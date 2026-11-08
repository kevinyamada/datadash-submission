import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
CUTOFFS = ['2023-07-01', '2024-03-01', '2024-07-01', '2025-03-01']
WEIGHTS = [0.5, 0.75, 1.0]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class SavedFiles:
    def __init__(self, roots):
        self.roots = list(dict.fromkeys(Path(p).resolve() for p in roots))
        self.provenance = []

    def read(self, relative, archive_name=None):
        import pandas as pd
        for root in self.roots:
            path = root / relative
            if path.is_file():
                raw = path.read_bytes()
                source = str(path)
                break
            archive = root / archive_name if archive_name else None
            if archive and archive.is_file():
                with ZipFile(archive) as zipped:
                    if relative not in zipped.namelist():
                        continue
                    raw = zipped.read(relative)
                    source = str(archive) + ':' + relative
                    break
        else:
            raise FileNotFoundError(f'Missing saved predictions: {relative}. Keep the earlier results folder or historical_stress_results.zip in the folder passed to --saved-root.')
        self.provenance.append({'file': relative, 'source': source, 'sha256': hashlib.sha256(raw).hexdigest()})
        return pd.read_csv(io.BytesIO(raw))


def validate(frame):
    import numpy as np
    required = {'ml_num', 'sold_price', 'predicted_price', 'month', 'house_type_name'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if not frame.ml_num.is_unique:
        raise ValueError('Duplicate home IDs in saved predictions')
    values = frame[['sold_price', 'predicted_price']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Prices must be finite and positive')
    return frame.sort_values('ml_num').reset_index(drop=True)


def metrics(actual, predicted):
    import numpy as np
    error = 100 * (np.asarray(predicted) / np.asarray(actual) - 1)
    return {'n': len(error), 'mdape': float(np.median(np.abs(error))), 'mape': float(np.mean(np.abs(error))), 'median_bias': float(np.median(error))}


def correction(prior, predicted):
    import numpy as np
    edges = [1_500_000, 2_000_000, np.inf]
    factors = np.ones(len(predicted))
    fitted = []
    for low, high in zip(edges[:-1], edges[1:]):
        past = prior[(prior.predicted_price >= low) & (prior.predicted_price < high)]
        n = len(past)
        delta = 0.0
        if n >= 200:
            residual = np.log(past.sold_price.to_numpy() / past.predicted_price.to_numpy())
            delta = float(np.clip(np.median(residual) * n / (n + 500), -np.log(1.10), np.log(1.10)))
        mask = (predicted >= low) & (predicted < high)
        factors[mask] = np.exp(delta)
        fitted.append({'lower_predicted_price': low, 'upper_predicted_price': None if np.isinf(high) else high, 'calibration_n': n, 'factor': float(np.exp(delta))})
    return predicted * factors, fitted


def radius(values):
    import numpy as np
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) < 100:
        raise ValueError('Too few earlier residuals to calibrate an interval')
    rank = min(len(ordered), int(np.ceil((len(ordered) + 1) * 0.90)))
    return float(ordered[rank - 1])


def interval_score(frame, prior, method):
    import numpy as np
    residual = np.abs(np.log(prior.sold_price.to_numpy() / prior.predicted_price.to_numpy()))
    global_radius = radius(residual)
    q = np.full(len(frame), global_radius)
    if method == 'property_conservative':
        for kind in frame.house_type_name.unique():
            group = prior[prior.house_type_name == kind]
            if len(group) >= 200:
                local = radius(np.abs(np.log(group.sold_price.to_numpy() / group.predicted_price.to_numpy())))
                q[frame.house_type_name.to_numpy() == kind] = max(global_radius, local)
    point = frame.predicted_price.to_numpy()
    actual = frame.sold_price.to_numpy()
    lower, upper = point * np.exp(-q), point * np.exp(q)
    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower
    score = width + 20 * np.maximum(lower - actual, 0) + 20 * np.maximum(actual - upper, 0)
    return {'n': len(frame), 'calibration_n': len(prior), 'coverage': float(100 * covered.mean()), 'median_width_pct_prediction': float(np.median(100 * width / point)), 'mean_interval_score_pct_actual': float(np.mean(100 * score / actual))}


def worker(args, out):
    import numpy as np
    import pandas as pd
    files = SavedFiles([args.saved_root, ROOT])
    history = {weight: [] for weight in WEIGHTS}
    list_history = []
    scores, breakdowns, intervals, calibrations = [], [], [], []
    for cutoff in CUTOFFS:
        print(f'Reading saved predictions for {cutoff}', flush=True)
        start = pd.Timestamp(cutoff)
        end = start + pd.DateOffset(months=4)
        loaded = {}
        names = [('off_market', 'selected'), ('off_market', 'lgb_l2_calibrated'), ('off_market', 'spline_text_ridge10'), ('list_aware', 'selected')]
        for track, name in names:
            relative = f'research_v2/historical_stress/pred_{cutoff}_{track}_{name}.csv'
            frame = validate(files.read(relative, 'historical_stress_results.zip'))
            months = pd.to_datetime(frame.month + '-01')
            if not ((months >= start) & (months < end)).all():
                raise ValueError(f'Unexpected evaluation dates in {relative}')
            if 'cutoff' in frame and not (frame.cutoff == cutoff).all():
                raise ValueError('Saved cutoff mismatch')
            loaded[(track, name)] = frame
        base = loaded[('off_market', 'selected')]
        tree = loaded[('off_market', 'lgb_l2_calibrated')]
        spline = loaded[('off_market', 'spline_text_ridge10')]
        for other in [tree, spline]:
            pd.testing.assert_frame_equal(base[['ml_num', 'sold_price', 'month', 'house_type_name']], other[['ml_num', 'sold_price', 'month', 'house_type_name']])
        reference = np.exp(0.5 * np.log(tree.predicted_price.to_numpy()) + 0.5 * np.log(spline.predicted_price.to_numpy()))
        np.testing.assert_allclose(reference, base.predicted_price.to_numpy(), rtol=1e-8, atol=0.01)
        for weight in WEIGHTS:
            prediction = np.exp(weight * np.log(tree.predicted_price.to_numpy()) + (1 - weight) * np.log(spline.predicted_price.to_numpy()))
            current = base.copy()
            current['predicted_price'] = prediction
            if history[weight]:
                prior = pd.concat(history[weight], ignore_index=True)
                if not (pd.to_datetime(prior.month + '-01') < start).all():
                    raise ValueError('Calibration includes current or later outcomes')
                for adjustment in ['none', 'price_band']:
                    adjusted = prediction.copy()
                    if adjustment == 'price_band':
                        adjusted, fitted = correction(prior, prediction)
                        calibrations.extend({'cutoff': cutoff, 'tree_weight': weight, **item} for item in fitted)
                    name = f'tree_{weight:g}_{adjustment}'
                    record = {'cutoff': cutoff, 'candidate': name, 'tree_weight': weight, 'correction': adjustment, **metrics(base.sold_price, adjusted)}
                    scores.append(record)
                    groups = {'property_type': base.house_type_name, 'actual_price': pd.cut(base.sold_price, [0, 500000, 1000000, 2000000, np.inf], labels=['below_500k', '500k_to_1m', '1m_to_2m', '2m_or_more'], right=False), 'month': base.month}
                    if 'listed_after_cutoff' in base:
                        groups['listed_after_cutoff'] = base.listed_after_cutoff.astype(str)
                    for grouping, values in groups.items():
                        for label in values.dropna().unique():
                            mask = (values == label).to_numpy()
                            breakdowns.append({'cutoff': cutoff, 'candidate': name, 'grouping': grouping, 'group': str(label), **metrics(base.sold_price.to_numpy()[mask], adjusted[mask])})
                    print(f'{cutoff} {name}: median error {record["mdape"]:.3f}%', flush=True)
            history[weight].append(current)
        current_list = loaded[('list_aware', 'selected')]
        if list_history:
            pooled = pd.concat(list_history, ignore_index=True)
            for method in ['pooled', 'latest_window', 'property_conservative']:
                prior = list_history[-1] if method == 'latest_window' else pooled
                if not (pd.to_datetime(prior.month + '-01') < start).all():
                    raise ValueError('Interval calibration includes later outcomes')
                intervals.append({'evaluation': 'historical_frozen', 'period': cutoff, 'method': method, **interval_score(current_list, prior, method)})
        list_history.append(current_list)
        pd.DataFrame(scores).to_csv(out / 'point_scores.csv', index=False)
        pd.DataFrame(breakdowns).to_csv(out / 'error_groups.csv', index=False)
        pd.DataFrame(intervals).to_csv(out / 'interval_scores.csv', index=False)
        atomic_json(out / 'price_corrections.json', calibrations)
        atomic_json(out / 'sources.json', files.provenance)
    try:
        official = files.read('research_v2/official/outputs/validation_predictions.csv')
    except FileNotFoundError:
        print('Official monthly predictions not found. Historical interval checks are still available.', flush=True)
    else:
        official = official[official.track == 'list_aware'].copy()
        official = validate(official)
        initial = official[official.month.isin(['2025-12', '2026-01'])]
        for month in ['2026-02', '2026-03', '2026-04', '2026-05']:
            current = official[official.month == month]
            if current.empty:
                raise ValueError(f'Missing official predictions for {month}')
            past = official[official.month < month]
            recent_months = sorted(past.month.unique())[-2:]
            for method in ['fixed_dec_jan', 'expanding', 'recent_two_months', 'property_conservative']:
                prior = initial if method == 'fixed_dec_jan' else past[past.month.isin(recent_months)] if method == 'recent_two_months' else past
                intervals.append({'evaluation': 'official_monthly_reused', 'period': month, 'method': method, **interval_score(current, prior, method)})
        print('Official interval comparisons are exploratory because these months were already inspected.', flush=True)
    table = pd.DataFrame(scores)
    reference = table[table.candidate == 'tree_0.5_none'].set_index('cutoff')
    ranking = []
    for name, group in table.groupby('candidate'):
        group = group.set_index('cutoff').loc[reference.index]
        delta = group.mdape - reference.mdape
        mape_delta = group.mape - reference.mape
        ranking.append({'candidate': name, 'windows': len(group), 'mean_mdape': float(group.mdape.mean()), 'mean_mape': float(group.mape.mean()), 'mdape_change_pp': float(delta.mean()), 'mape_change_pp': float(mape_delta.mean()), 'worst_window_change_pp': float(delta.max()), 'windows_improved': int((delta < 0).sum()), 'worth_confirmation': bool(delta.mean() <= -0.10 and mape_delta.mean() <= 0.10 and delta.max() <= 0.20 and (delta < 0).sum() >= 2)})
    ranked = pd.DataFrame(ranking).sort_values(['mean_mdape', 'mean_mape'])
    ranked.to_csv(out / 'ranking.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'interval_scores.csv', index=False)
    atomic_json(out / 'sources.json', files.provenance)
    atomic_json(out / 'status.json', {'status': 'complete', 'point_windows': list(reference.index), 'note': 'Exploratory comparisons on previously inspected periods. No automatic model replacement. No training or hidden test access.'})
    print('\nPoint prediction comparisons:', flush=True)
    print(ranked.to_string(index=False), flush=True)
    print('\nInterval results: coverage nearer 90% is useful only alongside interval width and interval score.', flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved-root', type=Path, default=ROOT)
    parser.add_argument('--minutes', type=float, default=15)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.minutes <= 120:
        parser.error('--minutes must be above zero and at most 120')
    if args.worker:
        worker(args, args.output)
        return
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = args.output or ROOT / 'focused_results' / f'{stamp}_{os.getpid()}'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'plan.json', {'tree_weights': WEIGHTS, 'price_correction': 'Earlier out of sample median log residuals in predicted price bands above 1.5m; at least 200 cases, shrink n/(n+500), cap factor at 1/1.10 to 1.10', 'calibration_only_window': CUTOFFS[0], 'evaluation_windows': CUTOFFS[1:], 'interval_target': 0.9, 'point_review_rule': 'Average MdAPE improvement at least 0.10 percentage points, MAPE worsening at most 0.10, worst window worsening at most 0.20, improvement in at least two of three windows', 'time_limit_minutes': args.minutes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'disclosure': 'All periods have been seen before. Treat results as exploratory. No model or submission files are changed.'})
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--saved-root', str(args.saved_root.resolve()), '--output', str(out.resolve())]
    print('Reusing saved predictions. No model training. No hidden test data. No paid compute.', flush=True)
    print(f'Time limit: {args.minutes:g} minutes. Results: {out}', flush=True)
    status = 'error'
    code = 1
    try:
        process = subprocess.Popen(command)
        try:
            code = process.wait(timeout=args.minutes * 60)
            status = 'complete' if code == 0 else 'error'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = 'time_limit'
            print('Time limit reached. Completed output files are preserved.', flush=True)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.kill()
            process.wait()
            status = 'interrupted'
    finally:
        atomic_json(out / 'run_status.json', {'status': status, 'return_code': code})
        destination = out.with_suffix('.zip')
        with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
            for path in out.iterdir():
                if path.is_file() and path.suffix in ['.csv', '.json']:
                    archive.write(path, path.name)
            archive.write(Path(__file__), Path(__file__).name)
        print(f'\nSEND THIS FILE: {destination}', flush=True)
    if status != 'complete':
        sys.exit(1)


if __name__ == '__main__':
    main()
