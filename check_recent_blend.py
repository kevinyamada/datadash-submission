import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from focused_research import SavedFiles, atomic_json, metrics, validate

ROOT = Path(__file__).resolve().parent
CUTOFFS = ['2024-03-01', '2024-11-01', '2025-07-01']


def read_json(roots, relative, provenance):
    for root in roots:
        path = root / relative
        archive = root / 'research_v3_results.zip'
        if path.is_file():
            raw = path.read_bytes()
            source = str(path)
        elif archive.is_file():
            with ZipFile(archive) as zipped:
                if relative not in zipped.namelist():
                    continue
                raw = zipped.read(relative)
            source = str(archive) + ':' + relative
        else:
            continue
        provenance.append({'file': relative, 'source': source, 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    raise FileNotFoundError(f'Missing {relative}. Keep the earlier research_v3 folder or research_v3_results.zip in --saved-root.')


def worker(args):
    import numpy as np
    import pandas as pd
    files = SavedFiles([args.saved_root, ROOT])
    manifest = read_json(files.roots, 'research_v3/manifest.json', files.provenance)
    members = manifest['selected']['tracks']['off_market']
    tree = next(item for item in members if item['name'] == 'lgb_l2_calibrated')
    if len(members) != 2 or any(item['weight'] != 0.5 for item in members):
        raise ValueError('Saved selected blend is not the expected equal weight blend')
    if tree['config'] != manifest['catalogue']['lgb_cal_3_0.5']:
        raise ValueError('Saved LightGBM configuration does not match the blend component')
    scores, groups, monthly = [], [], []
    for cutoff in CUTOFFS:
        print(f'Checking saved window starting {cutoff}', flush=True)
        loaded = {}
        for name in ['selected', 'lgb_cal_3_0.5']:
            stem = f'research_v3/predictions/{cutoff}_off_market_{name}'
            record = read_json(files.roots, stem + '.json', files.provenance)
            if record['status'] != 'ok' or record['cutoff'] != cutoff or record['name'] != name or record['track'] != 'off_market':
                raise ValueError(f'Unexpected metadata for {stem}')
            frame = validate(files.read(stem + '.csv', 'research_v3_results.zip'))
            start = pd.Timestamp(cutoff)
            end = start + pd.DateOffset(months=4)
            months = pd.to_datetime(frame.month + '-01')
            if len(frame) != record['n'] or not ((months >= start) & (months < end)).all():
                raise ValueError(f'Incorrect saved rows or dates for {stem}')
            actual = metrics(frame.sold_price, frame.predicted_price)
            for key in ['mdape', 'mape']:
                if not np.isclose(actual[key], record[key], atol=1e-7, rtol=1e-7):
                    raise ValueError(f'Saved predictions do not match recorded {key}')
            loaded[name] = frame
        base, lgb = loaded['selected'], loaded['lgb_cal_3_0.5']
        pd.testing.assert_frame_equal(base[['ml_num', 'sold_price', 'month', 'house_type_name']], lgb[['ml_num', 'sold_price', 'month', 'house_type_name']])
        proposed = np.exp(0.5 * np.log(base.predicted_price.to_numpy()) + 0.5 * np.log(lgb.predicted_price.to_numpy()))
        for name, prediction in [('current_50_50', base.predicted_price.to_numpy()), ('proposed_75_25', proposed)]:
            row = {'cutoff': cutoff, 'candidate': name, **metrics(base.sold_price, prediction)}
            scores.append(row)
            print(f'{name}: median error {row["mdape"]:.4f}%, mean error {row["mape"]:.4f}%', flush=True)
            labels = {'property_type': base.house_type_name, 'actual_price': pd.cut(base.sold_price, [0, 500000, 1000000, 2000000, np.inf], labels=['below_500k', '500k_to_1m', '1m_to_2m', '2m_or_more'], right=False)}
            if 'listed_after_cutoff' in base:
                labels['listed_after_cutoff'] = base.listed_after_cutoff.astype(str)
            for grouping, values in labels.items():
                for label in values.dropna().unique():
                    mask = (values == label).to_numpy()
                    groups.append({'cutoff': cutoff, 'candidate': name, 'grouping': grouping, 'group': str(label), **metrics(base.sold_price.to_numpy()[mask], prediction[mask])})
            for month in sorted(base.month.unique()):
                mask = (base.month == month).to_numpy()
                monthly.append({'cutoff': cutoff, 'month': month, 'candidate': name, **metrics(base.sold_price.to_numpy()[mask], prediction[mask])})
        pd.DataFrame(scores).to_csv(args.output / 'scores.csv', index=False)
        pd.DataFrame(groups).to_csv(args.output / 'groups.csv', index=False)
        pd.DataFrame(monthly).to_csv(args.output / 'monthly.csv', index=False)
        atomic_json(args.output / 'sources.json', files.provenance)
    table = pd.DataFrame(scores)
    recent = table[table.cutoff.isin(CUTOFFS[1:])]
    summary = recent.groupby('candidate').agg(windows=('cutoff', 'nunique'), mean_mdape=('mdape', 'mean'), mean_mape=('mape', 'mean')).reset_index()
    summary.to_csv(args.output / 'recent_summary.csv', index=False)
    print('\nRecent period comparison:', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('\nThe March 2024 window overlaps the previous check. All periods were inspected in earlier research. These are exploratory results, not fresh independent validation.', flush=True)


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
        worker(args)
        return
    out = args.output or ROOT / 'blend_check_results' / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'plan.json', {'cutoffs': CUTOFFS, 'recent_comparison': CUTOFFS[1:], 'candidates': ['current_50_50', 'proposed_75_25'], 'identity': 'sqrt(current_geometric_blend * tree) equals tree^0.75 * spline^0.25 when current is tree^0.5 * spline^0.5', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'time_limit_minutes': args.minutes, 'disclosure': 'Reuses previously inspected periods. No fitting, hidden test access or automatic model changes.'})
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--saved-root', str(args.saved_root.resolve()), '--output', str(out.resolve())]
    print('Reusing saved predictions only. No training or submission changes.', flush=True)
    status, code = 'error', 1
    try:
        process = subprocess.Popen(command)
        try:
            code = process.wait(timeout=args.minutes * 60)
            status = 'complete' if code == 0 else 'error'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = 'time_limit'
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
