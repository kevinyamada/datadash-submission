import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
TRACKS = ['off_market', 'list_aware']


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def analyse_prices(out):
    import numpy as np
    import pandas as pd
    path = ROOT / 'research_v2/official/outputs/validation_predictions.csv'
    frame = pd.read_csv(path)
    if set(frame.track.unique()) != set(TRACKS):
        raise ValueError('Expected both final model tracks in official predictions')
    if frame.duplicated(['track', 'ml_num']).any():
        raise ValueError('Duplicate predictions')
    prices = frame[['sold_price', 'predicted_price']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Invalid prices')
    a = frame[frame.track == 'off_market'].set_index('ml_num').sold_price.sort_index()
    b = frame[frame.track == 'list_aware'].set_index('ml_num').sold_price.sort_index()
    pd.testing.assert_series_equal(a, b)
    q10, q90 = a.quantile([0.1, 0.9]).to_numpy()
    rows = []
    error_quantiles = []
    for track, group in frame.groupby('track'):
        subsets = [('All homes', group), ('Cheapest 10%', group[group.sold_price <= q10]), ('Middle 80%', group[(group.sold_price > q10) & (group.sold_price < q90)]), ('Most expensive 10%', group[group.sold_price >= q90])]
        for label, subset in subsets:
            signed = 100 * (subset.predicted_price / subset.sold_price - 1)
            dollars = subset.predicted_price - subset.sold_price
            rows.append({'track': track, 'group': label, 'n': len(subset), 'mdape_pct': float(signed.abs().median()), 'mape_pct': float(signed.abs().mean()), 'median_bias_pct': float(signed.median()), 'mae_dollars': float(dollars.abs().mean()), 'rmse_dollars': float(np.sqrt(np.mean(dollars ** 2)))})
        error = 100 * (group.predicted_price / group.sold_price - 1).abs()
        error_quantiles.append({'track': track, 'error_p10_pct': float(error.quantile(0.1)), 'error_p50_pct': float(error.quantile(0.5)), 'error_p90_pct': float(error.quantile(0.9))})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'price_groups.csv', index=False)
    pd.DataFrame(error_quantiles).to_csv(out / 'error_percentiles.csv', index=False)
    save_json(out / 'price_thresholds.json', {'sold_price_p10': float(q10), 'sold_price_p90': float(q90), 'homes_per_track': len(a), 'source_sha256': digest(path), 'interpretation': 'Bottom and top 10 percent by actual validation sale price. Boundary ties are included, so counts can slightly exceed 10 percent. Diagnostic groups only, never prediction features. Error percentiles are reported separately because the original percentile request was ambiguous.'})
    print('\nPrice group results:', flush=True)
    print(scores.to_string(index=False, float_format=lambda x: f'{x:,.2f}'), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Cheapest 10%', 'Middle 80%', 'Most expensive 10%']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(3)
    for i, (track, name, colour) in enumerate([('off_market', 'Without asking price', '#2878a0'), ('list_aware', 'With asking price', '#c87332')]):
        values = scores[scores.track == track].set_index('group').loc[labels, 'mdape_pct']
        bars = ax.bar(positions + (i - 0.5) * 0.36, values, width=0.36, label=name, color=colour)
        ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Median absolute percentage error (%)')
    ax.set_title('Validation errors by actual sale price')
    ax.set_ylim(0, scores.mdape_pct.max() * 1.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'price_groups.png', dpi=180)
    plt.close(fig)


def benchmark(args):
    import numpy as np
    import data
    import models
    begin = time.perf_counter()
    train = data.load('listings')
    homes = data.hide_answers(data.load(args.worker), args.worker)
    load_seconds = time.perf_counter() - begin
    if train.date_sold.max() >= __import__('pandas').Timestamp('2026-06-01'):
        raise ValueError('Training includes sales after May 2026')
    if not homes.ml_num.is_unique:
        raise ValueError('Duplicate prediction home IDs')
    print(f'{args.worker}: fitting final model on {len(train):,} sales', flush=True)
    model = models.make_model(args.worker)
    begin = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - begin
    times = []
    first = None
    for repeat in range(4):
        begin = time.perf_counter()
        prediction = np.asarray(model.predict(homes), dtype=float)
        times.append(time.perf_counter() - begin)
        if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError('Invalid benchmark prediction')
        if first is None:
            first = prediction.copy()
        else:
            np.testing.assert_allclose(first, prediction, rtol=1e-10, atol=1e-6)
    median_seconds = float(np.median(times[1:]))
    record = {'track': args.worker, 'status': 'complete', 'training_homes': len(train), 'prediction_homes': len(homes), 'data_load_seconds': load_seconds, 'fit_seconds': fit_seconds, 'first_batch_prediction_seconds': times[0], 'repeat_batch_prediction_seconds': times[1:], 'median_repeat_batch_seconds': median_seconds, 'amortised_repeat_ms_per_home': 1000 * median_seconds / len(homes), 'fit_plus_first_prediction_seconds': fit_seconds + times[0], 'note': 'One training run on all labelled data. Full final settings. Fit includes feature preparation and internal calibration. Prediction includes feature transformation. Import and data loading time excluded from fit and predict. Per home time is batch throughput, not single request latency. Timings depend on hardware and load. Predictions are checked but not saved or scored against hidden outcomes.'}
    save_json(args.output / f'runtime_{args.worker}.json', record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=TRACKS)
    args = parser.parse_args()
    if args.threads < 1 or not 0 < args.minutes <= 120:
        parser.error('Use positive threads and a time limit of at most 120 minutes')
    for key in ['DASH_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = str(args.threads)
    if args.worker:
        benchmark(args)
        return
    started = time.monotonic()
    out = args.output or ROOT / 'final_checks' / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    out.mkdir(parents=True, exist_ok=True)
    sources = ['check_tails_runtime.py', 'models.py', 'advanced_models.py', 'spline_model.py', 'features.py', 'data.py', 'model_config_v2.json']
    manifest = {'source_hashes': {p: digest(ROOT / p) for p in sources}, 'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'threads': args.threads, 'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'catboost']}}
    import data
    manifest['data_hashes'] = {name: digest(data.find_file(name)) for name in ['listings', *TRACKS]}
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Code, data or environment changed. Use a new output directory.')
    save_json(path, manifest)
    save_json(out / 'run_settings.json', {'minutes': args.minutes, 'started_local': time.strftime('%Y-%m-%d %H:%M:%S')})
    statuses = {}
    child = None
    try:
        analyse_prices(out)
        for track in TRACKS:
            if (out / f'runtime_{track}.json').exists():
                statuses[track] = 'saved_result'
                print(f'{track}: reusing completed benchmark', flush=True)
                continue
            remaining = args.minutes * 60 - (time.monotonic() - started)
            if remaining <= 0:
                statuses[track] = 'time_limit'
                break
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', track, '--output', str(out.resolve()), '--threads', str(args.threads)]
            child = subprocess.Popen(command)
            deadline = time.monotonic() + remaining
            while child.poll() is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    child.kill()
                    child.wait()
                    statuses[track] = 'time_limit'
                    break
                try:
                    child.wait(timeout=min(20, left))
                except subprocess.TimeoutExpired:
                    print(f'{track}: running, {int(time.monotonic() - started)} seconds elapsed overall', flush=True)
            if track not in statuses:
                statuses[track] = 'complete' if child.returncode == 0 else 'error'
            child = None
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        statuses['run'] = 'interrupted'
    finally:
        save_json(out / 'run_status.json', {'tracks': statuses, 'elapsed_seconds': time.monotonic() - started})
        records = [json.loads(p.read_text()) for p in out.glob('runtime_*.json')]
        if records:
            import pandas as pd
            pd.DataFrame(records).to_csv(out / 'runtime_summary.csv', index=False)
        destination = out.with_suffix('.zip')
        with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
            for p in out.iterdir():
                if p.is_file() and p.suffix in ['.csv', '.json', '.png']:
                    archive.write(p, p.name)
            archive.write(Path(__file__), Path(__file__).name)
        print(f'\nSEND THIS FILE: {destination}', flush=True)
        print(f'To resume incomplete timings, use --output "{out}"', flush=True)
    if any(statuses.get(t) not in ['complete', 'saved_result'] for t in TRACKS):
        sys.exit(1)


if __name__ == '__main__':
    main()
