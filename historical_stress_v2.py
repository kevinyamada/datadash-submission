import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import data
import models
from advanced_models import BlendModel
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2' / 'historical_stress'
OUT.mkdir(parents=True, exist_ok=True)
CUTOFFS = ['2023-07-01', '2024-03-01', '2024-07-01', '2025-03-01']
TRACKS = ['off_market', 'list_aware']

def metrics(frame):
    signed = 100 * (frame.predicted_price / frame.sold_price - 1)
    absolute = signed.abs()
    return {'n': len(frame), 'mdape': float(absolute.median()), 'mape': float(absolute.mean()), 'median_bias': float(signed.median()), 'p90_error': float(absolute.quantile(0.9)), 'within_10_percent': float((absolute <= 10).mean() * 100)}

def save_predictions(homes, prediction, path, cutoff, track, name):
    prediction = np.asarray(prediction, dtype=float)
    assert len(prediction) == len(homes)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
    frame = homes[['ml_num', 'sold_price', 'house_type_name', 'community', 'date_listed']].copy()
    frame['month'] = homes.date_sold.dt.strftime('%Y-%m')
    frame['cutoff'] = cutoff
    frame['track'] = track
    frame['model'] = name
    frame['listed_after_cutoff'] = homes.date_listed >= pd.Timestamp(cutoff)
    frame['predicted_price'] = prediction
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    print(json.dumps({'cutoff': cutoff, 'track': track, 'model': name, **metrics(frame)}), flush=True)

def summarise():
    frames = [pd.read_csv(p) for p in sorted(OUT.glob('pred_*.csv'))]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summaries = []
    breakdowns = []
    for keys, group in combined.groupby(['cutoff', 'track', 'model']):
        base = dict(zip(['cutoff', 'track', 'model'], keys))
        summaries.append({**base, 'subset': 'all', **metrics(group)})
        safe = group[group.listed_after_cutoff.astype(str).str.lower() == 'true']
        if len(safe):
            summaries.append({**base, 'subset': 'listed_after_cutoff', **metrics(safe)})
        for column in ['month', 'house_type_name', 'community']:
            for label, subset in group.groupby(column):
                breakdowns.append({**base, 'grouping': column, 'group': label, **metrics(subset)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(breakdowns).to_csv(OUT / 'breakdown.csv', index=False)
    selected = summary[(summary.subset == 'all') & summary.model.isin(['previous', 'selected'])]
    print(selected.to_string(index=False), flush=True)
    with ZipFile(ROOT / 'historical_stress_results.zip', 'w', ZIP_DEFLATED) as archive:
        for path in OUT.iterdir():
            if path.is_file() and path.suffix in ['.csv', '.json']:
                archive.write(path, path.relative_to(ROOT))
        archive.write(Path(__file__), Path(__file__).name)
        archive.write(ROOT / 'model_config_v2.json', 'model_config_v2.json')
    print('SEND: historical_stress_results.zip', flush=True)

def main():
    config = json.loads((ROOT / 'model_config_v2.json').read_text())
    source_files = ['historical_stress_v2.py', 'advanced_models.py', 'spline_model.py', 'temporal_features.py', 'features.py', 'models.py', 'data.py', 'model_config.json', 'model_config_v2.json']
    manifest = {'cutoffs': CUTOFFS, 'horizon_months': 4, 'python': platform.python_version(), 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'catboost']}, 'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}}
    listings = data.load('listings')
    manifest['training_table_hash'] = hashlib.sha256(pd.util.hash_pandas_object(listings, index=True).values.tobytes()).hexdigest()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest != manifest:
            raise RuntimeError('Code, data or environment changed. Preserve the existing results and use a new output directory before rerunning.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        for cutoff in CUTOFFS:
            start = pd.Timestamp(cutoff)
            end = start + pd.DateOffset(months=4)
            train = listings[listings.date_sold < start].copy()
            homes = listings[(listings.date_sold >= start) & (listings.date_sold < end)].copy()
            assert len(train) and len(homes)
            assert train.date_sold.max() < homes.date_sold.min()
            assert homes.ml_num.is_unique
            assert not set(train.ml_num).intersection(homes.ml_num)
            for track in TRACKS:
                safe_homes = data.hide_answers(homes, track)
                baseline_path = OUT / f'pred_{cutoff}_{track}_previous.csv'
                if not baseline_path.exists():
                    print(f'START {cutoff} {track} previous', flush=True)
                    begin = perf_counter()
                    baseline = models.make_previous_model(track).fit(train)
                    save_predictions(homes, baseline.predict(safe_homes), baseline_path, cutoff, track, 'previous')
                    print(f'Finished in {perf_counter() - begin:.1f} seconds', flush=True)
                    del baseline
                members = config['tracks'][track]
                names = ['selected']
                if track == 'off_market':
                    names += [member['name'] for member in members]
                paths = {name: OUT / f'pred_{cutoff}_{track}_{name}.csv' for name in names}
                if all((path.exists() for path in paths.values())):
                    print(f'SKIP completed {cutoff} {track}', flush=True)
                    continue
                print(f'START {cutoff} {track} selected', flush=True)
                begin = perf_counter()
                model = BlendModel(track, members).fit(train)
                save_predictions(homes, model.predict(safe_homes), paths['selected'], cutoff, track, 'selected')
                if track == 'off_market':
                    for member, fitted in zip(members, model.models):
                        save_predictions(homes, fitted.predict(safe_homes), paths[member['name']], cutoff, track, member['name'])
                print(f'Finished in {perf_counter() - begin:.1f} seconds', flush=True)
                del model
    except KeyboardInterrupt:
        print('Interrupted. Completed predictions are saved.', flush=True)
    finally:
        summarise()
if __name__ == '__main__':
    if '--package-only' in sys.argv:
        summarise()
    else:
        main()
