import argparse
import gc
import hashlib
import itertools
import json
import platform
import time
import traceback
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import data
import models
from advanced_models import AdvancedModel, BlendModel
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v3'
PRED = OUT / 'predictions'
PRED.mkdir(parents=True, exist_ok=True)
CUTOFFS = ['2024-03-01', '2024-11-01', '2025-07-01']

def catalogue():
    configs = {}
    for loss, leaves, rate, half_life in itertools.product(['regression', 'regression_l1'], [15, 31, 63], [0.025, 0.05], [None, 18]):
        name = f'lgb_{loss}_{leaves}_{rate}_{half_life}'
        configs[name] = {'family': 'lgb', 'features': 'enhanced', 'half_life': half_life, 'params': {'objective': loss, 'num_leaves': leaves, 'learning_rate': rate, 'n_estimators': 1800 if rate == 0.025 else 900, 'min_child_samples': 80, 'reg_lambda': 10}}
    for months, shrink in itertools.product([3, 6], [0.25, 0.5, 0.75]):
        configs[f'lgb_cal_{months}_{shrink}'] = {'family': 'lgb', 'features': 'enhanced', 'calibration_months': months, 'calibration_shrink': shrink, 'params': {'objective': 'regression', 'num_leaves': 63, 'n_estimators': 1200}}
    for loss, depth, penalty in itertools.product(['RMSE', 'MAE'], [5, 7, 9], [5, 20]):
        configs[f'cat_{loss}_{depth}_{penalty}'] = {'family': 'cat', 'features': 'enhanced', 'params': {'loss_function': loss, 'depth': depth, 'l2_leaf_reg': penalty, 'iterations': 1600, 'learning_rate': 0.04}}
    for loss, depth, child in itertools.product(['reg:squarederror', 'reg:absoluteerror'], [3, 5, 7], [20, 80]):
        name = f"xgb_{loss.split(':')[1]}_{depth}_{child}"
        configs[name] = {'family': 'xgb', 'features': 'enhanced', 'native_categories': True, 'params': {'objective': loss, 'max_depth': depth, 'min_child_weight': child, 'n_estimators': 1400, 'learning_rate': 0.035, 'reg_lambda': 15}}
    for penalty, recent in itertools.product([10, 100, 1000], [24, 36]):
        configs[f'spline_text_{penalty}_{recent}'] = {'family': 'gam', 'features': 'enhanced', 'text_ridge': True, 'recent_months': recent, 'params': {'alpha': penalty}}
    for loss, leaves in itertools.product(['squared_error', 'absolute_error'], [15, 31]):
        configs[f'hist_{loss}_{leaves}'] = {'family': 'hist', 'features': 'base', 'params': {'loss': loss, 'max_leaf_nodes': leaves, 'max_iter': 1000, 'l2_regularization': 20, 'learning_rate': 0.035}}
    for leaf in [5, 20]:
        configs[f'extra_{leaf}'] = {'family': 'extra', 'features': 'base', 'params': {'n_estimators': 160, 'max_depth': 24, 'min_samples_leaf': leaf, 'max_features': 0.8}}
    for penalty in [10, 1000]:
        configs[f'ridge_{penalty}'] = {'family': 'ridge', 'features': 'enhanced', 'params': {'alpha': penalty}}
    return configs

def score(actual, predicted):
    signed = 100 * (predicted / actual - 1)
    absolute = np.abs(signed)
    return {'mdape': float(np.median(absolute)), 'mape': float(np.mean(absolute)), 'bias': float(np.median(signed)), 'p90_error': float(np.quantile(absolute, 0.9))}

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def summarise():
    records = [json.loads(path.read_text()) for path in sorted(PRED.glob('*.json'))]
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'all_runs.csv', index=False)
    successful = frame[frame.status == 'ok']
    if successful.empty:
        return
    ranking = successful.groupby(['track', 'name']).agg(completed_windows=('cutoff', 'nunique'), mean_mdape=('mdape', 'mean'), worst_mdape=('mdape', 'max'), mean_mape=('mape', 'mean'), total_seconds=('seconds', 'sum')).reset_index()
    ranking = ranking.sort_values(['track', 'mean_mdape', 'mean_mape'])
    ranking.to_csv(OUT / 'ranking.csv', index=False)
    complete = ranking[ranking.completed_windows == len(CUTOFFS)]
    print('\nCompleted three-window comparisons:', flush=True)
    for track in ['off_market', 'list_aware']:
        print(complete[complete.track == track].head(10).to_string(index=False), flush=True)

def package():
    summarise()
    destination = ROOT / 'research_v3_results.zip'
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in OUT.rglob('*'):
            if path.is_file() and path.suffix in ['.csv', '.json']:
                archive.write(path, path.relative_to(ROOT))
        archive.write(Path(__file__), Path(__file__).name)
    print(f'\nSEND: {destination}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-only', action='store_true')
    parser.add_argument('--max-new-runs', type=int, default=0)
    args = parser.parse_args()
    if args.package_only:
        package()
        return
    configs = catalogue()
    selected = json.loads((ROOT / 'model_config_v2.json').read_text())
    listings = data.load('listings')
    sources = ['extensive_research_v3.py', 'advanced_models.py', 'spline_model.py', 'temporal_features.py', 'features.py', 'models.py', 'data.py', 'model_config.json', 'model_config_v2.json']
    manifest = {'cutoffs': CUTOFFS, 'horizon_months': 4, 'python': platform.python_version(), 'packages': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'catboost', 'xgboost']}, 'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}, 'data_hash': hashlib.sha256(pd.util.hash_pandas_object(listings, index=True).values.tobytes()).hexdigest(), 'catalogue': configs, 'selected': selected}
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Code, configuration, data or package versions changed. Do not mix results. Send this error before continuing.')
    else:
        write_json(manifest_path, manifest)
    names = ['previous', 'selected', *configs]
    total = len(CUTOFFS) * 2 * len(names)
    position = 0
    new_runs = 0
    print(f'{len(configs)} candidates plus two references', flush=True)
    print(f'{total} total comparisons; completed runs will be skipped', flush=True)
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
            for track in ['off_market', 'list_aware']:
                safe_homes = data.hide_answers(homes, track)
                for name in names:
                    position += 1
                    stem = f'{cutoff}_{track}_{name}'
                    metadata_path = PRED / f'{stem}.json'
                    prediction_path = PRED / f'{stem}.csv'
                    if metadata_path.exists():
                        record = json.loads(metadata_path.read_text())
                        if record['status'] == 'error':
                            print(f'[{position}/{total}] SKIP recorded error {stem}', flush=True)
                            continue
                        if prediction_path.exists():
                            print(f'[{position}/{total}] SKIP {stem}', flush=True)
                            continue
                    if args.max_new_runs and new_runs >= args.max_new_runs:
                        return
                    print(f'[{position}/{total}] START {stem}', flush=True)
                    begin = time.perf_counter()
                    model = None
                    record = {'cutoff': cutoff, 'validation_end_exclusive': str(end.date()), 'track': track, 'name': name, 'train_n': len(train), 'n': len(homes)}
                    try:
                        if name == 'previous':
                            model = models.make_previous_model(track)
                        elif name == 'selected':
                            model = BlendModel(track, selected['tracks'][track])
                        else:
                            model = AdvancedModel(track, deepcopy(configs[name]))
                        model.fit(train)
                        prediction = np.asarray(model.predict(safe_homes), dtype=float)
                        assert len(prediction) == len(homes)
                        assert np.isfinite(prediction).all()
                        assert (prediction > 0).all()
                        output = homes[['ml_num', 'sold_price', 'house_type_name', 'community']].copy()
                        output['month'] = homes.date_sold.dt.strftime('%Y-%m')
                        output['listed_after_cutoff'] = homes.date_listed >= start
                        output['predicted_price'] = prediction
                        temporary = prediction_path.with_suffix('.tmp')
                        output.to_csv(temporary, index=False)
                        temporary.replace(prediction_path)
                        record.update(score(homes.sold_price.to_numpy(), prediction))
                        record['status'] = 'ok'
                    except Exception:
                        record['status'] = 'error'
                        record['error'] = traceback.format_exc()
                        print(record['error'], flush=True)
                    finally:
                        model = None
                        gc.collect()
                    record['seconds'] = time.perf_counter() - begin
                    write_json(metadata_path, record)
                    new_runs += 1
                    print(json.dumps(record), flush=True)
                    if new_runs % 10 == 0:
                        summarise()
    except KeyboardInterrupt:
        print('\nStopped. Completed comparisons are preserved.', flush=True)
    finally:
        package()
if __name__ == '__main__':
    main()
