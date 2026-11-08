import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

def main():
    analysis = ROOT / 'analysis'
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    comparison = pd.read_csv(analysis / 'comparison.csv')
    stress = pd.read_csv(analysis / 'frozen_four_month_scores.csv')
    intervals = pd.read_csv(analysis / 'interval_evaluation.csv')
    safe = pd.read_csv(analysis / 'scores_by_listed_after_cutoff.csv')
    price = pd.read_csv(analysis / 'scores_by_price_band.csv')

    def value(frame, track, field='mdape', **filters):
        rows = frame[frame.track == track]
        for key, val in filters.items():
            rows = rows[rows[key] == val]
        return float(rows.iloc[0][field])
    macros = {'OffMd': value(comparison, 'off_market', model='selected'), 'ListMd': value(comparison, 'list_aware', model='selected'), 'OffBase': value(comparison, 'off_market', model='baseline'), 'ListBase': value(comparison, 'list_aware', model='baseline'), 'SafeOff': value(safe, 'off_market', listed_after_cutoff=True), 'SafeList': value(safe, 'list_aware', listed_after_cutoff=True), 'FrozenOff': value(stress, 'off_market', model='selected'), 'FrozenList': value(stress, 'list_aware', model='selected'), 'LuxuryMd': value(price, 'off_market', price_band='over $2m'), 'LuxuryBias': value(price, 'off_market', 'median_signed_error', price_band='over $2m'), 'MiddleMd': value(price, 'off_market', price_band='$500k-$1m')}
    macros['OffGain'] = 100 * (1 - macros['OffMd'] / macros['OffBase'])
    macros['ListGain'] = 100 * (1 - macros['ListMd'] / macros['ListBase'])
    (reports / 'case_numbers.tex').write_text('\n'.join(('\\newcommand{\\' + key + '}{' + f'{val:.2f}' + '}' for key, val in macros.items())) + '\n')
    rows = ['\\begin{tabular}{llrr}', '\\toprule', 'Track & Method & MdAPE \\% & MAPE \\% \\\\', '\\midrule']
    for row in comparison.itertuples():
        name = 'Off-market' if row.track == 'off_market' else 'List-aware'
        method = 'LightGBM' if row.model == 'selected' else 'Recent median' if row.track == 'off_market' else 'Asking price'
        rows.append(f'{name} & {method} & {row.mdape:.2f} & {row.mape:.2f} \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    (reports / 'case_comparison.tex').write_text('\n'.join(rows) + '\n')
    rows = ['\\begin{tabular}{lrr}', '\\toprule', 'Forecast protocol & Off-market & List-aware \\\\', '\\midrule']
    for method, label in [('monthly_refit', 'Monthly refits'), ('frozen_four_month', 'Frozen four months')]:
        a = value(intervals, 'off_market', 'coverage_pct', evaluation=method)
        b = value(intervals, 'list_aware', 'coverage_pct', evaluation=method)
        rows.append(f'{label} & {a:.1f}\\% & {b:.1f}\\% \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    (reports / 'interval_table.tex').write_text('\n'.join(rows) + '\n')
    q = json.loads((analysis / 'interval_quantiles.json').read_text())
    for track in ['off_market', 'list_aware']:
        frame = pd.read_csv(ROOT / 'submissions' / f'submission_{track}.csv')
        frame['lower_historical_90'] = np.round(frame.predicted_price * np.exp(-q[track]))
        frame['upper_historical_90'] = np.round(frame.predicted_price * np.exp(q[track]))
        frame.to_csv(analysis / f'prediction_intervals_{track}.csv', index=False)
    manifest = {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'matplotlib', 'pyarrow']}, 'dataset': json.loads((ROOT / 'data_local' / 'manifest.json').read_text()), 'model_config': json.loads((ROOT / 'model_config.json').read_text()), 'submission_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'submissions').glob('*.csv')}, 'seed': 465, 'hidden_test_scores_available': False}
    (analysis / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    script = (ROOT / 'report' / 'video_script_template.md').read_text()
    for key, val in macros.items():
        script = script.replace('[' + key + ']', f'{val:.2f}')
    script = script.replace('Replace the bracketed result fields from the freshly generated tables before recording. ', '')
    (ROOT / 'report' / 'video_script.md').write_text(script)
    print(json.dumps(macros, indent=2))
if __name__ == '__main__':
    main()
