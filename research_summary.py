import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import score
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2'

def main():
    records = pd.read_csv(OUT / 'all_experiments.csv')
    valid = records[records.status == 'ok']
    catalogue = json.loads((OUT / 'catalogue.json').read_text())
    lines = ['# Model comparison results', '', f'Completed {len(valid)} successful window/track/model evaluations across {valid.name.nunique()} configurations.', '', 'Full logs include unsuccessful attempts. Screening uses July-October 2025; confirmation uses November 2024-February 2025. Neither period is a hidden competition test. December-May was previously viewed and remains a reporting period.', '', '## Best screening results', '']
    for track in ['off_market', 'list_aware']:
        lines += [f'### {track}', '', '| Model | MdAPE % | MAPE % | Median signed error % |', '|---|---:|---:|---:|']
        for row in valid[(valid.stage == 'screen') & (valid.track == track)].sort_values('mdape').head(8).itertuples():
            lines.append(f'| {row.name} | {row.mdape:.3f} | {row.mape:.3f} | {row.bias:+.3f} |')
        lines.append('')
    if (ROOT / 'model_config_v2.json').exists():
        config = json.loads((ROOT / 'model_config_v2.json').read_text())
        lines += ['## Locked final selection', '', config['selection'], '']
        for track, members in config['tracks'].items():
            lines.append(f'- {track}: ' + ', '.join((f"{m['name']} ({m['weight']:.2f})" for m in members)))
    old = OUT / 'incumbent_validation.csv'
    new = ROOT / 'outputs' / 'validation_predictions.csv'
    if old.exists() and new.exists():
        a = pd.read_csv(old)
        b = pd.read_csv(new)
        paired = a.merge(b, on=['ml_num', 'track'], suffixes=('_old', '_new'), validate='one_to_one')
        rows = []
        for track, g in paired.groupby('track'):
            actual = g.sold_price_new.to_numpy()
            p = g.predicted_price_new.to_numpy()
            q = g.predicted_price_old.to_numpy()
            before = score(actual, q)
            after = score(actual, p)
            rng = np.random.default_rng(465)
            months = g.month_new.unique()
            diff = []
            for _ in range(1000):
                sampled = rng.choice(months, size=len(months), replace=True)
                ix = np.concatenate([np.flatnonzero(g.month_new.to_numpy() == m) for m in sampled])
                diff.append(score(actual[ix], p[ix])['mdape'] - score(actual[ix], q[ix])['mdape'])
            lo, hi = np.quantile(diff, [0.025, 0.975])
            rows.append({'track': track, 'old_mdape': before['mdape'], 'new_mdape': after['mdape'], 'old_mape': before['mape'], 'new_mape': after['mape'], 'mdape_change': after['mdape'] - before['mdape'], 'month_block_change_ci_low': lo, 'month_block_change_ci_high': hi, 'within_10_pct': 100 * float(np.mean(abs(p - actual) / actual <= 0.1))})
        pd.DataFrame(rows).to_csv(OUT / 'official_comparison.csv', index=False)
        lines += ['', '## Official reporting comparison', '', 'These are reused historical validation months. The block bootstrap samples entire months; six months give limited information about future regime changes.', '', pd.DataFrame(rows).to_markdown(index=False, floatfmt='.3f')]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
if __name__ == '__main__':
    main()
