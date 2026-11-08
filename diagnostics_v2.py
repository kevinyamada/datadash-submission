from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2' / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

def metrics(frame):
    signed = 100 * (frame.predicted_price / frame.sold_price - 1)
    return {'n': len(frame), 'mdape': float(signed.abs().median()), 'mape': float(signed.abs().mean()), 'median_bias': float(signed.median()), 'within_10_percent': float((signed.abs() <= 10).mean() * 100)}
if __name__ == '__main__':
    predictions = pd.read_csv(ROOT / 'research_v2' / 'official' / 'outputs' / 'validation_predictions.csv')
    intervals = []
    errors = []
    for track in ['off_market', 'list_aware']:
        frame = predictions[predictions.track == track].copy()
        calibration = frame[frame.month.isin(['2025-12', '2026-01'])]
        residuals = np.abs(np.log(calibration.sold_price / calibration.predicted_price))
        rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * 0.9)))
        q = float(np.sort(residuals)[rank - 1])
        frozen = pd.read_csv(ROOT / 'research_v2' / 'frozen_check' / f'v2_{track}.csv')
        evaluations = {'monthly_retraining': frame[frame.month >= '2026-02'].copy(), 'frozen_four_months': frozen.copy()}
        for setting, evaluation in evaluations.items():
            evaluation['lower_price'] = evaluation.predicted_price * np.exp(-q)
            evaluation['upper_price'] = evaluation.predicted_price * np.exp(q)
            evaluation['covered'] = (evaluation.sold_price >= evaluation.lower_price) & (evaluation.sold_price <= evaluation.upper_price)
            evaluation.to_csv(OUT / f'intervals_{setting}_{track}.csv', index=False)
            for month, group in [('all', evaluation), *evaluation.groupby('month')]:
                intervals.append({'track': track, 'setting': setting, 'month': month, 'n': len(group), 'calibration_n': len(calibration), 'log_radius': q, 'coverage_percent': float(group.covered.mean() * 100), 'median_width_dollars': float((group.upper_price - group.lower_price).median())})
        frame['price_band'] = pd.cut(frame.sold_price, bins=[0, 500000, 1000000, 2000000, np.inf], labels=['under_500k', '500k_to_1m', '1m_to_2m', 'over_2m'], right=False)
        for column in ['month', 'house_type_name', 'price_band']:
            for group, subset in frame.groupby(column, observed=True):
                errors.append({'track': track, 'grouping': column, 'group': str(group), **metrics(subset)})
        print(track, json.dumps(metrics(frame)), flush=True)
    pd.DataFrame(errors).to_csv(OUT / 'error_breakdown.csv', index=False)
    coverage = pd.DataFrame(intervals)
    coverage.to_csv(OUT / 'interval_coverage.csv', index=False)
    print(coverage[coverage.month == 'all'].to_string(index=False))
