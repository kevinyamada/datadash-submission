import json
from pathlib import Path
import numpy as np
import pandas as pd
import data
import models
from experiments import score
ROOT = Path(__file__).resolve().parent

def recent_baseline(train, homes, cutoff):
    recent = train[train.date_sold >= cutoff - pd.DateOffset(months=12)]
    group = recent.groupby(['house_type_name', 'community']).sold_price.median()
    kinds = recent.groupby('house_type_name').sold_price.median()
    keys = pd.MultiIndex.from_frame(homes[['house_type_name', 'community']])
    p = pd.Series(group.reindex(keys).to_numpy(), index=homes.index)
    return p.fillna(homes.house_type_name.map(kinds)).fillna(recent.sold_price.median()).to_numpy()

def quantile(residuals, level=0.9):
    values = np.sort(np.asarray(residuals))
    rank = min(len(values), int(np.ceil((len(values) + 1) * level)))
    return float(values[rank - 1])

def main():
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    df = data.load('listings')
    pred = pd.read_csv(ROOT / 'outputs' / 'validation_predictions.csv')
    pred = pred.merge(df[['ml_num', 'community', 'date_listed']], on='ml_num', validate='many_to_one')
    pred['listed_after_cutoff'] = pred.date_listed >= pd.to_datetime(pred.month + '-01')
    pred['price_band'] = pd.cut(pred.sold_price, [0, 500000, 1000000, 2000000, np.inf], labels=['under $500k', '$500k-$1m', '$1m-$2m', 'over $2m'])
    pred['ape'] = 100 * abs(pred.predicted_price - pred.sold_price) / pred.sold_price
    pred['signed_error'] = 100 * (pred.predicted_price - pred.sold_price) / pred.sold_price
    for field in ['month', 'house_type_name', 'community', 'price_band', 'listed_after_cutoff']:
        rows = []
        for (track, value), g in pred.groupby(['track', field], observed=True):
            rows.append({'track': track, field: value, **score(g.sold_price.to_numpy(), g.predicted_price.to_numpy())})
        pd.DataFrame(rows).to_csv(out / f'scores_by_{field}.csv', index=False)
    records = []
    baseline_rows = []
    for month in pd.period_range('2025-12', '2026-05', freq='M'):
        past = df[df.date_sold < month.start_time]
        homes = df[df.date_sold.dt.to_period('M') == month]
        for track in ['off_market', 'list_aware']:
            p = recent_baseline(past, homes, month.start_time) if track == 'off_market' else homes.list_price.to_numpy()
            baseline_rows.append(pd.DataFrame({'track': track, 'sold_price': homes.sold_price, 'predicted_price': p}))
    base = pd.concat(baseline_rows)
    for track in ['off_market', 'list_aware']:
        for name, frame in [('baseline', base), ('selected', pred)]:
            g = frame[frame.track == track]
            records.append({'track': track, 'model': name, **score(g.sold_price.to_numpy(), g.predicted_price.to_numpy())})
    pd.DataFrame(records).to_csv(out / 'comparison.csv', index=False)
    cutoff = pd.Timestamp('2026-02-01')
    past = df[df.date_sold < cutoff]
    homes = df[(df.date_sold >= cutoff) & (df.date_sold < pd.Timestamp('2026-06-01'))]
    stress = []
    interval = []
    final_quantiles = {}
    for track in ['off_market', 'list_aware']:
        model = models.make_model(track).fit(past)
        p = model.predict(data.hide_answers(homes, track))
        cal = pred[(pred.track == track) & (pred.month < '2026-02')]
        q = quantile(abs(np.log(cal.sold_price / cal.predicted_price)))
        for name, frame in [('monthly_refit', pred[(pred.track == track) & (pred.month >= '2026-02')]), ('frozen_four_month', pd.DataFrame({'sold_price': homes.sold_price, 'predicted_price': p}))]:
            low = frame.predicted_price * np.exp(-q)
            high = frame.predicted_price * np.exp(q)
            interval.append({'track': track, 'evaluation': name, 'calibration_n': len(cal), 'calibration_months': '2025-12 to 2026-01', 'evaluation_months': '2026-02 to 2026-05', 'q_log': q, 'coverage_pct': 100 * float(((frame.sold_price >= low) & (frame.sold_price <= high)).mean()), 'median_width_relative_prediction_pct': 100 * (np.exp(q) - np.exp(-q)), **score(frame.sold_price.to_numpy(), frame.predicted_price.to_numpy())})
        b = recent_baseline(past, homes, cutoff) if track == 'off_market' else homes.list_price.to_numpy()
        stress.extend([{'track': track, 'model': 'selected', **score(homes.sold_price.to_numpy(), p)}, {'track': track, 'model': 'baseline', **score(homes.sold_price.to_numpy(), b)}])
        all_rows = pred[pred.track == track]
        final_quantiles[track] = quantile(abs(np.log(all_rows.sold_price / all_rows.predicted_price)))
    pd.DataFrame(stress).to_csv(out / 'frozen_four_month_scores.csv', index=False)
    pd.DataFrame(interval).to_csv(out / 'interval_evaluation.csv', index=False)
    (out / 'interval_quantiles.json').write_text(json.dumps(final_quantiles, indent=2))
    pred.nlargest(20, 'ape').to_csv(out / 'largest_errors.csv', index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(pd.DataFrame(stress).to_string(index=False))
    print(pd.DataFrame(interval).to_string(index=False))
if __name__ == '__main__':
    main()
