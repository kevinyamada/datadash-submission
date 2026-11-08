from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    frame = pd.read_csv(ROOT / 'research_v2/official/outputs/validation_predictions.csv')
    prices = frame[frame.track == 'off_market'].sold_price
    low, high = prices.quantile([0.1, 0.9])
    frame['tail'] = np.select([frame.sold_price <= low, frame.sold_price >= high], ['Cheapest 10%', 'Most expensive 10%'], default='Middle 80%')
    frame['ape'] = 100 * (frame.predicted_price / frame.sold_price - 1).abs()
    frame['bias'] = 100 * (frame.predicted_price / frame.sold_price - 1)
    result = frame[frame['tail'] != 'Middle 80%'].groupby(['tail', 'house_type_name', 'track']).agg(n=('ml_num', 'size'), mdape=('ape', 'median'), bias=('bias', 'median')).reset_index()
    out = ROOT / 'analysis/final_checks'
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'price_extremes_by_type.csv', index=False)
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()
