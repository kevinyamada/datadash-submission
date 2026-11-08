import json
from pathlib import Path
import numpy as np
import pandas as pd
import data
from models import CompetitionModel
ROOT = Path(__file__).resolve().parent

def score(y, p):
    error = (p - y) / y * 100
    return {'n': len(y), 'mdape': float(np.median(abs(error))), 'mape': float(np.mean(abs(error))), 'median_signed_error': float(np.median(error))}

def main():
    df = data.load('listings')
    cutoff = pd.Timestamp('2025-07-01')
    train = df[df.date_sold < cutoff]
    valid = df[(df.date_sold >= cutoff) & (df.date_sold < pd.Timestamp('2025-11-01'))]
    records = []
    for track, offset, text in [('off_market', False, True), ('off_market', True, True), ('off_market', False, False), ('list_aware', False, True), ('list_aware', False, False)]:
        model = CompetitionModel(track, area_offset=offset, use_text=text).fit(train)
        pred = model.predict(data.hide_answers(valid, track))
        row = {'track': track, 'area_offset': offset, 'use_text': text, 'train_before': str(cutoff.date()), 'evaluation': '2025-07 through 2025-10 frozen model', **score(valid.sold_price.to_numpy(), pred)}
        records.append(row)
        print(row, flush=True)
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out / 'development_experiments.csv', index=False)
    off = min([r for r in records if r['track'] == 'off_market' and r['use_text']], key=lambda r: r['mdape'])
    config = {'off_market_area_offset': off['area_offset'], 'selection': 'Lowest MdAPE among two prespecified off-market targets on July-October 2025; text ablations diagnostic only; no tuning on December-May scores.'}
    (ROOT / 'model_config.json').write_text(json.dumps(config, indent=2))
if __name__ == '__main__':
    main()
