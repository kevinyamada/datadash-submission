import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import data
from advanced_models import AdvancedModel
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2'
WINDOWS = {'screen': ('2025-07-01', '2025-11-01'), 'confirm': ('2024-11-01', '2025-03-01')}

def score(actual, predicted):
    signed = 100 * (predicted - actual) / actual
    return {'n': len(actual), 'mdape': float(np.median(abs(signed))), 'mape': float(np.mean(abs(signed))), 'bias': float(np.median(signed)), 'p90_ape': float(np.quantile(abs(signed), 0.9))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=WINDOWS, default='screen')
    parser.add_argument('--track', choices=['off_market', 'list_aware'], required=True)
    parser.add_argument('--names', default='all')
    args = parser.parse_args()
    catalogue = json.loads((OUT / 'catalogue.json').read_text())
    names = list(catalogue) if args.names == 'all' else args.names.split(',')
    train_all = data.load('listings')
    start, end = map(pd.Timestamp, WINDOWS[args.stage])
    train = train_all[train_all.date_sold < start]
    val = train_all[(train_all.date_sold >= start) & (train_all.date_sold < end)]
    safe = data.hide_answers(val, args.track)
    folder = OUT / 'predictions'
    folder.mkdir(exist_ok=True)
    for name in names:
        cfg = catalogue[name]
        signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        dest = folder / f'{args.stage}_{args.track}_{name}.csv'
        meta = dest.with_suffix('.json')
        if meta.exists() and dest.exists():
            old = json.loads(meta.read_text())
            if old['signature'] == signature and old.get('status') == 'ok':
                print('SKIP', args.stage, args.track, name, flush=True)
                continue
        began = time.monotonic()
        record = {'stage': args.stage, 'track': args.track, 'name': name, 'signature': signature, 'train_before': str(start.date()), 'validation_until_exclusive': str(end.date()), 'train_n': len(train)}
        try:
            model = AdvancedModel(args.track, cfg).fit(train)
            pred = model.predict(safe)
            if not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError('Invalid predictions')
            frame = pd.DataFrame({'ml_num': val.ml_num, 'sold_price': val.sold_price, 'predicted_price': pred, 'month': val.date_sold.dt.to_period('M').astype(str)})
            frame.to_csv(dest, index=False)
            record.update(score(val.sold_price.to_numpy(), pred))
            record['status'] = 'ok'
            del model
        except Exception as exc:
            record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        record['seconds'] = time.monotonic() - began
        meta.write_text(json.dumps(record, indent=2))
        with (OUT / f'journal_{args.stage}_{args.track}.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
if __name__ == '__main__':
    main()
