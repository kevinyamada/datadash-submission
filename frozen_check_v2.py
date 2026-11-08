from pathlib import Path
import numpy as np
import pandas as pd
import data
import models
from validate_v2 import make_model
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2' / 'frozen_check'
OUT.mkdir(parents=True, exist_ok=True)
if __name__ == '__main__':
    listings = data.load('listings')
    cutoff = pd.Timestamp('2026-02-01')
    train = listings[listings.date_sold < cutoff].copy()
    homes = listings[(listings.date_sold >= cutoff) & (listings.date_sold < pd.Timestamp('2026-06-01'))].copy()
    results = []
    for version, factory in [('previous', models.make_previous_model), ('v2', make_model)]:
        for track in ['off_market', 'list_aware']:
            print(f'Starting {version} {track}', flush=True)
            model = factory(track).fit(train)
            predicted = model.predict(data.hide_answers(homes, track))
            assert np.isfinite(predicted).all() and (predicted > 0).all()
            frame = homes[['ml_num', 'sold_price', 'house_type_name', 'community']].copy()
            frame['month'] = homes.date_sold.dt.strftime('%Y-%m')
            frame['version'] = version
            frame['track'] = track
            frame['predicted_price'] = np.round(predicted)
            frame.to_csv(OUT / f'{version}_{track}.csv', index=False)
            error = 100 * np.abs(frame.predicted_price / frame.sold_price - 1)
            results.append({'version': version, 'track': track, 'n': len(frame), 'mdape': float(np.median(error)), 'mape': float(np.mean(error))})
            print(results[-1], flush=True)
    pd.DataFrame(results).to_csv(OUT / 'scores.csv', index=False)
