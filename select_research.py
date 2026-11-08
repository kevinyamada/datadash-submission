import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import score
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2'

def collect():
    rows = [json.loads(p.read_text()) for p in (OUT / 'predictions').glob('*.json')]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['stage', 'track', 'name', 'status', 'mdape', 'mape', 'bias', 'seconds'])
    frame.to_csv(OUT / 'all_experiments.csv', index=False)
    return frame[frame.status == 'ok'].copy()

def shortlist(frame):
    selected = {}
    cat = json.loads((OUT / 'catalogue.json').read_text())
    for track in ['off_market', 'list_aware']:
        rows = frame[(frame.stage == 'screen') & (frame.track == track)].sort_values(['mdape', 'mape'])
        names = list(rows.head(4).name)
        for family in ['cat', 'lgb', 'gam', 'xgb']:
            candidates = rows[rows.name.map(lambda n: cat[n]['family'] == family)]
            if len(candidates):
                name = candidates.iloc[0]['name']
                if name not in names:
                    names.append(name)
        if 'incumbent' not in names:
            names.append('incumbent')
        selected[track] = names
    (OUT / 'shortlist.json').write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2))

def select(frame):
    shortlist = json.loads((OUT / 'shortlist.json').read_text())
    catalogue = json.loads((OUT / 'catalogue.json').read_text())
    chosen = {}
    all_rows = []
    for track, names in shortlist.items():
        predictions = {}
        for stage in ['screen', 'confirm']:
            base = None
            predictions[stage] = {}
            for name in names:
                path = OUT / 'predictions' / f'{stage}_{track}_{name}.csv'
                if not path.exists():
                    raise ValueError(f'Missing {path}')
                p = pd.read_csv(path).sort_values('ml_num')
                if base is None:
                    base = p
                if not np.array_equal(base.ml_num, p.ml_num):
                    raise ValueError('IDs do not match')
                predictions[stage][name] = p.predicted_price.to_numpy()
            predictions[stage]['actual'] = base.sold_price.to_numpy()
        candidates = [(name,) for name in names] + list(itertools.combinations(names, 2))
        rows = []
        for members in candidates:
            row = {'track': track, 'members': '+'.join(members), 'member_count': len(members)}
            for stage in ['screen', 'confirm']:
                p = np.exp(np.mean([np.log(predictions[stage][n]) for n in members], axis=0))
                metrics = score(predictions[stage]['actual'], p)
                for key in ['mdape', 'mape', 'bias']:
                    row[f'{stage}_{key}'] = metrics[key]
            row['mean_mdape'] = (row['screen_mdape'] + row['confirm_mdape']) / 2
            row['mean_mape'] = (row['screen_mape'] + row['confirm_mape']) / 2
            rows.append(row)
        ranking = pd.DataFrame(rows).sort_values(['mean_mdape', 'mean_mape'])
        best = ranking.iloc[0]
        single = ranking[ranking.member_count == 1].iloc[0]
        if single.mean_mdape - best.mean_mdape < 0.05:
            best = single
        members = best.members.split('+')
        chosen[track] = [{'name': name, 'weight': 1 / len(members), 'config': catalogue[name]} for name in members]
        all_rows.extend(rows)
        print(track, best.to_dict(), flush=True)
    pd.DataFrame(all_rows).to_csv(OUT / 'blend_comparison.csv', index=False)
    config = {'version': 2, 'selection': 'Equal-weight geometric means tested on two earlier frozen windows; select mean MdAPE, prefer single model if improvement below 0.05 percentage points. No December-May results used for this selection.', 'tracks': chosen}
    (ROOT / 'model_config_v2.json').write_text(json.dumps(config, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['summary', 'shortlist', 'select'])
    args = parser.parse_args()
    frame = collect()
    if args.action == 'shortlist':
        shortlist(frame)
    elif args.action == 'select':
        select(frame)
    else:
        print(frame[['stage', 'track', 'name', 'mdape', 'mape', 'bias', 'seconds']].sort_values(['track', 'stage', 'mdape']).to_string(index=False))
if __name__ == '__main__':
    main()
