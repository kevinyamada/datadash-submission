import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis' / 'eda'

def area_midpoint(series):
    values = series.astype('string').str.replace(',', '', regex=False).str.extract('^\\s*(\\d+(?:\\.\\d+)?)\\s*(?:-\\s*(\\d+(?:\\.\\d+)?))?\\s*$').astype(float)
    return (values[0] + values[1].fillna(values[0])) / 2

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(ROOT / 'data_local' / f'{name}.csv', low_memory=False) for name in ['listings', 'test_off_market', 'test_list_aware']}
    summary = {}
    for name, frame in frames.items():
        for col in ['date_listed', 'date_sold', 'external_avm_estimate_date']:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors='coerce')
        frame['area_midpoint_eda'] = area_midpoint(frame.house_area)
        pd.DataFrame({'dtype': frame.dtypes.astype(str), 'missing': frame.isna().sum(), 'missing_pct': frame.isna().mean() * 100, 'unique': frame.nunique()}).to_csv(OUT / f'{name}_schema.csv')
        frame.select_dtypes('number').describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_csv(OUT / f'{name}_numeric_summary.csv')
        summary[name] = {'rows': len(frame), 'original_columns': len(frame.columns) - 1, 'duplicate_ids': int(frame.ml_num.duplicated().sum()), 'invalid_area': int(frame.area_midpoint_eda.isna().sum()), 'listed_min': str(frame.date_listed.min()), 'listed_max': str(frame.date_listed.max())}
    train = frames['listings']
    summary['overlap'] = {name: len(set(train.ml_num) & set(frame.ml_num)) for name, frame in frames.items() if name != 'listings'}
    summary['test_overlap'] = len(set(frames['test_off_market'].ml_num) & set(frames['test_list_aware'].ml_num))
    summary['quality'] = {'nonpositive_prices': int((train.sold_price <= 0).sum()), 'sale_before_listing': int((train.date_sold < train.date_listed).sum()), 'external_avm_after_sale_pct': float((train.external_avm_estimate_date > train.date_sold).mean() * 100), 'area_ranges_pct': float(train.house_area.astype(str).str.contains('-').mean() * 100), 'repeated_coordinates': int(train.duplicated(['latitude', 'longitude']).sum()), 'communities': int(train.community.nunique())}
    train['month'] = train.date_sold.dt.to_period('M').astype(str)
    train['sold_to_list_eda'] = train.sold_price / train.list_price
    train['price_per_sqft_eda'] = train.sold_price / train.area_midpoint_eda
    monthly = train.groupby('month').agg(n=('sold_price', 'size'), median_price=('sold_price', 'median'), mean_price=('sold_price', 'mean'), median_sold_to_list=('sold_to_list_eda', 'median'))
    monthly.to_csv(OUT / 'monthly.csv')
    for group in ['house_type_name', 'community', 'municipality']:
        train.groupby(group).agg(n=('sold_price', 'size'), median_price=('sold_price', 'median'), price_p10=('sold_price', lambda x: x.quantile(0.1)), price_p90=('sold_price', lambda x: x.quantile(0.9)), median_area=('area_midpoint_eda', 'median'), median_price_per_sqft=('price_per_sqft_eda', 'median')).to_csv(OUT / f'by_{group}.csv')
        pd.concat({name: frame[group].value_counts(normalize=True) for name, frame in frames.items()}, axis=1).fillna(0).to_csv(OUT / f'{group}_distribution.csv')
    train.groupby('house_type_name')[['lot_front', 'lot_depth']].agg(lambda x: x.isna().mean() * 100).to_csv(OUT / 'lot_missingness.csv')
    train.select_dtypes('number').corr(method='spearman')['sold_price'].sort_values(ascending=False).to_csv(OUT / 'price_correlations.csv')
    summary['price'] = {'median': float(train.sold_price.median()), 'mean': float(train.sold_price.mean()), 'max': float(train.sold_price.max())}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(np.log10(train.sold_price), bins=60, color='#236b8e')
    ax[0, 0].set(xlabel='Log10 sale price (CAD)', ylabel='Homes', title='Prices have a long upper tail')
    monthly.median_price.div(1000000.0).plot(ax=ax[0, 1], color='#236b8e')
    ax[0, 1].set(ylabel='Median sale price ($M)', xlabel='Sale month', title='Composition and market change over time')
    monthly.median_sold_to_list.plot(ax=ax[1, 0], color='#bd572d')
    ax[1, 0].axhline(1, color='gray', linestyle='--')
    ax[1, 0].set(ylabel='Median sold / list', xlabel='Sale month', title='Asking price is informative, but biased')
    grouped = train.groupby('house_type_name').sold_price.median().sort_values().div(1000000.0)
    grouped.plot.barh(ax=ax[1, 1], color='#236b8e')
    ax[1, 1].set(xlabel='Median sale price ($M)', ylabel='', title='Property types have different price levels')
    fig.tight_layout()
    fig.savefig(OUT / 'eda_overview.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    monthly.median_sold_to_list.plot(ax=axes[0], color='#bd572d', linewidth=2)
    axes[0].set_xticks(range(0, len(monthly), 12), [m[:4] for m in monthly.index[::12]])
    axes[0].axhline(1, color='gray', linestyle='--')
    axes[0].set(xlabel='Sale month', ylabel='Median sold / list', title='Asking-price calibration changes')
    grouped.plot.barh(ax=axes[1], color='#236b8e')
    axes[1].set(xlabel='Median sale price ($M)', ylabel='', title='Property types differ')
    fig.tight_layout()
    fig.savefig(OUT / 'eda_market.png', dpi=180)
    plt.close(fig)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
