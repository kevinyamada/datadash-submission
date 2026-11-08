from bisect import bisect_left, insort
import numpy as np
import pandas as pd
import features

def rolling_medians(sale_days, values, query_days, window, min_count=5):
    order = np.argsort(sale_days, kind='stable')
    days = np.asarray(sale_days)[order]
    values = np.asarray(values)[order]
    med = np.full(len(query_days), np.nan)
    counts = np.zeros(len(query_days))
    left = 0
    right = 0
    active = []
    for i in np.argsort(query_days, kind='stable'):
        day = query_days[i]
        while right < len(days) and days[right] < day:
            insort(active, float(values[right]))
            right += 1
        while left < right and days[left] < day - window:
            active.pop(bisect_left(active, float(values[left])))
            left += 1
        n = len(active)
        counts[i] = n
        if n >= min_count:
            med[i] = (active[(n - 1) // 2] + active[n // 2]) / 2
    return (med, counts)

class TemporalComparables:

    def __init__(self, track):
        self.track = track

    def fit(self, train):
        x = train.copy()
        if self.track == 'off_market':
            values = np.log(x.sold_price / features.square_feet(x.house_area))
        else:
            values = np.log(x.sold_price / x.list_price)
        x['_historical_value'] = values
        x['_sale_day'] = x.date_sold.to_numpy().astype('datetime64[D]').astype(np.int64)
        x = x[np.isfinite(x._historical_value)]
        self.sources = {}
        for name, cols, window in [('community_type_365', ['community', 'house_type_name'], 365), ('type_90', ['house_type_name'], 90), ('type_365', ['house_type_name'], 365)]:
            groups = {}
            for key, g in x.groupby(cols, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                groups[key] = (g._sale_day.to_numpy(), g._historical_value.to_numpy())
            self.sources[name] = (cols, window, groups)
        return self

    def transform(self, homes):
        result = pd.DataFrame(index=homes.index)
        queries = homes.date_listed.to_numpy().astype('datetime64[D]').astype(np.int64)
        for name, (cols, window, groups) in self.sources.items():
            medians = np.full(len(homes), np.nan)
            counts = np.zeros(len(homes))
            keys = {}
            for i, key in enumerate(homes[cols].itertuples(index=False, name=None)):
                keys.setdefault(key, []).append(i)
            for key, positions in keys.items():
                if key not in groups:
                    continue
                med, count = rolling_medians(*groups[key], queries[positions], window)
                medians[positions] = med
                counts[positions] = count
            result['prior_' + name] = medians
            result['prior_' + name + '_count'] = np.log1p(counts)
        result['prior_type_trend'] = result.prior_type_90 - result.prior_type_365
        return result
