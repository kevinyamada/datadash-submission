"""Competition models and preserved starter baselines."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import features

def make_previous_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parent / 'model_config.json').read_text())
    return CompetitionModel(track, area_offset=track == 'off_market' and config['off_market_area_offset'])

def make_model(track):
    import json
    from pathlib import Path
    from advanced_models import BlendModel
    config = json.loads((Path(__file__).parent / 'model_config_v2.json').read_text())
    return BlendModel(track, config['tracks'][track])

class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self

    def predict(self, homes):
        return homes['list_price'].to_numpy(dtype=float)

class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ['ml_num', 'house_type_name', 'date_sold', 'latitude', 'longitude', 'sold_price']
        self.sales = train[columns].sort_values(['date_sold', 'ml_num'], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby('house_type_name')))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home['house_type_name'], self.sales)
        earlier = sales[sales['date_sold'] < home['date_listed']]
        if len(earlier) == 0:
            earlier = self.sales[self.sales['date_sold'] < home['date_listed']]
        dx = (earlier['longitude'].to_numpy() - home['longitude']) * 0.72
        dy = earlier['latitude'].to_numpy() - home['latitude']
        distance = np.sqrt(dx ** 2 + dy ** 2)
        nearest = np.argsort(distance, kind='stable')[:self.k]
        return float(np.median(earlier['sold_price'].to_numpy()[nearest]))

class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train['sold_price']))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))

class CompetitionModel:

    def __init__(self, track, area_offset=False, use_text=True):
        if track not in ('off_market', 'list_aware'):
            raise ValueError('Unknown track')
        self.track = track
        self.area_offset = area_offset
        self.use_text = use_text

    def transform(self, homes, fitting=False):
        import pandas as pd
        x = features.make_features(homes, self.track)
        if not self.use_text:
            x = x.drop(columns=[c for c in x if c.startswith('text_')])
        if fitting:
            self.categories = {c: sorted(x[c].dropna().astype(str).unique()) for c in features.CATEGORICAL}
        for c, categories in self.categories.items():
            x[c] = pd.Categorical(x[c], categories=categories)
        return x

    def offset(self, homes):
        if self.track == 'list_aware':
            value = homes['list_price'].to_numpy(dtype=float)
        elif self.area_offset:
            value = features.square_feet(homes['house_area']).fillna(self.area_fill).clip(lower=1).to_numpy()
        else:
            value = np.ones(len(homes))
        if not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Offsets must be finite and positive')
        return value

    def fit(self, train):
        from lightgbm import LGBMRegressor
        self.area_fill = float(features.square_feet(train['house_area']).median())
        x = self.transform(train, fitting=True)
        target = np.log(train['sold_price'].to_numpy()) - np.log(self.offset(train))
        self.model = LGBMRegressor(objective='regression_l1', n_estimators=350, learning_rate=0.05, num_leaves=31, min_child_samples=60, reg_lambda=5, colsample_bytree=0.9, random_state=465, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
        self.model.fit(x, target)
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.transform(homes))) * self.offset(homes)
