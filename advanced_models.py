import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import features
ROOT = Path(__file__).resolve().parent

def enhanced_features(frame, track):
    x = features.make_features(frame, track)
    x['text_needs_work'] = frame.remarks.fillna('').str.lower().str.contains('as[- ]is|needs work|needs updating|\\bworn\\b|\\bdated\\b|overhaul', regex=True).astype(int)
    text = frame.remarks.fillna('').str.lower()
    patterns = {'new_finishes': 'new kitchen|new bathroom|new flooring|fresh paint|updated kitchen|updated bathroom', 'major_renovation': 'fully renovat|fully redone|fully updated|completely renovat|top.to.bottom|gut.renovat', 'minor_renovation': 'partially updated|some updates|cosmetic|refreshed', 'poor_condition': 'needs renovation|needs repair|fixer|handyman|tlc|requires updating|requires work|water damage', 'good_condition': 'pristine|immaculate|meticulous|well.kept|well.cared', 'unfinished': 'unfinished|original finishes|original kitchen|dated kitchen|older finishes', 'premium': 'high.end|premium|luxur|chef|designer|custom.built|high.ceiling', 'view': 'unobstructed|panoramic|lake view|ravine|waterfront', 'income': 'separate entrance|rental income|income potential|in.law|secondary suite', 'light': 'bright|natural light|sun.filled|south.facing', 'corner': 'corner unit|end.unit|corner lot', 'fees': 'maintenance fee|monthly fee|utilities included'}
    for name, pattern in patterns.items():
        x['text_' + name] = text.str.contains(pattern, regex=True).astype(int)
    x['log_area'] = np.log(x.area_sqft.clip(lower=1))
    x['log_lot_area'] = np.log1p(x.lot_area.clip(lower=0))
    x['washrooms_per_bedroom'] = x.washrooms / x.bedrooms.clip(lower=1)
    x['area_per_washroom'] = x.area_sqft / x.washrooms.clip(lower=1)
    x['lot_to_floor'] = x.lot_area / x.area_sqft.clip(lower=1)
    x['community_type'] = frame.community.astype(str) + '|' + frame.house_type_name.astype(str)
    x['type_style'] = frame.house_type_name.astype(str) + '|' + frame.house_style.astype(str)
    x['location_cell'] = frame.latitude.round(3).astype(str) + '|' + frame.longitude.round(3).astype(str)
    x['listing_month'] = frame.date_listed.dt.month
    return x.replace([np.inf, -np.inf], np.nan)

class AdvancedModel:

    def __init__(self, track, config):
        self.track = track
        self.config = dict(config)
        self.text_model = None

    def numeric_features(self, frame):
        x = features.make_features(frame, self.track) if self.config.get('features', 'base') == 'base' else enhanced_features(frame, self.track)
        if self.config.get('features') == 'market':
            keep = ['house_type_name', 'house_style', 'community', 'months_since_2021', 'month_sin', 'month_cos']
            if self.track == 'list_aware':
                keep.append('log_list_price')
            x = x[keep]
        if self.config.get('comparables'):
            x = pd.concat([x, self.comparables.transform(frame)], axis=1)
        return x

    def text_features(self, train, frame=None):
        if not self.config.get('svd', 0):
            return None
        n = self.config['svd']
        if frame is None:
            digest = hashlib.sha256(pd.util.hash_pandas_object(train[['ml_num', 'remarks']], index=False).values.tobytes()).hexdigest()[:20]
            path = ROOT / 'research_v2' / 'cache' / f'text_{digest}_{n}.joblib'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                self.text_model, values = joblib.load(path)
                return values
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_df=0.98, max_features=6000, sublinear_tf=True, dtype=np.float32)
            matrix = vectorizer.fit_transform(train.remarks.fillna(''))
            svd = TruncatedSVD(n_components=n, n_iter=5, random_state=465)
            values = svd.fit_transform(matrix)
            self.text_model = (vectorizer, svd)
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            joblib.dump((self.text_model, values), temporary)
            temporary.replace(path)
            return values
        vectorizer, svd = self.text_model
        return svd.transform(vectorizer.transform(frame.remarks.fillna('')))

    def prepare(self, frame, fitting=False):
        x = self.numeric_features(frame)
        if self.config.get('svd', 0):
            values = self.text_features(frame) if fitting else self.text_features(None, frame)
            for k in range(values.shape[1]):
                x[f'text_svd_{k}'] = values[:, k]
        if fitting:
            self.categories = {c: sorted(x[c].dropna().astype(str).unique()) for c in x.select_dtypes(include=['object', 'string']).columns}
        family = self.config['family']
        for c, levels in self.categories.items():
            if family in ['cat', 'gam']:
                x[c] = x[c].fillna('__MISSING__').astype(str)
            elif family == 'lgb' or (family == 'xgb' and self.config.get('native_categories')):
                x[c] = pd.Categorical(x[c], categories=levels)
            else:
                x[c] = pd.Series(pd.Categorical(x[c], categories=levels).codes.astype(float), index=x.index).replace(-1, np.nan)
        if family in ['cat', 'extra', 'ridge', 'gam']:
            if fitting:
                self.fill = x.drop(columns=list(self.categories)).median().fillna(0)
            for c, v in self.fill.items():
                x[c] = x[c].fillna(v)
        if self.config.get('raw_text'):
            x['remarks'] = frame.remarks.fillna('empty').replace('', 'empty')
        return x

    def offset(self, frame):
        if self.track == 'list_aware':
            return frame.list_price.to_numpy(dtype=float)
        if self.config.get('area_offset', False):
            return features.square_feet(frame.house_area).fillna(self.area_fill).clip(lower=1).to_numpy()
        return np.ones(len(frame))

    def fit(self, train):
        cfg = self.config
        self.calibration_factor = 1.0
        if cfg.get('calibration_months'):
            import data
            cutoff = train.date_sold.max() - pd.DateOffset(months=cfg['calibration_months'])
            earlier = train[train.date_sold < cutoff]
            calibration = train[train.date_sold >= cutoff]
            inner_config = {k: v for k, v in cfg.items() if k not in ['calibration_months', 'calibration_shrink']}
            inner = AdvancedModel(self.track, inner_config).fit(earlier)
            residual = np.log(calibration.sold_price.to_numpy() / inner.predict(data.hide_answers(calibration, self.track)))
            self.calibration_factor = float(np.exp(cfg.get('calibration_shrink', 0.5) * np.median(residual)))
            del inner
        train = train.copy()
        if cfg.get('recent_months'):
            train = train[train.date_sold >= train.date_sold.max() - pd.DateOffset(months=cfg['recent_months'])]
        if cfg.get('comparables'):
            from temporal_features import TemporalComparables
            self.comparables = TemporalComparables(self.track).fit(train)
        self.area_fill = features.square_feet(train.house_area).median()
        x = self.prepare(train, True)
        target = np.log(train.sold_price.to_numpy()) - np.log(self.offset(train))
        weight = None
        if cfg.get('half_life'):
            age = (train.date_sold.max() - train.date_sold).dt.days.to_numpy() / 30.44
            weight = np.power(0.5, age / cfg['half_life'])
        family = cfg['family']
        params = dict(cfg.get('params', {}))
        threads = int(os.environ.get('DASH_THREADS', '4'))
        if family == 'lgb':
            from lightgbm import LGBMRegressor
            defaults = dict(objective='regression_l1', n_estimators=1000, learning_rate=0.04, num_leaves=31, min_child_samples=60, reg_lambda=5, colsample_bytree=0.9, random_state=465, n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)
            defaults.update(params)
            self.model = LGBMRegressor(**defaults)
            self.model.fit(x, target, sample_weight=weight)
        elif family == 'cat':
            from catboost import CatBoostRegressor
            defaults = dict(iterations=1200, depth=7, learning_rate=0.05, loss_function='RMSE', l2_leaf_reg=5, random_seed=465, thread_count=threads, verbose=False, allow_writing_files=False)
            defaults.update(params)
            if cfg.get('raw_text'):
                defaults['feature_calcers'] = ['BoW']
            self.model = CatBoostRegressor(**defaults)
            self.model.fit(x, target, cat_features=list(self.categories), text_features=['remarks'] if cfg.get('raw_text') else None, sample_weight=weight)
        elif family == 'xgb':
            from xgboost import XGBRegressor
            defaults = dict(n_estimators=1000, max_depth=6, learning_rate=0.04, min_child_weight=30, subsample=0.9, colsample_bytree=0.9, reg_lambda=10, objective='reg:squarederror', tree_method='hist', random_state=465, n_jobs=threads)
            defaults.update(params)
            defaults['enable_categorical'] = cfg.get('native_categories', False)
            self.model = XGBRegressor(**defaults)
            self.model.fit(x, target, sample_weight=weight)
        elif family == 'hist':
            from sklearn.ensemble import HistGradientBoostingRegressor
            defaults = dict(max_iter=600, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=5, early_stopping=False, random_state=465, categorical_features=list(self.categories))
            defaults.update(params)
            self.model = HistGradientBoostingRegressor(**defaults)
            self.model.fit(x, target, sample_weight=weight)
        elif family == 'extra':
            from sklearn.ensemble import ExtraTreesRegressor
            defaults = dict(n_estimators=250, min_samples_leaf=5, max_features=0.8, random_state=465, n_jobs=threads)
            defaults.update(params)
            self.model = ExtraTreesRegressor(**defaults)
            self.model.fit(x, target, sample_weight=weight)
        elif family == 'gam':
            from spline_model import SplineRegressor
            self.model = SplineRegressor(alpha=params.get('alpha', 100), text=cfg.get('text_ridge', False))
            self.model.fit(x, target, self.categories, train.remarks, weight)
        elif family == 'ridge':
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            cats = list(self.categories)
            nums = [c for c in x if c not in cats]
            self.preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cats), ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), nums)])
            matrix = self.preprocessor.fit_transform(x)
            self.model = Ridge(alpha=params.get('alpha', 100), solver='lsqr')
            self.model.fit(matrix, target, sample_weight=weight)
        else:
            raise ValueError(f'Unknown family {family}')
        return self

    def predict(self, homes):
        x = self.prepare(homes)
        if self.config['family'] == 'gam':
            return self.calibration_factor * np.exp(self.model.predict(x, homes.remarks)) * self.offset(homes)
        if self.config['family'] == 'ridge':
            x = self.preprocessor.transform(x)
        return self.calibration_factor * np.exp(self.model.predict(x)) * self.offset(homes)

class BlendModel:

    def __init__(self, track, members):
        self.track = track
        self.members = members

    def fit(self, train):
        self.models = [AdvancedModel(self.track, item['config']).fit(train) for item in self.members]
        return self

    def predict(self, homes):
        weights = np.array([m['weight'] for m in self.members])
        weights = weights / weights.sum()
        return np.exp(sum((w * np.log(m.predict(homes)) for w, m in zip(weights, self.models))))
