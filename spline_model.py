import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

class SplineRegressor:

    def __init__(self, alpha=100, text=False):
        self.alpha = alpha
        self.text = text

    def fit(self, x, y, categories, remarks, weight=None):
        cats = list(categories)
        continuous = [c for c in ['log_area', 'months_since_2021', 'latitude', 'longitude'] if c in x and x[c].nunique() > 5]
        others = [c for c in x if c not in cats + continuous]
        transformers = [('cat', OneHotEncoder(handle_unknown='ignore'), cats), ('other', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), others)]
        if continuous:
            transformers.append(('curves', make_pipeline(SimpleImputer(strategy='median'), SplineTransformer(n_knots=8, degree=3, extrapolation='linear'), StandardScaler()), continuous))
        self.preprocessor = ColumnTransformer(transformers, sparse_threshold=1)
        base = self.preprocessor.fit_transform(x)
        self.type_encoder = OneHotEncoder(handle_unknown='ignore')
        types = self.type_encoder.fit_transform(x[['house_type_name']])
        area = x.log_area.to_numpy()
        self.area_mean = np.nanmean(area)
        interaction = types.multiply((area - self.area_mean)[:, None])
        self.community_encoder = OneHotEncoder(handle_unknown='ignore')
        communities = self.community_encoder.fit_transform(x[['community']])
        time = x.months_since_2021.to_numpy() / 12
        self.time_mean = time.mean()
        trend = communities.multiply((time - self.time_mean)[:, None])
        blocks = [base, interaction, trend]
        if self.text:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_features=12000, sublinear_tf=True, dtype=np.float32)
            blocks.append(self.vectorizer.fit_transform(remarks.fillna('')) * 3)
        self.model = Ridge(alpha=self.alpha, solver='lsqr', tol=1e-05)
        self.model.fit(sparse.hstack(blocks, format='csr'), y, sample_weight=weight)
        return self

    def predict(self, x, remarks):
        blocks = [self.preprocessor.transform(x), self.type_encoder.transform(x[['house_type_name']]).multiply((x.log_area.to_numpy() - self.area_mean)[:, None]), self.community_encoder.transform(x[['community']]).multiply((x.months_since_2021.to_numpy() / 12 - self.time_mean)[:, None])]
        if self.text:
            blocks.append(self.vectorizer.transform(remarks.fillna('')) * 3)
        return self.model.predict(sparse.hstack(blocks, format='csr'))
