import numpy as np
import pandas as pd
import pytest
from advanced_models import AdvancedModel, enhanced_features
from test_starter import make_listings
import data

@pytest.mark.parametrize('family', ['lgb', 'cat', 'xgb', 'extra', 'ridge'])
def test_models_reject_leakage_and_predict_unknown_categories(family):
    train = make_listings(300)
    params = {'n_estimators': 8} if family in ['lgb', 'xgb', 'extra'] else {'iterations': 8} if family == 'cat' else {}
    model = AdvancedModel('off_market', {'family': family, 'features': 'enhanced', 'params': params}).fit(train)
    homes = make_listings(10, seed=11).assign(community='Unseen')
    clean = model.predict(data.hide_answers(homes, 'off_market'))
    dirty = homes.assign(list_price=1000000000000.0, sold_price=1000000000000000.0, days_on_market=999, date_sold=pd.Timestamp('2100-01-01'), external_avm_estimate=1000000000000000.0)
    np.testing.assert_allclose(clean, model.predict(dirty), rtol=1e-12, atol=1e-08)
    assert np.isfinite(clean).all()
    assert (clean > 0).all()

def test_updated_is_not_dated():
    homes = make_listings(2).assign(remarks=['An updated kitchen', 'A dated kitchen'])
    assert enhanced_features(homes, 'off_market').text_needs_work.tolist() == [0, 1]

def test_svd_training_vocabulary_excludes_unseen_text():
    train = make_listings(300)
    train['remarks'] = ['bright kitchen nice home' if i % 2 else 'original house garden basement' for i in range(300)]
    model = AdvancedModel('off_market', {'family': 'lgb', 'features': 'enhanced', 'svd': 2, 'params': {'n_estimators': 5}}).fit(train)
    homes = make_listings(10).assign(remarks='neverobservedword')
    assert 'neverobservedword' not in model.text_model[0].vocabulary_
    assert np.isfinite(model.predict(homes)).all()

def test_comparables_ignore_same_day_and_future_sales():
    from temporal_features import TemporalComparables
    train = make_listings(8).assign(date_sold=pd.Timestamp('2025-01-01'), house_type_name='Detached', community='A', house_area='1000', sold_price=500000)
    home = train.iloc[[0]].assign(date_listed=pd.Timestamp('2025-02-01'))
    later = train.iloc[[0]].assign(date_sold=pd.Timestamp('2025-02-01'), sold_price=1000000000000.0)
    a = TemporalComparables('off_market').fit(train).transform(home)
    b = TemporalComparables('off_market').fit(pd.concat([train, later], ignore_index=True)).transform(home)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_allclose(a.prior_community_type_365, np.log(500))

def test_comparables_expire_old_sales():
    from temporal_features import rolling_medians
    median, count = rolling_medians(np.array([0, 1, 2, 90]), np.array([1, 2, 3, 7]), np.array([100]), 30, min_count=1)
    assert median[0] == 7 and count[0] == 1

def test_spline_model_handles_unknown_categories():
    train = make_listings(300)
    model = AdvancedModel('off_market', {'family': 'gam', 'features': 'enhanced'}).fit(train)
    homes = make_listings(10).assign(community='Unseen')
    assert np.isfinite(model.predict(data.hide_answers(homes, 'off_market'))).all()

def test_recent_calibration_uses_training_only():
    train = make_listings(400)
    cfg = {'family': 'lgb', 'features': 'base', 'calibration_months': 3, 'params': {'n_estimators': 5}}
    model = AdvancedModel('off_market', cfg).fit(train)
    assert np.isfinite(model.calibration_factor) and model.calibration_factor > 0
    homes = make_listings(10, seed=13)
    np.testing.assert_allclose(model.predict(homes), model.predict(data.hide_answers(homes, 'off_market')))
