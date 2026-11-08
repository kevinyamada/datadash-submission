import numpy as np
import pandas as pd
import pytest
import data
import features
import models
from test_starter import make_listings

def test_off_market_predictions_ignore_forbidden_inputs():
    train = make_listings(300)
    homes = make_listings(20, seed=5)
    model = models.CompetitionModel('off_market').fit(train)
    clean = model.predict(data.hide_answers(homes, 'off_market'))
    changed = homes.assign(sold_price=1000000000000.0, list_price=1000000000000.0, days_on_market=1000000.0, date_sold=pd.Timestamp('2100-01-01'), external_avm_estimate=1000000000000.0)
    np.testing.assert_array_equal(clean, model.predict(changed))

def test_categories_are_learned_from_training_only():
    train = make_listings(300)
    model = models.CompetitionModel('off_market').fit(train)
    unseen = make_listings(2).assign(community='Never observed')
    assert 'Never observed' not in model.categories['community']
    assert model.transform(unseen)['community'].isna().all()
    assert np.isfinite(model.predict(unseen)).all()

def test_area_parser():
    actual = features.make_features(make_listings(3).assign(house_area=['600-699', '1200', '1,500']), 'off_market')['area_sqft']
    np.testing.assert_allclose(actual, [649.5, 1200, 1500])

def test_list_aware_excludes_post_sale_values():
    train = make_listings(300)
    homes = make_listings(10, seed=9)
    model = models.CompetitionModel('list_aware').fit(train)
    np.testing.assert_array_equal(model.predict(homes), model.predict(data.hide_answers(homes, 'list_aware')))

def test_invalid_track_rejected():
    with pytest.raises(ValueError):
        models.CompetitionModel('other')
