"""Explicit property feature sets for the two competition tracks."""
import pandas as pd
PROPERTY_TYPES = ['Condo Apt', 'Detached', 'Semi-Detached', 'Condo Townhouse', 'Freehold Townhouse', 'Co-Op Apt']

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split('-', expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors='coerce')
    high = pd.to_numeric(parts[1], errors='coerce').fillna(low)
    return (low + high) / 2

def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X['square_feet'] = square_feet(homes['house_area'])
    X['bedrooms'] = homes['bedrooms']
    X['washrooms'] = homes['washrooms']
    X['parking'] = homes['parking_total']
    X['latitude'] = homes['latitude']
    X['longitude'] = homes['longitude']
    X['months'] = (homes['date_listed'] - pd.Timestamp('2021-01-01')).dt.days / 30.44
    for kind in PROPERTY_TYPES:
        X['is_' + kind] = (homes['house_type_name'] == kind).astype(int)
    return X

def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X['list_price'] = homes['list_price']
    return X
import numpy as np
CATEGORICAL = ['house_type_name', 'house_style', 'municipality', 'community']
NUMERIC = ['bedrooms', 'washrooms', 'parking_total', 'garage', 'latitude', 'longitude', 'lot_front', 'lot_depth']
PATTERNS = {'renovated': 'renovat|fully updated|fully redone|turnkey', 'needs_work': 'as[- ]is|needs work|needs updating|worn|dated|overhaul', 'original': 'original|older|vintage', 'maintained': 'well[- ]maintained|cared for|good condition|move[- ]in', 'luxury': 'quartz|marble|custom|designer|spa[- ]like|heated', 'basement': 'basement', 'transit': 'transit|subway|commut', 'outdoor': 'balcony|terrace|backyard|garden'}

def make_features(frame, track):
    features = frame[CATEGORICAL + NUMERIC].copy()
    area = frame['house_area'].astype('string').str.replace(',', '', regex=False).str.extract('^\\s*(\\d+(?:\\.\\d+)?)\\s*(?:-\\s*(\\d+(?:\\.\\d+)?))?\\s*$').astype(float)
    features['area_sqft'] = (area[0] + area[1].fillna(area[0])) / 2
    features['area_is_range'] = area[1].notna().astype(int)
    features['area_range_width'] = area[1] - area[0]
    features['lot_area'] = frame['lot_front'] * frame['lot_depth']
    features['area_per_bedroom'] = features['area_sqft'] / frame['bedrooms'].clip(lower=1)
    features['months_since_2021'] = (frame['date_listed'] - pd.Timestamp('2021-01-01')).dt.days / 30.44
    month = frame['date_listed'].dt.month
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    remarks = frame['remarks'].fillna('').str.lower()
    for name, pattern in PATTERNS.items():
        features[f'text_{name}'] = remarks.str.contains(pattern, regex=True).astype(int)
    if track == 'list_aware':
        features['log_list_price'] = np.log(frame['list_price'])
        features['list_price_per_sqft'] = frame['list_price'] / features['area_sqft']
    return features.replace([np.inf, -np.inf], np.nan)
