from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "model_outputs"

OUTPUT.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "listings.csv")

for column in ["date_listed", "date_sold"]:
    train[column] = pd.to_datetime(train[column])

CATEGORICAL = [
    "house_type_name",
    "house_style",
    "municipality",
    "community",
]

NUMERIC = [
    "bedrooms",
    "washrooms",
    "parking_total",
    "garage",
    "latitude",
    "longitude",
    "lot_front",
    "lot_depth",
]

PATTERNS = {
    "renovated": r"renovat|fully updated|fully redone|turnkey",
    "needs_work": r"as[- ]is|needs work|needs updating|worn|dated|overhaul",
    "original": r"original|older|vintage",
    "maintained": r"well[- ]maintained|cared for|good condition|move[- ]in",
    "luxury": r"quartz|marble|custom|designer|spa[- ]like|heated",
    "basement": r"basement",
    "transit": r"transit|subway|commut",
    "outdoor": r"balcony|terrace|backyard|garden",
}


def make_features(frame, track):
    features = frame[CATEGORICAL + NUMERIC].copy()

    area = (
        frame["house_area"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(
            r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*$"
        )
        .astype(float)
    )

    features["area_sqft"] = (
        area[0] + area[1].fillna(area[0])
    ) / 2

    features["area_is_range"] = area[1].notna().astype(int)
    features["area_range_width"] = area[1] - area[0]

    features["lot_area"] = (
        frame["lot_front"] * frame["lot_depth"]
    )

    features["area_per_bedroom"] = (
        features["area_sqft"] / frame["bedrooms"].clip(lower=1)
    )

    features["months_since_2021"] = (
        frame["date_listed"] - pd.Timestamp("2021-01-01")
    ).dt.days / 30.44

    month = frame["date_listed"].dt.month

    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    remarks = frame["remarks"].fillna("").str.lower()

    for name, pattern in PATTERNS.items():
        features[f"text_{name}"] = (
            remarks.str.contains(pattern, regex=True).astype(int)
        )

    if track == "list_aware":
        features["log_list_price"] = np.log(frame["list_price"])
        features["list_price_per_sqft"] = (
            frame["list_price"] / features["area_sqft"]
        )

    return features.replace([np.inf, -np.inf], np.nan)


def prepare_features(past, future, track):
    x_train = make_features(past, track)
    x_future = make_features(future, track)

    for column in CATEGORICAL:
        categories = sorted(
            x_train[column].dropna().astype(str).unique()
        )

        x_train[column] = pd.Categorical(
            x_train[column], categories=categories
        )

        x_future[column] = pd.Categorical(
            x_future[column], categories=categories
        )

    return x_train, x_future


def fit_predict(past, future, track):
    x_train, x_future = prepare_features(past, future, track)

    target = np.log(past["sold_price"].to_numpy())

    if track == "list_aware":
        target = target - np.log(past["list_price"].to_numpy())

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=60,
        reg_lambda=5,
        colsample_bytree=0.9,
        random_state=465,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )

    model.fit(x_train, target)

    predictions = np.exp(model.predict(x_future))

    if track == "list_aware":
        predictions = predictions * future["list_price"].to_numpy()

    return predictions


def off_market_baseline(past, future, cutoff):
    recent = past[
        past["date_sold"] >= cutoff - pd.DateOffset(months=12)
    ]

    community_medians = recent.groupby(
        ["house_type_name", "community"]
    )["sold_price"].median()

    type_medians = recent.groupby(
        "house_type_name"
    )["sold_price"].median()

    keys = pd.MultiIndex.from_frame(
        future[["house_type_name", "community"]]
    )

    predictions = pd.Series(
        community_medians.reindex(keys).to_numpy(),
        index=future.index,
    )

    predictions = predictions.fillna(
        future["house_type_name"].map(type_medians)
    )

    return predictions.fillna(
        recent["sold_price"].median()
    ).to_numpy()


def metrics(frame):
    actual = frame["sold_price"].to_numpy()
    predicted = frame["predicted_price"].to_numpy()

    signed_error = 100 * (predicted - actual) / actual
    absolute_error = np.abs(signed_error)

    return {
        "homes": len(frame),
        "MdAPE_percent": np.median(absolute_error),
        "MAPE_percent": np.mean(absolute_error),
        "MAE_dollars": np.mean(np.abs(predicted - actual)),
        "median_bias_percent": np.median(signed_error),
    }


def summarize(frame, columns):
    rows = []

    for keys, group in frame.groupby(columns, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(columns, keys))
        row.update(metrics(group))
        rows.append(row)

    return pd.DataFrame(rows)


assert train["ml_num"].is_unique
assert (train["sold_price"] > 0).all()
assert (train["list_price"] > 0).all()
assert train["date_sold"].notna().all()
assert train["date_listed"].notna().all()

results = []

for month in pd.period_range("2025-12", "2026-05", freq="M"):
    cutoff = month.start_time
    end = (month + 1).start_time

    past = train[train["date_sold"] < cutoff].copy()

    future = train[
        (train["date_sold"] >= cutoff)
        & (train["date_sold"] < end)
    ].copy()

    assert len(past) > 0
    assert len(future) > 0
    assert past["date_sold"].max() < future["date_sold"].min()

    print(
        f"\n{month}: training on {len(past):,} sales, "
        f"validating on {len(future):,} sales",
        flush=True,
    )

    predictions = {
        "off_market_baseline": off_market_baseline(
            past, future, cutoff
        ),
        "list_aware_baseline": future["list_price"].to_numpy(),
    }

    for track in ["off_market", "list_aware"]:
        print(f"Training {track} model...", flush=True)
        predictions[f"{track}_model"] = fit_predict(
            past, future, track
        )

    for name, predicted in predictions.items():
        assert np.isfinite(predicted).all()
        assert (predicted > 0).all()

        result = future[
            [
                "ml_num",
                "sold_price",
                "house_type_name",
                "community",
                "date_listed",
            ]
        ].copy()

        result["month"] = str(month)
        result["model"] = name
        result["predicted_price"] = predicted
        result["listed_after_cutoff"] = (
            future["date_listed"] >= cutoff
        )

        results.append(result)

        scores = metrics(result)

        print(
            f"{name}: "
            f"MdAPE = {scores['MdAPE_percent']:.2f}% | "
            f"MAPE = {scores['MAPE_percent']:.2f}%",
            flush=True,
        )

validation = pd.concat(results, ignore_index=True)

validation.to_csv(
    OUTPUT / "validation_predictions.csv",
    index=False,
)

overall = summarize(validation, ["model"])

by_month = summarize(
    validation, ["model", "month"]
)

by_type = summarize(
    validation, ["model", "house_type_name"]
)

by_community = summarize(
    validation, ["model", "community"]
)

validation["price_band"] = pd.cut(
    validation["sold_price"],
    bins=[0, 500000, 1000000, 2000000, np.inf],
    labels=["Under 500k", "500k to 1m", "1m to 2m", "Over 2m"],
)

by_price = summarize(
    validation, ["model", "price_band"]
)

listing_safe = summarize(
    validation[validation["listed_after_cutoff"]],
    ["model"],
)

tables = {
    "overall_scores": overall,
    "scores_by_month": by_month,
    "scores_by_property_type": by_type,
    "scores_by_community": by_community,
    "scores_by_price_band": by_price,
    "scores_listing_safe_subset": listing_safe,
}

report = [
    "Monthly walk-forward validation: December 2025 to May 2026.",
    "All training sales precede the evaluated sale month.",
    "Scores are historical validation results, not hidden-test results.",
    "The listing-safe subset also requires listing on or after the cutoff.",
    "The subset contains different homes and is not directly comparable.",
    "Fixed model settings were used without hyperparameter search.",
]

for name, table in tables.items():
    table.to_csv(OUTPUT / f"{name}.csv", index=False)

    if name != "scores_by_community":
        report.append(f"\n{name}\n")
        report.append(
            table.round(3).to_string(index=False)
        )

report_text = "\n".join(report)

(OUTPUT / "model_summary.txt").write_text(
    report_text,
    encoding="utf-8",
)

print("\n" + report_text)
print(f"\nFinished. Results saved in:\n{OUTPUT}")