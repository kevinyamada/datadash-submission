from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "eda_outputs"

DATA.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/Baliyang/SSU_data_dash/resolve/main"
NAMES = ["listings", "test_off_market", "test_list_aware"]

datasets = {}
hashes = {}
summary = []

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def record(title, value):
    text = f"\n{title}\n{'=' * len(title)}\n{value}\n"
    print(text)
    summary.append(text)


def save_table(name, table):
    table.to_csv(OUTPUT / f"{name}.csv")
    record(name, table.to_string())


def save_chart(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close()


for name in NAMES:
    path = DATA / f"{name}.csv"

    if not path.exists():
        print(f"Downloading {name}...")
        temporary = path.with_suffix(".download")
        urlretrieve(f"{BASE_URL}/{name}.csv", temporary)
        temporary.replace(path)

    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)

    for column in [
        "date_listed",
        "date_sold",
        "external_avm_estimate_date",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    datasets[name] = frame

train = datasets["listings"].copy()

for name, frame in datasets.items():
    record(f"{name}: shape", frame.shape)

    schema = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_percent": frame.isna().mean() * 100,
            "unique_values": frame.nunique(),
        }
    )

    save_table(f"{name}_schema", schema)

    record(
        f"{name}: duplicate checks",
        {
            "duplicate_ids": int(frame["ml_num"].duplicated().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
        },
    )

    frame.head(10).to_csv(OUTPUT / f"{name}_sample.csv", index=False)

train_ids = set(train["ml_num"])
off_ids = set(datasets["test_off_market"]["ml_num"])
list_ids = set(datasets["test_list_aware"]["ml_num"])

record(
    "ID overlap",
    {
        "train_and_off_market": len(train_ids & off_ids),
        "train_and_list_aware": len(train_ids & list_ids),
        "between_test_sets": len(off_ids & list_ids),
    },
)

for name in ["test_off_market", "test_list_aware"]:
    record(
        f"{name}: training columns absent from test",
        sorted(set(train.columns) - set(datasets[name].columns)),
    )

area_text = train["house_area"].astype("string").str.replace(",", "", regex=False)

area_parts = area_text.str.extract(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*$"
).astype(float)

train["area_sqft"] = (
    area_parts[0] + area_parts[1].fillna(area_parts[0])
) / 2

train["area_is_range"] = area_parts[1].notna()
train["lot_area_sqft"] = train["lot_front"] * train["lot_depth"]
train["sale_month"] = train["date_sold"].dt.to_period("M").astype(str)

train["sold_to_list_ratio"] = (
    train["sold_price"] / train["list_price"].where(train["list_price"] > 0)
)

train["price_per_sqft"] = (
    train["sold_price"] / train["area_sqft"].where(train["area_sqft"] > 0)
)

train["computed_days_on_market"] = (
    train["date_sold"] - train["date_listed"]
).dt.days

record(
    "Data quality checks",
    {
        "nonpositive_sold_price": int((train["sold_price"] <= 0).sum()),
        "nonpositive_list_price": int((train["list_price"] <= 0).sum()),
        "listing_after_sale": int(
            (train["date_listed"] > train["date_sold"]).sum()
        ),
        "days_on_market_mismatches": int(
            (
                train["computed_days_on_market"].notna()
                & train["days_on_market"].notna()
                & (
                    train["computed_days_on_market"]
                    != train["days_on_market"]
                )
            ).sum()
        ),
        "unparsed_nonmissing_area": int(
            (
                train["house_area"].notna()
                & train["area_sqft"].isna()
            ).sum()
        ),
        "area_range_percent": float(train["area_is_range"].mean() * 100),
        "repeated_coordinate_rows": int(
            train.duplicated(["latitude", "longitude"]).sum()
        ),
        "earliest_sale": str(train["date_sold"].min()),
        "latest_sale": str(train["date_sold"].max()),
        "number_of_communities": int(train["community"].nunique()),
    },
)

numeric_columns = [
    "sold_price",
    "list_price",
    "bedrooms",
    "washrooms",
    "parking_total",
    "garage",
    "area_sqft",
    "lot_front",
    "lot_depth",
    "lot_area_sqft",
    "days_on_market",
    "sold_to_list_ratio",
    "price_per_sqft",
]

save_table(
    "numeric_summary",
    train[numeric_columns].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T,
)

property_summary = train.groupby("house_type_name").agg(
    count=("sold_price", "size"),
    median_price=("sold_price", "median"),
    mean_price=("sold_price", "mean"),
    median_area=("area_sqft", "median"),
    median_price_per_sqft=("price_per_sqft", "median"),
    median_sold_to_list=("sold_to_list_ratio", "median"),
)

save_table(
    "property_type_summary",
    property_summary.sort_values("count", ascending=False),
)

save_table(
    "lot_missingness_by_type",
    train.groupby("house_type_name")[["lot_front", "lot_depth"]].agg(
        lambda values: values.isna().mean() * 100
    ),
)

community_summary = train.groupby("community").agg(
    count=("sold_price", "size"),
    median_price=("sold_price", "median"),
    median_area=("area_sqft", "median"),
    median_price_per_sqft=("price_per_sqft", "median"),
)

save_table(
    "community_summary",
    community_summary.sort_values("median_price", ascending=False),
)

municipality_summary = train.groupby("municipality").agg(
    count=("sold_price", "size"),
    median_price=("sold_price", "median"),
    median_area=("area_sqft", "median"),
)

save_table("municipality_summary", municipality_summary)

monthly_summary = train.groupby("sale_month").agg(
    count=("sold_price", "size"),
    median_price=("sold_price", "median"),
    median_sold_to_list=("sold_to_list_ratio", "median"),
    median_days_on_market=("days_on_market", "median"),
)

save_table("monthly_summary", monthly_summary)

monthly_types = train.pivot_table(
    index="sale_month",
    columns="house_type_name",
    values="sold_price",
    aggfunc="median",
)

save_table("monthly_prices_by_type", monthly_types)

valid_avm_dates = (
    train["external_avm_estimate_date"].notna()
    & train["date_sold"].notna()
)

avm_after_sale = (
    train["external_avm_estimate_date"] > train["date_sold"]
)

record(
    "External AVM availability audit",
    {
        "missing_estimates": int(train["external_avm_estimate"].isna().sum()),
        "valid_date_pairs": int(valid_avm_dates.sum()),
        "dated_after_sale_count": int(avm_after_sale.sum()),
        "dated_after_sale_percent_of_valid_dates": (
            float(avm_after_sale[valid_avm_dates].mean() * 100)
            if valid_avm_dates.any()
            else None
        ),
        "available_in_off_market_test": (
            "external_avm_estimate"
            in datasets["test_off_market"].columns
        ),
        "available_in_list_aware_test": (
            "external_avm_estimate"
            in datasets["test_list_aware"].columns
        ),
    },
)

patterns = {
    "renovation_mentions": r"renovat|fully updated|fully redone|turnkey",
    "repair_mentions": r"as[- ]is|needs work|needs updating|worn|dated|overhaul",
    "original_finishes": r"original|older|vintage",
    "luxury_mentions": r"quartz|marble|custom|designer|spa[- ]like|heated",
    "transit_mentions": r"transit|subway|commut",
}

remarks = train["remarks"].fillna("").str.lower()
text_rows = []

for name, pattern in patterns.items():
    flag = remarks.str.contains(pattern, regex=True)

    text_rows.append(
        {
            "feature": name,
            "matching_rows": int(flag.sum()),
            "matching_percent": float(flag.mean() * 100),
            "median_price_with_flag": train.loc[flag, "sold_price"].median(),
            "median_price_without_flag": train.loc[~flag, "sold_price"].median(),
        }
    )

save_table("description_keyword_summary", pd.DataFrame(text_rows).set_index("feature"))

correlations = train[
    [
        "sold_price",
        "bedrooms",
        "washrooms",
        "parking_total",
        "garage",
        "area_sqft",
        "lot_area_sqft",
        "latitude",
        "longitude",
    ]
].corr(method="spearman")

save_table("spearman_correlations", correlations)

positive_prices = train.loc[train["sold_price"] > 0, "sold_price"]
plot_limit = positive_prices.quantile(0.99)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(
    positive_prices[positive_prices <= plot_limit] / 1_000_000,
    bins=60,
    color="#167d9a",
)
axes[0].set_title("Sale prices: values up to the 99th percentile")
axes[0].set_xlabel("Sale price ($ millions)")
axes[0].set_ylabel("Number of homes")

axes[1].hist(np.log10(positive_prices), bins=60, color="#e18a43")
axes[1].set_title("All positive sale prices on a log scale")
axes[1].set_xlabel("log10(sale price)")
axes[1].set_ylabel("Number of homes")

save_chart("01_price_distributions")

fig, ax = plt.subplots(figsize=(9, 5))

ordered_types = property_summary.sort_values("median_price")
ax.barh(
    ordered_types.index,
    ordered_types["median_price"] / 1_000_000,
    color="#167d9a",
)
ax.set_xlabel("Median sale price ($ millions)")
ax.set_title("Median sale price by property type: all training years")

save_chart("02_prices_by_property_type")

month_dates = pd.to_datetime(monthly_summary.index)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(
    month_dates,
    monthly_summary["median_price"] / 1_000_000,
    color="#167d9a",
)
axes[0].set_ylabel("Median price ($ millions)")
axes[0].set_title("Monthly sale prices and sample counts")

axes[1].bar(month_dates, monthly_summary["count"], width=22, color="#e18a43")
axes[1].set_ylabel("Number of sales")
axes[1].set_xlabel("Sale month")

save_chart("03_monthly_prices_and_counts")

fig, ax = plt.subplots(figsize=(11, 5))

for property_type in monthly_types.columns:
    ax.plot(
        pd.to_datetime(monthly_types.index),
        monthly_types[property_type] / 1_000_000,
        label=property_type,
    )

ax.set_ylabel("Median sale price ($ millions)")
ax.set_xlabel("Sale month")
ax.set_title("Monthly prices by type: changing sale composition can affect medians")
ax.legend(fontsize=8, ncol=2)

save_chart("04_monthly_prices_by_type")

scatter_data = train[
    (train["area_sqft"] > 0)
    & (train["sold_price"] > 0)
].sample(
    n=min(
        10000,
        int(((train["area_sqft"] > 0) & (train["sold_price"] > 0)).sum()),
    ),
    random_state=465,
)

fig, ax = plt.subplots(figsize=(9, 6))

for property_type, group in scatter_data.groupby("house_type_name"):
    ax.scatter(
        group["area_sqft"],
        group["sold_price"],
        s=8,
        alpha=0.25,
        label=property_type,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Floor area (sq ft, log scale)")
ax.set_ylabel("Sale price ($, log scale)")
ax.set_title("Floor area versus price: sample of up to 10,000 homes")
ax.legend(fontsize=8)

save_chart("05_area_vs_price")

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    month_dates,
    monthly_summary["median_sold_to_list"],
    color="#167d9a",
)
ax.axhline(1, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Median sold price / asking price")
ax.set_xlabel("Sale month")
ax.set_title("Sale-to-asking-price ratio over time")

save_chart("06_sold_to_list_over_time")

fig, ax = plt.subplots(figsize=(9, 7))

matrix = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(
    range(len(correlations.columns)),
    correlations.columns,
    rotation=55,
    ha="right",
)
ax.set_yticks(range(len(correlations.index)), correlations.index)
ax.set_title("Spearman correlations: associations, not causal effects")
fig.colorbar(matrix, ax=ax, label="Correlation")

save_chart("07_correlations")

record(
    "Interpretation cautions",
    "\n".join(
        [
            "All homes and locations are synthetic.",
            "Description keyword comparisons are unadjusted associations.",
            "Monthly medians can change because the mix of homes changes.",
            "Price-per-square-foot calculations use approximated range midpoints.",
            "Do not use sold_price-derived EDA columns as prediction features.",
            "Do not use date_sold or days_on_market as prediction features.",
            "Do not use list_price in the off-market model.",
            "External AVM estimates are absent from both test sets.",
            "Fit learned preprocessing only on the training portion of each time split.",
            "No prediction model has been trained by this script.",
        ]
    ),
)

(OUTPUT / "source_hashes.json").write_text(
    json.dumps(hashes, indent=2),
    encoding="utf-8",
)

(OUTPUT / "eda_summary.txt").write_text(
    "\n".join(summary),
    encoding="utf-8",
)

print(f"\nFinished. All outputs are saved in:\n{OUTPUT}")