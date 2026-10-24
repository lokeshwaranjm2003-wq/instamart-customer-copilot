
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SEPTEMBER_END = pd.Timestamp("2025-10-01 00:00:00")

customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")
products = pd.read_csv(CLEAN / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

for col in ["order_amount", "discount_pct", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

# Features must use only orders before 1 September 2025
before = orders[
    (orders["order_ts"] < CUTOFF)
    & orders["order_ts"].notna()
].copy()

completed_before = before[before["status"] == "completed"].copy()

# Cohort: customers with at least one completed order before cutoff
cohort_ids = completed_before["customer_id"].unique()

# Label window: 1 September through 30 September 2025
september_completed = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < SEPTEMBER_END)
    & (orders["status"] == "completed")
]

active_ids = set(september_completed["customer_id"])
cohort = pd.DataFrame({"customer_id": cohort_ids})
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_ids)
).astype(int)

# RFM-style features, calculated only from pre-cutoff completed orders
features = completed_before.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    average_discount=("discount_pct", "mean"),
    average_delivery_minutes=("delivery_minutes", "mean"),
).reset_index()

features["recency"] = (
    CUTOFF - features["last_order"]
).dt.days

features = features.drop(columns=["last_order"])

# Cancellation rate: cancelled / all final-status orders before cutoff
final_before = before[before["status"].isin(["completed", "cancelled"])]

final_counts = final_before.groupby("customer_id").agg(
    final_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status", lambda s: (s == "cancelled").sum()
    ),
).reset_index()

final_counts["cancellation_rate"] = (
    final_counts["cancelled_orders"]
    / final_counts["final_orders"]
)

# Favourite product category from completed pre-cutoff orders
with_products = completed_before.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite = (
    with_products.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_orders")
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Combine features with customer demographics
data = cohort.merge(features, on="customer_id", how="left")
data = data.merge(final_counts, on="customer_id", how="left")
data = data.merge(favourite, on="customer_id", how="left")
data = data.merge(customers, on="customer_id", how="left")

data["tenure_days"] = (
    CUTOFF - data["signup_date"]
).dt.days

# Keep required columns only
columns = [
    "customer_id", "lapsed", "recency", "frequency", "monetary",
    "average_discount", "cancellation_rate",
    "average_delivery_minutes", "favourite_category",
    "city", "age", "gender", "tenure_days"
]
data = data[columns]

# Fill missing values using cohort-derived values
numeric_cols = [
    "recency", "frequency", "monetary", "average_discount",
    "cancellation_rate", "average_delivery_minutes", "age",
    "tenure_days"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

for col in ["favourite_category", "city", "gender"]:
    data[col] = data[col].fillna("Unknown").astype(str)

# Check no feature value is missing
feature_cols = [c for c in data.columns if c not in ["customer_id", "lapsed"]]
assert not data[feature_cols].isna().any().any(), "Missing feature values found!"

# Stratified train/test split
X = data[feature_cols]
y = data["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

data.to_csv(OUTPUTS / "lapse_features.csv", index=False)

print("Cohort customers:", len(data))
print("Lapsed customers:", int(y.sum()))
print("Lapsed percentage:", round(y.mean() * 100, 2), "%")
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Missing feature values:", int(X.isna().sum().sum()))
print("Saved: outputs/lapse_features.csv")
