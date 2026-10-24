
from pathlib import Path
import pandas as pd

# Project folders
RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw data
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")

print("Raw customers shape:", customers.shape)
print("Raw products shape:", products.shape)

# 1. Standardise city names
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# 2. Standardise gender values
def clean_gender(value):
    value = str(value).strip().lower()

    if value in {"m", "male", "man"}:
        return "male"
    if value in {"f", "female", "woman"}:
        return "female"
    return "unknown"

customers["gender"] = customers["gender"].apply(clean_gender)

# 3. Convert invalid ages to missing values
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = float("nan")

# Fill missing ages using the median age within each city
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# 4. Convert signup date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Save cleaned files
customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)

print("\nCleaned city counts:")
print(customers["city"].value_counts())

print("\nCleaned gender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Customer and product files saved in data/clean/")

# 5. Load raw orders
orders = pd.read_csv(RAW / "orders.csv")

# Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Keep the latest record for each order_id
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
)

# Convert amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Convert quantity and discount to numeric
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Get product prices and discounts
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Fill missing or invalid amounts using product price
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# Make negative delivery times missing
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
orders.loc[
    orders["delivery_minutes"] < 0, "delivery_minutes"
] = float("nan")

# Fill missing delivery times for completed orders
completed = orders["status"].eq("completed")
category_median = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(category_median)
)

# Cancelled orders must not have delivery times
orders.loc[
    orders["status"].eq("cancelled"), "delivery_minutes"
] = float("nan")

# Remove temporary product columns
orders = orders.drop(
    columns=["price_inr", "category"],
    errors="ignore"
)


# Save cleaned orders
orders.to_csv(CLEAN / "orders.csv", index=False)

print("\nCleaned orders:", orders.shape)
print("Order status counts:")
print(orders["status"].value_counts())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times in completed orders:",
    orders.loc[orders["status"].eq("completed"),
               "delivery_minutes"].isna().sum()
)
print("Cleaned orders saved to data/clean/orders.csv")


# 6. Save cleaned data into SQLite database
import sqlite3

DB_PATH = Path("data/instamart.db")

# Store order timestamps in the required text format
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Save all three tables to SQLite
with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Verify table row counts
    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}", conn
        )["total"].iloc[0]
        print(f"Database table {table}: {count} rows")

print(f"SQLite database saved to {DB_PATH}")
