
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")
products = pd.read_csv(CLEAN / "products.csv")

# Keep completed orders only for revenue and spend analysis
completed = orders.loc[orders["status"].eq("completed")].copy()

# Make numeric columns safe for calculations
for col in ["order_amount", "quantity", "discount_pct", "delivery_minutes"]:
    completed[col] = pd.to_numeric(completed[col], errors="coerce")

print("=" * 55)
print("PART A: PANDAS CROSS-CHECK AND EXPLORATORY ANALYSIS")
print("=" * 55)

# 1. Pivot table: city x product category
city_orders = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)
city_orders = city_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

pivot = pd.pivot_table(
    city_orders,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
).round(2)

print("\n1. Revenue pivot table (grand total included):")
print(pivot)

# Compare pivot grand total with SQLite SQL total
with sqlite3.connect("data/instamart.db") as conn:
    sql_total = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS total
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    )["total"].iloc[0]

pandas_total = completed["order_amount"].sum()

print(f"\nSQL completed revenue:    INR {sql_total:,.2f}")
print(f"Pandas completed revenue: INR {pandas_total:,.2f}")

# 2. Sum, mean, and count by city
city_summary = city_orders.groupby("city").agg(
    revenue_sum=("order_amount", "sum"),
    revenue_mean=("order_amount", "mean"),
    order_count=("order_amount", "count")
).round(2)

print("\n2. Completed orders summary by city:")
print(city_summary)

# 3. Descriptive statistics
amounts = completed["order_amount"].dropna()

q1 = amounts.quantile(0.25)
median = amounts.median()
q3 = amounts.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\n3. Order amount statistics (completed orders):")
print(f"Mean: INR {amounts.mean():,.2f}")
print(f"Median: INR {median:,.2f}")
print(f"Skewness: {amounts.skew():.4f}")
print(f"Minimum: INR {amounts.min():,.2f}")
print(f"Q1 (25%): INR {q1:,.2f}")
print(f"Q2 (50%): INR {median:,.2f}")
print(f"Q3 (75%): INR {q3:,.2f}")
print(f"Maximum: INR {amounts.max():,.2f}")
print(f"IQR: INR {iqr:,.2f}")
print(f"Lower fence: INR {lower_fence:,.2f}")
print(f"Upper fence: INR {upper_fence:,.2f}")
print(f"Lower outliers: {(amounts < lower_fence).sum()}")
print(f"Upper outliers: {(amounts > upper_fence).sum()}")

# 4. Correlation matrix
corr_cols = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

corr_data = completed[corr_cols].apply(
    pd.to_numeric, errors="coerce"
)
correlation = corr_data.corr()

print("\n4. Correlation matrix:")
print(correlation.round(4))

# Find the pair with the largest absolute correlation
pairs = correlation.abs().where(
    pd.DataFrame(
        [[i != j for j in range(len(corr_cols))]
         for i in range(len(corr_cols))],
        index=corr_cols,
        columns=corr_cols
    )
).stack()

if not pairs.empty:
    strongest_pair = pairs.idxmax()
    actual_corr = correlation.loc[strongest_pair[0], strongest_pair[1]]
    print(
        f"\nLargest absolute correlation: "
        f"{strongest_pair[0]} and {strongest_pair[1]} "
        f"({actual_corr:.4f})"
    )

# Save analysis outputs
pivot.to_csv(OUTPUTS / "revenue_pivot.csv")
city_summary.to_csv(OUTPUTS / "city_summary.csv")
correlation.to_csv(OUTPUTS / "correlation_matrix.csv")

print("\nAnalysis files saved in outputs/")
