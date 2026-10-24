
from pathlib import Path
import sqlite3
import pandas as pd

clean = Path("data/clean")

customers = pd.read_csv(clean / "customers.csv")
products = pd.read_csv(clean / "products.csv")
orders = pd.read_csv(clean / "orders.csv")

checks = []

# Customer checks
checks.append(len(customers) == 1200)
checks.append(len(products) == 40)
checks.append(customers["age"].isna().sum() == 0)
checks.append(customers["city"].nunique() == 6)
checks.append(set(customers["gender"].unique()).issubset(
    {"male", "female", "unknown"}
))

# Order checks
checks.append(len(orders) == 12309)
checks.append(orders["order_id"].nunique() == 12309)
checks.append((orders["status"] == "completed").sum() == 11110)
checks.append((orders["status"] == "cancelled").sum() == 1199)
checks.append(orders["order_amount"].isna().sum() == 0)

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

checks.append(completed["delivery_minutes"].isna().sum() == 0)
checks.append(cancelled["delivery_minutes"].isna().sum() == 1199)

# SQLite database checks
with sqlite3.connect("data/instamart.db") as conn:
    for table, expected in [
        ("customers", 1200),
        ("products", 40),
        ("orders", 12309),
    ]:
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table}", conn
        )["n"].iloc[0]
        checks.append(actual == expected)

if all(checks):
    print("ALL CHECKPOINTS PASSED")
else:
    print(f"VALIDATION FAILED: {checks.count(False)} check(s) failed")
    for i, passed in enumerate(checks, start=1):
        print(f"Check {i}: {'PASS' if passed else 'FAIL'}")
