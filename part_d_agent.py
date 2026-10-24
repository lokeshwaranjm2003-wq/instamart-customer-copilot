
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/instamart.db")

print("=" * 55)
print("PART D: REFUND AGENT - DATABASE INSPECTION")
print("=" * 55)

with sqlite3.connect(DB_PATH) as conn:
    # Check orders table structure
    columns = pd.read_sql_query(
        "PRAGMA table_info(orders)", conn
    )

    print("\nOrders table columns:")
    print(columns[["name", "type"]].to_string(index=False))

    # Check completed order count
    completed = pd.read_sql_query(
        """
        SELECT COUNT(*) AS total
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    )

    print("\nCompleted orders:", completed["total"].iloc[0])

    # Check refund-relevant data
    summary = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_completed,
            SUM(CASE WHEN delivery_minutes > 45
                THEN 1 ELSE 0 END) AS late_orders,
            SUM(CASE WHEN delivery_minutes <= 45
                THEN 1 ELSE 0 END) AS on_time_orders
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    )

    print("\nRefund eligibility summary:")
    print(summary.to_string(index=False))

print("\nREFUND AGENT DATABASE INSPECTION COMPLETED")
