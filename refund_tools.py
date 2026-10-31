
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/instamart.db")
STATE_DIR = Path("state")
STATE_DIR.mkdir(exist_ok=True)
STATE_DB = STATE_DIR / "refunds_test.db"

MAX_AUTO_REFUND = 1000.0
LATE_DELIVERY_MINUTES = 45


def init_state():
    with sqlite3.connect(STATE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS refunds (
                order_id TEXT PRIMARY KEY,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def lookup_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, customer_id, status,
                   order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    return dict(row) if row else None


def calculate_refund(order):
    if order is None:
        return {
            "eligible": False,
            "amount": 0.0,
            "reason": "Order not found"
        }

    if order["status"] != "completed":
        return {
            "eligible": False,
            "amount": 0.0,
            "reason": "Order is not completed"
        }

    delivery = order["delivery_minutes"]
    amount = order["order_amount"]

    if delivery is None or amount is None:
        return {
            "eligible": False,
            "amount": 0.0,
            "reason": "Missing delivery or amount data"
        }

    if float(delivery) <= LATE_DELIVERY_MINUTES:
        return {
            "eligible": False,
            "amount": 0.0,
            "reason": "Delivery was within 45 minutes"
        }

    return {
        "eligible": True,
        "amount": round(float(amount), 2),
        "reason": "Late delivery exceeds 45 minutes"
    }


def issue_refund(order_id, amount):
    init_state()

    order = lookup_order(order_id)
    decision = calculate_refund(order)

    if not decision["eligible"]:
        return {
            "status": "rejected",
            "reason": decision["reason"]
        }

    if abs(float(amount) - decision["amount"]) > 0.001:
        return {
            "status": "rejected",
            "reason": "Refund amount does not match policy"
        }

    if amount > MAX_AUTO_REFUND:
        return {
            "status": "escalated",
            "reason": "Human approval required above INR 1000"
        }

    with sqlite3.connect(STATE_DB) as conn:
        existing = conn.execute(
            "SELECT status FROM refunds WHERE order_id = ?",
            (order_id,)
        ).fetchone()

        if existing:
            return {
                "status": "duplicate",
                "reason": "Refund already recorded"
            }

        conn.execute(
            """
            INSERT INTO refunds
            (order_id, amount, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                order_id,
                amount,
                "issued",
                datetime.now(timezone.utc).isoformat()
            )
        )

    return {
        "status": "issued",
        "order_id": order_id,
        "amount": amount
    }


def escalate_to_human(order_id, reason):
    return {
        "status": "escalated",
        "order_id": order_id,
        "reason": reason
    }


def policy_search(question):
    from part_d_rag import retrieve
    return retrieve(question, k=3)


if __name__ == "__main__":
    init_state()
    print("Refund state database initialized:", STATE_DB)

    with sqlite3.connect(DB_PATH) as conn:
        sample = conn.execute("""
            SELECT order_id
            FROM orders
            WHERE status = 'completed'
              AND delivery_minutes > 45
              AND order_amount > 0
              AND order_amount <= 1000
            LIMIT 1
        """).fetchone()

    if sample:
        order = lookup_order(sample[0])
        print("\nSample order:", order)
        print("Eligibility:", calculate_refund(order))
    else:
        print("No eligible sample order found.")

    print("\nREFUND TOOLS INITIALIZED")
