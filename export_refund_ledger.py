
import json
import sqlite3
from pathlib import Path

from refund_tools import STATE_DB, init_state


OUTPUT_PATH = Path("outputs/refund_ledger.jsonl")


def export_refund_ledger():
    init_state()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with sqlite3.connect(STATE_DB) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                order_id,
                amount,
                status,
                created_at
            FROM refunds
            ORDER BY created_at, order_id
        """).fetchall()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            record = {
                "order_id": row["order_id"],
                "refund_inr": row["amount"],
                "status": row["status"],
                "created_at": row["created_at"],
            }

            file.write(
                json.dumps(record) + "\n"
            )

    print("Refund ledger exported:", OUTPUT_PATH)
    print("Total refund records:", len(rows))


if __name__ == "__main__":
    export_refund_ledger()
