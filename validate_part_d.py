
import sqlite3

from refund_tools import (
    DB_PATH,
    STATE_DB,
    init_state,
    lookup_order,
    calculate_refund,
    issue_refund,
    policy_search
)

from refund_agent import process_refund

print("=" * 55)
print("PART D: FINAL VALIDATION")
print("=" * 55)

init_state()

with sqlite3.connect(DB_PATH) as conn:
    eligible = conn.execute("""
        SELECT order_id
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes > 45
          AND order_amount > 0
          AND order_amount <= 1000
        LIMIT 1
    """).fetchone()

    on_time = conn.execute("""
        SELECT order_id
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes <= 45
        LIMIT 1
    """).fetchone()

    high_value = conn.execute("""
        SELECT order_id
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes > 45
          AND order_amount > 1000
        LIMIT 1
    """).fetchone()

    cancelled = conn.execute("""
        SELECT order_id
        FROM orders
        WHERE status = 'cancelled'
        LIMIT 1
    """).fetchone()


# Test 1: Eligible refund
assert eligible is not None
order = lookup_order(eligible[0])
decision = calculate_refund(order)
assert decision["eligible"] is True
assert decision["amount"] == round(order["order_amount"], 2)
print("PASS: Eligible refund calculation")


# Test 2: On-time delivery
assert on_time is not None
decision = calculate_refund(lookup_order(on_time[0]))
assert decision["eligible"] is False
assert decision["amount"] == 0
print("PASS: On-time delivery rejected")


# Test 3: Cancelled order
assert cancelled is not None
decision = calculate_refund(lookup_order(cancelled[0]))
assert decision["eligible"] is False
print("PASS: Cancelled order rejected")


# Test 4: Nonexistent order
assert lookup_order("ORD-NOT-FOUND") is None
result = process_refund("ORD-NOT-FOUND")
assert result["status"] == "rejected"
print("PASS: Nonexistent order rejected")


# Test 5: High-value refund
if high_value:
    result = process_refund(high_value[0])
    assert result["status"] == "escalated"
    print("PASS: High-value refund escalated")
else:
    print("SKIP: No high-value eligible order available")


# Test 6: Duplicate protection
result = issue_refund(eligible[0], order["order_amount"])
assert result["status"] in ["issued", "duplicate"]

second_result = issue_refund(eligible[0], order["order_amount"])
assert second_result["status"] == "duplicate"
print("PASS: Duplicate refund prevented")


# Test 7: Incorrect refund amount
wrong_amount = round(order["order_amount"] + 10, 2)
result = issue_refund(eligible[0], wrong_amount)
assert result["status"] == "rejected"
print("PASS: Incorrect refund amount rejected")


# Test 8: Policy retrieval
policies = policy_search("late delivery refund policy")
assert len(policies) == 3
assert all("source" in p for p in policies)
print("PASS: Policy retrieval")


# Test 9: State persistence
with sqlite3.connect(STATE_DB) as conn:
    saved = conn.execute(
        "SELECT COUNT(*) FROM refunds WHERE order_id = ?",
        (eligible[0],)
    ).fetchone()[0]

assert saved == 1
print("PASS: Refund state persisted")


print("\nALL PART D VALIDATION CHECKS PASSED")
