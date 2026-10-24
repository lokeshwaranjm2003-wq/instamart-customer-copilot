
import sqlite3

from refund_tools import (
    DB_PATH,
    lookup_order,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    policy_search
)


def process_refund(order_id):
    print("\n" + "=" * 55)
    print("REFUND REQUEST:", order_id)
    print("=" * 55)

    # Step 1: Lookup the order
    order = lookup_order(order_id)

    if order is None:
        print("Decision: REJECTED - Order not found")
        return {
            "status": "rejected",
            "reason": "Order not found"
        }

    print("Order status:", order["status"])
    print("Order amount:", order["order_amount"])
    print("Delivery minutes:", order["delivery_minutes"])

    # Step 2: Retrieve relevant policy context
    policies = policy_search(
        "late delivery refund eligibility and human approval"
    )

    print("\nRetrieved policy sources:")
    for policy in policies:
        print("-", policy["source"])

    # Step 3: Calculate eligibility
    decision = calculate_refund(order)

    if not decision["eligible"]:
        print("\nDecision: REJECTED")
        print("Reason:", decision["reason"])

        return {
            "status": "rejected",
            "reason": decision["reason"]
        }

    amount = decision["amount"]

    # Step 4: Human approval for large refunds
    if amount > 1000:
        result = escalate_to_human(
            order_id,
            "Refund exceeds INR 1000"
        )

        print("\nDecision: ESCALATED")
        return result

    # Step 5: Issue eligible refund
    result = issue_refund(order_id, amount)

    print("\nDecision:", result["status"].upper())
    print("Result:", result)

    return result


def find_test_orders():
    with sqlite3.connect(DB_PATH) as conn:
        eligible = conn.execute("""
            SELECT order_id FROM orders
            WHERE status = 'completed'
              AND delivery_minutes > 45
              AND order_amount > 0
              AND order_amount <= 1000
            LIMIT 1
        """).fetchone()

        ineligible = conn.execute("""
            SELECT order_id FROM orders
            WHERE status = 'completed'
              AND delivery_minutes <= 45
            LIMIT 1
        """).fetchone()

        high_value = conn.execute("""
            SELECT order_id FROM orders
            WHERE status = 'completed'
              AND delivery_minutes > 45
              AND order_amount > 1000
            LIMIT 1
        """).fetchone()

    return (
        eligible[0] if eligible else None,
        ineligible[0] if ineligible else None,
        high_value[0] if high_value else None
    )


if __name__ == "__main__":
    eligible, ineligible, high_value = find_test_orders()

    if eligible:
        print("\nTEST 1: ELIGIBLE REFUND")
        process_refund(eligible)

        print("\nTEST 2: DUPLICATE REFUND")
        process_refund(eligible)

    if ineligible:
        print("\nTEST 3: INELIGIBLE REFUND")
        process_refund(ineligible)

    if high_value:
        print("\nTEST 4: HIGH-VALUE REFUND")
        process_refund(high_value)
    else:
        print("\nNo high-value eligible sample found.")

    print("\nTEST 5: NONEXISTENT ORDER")
    process_refund("ORD-NOT-FOUND")

    print("\nREFUND AGENT TESTS COMPLETED")
