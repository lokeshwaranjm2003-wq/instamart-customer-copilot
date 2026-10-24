
import json
import re
import random
import time

import pandas as pd
import mock_llm

ALLOWED = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only:
{"category": "...", "reason": "..."}
"""

FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only:
{"category": "...", "reason": "..."}

EXAMPLES:
Input: My account was hacked and someone placed an unauthorized order.
Output: {"category": "fraud", "reason": "unauthorized activity"}

Input: My delivery rider arrived late.
Output: {"category": "delivery", "reason": "late delivery"}

Input: The app crashes during login.
Output: {"category": "technical", "reason": "app problem"}
"""

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE = r"\b[6-9]\d{9}\b"
ORDER_ID = r"ORD-\d{6}"

def validate_response(content):
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", str(content).lower()))
        found = words & ALLOWED
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""

    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED:
        return "classified_json", parsed["category"]

    return "flagged_invalid_output", ""


def classify_with_retry(ticket_id, text, system_prompt):
    # Verify that only redacted text reaches the model
    assert not re.search(EMAIL, text)
    assert not re.search(PHONE, text)
    assert not re.search(ORDER_ID, text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    for retry in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry == 3:
                raise

            wait = 0.05 * (2 ** retry) * random.uniform(0.9, 1.15)
            print(
                f"Retry ticket={ticket_id}, "
                f"retry={retry}, wait={wait:.4f}s"
            )
            time.sleep(wait)


def run_classification(prompt, run_name):
    tickets = pd.read_csv(
        "data/raw/tickets.csv",
        keep_default_na=False
    )

    redacted = pd.read_csv(
        "outputs/redacted_tickets.csv",
        keep_default_na=False
    )

    tickets = tickets.merge(
        redacted,
        on=["ticket_id", "customer_id"],
        how="left",
        validate="one_to_one"
    )

    mock_llm.reset()
    results = []
    total_tokens = 0

    for _, row in tickets.iterrows():
        original = str(row["text"])
        clean = str(row["redacted_text"])

        if not original.strip():
            status, category, tokens = "skipped_empty", "", 0

        elif mock_llm.est_tokens(original) > 400:
            status, category, tokens = "flagged_long", "", 0

        else:
            response = classify_with_retry(
                row["ticket_id"],
                clean,
                prompt
            )

            tokens = response.usage.total_tokens
            total_tokens += tokens

            content = response.choices[0].message.content
            status, category = validate_response(content)

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": status,
            "category": category,
            "tokens_used": tokens
        })

    result = pd.DataFrame(results)

    classified = result[
        result["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if len(classified) else 0
    )

    print(f"\n{run_name.upper()} RESULTS")
    print(result["status"].value_counts())
    print("Classified tickets:", len(classified))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) else 0
    )

    if run_name == "zero_shot":
        print("\nZero-shot confusion table:")
        print(pd.crosstab(
            classified["true_category"],
            classified["category"]
        ))

    result.to_csv(
        f"outputs/{run_name}_classification.csv",
        index=False
    )

    return result


if __name__ == "__main__":
    run_classification(ZERO_SHOT_PROMPT, "zero_shot")
    run_classification(FEW_SHOT_PROMPT, "few_shot")
