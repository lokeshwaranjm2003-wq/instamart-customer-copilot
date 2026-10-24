
from pathlib import Path
import pandas as pd
import mock_llm

from part_c_classify import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    classify_with_retry,
    validate_response,
)

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

def run_triage(prompt, budget, output_file):
    tickets = pd.read_csv(
        "data/raw/tickets.csv",
        keep_default_na=False
    )

    redacted = pd.read_csv(
        OUTPUTS / "redacted_tickets.csv",
        keep_default_na=False
    )

    segments = pd.read_csv(
        OUTPUTS / "customer_segments.csv"
    )

    tickets = tickets.merge(
        redacted,
        on=["ticket_id", "customer_id"],
        how="left",
        validate="one_to_one"
    )

    tickets = tickets.merge(
        segments[
            ["customer_id", "segment_name", "is_premium"]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    mock_llm.reset()
    total_tokens = 0
    results = []

    for _, row in tickets.iterrows():
        original = str(row["text"])
        clean = str(row["redacted_text"])

        status = ""
        category = ""
        tokens = 0

        if not original.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(original) > 400:
            status = "flagged_long"

        elif total_tokens >= budget:
            status = "budget_exhausted"

        else:
            response = classify_with_retry(
                row["ticket_id"],
                clean,
                prompt
            )

            tokens = response.usage.total_tokens
            total_tokens += tokens

            status, category = validate_response(
                response.choices[0].message.content
            )

        # Routing rules
        premium = str(row["is_premium"]).lower() == "true"

        if status not in [
            "classified_json",
            "classified_keyword"
        ]:
            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif premium:
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        segment_name = (
            str(row["segment_name"])
            if pd.notna(row["segment_name"])
            else ""
        )

        if status not in [
            "classified_json",
            "classified_keyword"
        ]:
            segment_name = ""

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens
        })

    result = pd.DataFrame(results)
    result.to_csv(OUTPUTS / output_file, index=False)

    print("\nFILE:", output_file)
    print("Total tickets:", len(result))
    print("Total tokens:", total_tokens)
    print("\nStatus counts:")
    print(result["status"].value_counts())
    print("\nRouting counts:")
    print(result["route"].value_counts())
    print("\nPriority counts:")
    print(result["priority"].value_counts())

    return result


# Budget demonstration using zero-shot prompt
budget_result = run_triage(
    ZERO_SHOT_PROMPT,
    6000,
    "triage_budget_demo.csv"
)

budget_exhausted = (
    budget_result["status"] == "budget_exhausted"
).sum()

print("\nBudget exhausted tickets:", budget_exhausted)

assert budget_exhausted > 0, (
    "No tickets exhausted the demo budget. "
    "Reduce demo budget to 5000."
)

# Final few-shot run with a large budget
final_result = run_triage(
    FEW_SHOT_PROMPT,
    100000,
    "triage_results.csv"
)

assert len(final_result) == 164
assert (
    final_result["status"] == "budget_exhausted"
).sum() == 0

print("\nPART C ROUTING COMPLETED")
