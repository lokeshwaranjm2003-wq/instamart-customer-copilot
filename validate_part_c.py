
from pathlib import Path
import pandas as pd

OUTPUTS = Path("outputs")

triage = pd.read_csv(
    OUTPUTS / "triage_results.csv",
    keep_default_na=False
)

budget = pd.read_csv(
    OUTPUTS / "triage_budget_demo.csv",
    keep_default_na=False
)

assert len(triage) == 164, "Ticket count incorrect"
assert triage["ticket_id"].is_unique, "Duplicate ticket IDs"
assert len(budget) == 164, "Budget demo count incorrect"

classified = triage["status"].isin([
    "classified_json",
    "classified_keyword"
])

assert classified.sum() == 147, "Classified count incorrect"
assert (triage["status"] == "skipped_empty").sum() == 4
assert (triage["status"] == "flagged_long").sum() == 4
assert (triage["status"] == "flagged_invalid_output").sum() == 9
assert (triage["status"] == "budget_exhausted").sum() == 0

assert (
    triage.loc[triage["category"] == "fraud", "route"]
    == "security_team"
).all()

assert (
    triage.loc[~classified, "route"] == "human_review"
).all()

assert (
    triage.loc[triage["priority"] == "HIGH", "route"]
    .isin(["security_team", "premium_desk"])
).all()

assert (
    budget["status"] == "budget_exhausted"
).any(), "Budget demonstration did not trigger"

print("ALL PART C VALIDATION CHECKS PASSED")
print("Total tickets:", len(triage))
print("Classified:", classified.sum())
print("HIGH priority:", (triage["priority"] == "HIGH").sum())
print("NORMAL priority:", (triage["priority"] == "NORMAL").sum())
print("Budget demo: PASS")
