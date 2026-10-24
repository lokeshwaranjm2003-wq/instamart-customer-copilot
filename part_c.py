
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

# Load customer support tickets
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("=" * 50)
print("PART C: SUPPORT TICKET TRIAGE")
print("=" * 50)

print("\nTotal tickets:", len(tickets))
print("\nColumns:")
print(tickets.columns.tolist())

print("\nTrue category counts:")
print(tickets["true_category"].value_counts())

# Check empty tickets
empty_tickets = tickets[
    tickets["text"].astype(str).str.strip().eq("")
]

print("\nEmpty tickets:", len(empty_tickets))

# Estimate tokens using the project rule
from mock_llm import est_tokens

tickets["estimated_tokens"] = tickets["text"].apply(est_tokens)

# Check long tickets
long_tickets = tickets[
    tickets["estimated_tokens"] > 400
]

print("Long tickets:", len(long_tickets))

print("\nMaximum estimated tokens:",
      tickets["estimated_tokens"].max())

print("\nPART C DATA INSPECTION COMPLETED")


# PART C STEP 2: PRIVACY REDACTION

import re

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

def redact_text(text):
    text = str(text)

    # Replace email addresses
    text, email_count = re.subn(
        EMAIL_PATTERN, "[EMAIL]", text
    )

    # Replace phone numbers
    text, phone_count = re.subn(
        PHONE_PATTERN, "[PHONE]", text
    )

    # Replace order IDs
    text, order_count = re.subn(
        ORDER_PATTERN, "[ORDER_ID]", text
    )

    return text, email_count, phone_count, order_count


redacted_texts = []
total_emails = 0
total_phones = 0
total_order_ids = 0

for text in tickets["text"]:
    clean_text, emails, phones, order_ids = redact_text(text)

    redacted_texts.append(clean_text)

    total_emails += emails
    total_phones += phones
    total_order_ids += order_ids

tickets["redacted_text"] = redacted_texts

# Verify that no personal information remains
for text in tickets["redacted_text"]:
    assert not re.search(EMAIL_PATTERN, text)
    assert not re.search(PHONE_PATTERN, text)
    assert not re.search(ORDER_PATTERN, text)

print("\n" + "=" * 50)
print("PART C: PRIVACY REDACTION RESULTS")
print("=" * 50)

print("Emails redacted:", total_emails)
print("Phone numbers redacted:", total_phones)
print("Order IDs redacted:", total_order_ids)

total_redactions = (
    total_emails + total_phones + total_order_ids
)

print("Total redactions:", total_redactions)
print("Personal information remaining: 0")

# Save only redacted text, not the original ticket text
tickets[
    ["ticket_id", "customer_id", "redacted_text"]
].to_csv(
    OUTPUTS / "redacted_tickets.csv",
    index=False
)

print("Saved: outputs/redacted_tickets.csv")
