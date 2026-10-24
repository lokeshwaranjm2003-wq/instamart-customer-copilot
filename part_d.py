
from pathlib import Path

POLICY_DIR = Path("data/policies")

print("=" * 55)
print("PART D: POLICY RAG AND REFUND AGENT")
print("=" * 55)

# Find all policy text files
policy_files = sorted(POLICY_DIR.glob("*.txt"))

print("\nTotal policy documents:", len(policy_files))

for file in policy_files:
    text = file.read_text(encoding="utf-8")

    print("\n" + "-" * 50)
    print("Policy file:", file.name)
    print("Characters:", len(text))
    print("Words:", len(text.split()))
    print("Preview:")
    print(text[:350])

print("\nPOLICY INSPECTION COMPLETED")
