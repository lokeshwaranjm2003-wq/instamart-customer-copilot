
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Part A - Data Cleaning & SQLite", "part_a.py"),
    ("Part B - Customer Segmentation", "part_b.py"),
    ("Part B - Lapse Features", "part_b_lapse.py"),
    ("Part B - Prediction Model", "part_b_model.py"),
    ("Part C - Privacy Redaction", "part_c.py"),
    ("Part C - Ticket Classification", "part_c_classify.py"),
    ("Part C - Ticket Routing", "part_c_routing.py"),
    ("Part D - Policy Inspection", "part_d.py"),
    ("Part D - Policy Retrieval", "part_d_rag.py"),
    ("Part D - Policy Assistant", "part_d_assistant.py"),
    ("Part D - Refund Agent", "react_refund_agent.py"),
]

def main():
    print("=" * 60)
    print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("FULL PROJECT RUN")
    print("=" * 60, flush=True)

    total_start = time.monotonic()

    for number, (title, filename) in enumerate(STEPS, start=1):
        script = PROJECT_ROOT / filename

        print(f"\n[{number}/{len(STEPS)}] {title}", flush=True)
        print(f"Running: {filename}", flush=True)

        if not script.is_file():
            print(f"FAILED: File not found: {filename}")
            sys.exit(1)

        start = time.monotonic()

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        elapsed = time.monotonic() - start

        if result.returncode != 0:
            print(f"\nFAILED: {title}")
            print(f"Exit code: {result.returncode}")
            print(f"Time: {elapsed:.2f} seconds")
            sys.exit(result.returncode)

        print(f"PASS: {title} ({elapsed:.2f} seconds)", flush=True)

    total_time = time.monotonic() - total_start

    print("\n" + "=" * 60)
    print("ALL PROJECT STEPS COMPLETED SUCCESSFULLY")
    print(f"Total time: {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
