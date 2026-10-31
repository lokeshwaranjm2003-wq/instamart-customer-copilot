
import pandas as pd
import mock_llm

from react_refund_agent import run_agent


def main():
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")

    print("=" * 60)
    print("REACT REFUND AGENT - SEVEN SCENARIO TEST")
    print("=" * 60)

    for _, row in scenarios.iterrows():
        mock_llm.reset()

        print("\n" + "-" * 60)
        print("Scenario:", row["scenario_id"])
        print("Expected:", row["expected_outcome"])
        print("Expected refund:", row["expected_refund_inr"])

        result = run_agent(
            row["message"],
            max_iterations=6,
        )

        print("Actual result:", result)

    print("\nALL SEVEN SCENARIOS EXECUTED")


if __name__ == "__main__":
    main()
