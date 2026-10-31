
from pathlib import Path
from uuid import uuid4
import time
import json
import mock_llm

from refund_tools import (
    lookup_order,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    policy_search,
)


CHECKPOINT_DIR = Path("state")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step,
    tools_executed,
):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tools_executed,
    }

    path = CHECKPOINT_DIR / f"{run_id}.json"
    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(checkpoint, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)

    print("Checkpoint saved:", path)


def load_checkpoint(run_id):
    path = CHECKPOINT_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    print("Checkpoint loaded:", path)
    print("Saved step:", checkpoint["step"])
    print(
        "Completed tools:",
        [
            tool["name"]
            for tool in checkpoint["tools_executed"]
        ],
    )

    return checkpoint


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AGENT_LOG = OUTPUT_DIR / "agent_runs.jsonl"


def log_agent_run(
    run_id,
    scenario_id,
    status,
    tokens_used,
    tools_executed,
):
    record = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "tokens_used": tokens_used,
        "tool_count": len(tools_executed),
        "tools": [
            tool["name"]
            for tool in tools_executed
        ],
    }

    with AGENT_LOG.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(record) + "\n")

    print("Agent run logged:", AGENT_LOG)


def execute_tool(name, args):
    if name == "get_order":
        order = lookup_order(args["order_id"])

        return (
            order
            if order
            else {"error": "Order not found"}
        )

    if name == "search_policy":
        return policy_search(args["query"])

    if name == "calculate_refund":
        order = {
            "status": "completed",
            "order_amount": args["order_amount"],
            "delivery_minutes": args["delivery_minutes"],
        }

        result = calculate_refund(order)

        return {
            "refund_inr": result["amount"]
        }

    if name == "issue_refund":
        result = issue_refund(
            args["order_id"],
            args["refund_inr"],
        )

        return {
            "refund_inr": args["refund_inr"],
            "status": result["status"],
        }

    if name == "escalate_to_human":
        result = escalate_to_human(
            args["order_id"],
            "Refund above INR 1000",
        )

        return {
            "refund_inr": args["refund_inr"],
            "status": result["status"],
        }

    return {
        "error": f"Unknown tool: {name}"
    }


def run_agent(
    message,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=60,
    run_id=None,
    scenario_id="manual",
    simulate_crash_after=None,
):
    if run_id is None:
        run_id = uuid4().hex

    tools_executed = []
    tokens_used = 0
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT: Follow refund policy using tools."},
        {"role": "user", "content": message},
    ]

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
            print("Final status:", status)
            return status

        response = mock_llm.chat(messages, tools=[])
        tokens_used += response.usage.total_tokens
        print("Tokens used:", tokens_used)

        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
            print("Final status:", status)
            return status

        reply = response.choices[0].message
        if not reply.tool_calls:
            final_answer = reply.content
            for msg in reversed(messages):
                if msg["role"] == "tool" and msg["name"] == "issue_refund":
                    result = json.loads(msg["content"])
                    if result["status"] == "duplicate":
                        final_answer = (
                            "ALREADY_REFUNDED: Refund was previously recorded for this order."
                        )
                    elif result["status"] != "issued":
                        final_answer = f"REFUND_FAILED: Tool returned {result['status']}."
                    break
            log_agent_run(run_id, scenario_id, final_answer, tokens_used, tools_executed)
            print("Final answer:", final_answer)
            return final_answer

        tool = reply.tool_calls[0]
        name = tool.name
        args = tool.arguments
        if isinstance(args, str):
            args = json.loads(args)

        print(f"Step {iteration + 1}: {name}")
        result = execute_tool(name, args)
        print("Tool result:", result)
        tools_executed.append({"name": name, "arguments": args, "result": result})
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool.id, "name": name, "arguments": args}],
        })
        messages.append({
            "role": "tool", "name": name,
            "tool_call_id": tool.id, "content": json.dumps(result),
        })
        save_checkpoint(
            run_id=run_id, scenario_id=scenario_id, messages=messages,
            tokens_used=tokens_used, step=iteration + 1,
            tools_executed=tools_executed,
        )
        if simulate_crash_after is not None and iteration + 1 == simulate_crash_after:
            raise RuntimeError(f"SIMULATED_CRASH_AFTER_STEP_{iteration + 1}")

    status = "FALLBACK_MAX_ITERATIONS"
    log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
    print("Final status:", status)
    return status

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=60,
):
    checkpoint = load_checkpoint(run_id)
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    tools_executed = checkpoint["tools_executed"]
    start_step = checkpoint["step"]
    scenario_id = checkpoint["scenario_id"]
    start_time = time.monotonic()
    print("Resuming from step:", start_step)

    for iteration in range(start_step, max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
            return status

        response = mock_llm.chat(messages, tools=[])
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
            return status

        reply = response.choices[0].message
        if not reply.tool_calls:
            final_answer = reply.content
            for msg in reversed(messages):
                if msg["role"] == "tool" and msg["name"] == "issue_refund":
                    result = json.loads(msg["content"])
                    if result["status"] == "duplicate":
                        final_answer = "ALREADY_REFUNDED: Refund was previously recorded."
                    elif result["status"] != "issued":
                        final_answer = f"REFUND_FAILED: Tool returned {result['status']}."
                    break

            log_agent_run(
                run_id=run_id,
                scenario_id=checkpoint["scenario_id"],
                status=final_answer,
                tokens_used=tokens_used,
                tools_executed=tools_executed,
            )
            print("Resumed final answer:", final_answer)
            return final_answer

        tool = reply.tool_calls[0]
        name = tool.name
        args = tool.arguments
        if isinstance(args, str):
            args = json.loads(args)
        print(f"Resume step {iteration + 1}: {name}")
        result = execute_tool(name, args)
        tools_executed.append({"name": name, "arguments": args, "result": result})
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": tool.id, "name": name, "arguments": args}],
        })
        messages.append({
            "role": "tool", "name": name,
            "tool_call_id": tool.id, "content": json.dumps(result),
        })
        save_checkpoint(
            run_id=run_id, scenario_id=scenario_id,
            messages=messages, tokens_used=tokens_used,
            step=iteration + 1, tools_executed=tools_executed,
        )

    status = "FALLBACK_MAX_ITERATIONS"
    log_agent_run(run_id, scenario_id, status, tokens_used, tools_executed)
    return status

if __name__ == "__main__":
    mock_llm.reset()

    run_agent(
        "My order ORD-100048 reached me very late. "
        "I would like a refund."
    )
