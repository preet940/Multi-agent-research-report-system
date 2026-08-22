"""
Owns ALL state for a run. Agents are stateless functions we call in order;
this file is the only place that knows "what step are we on."

Run: python orchestrator.py "your topic here"
"""

import sys
import json
import time
from datetime import datetime, timezone

from agents import planner_agent, worker_agent, writer_agent, critic_agent, reset_call_log, get_run_cost_summary
from guardrails import apply_output_guardrail

MAX_REVISION_ROUNDS = 2  # hard cap -> prevents infinite critic<->writer loop


def run(topic: str) -> dict:
    reset_call_log()  # so cost is per-run, not cumulative across topics
    trace = {"topic": topic, "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}

    def log(step_name: str, data: dict):
        trace["steps"].append({"step": step_name, "at": time.time(), **data})
        print(f"\n=== {step_name} ===")
        print(json.dumps(data, indent=2)[:800])  # truncate for readability

    # 1. PLAN
    sub_questions = planner_agent(topic)
    log("planner", {"sub_questions": sub_questions})

    # 2. RESEARCH (sequential for simplicity; parallelize with
    #    concurrent.futures.ThreadPoolExecutor once this works)
    worker_outputs = []
    for q in sub_questions:
        result = worker_agent(q)
        worker_outputs.append(result)
        log("worker", result)

    # 3. WRITE (initial draft)
    draft = writer_agent(topic, worker_outputs)
    log("writer", {"draft_preview": draft[:300]})

    # 4. CRITIQUE LOOP
    for round_num in range(1, MAX_REVISION_ROUNDS + 1):
        review = critic_agent(topic, draft)
        log("critic", {"round": round_num, **review})

        if review["verdict"] == "approve":
            break

        draft = writer_agent(topic, worker_outputs, feedback=review["feedback"])
        log("writer_revision", {"round": round_num, "draft_preview": draft[:300]})
    else:
        log("critic", {"note": f"Max {MAX_REVISION_ROUNDS} revision rounds hit, shipping best draft"})

    # 5. OUTPUT GUARDRAIL — runs regardless of critic verdict. The critic
    #    is an LLM judging its own kind of output (a report); the guardrail
    #    is a separate, narrower, blocking check specifically on
    #    groundedness. Different from the critic loop above: this doesn't
    #    trigger a rewrite, it decides whether to show the report AT ALL.
    guardrail_result = apply_output_guardrail(topic, draft)
    log("output_guardrail", guardrail_result)

    trace["final_report"] = guardrail_result["final_output"]
    trace["guardrail_passed"] = guardrail_result["guardrail_passed"]
    trace["cost_summary"] = get_run_cost_summary()
    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    return trace


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "The impact of retrieval-augmented generation on LLM factuality"
    result = run(topic)

    # Save full trace (your observability artifact — show this in interviews)
    with open("trace.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(json.dumps(result["cost_summary"], indent=2))
    print(f"\n(Full trace saved to trace.json)")