"""
Output guardrail: runs BEFORE a report is returned to the user, not
after the fact in a batch eval. Same scoring logic as an eval, different
job -- an eval measures and reports; a guardrail measures and DECIDES
whether to let the output through.

This is the same judge_groundedness function you'd write for
eval_harness.py -- reused here as a live blocking check.
"""

from agents import _ask_json

GROUNDEDNESS_THRESHOLD = 3  # below this score (out of 5), block the report


def check_groundedness(topic: str, report: str) -> dict:
    """Same check as an eval, called inline instead of in a batch run."""
    return _ask_json(
        system=(
            "You are auditing a research report for groundedness. "
            "For each specific statistic or strong factual claim, judge "
            "whether it is stated with a scope/cause that seems plausible "
            "and not artificially narrowed (e.g. attributing a combined-"
            "method result to a single method). Score 1-5: 5 = claims "
            "look precisely scoped and appropriately hedged, 1 = claims "
            "look overstated or suspiciously precise without clear scope."
        ),
        user=f"Topic: {topic}\n\nReport:\n{report}\n\n"
             f'Return JSON: {{"score": 1-5, "reasoning": "...", '
             f'"flagged_claims": ["..."]}}',
        call_type="output_guardrail",
    )


def apply_output_guardrail(topic: str, report: str) -> dict:
    """
    Returns a dict describing what happened: whether the report passed,
    and either the original report or a safe fallback message.

    This is where you make the real design decision from the theory
    discussion: block-and-fallback vs block-and-retry vs log-and-allow.
    Here we do block-and-fallback (simplest, safest default) -- the
    orchestrator could instead choose block-and-retry by feeding the
    flagged claims back to writer_agent as feedback.
    """
    check = check_groundedness(topic, report)
    passed = check["score"] >= GROUNDEDNESS_THRESHOLD

    if passed:
        return {
            "guardrail_passed": True,
            "groundedness_score": check["score"],
            "reasoning": check["reasoning"],
            "final_output": report,
        }

    fallback_message = (
        "I wasn't able to produce a report I'm confident is accurately "
        "grounded in its sources for this topic. Rather than show "
        "potentially misleading claims, here's what I found flagged "
        f"as uncertain: {'; '.join(check.get('flagged_claims', []))}\n\n"
        "You may want to research this topic further with additional "
        "sources, or try rephrasing the topic to be more specific."
    )
    return {
        "guardrail_passed": False,
        "groundedness_score": check["score"],
        "reasoning": check["reasoning"],
        "flagged_claims": check.get("flagged_claims", []),
        "final_output": fallback_message,
    }