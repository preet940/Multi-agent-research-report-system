# Multi-Agent Research Report System

Planner -> Workers (parallel research) -> Writer -> Critic (revision loop, capped)

## Setup

```bash
pip install openai requests
export GROQ_API_KEY=your_key     # from console.groq.com
export TAVILY_API_KEY=your_key   # optional — falls back to stub data without it
python orchestrator.py "How is RAG used to reduce LLM hallucination?"
```

Note: uses Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`)
via the `openai` Python SDK — no Groq-specific SDK needed. Default model is
`llama-3.3-70b-versatile`; change `MODEL` in `agents.py` to any model your
Groq account has access to.

Output: printed report + `trace.json` (full step-by-step log of every
agent's input/output — this is your observability layer).

## Architecture decisions (and why — use these in interviews)

- **Agents are stateless functions.** No agent stores its own memory;
  the orchestrator owns all state. Makes each agent unit-testable in
  isolation and keeps the whole run traceable from outside.
- **Critic sees only the final draft, not the planner/worker reasoning.**
  Mirrors how a real reader encounters the report — catches gaps an
  "insider" reviewing its own chain of thought would miss.
- **Hard cap on revision rounds (`MAX_REVISION_ROUNDS = 2`).** Without
  this, critic<->writer could loop indefinitely if the critic keeps
  finding fault. Real agent systems always need a stop condition beyond
  "keep going until the model is happy."
- **Tools are separate from agent logic.** `search_web()` knows nothing
  about LLMs — swap the implementation without touching any agent code.
- **Workers run sequentially first.** Correct-then-fast: prove the
  control flow works before optimizing with
  `concurrent.futures.ThreadPoolExecutor` for parallel research.

## Known limitations (say these out loud — shows maturity, not weakness)

- No retry/backoff on API failures yet
- No cost tracking (each run = 1 planner + N worker + 1+ writer + 1+
  critic calls — worth logging token usage per step)
- Critic feedback isn't validated — a critic could reject forever if
  MAX_REVISION_ROUNDS weren't capped
- Search tool has no dedup/relevance filtering on results

## A real issue found during testing

On a run with real Tavily search data, the report cited a "71-89% hallucination
reduction" statistic and attributed it to RAG techniques alone. Checking the
original source (`futureagi.com`) showed the actual number applied to a
**combined guardrail stack** (system prompts + RAG + real-time monitoring),
not RAG in isolation. The worker had correctly extracted the number but
narrowed its scope while summarizing.

This is a **faithfulness/groundedness failure**, not classic hallucination —
the number was real, but the claim it was attached to was distorted. Fixed
by adding an explicit grounding rule to the worker prompt (don't narrow a
statistic's stated scope) and a corresponding check to the critic prompt.
Not fully solved — a proper fix needs an automated groundedness check
(comparing each cited claim against its source snippet), which is the
natural next step below.

## Natural next steps (this is your bridge into evals/guardrails study)

- Build an eval set: 10-15 topics with a rubric (factual accuracy,
  citation coverage, structure) — score `trace.json` outputs against it
- Add a guardrail step: check the final report for unsupported claims
  before it's returned to the user
- Track cost/latency per run as a quality metric, not just correctness