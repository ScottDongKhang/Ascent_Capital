# Lean fix: AI PM `run_quant_agent` no longer mandates all four agents

## Problem confirmed

`agents/ai_pm_agent.py`'s `run_quant_agent` tool enum (line 434) covers all four specialist
agents (`us_equities`, `macro`, `international`, `alternatives`). Two system-prompt sections
told the model to call all four every rebalance:

- Line 994 (`_SYSTEM_PROMPT`, single-phase fallback path, "PHASE 2 — QUANT BASELINE")
- Line 1309 (`_SYNTHESIS_PROMPT_TEMPLATE`, two-phase Phase 2 synthesis, "PHASE 2 TOOLS")

`run_all_agents.py` only ever dispatches `us_equities` daily, so the `agent_outputs` list
passed into `run_ai_pm(quant_outputs=...)` only ever contains that one agent's output. That
list becomes the `precomputed` cache built at lines 2572–2617 and threaded into
`_make_executor(..., precomputed, ...)` (call sites: line 2712 main Phase 2 executor, line
2827 fallback single-phase executor). `_run_quant_agent_cached` (lines 2228–2233) only serves
from that cache; any `agent_id` not in it (i.e. `macro`/`international`/`alternatives`) falls
through to `_tool_run_quant_agent` (line 1408), which does a real `importlib.import_module`
and calls `run_macro_agent()` / `run_international_agent()` / `run_alternatives_agent()` live —
their full pipelines, at real wall-clock and LLM/API cost — on every scheduled rebalance,
because the prompt told the model calling all four was "Required."

None of those three agents' outputs feed live capital: `orchestrator/central_intelligence.py`
only ever sees `us_equities` in `agent_weights` (daily run dispatches only that one agent), so
the correlation guard and any cross-agent logic are structurally unreachable for
macro/international/alternatives regardless of what the AI PM does with them.

## Fix

Changed both "Required: run_quant_agent for all four agents" / "Required: run_quant_agent ×4"
instructions to make `us_equities` required and the other three explicitly optional, with the
reasoning stated in-prompt (their signal isn't proven bad, just unused for live capital; each
extra call has a real time/cost, so call them only if the model's judgment genuinely wants more
context).

### Line 994 (`_SYSTEM_PROMPT`)

Before:
```
Required: run_quant_agent for all four agents. Then call get_position_momentum on ALL combined names.
```

After:
```
Required: run_quant_agent for us_equities — it is the only agent whose output feeds live
capital allocation. Optional: run_quant_agent for macro / international / alternatives if you
want additional market context for your qualitative judgment — their signal is not proven bad,
it's simply not wired to live weights, so calling them costs real wall-clock/API time for
context only, not for sizing. Then call get_position_momentum on ALL combined names you have.
```

### Line 1309 (`_SYNTHESIS_PROMPT_TEMPLATE`)

Before:
```
Required: run_quant_agent ×4. Then get_position_momentum on all combined names.
```

After:
```
Required: run_quant_agent for us_equities — the only agent that feeds live capital. Optional:
run_quant_agent for macro / international / alternatives for extra context (not proven bad,
just unused for sizing — each additional call has real wall-clock/API cost). Then
get_position_momentum on all combined names you have.
```

(Exact line numbers after edit shifted slightly since the new text is longer — the "Required:"
line itself is still the first line of each block, now at 994 and ~1313 respectively; grep
`Required: run_quant_agent` to relocate precisely if the file changes again.)

## What was NOT changed (per task constraints)

- Tool enum (line 434) — left as `["us_equities", "macro", "international", "alternatives"]`.
  Calling any of the three remains fully possible; the model just isn't told it must.
- `agents/macro_agent.py`, `international_agent.py`, `alternatives_agent.py` — untouched.
- `orchestrator/central_intelligence.py`, `ascent/alpha/`, `ascent/analyst/proof_audit/agent_signals.py`
  — untouched, out of scope.
- No other prompt section penalizes or scores partial agent calls — grepped the whole file for
  "four agent", "all 4", "×4", "x4", "checklist" and confirmed only the two edited lines
  matched (before the edit). The sizing/rubric sections (`HOW TO USE THE QUANT OUTPUT`,
  `SIZING DISCIPLINE`) talk generically about "quant" signal, not about needing all four agents,
  so no contradictory instruction remains.

## Item 3 — cheaper way to serve macro/international/alternatives context

Checked for an existing cached/precomputed summary of these agents' last known signal that
could substitute for a live run:

- `data_cache/` has raw price caches (`prices_macro.parquet`, `prices_international.parquet`,
  `prices_alternatives.parquet`, `macro_live.parquet`/`macro_simulated.parquet`,
  `macro_causal_dag.json`) but **no stored `AgentOutput`-shaped artifact** (target weights /
  regime signal / skill score) for macro, international, or alternatives agents.
- `ascent/analyst/proof_audit/agent_signals.py` computes/scores these agents' signal but does
  not persist a reusable "last known output" object either, and the task explicitly puts that
  file's hard imports out of scope.
- Conclusion: no clean existing shortcut exists today. Building one (e.g. a small
  `data_cache/agent_output_cache/{macro,international,alternatives}.json` written the last time
  each agent actually ran, with an age/staleness check) would be a reasonable follow-up but is
  a new caching layer, not a straightforward reuse of something already there — left as a
  nice-to-have, not built, per the task's own guidance not to build new caching infra if it
  isn't a clean fit.

## Tests

Ran the structural/prompt-contract test suites covering `agents/ai_pm_agent.py` — none require
a live LLM call (they test prompt strings, tool schema, executor wiring, and mocked flows):

```
tests/test_ai_pm_prompt_contract.py ..............   (14 passed)
tests/test_ai_pm_agent.py .................           (17 passed)
tests/test_ai_pm_cheap_bugs.py .................       (17 passed)
tests/agents/test_ai_pm_fallback_fix.py ........        (8 passed)
tests/agents/test_ai_pm_phase2_force_seal.py ....        (4 passed)
tests/agents/test_ai_pm_prethesis_fixes.py ...........  (11 passed)

71 passed in 5.33s
```

`ast.parse` on the edited file after the patch: OK, no syntax errors.

No real LLM API calls were made to test this change, per instructions — verified by reading
the prompt text and the executor/caching code path, not by invoking the AI PM end-to-end.
