# Deployable Multi-Agent Architecture — Which Roles Are Actually LLM Agents

Cross-references `01`-`06`'s ~28 roles against the codebase's already-working
LLM infrastructure (`ascent/llm/client.py`, `agents/ai_pm_agent.py`,
`agents/red_team_agent.py`, `debate/judge.py`) to classify each role as (a)
deterministic code, (b) LLM-backed agent, or (c) scheduled/cron job, so the
department blueprints translate into an actual runnable system, not just
Python modules.

## The headline finding

**Of ~28 roles across all 6 departments, only 6 are genuinely LLM-backed —
and 3 of those already exist and need no new agent** (the Judge, Red Team,
and the Signal Researcher's LLM-guided variant generation in
`self_improve.py::generate_variants()`). Every role whose decision logic is
expressed as a numeric threshold in the blueprints (VaR%, correlation cap,
DSR-p, PBO%, slippage-bps, discrepancy-$, ladder gate booleans) is
deterministic (a) — these are roles the blueprints themselves already
describe with `if metric > threshold` pseudocode. Giving them an LLM would
add cost, latency, and nondeterminism to something fully specified.

The only genuinely new (b) roles are the ones synthesizing multiple
already-computed verdicts into prose, or making a discretionary call the
blueprint explicitly reserves for judgment on a borderline case:

| Role | Model | Tools/wrapper | Core responsibility | Phase dependency |
|---|---|---|---|---|
| **CRO synthesis** (01) | Haiku | `chat_completion`, reads `RiskDecision` lists from roles 1-4 | Turn 4 roles' structured decisions into one paragraph + a `halt_requested` flag; never invents a number not in the inputs; escalates ambiguity rather than resolving it silently | Phase 1 (needs roles 1-4's code to exist) |
| **Promotion/Demotion Gatekeeper — discretion** (04) | Sonnet | `tool_completion`, reads paper-PnL + correlation logs | Adjudicate a borderline case (correlation 0.5-0.6, or Sharpe just under threshold) given the mechanical gate values; must cite the specific value being overridden; may never override a WFE/DSR failure | Phase 4 (needs Strategy Allocation Analyst's math + Phase 2's DSR/PBO) |
| **Investor Reporting Lead** (04) | Sonnet | `chat_completion`, reads `eod_log.jsonl`, verdicts, allocation history | Produce the four-part rebalance-recap narrative rolled up to fund level, quoting verdict reasoning verbatim, flagging untraceable figures | Phase 4, but **a fund-level version can ship now** against `us_equities`-only data |
| **CIO — vol-budget veto narrative** (04) | Opus | `tool_completion`, reads `risk_budget.yaml` + proposed weights | Explain whether today's proposed allocation respects the fund vol budget; may only shrink, never up-scale | Phase 4 |
| **Escalation Authority — severity classifier** (06) | Haiku | `chat_completion`, reads `get_gated_authority()` | Classify a disagreement into severity 1/2/3 per pre-defined numeric thresholds (the boundaries are config, not model discretion); produce a one-line reason string | Phase 6 |

Every deterministic (a) role is an **inline call within `run_all_agents.py`'s
existing sequence**, at the file:line insertion points each blueprint's
Layer 5 already names — not a standalone script.

## Legitimately (c) — standalone/cron, not inline

- **Reconciliation Analyst** (05) — runs the *next* morning, before
  `check_halt_state()`, because Alpaca settlement isn't same-day-reliable.
- **Outcome Tracker / `score_pending_interventions`** — scores interventions
  ≥14 calendar days old; inherently periodic batch, not per-run.
- **Investor Reporting Lead** — weekly/monthly, cron-scheduled separately
  from the daily pipeline invocation.

## What ships this week vs. what needs a prior phase

**Ships now, no prerequisite phase:**
- Wiring `compliance/audit_trail.py` into halt/override/kill-switch sites
  (Phase 0 item 2) — pure plumbing against code that already exists.
- **Data Integrity Officer** — `scripts/reconcile_numbers.py` already
  computes duplicate/phantom-row counts; wrapping it as a pre-flight gate is
  a refactor, not new statistics.
- **A fund-level Investor Reporting Lead** covering `us_equities` only
  (the only live strategy today) — the four-part recap format and model
  tier are already fully specified in CLAUDE.md's "Rebalance recap" section;
  this generalizes an existing manual practice into a scheduled Sonnet call.

**Blocked on Phase 1** (IRM roles 1-4 don't exist yet): the CRO synthesis
agent — an LLM cannot synthesize numbers that don't get computed.

**Blocked on Phase 2** (DSR/PBO don't exist yet): the Gatekeeper's LLM
discretion — there's no borderline case to adjudicate without the statistic.

**Blocked on Phase 4** (CIO committee scaffolding): roles requiring
`orchestrator/cio_committee.py`'s deterministic math first — the longest
dependency chain in the whole set (Phase 2 → Phase 4 → these roles).

**Blocked on Phase 6** (ladder bug fix): the Escalation Authority reads
`get_gated_authority()`, net-new per the blueprint — building the LLM
classifier before the promotion-path bug is fixed just formalizes a broken
ladder.

## Bottom line

The transformation plan's phases are almost entirely deterministic-code
work — VaR math, DSR/PBO, compliance thresholds, reconciliation diffs. The
LLM budget should go to exactly 5 new synthesis points (CRO, Gatekeeper
discretion, Investor Reporting Lead, CIO narrative, Escalation classifier)
sitting on top of a much larger base of hard-coded thresholds — the same
ratio the codebase already has today (a handful of LLM synthesis calls:
AI PM, Judge, Red Team — over a much larger deterministic pipeline), not an
inflation of LLM surface area to match the blueprint's role count.
