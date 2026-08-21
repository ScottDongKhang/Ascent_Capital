# CIO / Capital Allocation Committee — Target Architecture

Grounded in `orchestrator/central_intelligence.py` (878 lines) and
`agents/ai_pm_agent.py`/`run_all_agents.py` as they exist today: `BASE_ALLOCATION =
{"us_equities": 1.0}`, `_compute_early_sharpe`/`_apply_early_zeros` (21-day,
Sharpe≤0 zero-out, exempt for `DEFENSIVE_AGENTS={"macro","alternatives"}` outside
stressed/crisis), `merge_agent_outputs`, and `agent_tasks =
[("us_equities", run_us_equities_agent)]` in `run_all_agents.py`.

## Layer 1 — Department Mandate

**Ascent Capital Investment Committee.** Owns four things, none of which any single
strategy agent may set for itself:

1. **Fund-level vol target and risk budget** (e.g. target annualized vol 10-14%,
   decomposed into per-strategy vol budgets that sum, after correlation, to the
   fund target — not a simple sum).
2. **Capital allocation across strategies/sleeves** — today just `us_equities`; in
   the target state, `us_equities`, `macro`, `international`, `alternatives`, and
   their internal `meanrev`/`statarb` sleeves.
3. **Promotion/demotion authority** — the only body permitted to move a strategy
   between "dormant research" (AI-PM-optional context call) and "live
   capital-allocating" (a line in `agent_tasks`, a nonzero key in
   `BASE_ALLOCATION`). This authority is currently exercised implicitly and
   manually — the code comments in `central_intelligence.py` documenting why
   macro/international were cut are the paper trail; there is no agent role that
   owns this decision going forward.
4. **Investor reporting** — periodic risk-adjusted-return narrative to "investors"
   (the owner), reusing the existing rebalance-recap discipline from CLAUDE.md but
   rolled up to fund level instead of per-rebalance.

The mandate is explicitly **risk-adjusted return, not raw return** — this must be
encoded as the committee's objective function (maximize fund Sharpe/Sortino subject
to the vol budget), not "beat SPY," which the existing docs already reject.

## Layer 2 — Roles

| Role | Analogue | Core question it answers |
|---|---|---|
| **CIO** | Fund CIO | "What's the fund's total risk budget, and does today's allocation respect it?" Final sign-off authority; the only role that can override the Strategy Allocation Analyst's proposed weights. |
| **Strategy Allocation Analyst** | Portfolio construction / risk analyst | "Given each live strategy's trailing Sharpe and its correlation to every other live strategy, what capital share maximizes IR ≈ IC × breadth?" |
| **Promotion/Demotion Gatekeeper** | IC voting member / seeding committee | "Has this dormant strategy earned live capital? Has this live strategy lost its right to it?" |
| **Investor Reporting Lead** | IR / LP reporting | "Turn the allocation ledger + P&L attribution into the periodic narrative." |

All four are software agents (LLM-backed where judgment is needed, deterministic
where it isn't), sitting **above** `orchestrator/central_intelligence.py`, not
inside it. The orchestrator remains the mechanical merge/cap/coherence layer; the
Committee decides its *inputs* (`BASE_ALLOCATION`, which agents are even in
`agent_tasks`).

## Layer 3 — Per-Role Responsibilities and Decision Logic

**Strategy Allocation Analyst — sizing logic.**
Inputs: each live strategy's trailing 63-day Sharpe (reuse `_compute_early_sharpe`'s
math, generalized beyond the 21-day/negative-only gate) and the full N×N
cross-strategy correlation matrix (reuse the existing
`ascent/risk/correlation_guard.py` machinery, currently unreachable because
`len(agent_weights) >= 2` never holds).
Logic: base weight ∝ Sharpe; then a **diversification multiplier** = `1 / (1 +
avg_pairwise_correlation_to_other_live_strategies)` — this is the concrete
implementation of "IR ≈ IC × breadth" (more weakly-correlated strategies compound
into higher fund-level IR even at equal individual Sharpe). A strategy correlated
>0.6 to an existing live strategy gets penalized even if its own Sharpe is fine,
because it isn't adding breadth.

**Promotion/Demotion Gatekeeper — thresholds** (extends `_apply_early_zeros`,
doesn't replace it — that function still runs *within* a live strategy's early-warmup
window; the Gatekeeper governs the dormant->live and live->dormant *boundary*):

- **Promotion eligibility**: a dormant strategy (macro/international/alternatives)
  is eligible after **N=10 consecutive scheduled rebalances** of positive
  risk-adjusted contribution in a **paper-traded combined book** (i.e., simulate
  what the fund's Sharpe would have been with this strategy folded in at a small
  weight, using its already-running independent pipeline's outputs — this is
  exactly what `agents/ai_pm_agent.py`'s optional `run_quant_agent` calls already
  produce as exhaust; today that exhaust is discarded after the AI PM reads it),
  AND correlation to every existing live strategy below **X=0.5** over the same
  window. Both conditions, not either — a strategy that's profitable but redundant
  doesn't improve fund IR.
- **Demotion**: a live strategy is cut after **M=5 consecutive rebalances** of
  trailing-63-day Sharpe below **Y=0** (a direct generalization of
  `_apply_early_zeros`'s existing `sharpe <= 0` test, but promoted from "zero this
  rebalance, reinstate automatically next rebalance if Sharpe recovers" to "flag
  for Committee review after 5 consecutive failures") **or** a single drawdown
  episode whose contribution to fund NAV drawdown exceeds **Z=25%** of the total
  drawdown (a concentration-of-pain test the current code has no equivalent for —
  `_water_fill_cap` only caps *position* weight, never *drawdown attribution*).
- Demotion is not instant deletion: a demoted strategy returns to "dormant
  research" status (still runs its pipeline, still feeds the AI PM as context)
  rather than being deleted, mirroring how macro/international already sit today —
  this makes demotion reversible and low-stakes to execute, which matters for
  getting the Gatekeeper agent to actually pull the trigger instead of defaulting
  to "give it more time."

**Investor Reporting Lead**: reads the allocation ledger (Layer 4) plus
per-strategy P&L attribution and produces the four-part recap format already
mandated in CLAUDE.md (reasoning / watch items / performance / why), rolled up
across strategies rather than confined to one rebalance's trade list.

**CIO**: sets the top-level vol target (a single scalar, e.g. read from a config
file the Committee owns) and has veto power over the Strategy Allocation Analyst's
proposed weights if they'd push the *fund-level* vol (computed from the
correlation-adjusted weighted sum, not the naive sum) above budget. This is the
missing piece today — there is currently no fund-level vol target anywhere in the
codebase; `_water_fill_cap` and `MAX_POSITION_WEIGHT` cap position concentration,
not portfolio volatility.

## Layer 4 — Interfaces / Data Contracts

- **Strategy Allocation Analyst** reads: `dashboard/agent_skill_scores.json`
  (already exists, already has the trailing-Sharpe shape `_load_skill_scores()`
  consumes) + a new `dashboard/strategy_correlation_matrix.json` (generalization of
  `ascent/risk/correlation_guard.py`'s pairwise return-correlation computation,
  made to run whenever ≥2 strategies are live, not gated to the orchestrator's dead
  `len(agent_weights)>=2` branch). Emits: `data_cache/cio_target_allocation.json` —
  `{strategy_id: target_capital_pct, as_of, rationale}` — which becomes the new
  `BASE_ALLOCATION` input to `_compute_allocation()`, replacing the hardcoded dict.
- **Promotion/Demotion Gatekeeper** reads: the paper-traded combined-book Sharpe
  history for each dormant strategy (new artifact,
  `logs/dormant_strategy_paper_pnl.jsonl`, populated by capturing
  `run_quant_agent`'s output inside `ai_pm_agent.py` instead of discarding it) +
  the correlation matrix above + the live-strategy demotion counters. Emits:
  `data_cache/cio_promotion_decisions.json` — an append-only decision log
  (`strategy_id, action: promote\|demote\|hold, as_of, evidence_refs`) that
  `run_all_agents.py` reads at startup to construct `agent_tasks`.
- **CIO** reads: the Strategy Allocation Analyst's proposed allocation + a fund
  vol-target config (`config/risk_budget.yaml`, new). Emits: the *approved*
  `cio_target_allocation.json` (may down-scale all weights, never up-scale past the
  analyst's proposal — a CIO can say no, not print money).
- **Investor Reporting Lead** reads: `logs/eod_log.jsonl`,
  `outputs/debate_log/verdict_<date>.json`, and `cio_target_allocation.json`'s
  history. Emits: a periodic markdown recap (reuses the existing rebalance-recap
  format).

## Layer 5 — Concrete Implementation Mapping

**New modules:**
- `orchestrator/cio_committee.py` — new top-level module, called by
  `run_all_agents.py` **before** `agent_tasks` is constructed (not after, unlike
  `run_orchestrator` which runs after agent execution). Contains `class
  StrategyAllocationAnalyst`, `class PromotionGatekeeper`, `class CIO`, `class
  InvestorReportingLead`, and `run_cio_committee() -> dict[str, float]` returning
  the approved allocation dict.
- `ascent/risk/fund_vol_budget.py` — new: computes fund-level vol from
  per-strategy vol + correlation matrix; used by `CIO.check_vol_budget()`.

**Changes to `run_all_agents.py`:**
- Before line 896 (`agent_tasks = [("us_equities", run_us_equities_agent)]`),
  insert a call to `run_cio_committee()` that reads
  `data_cache/cio_promotion_decisions.json` and dynamically builds `agent_tasks` —
  e.g. if macro was last promoted, `agent_tasks.append(("macro",
  run_macro_agent))`. This is the literal mechanism by which macro/international/
  alternatives stop being AI-PM-optional and become real capital-allocating agents:
  **they get added to `agent_tasks` and run in the same `ThreadPoolExecutor`,
  producing `AgentOutput`s that `run_orchestrator` already knows how to merge.** No
  change needed to `merge_agent_outputs` itself — it already supports N agents;
  only `BASE_ALLOCATION` needs to stop being a hardcoded 1.0 and instead be
  populated from `data_cache/cio_target_allocation.json`.

**Changes to `orchestrator/central_intelligence.py`:**
- `BASE_ALLOCATION`, `STRESSED_ALLOCATION`, `CRISIS_ALLOCATION` become loaded from
  `cio_target_allocation.json` (with a fallback to `{"us_equities": 1.0}` if the
  Committee hasn't run or the file is stale — same staleness-guard pattern as
  `_load_skill_scores()`).
- `_apply_early_zeros`'s single-agent degenerate-case warning (lines 274-319)
  becomes exercisable in the multi-agent case again once ≥2 agents run, restoring
  its original intended behavior.
- The correlation guard's dead `len(agent_weights) >= 2` gate (line 830) becomes
  live the moment a second agent is promoted — no code change required there, just
  a state change.
- `_apply_crisis_veto` (currently defined-but-uncalled per its own module comment)
  should be wired back into `run_orchestrator()` conditioned on `"macro" in
  allocation and allocation["macro"] > 0`, once/if macro is promoted — the
  function's logic doesn't need to change, just its call site's dead-code
  justification goes away.

**AI PM's role narrows correctly under this design**: `agents/ai_pm_agent.py`'s
`run_quant_agent` calls for macro/international/alternatives (currently "optional
context") remain useful — they're now also the data source for the Gatekeeper's
paper-traded promotion evidence — but the AI PM itself never decides promotion; it
stays advisory per integrity constraint #5, and the Committee, not the AI PM, is
the new (also advisory-until-proven, machine-checkable) layer that actually flips a
strategy from dormant to live.
