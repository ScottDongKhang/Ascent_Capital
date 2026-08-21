# Transformation Plan — Closing the Gaps

> **v2 addendum — read this before the plan below.** A second attack wave
> (`17`-`21`) found five things this v1 plan got wrong or omitted, in order
> of how much they should change what you actually do next:
>
> 1. **`21_alpha_edge_audit.md`: the underlying edge was never audited, and
>    when read directly the two live sleeves are textbook, heavily-crowded
>    short-term reversal (RSI/Bollinger/z-score/momentum) with beta 0.95 to
>    SPY and zero published per-sleeve IC.** Insert an alpha-IC-triage step
>    (§Phase 2 in `21`) **before** building the DSR/PBO machinery below —
>    otherwise Phase 2 rigorously validates a signal pair already plausibly
>    indistinguishable from beta.
> 2. **`20_cost_timeline_reality_check.md`: this plan has no real timeline.**
>    Converted to solo/part-time weekends, the full 8 phases are ~20-28
>    weekends over 6-9 months, not a "sequenced backlog." A minimum-viable
>    4-6-weekend cut is specified there — read it before starting anything.
> 3. **`20` also flags this directly: 16+ planning documents, zero lines of
>    implementation.** Two of the later documents already found this plan's
>    own effort estimates wrong. That pattern is a real signal to stop
>    planning and ship the Phase 0 weekend next — not to write document 22.
> 4. **`17_red_team_findings.md`: entire categories of operational risk are
>    absent** — disaster recovery, credential expiry, broker/LLM-provider
>    outage mid-run, key-person coverage, single-machine failure. None of
>    Phases 0-7 below address any of these; they're all signal/portfolio
>    risk, not operational risk. Treat this as an unscoped Phase 8 candidate.
> 5. **`19_deployable_agent_architecture.md`: most of what follows is not an
>    "agent," it's deterministic code.** Only 5 new LLM-backed roles exist
>    across all 6 departments (CRO synthesis, Gatekeeper discretion,
>    Investor Reporting Lead, CIO narrative, Escalation classifier) — useful
>    for scoping actual implementation effort per phase below.
>
> The phase structure below is otherwise still the right shape — these are
> corrections and one missing category, not a rewrite.

Synthesizes `00`-`10` into a single prioritized, sequenced backlog: what to cut,
what to add, in what order, and why. Every item traces back to a specific finding
in the audit/blueprint/lifecycle docs — nothing here is invented fresh.

**Sequencing principle**: cheapest-and-highest-value first. Phase 0 is
housekeeping (fix known-wrong claims, wire code that already exists). Phases 1-3
are the load-bearing gaps that most compromise "does this behave like a real
fund." Phases 4-6 are structural/organizational maturity. Phase 7 is the piece
`10_investment_thesis_lifecycle.md` found missing industry-wide, not just here
— building even a thin version puts Ascent ahead of typical practice.

---

## Phase 0 — Housekeeping (cheap, do first)

1. **Fix CLAUDE.md's stale "four agents required" claim** (`00`, `03` finding).
   `ai_pm_agent.py:972-975` already made macro/international/alternatives
   optional; the top-of-file doc description still says "Required." One-line
   doc fix, zero code risk.
2. **Wire `compliance/audit_trail.py` into the gaps `05` identified** — it
   already exists (hash-chained, already used for `order_submitted` etc.) but
   doesn't yet cover halt/override events (`run_all_agents.py::check_halt_state`,
   lines 360-402) or kill-switch trip/reset. Add `_audit(...)` calls at those
   sites. This is instrumentation, not new infrastructure — low risk, immediate
   value (closes a real audit-trail gap with code that's already proven).
3. **Fix the two duplicate order-submission paths** (`03` finding):
   `run_eod()` and `run_eod_with_weights()` independently reimplement
   kill-switch-call/order-loop logic with subtly different behavior (one
   swallows a non-`KillSwitchTriggered` exception, the other doesn't). Collapse
   to one shared path before adding any new gate (compliance, reconciliation) —
   otherwise every new gate has to be built twice and will drift.
4. **Regime risk multiplier**: currently wrapped in a bare `try/except: pass`
   that silently no-ops (`00` finding). Either wire it for real or delete it —
   dead code disguised as live risk machinery is worse than no code, because it
   reads as a safeguard that isn't one.

**Effort**: all four are S (small) — days, not weeks, each touching 1-3 files.
**Do this before anything else** — it's mostly deleting false confidence or
connecting wires that already exist.

---

## Phase 1 — Independent Risk Management (shadow mode)

Closes the audit's single biggest structural gap: no independent second line
that can reject/shrink one position without the desk's sign-off (`00` scorecard
— "Absent"; full design in `01`).

- Build `ascent/risk/irm/` per `01`'s Layer 5: `model_risk_reviewer.py`
  (cache-staleness/NaN-rate pre-flight — cheapest to build, highest immediate
  value since it would have caught the recurring `prices_live` corruption
  class of bug), `market_risk_analyst.py` (VaR/CVaR, reuses existing
  `covariance_model.py`), `credit_risk.py` (thin — single broker), `cro.py`
  (aggregator).
- **Ship in shadow mode first** (`SHADOW_MODE=True`, logs decisions via
  `applied=False`, same pattern as the debate judge/earned-authority/falsifier
  trim already use) — do not give it live veto power until it has a validation
  window with an artifact-backed positive result, per the project's own
  integrity constraint #5 precedent.
- **Skip the Stress-Test Lead role initially** — it's the most build-effort
  (scenario-replay infrastructure) for the least immediate risk reduction given
  Ascent is single-strategy, not multi-desk. Defer to Phase 4+ once multiple
  agents are live and correlated-tail-risk actually matters more.

**Effort**: M-L. Model Risk Reviewer alone is S and should ship first,
independently of the rest — it's the one role that would have prevented a
bug that has actually recurred three times in this codebase.

---

## Phase 2 — Alpha Research: overfitting correction + capital ramp

Closes the audit's "Partial" verdict on the alpha validation pipeline (`00`;
full design in `02`; statistical grounding rigorously re-verified in `10`
Stages 2-4).

- Add `PerformanceAnalyzer.deflated_sharpe_ratio()` and `.pbo_cscv()` to
  `wf_framework/metrics.py`, per `02` Layer 5 — the walk-forward framework
  already produces the fold results these methods need; this is additive, not
  a rewrite.
- Make `ParameterOptimizer.optimize()` return `n_trials` alongside the winning
  config — a small, low-risk signature change, but mandatory: DSR/PBO are
  meaningless without trial count.
- **Do not build a numeric staged capital-ramp schedule as if it's an
  industry-standard number** — `10` Stage 6 found no such schedule is
  publicly documented anywhere; any specific percentage/day-count is Ascent's
  own reasoned design choice, and should be labeled as such in code comments,
  not presented as "how real funds do it."
- Leave `SELF_MODIFY_ENABLED = False` until DSR/PBO gating exists — flipping
  it on before the statistical correction exists just automates the
  overfitting problem faster.

**Effort**: M. The statistics module is self-contained and testable
independently of the rest of the pipeline.

---

## Phase 3 — Trading & Execution: compliance gate + reconciliation

Closes two real gaps `03` found: `LARGE_TRADE_THRESHOLD_PCT` is a dead
constant with no enforcement site, and nothing compares Alpaca's book to the
internal ledger.

- `ascent/execution/compliance_gate.py` — restricted list (empty list is fine
  initially, the mechanism matters more than populated content today),
  position-limit re-check against *live* broker positions (not the stale
  weights snapshot), buying-power check, and the large-order approval gate
  `LARGE_TRADE_THRESHOLD_PCT` was always meant to enforce.
- `ascent/execution/reconciliation.py` — nightly Alpaca-vs-internal-ledger
  diff, thresholds per `03`/`05` ($250 or 1 share single-position, $1,000
  aggregate → halt). This is genuinely novel infrastructure, not a wire-up —
  budget real time for it.
- Depends on Phase 0 item 3 (collapsed order path) — build this after, not
  before, or it has to be built twice.

**Effort**: M. Compliance gate is the more valuable of the two and can ship
first independently.

---

## Phase 4 — CIO / Capital Allocation: promote a second real strategy

Closes the audit's "Partial" verdict on multi-desk structure (`00`; design in
`04`). This is the highest-effort, highest-payoff structural phase — it's what
actually gives Ascent breadth (Grinold-Kahn IR ≈ IC × √breadth), which is the
one lever the whole document set keeps returning to as the honest way to
improve risk-adjusted return without adding leverage.

- Build `orchestrator/cio_committee.py` per `04` Layer 5.
- **Before promoting anything**, compute the actual correlation between
  `us_equities` and each of macro/international/alternatives using their
  already-running independent pipelines' historical outputs — the whole
  point of breadth is that it only pays off if the added strategy is
  genuinely weakly correlated (`00`'s verified caveat on the Fundamental Law:
  IC and breadth aren't independent; correlated "breadth" doesn't count).
  Don't promote on Sharpe alone.
- `alternatives_agent` is likely the best breadth candidate precisely because
  its universe (commodities/alts) is structurally least correlated with
  `us_equities` — but per CLAUDE.md, it's currently unmeasurable by the
  existing harness (universe too small for the long-short leg construction).
  **Fixing that measurement gap may be a prerequisite**, not just a nice-to-have.
- Wire the correlation guard's currently-dead `len(agent_weights) >= 2` gate
  (`central_intelligence.py:830`) — it becomes live automatically the moment a
  second agent is promoted; no code change needed there, just the state change
  from promotion.

**Effort**: L. This is the phase most worth doing carefully rather than fast —
get the correlation-gated promotion criteria right before flipping any
capital-allocating switch.

---

## Phase 5 — Compliance & Middle Office: the remaining net-new pieces

Phase 0 already wires the existing audit trail into its gaps. What's left,
per `05`:

- `ascent/compliance/surveillance.py` — post-submission deviation/rejection/
  silent-drop flagging, thresholds per `05`.
- `ascent/compliance/data_integrity.py` — pre-flight cache checker, built
  directly on `scripts/reconcile_numbers.py`'s existing duplicate/phantom-row
  logic (don't reimplement — that script becomes this module's library).
  **This one is worth pulling forward earlier** if Phase 1's Model Risk
  Reviewer isn't built first — they overlap significantly in purpose (catching
  bad data before it propagates) and should share the underlying check logic
  regardless of which phase ships first.

**Effort**: S-M, mostly because the hard part (audit trail) is Phase 0, and
the data-integrity logic already exists in `reconcile_numbers.py`.

---

## Phase 6 — Judgment & Governance: fix the promotion-path bug

Closes a real, narrow, high-value bug `06` identified: `earned_authority.py`'s
promotion path is logically complete but practically dead because
`update_authority()` requires both `track_d_return` and `track_astar_return`
non-`None` on the same call, and upstream callers don't reliably supply both
every day — so the buffer never reaches the required window length and
promotion never fires, only demotion.

- Audit every call site of `update_authority()` in the daily pipeline; ensure
  both returns are computed and passed every trading day the buffer should
  accrue.
- Add the `PROMOTION_PATH_STALLED` alarm `06` specifies — distinct from
  `is_stuck()`, which currently conflates "no promotion earned" with
  "promotion mechanically impossible."
- **This is a bug fix, not new infrastructure** — cheap, and directly closes
  the gap between "demotion-only in practice" and the asymmetric-but-bidirectional
  authority ladder the design already specifies on paper.

**Effort**: S. Do this early — it's a one-file, well-understood fix with an
outsized correctness payoff (the promotion mechanism has existed, unused, this
whole time).

---

## Phase 7 — The lifecycle's missing back half (new, from `10`)

`10_investment_thesis_lifecycle.md`'s most important finding: Stages 7-9 (live
decay monitoring, exit criteria, post-mortem) are the least-documented part of
the *entire industry*, and Ascent's code has zero coverage of any of them —
not because Ascent skipped a step everyone else has solved, but because this
is genuinely the industry's weakest-operationalized stage. That makes it high
leverage: a thin version here plausibly exceeds typical practice.

1. **Rolling IC/Sharpe monitor per sleeve** — a small, self-contained
   dashboard/log extending `dashboard/agent_skill_scores.json`'s existing
   shape to track rolling IC (not just Sharpe) per sleeve, with the explicit
   caveat from `10` Stage 7 built into any alerting: distinguishing decay from
   noise needs years of data at realistic Sharpe levels, so this is a
   trend-flagging tool, not an auto-decision trigger.
2. **A pre-committed, mechanical sleeve-level cut rule** — distinct from the
   whole-book kill switch, which already exists. `10` Stage 8's finding: the
   entire reason mechanical rules exist is to remove the sleeve's own
   advocate (whoever/whatever built it) from the cut decision, countering
   sunk-cost bias. A simple version: N consecutive rebalances of sleeve Sharpe
   below a pre-set threshold auto-flags for demotion review (this already
   partially exists for cross-strategy demotion in `04`'s Gatekeeper design —
   extend the same mechanism down to the sleeve level within `us_equities`).
3. **A plain, append-only "rejected hypothesis" log** — signal ID, thesis,
   why it failed (DSR/PBO fail, paper-trading fail, live decay), date. Cheap
   to build (one JSONL file, one append function), and directly prevents the
   single most wasteful failure mode in quant research: re-testing the same
   already-falsified idea because nobody remembers it was tried.

**Effort**: S per item. All three are cheap, additive, and don't touch the
live-trading path at all — good candidates for early, low-risk shipping
alongside Phase 0.

---

## What to cut

- **The dead `_apply_crisis_veto`** (`central_intelligence.py`) — keep, don't
  cut, but only until Phase 4 promotes a second agent; if Phase 4 is
  deprioritized long-term, revisit whether to delete it rather than let it
  keep reading as "risk machinery" that isn't reachable.
- **The regime risk multiplier's silent try/except** (Phase 0 item 4) — cut
  or fix, don't leave as-is.
- **Nothing else in the current codebase should be deleted.** Every other
  "theater" item this audit found (falsifier trim, judge position-change,
  earned-authority blend) is *already* correctly inert-but-logged per
  integrity constraint #5 — that's the right state for an unproven mechanism,
  not a bug to fix by deleting the logging.

---

## Suggested execution order (single-operator pace, not a team sprint)

1. Phase 0 (all 4 items) + Phase 7 items 1 and 3 (cheapest, no live-path risk)
2. Phase 6 (bug fix, high payoff, low effort)
3. Phase 1's Model Risk Reviewer alone (highest-value single component)
4. Phase 2 (statistics module, self-contained)
5. Phase 3's compliance gate
6. Phase 5's remaining pieces (surveillance, data integrity — much of the hard
   part already exists)
7. Phase 1's remaining IRM roles, Phase 3's reconciliation, Phase 7 item 2
8. Phase 4 last — highest effort, and depends on getting the correlation-gated
   promotion criteria right, which depends on measurement work
   (`alternatives_agent`'s universe-size gap) that should happen before, not
   during, the promotion decision.

This order front-loads everything that's cheap, self-contained, and doesn't
touch the live order-submission path, and defers the one phase (CIO/multi-desk)
that's both the most valuable long-term and the easiest to get wrong if rushed.
