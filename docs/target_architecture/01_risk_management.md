# Independent Risk Management — Target Architecture

## Layer 1 — Department Mandate

**Purpose.** Independent Risk Management (IRM) exists to protect NAV from tail loss,
hidden concentration, and broken models — independently of the desk that generates
the trades. Today, `ascent/portfolio/optimizer.py::_water_fill_cap()` sizes the book
*and* caps it in the same code path, and `ascent/execution/kill_switch.py` is the
only thing that reaches live behavior — a binary halt at 15% drawdown. There is no
second line: no independently-owned limit that can reject or shrink one specific
position without touching the rest of the book.

**Authority.** IRM is a second line under the Three Lines Model:
- It can **reject a specific proposed trade** (an individual `target_weights[symbol]`
  entry) without touching any other position, without needing PM sign-off, and
  without halting the book.
- It can **shrink** a position to a computed max-safe size and pass the rest of the
  portfolio through unmodified.
- It **cannot increase** a position, cannot select instruments, and cannot author
  alpha — that stays first line (`agents/us_equities_agent.py`,
  `ascent/portfolio/optimizer.py`).
- It **can** escalate to a book-wide halt only through the existing kill switch
  mechanism (unchanged, still binary), and only when a stress test or aggregate VaR
  breach is unresolvable by position-level shrink alone.
- Per integrity constraint #5's own logic: any override IRM applies to live weights
  is only earned once measured — so *on launch*, IRM ships in **shadow mode** (log
  every decision it would have made, `applied=False`), and is promoted to a live veto
  only after a defined validation window, exactly like the debate judge and
  earned-authority mechanisms were required to be before they were cut. This is the
  one deliberate deviation from "give it real teeth immediately" — it's how this
  codebase already earns live-write privileges.

**Reporting.** IRM's Chief Risk Officer (CRO) role emits a daily
`risk_report_<date>.json` (parallel to `outputs/debate_log/verdict_<date>.json`),
consumed by: the PM (as an input, not a request), `run_all_agents.py`'s rebalance
recap (Part 2 "things to watch"), and a new `risk_log.jsonl` under `logs/` for
backtestable audit. It reports to the same place the AI PM's verdict reports to —
logged and scored, promotable to real authority only with artifact-backed evidence.

## Layer 2 — Roles

| Role | Owns | Can veto | Escalates to |
|---|---|---|---|
| **Chief Risk Officer (CRO)** | Limit policy, the limit register itself (thresholds), aggregation across all other roles' findings into one report | Nothing directly — synthesizes and can trigger book-wide halt request to kill switch | Nobody (top of IRM); publishes to PM + logs |
| **Market Risk Analyst** | Position-level and portfolio-level VaR/CVaR, volatility, correlation/concentration limits | Individual position shrink/reject pre-trade | CRO, if aggregate portfolio VaR is breached even after all individual shrinks |
| **Credit/Counterparty Risk** | Broker margin usage, buying-power headroom, concentration in a single counterparty (currently only Alpaca, so this role is thin but still owns the check) | Order-level reject if margin/BP would be breached | CRO, if buying power methodology itself looks wrong (data issue) |
| **Stress-Test Lead** | Scenario/historical stress runs (2020 COVID crash, 2022 rate shock, custom shocks) against the *proposed* post-trade book | Book-wide shrink-to-target-vol recommendation (not per-symbol) | CRO always — stress breaches are portfolio-level by construction, not resolvable at position level |
| **Model Risk Reviewer** | Validates that the alpha/portfolio/regime models feeding today's proposal aren't themselves broken (NaN propagation, feature drift, stale cache, out-of-distribution inputs) | Reject the entire day's proposal and force fallback to last-known-good weights | CRO, always — a broken model invalidates every other role's inputs too |

## Layer 3 — Per-Role Responsibilities and Decision Logic

### Market Risk Analyst
- **Position VaR check.** For each proposed `target_weights[symbol]`, compute 1-day
  95% historical VaR from the trailing 252-day return series (reuse
  `ascent/risk/covariance_model.py`'s covariance estimate). Trigger: **position VaR
  > 2.5% of NAV** → shrink that position's weight until its VaR contribution equals
  2.5% of NAV; do not touch other weights.
- **Portfolio CVaR check.** Compute 1-day 95% CVaR (expected shortfall) on the full
  proposed post-trade book. Trigger: **portfolio CVaR > 4.0% of NAV** →
  proportionally shrink the 3 largest contributors to CVaR (by marginal CVaR) until
  aggregate CVaR ≤ 4.0%. If shrinking those 3 alone cannot bring it under threshold
  without dropping any one below `MIN_POSITIONS` viability, escalate to CRO.
- **Correlation/concentration limit.** Reuse `ascent/risk/correlation_guard.py`'s
  `CORRELATION_CAP = 0.70` (63-day trailing), but apply it *within* the single
  agent's book (today it's gated `len(agent_weights) >= 2` and unreachable per
  CLAUDE.md — this role's job is to make the equivalent check reachable
  intra-book). Trigger: **any two positions both >5% NAV with pairwise correlation >
  0.70** → halve the smaller of the two positions.
- **Vol-target check.** Trailing 20-day realized portfolio vol vs. target (read from
  `ascent/regime/`'s vol-target overlay config). Trigger: **realized vol > 1.5x
  target** → flag to CRO (informational, not auto-shrink — this overlaps the existing
  overlay and duplicate action would double-count).

### Credit/Counterparty Risk
- Reads Alpaca account buying power via `ascent/execution/alpaca_broker.py`.
  Trigger: **post-trade estimated margin usage > 90% of available buying power** →
  reject the marginal orders that would push it over 90%, largest orders first, until
  under threshold.
- Trigger: **single counterparty (Alpaca) exposure = 100% by construction** — logged
  as a standing structural finding every run (not actionable today, single-broker),
  escalated to CRO as a standing item, not per-trade noise.

### Stress-Test Lead
- Runs 3 canned historical scenarios (2020-02-19→2020-03-23 COVID drawdown, 2022
  rate-shock quarter, a synthetic +3-sigma single-day equity shock) by replaying
  each proposed symbol's historical beta/vol through the scenario's realized factor
  moves. Trigger: **worst-scenario portfolio loss > 12% of NAV** → recommend
  book-wide gross exposure cut to bring worst-case loss to ≤ 10% of NAV (a
  target-vol scale-down applied uniformly, not symbol selection — that's IRM's
  structural boundary: it can say "smaller," never "different").
- Escalates every stress-test breach to CRO regardless of magnitude — this role
  never self-resolves, by design (portfolio-level actions require CRO
  sign-off/aggregation so the daily report has one place stress-driven scale-downs
  are visible).

### Model Risk Reviewer
- Runs pre-flight validity checks: cache staleness (`prices_live` last row date vs.
  `market_today()`, reuse `ascent/utils/market_time.py`), NaN rate in the feature
  matrix feeding the optimizer (`_SPARSE_FILL_ZERO` panels), and feature-set drift
  (cached `feature_names` mismatch, per the existing ML-sleeve gotcha). Trigger:
  **any check fails** (stale cache > 1 trading day, NaN rate > 5% in any required
  panel, feature-name mismatch) → reject the entire day's proposal and signal
  `run_all_agents.py` to fall back to `_live_book_or(fallback)` — the existing "hold
  current book" pattern already used elsewhere in the file.
- Trigger: **regime label staleness** — cross-check `dashboard/regime_labels.csv`
  against `data_cache/ai_regime_assessment.json` (already a known lag per
  CLAUDE.md). If they disagree, flag (not reject) to CRO — this is informational
  because it's a known, tolerated lag, not a new failure mode.

### Chief Risk Officer
- Aggregates all four roles' findings into `risk_report_<date>.json`. Owns two
  thresholds nobody else does: **book-wide halt request** — if Stress-Test Lead's
  scale-down plus Market Risk Analyst's shrinks together still leave portfolio CVaR
  > 4.0% NAV after resolution, CRO requests `kill_switch` intervention (soft-warn
  path, not hard-stop, unless combined with an existing drawdown breach). And
  **limit register changes** — any threshold in this document is CRO-owned config,
  versioned, never silently edited by another role's code.

## Layer 4 — Interfaces / Data Contracts

All roles share one input contract and one output contract so they compose as a
pipeline stage, not four bespoke integrations.

**Common input** (each role's `check()` function receives):
```
RiskCheckInput = {
  "proposed_weights": dict[str, float],       # from optimizer/orchestrator merge, pre-execution
  "current_holdings": dict[str, float],       # live book, from alpaca_broker.get_positions()
  "price_history": pd.DataFrame,               # prices_live, wide, date-indexed
  "nav": float,                                 # current account equity
  "as_of_date": date,
}
```

**Common output** (each role emits one `RiskDecision` per symbol it acts on, plus
zero-or-more portfolio-level flags):
```
RiskDecision = {
  "role": str,                                  # "market_risk" | "credit" | "stress" | "model_risk"
  "symbol": str | None,                         # None = portfolio-level
  "action": "approve" | "shrink" | "reject" | "escalate" | "flag",
  "original_weight": float | None,
  "adjusted_weight": float | None,               # None if reject
  "reason": str,                                 # human-readable, cites the specific threshold breached
  "metric_value": float,                         # the actual computed VaR/CVaR/corr/etc.
  "threshold": float,
}
```

- **Market Risk Analyst** reads `proposed_weights` + covariance from
  `ascent/risk/covariance_model.py`; emits per-symbol `RiskDecision`s consumed by an
  aggregator before execution.
- **Credit/Counterparty** reads `proposed_weights` + live buying power from
  `alpaca_broker`; emits per-symbol reject decisions.
- **Stress-Test Lead** reads `proposed_weights` + historical scenario return series
  (new cache, `data_cache/stress_scenarios.parquet`); emits one portfolio-level
  `escalate` or `approve`.
- **Model Risk Reviewer** reads cache metadata (mtimes, NaN counts) — no price
  data; emits one portfolio-level `reject` or `approve` gating everything downstream
  (it runs *first*, since a broken model invalidates the rest).
- **CRO** reads the full list of `RiskDecision`s from the other three roles; emits
  `risk_report_<date>.json` = `{decisions: [...], net_weights: dict, halt_requested:
  bool, summary: str}`, consumed by (a) `run_eod_with_weights()` as the actual
  weights to submit, (b) the rebalance recap, (c) `logs/risk_log.jsonl` for
  backtest/audit.

## Layer 5 — Concrete Implementation Mapping

New package: **`ascent/risk/irm/`** (independent from `ascent/portfolio/` — this
separation *is* the point; it must not import from `ascent/portfolio/optimizer.py`
internals, only consume its output weights).

| Role | Module / class | Existing code reused | Net-new |
|---|---|---|---|
| Model Risk Reviewer | `ascent/risk/irm/model_risk_reviewer.py::ModelRiskReviewer.check()` | cache staleness pattern from `ascent/main.py` | NaN-rate / feature-drift checker |
| Market Risk Analyst | `ascent/risk/irm/market_risk_analyst.py::MarketRiskAnalyst.check()` | `ascent/risk/covariance_model.py`, `ascent/risk/correlation_guard.py` (generalize `CORRELATION_CAP` off its current `len(agent_weights)>=2` gate so it's reachable intra-book) | VaR/CVaR calculator (new, ~60 lines using historical simulation over `price_history`) |
| Credit/Counterparty | `ascent/risk/irm/credit_risk.py::CreditRiskAnalyst.check()` | `ascent/execution/alpaca_broker.py` (add a `get_buying_power()` read) | thin wrapper |
| Stress-Test Lead | `ascent/risk/irm/stress_test.py::StressTestLead.check()` | none directly — closest existing pattern is `ascent/backtest/` replay logic, reused for scenario replay | `data_cache/stress_scenarios.parquet` builder script |
| CRO aggregator | `ascent/risk/irm/cro.py::ChiefRiskOfficer.aggregate()` | pattern-matches `ascent/risk/pm_risk_validator.py::validate()`'s `(ok, violations)` shape, extended to per-symbol decisions | new |

**Insertion point.** `ascent/risk/pm_risk_validator.py::validate()` already runs at
`run_all_agents.py:1480` against the AI PM's *proposed* portfolio — but it's a
static hard-limit check (`MAX_POSITION`, `MAX_SECTOR`), not model-driven, and it does
not gate the *quant* weights, only the AI PM's advisory blend candidate. IRM needs to
sit downstream of both: after `merged_weights` is finalized (the value passed into
`run_eod_with_weights` at `run_all_agents.py:1897` and `:2973`) and *before*
`ascent/execution/eod_runner.py::run_eod_with_weights()` reaches order computation.

Concretely, insert inside `run_eod_with_weights()` (`ascent/execution/eod_runner.py`),
right after the existing "1.5. Debate layer" block (around line 830, once the debate
verdict has already potentially reduced size) and **before** the kill-switch check
at line 1012:

```python
# 1.6. Independent Risk Management (second line of defense)
from ascent.risk.irm.cro import ChiefRiskOfficer
cro = ChiefRiskOfficer()
risk_report = cro.aggregate(merged_weights, current_holdings, today_str)
_write_risk_log(today_str, risk_report)   # logs/risk_log.jsonl, mirrors _log_multi_run
if risk_report["halt_requested"]:
    print("[IRM] CRO requested halt -- routing to kill_switch soft-warn")
    # existing kill_switch.check() path, not a new halt mechanism
if SHADOW_MODE:
    print(f"[IRM] shadow mode: would apply {len(risk_report['decisions'])} decisions -- not applied")
else:
    merged_weights = risk_report["net_weights"]
```

This is the one place with real veto power over order submission — everything
upstream (AI PM Phase 2, debate judge, `pm_risk_validator`) only ever touches
`merged_weights` before it's finalized; this insertion touches the value that
literally becomes the order list a few hundred lines later in the same function.
Wiring `SHADOW_MODE=True` by default (config-driven, `get_config()`) is what makes
this consistent with integrity constraint #5's precedent: log everything,
`applied=False`, promote to real writes only after a validation window with an
artifact-backed positive result — exactly the standard the debate judge,
earned-authority blend, and falsifier trim were all held to and all failed or were
never tested against.
