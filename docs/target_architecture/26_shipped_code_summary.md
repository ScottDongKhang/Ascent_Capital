# Shipped Code — Five Real Implementation Tasks

Following the "stop planning, ship code" recommendation from `20`, five
agents implemented real, tested code against the specific gaps this project
had already found and fully specified. Nothing here is committed to git —
working-tree changes only, pending review.

## 1. Governance bug fix (`13`)

**Files**: `CLAUDE.md` (doc correction), `ascent/strategy/earned_authority.py`
(additive).

Added `days_since_last_buffer_append` to state, incremented on every
buffer-skip (the `None`-guard path), reset on every real append. When it
reaches `2 * window` for the current promotion level, logs a
`PROMOTION_PATH_STALLED` warning — distinct from `is_stuck()`, which
conflated "no promotion earned" with "promotion mechanically impossible."
No promotion/demotion decision logic changed.

**Verified**: `pytest tests/test_ai_pm_authority.py tests/test_ai_pm_counterfactual.py`
— 38 passed.

## 2. Audit trail wiring (`12` item 2)

**File**: `run_all_agents.py` (additive), `compliance/audit_trail.py`
(2 new event-type strings added: `halt_triggered`, `halt_overridden`).

`check_halt_state()` now writes both events to the existing hash-chained
audit trail (`compliance/audit_trail.py`, already live for
`order_submitted`), alongside the existing print statements. Control flow
and return values unchanged — confirmed by diff inspection.

**Verified**: `pytest tests/ -k "halt"` — 9 passed, including two tests that
specifically assert `halt_state.json` write behavior on halt vs. proceed
verdicts.

## 3. Rejected-hypothesis registry (`15` item 3)

**Files**: `ascent/research/hypothesis_registry.py` (new),
`ascent/research/self_improve.py` (additive, 22 lines).

`_config_hash()`, `was_previously_rejected()`, `record_verdict()` — an
already-tested config gets skipped (with its prior verdict reused) rather
than re-evaluated; every variant's verdict (promoted or not) gets appended
to `logs/hypothesis_registry.jsonl`. Reuses `edge`/`oos_sharpe`/`promoted`
values `self_improve.py` already computes — no new evaluation logic.

**Verified**: `pytest tests/ -k "self_improve or hypothesis"` — 14 passed
(4 new tests for the registry itself, 10 pre-existing self-improve tests
confirming no regression).

## 4. Model Risk Reviewer (`14`)

**Files**: `ascent/risk/irm/__init__.py`, `ascent/risk/irm/
model_risk_reviewer.py` (new, standalone — not yet wired into the live
pipeline), `tests/risk/test_model_risk_reviewer.py` (new).

**A real bug in the original skeleton was caught and fixed during
implementation**: the skeleton's cache-freshness check matched on the
literal word `"stale"` in the failure reason string. Reading
`validate_cache()`'s actual failure strings (`ascent/data/store/
parquet.py:184-272`) showed none of them contain that word — the literal
skeleton check would have silently passed every real failure mode (missing
cache file, missing symbols, phantom-row corruption, a stale-by-days cache
whose message doesn't use "stale" verbatim). Replaced with the correct
contract: any non-empty reason string fails, and a `prices_live_fallback_
simulated`/`prices_simulated` cache name fails even with an empty reason
since it's degraded data by construction.

**Verified**: `pytest tests/risk/test_model_risk_reviewer.py -v` — 14
passed, including a locked-in regression test for the exact bug above
(`test_nonempty_reason_fails_even_without_word_stale`).

**Not yet wired into `run_all_agents.py`/`ascent/main.py`** — deliberately
scoped standalone this round to avoid overlapping parallel work on the live
pipeline. Wiring is the next step, not done here.

## 5. Walk-forward runner persistence patch — real beta-hedged Sharpe

**File**: `ascent/research/walk_forward_runner.py` (+43/-1 lines): a new
persistence block writing `outputs/wf_results/wf_daily_returns_<date>.csv`
(date, strategy_return, benchmark_return) from `result.portfolio_returns`/
`result.benchmark_returns`, additive only — the existing JSON report and
ledger CSVs are unchanged. **One incidental bug fixed in the same diff**: an
unconditional `from ascent.dashboard.export_dashboard_data import
export_to_dashboard` outside the try/except was raising `ModuleNotFoundError`
unconditionally (that module was deleted in commit `ab392bb`), which would
have aborted the pipeline before reaching the ledger/report save step. Moved
inside the try block.

**A full 165-fold walk-forward run was executed** (not just a small-scale
patch validation) — `outputs/wf_results/wf_daily_returns_2026-08-20.csv`,
1641 rows, 2020-01-02 → 2026-07-15, matching the canonical artifact's
`n_oos_days` exactly.

### The real number

Computed directly from the persisted daily return series (not algebraically
estimated):

```
OLS-estimated beta from this data:  0.9468  (reported beta: 0.947 — matches almost exactly)
Correlation (strategy, SPY):        0.7830
Annualized hedged vol:              0.1529
Annualized hedged Sharpe:           -0.097
```

**Beta-hedged Sharpe ≈ -0.10.** This confirms the sign of `24`'s algebraic
estimate (≈-0.19) with real data — negative, not merely diminished — but at
roughly half the magnitude. The OLS beta matching the artifact's reported
beta to three decimal places is strong independent validation that this CSV
correctly reproduces the canonical run's actual return series, not a
different or corrupted computation.

**Updated verdict**: the "beta wearing a governance costume" finding from
`21`/`24` is now confirmed with real, directly-computed data, not an
algebraic approximation. The magnitude is somewhat less severe than the
algebraic estimate suggested (-0.10 vs -0.19), but the conclusion is
unchanged: after removing SPY beta exposure, this strategy has a negative
risk-adjusted return over the full canonical OOS window.

---

## Net effect

Five real code changes, all tests passing (75 total: 38+9+14+14 across
tasks 1-4, plus a full backtest run for task 5), zero regressions, one
genuine skeleton bug caught before it could ship (`model_risk_reviewer.py`'s
stale-string-matching check), one unrelated pipeline-breaking import bug
fixed as a side effect (the dashboard import), and the single most
important open empirical question in the project — is there real alpha
here — now has a real, computed, negative answer instead of an estimate.

Nothing is committed. Recommend reviewing the diffs, running the full test
suite once more, and deciding whether/when to commit — and treating the
walk-forward result as the decisive input to the `25` IC memo's "pause and
re-underwrite" recommendation, which is now confirmed rather than merely
well-argued.
