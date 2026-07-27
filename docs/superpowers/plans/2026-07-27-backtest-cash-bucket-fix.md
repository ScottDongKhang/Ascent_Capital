# Backtest Cash-Bucket Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `BacktestEngine` from silently re-levering a partially-invested book back to 100% gross exposure on every non-rebalance day, so that de-risking overlays (200MA cut, vol targeting, stop-losses) actually take effect in research.

**Architecture:** The engine drifts weights day-to-day with `drifted / drifted.sum()`, which renormalizes to gross 1.0 unconditionally. The fix divides by the *portfolio* value factor (invested + cash) instead of the invested-only sum, so a book at 0.70 gross stays near 0.70 as prices move. Single-line semantic change plus a regression test; no API change.

**Tech Stack:** Python 3.12, pandas, numpy, pytest.

## Global Constraints

- Always use `.venv/bin/python`. Never the system Python.
- Use `import logging`; **never** `from loguru import logger` (loguru is not installed).
- Config access via `get_config()` — never `Config()` directly.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- Do not change `BacktestEngine.run()`'s signature or `BacktestResult`'s fields.
- Cash earns 0% in this model. Introducing a risk-free rate is explicitly **out of scope**.
- This is a **bug fix**, not a feature — it is NOT config-gated. It changes existing backtest numbers on purpose.

---

## Why this is a prerequisite

Empirically verified on 2026-07-27 with a synthetic book targeting a constant 0.50 gross:

```
2025-01-30    0.5   <- rebalance day, correct
2025-01-31    1.0   <- next day, silently re-levered 2x
2025-02-03    1.0
... stays 1.0 until the next rebalance
```

Consequences:
1. The 200MA cut (`×0.70`) and vol targeting (floor `×0.25`) survive exactly **one day per 21-day cycle** in research.
2. Any stop-loss that exits to cash is erased the next day.
3. The WF OOS figures in `CURRENT_VERIFIED_NUMBERS.md` (Sharpe 0.41, CAGR +10.3%, max DD −32.9%) were produced with both portfolio-level risk overlays effectively disabled. Those numbers must be re-run after this fix and the doc updated.

Production is **not** affected — `ascent/main.py` submits target weights to Alpaca directly and does not route through `BacktestEngine`. This is a research-vs-live divergence of the same class that `ascent/portfolio/exposure.py` was created to prevent.

---

## File Structure

- `ascent/backtest/engine.py` — the drift block at lines 74-82. One semantic change.
- `tests/backtest/test_engine_cash_bucket.py` — **new**. Owns the gross-exposure-preservation contract.
- `CURRENT_VERIFIED_NUMBERS.md` — WF figures re-run and restated after the fix.

---

### Task 1: Preserve gross exposure through weight drift

**Files:**
- Create: `tests/backtest/test_engine_cash_bucket.py`
- Modify: `ascent/backtest/engine.py:74-82`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no new public symbols. `BacktestEngine.run()` keeps its exact signature
  `run(target_weights: pd.DataFrame, close_prices: pd.DataFrame, open_prices: pd.DataFrame, benchmark_prices: pd.Series | None = None) -> BacktestResult`.
  Behavioural contract added: `result.held_weights.sum(axis=1)` tracks the target gross rather than snapping to 1.0.

- [ ] **Step 1: Create the test directory marker if missing**

```bash
mkdir -p tests/backtest && touch tests/backtest/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/backtest/test_engine_cash_bucket.py`:

```python
# tests/backtest/test_engine_cash_bucket.py
"""
Backtest engine must preserve a partially-invested book.

Before this fix the daily drift step renormalized weights by the invested
sum alone (`drifted / drifted.sum()`), which snaps gross exposure back to
1.0 on every non-rebalance day. That silently erased the 200MA cut, vol
targeting, and any stop-loss-to-cash rule everywhere except the single
rebalance day itself.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.backtest.engine import BacktestEngine


def _synthetic_market(n_days: int = 60, n_syms: int = 2, seed: int = 0):
    """Deterministic price panel; returns (close, open_) frames."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days)) for s in syms},
        index=dates,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    return close, open_


class TestCashBucketPreserved:
    def test_half_invested_book_is_not_relevered(self):
        """A book targeting 0.50 gross must stay near 0.50, not snap to 1.0."""
        close, open_ = _synthetic_market()
        # 0.25 + 0.25 = 0.50 gross on every date.
        tw = pd.DataFrame(0.25, index=close.index, columns=close.columns)

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        gross = res.held_weights.sum(axis=1)
        invested = gross[gross > 1e-9]  # skip the pre-first-signal cash period

        assert not invested.empty, "expected at least one invested day"
        # Drift around 0.50 is fine; snapping to 1.0 is the bug.
        assert invested.max() < 0.60, (
            f"gross exposure re-levered to {invested.max():.4f}; "
            "expected to stay near the 0.50 target"
        )

    def test_fully_invested_book_still_sums_to_one(self):
        """Regression: the common gross==1.0 case must be unchanged."""
        close, open_ = _synthetic_market()
        tw = pd.DataFrame(0.5, index=close.index, columns=close.columns)  # 1.0 gross

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        gross = res.held_weights.sum(axis=1)
        invested = gross[gross > 1e-9]

        assert not invested.empty
        assert invested.max() == pytest.approx(1.0, abs=1e-6)
        assert invested.min() == pytest.approx(1.0, abs=1e-6)

    def test_derisked_book_earns_less_than_full_book(self):
        """The economic point: half the exposure means roughly half the return."""
        close, open_ = _synthetic_market(seed=3)
        half = pd.DataFrame(0.25, index=close.index, columns=close.columns)
        full = pd.DataFrame(0.50, index=close.index, columns=close.columns)

        eng = lambda: BacktestEngine(rebalance_freq_days=21, execution_delay=1)
        r_half = eng().run(half, close, open_).portfolio_returns
        r_full = eng().run(full, close, open_).portfolio_returns

        # Compare only days where the full book is actually invested.
        mask = r_full.abs() > 1e-12
        assert mask.any()
        ratio = r_half[mask].abs().sum() / r_full[mask].abs().sum()
        # Before the fix this ratio is ~1.0 (identical books after day 1).
        assert ratio < 0.75, (
            f"de-risked book moved {ratio:.3f}x as much as the full book; "
            "expected roughly half"
        )

    def test_all_cash_book_does_not_divide_by_zero(self):
        """A zero-weight book must stay flat rather than error or NaN."""
        close, open_ = _synthetic_market()
        tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        res = BacktestEngine(rebalance_freq_days=21, execution_delay=1).run(
            tw, close, open_
        )
        assert res.held_weights.sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
        assert not res.portfolio_returns.isna().any()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/backtest/test_engine_cash_bucket.py -v`

Expected: `test_half_invested_book_is_not_relevered` FAILS with gross exposure ≈ 1.0, and `test_derisked_book_earns_less_than_full_book` FAILS with ratio ≈ 1.0. The other two should already pass.

- [ ] **Step 4: Apply the fix**

In `ascent/backtest/engine.py`, replace the drift block (currently lines 74-82):

```python
            # Drift weights from previous day
            if i > 0:
                prev_dt = common_dates[i - 1]
                ret     = daily_returns.loc[prev_dt]
                drifted = prev_weights * (1 + ret)
                total   = drifted.sum()
                current_weights = drifted / total if total > 0 else prev_weights.copy()
            else:
                current_weights = prev_weights.copy()
```

with:

```python
            # Drift weights from previous day.
            #
            # Weights are fractions of TOTAL portfolio value, including the
            # cash bucket (1 - gross). Renormalizing by the invested sum alone
            # silently re-levers a partially-invested book back to gross 1.0
            # on every non-rebalance day, erasing the 200MA cut, vol targeting
            # and any stop-loss-to-cash rule. Divide by the portfolio value
            # factor instead: invested_after_drift + cash (cash returns 0).
            if i > 0:
                prev_dt = common_dates[i - 1]
                ret     = daily_returns.loc[prev_dt]
                drifted = prev_weights * (1 + ret)
                cash    = 1.0 - float(prev_weights.sum())
                port_factor = float(drifted.sum()) + cash
                current_weights = (
                    drifted / port_factor if abs(port_factor) > 1e-12
                    else prev_weights.copy()
                )
            else:
                current_weights = prev_weights.copy()
```

Sanity of the arithmetic:
- gross 1.0, cash 0.0 → `port_factor == drifted.sum()` → identical to old behaviour (regression safe).
- gross 0.5, cash 0.5, flat returns → `port_factor = 0.5 + 0.5 = 1.0` → gross stays 0.5.
- gross 0.5, assets +10% → `port_factor = 0.55 + 0.5 = 1.05` → gross 0.5238 (assets correctly grow relative to cash).
- gross 0.0, cash 1.0 → `port_factor = 1.0` → stays 0.0, no division by zero.

- [ ] **Step 5: Verify the file still parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('ascent/backtest/engine.py').read())"`
Expected: no output.

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/backtest/test_engine_cash_bucket.py -v`
Expected: 4 passed.

- [ ] **Step 7: Run the surrounding suites and separate pre-existing failures**

Run:
```bash
.venv/bin/python -m pytest tests/backtest tests/portfolio tests/test_mvo_optimizer.py -v 2>&1 | tail -40
```

Expected: the new tests pass. Some backtest/WF tests will show **different numbers** — that is the intended effect of the fix, not a regression. For each failure, classify it explicitly:
- **Expected-change:** the test asserts a return/Sharpe/drawdown number produced by the old re-levering behaviour. Update the expected value and add a comment citing this plan.
- **Real regression:** the test asserts an invariant (weights sum, no-NaN, ordering). Do NOT update the expectation — fix the code.

Confirm the baseline first so you can tell the two apart:
```bash
git stash && .venv/bin/python -m pytest tests/backtest tests/portfolio -v 2>&1 | tail -20 && git stash pop
```

- [ ] **Step 8: Commit**

```bash
git add ascent/backtest/engine.py tests/backtest/__init__.py tests/backtest/test_engine_cash_bucket.py
git commit -m "fix(backtest): preserve cash bucket in weight drift

The daily drift step renormalized by the invested sum alone, snapping a
partially-invested book back to gross 1.0 on every non-rebalance day. The
200MA cut, vol targeting and any stop-loss-to-cash rule therefore survived
exactly one day per rebalance cycle in research. Divide by the portfolio
value factor (invested + cash) instead.

Verified: a book targeting 0.50 gross held 0.50 for one day then 1.00 for
the next 20. Changes existing backtest numbers by design."
```

---

### Task 2: Re-run walk-forward and restate the verified numbers

**Files:**
- Modify: `CURRENT_VERIFIED_NUMBERS.md`
- Modify: `CLAUDE.md` (the "Current state" WF bullet)
- Create: `outputs/wf_results/wf_report_cashfix_2026-07-27.json` (produced by the run)

**Interfaces:**
- Consumes: the corrected `BacktestEngine` from Task 1.
- Produces: a restated WF OOS figure that supersedes the Sharpe 0.41 / CAGR +10.3% line.

- [ ] **Step 1: Locate the walk-forward entrypoint**

Run: `grep -rn "def run_walk_forward\|wf_report" ascent/research/wf_framework/*.py scripts/*.py | head -20`

Use the same entrypoint and configuration that produced `outputs/wf_results/wf_report_clean_2026-06-22.json`. Read that artifact's header first to copy the exact fold count, window, and sleeve settings (`llm_fundamental` and `narrative` were zeroed and logged as `skipped`).

- [ ] **Step 2: Run walk-forward on the corrected engine**

Run the WF with output to `outputs/wf_results/wf_report_cashfix_2026-07-27.json`.

Expect this to take a while. Do **not** change any strategy parameters — the only difference from the 2026-06-22 run must be the engine fix, so the delta is attributable.

- [ ] **Step 3: Compare against the prior artifact**

```bash
.venv/bin/python - <<'PY'
import json
old = json.load(open('outputs/wf_results/wf_report_clean_2026-06-22.json'))
new = json.load(open('outputs/wf_results/wf_report_cashfix_2026-07-27.json'))
for k in ('sharpe', 'cagr', 'max_drawdown', 'beta', 'win_rate'):
    print(f"{k:15s} old={old.get(k)!r:>12} new={new.get(k)!r:>12}")
PY
```

Note the key names may differ — inspect `old.keys()` first and adapt.

**Directional expectation to sanity-check against:** with the overlays now actually biting, gross exposure is lower on average, so CAGR should *fall* and max drawdown should *shrink* (less negative). If drawdown got worse, stop and investigate — that contradicts the mechanism and means something else changed.

- [ ] **Step 4: Update the single source of truth**

In `CURRENT_VERIFIED_NUMBERS.md` §1, replace the Sharpe 0.41 / CAGR +10.3% / max DD −32.9% figures with the new run. Keep the old numbers visible in a superseded line with the reason:

> Superseded 2026-07-27. The 2026-06-22 figures were produced by a backtest engine that re-levered any partially-invested book back to gross 1.0 on every non-rebalance day, so the 200MA cut and vol targeting were inactive for ~20 of every 21 days. See `docs/superpowers/plans/2026-07-27-backtest-cash-bucket-fix.md`.

Mirror the same correction in the `CLAUDE.md` "Current state" WF bullet.

- [ ] **Step 5: Commit**

```bash
git add CURRENT_VERIFIED_NUMBERS.md CLAUDE.md outputs/wf_results/wf_report_cashfix_2026-07-27.json
git commit -m "docs: restate WF OOS numbers after backtest cash-bucket fix"
```

---

## Self-Review

- **Spec coverage:** the empirical bug (re-levering), the fix, the regression guard for gross==1.0, the divide-by-zero edge, and the downstream restatement of the verified numbers are all covered by Tasks 1-2.
- **Placeholders:** none. Every code step contains the literal code.
- **Type consistency:** no new public symbols introduced; `BacktestEngine.run()` and `BacktestResult` are untouched.
- **Known gap, deliberate:** cash earns 0% rather than the risk-free rate. Documented in Global Constraints as out of scope; it makes the de-risked book marginally conservative, which is the safe direction.
