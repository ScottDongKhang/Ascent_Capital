# Strategy-Own Volatility Targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale portfolio exposure by the inverse of the **strategy's own** trailing realized volatility rather than SPY's, implementing Barroso & Santa-Clara (2015) and Moreira & Muir (2017).

**Architecture:** `vol_target_scale()` currently uses trailing SPY volatility as a proxy for portfolio volatility. For a book whose alpha is 70% trend, that is the wrong reference series — momentum's own volatility spikes ahead of momentum crashes while the market stays calm, which is exactly the 2026-06/07 pattern. This plan extracts a generic `realized_vol_scale()`, adds a causal `strategy_return_proxy()` computed from the weights and price panels, and lets `apply_exposure_overlays()` choose its reference series via config. SPY remains the default until walk-forward says otherwise.

**Tech Stack:** Python 3.12, pandas, numpy, pytest.

## Global Constraints

- **PREREQUISITE:** `docs/superpowers/plans/2026-07-27-backtest-cash-bucket-fix.md` must land first. Until it does, `BacktestEngine` re-levers any sub-1.0 gross book back to 1.0 the next day, so *no* exposure overlay can be validated — including this one. This plan's Task 5 is meaningless without it.
- **SEQUENCING:** this plan and `2026-07-27-momentum-crash-indicator.md` both modify `ascent/portfolio/exposure.py`. Land one, then rebase the other. Do not run them in parallel worktrees against the same file.
- Always use `.venv/bin/python`. Never the system Python.
- Use `import logging`; **never** `from loguru import logger` (loguru is not installed).
- Config access via `get_config()` — never `Config()` directly.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- `vol_target_scale()`'s existing signature and behaviour must be **bit-identical** after refactoring. `tests/portfolio/test_exposure.py` is the regression guard and must pass untouched.
- New config defaults to the **current** behaviour (`vol_target_reference = "spy"`). Switching the default is a separate, evidence-gated decision (Task 5).
- All computations stay causal: only returns strictly before each decision date.
- Overlays never raise. On failure, log and fall back to the SPY reference.

---

## Evidence

**Barroso & Santa-Clara (2015), "Momentum Has Its Moments":** momentum's risk is highly time-varying and predictable from *its own* realized variance — momentum volatility rises going into momentum crashes. Scaling the momentum portfolio by its own trailing realized volatility roughly doubled the strategy's Sharpe ratio in their sample.

**Moreira & Muir (2017), "Volatility-Managed Portfolios" (Journal of Finance):** the same mechanism generalizes. Scaling a factor by the inverse of its own previous-month realized variance produces large alphas and higher Sharpe ratios across the market, value, **momentum**, profitability, and investment factors, plus currency carry. Volatility timing works because factor volatility does not move proportionally with expected return — cutting exposure in high-vol periods sacrifices little expected return.

**Why the current code has the wrong reference:** `ascent/portfolio/exposure.py:78-109` computes `realized = SPY.pct_change().rolling(21).std() * sqrt(252)` and scales by `target_vol / realized`. During 2026-06-29 → 2026-07-24 SPY was roughly flat (−0.4%) while the book fell −3.4%, driven by single-name momentum reversals. A SPY-referenced overlay saw a calm market and did not de-risk. A strategy-referenced overlay would have seen its own volatility rising.

**Honest scope limit:** this would *not* have prevented the ALGM/MRNA loss on its own. It is a portfolio-level exposure control, so it scales the whole book down rather than targeting the two offending names. Its value is systemic — it makes the de-risking overlay respond to the risk the strategy is actually taking. The position-level fix is `2026-07-27-position-stop-loss.md`.

---

## File Structure

- `ascent/portfolio/exposure.py` — extract `realized_vol_scale()`, add `strategy_return_proxy()`, add a `vol_reference` parameter to `apply_exposure_overlays()`. `vol_target_scale()` stays as a thin, behaviour-preserving wrapper.
- `ascent/config/settings.py` — one new `BacktestConfig` field.
- `ascent/main.py` — pass the configured reference and the weights/price panels.
- `ascent/research/wf_framework/ascent_strategy.py` — same, in `_apply_vol_target`.
- `tests/portfolio/test_exposure_strategy_vol.py` — **new**.
- `tests/portfolio/test_exposure.py` — **must pass unchanged** (regression guard).

---

### Task 1: Extract a generic realized-volatility scaler

**Files:**
- Modify: `ascent/portfolio/exposure.py:78-109`
- Create: `tests/portfolio/test_exposure_strategy_vol.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `realized_vol_scale(returns: pd.Series, dates: pd.Index, target_vol: float = VOL_TARGET, lookback: int = VOL_LOOKBACK, floor: float = VOL_FLOOR, cap: float = VOL_CAP) -> pd.Series`.
  `vol_target_scale(spy_close, dates, ...)` keeps its exact existing signature and delegates.

- [ ] **Step 1: Write the failing test**

Create `tests/portfolio/test_exposure_strategy_vol.py`:

```python
# tests/portfolio/test_exposure_strategy_vol.py
"""
Strategy-own volatility targeting — Barroso & Santa-Clara (2015),
Moreira & Muir (2017).

vol_target_scale() referenced SPY as a proxy for portfolio volatility.
For a 70%-trend book that is the wrong series: momentum volatility rises
into momentum crashes while the market stays calm.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.exposure import (
    VOL_TARGET, VOL_FLOOR, VOL_CAP,
    realized_vol_scale,
    vol_target_scale,
)


def _returns(vol_ann: float, n: int = 120, seed: int = 0) -> pd.Series:
    """Daily returns with a known annualized volatility."""
    rng = np.random.default_rng(seed)
    daily = vol_ann / np.sqrt(252)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series(rng.normal(0.0, daily, n), index=idx)


class TestRealizedVolScale:
    def test_high_vol_scales_exposure_down(self):
        r = _returns(0.40)                     # 40% ann vs a 15% target
        dates = r.index[-20:]
        out = realized_vol_scale(r, dates, target_vol=0.15)
        assert (out < 1.0).all()
        assert out.mean() == pytest.approx(0.15 / 0.40, rel=0.35)

    def test_low_vol_is_capped_at_one(self):
        r = _returns(0.05)                     # calmer than target
        out = realized_vol_scale(r, r.index[-20:], target_vol=0.15, cap=1.0)
        assert (out <= 1.0 + 1e-12).all()

    def test_scale_respects_floor(self):
        r = _returns(3.00)                     # violently volatile
        out = realized_vol_scale(r, r.index[-20:], target_vol=0.15,
                                 floor=0.25, cap=1.0)
        assert (out >= 0.25 - 1e-12).all()

    def test_is_causal_future_returns_do_not_leak(self):
        """A spike AFTER date d must not change the scale AT date d."""
        r = _returns(0.15, n=80, seed=7)
        d = r.index[60]
        base = realized_vol_scale(r, pd.Index([d]), target_vol=0.15).iloc[0]

        spiked = r.copy()
        spiked.iloc[61:] = spiked.iloc[61:] * 20.0
        after = realized_vol_scale(spiked, pd.Index([d]), target_vol=0.15).iloc[0]

        assert base == pytest.approx(after, abs=1e-12)

    def test_insufficient_history_returns_one(self):
        r = _returns(0.40, n=3)
        out = realized_vol_scale(r, r.index, target_vol=0.15)
        assert (out == 1.0).all()

    def test_zero_volatility_returns_one(self):
        idx = pd.bdate_range("2025-01-01", periods=40)
        r = pd.Series(0.0, index=idx)
        out = realized_vol_scale(r, idx[-10:], target_vol=0.15)
        assert (out == 1.0).all()

    def test_empty_dates_returns_empty(self):
        r = _returns(0.20)
        assert realized_vol_scale(r, pd.Index([]), target_vol=0.15).empty


class TestVolTargetScaleUnchanged:
    """vol_target_scale() must be bit-identical after the refactor."""

    def test_delegates_to_realized_vol_scale_with_spy_returns(self):
        idx = pd.bdate_range("2025-01-01", periods=90)
        rng = np.random.default_rng(11)
        spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 90)),
                        index=idx)
        dates = idx[-15:]

        legacy = vol_target_scale(spy, dates, target_vol=VOL_TARGET,
                                  floor=VOL_FLOOR, cap=VOL_CAP)
        direct = realized_vol_scale(spy.pct_change().dropna(), dates,
                                    target_vol=VOL_TARGET,
                                    floor=VOL_FLOOR, cap=VOL_CAP)
        pd.testing.assert_series_equal(legacy, direct)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_exposure_strategy_vol.py -v`
Expected: FAIL — `ImportError: cannot import name 'realized_vol_scale'`.

- [ ] **Step 3: Refactor `exposure.py`**

Replace the body of `vol_target_scale` (lines 78-109) with:

```python
def realized_vol_scale(
    returns: pd.Series,
    dates: pd.Index,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    floor: float = VOL_FLOOR,
    cap: float = VOL_CAP,
) -> pd.Series:
    """
    Per-date exposure multiplier targeting `target_vol` annualized against an
    arbitrary daily return series.

    scale(d) = clip(target_vol / realized_vol(d), floor, cap), where
    realized_vol(d) uses returns strictly before d (fully causal).
    Dates with <5 trailing observations get scale 1.0.

    Barroso & Santa-Clara (2015) and Moreira & Muir (2017) both scale a
    factor by ITS OWN trailing realized volatility. Passing SPY returns here
    reproduces the legacy market-referenced behaviour; passing the strategy's
    own returns implements the papers.
    """
    if len(dates) == 0:
        return pd.Series(dtype=float)

    rets = returns.sort_index().dropna()

    scales = []
    for d in dates:
        past = rets[rets.index < d].iloc[-lookback:]
        if len(past) < 5:
            scales.append(1.0)
            continue
        realized = float(past.std() * np.sqrt(252))
        if realized < 1e-6:
            scales.append(1.0)
            continue
        scales.append(float(np.clip(target_vol / realized, floor, cap)))
    return pd.Series(scales, index=dates)


def vol_target_scale(
    spy_close: pd.Series,
    dates: pd.Index,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    floor: float = VOL_FLOOR,
    cap: float = VOL_CAP,
) -> pd.Series:
    """
    Market-referenced vol targeting: `realized_vol_scale` over SPY returns.

    Retained with its original signature so existing callers and tests are
    unaffected. New code should prefer `realized_vol_scale` with the
    strategy's own return series — see `strategy_return_proxy`.
    """
    spy_close = spy_close.sort_index()
    spy_close = spy_close[~spy_close.index.duplicated(keep="last")]
    return realized_vol_scale(
        spy_close.pct_change().dropna(), dates,
        target_vol=target_vol, lookback=lookback, floor=floor, cap=cap,
    )
```

- [ ] **Step 4: Verify parse, run new tests AND the existing regression guard**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/exposure.py').read())"
.venv/bin/python -m pytest tests/portfolio/test_exposure_strategy_vol.py tests/portfolio/test_exposure.py -v
```
Expected: all pass. `tests/portfolio/test_exposure.py` must pass **without edits** — if it does not, the refactor changed behaviour and must be corrected, not the test.

- [ ] **Step 5: Commit**

```bash
git add ascent/portfolio/exposure.py tests/portfolio/test_exposure_strategy_vol.py
git commit -m "refactor(portfolio): extract realized_vol_scale from vol_target_scale

Generic over any return series. vol_target_scale is now a behaviour-
preserving wrapper passing SPY returns. Prepares strategy-own vol
targeting per Barroso & Santa-Clara (2015) / Moreira & Muir (2017)."
```

---

### Task 2: Causal strategy return proxy

**Files:**
- Modify: `ascent/portfolio/exposure.py`
- Modify: `tests/portfolio/test_exposure_strategy_vol.py`

**Interfaces:**
- Consumes: nothing from Task 1 at runtime (used together by Task 3).
- Produces: `strategy_return_proxy(weights: pd.DataFrame, close: pd.DataFrame) -> pd.Series` — daily return series of the book, `r(t) = sum_i w_i(t-1) * ret_i(t)`.

**Why a proxy:** the strategy's realized return series is not available inside the strategy before the backtest runs. It is reconstructible from the two panels the strategy already holds, using yesterday's weights against today's returns — which is both the economically correct definition and causal by construction.

- [ ] **Step 1: Write the failing test**

Append to `tests/portfolio/test_exposure_strategy_vol.py`:

```python
from ascent.portfolio.exposure import strategy_return_proxy


class TestStrategyReturnProxy:
    def test_single_asset_full_weight_reproduces_asset_return(self):
        idx = pd.bdate_range("2025-01-01", periods=4)
        close = pd.DataFrame({"A": [100.0, 110.0, 99.0, 108.9]}, index=idx)
        w = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
        out = strategy_return_proxy(w, close)
        assert out.loc[idx[1]] == pytest.approx(0.10)
        assert out.loc[idx[2]] == pytest.approx(-0.10)

    def test_half_weight_earns_half_the_return(self):
        idx = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
        out = strategy_return_proxy(w, close)
        assert out.loc[idx[1]] == pytest.approx(0.05)

    def test_uses_yesterdays_weights_not_todays(self):
        """Causality: a weight set on day t must not earn day t's return."""
        idx = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [100.0, 200.0, 200.0]}, index=idx)
        # Zero weight on day 0 -> the +100% move on day 1 must NOT be earned.
        w = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
        out = strategy_return_proxy(w, close)
        assert out.loc[idx[1]] == pytest.approx(0.0)

    def test_cash_position_contributes_zero(self):
        idx = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [100.0, 110.0, 121.0],
                              "B": [50.0, 55.0, 60.5]}, index=idx)
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.0, 0.0, 0.0]}, index=idx)
        out = strategy_return_proxy(w, close)
        assert out.loc[idx[1]] == pytest.approx(0.05)

    def test_missing_price_column_is_ignored_not_fatal(self):
        idx = pd.bdate_range("2025-01-01", periods=3)
        close = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "GHOST": [0.5, 0.5, 0.5]},
                         index=idx)
        out = strategy_return_proxy(w, close)
        assert out.loc[idx[1]] == pytest.approx(0.05)
        assert not out.isna().any()

    def test_empty_inputs_return_empty(self):
        assert strategy_return_proxy(pd.DataFrame(), pd.DataFrame()).empty
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_exposure_strategy_vol.py::TestStrategyReturnProxy -v`
Expected: FAIL — `ImportError: cannot import name 'strategy_return_proxy'`.

- [ ] **Step 3: Write the implementation**

Append to `ascent/portfolio/exposure.py`:

```python
def strategy_return_proxy(
    weights: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.Series:
    """
    Daily return series of the book: r(t) = sum_i w_i(t-1) * ret_i(t).

    Causal by construction — yesterday's weights against today's returns.
    Used as the reference series for strategy-own volatility targeting
    (Barroso & Santa-Clara 2015, Moreira & Muir 2017), because the realized
    return series is not available inside the strategy before it runs.

    Symbols present in `weights` but absent from `close` contribute zero
    rather than NaN, so one missing ticker cannot blank the whole series.
    """
    if weights is None or weights.empty or close is None or close.empty:
        return pd.Series(dtype=float)

    cols = [c for c in weights.columns if c in close.columns]
    if not cols:
        return pd.Series(0.0, index=weights.index)

    w = weights[cols].astype(float).fillna(0.0)
    rets = close[cols].reindex(w.index).pct_change().fillna(0.0)

    # shift(1): weights known at the close of t-1 earn t's return.
    return (w.shift(1).fillna(0.0) * rets).sum(axis=1)
```

- [ ] **Step 4: Verify parse and run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/exposure.py').read())"
.venv/bin/python -m pytest tests/portfolio/test_exposure_strategy_vol.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ascent/portfolio/exposure.py tests/portfolio/test_exposure_strategy_vol.py
git commit -m "feat(portfolio): add causal strategy_return_proxy

r(t) = sum w(t-1) * ret(t). Reference series for strategy-own vol targeting."
```

---

### Task 3: Reference selection in `apply_exposure_overlays`

**Files:**
- Modify: `ascent/portfolio/exposure.py:112-157`
- Modify: `tests/portfolio/test_exposure_strategy_vol.py`

**Interfaces:**
- Consumes: `realized_vol_scale`, `strategy_return_proxy`.
- Produces: `apply_exposure_overlays(..., vol_reference: str = "spy", close: pd.DataFrame | None = None)`. Every existing parameter keeps its position and default. Returned `meta` gains `"vol_reference": str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/portfolio/test_exposure_strategy_vol.py`:

```python
from ascent.portfolio.exposure import apply_exposure_overlays


def _calm_spy(idx):
    """SPY drifting up quietly: ~6% annualized vol."""
    rng = np.random.default_rng(2)
    return pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.06 / np.sqrt(252), len(idx))),
        index=idx,
    )


class TestVolReferenceSelection:
    def test_default_is_spy_and_unchanged(self):
        idx = pd.bdate_range("2025-01-01", periods=90)
        spy = _calm_spy(idx)
        w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
        a, meta_a = apply_exposure_overlays(w, spy)
        b, meta_b = apply_exposure_overlays(w, spy, vol_reference="spy")
        pd.testing.assert_frame_equal(a, b)
        assert meta_b["vol_reference"] == "spy"

    def test_strategy_reference_derisks_when_book_is_wild_but_spy_is_calm(self):
        """The 2026-06/07 pattern: flat market, violent single names."""
        idx = pd.bdate_range("2025-01-01", periods=90)
        spy = _calm_spy(idx)
        rng = np.random.default_rng(5)
        # Book holds one very volatile name (~60% ann).
        close = pd.DataFrame(
            {"WILD": 100 * np.cumprod(
                1 + rng.normal(0.0, 0.60 / np.sqrt(252), len(idx)))},
            index=idx,
        )
        w = pd.DataFrame(1.0, index=idx, columns=["WILD"])

        spy_scaled, _ = apply_exposure_overlays(
            w, spy, vol_reference="spy", close=close, rebalance_only=False)
        str_scaled, meta = apply_exposure_overlays(
            w, spy, vol_reference="strategy", close=close, rebalance_only=False)

        assert meta["vol_reference"] == "strategy"
        # Strategy-referenced must cut exposure harder than the calm-SPY view.
        assert str_scaled.iloc[-1].sum() < spy_scaled.iloc[-1].sum()

    def test_strategy_reference_without_close_falls_back_to_spy(self, caplog):
        idx = pd.bdate_range("2025-01-01", periods=90)
        spy = _calm_spy(idx)
        w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
        with caplog.at_level("WARNING"):
            out, meta = apply_exposure_overlays(w, spy, vol_reference="strategy")
        expected, _ = apply_exposure_overlays(w, spy, vol_reference="spy")
        pd.testing.assert_frame_equal(out, expected)
        assert meta["vol_reference"] == "spy"

    def test_unknown_reference_falls_back_to_spy(self, caplog):
        idx = pd.bdate_range("2025-01-01", periods=60)
        spy = _calm_spy(idx)
        w = pd.DataFrame(0.5, index=idx, columns=["A"])
        with caplog.at_level("WARNING"):
            _, meta = apply_exposure_overlays(w, spy, vol_reference="banana")
        assert meta["vol_reference"] == "spy"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/portfolio/test_exposure_strategy_vol.py::TestVolReferenceSelection -v`
Expected: FAIL — `TypeError: apply_exposure_overlays() got an unexpected keyword argument 'vol_reference'`.

- [ ] **Step 3: Modify `apply_exposure_overlays`**

Add the two parameters at the **end** of the signature (so positional callers are unaffected):

```python
    vol_targeting_enabled: bool = True,
    vol_reference: str = "spy",
    close: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
```

Replace the `if vol_targeting_enabled:` block with:

```python
    ref = str(vol_reference or "spy").lower()
    if ref not in ("spy", "strategy"):
        log.warning("[Exposure] Unknown vol_reference %r — using 'spy'",
                    vol_reference)
        ref = "spy"
    if ref == "strategy" and (close is None or close.empty):
        log.warning("[Exposure] vol_reference='strategy' needs a close panel "
                    "— falling back to 'spy'")
        ref = "spy"

    if vol_targeting_enabled:
        if ref == "strategy":
            strat_rets = strategy_return_proxy(weights, close)
            v_scale = realized_vol_scale(strat_rets, dates,
                                         target_vol=target_vol,
                                         floor=vol_floor, cap=vol_cap)
        else:
            v_scale = vol_target_scale(spy_close, dates, target_vol=target_vol,
                                       floor=vol_floor, cap=vol_cap)
    else:
        v_scale = pd.Series(1.0, index=dates)
```

Add `"vol_reference": ref` to the returned `meta` dict.

- [ ] **Step 4: Verify parse and run the full exposure suite**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/portfolio/exposure.py').read())"
.venv/bin/python -m pytest tests/portfolio/ -v
```
Expected: all pass, `tests/portfolio/test_exposure.py` still unedited.

- [ ] **Step 5: Commit**

```bash
git add ascent/portfolio/exposure.py tests/portfolio/test_exposure_strategy_vol.py
git commit -m "feat(portfolio): vol_reference option on apply_exposure_overlays

'spy' (default, unchanged) or 'strategy' (Barroso & Santa-Clara). Falls
back to spy when the close panel is missing or the value is unknown."
```

---

### Task 4: Config flag and wiring

**Files:**
- Modify: `ascent/config/settings.py` (after `stop_loss_*` if that plan landed, else after `risk_budget_per_name`)
- Modify: `ascent/main.py:825` (the `apply_exposure_overlays` call)
- Modify: `ascent/research/wf_framework/ascent_strategy.py:339-375` (`_apply_vol_target`)
- Create: `tests/test_vol_reference_config.py`

**Interfaces:**
- Consumes: `apply_exposure_overlays(..., vol_reference=, close=)`.
- Produces: `BacktestConfig.vol_target_reference: str = "spy"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vol_reference_config.py`:

```python
# tests/test_vol_reference_config.py
"""Vol-targeting reference series. Defaults to the CURRENT behaviour."""
from ascent.config.settings import get_config


def test_vol_reference_defaults_to_spy():
    assert get_config().backtest.vol_target_reference == "spy", (
        "must default to the existing market-referenced behaviour until the "
        "walk-forward comparison in the strategy-own-vol-targeting plan"
    )


def test_vol_reference_is_a_known_value():
    assert get_config().backtest.vol_target_reference in ("spy", "strategy")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_vol_reference_config.py -v`
Expected: FAIL — `AttributeError: 'BacktestConfig' object has no attribute 'vol_target_reference'`.

- [ ] **Step 3: Add the config field**

```python
    # Reference series for vol targeting: "spy" (market-referenced, legacy) or
    # "strategy" (the book's own trailing realized vol — Barroso &
    # Santa-Clara 2015, Moreira & Muir 2017). See
    # docs/superpowers/plans/2026-07-27-strategy-own-vol-targeting.md.
    vol_target_reference: str = "spy"
```

- [ ] **Step 4: Wire production**

In `ascent/main.py`, extend the existing `apply_exposure_overlays(...)` call (around line 825) with:

```python
            vol_reference=cfg.backtest.vol_target_reference,
            close=builder.close,
```

`builder.close` is the price panel already used for the inverse-vol tilt in the same function, so no new data dependency is introduced.

- [ ] **Step 5: Wire research**

In `ascent/research/wf_framework/ascent_strategy.py`, `_apply_vol_target` currently calls `vol_target_scale`. Change it to branch on the same config value, using the close panel the method already builds:

```python
            from ascent.portfolio.exposure import (
                vol_target_scale, realized_vol_scale, strategy_return_proxy,
            )
            try:
                from ascent.config.settings import get_config as _gc
                _ref = str(getattr(_gc().backtest, "vol_target_reference", "spy")).lower()
            except Exception:
                _ref = "spy"

            if _ref == "strategy" and close_panel is not None and not close_panel.empty:
                scale = realized_vol_scale(
                    strategy_return_proxy(weights, close_panel),
                    weights.index, target_vol=target_vol,
                )
            else:
                scale = vol_target_scale(spy_close, weights.index,
                                         target_vol=target_vol)
```

Read the existing method body first (`sed -n '339,380p' ascent/research/wf_framework/ascent_strategy.py`) and adapt the local variable names — `close_panel`, `weights`, `spy_close`, and `target_vol` above are placeholders for whatever that method actually calls them. Do not rename existing locals.

- [ ] **Step 6: Verify parse and run**

```bash
for f in ascent/config/settings.py ascent/main.py ascent/research/wf_framework/ascent_strategy.py; do
  .venv/bin/python -c "import ast; ast.parse(open('$f').read())" || echo "PARSE FAIL $f"
done
.venv/bin/python -m pytest tests/test_vol_reference_config.py tests/portfolio -v
```
Expected: all pass. With the default `"spy"`, no behaviour changes anywhere.

- [ ] **Step 7: Commit**

```bash
git add ascent/config/settings.py ascent/main.py ascent/research/wf_framework/ascent_strategy.py tests/test_vol_reference_config.py
git commit -m "feat: wire vol_target_reference through production and research

Defaults to 'spy' so behaviour is unchanged until walk-forward validates
the strategy-referenced variant."
```

---

### Task 5: Walk-forward comparison and decision

**Files:**
- Create: `outputs/wf_results/wf_report_stratvol_2026-07-27.json`
- Modify: `ascent/config/settings.py` (only if the evidence supports switching)
- Modify: `CURRENT_VERIFIED_NUMBERS.md`, `CLAUDE.md`

- [ ] **Step 1: Confirm the prerequisite landed**

```bash
.venv/bin/python -m pytest tests/backtest/test_engine_cash_bucket.py -v
```
Expected: 4 passed. If not, **stop** — with the engine re-levering, every vol-scale is erased the next day and the comparison measures nothing.

- [ ] **Step 2: Run both references**

Baseline: `vol_target_reference="spy"` (reuse `wf_report_cashfix_2026-07-27.json` if already produced).
Treatment: `vol_target_reference="strategy"` → `wf_report_stratvol_2026-07-27.json`.

Change nothing else between the two runs.

- [ ] **Step 3: Compare**

Record Sharpe, CAGR, max drawdown, beta, turnover, and mean/min vol scale for both.

The papers predict a **higher Sharpe** primarily through a lower denominator — expect CAGR to fall somewhat while volatility and drawdown fall more.

- [ ] **Step 4: Apply the decision rule**

Switch the default to `"strategy"` only if:
- Sharpe improves by at least 0.05, **and**
- max drawdown improves (less negative), **and**
- mean vol scale does not collapse to the floor (0.25) — a permanently floored scale means the book is simply de-levered, not vol-*targeted*, which is a different change that should be made explicitly rather than as a side effect.

Otherwise leave the default at `"spy"` and record why. The machinery stays in the tree either way.

- [ ] **Step 5: Record the outcome and commit**

Add a `CLAUDE.md` session-log entry with the comparison table and the decision. Update `CURRENT_VERIFIED_NUMBERS.md` if the default changed.

```bash
git add -A outputs/wf_results/ CURRENT_VERIFIED_NUMBERS.md CLAUDE.md ascent/config/settings.py
git commit -m "test(portfolio): walk-forward comparison of vol-targeting reference"
```

---

## Self-Review

- **Spec coverage:** generic scaler (Task 1), causal proxy series (Task 2), reference selection with fallbacks (Task 3), config + both wiring paths (Task 4), evidence gate (Task 5). The Barroso/Santa-Clara and Moreira/Muir mechanism — scale by the factor's *own* vol — is what Tasks 1-3 implement.
- **Placeholders:** none. Task 4 Step 5 explicitly flags that local variable names must be read from the real method rather than guessed, which is an instruction, not a placeholder.
- **Type consistency:** `realized_vol_scale(returns: pd.Series, dates: pd.Index) -> pd.Series` is called with `strategy_return_proxy(...) -> pd.Series` and with `spy_close.pct_change().dropna()`. `vol_reference` is a `str` in the config, the signature, and the `meta` dict.
- **Known limitation, stated not hidden:** this is a portfolio-level control. It would have de-risked the whole book during the 2026-06/07 episode, not targeted ALGM and MRNA. The position-level answer is the stop-loss plan; these are complements, not substitutes.
