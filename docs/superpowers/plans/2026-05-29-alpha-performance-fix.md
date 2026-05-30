# Alpha Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the performance gap vs SPY by (1) preventing false stressed-regime exposure cuts when VIX doesn't confirm stress, and (2) eliminating the fundamental sleeve's statistically significant negative IC in calm_bull regimes.

**Architecture:** Two independent changes. Regime fix: post-process the HMM signal cache in `engine.py` to moderate `risk_multiplier` when VIX < 20 despite "stressed" label; mirror this in `main.py`'s SPY 200MA overlay. Fundamental fix: add per-regime alpha weight defaults in `stack.py` (fundamental=0% in calm_bull) plus a rolling IC gate that auto-zeros any sleeve whose 5-day mean IC < -0.010.

**Tech Stack:** Python, pandas, existing `logs/sleeve_ic_log.jsonl`, existing `RegimeSignal.risk_multiplier` field.

**Note:** Calm_bull allocation is already 70% US equity (changed in commit `3f400c1`). Skip that item.

---

## Task 1: VIX Confirmation for Stressed Regime Risk Multiplier

The HMM can call "stressed" on price-momentum alone. When VIX < 20, that's a false positive — markets aren't actually fearful. We moderate `risk_multiplier` back to 1.0 for those dates so the 35% exposure cut doesn't fire in a bull market.

**Files:**
- Modify: `ascent/regime/engine.py`
- Test: `tests/test_regime_features.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regime_features.py`:

```python
import pandas as pd
import numpy as np
import pytest
from ascent.regime.engine import _apply_vix_confirmation, VIX_STRESSED_CONFIRMATION


def _make_signal_cache(labels, risk_mults, dates=None):
    if dates is None:
        dates = pd.date_range("2026-01-01", periods=len(labels), freq="B")
    return pd.DataFrame({
        "label": labels,
        "risk_multiplier": risk_mults,
        "entropy": [0.3] * len(labels),
        "transition_flag": [False] * len(labels),
        "dwell_days": [1] * len(labels),
    }, index=dates)


def _make_vix(dates, values):
    return pd.Series(values, index=dates, name="Close")


def test_vix_confirmation_moderates_stressed_when_vix_low():
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    cache = _make_signal_cache(["stressed", "stressed", "calm_bull"], [0.65, 0.65, 1.0], dates)
    vix = _make_vix(dates, [15.0, 18.0, 12.0])  # all below threshold
    result = _apply_vix_confirmation(cache, vix)
    # stressed rows with low VIX → risk_multiplier set to 1.0
    assert result.loc[dates[0], "risk_multiplier"] == pytest.approx(1.0)
    assert result.loc[dates[1], "risk_multiplier"] == pytest.approx(1.0)
    # calm_bull unchanged
    assert result.loc[dates[2], "risk_multiplier"] == pytest.approx(1.0)


def test_vix_confirmation_keeps_cut_when_vix_high():
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    cache = _make_signal_cache(["stressed", "stressed"], [0.65, 0.65], dates)
    vix = _make_vix(dates, [25.0, 32.0])  # above threshold
    result = _apply_vix_confirmation(cache, vix)
    assert result.loc[dates[0], "risk_multiplier"] == pytest.approx(0.65)
    assert result.loc[dates[1], "risk_multiplier"] == pytest.approx(0.65)


def test_vix_confirmation_graceful_on_missing_vix():
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    cache = _make_signal_cache(["stressed", "stressed"], [0.65, 0.65], dates)
    # Pass None — should return cache unchanged
    result = _apply_vix_confirmation(cache, None)
    assert result.loc[dates[0], "risk_multiplier"] == pytest.approx(0.65)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_regime_features.py::test_vix_confirmation_moderates_stressed_when_vix_low -v
```

Expected: `FAILED — ImportError: cannot import name '_apply_vix_confirmation'`

- [ ] **Step 3: Add constant and function to `ascent/regime/engine.py`**

After the existing `EMERGENCY_BREAK_ZSCORE_THRESHOLD` constant (line ~53), add:

```python
# VIX below this level means the HMM's "stressed" call is unconfirmed by fear.
# When stressed but VIX < 20, we keep the label (for sleeve weighting) but
# restore risk_multiplier to 1.0 — no exposure cut.
VIX_STRESSED_CONFIRMATION = 20.0
VIX_CONFIRMATION_LABELS   = {"stressed"}   # labels that require VIX confirmation
```

After the `check_emergency_refit_triggers` function (before the `RegimeEngine` class), add:

```python
def _apply_vix_confirmation(
    signal_df: pd.DataFrame,
    vix_prices,
) -> pd.DataFrame:
    """
    Post-process the regime signal cache: for any day labeled "stressed",
    if VIX < VIX_STRESSED_CONFIRMATION, restore risk_multiplier to 1.0.

    The HMM can fire "stressed" on price momentum alone during a relief rally.
    VIX < 20 means options markets don't agree — no exposure cut applied.
    Label is kept as "stressed" so sleeve weighting still tilts defensive.

    Returns a modified copy of signal_df.
    """
    if vix_prices is None or signal_df.empty:
        return signal_df

    df = signal_df.copy()

    # Align VIX to signal dates — forward-fill gaps (VIX has same calendar as SPY)
    if hasattr(vix_prices, "columns"):
        vix_close = vix_prices.iloc[:, 0] if "Close" not in vix_prices.columns else vix_prices["Close"]
    else:
        vix_close = vix_prices

    vix_aligned = vix_close.reindex(df.index, method="ffill")

    moderated = 0
    for dt in df.index:
        if str(df.at[dt, "label"]) not in VIX_CONFIRMATION_LABELS:
            continue
        vix_val = vix_aligned.get(dt)
        if vix_val is not None and not np.isnan(float(vix_val)) and float(vix_val) < VIX_STRESSED_CONFIRMATION:
            df.at[dt, "risk_multiplier"] = 1.0
            moderated += 1

    if moderated > 0:
        log.info(
            f"regime.engine: VIX confirmation moderated {moderated} stressed dates "
            f"(VIX < {VIX_STRESSED_CONFIRMATION}) → risk_multiplier restored to 1.0"
        )
    return df
```

- [ ] **Step 4: Call `_apply_vix_confirmation` in `RegimeEngine.fit()`**

In `ascent/regime/engine.py`, in `RegimeEngine.fit()`, immediately after:
```python
self._signal_cache = self._decision_engine.process_to_frame(prob_df)
```
add:
```python
self._signal_cache = _apply_vix_confirmation(self._signal_cache, vix_prices)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_regime_features.py -k "vix_confirmation" -v
```

Expected: 3 PASSED

- [ ] **Step 6: Verify AST**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/regime/engine.py').read()); print('AST OK')"
```

- [ ] **Step 7: Commit**

```bash
git add ascent/regime/engine.py tests/test_regime_features.py
git commit -m "feat: VIX confirmation gate on stressed regime risk_multiplier

When HMM calls stressed but VIX < 20, the equity market isn't actually
fearful — relief rallies produce false positives. Restore risk_multiplier
to 1.0 for unconfirmed stressed dates to prevent the 35% exposure cut
firing in momentum bull markets.

Sleeve weighting still tilts defensive (label kept); only the gross
exposure multiplier is moderated."
```

---

## Task 2: VIX Confirmation for SPY 200MA Overlay in Backtest

The backtest-mode SPY 200MA overlay in `main.py` cuts exposure 30% whenever SPY < 200MA. This fires ~22% of backtest dates regardless of whether fear is actually elevated. Add the same VIX confirmation.

**Files:**
- Modify: `ascent/main.py` (lines 646–660)

- [ ] **Step 1: Locate the overlay block**

The block starts at line ~646 in `ascent/main.py`:
```python
try:
    spy_close         = benchmark_df.set_index("date")["close"].sort_index()
    ...
    below_dates       = spy_below_aligned[spy_below_aligned].index
    if len(below_dates) > 0:
        target_weights.loc[below_dates] = target_weights.loc[below_dates] * 0.70
```

`vix_series` is in scope at this point (built earlier in the same function).

- [ ] **Step 2: Replace the overlay block**

Replace the existing block with:

```python
try:
    spy_close         = benchmark_df.set_index("date")["close"].sort_index()
    spy_close         = spy_close[~spy_close.index.duplicated(keep="last")]
    spy_ma200         = spy_close.rolling(200, min_periods=150).mean()
    spy_below_ma      = spy_close < spy_ma200

    # VIX confirmation: only cut exposure when both SPY < 200MA AND VIX > 20.
    # SPY-alone fires during relief rallies (April 2026 pattern) where markets
    # recover faster than the MA catches up. VIX < 20 = options market disagrees.
    VIX_MA_THRESHOLD = 20.0
    if vix_series is not None and not (hasattr(vix_series, "empty") and vix_series.empty):
        if hasattr(vix_series, "columns"):
            vix_close = vix_series.iloc[:, 0] if "Close" not in vix_series.columns else vix_series["Close"]
        else:
            vix_close = vix_series
        vix_aligned  = vix_close.reindex(spy_below_ma.index, method="ffill").fillna(0.0)
        vix_confirmed = vix_aligned > VIX_MA_THRESHOLD
        spy_below_confirmed = spy_below_ma & vix_confirmed
    else:
        spy_below_confirmed = spy_below_ma  # no VIX data — fall back to MA-only

    spy_below_aligned = spy_below_confirmed.reindex(target_weights.index, method="ffill").fillna(False)
    below_dates       = spy_below_aligned[spy_below_aligned].index
    if len(below_dates) > 0:
        target_weights.loc[below_dates] = target_weights.loc[below_dates] * 0.70
        pct = len(below_dates) / max(len(target_weights), 1) * 100
        print(f"[Portfolio] SPY 200d MA filter: {len(below_dates)} dates below MA+VIX>20 ({pct:.1f}%) → 30% exposure cut")
    else:
        print("[Portfolio] SPY 200d MA filter: no confirmed stress dates — no cuts applied")
except Exception as _e:
    print(f"[Portfolio] SPY 200d MA filter skipped: {_e}")
```

- [ ] **Step 3: Verify AST**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/main.py').read()); print('AST OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short -x
```

Expected: all pass (same count as before ± new tests)

- [ ] **Step 5: Commit**

```bash
git add ascent/main.py
git commit -m "feat: require VIX>20 confirmation for SPY 200MA exposure cut

SPY below 200MA alone fires during relief rallies — April 2026 showed
22% of backtest dates getting a 30% cut, many in bull markets. Now
requires VIX > 20 to confirm genuine market stress before cutting.
Falls back to MA-only if VIX data is unavailable."
```

---

## Task 3: Fundamental Sleeve — Regime-Conditional Weights + Rolling IC Gate

**Two sub-changes:**
- **3a**: Zero fundamental weight in calm_bull (redirect to trend). Fundamental factors (GP, accruals, asset_growth) are value/quality signals that systematically underperform in momentum bull markets. IC=-0.0078, t=-4.63 confirms this.
- **3b**: Auto-zero any sleeve whose 5-day rolling mean IC < -0.010 (hard floor). This catches future IC decay across all sleeves.

**Files:**
- Modify: `ascent/alpha/stack.py`
- Test: `tests/alpha/test_fundamental_alpha.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/alpha/test_fundamental_alpha.py`:

```python
import json
import tempfile
from pathlib import Path
import pytest
from ascent.alpha.stack import (
    _load_active_alpha_weights,
    _get_gated_weights,
    IC_GATE_THRESHOLD,
)


def test_calm_bull_zeroes_fundamental():
    weights = _load_active_alpha_weights(regime="calm_bull")
    assert weights["fundamental"] == 0.0, "fundamental must be 0 in calm_bull"
    assert weights["trend"] > 0.38, "trend should be higher in calm_bull to absorb fundamental weight"


def test_stressed_keeps_fundamental():
    weights = _load_active_alpha_weights(regime="stressed")
    assert weights["fundamental"] > 0.0, "fundamental should be active in stressed regime"


def test_ic_gate_zeroes_negative_sleeve(tmp_path):
    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    # Write 5 entries all showing fundamental IC < threshold
    for i in range(5):
        entry = {
            "date": f"2026-05-{20+i:02d}",
            "sleeves": {
                "fundamental": {"mean_ic": -0.015, "t_stat": -3.5, "n": 900},
                "trend":       {"mean_ic":  0.016, "t_stat":  3.8, "n": 900},
            }
        }
        ic_log.write_text(ic_log.read_text() if ic_log.exists() else "" + json.dumps(entry) + "\n")

    base = {"fundamental": 0.05, "trend": 0.38, "meanrev": 0.05}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))
    assert result["fundamental"] == 0.0, "fundamental IC < threshold should be gated to 0"
    assert result["trend"] > 0.38, "zeroed weight redistributed to trend"


def test_ic_gate_preserves_positive_sleeve(tmp_path):
    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    entry = {
        "date": "2026-05-29",
        "sleeves": {"fundamental": {"mean_ic": 0.01, "t_stat": 2.1, "n": 900}}
    }
    ic_log.write_text(json.dumps(entry) + "\n")
    base = {"fundamental": 0.05, "trend": 0.38}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))
    assert result["fundamental"] == pytest.approx(0.05)
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/alpha/test_fundamental_alpha.py::test_calm_bull_zeroes_fundamental -v
```

Expected: `FAILED — ImportError: cannot import name '_get_gated_weights'`

- [ ] **Step 3: Add per-regime defaults and IC gate to `ascent/alpha/stack.py`**

After the `DEFAULT_ALPHA_WEIGHTS` dict (around line 31), add:

```python
IC_GATE_THRESHOLD = -0.010  # zero out any sleeve with 5-day rolling mean IC below this

# Per-regime overrides for calm_bull (value/quality factors underperform in momentum bull).
# In stressed/crisis, fundamental and earnings factors tend to work.
# These are defaults — active_alpha_config.json overrides them when self-improve runs.
DEFAULT_ALPHA_WEIGHTS_BY_REGIME = {
    "calm_bull": {
        **DEFAULT_ALPHA_WEIGHTS,
        "fundamental": 0.00,
        "trend":       0.43,  # absorbs the 0.05 freed from fundamental
    },
    "stressed": {
        **DEFAULT_ALPHA_WEIGHTS,
        "fundamental": 0.08,
        "trend":       0.33,  # reduce trend slightly to fund extra fundamental weight
    },
    "crisis": {
        **DEFAULT_ALPHA_WEIGHTS,
        "fundamental": 0.08,
        "trend":       0.30,
        "volatility":  0.10,
    },
}
```

Modify `_load_active_alpha_weights` to check per-regime defaults before the flat default:

```python
def _load_active_alpha_weights(regime: str = None) -> dict:
    import json as _json
    from pathlib import Path as _Path

    config_path = _Path("data_cache/active_alpha_config.json")
    if config_path.exists():
        try:
            config = _json.loads(config_path.read_text())
            if regime:
                regime_weights = config.get("by_regime", {}).get(str(regime).lower())
                if regime_weights and isinstance(regime_weights, dict):
                    return {k: float(v) for k, v in regime_weights.items()}
            global_weights = config.get("global")
            if global_weights and isinstance(global_weights, dict):
                return {k: float(v) for k, v in global_weights.items()}
        except Exception as exc:
            log.warning("_load_active_alpha_weights: failed to load config (%s) — using defaults", exc)

    # Check per-regime built-in defaults before flat default
    if regime:
        regime_key = str(regime).lower()
        if regime_key in DEFAULT_ALPHA_WEIGHTS_BY_REGIME:
            return DEFAULT_ALPHA_WEIGHTS_BY_REGIME[regime_key].copy()

    return DEFAULT_ALPHA_WEIGHTS.copy()
```

After `_load_active_alpha_weights`, add:

```python
def _get_gated_weights(
    alpha_weights: dict,
    ic_log_path: str = "logs/sleeve_ic_log.jsonl",
    window: int = 5,
) -> dict:
    """
    Read the last `window` unique-date entries from sleeve_ic_log.jsonl.
    Zero out any sleeve whose rolling mean IC < IC_GATE_THRESHOLD.
    Redistribute freed weight to trend (the most reliable sleeve by IC t-stat).
    Returns a copy of alpha_weights with gated sleeves zeroed.
    """
    import json as _json
    from collections import defaultdict
    from pathlib import Path as _Path

    log_path = _Path(ic_log_path)
    if not log_path.exists():
        return alpha_weights  # no history yet — pass through

    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    seen, recent = set(), []
    for line in reversed(lines):
        try:
            entry = _json.loads(line)
            d = entry.get("date", "")
            if d and d not in seen:
                seen.add(d)
                recent.append(entry)
            if len(recent) >= window:
                break
        except Exception:
            continue

    if not recent:
        return alpha_weights

    sleeve_ics: dict = defaultdict(list)
    for entry in recent:
        for sleeve, stats in entry.get("sleeves", {}).items():
            ic = stats.get("mean_ic")
            if ic is not None:
                sleeve_ics[sleeve].append(float(ic))

    gated = {}
    for sleeve, ics in sleeve_ics.items():
        if len(ics) >= window and sum(ics) / len(ics) < IC_GATE_THRESHOLD:
            gated[sleeve] = sum(ics) / len(ics)

    if not gated:
        return alpha_weights

    result = dict(alpha_weights)
    freed = sum(result.get(s, 0.0) for s in gated)
    for sleeve in gated:
        log.warning(
            "[Stack] IC gate: zeroing %s (rolling mean_ic=%.4f < %.3f)",
            sleeve, gated[sleeve], IC_GATE_THRESHOLD,
        )
        result[sleeve] = 0.0
    if freed > 0 and "trend" in result:
        result["trend"] = round(result["trend"] + freed, 4)

    return result
```

- [ ] **Step 4: Call `_get_gated_weights` in `build_alpha_stack`**

In `ascent/alpha/stack.py`, in the `build_alpha_stack` function, find where `alpha_weights` is loaded:

```python
alpha_weights = _load_active_alpha_weights(regime=regime)
```

Add immediately after:

```python
alpha_weights = _get_gated_weights(alpha_weights)
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/alpha/test_fundamental_alpha.py -v
```

Expected: all 4 new tests PASS

- [ ] **Step 6: Verify AST**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/alpha/stack.py').read()); print('AST OK')"
```

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: 629+ passed, 1 skipped

- [ ] **Step 8: Commit**

```bash
git add ascent/alpha/stack.py tests/alpha/test_fundamental_alpha.py
git commit -m "feat: regime-conditional fundamental sleeve + rolling IC gate

Fundamental (GP, accruals, asset_growth) has IC=-0.0078, t=-4.63 in
calm_bull — value factors systematically underperform momentum bull.
Now: 0% weight in calm_bull (redirected to trend), 8% in stressed/crisis
where quality factors earn their keep.

Rolling IC gate (5-day window, threshold=-0.010) auto-zeros any sleeve
with persistently negative IC and redistributes to trend. Applies to
all sleeves as a safety net."
```

---

## Self-Review

**Spec coverage:**
- ✅ Regime confirmation logic → Tasks 1 + 2
- ✅ Calm_bull allocation → Already done (noted, skipped)
- ✅ Fundamental sleeve fix → Task 3

**Placeholder scan:** None found.

**Type consistency:**
- `_apply_vix_confirmation(signal_df, vix_prices)` — used in Task 1 tests and in `engine.py` body, signatures match
- `_get_gated_weights(alpha_weights, ic_log_path, window)` — used in Task 3 tests and in `build_alpha_stack`, signatures match
- `IC_GATE_THRESHOLD` — defined in Task 3 step 3, imported in tests ✅
- `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` — defined in Task 3 step 3, not exported directly but used via `_load_active_alpha_weights` ✅
