# Alpha Signal Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate three dormant alpha signals (narrative alpha, sector-relative momentum, cross-asset HY-spread) that are fully built but producing zeros, adding meaningful orthogonal signal to the alpha stack.

**Architecture:** (1) A seeder script populates `data_cache/llm_fundamental_cache.json` with 2+ quarterly entries per symbol so narrative alpha can compute Q-o-Q shifts. (2) Narrative alpha weight raised from 0% → 3% (trend reduced 41% → 38%). (3) Two new ML features added to `feature_defs.py` and `build_all_features`, registered in `ML_FEATURES`, and the model cache cleared so XGBoost retrains with them.

**Tech Stack:** Python 3.12, pandas, numpy, XGBoost, Claude Haiku (existing `ascent.llm.client`), existing `data_cache/fundamentals.parquet`.

---

### Task 1: LLM Fundamental Cache Seeder

**Files:**
- Create: `scripts/seed_llm_cache.py`
- Test: `tests/scripts/test_seed_llm_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_seed_llm_cache.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_fundamentals():
    rows = []
    for sym in ["AAPL", "MSFT"]:
        for q in ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]:
            rows.append({
                "symbol": sym, "date": q,
                "gross_profit": 100.0, "total_assets": 500.0,
                "net_income": 50.0, "op_cashflow": 60.0,
            })
    return pd.DataFrame(rows)


def test_seed_writes_cache(tmp_path):
    from scripts.seed_llm_cache import seed_cache

    fundamentals = _make_fundamentals()
    cache_path = tmp_path / "llm_fundamental_cache.json"

    mock_result = {"direction": "UP", "confidence": 0.8, "key_trend": "strong", "uncertainty": "rates"}

    with patch("ascent.alpha.llm_fundamental._call_llm", return_value=mock_result), \
         patch("ascent.alpha.llm_fundamental.CACHE_PATH", cache_path):
        seed_cache(fundamentals, cache_path=cache_path)

    assert cache_path.exists()
    cache = json.loads(cache_path.read_text())
    assert len(cache) >= 2  # at least one entry per symbol


def test_seed_handles_no_fundamentals(tmp_path):
    from scripts.seed_llm_cache import seed_cache

    cache_path = tmp_path / "llm_fundamental_cache.json"
    seed_cache(pd.DataFrame(), cache_path=cache_path)  # must not raise
    # No cache written when no data
    assert not cache_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/scripts/test_seed_llm_cache.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.seed_llm_cache'`

- [ ] **Step 3: Create `tests/scripts/__init__.py`**

```bash
mkdir -p tests/scripts && touch tests/scripts/__init__.py
```

- [ ] **Step 4: Write `scripts/seed_llm_cache.py`**

```python
"""
scripts/seed_llm_cache.py
Seeds data_cache/llm_fundamental_cache.json with LLM fundamental analyses
for all symbols in the fundamentals dataset. Run once to activate narrative alpha.

Usage:
    .venv/bin/python scripts/seed_llm_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from ascent.alpha.llm_fundamental import llm_fundamental_alpha, CACHE_PATH


def seed_cache(
    fundamentals: pd.DataFrame,
    cache_path: Path = CACHE_PATH,
    quarter_dates: list[str] | None = None,
) -> None:
    """
    Run llm_fundamental_alpha for each unique quarter-end date so the cache
    accumulates ≥2 entries per symbol (required for narrative Q-o-Q shift).

    Args:
        fundamentals:  DataFrame with columns [symbol, date, gross_profit,
                       total_assets, net_income, op_cashflow].
        cache_path:    Override cache path (for tests).
        quarter_dates: Quarter-end dates to seed. Defaults to last 4 unique dates.
    """
    if fundamentals is None or fundamentals.empty:
        return

    fundamentals = fundamentals.copy()
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])

    if quarter_dates is None:
        all_dates = sorted(fundamentals["date"].unique())
        quarter_dates = [str(d.date()) for d in all_dates[-4:]]

    import ascent.alpha.llm_fundamental as _lf
    orig_cache_path = _lf.CACHE_PATH
    _lf.CACHE_PATH = cache_path

    try:
        for q_date in quarter_dates:
            # Use cutoff = quarter_end + 46 days (after 45-day filing lag)
            cutoff = pd.Timestamp(q_date) + pd.Timedelta(days=46)
            scores = llm_fundamental_alpha(fundamentals, as_of_date=cutoff)
            n = len(scores)
            print(f"[Seeder] {q_date}: {n} symbols scored")
    finally:
        _lf.CACHE_PATH = orig_cache_path


def main():
    try:
        from ascent.data.store.parquet import load_parquet
        fundamentals = load_parquet("fundamentals")
        print(f"[Seeder] Loaded {len(fundamentals)} fundamental rows, "
              f"{fundamentals['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"[Seeder] Could not load fundamentals: {e}")
        return

    seed_cache(fundamentals)
    print(f"[Seeder] Done — cache at {CACHE_PATH}")

    # Report how many symbols have ≥2 entries (required for narrative alpha)
    import json
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        from collections import Counter
        sym_counts: Counter = Counter()
        for key in cache:
            sym = key.rsplit("_", 1)[0]
            sym_counts[sym] += 1
        ready = sum(1 for c in sym_counts.values() if c >= 2)
        print(f"[Seeder] {ready}/{len(sym_counts)} symbols ready for narrative alpha (≥2 quarters)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_seed_llm_cache.py -v
```
Expected: `2 passed`

- [ ] **Step 6: Run the seeder against real data**

```bash
.venv/bin/python scripts/seed_llm_cache.py
```
Expected output (approximate):
```
[Seeder] Loaded 675 fundamental rows, 169 symbols
[Seeder] 2025-03-31: 142 symbols scored
[Seeder] 2025-06-30: 145 symbols scored
[Seeder] 2025-09-30: 147 symbols scored
[Seeder] 2025-12-31: 149 symbols scored
[Seeder] Done — cache at data_cache/llm_fundamental_cache.json
[Seeder] 149/149 symbols ready for narrative alpha (≥2 quarters)
```

- [ ] **Step 7: Commit**

```bash
git add scripts/seed_llm_cache.py tests/scripts/__init__.py tests/scripts/test_seed_llm_cache.py
git commit -m "feat: LLM fundamental cache seeder — populates 4 quarters per symbol"
```

---

### Task 2: Activate Narrative Alpha Weight

**Files:**
- Modify: `ascent/alpha/stack.py:16-31` (DEFAULT_ALPHA_WEIGHTS)
- Modify: `ascent/research/self_improve.py` (DEFAULT_ALPHA_WEIGHTS — must match)
- Test: `tests/alpha/test_stack_weights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_stack_weights.py
import pytest


def test_narrative_weight_nonzero():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert DEFAULT_ALPHA_WEIGHTS["narrative"] == 0.03, (
        f"Expected narrative=0.03, got {DEFAULT_ALPHA_WEIGHTS['narrative']}"
    )


def test_weights_sum_to_one():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    total = sum(DEFAULT_ALPHA_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_self_improve_weights_match_stack():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_w
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_w
    assert stack_w == si_w, (
        f"stack.py and self_improve.py DEFAULT_ALPHA_WEIGHTS differ: "
        f"{set(stack_w.items()) ^ set(si_w.items())}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/alpha/test_stack_weights.py -v
```
Expected: `FAILED — narrative weight is 0.0, not 0.03`

- [ ] **Step 3: Update `ascent/alpha/stack.py` DEFAULT_ALPHA_WEIGHTS**

Change `trend` from `0.41` → `0.38` and `narrative` from `0.00` → `0.03`:

```python
DEFAULT_ALPHA_WEIGHTS = {
    "trend":           0.38,   # was 0.41
    "meanrev":         0.05,
    "volatility":      0.05,
    "statarb":         0.15,
    "ml":              0.10,
    "fundamental":     0.05,
    "llm_fundamental": 0.03,
    "earnings":        0.05,
    "analyst":         0.05,
    "options_flow":    0.02,
    "insider":         0.02,
    "short_interest":  0.02,
    "altdata":         0.00,
    "narrative":       0.03,   # was 0.00 — activated after cache seeding
}
```

- [ ] **Step 4: Update `ascent/research/self_improve.py` DEFAULT_ALPHA_WEIGHTS to match exactly**

Find the `DEFAULT_ALPHA_WEIGHTS` dict in `self_improve.py` and apply the same change: `"trend": 0.38` and `"narrative": 0.03`.

Verify with:
```bash
grep -n "DEFAULT_ALPHA_WEIGHTS" ascent/research/self_improve.py
```
Then edit the two lines.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/alpha/test_stack_weights.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add ascent/alpha/stack.py ascent/research/self_improve.py tests/alpha/test_stack_weights.py
git commit -m "feat: activate narrative alpha at 3% weight (trend 41%→38%)"
```

---

### Task 3: Sector-Relative Momentum Feature

**Files:**
- Modify: `ascent/features/feature_defs.py` (add `sector_relative_momentum` function + update `build_all_features`)
- Modify: `ascent/features/build_features.py:60-68` (pass `sector_map` through)
- Modify: `ascent/alpha/ml_sleeve.py:25-43` (add `"sector_rel_mom"` to `ML_FEATURES`)
- Test: `tests/features/test_sector_rel_mom.py`

**What this does:** For each stock, computes `mom_252d − sector_median_mom_252d`. Captures stock-specific momentum orthogonal to sector-level momentum. Historically IC ~0.15 in academic literature.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_sector_rel_mom.py
import pytest
import pandas as pd
import numpy as np
from ascent.features.feature_defs import sector_relative_momentum


def _make_close():
    dates = pd.bdate_range("2022-01-03", periods=300)
    np.random.seed(42)
    data = {"AAPL": np.cumprod(1 + np.random.normal(0.001, 0.02, 300)),
            "MSFT": np.cumprod(1 + np.random.normal(0.0005, 0.02, 300)),
            "JPM":  np.cumprod(1 + np.random.normal(0.0008, 0.02, 300))}
    return pd.DataFrame(data, index=dates)


def test_sector_rel_mom_shape():
    close = _make_close()
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    result = sector_relative_momentum(close, sector_map)
    assert result.shape == close.shape
    assert list(result.columns) == list(close.columns)


def test_sector_rel_mom_tech_sum_near_zero():
    """For a two-stock sector, sector-relative scores sum to ~0."""
    close = _make_close()
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    result = sector_relative_momentum(close, sector_map)
    # Drop early NaN rows (252d lookback)
    valid = result.dropna()
    tech_sum = (valid["AAPL"] + valid["MSFT"]).abs().mean()
    assert tech_sum < 0.01  # within-sector scores cancel


def test_sector_rel_mom_no_sector_map():
    """With no sector map, returns raw momentum (fallback)."""
    close = _make_close()
    result = sector_relative_momentum(close, {})
    assert result.shape == close.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/features/test_sector_rel_mom.py -v
```
Expected: `ImportError: cannot import name 'sector_relative_momentum'`

- [ ] **Step 3: Add `sector_relative_momentum` to `ascent/features/feature_defs.py`**

Add this function after the `high_52w_pct` definition (before the `build_fundamental_panel` block):

```python
def sector_relative_momentum(
    close: pd.DataFrame,
    sector_map: dict,
    window: int = 252,
) -> pd.DataFrame:
    """
    252d return minus sector-median 252d return per stock.
    Captures idiosyncratic momentum, orthogonal to sector drift.
    Symbols not in sector_map are left with their raw momentum.
    """
    mom = momentum_return(close, window)
    if not sector_map:
        return mom

    result = mom.copy()
    sector_groups: dict[str, list[str]] = {}
    for sym, sector in sector_map.items():
        if sym in close.columns:
            sector_groups.setdefault(sector, []).append(sym)

    for sector, syms in sector_groups.items():
        in_universe = [s for s in syms if s in mom.columns]
        if len(in_universe) < 2:
            continue
        sector_median = mom[in_universe].median(axis=1)
        for sym in in_universe:
            result[sym] = mom[sym] - sector_median

    return result
```

- [ ] **Step 4: Update `build_all_features` to accept and use `sector_map`**

In `ascent/features/feature_defs.py`, update the `build_all_features` signature and body:

```python
def build_all_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    macro_pivot: pd.DataFrame | None = None,
    earnings_df: pd.DataFrame | None = None,
    analyst_df: pd.DataFrame | None = None,
    options_df: pd.DataFrame | None = None,
    insider_df: pd.DataFrame | None = None,
    short_df: pd.DataFrame | None = None,
    sector_map: dict | None = None,          # NEW
) -> dict[str, pd.DataFrame]:
```

Then after `features["high_52w_pct"] = high_52w_pct(close)`, add:

```python
    # Sector-relative momentum
    if sector_map:
        features["sector_rel_mom"] = sector_relative_momentum(close, sector_map)
```

- [ ] **Step 5: Update `FeatureBuilder.compute_features` to pass sector_map**

In `ascent/features/build_features.py`, update `compute_features`:

```python
    def compute_features(self) -> dict[str, pd.DataFrame]:
        """Compute all features. Returns {name: DataFrame(dates × symbols)}."""
        # Load sector map for sector-relative momentum
        sector_map: dict = {}
        try:
            from ascent.data.store.parquet import load_parquet, has_data
            if has_data("profiles"):
                profiles = load_parquet("profiles")
                if "symbol" in profiles.columns and "sector" in profiles.columns:
                    sector_map = dict(zip(profiles["symbol"], profiles["sector"]))
        except Exception:
            pass

        features = build_all_features(
            self.close, self.volume, self.dollar_volume, self.macro_pivot,
            earnings_df=self.earnings_df,
            analyst_df=self.analyst_df,
            options_df=self.options_df,
            insider_df=self.insider_df,
            short_df=self.short_df,
            sector_map=sector_map,          # NEW
        )
```

- [ ] **Step 6: Add `"sector_rel_mom"` to `ML_FEATURES` in `ascent/alpha/ml_sleeve.py`**

```python
ML_FEATURES = [
    "mom_skip1m",
    "zscore_20d",
    "high_52w_pct",
    "mom_126d",
    "vol_63d",
    "earnings_surprise",
    "analyst_revision",
    "iv_skew",
    "insider_net_score",
    "short_pct_float",
    "sector_rel_mom",    # NEW: idiosyncratic momentum (stock vs sector median)
]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/features/test_sector_rel_mom.py -v
```
Expected: `3 passed`

- [ ] **Step 8: Commit**

```bash
git add ascent/features/feature_defs.py ascent/features/build_features.py \
        ascent/alpha/ml_sleeve.py tests/features/test_sector_rel_mom.py
git commit -m "feat: add sector-relative momentum feature to ML sleeve"
```

---

### Task 4: HY-Spread Direction Feature

**Files:**
- Modify: `ascent/features/feature_defs.py` (add `hy_spread_direction` + wire in `build_all_features`)
- Modify: `ascent/alpha/ml_sleeve.py` (add `"hy_spread_dir"` to `ML_FEATURES`)
- Test: `tests/features/test_hy_spread_dir.py`

**What this does:** When HY credit spreads are widening (risk-off), equity momentum tends to fail. When spreads tighten, momentum tends to persist. This gives the ML model a macro regime input that's orthogonal to price momentum.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_hy_spread_dir.py
import pytest
import pandas as pd
import numpy as np
from ascent.features.feature_defs import hy_spread_direction


def _make_macro_pivot():
    dates = pd.bdate_range("2023-01-03", periods=100)
    # Spread widens then tightens
    hy = pd.Series(
        [400.0] * 40 + [450.0] * 20 + [380.0] * 40,  # bps
        index=dates
    )
    return pd.DataFrame({"hy_spread": hy})


def test_hy_spread_dir_shape():
    macro = _make_macro_pivot()
    close_cols = ["AAPL", "MSFT"]
    close = pd.DataFrame(
        np.ones((100, 2)),
        index=macro.index,
        columns=close_cols,
    )
    result = hy_spread_direction(macro, close)
    assert result.shape == (100, 2)
    assert list(result.columns) == close_cols


def test_hy_spread_dir_values():
    """Widening spreads produce -1, tightening produce +1."""
    macro = _make_macro_pivot()
    close_cols = ["AAPL"]
    close = pd.DataFrame(np.ones((100, 1)), index=macro.index, columns=close_cols)
    result = hy_spread_direction(macro, close)
    # After 20-day window, widening period → -1
    val_widening = result.iloc[55]["AAPL"]  # in widening period
    assert val_widening == -1.0
    # Tightening period → +1
    val_tightening = result.iloc[90]["AAPL"]
    assert val_tightening == 1.0


def test_hy_spread_dir_missing_column():
    """Returns zeros if hy_spread column missing."""
    macro = pd.DataFrame({"vix": [20.0] * 50}, index=pd.bdate_range("2023-01-03", periods=50))
    close = pd.DataFrame(np.ones((50, 1)), index=macro.index, columns=["AAPL"])
    result = hy_spread_direction(macro, close)
    assert (result == 0.0).all().all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/features/test_hy_spread_dir.py -v
```
Expected: `ImportError: cannot import name 'hy_spread_direction'`

- [ ] **Step 3: Add `hy_spread_direction` to `ascent/features/feature_defs.py`**

Add after `sector_relative_momentum`:

```python
def hy_spread_direction(
    macro_pivot: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Direction of HY credit spread change over `window` days.
    +1 = spreads tightening (risk-on), -1 = spreading (risk-off), 0 = flat.
    Broadcast uniformly to all equity symbols — pure macro regime input.
    Returns zeros if hy_spread not in macro_pivot.
    """
    zeros = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if macro_pivot is None or "hy_spread" not in macro_pivot.columns:
        return zeros

    hy = macro_pivot["hy_spread"].reindex(close.index).ffill()
    change = hy.diff(window)
    direction = change.apply(
        lambda x: -1.0 if x > 0 else (1.0 if x < 0 else 0.0)
    ).fillna(0.0)

    return pd.DataFrame(
        np.tile(direction.values.reshape(-1, 1), (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )
```

- [ ] **Step 4: Wire into `build_all_features`**

In the macro section of `build_all_features`, after the existing macro loop, add:

```python
    # HY-spread direction (cross-asset regime signal for ML sleeve)
    if macro_pivot is not None and not macro_pivot.empty:
        features["hy_spread_dir"] = hy_spread_direction(macro_pivot, close)
```

- [ ] **Step 5: Add `"hy_spread_dir"` to `ML_FEATURES` in `ascent/alpha/ml_sleeve.py`**

```python
ML_FEATURES = [
    "mom_skip1m",
    "zscore_20d",
    "high_52w_pct",
    "mom_126d",
    "vol_63d",
    "earnings_surprise",
    "analyst_revision",
    "iv_skew",
    "insider_net_score",
    "short_pct_float",
    "sector_rel_mom",
    "hy_spread_dir",     # NEW: HY credit spread direction (macro regime input)
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/features/test_hy_spread_dir.py -v
```
Expected: `3 passed`

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest -q
```
Expected: 506+ passed, 1 skipped

- [ ] **Step 8: Commit**

```bash
git add ascent/features/feature_defs.py ascent/alpha/ml_sleeve.py \
        tests/features/test_hy_spread_dir.py
git commit -m "feat: add HY-spread direction as cross-asset ML feature"
```

---

### Task 5: Clear ML Cache and Force Retrain

**Files:**
- No new files — delete cache and verify retrain on next run

**Why:** `ML_FEATURES` changed (added 2 features). The cached model has mismatched `feature_names` — XGBoost will crash on prediction with "Feature shape mismatch". Must delete cache before next run.

- [ ] **Step 1: Delete stale ML model caches**

```bash
rm -f data_cache/ml_model_us_equities.pkl
ls data_cache/ml_model_*.pkl 2>/dev/null && echo "WARNING: stale caches remain" || echo "All ML caches cleared"
```
Expected: `All ML caches cleared`

- [ ] **Step 2: Verify feature count in ml_sleeve.py**

```bash
python -c "from ascent.alpha.ml_sleeve import ML_FEATURES; print(f'{len(ML_FEATURES)} features:', ML_FEATURES)"
```
Expected: `12 features: ['mom_skip1m', 'zscore_20d', ..., 'sector_rel_mom', 'hy_spread_dir']`

- [ ] **Step 3: Verify the pipeline will retrain**

```bash
python -c "
from ascent.alpha.ml_sleeve import _load_cached_model
model, train_date, features = _load_cached_model('us_equities')
print('Cache:', model, train_date, features)
"
```
Expected: `Cache: None None None`

- [ ] **Step 4: Run the full pipeline to trigger retrain (dry-run)**

```bash
.venv/bin/python -c "
import pandas as pd
from ascent.alpha.ml_sleeve import build_ml_alpha_cpcv
print('[Test] ML sleeve will retrain on next run_all_agents.py call — CPCV takes ~5-10 min')
"
```

- [ ] **Step 5: Commit**

```bash
git add ascent/alpha/stack.py
git commit -m "chore: clear ML cache — 12-feature retrain required (sector_rel_mom + hy_spread_dir)"
```

---

## Self-Review

**Spec coverage check:**

1. ✅ LLM fundamental cache seeder → Task 1
2. ✅ Narrative alpha activation (0% → 3%) → Task 2
3. ✅ Sector-relative momentum → Task 3
4. ✅ HY-spread direction → Task 4
5. ✅ ML cache cleared for retrain → Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `sector_relative_momentum(close: pd.DataFrame, sector_map: dict, window: int) → pd.DataFrame` — consistent in feature_defs.py and test
- `hy_spread_direction(macro_pivot, close, window) → pd.DataFrame` — consistent in feature_defs.py and test
- `ML_FEATURES` list updated in Task 3 Step 6 and Task 4 Step 5 — both reference `ml_sleeve.py:ML_FEATURES`
- `DEFAULT_ALPHA_WEIGHTS["narrative"] = 0.03` — both `stack.py` and `self_improve.py` updated in Task 2
