# Ascent Portfolio Strategy Adapter — Design Spec

**Date:** 2026-06-03
**Goal:** Wire the Ascent quant alpha stack into the Walk-Forward OOS framework as a portfolio-weight strategy, enabling honest IS-parameter optimization and OOS evaluation across all 5 key parameters with smart caching (~30 min runtime).

---

## Context

The WF framework at `ascent/research/wf_framework/` was built with a single-asset signal interface (`BaseStrategy.generate_signals → pd.Series of {-1,0,1}`). Ascent's alpha stack outputs cross-sectional portfolio weights across 15–20 symbols — a fundamentally different shape. This spec extends the framework with a parallel portfolio path while keeping single-asset strategies fully backward compatible.

---

## What Gets Built

### 1. `ascent/research/wf_framework/portfolio_strategy.py` (new)

Abstract base class for multi-asset portfolio strategies.

```python
class PortfolioBaseStrategy(ABC):
    def __init__(self, **params): ...          # same as BaseStrategy
    param_grid: dict[str, list]               # abstract property
    generate_signals(data: pd.DataFrame)      # returns pd.DataFrame (dates × symbols)
                     → pd.DataFrame           # weights, each row sums to ≤ 1.0
```

**Interface contract:**
- `data` is long-format multi-symbol OHLCV with columns `[date, symbol, open, high, low, close, volume]`
- Returns `pd.DataFrame` indexed by date, columns = symbol tickers, values = portfolio weights
- Each row must sum to ≤ 1.0 (cash allowed), no weight > `max_weight` param
- Strictly causal: weight at date t uses only data at dates ≤ t

---

### 2. `ascent/research/wf_framework/ascent_strategy.py` (new)

Concrete implementation wrapping the Ascent pipeline.

```python
class AscentPortfolioStrategy(PortfolioBaseStrategy):
    param_grid = {
        "top_n":          [10, 15, 20],
        "max_weight":     [0.08, 0.10, 0.12],
        "trend_weight":   [0.30, 0.38, 0.50],
        "statarb_weight": [0.10, 0.15, 0.20],
        "mom_window":     [63, 126, 252],
    }
```

**`generate_signals(data)` pipeline:**
1. Pivot long-format data → wide format for FeatureBuilder
2. `FeatureBuilder(wide_prices, macro_df=None)` → `features` dict
3. Override `DEFAULT_ALPHA_WEIGHTS`: inject `trend_weight`, `statarb_weight`; renormalize remaining sleeves proportionally to sum to 1.0
4. `build_alpha_stack(features, agent_id="us_equities")` → `alpha_scores` DataFrame
5. For each rebalance date in the IS/OOS window: `sector_constrained_weighted(alpha_row, n=top_n, max_weight=max_weight)` → weights row
6. Forward-fill weights between rebalance dates
7. Return `pd.DataFrame(dates × symbols)`

**Sleeve weight injection detail:**
When `trend_weight` and `statarb_weight` are set by the grid, the remaining 12 sleeves are scaled proportionally:
```
remaining_budget = 1.0 - trend_weight - statarb_weight
scale_factor = remaining_budget / sum(DEFAULT_WEIGHTS for other sleeves)
other_sleeve_weight = DEFAULT_WEIGHT * scale_factor
```
This ensures weights always sum to 1.0 regardless of grid values.

**Rebalance cadence:** every 10 business days (matches live `rebalance_freq_days=10`), not daily — consistent with live behavior and the existing walk-forward runner.

**Sector map:** loaded from `profiles.parquet` if available; empty dict fallback (sector caps skipped with warning, per existing behavior).

---

### 3. Smart Caching (inside `AscentPortfolioStrategy`)

Two instance-level caches, keyed on IS data identity:

```
_feature_cache:  (is_start, is_end, n_symbols, mom_window) → features dict
_alpha_cache:    (is_start, is_end, n_symbols, mom_window,
                  trend_weight, statarb_weight)             → alpha_scores DataFrame
```

Cache key uses `(data.index.min(), data.index.max(), len(data.columns), ...)` — cheap to compute, uniquely identifies the IS slice without hashing large DataFrames.

**Result:** 243 grid combos collapse per fold to:
- 3 FeatureBuilder calls (one per `mom_window`)
- 27 `build_alpha_stack` calls (one per `mom_window × trend/statarb combo`)
- 243 `sector_constrained_weighted` calls (fast, ~0.05s each)

Cache is cleared between folds via `strategy.clear_cache()` called by the engine before each new IS window.

---

### 4. `execution.py` — Portfolio path extension (backward compatible)

`ExecutionModel.compute_returns` gains type dispatch:

```python
def compute_returns(self, data, signals):
    if isinstance(signals, pd.DataFrame):
        return self._portfolio_returns(data, signals)
    else:
        return self._single_asset_returns(data, signals)   # existing path unchanged
```

**`_portfolio_returns(data, weight_df)`:**
- `data`: long-format multi-symbol OHLCV
- `weight_df`: dates × symbols weight DataFrame from `generate_signals`
- Daily portfolio return: `(w_{t-1} × r_t).sum(axis=1)` where `r_t` = close-to-close per symbol
- Execution delay: weights shifted by 1 day (consistent with single-asset path)
- Transaction costs: `commission_pct × |w_t - w_{t-1}|.sum()` on each rebalance bar (turnover × commission)
- Slippage: `atr_multiplier × mean_ATR / mean_price × |w_t - w_{t-1}|.sum()` per rebalance bar
- Borrow cost: not applicable (long-only portfolio)
- Returns `pd.Series` of daily net portfolio returns — same output type as single-asset path

**Symbol alignment:** `weight_df.columns` and `data['symbol'].unique()` are aligned via `reindex(fill_value=0)` before computation. Symbols in weights but missing from data receive zero return.

---

### 5. `engine.py` — Portfolio detection (minimal change)

One new method `_is_portfolio_strategy()` checks `isinstance(self.strategy_cls, type) and issubclass(self.strategy_cls, PortfolioBaseStrategy)`.

When portfolio strategy detected:
- `data` passed to `run()` must be long-format multi-symbol (validated at start)
- `is_data = data[data['date'].isin(is_dates)]` instead of `data.loc[is_dates]`
- Strategy `generate_signals` called on full context window (IS start → OOS end), OOS weights harvested by date filter
- `strategy.clear_cache()` called before each fold's IS optimization

---

## Data Contract for `run()`

When using `AscentPortfolioStrategy`, call:
```python
prices = load_parquet("prices_live")   # long-format: date, symbol, open, high, low, close, volume
engine.run(data=prices, benchmark=spy_returns)
```

The engine validates that `data` contains `['date', 'symbol', 'close']` columns when a portfolio strategy is detected.

---

## Parameter Grid — Full Spec

| Parameter | Values | What it tests |
|---|---|---|
| `top_n` | [10, 15, 20] | Portfolio concentration |
| `max_weight` | [0.08, 0.10, 0.12] | Position size cap |
| `trend_weight` | [0.30, 0.38, 0.50] | Momentum sleeve dominance |
| `statarb_weight` | [0.10, 0.15, 0.20] | Sector-residual mean-rev contribution |
| `mom_window` | [63, 126, 252] | Feature lookback horizon |

**Total combos:** 3 × 3 × 3 × 3 × 3 = **243**
**Constraint:** `trend_weight + statarb_weight ≤ 0.75` (leaves ≥25% for other 12 sleeves)
**Estimated runtime:** ~30 min (22 folds × ~82s per fold with caching)

---

## IS Optimization vs OOS Evaluation — Regime Treatment

| Phase | Regime | Reason |
|---|---|---|
| IS grid search | **Off** — fixed sleeve weights from grid | Cleaner comparison across combos; regime would add a hidden variable |
| OOS evaluation | **On** — fold-local regime signal fitted on IS data | Realistic; matches live behavior where regime adapts sleeve weights |

The regime engine for OOS is fit identically to `walk_forward_runner.py`: `RegimeEngine(config=cfg.regime.to_engine_dict())` trained on IS slice only, `run_model_selection=False` for speed.

---

## Files Changed / Created

| File | Action | Change |
|---|---|---|
| `ascent/research/wf_framework/portfolio_strategy.py` | Create | `PortfolioBaseStrategy` ABC |
| `ascent/research/wf_framework/ascent_strategy.py` | Create | `AscentPortfolioStrategy` + caching |
| `ascent/research/wf_framework/execution.py` | Modify | Add `_portfolio_returns`, type-dispatch in `compute_returns` |
| `ascent/research/wf_framework/engine.py` | Modify | Portfolio detection, long-format data slicing, `clear_cache()` calls |
| `ascent/research/wf_framework/__init__.py` | Modify | Export `PortfolioBaseStrategy`, `AscentPortfolioStrategy` |
| `tests/test_wf_framework/test_portfolio_strategy.py` | Create | Unit tests for `PortfolioBaseStrategy` and `AscentPortfolioStrategy` |
| `tests/test_wf_framework/test_portfolio_execution.py` | Create | Unit tests for `_portfolio_returns` |
| `tests/test_wf_framework/test_ascent_engine.py` | Create | Integration test: full WF run with `AscentPortfolioStrategy` on real data |

---

## Test Plan

**`test_portfolio_strategy.py`:**
- `AscentPortfolioStrategy` is subclass of `PortfolioBaseStrategy`
- `generate_signals` returns `pd.DataFrame` with symbol columns
- Weights sum to ≤ 1.0 on every row
- No look-ahead: partial slice produces same weights on overlapping dates
- Caching: second call with same data is faster (cached)
- `clear_cache()` forces recomputation

**`test_portfolio_execution.py`:**
- Portfolio returns match manual `(w × r).sum()` calculation
- Transaction costs deducted on rebalance bars only
- Zero friction path: returns match raw weighted close-to-close
- Single-asset path unchanged (regression test)

**`test_ascent_engine.py`:**
- Engine runs end-to-end on `prices_live` parquet (or synthetic multi-symbol data if parquet absent)
- `equity_curve.iloc[0] == 1.0`
- No duplicate OOS dates
- `report["wfe"]` is finite
- IS optimizer receives only IS-window data (verified via mock)

---

## Integrity Constraints Preserved

1. **No look-ahead**: `generate_signals` called on `[is_start → oos_end]` context; weights at OOS dates use only history up to that date via `FeatureBuilder`'s causal rolling operations.
2. **No simulated data under live cache names**: adapter reads `prices_live` only; if absent, raises `FileNotFoundError` with clear message.
3. **Max-weight hard cap**: `sector_constrained_weighted` enforces `max_weight` via `_water_fill_cap()` with post-condition check — same as live pipeline.
4. **Sector constraint fallback**: `< 80%` coverage → skip caps + warn — same as live pipeline.
5. **Regime IS-only**: fold-local regime engine trained on IS slice only, identical to `walk_forward_runner.py` fix.
