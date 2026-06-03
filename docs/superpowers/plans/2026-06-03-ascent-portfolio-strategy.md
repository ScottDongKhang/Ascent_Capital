# Ascent Portfolio Strategy Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Ascent quant alpha stack into the Walk-Forward OOS framework as a portfolio-weight strategy, enabling full 243-combo grid search across top_n, max_weight, trend_weight, statarb_weight, and mom_window with smart caching (~30 min runtime on 22 folds).

**Architecture:** Four tasks: (1) `PortfolioBaseStrategy` ABC + portfolio execution path in `ExecutionModel`; (2) `AscentPortfolioStrategy` with class-level smart caching; (3) engine extension for long-format portfolio data; (4) end-to-end runner script. The existing single-asset path is untouched throughout.

**Tech Stack:** Python 3.12, pandas, numpy, existing Ascent pipeline (FeatureBuilder, build_alpha_stack, sector_constrained_weighted).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ascent/research/wf_framework/portfolio_strategy.py` | Create | `PortfolioBaseStrategy` ABC |
| `ascent/research/wf_framework/execution.py` | Modify | Add `_portfolio_returns` + type-dispatch in `compute_returns` |
| `ascent/research/wf_framework/ascent_strategy.py` | Create | `AscentPortfolioStrategy` with class-level caching |
| `ascent/research/wf_framework/engine.py` | Modify | Portfolio branch: long-format data slicing, `clear_cache()` calls |
| `ascent/research/wf_framework/__init__.py` | Modify | Export new classes |
| `tests/test_wf_framework/test_portfolio_execution.py` | Create | Portfolio execution path unit tests |
| `tests/test_wf_framework/test_ascent_strategy.py` | Create | AscentPortfolioStrategy unit tests |
| `tests/test_wf_framework/test_ascent_engine.py` | Create | Integration test on synthetic multi-symbol data |
| `scripts/run_ascent_wf.py` | Create | End-to-end runner on real prices_live data |

---

## Task 1: PortfolioBaseStrategy ABC + portfolio execution path

**Files:**
- Create: `ascent/research/wf_framework/portfolio_strategy.py`
- Modify: `ascent/research/wf_framework/execution.py`
- Create: `tests/test_wf_framework/test_portfolio_execution.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_portfolio_execution.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.portfolio_strategy import PortfolioBaseStrategy
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig


def make_multi_symbol_ohlcv(n_days=100, symbols=("A", "B", "C"), seed=42):
    """Long-format OHLCV fixture for 3 symbols × n_days."""
    np.random.seed(seed)
    rows = []
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    for sym in symbols:
        close = pd.Series(100 * (1 + np.random.randn(n_days) * 0.01).cumprod(), index=idx)
        for dt, c in close.items():
            rows.append({
                "date": dt, "symbol": sym,
                "open": c * 0.999, "high": c * 1.005,
                "low": c * 0.995, "close": c, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_ohlcv():
    return make_multi_symbol_ohlcv()


@pytest.fixture
def equal_weights(multi_ohlcv):
    """Equal-weight DataFrame across 3 symbols."""
    dates   = sorted(multi_ohlcv["date"].unique())
    symbols = sorted(multi_ohlcv["symbol"].unique())
    w = 1.0 / len(symbols)
    return pd.DataFrame(w, index=pd.DatetimeIndex(dates), columns=symbols)


def test_cannot_instantiate_portfolio_base():
    with pytest.raises(TypeError):
        PortfolioBaseStrategy()


def test_portfolio_returns_series(multi_ohlcv, equal_weights):
    em = ExecutionModel(ExecutionConfig(commission_pct=0.0,
                                        slippage_model="fixed_pct", fixed_pct=0.0))
    returns = em.compute_returns(multi_ohlcv, equal_weights)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(equal_weights)


def test_portfolio_zero_friction_matches_manual(multi_ohlcv, equal_weights):
    em = ExecutionModel(ExecutionConfig(commission_pct=0.0,
                                        slippage_model="fixed_pct", fixed_pct=0.0))
    returns = em.compute_returns(multi_ohlcv, equal_weights)
    # Manual: equal-weight close-to-close (delayed by 1)
    close_wide = multi_ohlcv.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    sym_ret = close_wide.pct_change().fillna(0)
    delayed_w = equal_weights.shift(1).fillna(0).reindex(
        index=sym_ret.index, columns=sym_ret.columns, fill_value=0
    )
    expected = (delayed_w * sym_ret).sum(axis=1).rename("net_return")
    pd.testing.assert_series_equal(returns, expected, atol=1e-6, check_names=False)


def test_portfolio_commission_reduces_returns(multi_ohlcv, equal_weights):
    cfg_no  = ExecutionConfig(commission_pct=0.0, slippage_model="fixed_pct", fixed_pct=0.0)
    cfg_yes = ExecutionConfig(commission_pct=0.001, slippage_model="fixed_pct", fixed_pct=0.0)
    r_no  = ExecutionModel(cfg_no).compute_returns(multi_ohlcv, equal_weights)
    r_yes = ExecutionModel(cfg_yes).compute_returns(multi_ohlcv, equal_weights)
    assert r_yes.sum() < r_no.sum()


def test_portfolio_atr_slippage_reduces_returns(multi_ohlcv, equal_weights):
    cfg_no  = ExecutionConfig(slippage_model="atr", atr_multiplier=0.0, commission_pct=0.0)
    cfg_yes = ExecutionConfig(slippage_model="atr", atr_multiplier=0.1, commission_pct=0.0)
    r_no  = ExecutionModel(cfg_no).compute_returns(multi_ohlcv, equal_weights)
    r_yes = ExecutionModel(cfg_yes).compute_returns(multi_ohlcv, equal_weights)
    assert r_yes.sum() < r_no.sum()


def test_single_asset_path_unchanged(multi_ohlcv):
    """Existing single-asset path must not be affected."""
    ohlcv_single = multi_ohlcv[multi_ohlcv["symbol"] == "A"].copy()
    # Rebuild as single-symbol OHLCV (no 'symbol' column, date as index)
    ohlcv_single = ohlcv_single.set_index("date").drop(columns=["symbol"])
    signals = pd.Series(1, index=ohlcv_single.index)
    em = ExecutionModel(ExecutionConfig())
    returns = em.compute_returns(ohlcv_single, signals)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(ohlcv_single)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_wf_framework/test_portfolio_execution.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'PortfolioBaseStrategy'`

- [ ] **Step 3: Create `portfolio_strategy.py`**

```python
# ascent/research/wf_framework/portfolio_strategy.py
"""
Portfolio Strategy Interface
============================
PortfolioBaseStrategy is the ABC for multi-asset walk-forward strategies.

Implementing a portfolio strategy
----------------------------------
1. Subclass PortfolioBaseStrategy.
2. Define `param_grid` property — dict of param_name → list of candidate values.
3. Implement `generate_signals(data)` — accepts long-format OHLCV DataFrame,
   returns pd.DataFrame (dates × symbols) of portfolio weights.
4. Optionally override `clear_cache()` if the strategy caches intermediate results.

Boundary defense: `generate_signals` must be strictly causal — weight at date t
may only use data at dates <= t. The engine enforces this by passing only the
[IS_start → OOS_end] context window.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class PortfolioBaseStrategy(ABC):
    """Abstract base for multi-asset portfolio strategies."""

    def __init__(self, **params):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    @property
    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return {param_name: [candidate_values]} for grid search."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : long-format OHLCV DataFrame with columns
               [date, symbol, open, high, low, close, volume].
               Sorted ascending by date. No future data allowed.

        Returns
        -------
        pd.DataFrame indexed by date, columns = symbol tickers.
        Values = portfolio weights in [0, max_weight].
        Each row sums to <= 1.0 (cash allowed).
        """

    def clear_cache(self) -> None:
        """Clear any internal caches. Engine calls this between folds."""
        pass
```

- [ ] **Step 4: Refactor `execution.py` — add portfolio path**

Read the current `execution.py` (already done above), then replace with:

```python
# ascent/research/wf_framework/execution.py
"""
Execution Friction Model
========================
Converts a signal (single-asset Series or portfolio weight DataFrame) into a
net-of-friction daily return series.

Single-asset path  : signals is pd.Series of {-1, 0, +1}
Portfolio path     : signals is pd.DataFrame (dates × symbols) of weights

Three friction layers (both paths):
  1. Slippage  — ATR-based or fixed-pct, applied on position-change bars.
  2. Commission — flat pct of notional traded (turnover).
  3. Borrow    — overnight financing for shorts (single-asset path only).

Boundary defense: 1-day execution delay — signal/weight at date t is active
from date t+1 onward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    slippage_model:     Literal["atr", "fixed_pct"] = "atr"
    atr_multiplier:     float = 0.10
    fixed_pct:          float = 0.0005
    atr_window:         int   = 14
    commission_pct:     float = 0.0005
    borrow_rate_annual: float = 0.0
    execution_delay:    int   = 1


class ExecutionModel:
    def __init__(self, config: ExecutionConfig | None = None):
        self.cfg = config or ExecutionConfig()

    # ------------------------------------------------------------------
    # Public API — type-dispatches on signals type
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        data:    pd.DataFrame,
        signals: Union[pd.Series, pd.DataFrame],
    ) -> pd.Series:
        if isinstance(signals, pd.DataFrame):
            return self._portfolio_returns(data, signals)
        return self._single_asset_returns(data, signals)

    # ------------------------------------------------------------------
    # Single-asset path (unchanged from original)
    # ------------------------------------------------------------------

    def _single_asset_returns(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> pd.Series:
        cfg   = self.cfg
        delay = cfg.execution_delay

        position  = signals.shift(delay).fillna(0)
        raw_ret   = data["close"].pct_change().fillna(0)
        gross_ret = position * raw_ret

        pos_change = position.diff().fillna(position)
        is_trade   = pos_change.abs() > 0

        if cfg.slippage_model == "atr":
            atr        = self._atr_single(data, cfg.atr_window)
            exec_price = data["open"].shift(-delay).clip(lower=1e-8)
            slip_cost  = cfg.atr_multiplier * atr / exec_price
        else:
            slip_cost = pd.Series(cfg.fixed_pct, index=data.index)

        slip_cost = slip_cost.fillna(0)
        slip_deduction       = is_trade * pos_change.abs() * slip_cost
        commission_deduction = is_trade * cfg.commission_pct
        daily_borrow_rate    = cfg.borrow_rate_annual / 252
        borrow_deduction     = (position < 0).astype(float) * daily_borrow_rate

        net_ret = gross_ret - slip_deduction - commission_deduction - borrow_deduction
        return net_ret.rename("net_return")

    # ------------------------------------------------------------------
    # Portfolio path (new)
    # ------------------------------------------------------------------

    def _portfolio_returns(
        self, data: pd.DataFrame, weight_df: pd.DataFrame
    ) -> pd.Series:
        """
        data      : long-format OHLCV with 'date' and 'symbol' columns.
        weight_df : pd.DataFrame (dates × symbols) from PortfolioBaseStrategy.
        """
        cfg   = self.cfg
        delay = cfg.execution_delay

        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        close_w = data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        open_w  = data.pivot_table(index="date", columns="symbol", values="open",  aggfunc="last").sort_index()
        high_w  = data.pivot_table(index="date", columns="symbol", values="high",  aggfunc="last").sort_index()
        low_w   = data.pivot_table(index="date", columns="symbol", values="low",   aggfunc="last").sort_index()

        # Align weights and prices to common dates and symbols
        common_dates = weight_df.index.intersection(close_w.index)
        all_syms     = weight_df.columns.intersection(close_w.columns)

        weights = weight_df.reindex(index=common_dates, columns=all_syms, fill_value=0.0)
        close   = close_w.reindex(index=common_dates,  columns=all_syms)
        open_   = open_w.reindex(index=common_dates,   columns=all_syms)
        high_   = high_w.reindex(index=common_dates,   columns=all_syms)
        low_    = low_w.reindex(index=common_dates,    columns=all_syms)

        # Daily returns per symbol
        sym_ret = close.pct_change().fillna(0)

        # Portfolio return: delayed weights × symbol returns
        delayed_w = weights.shift(delay).fillna(0)
        gross_ret = (delayed_w * sym_ret).sum(axis=1)

        # Turnover: sum of abs weight changes per bar
        w_diff = weights.diff()
        w_diff.iloc[0] = weights.iloc[0].abs()
        turnover = w_diff.abs().sum(axis=1)

        # Commission on turnover
        commission_deduction = turnover * cfg.commission_pct

        # Slippage on turnover
        if cfg.slippage_model == "atr":
            prev_c  = close.shift(1)
            tr_vals = np.maximum(
                np.maximum(
                    (high_ - low_).fillna(0).values,
                    (high_ - prev_c).abs().fillna(0).values,
                ),
                (low_ - prev_c).abs().fillna(0).values,
            )
            tr  = pd.DataFrame(tr_vals, index=common_dates, columns=all_syms)
            atr = tr.ewm(alpha=1 / cfg.atr_window, adjust=False).mean()
            slip_per_sym   = cfg.atr_multiplier * atr / open_.clip(lower=1e-8)
            slip_deduction = (w_diff.abs() * slip_per_sym.fillna(0)).sum(axis=1)
        else:
            slip_deduction = turnover * cfg.fixed_pct

        net_ret = gross_ret - commission_deduction - slip_deduction
        return net_ret.rename("net_return")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atr_single(data: pd.DataFrame, window: int) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False).mean()
```

- [ ] **Step 5: Run tests — all 6 must pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_portfolio_execution.py -v
```
Expected: 6 PASSED.

- [ ] **Step 6: Run full suite — no regressions**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/ -q --tb=no
```
Expected: 41 passed, 0 failed (35 prior + 6 new).

- [ ] **Step 7: Commit**

```bash
git add ascent/research/wf_framework/portfolio_strategy.py \
        ascent/research/wf_framework/execution.py \
        tests/test_wf_framework/test_portfolio_execution.py
git commit -m "feat: wf-framework portfolio path — PortfolioBaseStrategy ABC + portfolio execution"
```

---

## Task 2: AscentPortfolioStrategy with smart caching

**Files:**
- Create: `ascent/research/wf_framework/ascent_strategy.py`
- Create: `tests/test_wf_framework/test_ascent_strategy.py`

The key design: caches are **class-level** so they're shared across instances created during the same IS optimization loop. The engine calls `AscentPortfolioStrategy.clear_cache()` (classmethod) before each fold to prevent stale entries.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_ascent_strategy.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.portfolio_strategy import PortfolioBaseStrategy
from ascent.research.wf_framework.ascent_strategy import AscentPortfolioStrategy


def make_multi_ohlcv(n_days=300, n_symbols=20, seed=7):
    """Synthetic long-format OHLCV — does not require prices_live.parquet."""
    np.random.seed(seed)
    rows = []
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    for sym in symbols:
        close = pd.Series(100 * (1 + np.random.randn(n_days) * 0.012).cumprod(), index=idx)
        for dt, c in close.items():
            rows.append({
                "date": dt, "symbol": sym,
                "open": c * 0.999, "high": c * 1.006,
                "low": c * 0.994, "close": c, "volume": 500_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ohlcv():
    return make_multi_ohlcv()


def test_is_portfolio_base_strategy():
    assert issubclass(AscentPortfolioStrategy, PortfolioBaseStrategy)


def test_param_grid_keys():
    s = AscentPortfolioStrategy()
    for key in ["top_n", "max_weight", "trend_weight", "statarb_weight", "mom_window"]:
        assert key in s.param_grid, f"Missing param_grid key: {key}"


def test_param_grid_values():
    s = AscentPortfolioStrategy()
    assert s.param_grid["top_n"]          == [10, 15, 20]
    assert s.param_grid["max_weight"]     == [0.08, 0.10, 0.12]
    assert s.param_grid["trend_weight"]   == [0.30, 0.38, 0.50]
    assert s.param_grid["statarb_weight"] == [0.10, 0.15, 0.20]
    assert s.param_grid["mom_window"]     == [63, 126, 252]


def test_generate_signals_returns_dataframe(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    s = AscentPortfolioStrategy(top_n=5, max_weight=0.20, mom_window=63)
    result = s.generate_signals(ohlcv)
    assert isinstance(result, pd.DataFrame), "generate_signals must return pd.DataFrame"


def test_weights_sum_leq_one(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    s = AscentPortfolioStrategy(top_n=5, max_weight=0.20, mom_window=63)
    result = s.generate_signals(ohlcv)
    row_sums = result.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all(), f"Row sums exceed 1.0: {row_sums.max():.4f}"


def test_no_weight_exceeds_max(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    max_w = 0.15
    s = AscentPortfolioStrategy(top_n=5, max_weight=max_w, mom_window=63)
    result = s.generate_signals(ohlcv)
    assert (result.values <= max_w + 1e-6).all(), "A weight exceeds max_weight"


def test_make_alpha_weights_sums_to_one():
    s = AscentPortfolioStrategy(trend_weight=0.50, statarb_weight=0.20)
    w = s._make_alpha_weights()
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-9, f"Alpha weights sum {total:.6f} != 1.0"


def test_make_alpha_weights_trend_statarb_set():
    s = AscentPortfolioStrategy(trend_weight=0.50, statarb_weight=0.20)
    w = s._make_alpha_weights()
    assert abs(w["trend"]   - 0.50) < 1e-9
    assert abs(w["statarb"] - 0.20) < 1e-9


def test_caching_shared_across_instances(ohlcv):
    """Two instances with same mom_window must share the feature cache."""
    AscentPortfolioStrategy.clear_cache()
    s1 = AscentPortfolioStrategy(top_n=5, mom_window=63)
    s2 = AscentPortfolioStrategy(top_n=10, mom_window=63)  # different top_n, same mom_window
    s1.generate_signals(ohlcv)
    n_after_s1 = len(AscentPortfolioStrategy._feature_cache)
    s2.generate_signals(ohlcv)
    n_after_s2 = len(AscentPortfolioStrategy._feature_cache)
    assert n_after_s1 == n_after_s2, "Feature cache grew on second call with same mom_window"


def test_clear_cache_resets():
    AscentPortfolioStrategy.clear_cache()
    assert len(AscentPortfolioStrategy._feature_cache) == 0
    assert len(AscentPortfolioStrategy._alpha_cache)   == 0


def test_no_lookahead(ohlcv):
    """Signals on first 150 bars match when computed on 150 vs 300 bars."""
    AscentPortfolioStrategy.clear_cache()
    s_full    = AscentPortfolioStrategy(top_n=5, mom_window=63)
    s_partial = AscentPortfolioStrategy(top_n=5, mom_window=63)
    cutoff = sorted(ohlcv["date"].unique())[149]
    partial_data = ohlcv[ohlcv["date"] <= cutoff]

    AscentPortfolioStrategy.clear_cache()
    result_partial = s_partial.generate_signals(partial_data)

    AscentPortfolioStrategy.clear_cache()
    result_full = s_full.generate_signals(ohlcv)

    common_dates = result_partial.index
    common_syms  = result_partial.columns.intersection(result_full.columns)
    pd.testing.assert_frame_equal(
        result_full.loc[common_dates, common_syms].fillna(0),
        result_partial[common_syms].fillna(0),
        atol=1e-6,
        check_names=False,
    )
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_ascent_strategy.py -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'AscentPortfolioStrategy'`

- [ ] **Step 3: Implement `ascent_strategy.py`**

```python
# ascent/research/wf_framework/ascent_strategy.py
"""
Ascent Portfolio Strategy
==========================
Wraps the Ascent quant alpha pipeline as a walk-forward portfolio strategy.

Smart caching
-------------
Feature computation (FeatureBuilder) and alpha computation (build_alpha_stack)
are cached at the CLASS level so all instances in a single optimizer grid search
share results. The engine calls `AscentPortfolioStrategy.clear_cache()` before
each fold's IS optimization to prevent stale entries from bleeding across folds.

Per fold, 243 combos collapse to:
  - 3 FeatureBuilder calls  (one per mom_window value)
  - 27 build_alpha_stack calls  (one per mom_window × trend_weight × statarb_weight combo)
  - 243 sector_constrained_weighted calls  (fast, ~0.05s each)

Boundary defense
----------------
generate_signals is strictly causal: it uses only data[date <= call_date] via
FeatureBuilder's rolling operations. No forward returns or future prices are used.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .portfolio_strategy import PortfolioBaseStrategy

from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS, build_alpha_stack
from ascent.features.build_features import FeatureBuilder
from ascent.portfolio.optimizer import sector_constrained_weighted


class AscentPortfolioStrategy(PortfolioBaseStrategy):
    """Ascent quant alpha stack as a PortfolioBaseStrategy."""

    # Class-level caches: shared across all instances in the same optimize() loop
    _feature_cache: dict = {}
    _alpha_cache:   dict = {}

    def __init__(
        self,
        top_n:          int   = 15,
        max_weight:     float = 0.10,
        trend_weight:   float = 0.38,
        statarb_weight: float = 0.15,
        mom_window:     int   = 252,
        sector_map:     Optional[dict] = None,
        rebalance_freq: int   = 10,
    ):
        super().__init__(
            top_n=top_n,
            max_weight=max_weight,
            trend_weight=trend_weight,
            statarb_weight=statarb_weight,
            mom_window=mom_window,
        )
        self.sector_map     = sector_map or self._load_sector_map()
        self.rebalance_freq = rebalance_freq

    # ------------------------------------------------------------------
    # PortfolioBaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "top_n":          [10, 15, 20],
            "max_weight":     [0.08, 0.10, 0.12],
            "trend_weight":   [0.30, 0.38, 0.50],
            "statarb_weight": [0.10, 0.15, 0.20],
            "mom_window":     [63, 126, 252],
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear class-level caches. Engine calls this before each fold."""
        cls._feature_cache.clear()
        cls._alpha_cache.clear()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : long-format OHLCV with 'date' and 'symbol' columns.
        Returns pd.DataFrame (dates × symbols) of portfolio weights, ffilled.
        """
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        all_dates = sorted(data["date"].unique())
        if not all_dates:
            return pd.DataFrame()

        # Cache key: uniquely identifies this data window
        data_key = (all_dates[0], all_dates[-1], data["symbol"].nunique())

        # --- Step 1: Features (cached per mom_window) ---
        feat_key = (*data_key, self.mom_window)
        if feat_key not in AscentPortfolioStrategy._feature_cache:
            # Trim to last mom_window bars for feature computation
            if self.mom_window < len(all_dates):
                cutoff     = all_dates[-self.mom_window]
                data_slice = data[data["date"] >= cutoff]
            else:
                data_slice = data
            builder = FeatureBuilder(data_slice, macro_df=None)
            AscentPortfolioStrategy._feature_cache[feat_key] = builder.compute_features()
        features = AscentPortfolioStrategy._feature_cache[feat_key]

        # --- Step 2: Alpha stack (cached per mom_window + sleeve blend) ---
        alpha_key = (*data_key, self.mom_window, round(self.trend_weight, 4),
                     round(self.statarb_weight, 4))
        if alpha_key not in AscentPortfolioStrategy._alpha_cache:
            alpha_weights = self._make_alpha_weights()
            AscentPortfolioStrategy._alpha_cache[alpha_key] = build_alpha_stack(
                features, alpha_weights=alpha_weights, agent_id="us_equities"
            )
        alpha_scores = AscentPortfolioStrategy._alpha_cache[alpha_key]

        # --- Step 3: Portfolio construction at rebalance dates ---
        alpha_dates = alpha_scores.index
        if len(alpha_dates) == 0:
            return pd.DataFrame()

        rebal_dates = [alpha_dates[i] for i in range(0, len(alpha_dates), self.rebalance_freq)]
        alpha_at_rebal = alpha_scores.loc[rebal_dates]

        weights_at_rebal = sector_constrained_weighted(
            alpha_at_rebal,
            n=self.top_n,
            max_weight=self.max_weight,
            sector_map=self.sector_map,
            regime_signal=None,   # regime off during IS optimization
        )

        # Forward-fill to every trading day in the alpha index
        weights_ffilled = (
            weights_at_rebal
            .reindex(alpha_dates)
            .ffill()
            .fillna(0.0)
        )
        return weights_ffilled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_alpha_weights(self) -> dict:
        """Return sleeve weight dict with trend/statarb overrides, others scaled."""
        base = dict(DEFAULT_ALPHA_WEIGHTS)
        base["trend"]   = self.trend_weight
        base["statarb"] = self.statarb_weight

        other_keys       = [k for k in base if k not in ("trend", "statarb")]
        other_default_sum = sum(DEFAULT_ALPHA_WEIGHTS[k] for k in other_keys)
        remaining        = 1.0 - self.trend_weight - self.statarb_weight

        if other_default_sum > 1e-9 and remaining > 0:
            scale = remaining / other_default_sum
            for k in other_keys:
                base[k] = DEFAULT_ALPHA_WEIGHTS[k] * scale

        return base

    @staticmethod
    def _load_sector_map() -> dict:
        """Load sector map from profiles.parquet if available; empty dict otherwise."""
        try:
            from ascent.data.store.parquet import load_parquet, has_data
            if has_data("profiles"):
                profiles = load_parquet("profiles")
                return dict(zip(profiles["symbol"], profiles["sector"]))
        except Exception:
            pass
        return {}
```

- [ ] **Step 4: Run tests — all 11 must pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_ascent_strategy.py -v
```
Expected: 11 PASSED. (The `test_no_lookahead` test may be slow — ~10s — that's normal.)

- [ ] **Step 5: Run full suite — no regressions**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/ -q --tb=no
```
Expected: 52 passed (41 prior + 11 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add ascent/research/wf_framework/ascent_strategy.py \
        tests/test_wf_framework/test_ascent_strategy.py
git commit -m "feat: wf-framework AscentPortfolioStrategy — 243-combo grid with class-level caching"
```

---

## Task 3: Engine extension + exports + integration test

**Files:**
- Modify: `ascent/research/wf_framework/engine.py`
- Modify: `ascent/research/wf_framework/__init__.py`
- Create: `tests/test_wf_framework/test_ascent_engine.py`

The engine needs a portfolio branch that:
1. Extracts unique dates from `data["date"]` (not `data.index`)
2. Slices IS/OOS data by date filter (not `.loc[]`)
3. Calls `strategy_cls.clear_cache()` before each fold's IS optimization
4. Harvests OOS weights by `.loc[oos_dates]` from the signals DataFrame

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_wf_framework/test_ascent_engine.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from ascent.research.wf_framework.engine import WalkForwardEngine
from ascent.research.wf_framework.windows import WindowGenerator
from ascent.research.wf_framework.execution import ExecutionConfig
from ascent.research.wf_framework.ascent_strategy import AscentPortfolioStrategy
from ascent.research.wf_framework.optimizer import ParameterOptimizer


def make_multi_ohlcv(n_days=600, n_symbols=30, seed=99):
    np.random.seed(seed)
    rows = []
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    for sym in symbols:
        close = pd.Series(100 * (1 + np.random.randn(n_days) * 0.012).cumprod(), index=idx)
        for dt, c in close.items():
            rows.append({
                "date": dt, "symbol": sym,
                "open": c * 0.999, "high": c * 1.006,
                "low": c * 0.994, "close": c, "volume": 500_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def multi_ohlcv():
    return make_multi_ohlcv()


@pytest.fixture(scope="module")
def benchmark(multi_ohlcv):
    close_wide = multi_ohlcv.pivot_table(
        index="date", columns="symbol", values="close"
    ).mean(axis=1)
    return close_wide.pct_change().fillna(0).rename("benchmark")


def make_engine():
    return WalkForwardEngine(
        strategy_cls     = AscentPortfolioStrategy,
        window_generator = WindowGenerator(is_days=252, oos_days=63,
                                           purge_days=21, embargo_days=5),
        exec_config      = ExecutionConfig(commission_pct=0.0005,
                                           slippage_model="fixed_pct", fixed_pct=0.0005),
        constraint_fn    = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
    )


def test_engine_runs_with_portfolio_strategy(multi_ohlcv, benchmark):
    engine = make_engine()
    equity_curve, report = engine.run(multi_ohlcv, benchmark)
    assert isinstance(equity_curve, pd.Series)
    assert isinstance(report, dict)
    assert "sharpe" in report and "wfe" in report


def test_equity_curve_starts_at_one(multi_ohlcv):
    engine = make_engine()
    equity_curve, _ = engine.run(multi_ohlcv)
    assert abs(equity_curve.iloc[0] - 1.0) < 1e-6


def test_no_duplicate_oos_dates(multi_ohlcv):
    engine = make_engine()
    equity_curve, _ = engine.run(multi_ohlcv)
    assert not equity_curve.index.duplicated().any()


def test_wfe_finite(multi_ohlcv):
    engine = make_engine()
    _, report = engine.run(multi_ohlcv)
    assert np.isfinite(report["wfe"])


def test_cache_cleared_between_folds(multi_ohlcv):
    """Verify clear_cache() is called before each fold's IS optimization."""
    clear_calls = []
    original_clear = AscentPortfolioStrategy.clear_cache

    @classmethod
    def recording_clear(cls):
        clear_calls.append(1)
        original_clear.__func__(cls)

    engine = make_engine()
    with patch.object(AscentPortfolioStrategy, "clear_cache", recording_clear):
        engine.run(multi_ohlcv)

    n_folds = len(engine.last_windows_)
    assert len(clear_calls) >= n_folds, \
        f"clear_cache called {len(clear_calls)} times, expected >= {n_folds}"


def test_optimizer_receives_is_data_only(multi_ohlcv):
    """IS optimizer must never see OOS dates."""
    max_dates_seen = []
    original_optimize = ParameterOptimizer.optimize

    def recording_optimize(self, is_data):
        if "date" in is_data.columns:
            max_dates_seen.append(pd.to_datetime(is_data["date"]).max())
        return original_optimize(self, is_data)

    engine = make_engine()
    with patch.object(ParameterOptimizer, "optimize", recording_optimize):
        engine.run(multi_ohlcv)

    for i, (w, max_dt) in enumerate(zip(engine.last_windows_, max_dates_seen)):
        assert max_dt < w.oos_start, \
            f"Fold {i}: optimizer saw data up to {max_dt.date()} >= OOS start {w.oos_start.date()}"


def test_single_asset_engine_still_works():
    """Existing single-asset MACrossStrategy must still work — no regression."""
    from ascent.research.wf_framework.strategy import MACrossStrategy
    np.random.seed(1)
    n = 400
    idx = pd.bdate_range("2019-01-02", periods=n)
    close = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=idx)
    data = pd.DataFrame({
        "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": 1_000_000,
    })
    engine = WalkForwardEngine(
        strategy_cls     = MACrossStrategy,
        window_generator = WindowGenerator(is_days=252, oos_days=63,
                                           purge_days=21, embargo_days=5),
        constraint_fn    = lambda p: p["fast"] < p["slow"],
        verbose          = False,
    )
    equity_curve, report = engine.run(data)
    assert isinstance(equity_curve, pd.Series)
    assert np.isfinite(report["sharpe"])
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_ascent_engine.py::test_engine_runs_with_portfolio_strategy -v 2>&1 | head -15
```
Expected: `AttributeError` or `TypeError` — engine doesn't yet have portfolio branch.

- [ ] **Step 3: Update `engine.py` with portfolio branch**

Replace the full contents of `ascent/research/wf_framework/engine.py`:

```python
# ascent/research/wf_framework/engine.py
"""
Walk-Forward Engine
===================
Top-level orchestrator. Supports both:
  - Single-asset strategies (BaseStrategy → pd.Series signals)
  - Portfolio strategies  (PortfolioBaseStrategy → pd.DataFrame weights)

Call `engine.run(data, benchmark)` to get the stitched OOS equity curve
and full performance report.

Boundary defenses
-----------------
1. IS optimizer receives IS-slice data only — cannot see OOS dates.
   Single-asset: `data.loc[is_dates]`
   Portfolio:    `data[data["date"].isin(is_dates)]`

2. Strategy's generate_signals called on [IS_start → OOS_end] for warmup;
   only OOS signals/weights harvested for evaluation.

3. For portfolio strategies: `strategy_cls.clear_cache()` called before each
   fold's IS optimization to prevent cross-fold cache contamination.

4. `last_windows_` stored for post-run inspection and testing.
"""
from __future__ import annotations
from typing import Callable, Type
import numpy as np
import pandas as pd

from .windows import WindowGenerator, SplitWindow
from .strategy import BaseStrategy
from .execution import ExecutionModel, ExecutionConfig
from .optimizer import ParameterOptimizer
from .metrics import PerformanceAnalyzer, FoldResult


def _is_portfolio(strategy_cls) -> bool:
    """Return True if strategy_cls is a PortfolioBaseStrategy subclass."""
    try:
        from .portfolio_strategy import PortfolioBaseStrategy
        return issubclass(strategy_cls, PortfolioBaseStrategy)
    except ImportError:
        return False


class WalkForwardEngine:

    def __init__(
        self,
        strategy_cls:     Type[BaseStrategy],
        window_generator: WindowGenerator | None = None,
        exec_config:      ExecutionConfig | None = None,
        objective:        str = "sharpe",
        constraint_fn:    Callable[[dict], bool] | None = None,
        rf_annual:        float = 0.0,
        verbose:          bool = True,
    ):
        self.strategy_cls     = strategy_cls
        self.wg               = window_generator or WindowGenerator()
        self.exec_model       = ExecutionModel(exec_config or ExecutionConfig())
        self.optimizer        = ParameterOptimizer(
            strategy_cls, self.exec_model, objective, constraint_fn, rf_annual
        )
        self.analyser         = PerformanceAnalyzer(rf_annual)
        self.verbose          = verbose
        self.last_windows_: list[SplitWindow] = []
        self._portfolio_mode  = _is_portfolio(strategy_cls)

    def run(
        self,
        data:      pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> tuple[pd.Series, dict]:
        if self._portfolio_mode:
            return self._run_portfolio(data, benchmark)
        return self._run_single_asset(data, benchmark)

    # ------------------------------------------------------------------
    # Single-asset path (original, unchanged)
    # ------------------------------------------------------------------

    def _run_single_asset(
        self, data: pd.DataFrame, benchmark: pd.Series | None
    ) -> tuple[pd.Series, dict]:
        dates   = data.index.drop_duplicates().sort_values()
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            self._print_header(len(windows))

        fold_results:      list[FoldResult] = []
        oos_return_chunks: list[pd.Series]  = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            is_data        = data.loc[is_dates]
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            full_context_dates = dates[(dates >= w.is_start) & (dates <= w.oos_end)]
            full_context_data  = data.loc[full_context_dates]
            strategy           = self.strategy_cls(**best_params)
            all_signals        = strategy.generate_signals(full_context_data)
            oos_signals        = all_signals.loc[oos_dates]
            oos_data           = data.loc[oos_dates]
            oos_returns        = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                self._print_fold(w, best_params, is_sharpe,
                                 self.analyser.sharpe(oos_returns))

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        return self._finalise(oos_return_chunks, fold_results, benchmark)

    # ------------------------------------------------------------------
    # Portfolio path (new)
    # ------------------------------------------------------------------

    def _run_portfolio(
        self, data: pd.DataFrame, benchmark: pd.Series | None
    ) -> tuple[pd.Series, dict]:
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        # Extract unique trading dates from the 'date' column
        dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            self._print_header(len(windows))

        fold_results:      list[FoldResult] = []
        oos_return_chunks: list[pd.Series]  = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            # BOUNDARY DEFENSE: clear cache and give optimizer IS slice only
            self.strategy_cls.clear_cache()
            is_dates_set          = set(is_dates)
            is_data               = data[data["date"].isin(is_dates_set)]
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            # Full context: IS_start → OOS_end for warmup
            full_context_dates = set(dates[(dates >= w.is_start) & (dates <= w.oos_end)])
            full_context_data  = data[data["date"].isin(full_context_dates)]

            strategy    = self.strategy_cls(**best_params)
            all_signals = strategy.generate_signals(full_context_data)   # DataFrame

            oos_dates_set = set(oos_dates)
            oos_signals   = all_signals.loc[all_signals.index.isin(oos_dates_set)]
            oos_data      = data[data["date"].isin(oos_dates_set)]
            oos_returns   = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                self._print_fold(w, best_params, is_sharpe,
                                 self.analyser.sharpe(oos_returns))

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        return self._finalise(oos_return_chunks, fold_results, benchmark)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        oos_return_chunks: list[pd.Series],
        fold_results:      list[FoldResult],
        benchmark:         pd.Series | None,
    ) -> tuple[pd.Series, dict]:
        if not oos_return_chunks:
            raise RuntimeError("All folds skipped — no OOS returns produced.")

        stitched = pd.concat(oos_return_chunks).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]

        equity_curve = (1 + stitched).cumprod()
        equity_curve = equity_curve / equity_curve.iloc[0]

        bm_aligned = None
        if benchmark is not None:
            bm_aligned = benchmark.reindex(stitched.index).fillna(0)

        report = self.analyser.full_report(stitched, bm_aligned, fold_results)

        if self.verbose:
            print()
            self.analyser.print_report(report)

        return equity_curve, report

    def _print_header(self, n_folds: int) -> None:
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD ENGINE — {n_folds} folds  "
              f"({'portfolio' if self._portfolio_mode else 'single-asset'})")
        print(f"  IS: {self.wg.is_days}d | OOS: {self.wg.oos_days}d | "
              f"Purge: {self.wg.purge_days}d | Embargo: {self.wg.embargo_days}d")
        print(f"{'='*60}")

    def _print_fold(
        self, w: SplitWindow, params: dict, is_sharpe: float, oos_sharpe: float
    ) -> None:
        print(
            f"  Fold {w.fold_id}: "
            f"IS [{w.is_start.date()} → {w.purge_start.date()}) "
            f"OOS [{w.oos_start.date()} → {w.oos_end.date()}] "
            f"IS_Sh={is_sharpe:.2f} OOS_Sh={oos_sharpe:.2f} "
            f"params={params}"
        )
```

- [ ] **Step 4: Update `__init__.py`**

```python
# ascent/research/wf_framework/__init__.py
from .windows import WindowGenerator, SplitWindow

try:
    from .strategy import BaseStrategy
except ImportError:
    pass

try:
    from .portfolio_strategy import PortfolioBaseStrategy
except ImportError:
    pass

try:
    from .execution import ExecutionModel, ExecutionConfig
except ImportError:
    pass

try:
    from .optimizer import ParameterOptimizer
except ImportError:
    pass

try:
    from .metrics import PerformanceAnalyzer
except ImportError:
    pass

try:
    from .engine import WalkForwardEngine
except ImportError:
    pass

try:
    from .ascent_strategy import AscentPortfolioStrategy
except ImportError:
    pass

__all__ = [
    "WindowGenerator", "SplitWindow",
    "BaseStrategy",
    "PortfolioBaseStrategy",
    "ExecutionModel", "ExecutionConfig",
    "ParameterOptimizer",
    "PerformanceAnalyzer",
    "WalkForwardEngine",
    "AscentPortfolioStrategy",
]
```

- [ ] **Step 5: Run integration tests**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_ascent_engine.py -v
```
Expected: 7 PASSED. (Tests will be slow — 3-5 min — that's expected for a full WF run on synthetic data.)

- [ ] **Step 6: Run full suite — no regressions**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/ -q --tb=no
```
Expected: 59 passed (52 prior + 7 new), 0 failed.

- [ ] **Step 7: Commit**

```bash
git add ascent/research/wf_framework/engine.py \
        ascent/research/wf_framework/__init__.py \
        tests/test_wf_framework/test_ascent_engine.py
git commit -m "feat: wf-framework engine portfolio branch — long-format data, cache clearing"
```

---

## Task 4: End-to-end runner on real prices_live data

**Files:**
- Create: `scripts/run_ascent_wf.py`

This is the runner that performs the full 243-combo × 22-fold evaluation on your real data and saves results to `outputs/wf_results/`.

- [ ] **Step 1: Create the runner script**

```python
#!/usr/bin/env python
# scripts/run_ascent_wf.py
"""
Ascent Walk-Forward OOS Evaluation
====================================
Full 243-combo grid search on real prices_live data.
Runtime: ~30 minutes.

Usage:
    .venv/bin/python scripts/run_ascent_wf.py

Output:
    outputs/wf_results/wf_report_YYYY-MM-DD.json   — performance report
    outputs/wf_results/wf_equity_YYYY-MM-DD.csv    — equity curve
    outputs/wf_results/wf_best_params_YYYY-MM-DD.json — best params per fold
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from ascent.data.store.parquet import load_parquet, has_data
from ascent.research.wf_framework import (
    WalkForwardEngine,
    WindowGenerator,
    ExecutionConfig,
    AscentPortfolioStrategy,
)


def main():
    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    if not has_data("prices_live"):
        print("ERROR: prices_live.parquet not found. Run main pipeline first.")
        sys.exit(1)

    print("Loading prices_live.parquet...")
    prices = load_parquet("prices_live")
    prices["date"] = pd.to_datetime(prices["date"])
    n_symbols = prices["symbol"].nunique()
    date_range = f"{prices['date'].min().date()} → {prices['date'].max().date()}"
    print(f"  {n_symbols} symbols, {date_range}")

    # SPY benchmark
    spy = prices[prices["symbol"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date").set_index("date")["close"].pct_change().fillna(0)
    spy.name = "SPY"

    # ------------------------------------------------------------------ #
    # 2. Build engine                                                      #
    # ------------------------------------------------------------------ #
    engine = WalkForwardEngine(
        strategy_cls = AscentPortfolioStrategy,
        window_generator = WindowGenerator(
            is_days     = 252,
            oos_days    = 63,
            purge_days  = 21,
            embargo_days= 5,
            window_type = "rolling",
            step_days   = 63,
        ),
        exec_config = ExecutionConfig(
            slippage_model     = "atr",
            atr_multiplier     = 0.10,
            commission_pct     = 0.0005,   # 5 bps round-trip
            borrow_rate_annual = 0.0,
        ),
        # Constraint: trend + statarb <= 75% (leaves >= 25% for 12 other sleeves)
        constraint_fn = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
        rf_annual     = 0.04,              # 4% risk-free for Sharpe
        verbose       = True,
    )

    n_combos = 1
    for v in AscentPortfolioStrategy().param_grid.values():
        n_combos *= len(v)
    print(f"\nGrid: {n_combos} combos × {engine.wg.is_days}d IS / {engine.wg.oos_days}d OOS")
    print("Estimated runtime: ~30 minutes\n")

    # ------------------------------------------------------------------ #
    # 3. Run                                                               #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    equity_curve, report = engine.run(prices, spy)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    # ------------------------------------------------------------------ #
    # 4. Save results                                                      #
    # ------------------------------------------------------------------ #
    out_dir = Path("outputs/wf_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    # Report JSON
    report_serializable = {
        k: (float(v) if isinstance(v, (float, np.floating)) else
            int(v)   if isinstance(v, (int, np.integer))   else v)
        for k, v in report.items()
    }
    report_path = out_dir / f"wf_report_{today}.json"
    with open(report_path, "w") as f:
        json.dump(report_serializable, f, indent=2)
    print(f"\nReport saved → {report_path}")

    # Equity curve CSV
    equity_path = out_dir / f"wf_equity_{today}.csv"
    equity_curve.to_csv(equity_path, header=["equity"])
    print(f"Equity curve saved → {equity_path}")

    # Per-fold best params
    fold_params = []
    for w in engine.last_windows_:
        fold_params.append({
            "fold_id":   w.fold_id,
            "oos_start": str(w.oos_start.date()),
            "oos_end":   str(w.oos_end.date()),
        })
    params_path = out_dir / f"wf_best_params_{today}.json"
    with open(params_path, "w") as f:
        json.dump(fold_params, f, indent=2)
    print(f"Fold metadata saved → {params_path}")

    # ------------------------------------------------------------------ #
    # 5. Print key numbers                                                 #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 50)
    print(f"  CAGR:       {report['cagr']*100:+.2f}%")
    print(f"  Sharpe:     {report['sharpe']:.3f}")
    print(f"  Max DD:     {report['max_drawdown']*100:.2f}%")
    print(f"  WFE:        {report['wfe']:.3f}  "
          f"({'acceptable' if report['wfe'] >= 0.5 else 'OVERFIT WARNING'})")
    if 'alpha' in report:
        print(f"  Alpha/SPY:  {report['alpha']*100:+.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses correctly (no syntax errors)**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/run_ascent_wf.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Run a quick smoke test with a small subset (optional but recommended)**

```bash
.venv/bin/python - << 'EOF'
import sys
sys.path.insert(0, ".")
import pandas as pd
from ascent.data.store.parquet import load_parquet, has_data
from ascent.research.wf_framework import (
    WalkForwardEngine, WindowGenerator, ExecutionConfig, AscentPortfolioStrategy
)

prices = load_parquet("prices_live")
prices["date"] = pd.to_datetime(prices["date"])

# Use only 2019–2021 for a fast smoke test (~3 folds)
prices_sub = prices[prices["date"] <= "2021-12-31"]

engine = WalkForwardEngine(
    strategy_cls     = AscentPortfolioStrategy,
    window_generator = WindowGenerator(is_days=252, oos_days=63,
                                        purge_days=21, embargo_days=5),
    exec_config      = ExecutionConfig(slippage_model="fixed_pct", fixed_pct=0.0005,
                                       commission_pct=0.0005),
    constraint_fn    = lambda p: p["trend_weight"] + p["statarb_weight"] <= 0.75,
    rf_annual        = 0.04,
    verbose          = True,
)

equity_curve, report = engine.run(prices_sub)
print(f"\nSmoke test: {len(engine.last_windows_)} folds, Sharpe={report['sharpe']:.3f}, WFE={report['wfe']:.3f}")
EOF
```
Expected: 3-5 folds printed, report shows finite Sharpe and WFE. Runtime ~5 min.

- [ ] **Step 4: Run the full evaluation**

```bash
.venv/bin/python scripts/run_ascent_wf.py
```
Expected: 22 folds, ~30 min, results saved to `outputs/wf_results/`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_ascent_wf.py outputs/wf_results/
git commit -m "feat: run_ascent_wf.py — full 243-combo WF OOS evaluation on prices_live"
```

---

## Self-Review

**Spec coverage:**
- ✅ `PortfolioBaseStrategy` ABC — Task 1
- ✅ Portfolio execution path (`_portfolio_returns`) — Task 1
- ✅ `AscentPortfolioStrategy` — Task 2
- ✅ param_grid: top_n/max_weight/trend_weight/statarb_weight/mom_window — Task 2
- ✅ Class-level smart caching (feature + alpha) — Task 2
- ✅ `clear_cache()` classmethod — Task 2
- ✅ `_make_alpha_weights()` renormalizes remaining sleeves — Task 2
- ✅ Engine portfolio branch: long-format date extraction, IS slicing — Task 3
- ✅ `clear_cache()` called before each fold IS optimization — Task 3
- ✅ Single-asset path unchanged (regression test) — Task 3
- ✅ IS optimizer receives IS data only (boundary defense, tested) — Task 3
- ✅ End-to-end runner script — Task 4
- ✅ `constraint_fn`: trend + statarb <= 0.75 — Task 4

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:**
- `PortfolioBaseStrategy.clear_cache()` is instance method (no-op) in Task 1
- `AscentPortfolioStrategy.clear_cache()` is classmethod in Task 2 — correctly overrides
- Engine calls `self.strategy_cls.clear_cache()` in Task 3 — works for classmethod
- `_portfolio_returns(data, weight_df)` in Task 1; called in Task 3 as `exec_model.compute_returns(oos_data, oos_signals)` — type dispatch correct
- `all_signals.loc[all_signals.index.isin(oos_dates_set)]` — `oos_dates_set` is a `set`, `.isin()` accepts sets ✅
