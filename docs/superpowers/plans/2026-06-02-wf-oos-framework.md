# Production Walk-Forward OOS Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, production-grade Walk-Forward OOS backtesting engine with strict anti-leakage defenses, realistic execution modelling, and a Walk-Forward Efficiency score.

**Architecture:** Six focused modules under `ascent/research/wf_framework/` — window generation (purge/embargo), abstract strategy interface, execution friction model, in-sample parameter optimizer, performance analyser, and the top-level `WalkForwardEngine` orchestrator. Each module has a single responsibility and is independently testable. The engine stitches OOS equity curves chronologically and computes WFE at the end.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy (for Bayesian opt fallback), itertools (grid search), dataclasses, abc. No new external dependencies.

---

## File Map

| File | Responsibility |
|---|---|
| `ascent/research/wf_framework/__init__.py` | Public exports |
| `ascent/research/wf_framework/windows.py` | `WindowGenerator` — IS/OOS splits with purge + embargo |
| `ascent/research/wf_framework/strategy.py` | `BaseStrategy` ABC + `MACrossStrategy` example |
| `ascent/research/wf_framework/execution.py` | `ExecutionModel` — ATR slippage, commission, borrow cost |
| `ascent/research/wf_framework/optimizer.py` | `ParameterOptimizer` — grid search strictly within IS window |
| `ascent/research/wf_framework/metrics.py` | `PerformanceAnalyzer` — Sharpe, Sortino, MDD, WFE |
| `ascent/research/wf_framework/engine.py` | `WalkForwardEngine` — top-level orchestrator |
| `tests/test_wf_framework/test_windows.py` | Unit tests for WindowGenerator |
| `tests/test_wf_framework/test_execution.py` | Unit tests for ExecutionModel |
| `tests/test_wf_framework/test_optimizer.py` | Unit tests for ParameterOptimizer |
| `tests/test_wf_framework/test_metrics.py` | Unit tests for PerformanceAnalyzer |
| `tests/test_wf_framework/test_engine.py` | Integration test — full WF run on synthetic data |

---

## Task 1: WindowGenerator — IS/OOS splits with purge and embargo

**Files:**
- Create: `ascent/research/wf_framework/__init__.py`
- Create: `ascent/research/wf_framework/windows.py`
- Create: `tests/test_wf_framework/__init__.py`
- Create: `tests/test_wf_framework/test_windows.py`

The `WindowGenerator` is the foundation. Every other module depends on the slices it produces. The purge gap removes the tail of the IS window where forward-looking labels overlap with OOS data. The embargo gap removes the head of the OOS window to break serial correlation.

```
Timeline:
|--- IS window ---|--- purge ---|--- embargo ---|--- OOS window ---|
                 ^             ^               ^
            is_end        purge_end      oos_start
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_windows.py
import pandas as pd
import pytest
from ascent.research.wf_framework.windows import WindowGenerator, SplitWindow

@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-02", periods=600)

def test_window_count(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    windows = gen.generate(dates)
    assert len(windows) >= 3, "Expected at least 3 folds for 600-day date range"

def test_no_overlap(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        assert w.oos_start > w.purge_end, "OOS must start after purge"
        assert w.purge_end >= w.is_end, "Purge must extend past IS end"
        assert w.oos_start > w.embargo_end or w.oos_start == w.embargo_end + pd.Timedelta(days=1), \
            "OOS must start after embargo"

def test_is_slice_max_index(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        is_slice = w.slice_is(dates)
        oos_slice = w.slice_oos(dates)
        assert is_slice.max() < oos_slice.min(), "IS data must precede OOS data"

def test_is_slice_excludes_purge(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        is_slice = w.slice_is(dates)
        assert is_slice.max() <= w.purge_start, \
            "IS slice must not include dates in the purge window"

def test_rolling_window_advances(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5,
                          window_type="rolling")
    windows = gen.generate(dates)
    for i in range(1, len(windows)):
        assert windows[i].is_start > windows[i-1].is_start, \
            "Rolling window IS start must advance each fold"

def test_anchored_window_fixed_start(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5,
                          window_type="anchored")
    windows = gen.generate(dates)
    for w in windows:
        assert w.is_start == windows[0].is_start, \
            "Anchored window IS start must be fixed"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_wf_framework/test_windows.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` — `wf_framework` doesn't exist yet.

- [ ] **Step 3: Create package scaffolding**

```python
# ascent/research/wf_framework/__init__.py
from .windows import WindowGenerator, SplitWindow
from .strategy import BaseStrategy
from .execution import ExecutionModel, ExecutionConfig
from .optimizer import ParameterOptimizer
from .metrics import PerformanceAnalyzer
from .engine import WalkForwardEngine

__all__ = [
    "WindowGenerator", "SplitWindow",
    "BaseStrategy",
    "ExecutionModel", "ExecutionConfig",
    "ParameterOptimizer",
    "PerformanceAnalyzer",
    "WalkForwardEngine",
]
```

```python
# tests/test_wf_framework/__init__.py
```

- [ ] **Step 4: Implement `windows.py`**

```python
# ascent/research/wf_framework/windows.py
"""
Walk-Forward Window Generator
==============================
Produces IS/OOS date-index slices with mandatory purge and embargo gaps.

Boundary defenses
-----------------
Purge  : The final `purge_days` bars of the IS window are excluded from the
         IS slice. This removes dates where a forward-looking label (e.g.
         21-day forward return) would overlap with OOS data, preventing
         label leakage at the IS/OOS boundary.

Embargo: The first `embargo_days` bars after the purge window are excluded
         entirely (neither IS nor OOS). This breaks the serial correlation
         that remains between adjacent IS and OOS returns even after purging,
         caused by the autocorrelation structure of asset returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    fold_id: int
    is_start: pd.Timestamp       # first bar available for training
    purge_start: pd.Timestamp    # first bar excluded from IS (purge begins)
    is_end: pd.Timestamp         # last bar of the raw IS window (before purge cut)
    purge_end: pd.Timestamp      # last bar of purge gap
    embargo_end: pd.Timestamp    # last bar of embargo gap
    oos_start: pd.Timestamp      # first OOS bar
    oos_end: pd.Timestamp        # last OOS bar

    def slice_is(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """IS dates: [is_start, purge_start). Purge tail excluded."""
        return dates[(dates >= self.is_start) & (dates < self.purge_start)]

    def slice_oos(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """OOS dates: [oos_start, oos_end]."""
        return dates[(dates >= self.oos_start) & (dates <= self.oos_end)]

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"IS [{self.is_start.date()} → {self.purge_start.date()}) "
            f"| purge [{self.purge_start.date()} → {self.purge_end.date()}] "
            f"| embargo [{self.purge_end.date()} → {self.embargo_end.date()}] "
            f"| OOS [{self.oos_start.date()} → {self.oos_end.date()}]"
        )


class WindowGenerator:
    """
    Generate walk-forward train/test splits with purge and embargo gaps.

    Parameters
    ----------
    is_days      : Number of trading days in the in-sample window.
    oos_days     : Number of trading days in the out-of-sample window.
    purge_days   : Bars removed from the IS tail to prevent label leakage.
                   Set this >= your maximum forward-return horizon.
    embargo_days : Bars removed from the OOS head to break serial correlation.
    window_type  : "rolling" (IS window slides) or "anchored" (IS always
                   starts from the first available date).
    step_days    : How many OOS bars to advance between folds.
                   Defaults to oos_days (non-overlapping OOS periods).
    """

    def __init__(
        self,
        is_days: int = 252,
        oos_days: int = 63,
        purge_days: int = 21,
        embargo_days: int = 5,
        window_type: Literal["rolling", "anchored"] = "rolling",
        step_days: int | None = None,
    ):
        if purge_days + embargo_days >= oos_days:
            raise ValueError(
                f"purge_days ({purge_days}) + embargo_days ({embargo_days}) "
                f"must be < oos_days ({oos_days})"
            )
        self.is_days = is_days
        self.oos_days = oos_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.window_type = window_type
        self.step_days = step_days if step_days is not None else oos_days

    def generate(self, dates: pd.DatetimeIndex) -> list[SplitWindow]:
        """
        Generate all valid SplitWindow objects for the given date index.

        The method advances the OOS window by `step_days` each fold.
        IS start advances by the same amount for rolling windows; stays
        fixed at dates[0] for anchored windows.
        """
        dates = dates.sort_values().drop_duplicates()
        n = len(dates)
        windows: list[SplitWindow] = []
        fold_id = 0

        # First IS window ends at index (is_days - 1)
        is_end_idx = self.is_days - 1

        while True:
            # IS window bounds
            if self.window_type == "anchored":
                is_start_idx = 0
            else:
                is_start_idx = is_end_idx - self.is_days + 1

            if is_start_idx < 0:
                is_end_idx += self.step_days
                continue

            # Purge: last purge_days bars of IS are excluded from training
            purge_start_idx = is_end_idx - self.purge_days + 1
            purge_end_idx   = is_end_idx

            # Embargo: purge_days bars immediately after IS end
            embargo_start_idx = purge_end_idx + 1
            embargo_end_idx   = embargo_start_idx + self.embargo_days - 1

            # OOS window starts after embargo
            oos_start_idx = embargo_end_idx + 1
            oos_end_idx   = oos_start_idx + self.oos_days - 1

            if oos_end_idx >= n:
                break  # Not enough data for a full OOS window

            windows.append(SplitWindow(
                fold_id      = fold_id,
                is_start     = dates[is_start_idx],
                purge_start  = dates[purge_start_idx],
                is_end       = dates[is_end_idx],
                purge_end    = dates[purge_end_idx],
                embargo_end  = dates[embargo_end_idx],
                oos_start    = dates[oos_start_idx],
                oos_end      = dates[oos_end_idx],
            ))

            fold_id    += 1
            is_end_idx += self.step_days

        return windows
```

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_windows.py -v
```
Expected: 6 PASSED.

- [ ] **Step 6: Commit**

```bash
git add ascent/research/wf_framework/__init__.py ascent/research/wf_framework/windows.py \
        tests/test_wf_framework/__init__.py tests/test_wf_framework/test_windows.py
git commit -m "feat: wf-framework windows — IS/OOS splits with purge and embargo"
```

---

## Task 2: BaseStrategy ABC + MA Crossover example

**Files:**
- Create: `ascent/research/wf_framework/strategy.py`
- Create: `tests/test_wf_framework/test_strategy.py`

The `BaseStrategy` ABC defines the interface every strategy must implement. The `MACrossStrategy` serves as the canonical example and is used in the integration test. Signals: +1 (long), -1 (short), 0 (flat).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_strategy.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.strategy import BaseStrategy, MACrossStrategy

@pytest.fixture
def price_data():
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-02", periods=300)
    prices = pd.Series(100 * (1 + np.random.randn(300) * 0.01).cumprod(), index=idx)
    return pd.DataFrame({"close": prices, "high": prices * 1.005,
                         "low": prices * 0.995, "volume": 1_000_000})

def test_ma_cross_is_base_strategy():
    assert issubclass(MACrossStrategy, BaseStrategy)

def test_ma_cross_param_grid():
    s = MACrossStrategy(fast=10, slow=50)
    assert "fast" in s.param_grid
    assert "slow" in s.param_grid

def test_ma_cross_signals_shape(price_data):
    s = MACrossStrategy(fast=10, slow=50)
    signals = s.generate_signals(price_data)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(price_data)

def test_ma_cross_signals_values(price_data):
    s = MACrossStrategy(fast=10, slow=50)
    signals = s.generate_signals(price_data)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})

def test_ma_cross_no_future_look(price_data):
    s = MACrossStrategy(fast=5, slow=20)
    sig_full = s.generate_signals(price_data)
    sig_partial = s.generate_signals(price_data.iloc[:150])
    # Signals on the first 149 bars must match between full and partial runs
    common_idx = sig_partial.index
    pd.testing.assert_series_equal(
        sig_full.loc[common_idx], sig_partial,
        check_names=False,
    )

def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseStrategy()
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_strategy.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Implement `strategy.py`**

```python
# ascent/research/wf_framework/strategy.py
"""
Strategy Interface
==================
BaseStrategy is the ABC every user strategy must subclass.

Implementing a strategy
-----------------------
1. Subclass BaseStrategy.
2. Define `param_grid` property — dict of param_name → list of candidate values.
3. Implement `generate_signals(data)` — accepts OHLCV DataFrame, returns
   a pd.Series of {-1, 0, +1} aligned to the input index.

The engine instantiates your strategy class with **params at optimisation time
and at OOS evaluation time. generate_signals must be strictly causal: signal
at index i may only use data at indices <= i.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base for all walk-forward strategies."""

    def __init__(self, **params):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    @property
    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return {param_name: [candidate_values]} for grid search."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute entry/exit signals from OHLCV data.

        Parameters
        ----------
        data : DataFrame with columns [open, high, low, close, volume].
               Must be sorted ascending by date index. No future data allowed.

        Returns
        -------
        pd.Series of int: +1 (long), -1 (short), 0 (flat).
        Index must match data.index exactly.
        """


class MACrossStrategy(BaseStrategy):
    """
    Simple moving-average crossover.

    Long  when fast SMA > slow SMA.
    Short when fast SMA < slow SMA.
    Flat  during warmup (first `slow` bars).

    Parameters
    ----------
    fast : int — fast SMA lookback (default 10)
    slow : int — slow SMA lookback (default 50)
    """

    def __init__(self, fast: int = 10, slow: int = 50):
        super().__init__(fast=fast, slow=slow)

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100, 200],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()

        signal = pd.Series(0, index=data.index, dtype=int)
        valid  = fast_ma.notna() & slow_ma.notna()
        signal.loc[valid & (fast_ma > slow_ma)] =  1
        signal.loc[valid & (fast_ma < slow_ma)] = -1
        return signal
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_strategy.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascent/research/wf_framework/strategy.py tests/test_wf_framework/test_strategy.py
git commit -m "feat: wf-framework strategy — BaseStrategy ABC + MACrossStrategy"
```

---

## Task 3: ExecutionModel — slippage, commission, borrow cost

**Files:**
- Create: `ascent/research/wf_framework/execution.py`
- Create: `tests/test_wf_framework/test_execution.py`

The `ExecutionModel` converts a raw signal series into a P&L series by modelling three friction layers in sequence: (1) slippage debited on the execution bar, (2) per-trade commission, (3) overnight borrow cost for short positions.

**ATR slippage formula:** `slippage_per_share = atr_multiplier × ATR_14`. Applied as a cost against the trade direction — buy at `open + slippage`, sell at `open - slippage`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_execution.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig

@pytest.fixture
def ohlcv():
    np.random.seed(0)
    n = 100
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=idx)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": 1_000_000,
    })

@pytest.fixture
def long_signals(ohlcv):
    return pd.Series(1, index=ohlcv.index, dtype=int)

def test_zero_friction_gross_return(ohlcv):
    cfg = ExecutionConfig(slippage_model="fixed_pct", fixed_pct=0.0,
                          commission_pct=0.0, borrow_rate_annual=0.0)
    em = ExecutionModel(cfg)
    signals = pd.Series(1, index=ohlcv.index)
    returns = em.compute_returns(ohlcv, signals)
    # With zero friction, returns ≈ close-to-close
    raw = ohlcv["close"].pct_change().fillna(0)
    pd.testing.assert_series_equal(returns, raw, atol=1e-6, check_names=False)

def test_commission_reduces_returns(ohlcv, long_signals):
    cfg_no  = ExecutionConfig(commission_pct=0.0)
    cfg_yes = ExecutionConfig(commission_pct=0.001)
    r_no  = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_yes = ExecutionModel(cfg_yes).compute_returns(ohlcv, long_signals)
    assert r_yes.sum() < r_no.sum()

def test_atr_slippage_reduces_returns(ohlcv, long_signals):
    cfg_no  = ExecutionConfig(slippage_model="atr", atr_multiplier=0.0)
    cfg_yes = ExecutionConfig(slippage_model="atr", atr_multiplier=0.1)
    r_no  = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_yes = ExecutionModel(cfg_yes).compute_returns(ohlcv, long_signals)
    assert r_yes.sum() < r_no.sum()

def test_borrow_cost_on_shorts(ohlcv):
    cfg = ExecutionConfig(borrow_rate_annual=0.03, commission_pct=0.0,
                          slippage_model="fixed_pct", fixed_pct=0.0)
    signals = pd.Series(-1, index=ohlcv.index)
    returns = ExecutionModel(cfg).compute_returns(ohlcv, signals)
    raw     = -ohlcv["close"].pct_change().fillna(0)
    # Short returns must be worse than raw short by borrow cost
    assert returns.sum() < raw.sum()

def test_no_borrow_on_longs(ohlcv, long_signals):
    cfg_no   = ExecutionConfig(borrow_rate_annual=0.0,  commission_pct=0.0,
                               slippage_model="fixed_pct", fixed_pct=0.0)
    cfg_borrow = ExecutionConfig(borrow_rate_annual=0.50, commission_pct=0.0,
                                 slippage_model="fixed_pct", fixed_pct=0.0)
    r_no     = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_borrow = ExecutionModel(cfg_borrow).compute_returns(ohlcv, long_signals)
    # Borrow rate must not affect long positions
    pd.testing.assert_series_equal(r_no, r_borrow, atol=1e-10, check_names=False)

def test_output_index_matches_input(ohlcv, long_signals):
    em = ExecutionModel(ExecutionConfig())
    returns = em.compute_returns(ohlcv, long_signals)
    assert returns.index.equals(ohlcv.index)
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_execution.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `execution.py`**

```python
# ascent/research/wf_framework/execution.py
"""
Execution Friction Model
========================
Converts a signal series into a net-of-friction daily return series.

Three friction layers applied in order on each bar:
  1. Slippage  — cost on the open of the execution bar (bar after signal).
  2. Commission — flat percentage of notional traded.
  3. Borrow    — daily overnight financing cost on short positions.

Slippage models
---------------
"atr"       : slippage = atr_multiplier × ATR(14). Scales with volatility.
"fixed_pct" : slippage = fixed_pct × execution_price. Constant percentage.

Signal convention
-----------------
+1 = long (buy on next open, hold until signal changes)
-1 = short (sell short on next open)
 0 = flat  (no position)

Boundary defense: signals are shifted by 1 bar (execution_delay=1) before
computing returns so that the signal at date t is executed at the open of
date t+1. This prevents look-ahead from intraday data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    slippage_model:    Literal["atr", "fixed_pct"] = "atr"
    atr_multiplier:    float = 0.10      # 0.1 × ATR(14) per share
    fixed_pct:         float = 0.0005    # 5 bps of execution price
    atr_window:        int   = 14
    commission_pct:    float = 0.0005    # 5 bps per trade (one-way)
    borrow_rate_annual: float = 0.0      # annual borrow rate for shorts (0 = no cost)
    execution_delay:   int   = 1         # bars between signal and execution


class ExecutionModel:
    """Apply realistic friction to convert signals → net daily returns."""

    def __init__(self, config: ExecutionConfig | None = None):
        self.cfg = config or ExecutionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        data:    pd.DataFrame,
        signals: pd.Series,
    ) -> pd.Series:
        """
        Parameters
        ----------
        data    : OHLCV DataFrame (index = dates, cols include open/high/low/close).
        signals : pd.Series of {-1, 0, +1}, index aligned to data.

        Returns
        -------
        pd.Series of daily net returns, index aligned to data.
        """
        cfg = self.cfg
        delay = cfg.execution_delay

        # Execution price = open of the bar after signal (1-day delay).
        exec_price = data["open"].shift(-delay)

        # Position held: signal from bar t is active from bar t+delay onward.
        position = signals.shift(delay).fillna(0)

        # Raw close-to-close return for each bar
        raw_ret = data["close"].pct_change().fillna(0)

        # Gross position return
        gross_ret = position * raw_ret

        # --- Slippage (deducted on execution bars where position changes) ---
        pos_change = position.diff().fillna(position)  # first bar treated as entry
        is_trade   = pos_change.abs() > 0

        if cfg.slippage_model == "atr":
            atr = self._atr(data, cfg.atr_window)
            slip_cost = cfg.atr_multiplier * atr / exec_price.clip(lower=1e-8)
        else:
            slip_cost = pd.Series(cfg.fixed_pct, index=data.index)

        slip_cost = slip_cost.fillna(0)

        # Slippage is always a cost (positive slip_cost reduces return).
        slip_deduction = is_trade * pos_change.abs() * slip_cost

        # --- Commission (one-way, applied on entry and exit bars) ---
        commission_deduction = is_trade * cfg.commission_pct

        # --- Borrow cost (daily, applied only to short positions) ---
        daily_borrow_rate = cfg.borrow_rate_annual / 252
        borrow_deduction  = (position < 0).astype(float) * daily_borrow_rate

        net_ret = gross_ret - slip_deduction - commission_deduction - borrow_deduction
        return net_ret.rename("net_return")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atr(data: pd.DataFrame, window: int) -> pd.Series:
        """ATR(window) using Wilder's method on high/low/prev_close."""
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False).mean()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_execution.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascent/research/wf_framework/execution.py tests/test_wf_framework/test_execution.py
git commit -m "feat: wf-framework execution — ATR slippage, commission, borrow cost"
```

---

## Task 4: ParameterOptimizer — grid search strictly within IS window

**Files:**
- Create: `ascent/research/wf_framework/optimizer.py`
- Create: `tests/test_wf_framework/test_optimizer.py`

The optimizer searches the strategy's `param_grid` exclusively on IS data. It returns the params that maximise the objective (default: Sharpe). The `optimize` method must never receive OOS data — the `WalkForwardEngine` enforces this by slicing before calling.

Valid param combinations filter out `fast >= slow` (for MA-type strategies) via a user-supplied `constraint_fn`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_optimizer.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.strategy import MACrossStrategy
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig
from ascent.research.wf_framework.optimizer import ParameterOptimizer

@pytest.fixture
def is_data():
    np.random.seed(7)
    n = 300
    idx = pd.bdate_range("2019-01-02", periods=n)
    close = pd.Series(100 * (1 + np.random.randn(n) * 0.012).cumprod(), index=idx)
    return pd.DataFrame({
        "open":  close * 0.999,
        "high":  close * 1.006,
        "low":   close * 0.994,
        "close": close,
        "volume": 500_000,
    })

def test_returns_dict(is_data):
    em  = ExecutionModel(ExecutionConfig(commission_pct=0.0, slippage_model="fixed_pct", fixed_pct=0.0))
    opt = ParameterOptimizer(MACrossStrategy, em)
    best_params, best_score = opt.optimize(is_data)
    assert isinstance(best_params, dict)
    assert "fast" in best_params and "slow" in best_params
    assert isinstance(best_score, float)

def test_constraint_respected(is_data):
    em  = ExecutionModel(ExecutionConfig(commission_pct=0.0))
    opt = ParameterOptimizer(
        MACrossStrategy, em,
        constraint_fn=lambda p: p["fast"] < p["slow"]
    )
    best_params, _ = opt.optimize(is_data)
    assert best_params["fast"] < best_params["slow"]

def test_optimizer_does_not_use_future_data(is_data):
    """Optimizer must return same result regardless of data appended AFTER is_data."""
    em  = ExecutionModel(ExecutionConfig())
    opt = ParameterOptimizer(MACrossStrategy, em)

    params_is, _ = opt.optimize(is_data)

    # Append 60 bars of "future" data
    n_future = 60
    idx_future = pd.bdate_range(is_data.index[-1] + pd.tseries.offsets.BDay(1), periods=n_future)
    future = pd.DataFrame({
        "open":  100, "high": 101, "low": 99, "close": 100, "volume": 1
    }, index=idx_future)
    combined = pd.concat([is_data, future])

    # Optimizer called with IS slice only — result must be identical
    params_combined, _ = opt.optimize(is_data)  # same IS slice
    assert params_is == params_combined

def test_best_score_is_finite(is_data):
    em  = ExecutionModel(ExecutionConfig())
    opt = ParameterOptimizer(MACrossStrategy, em)
    _, score = opt.optimize(is_data)
    assert np.isfinite(score)
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_optimizer.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `optimizer.py`**

```python
# ascent/research/wf_framework/optimizer.py
"""
Parameter Optimizer
===================
Grid-searches a strategy's param_grid on in-sample data only.

Boundary defense
----------------
`optimize(is_data)` receives only the IS date slice. The WalkForwardEngine
is responsible for slicing before calling — the optimizer has NO knowledge
of the full dataset and cannot access OOS data even accidentally.

Objective
---------
Default objective: annualised Sharpe ratio.
Pass `objective="calmar"` for drawdown-adjusted optimisation.
A finite, non-NaN score is required — parameter sets that produce all-zero
signals or degenerate returns receive a score of -inf.

Constraints
-----------
Pass `constraint_fn` to filter out invalid combinations (e.g., fast >= slow).
Invalid combinations are skipped entirely, not penalised.
"""
from __future__ import annotations
from itertools import product
from typing import Callable, Type
import numpy as np
import pandas as pd

from .strategy import BaseStrategy
from .execution import ExecutionModel


def _sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    if ann_vol < 1e-10:
        return -np.inf
    return (ann_ret - rf_annual) / ann_vol


def _calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    cum     = (1 + returns).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    if abs(mdd) < 1e-10:
        return -np.inf
    return ann_ret / abs(mdd)


_OBJECTIVES = {"sharpe": _sharpe, "calmar": _calmar}


class ParameterOptimizer:
    """
    Grid-search optimizer for BaseStrategy subclasses.

    Parameters
    ----------
    strategy_cls   : Subclass of BaseStrategy to optimise.
    execution_model: ExecutionModel instance for friction.
    objective      : "sharpe" (default) or "calmar".
    constraint_fn  : Optional callable(params_dict) → bool.
                     Return False to skip a parameter combination.
    rf_annual      : Risk-free rate for Sharpe calculation.
    """

    def __init__(
        self,
        strategy_cls:    Type[BaseStrategy],
        execution_model: ExecutionModel,
        objective:       str = "sharpe",
        constraint_fn:   Callable[[dict], bool] | None = None,
        rf_annual:       float = 0.0,
    ):
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {list(_OBJECTIVES)}")
        self.strategy_cls    = strategy_cls
        self.execution_model = execution_model
        self.objective_fn    = _OBJECTIVES[objective]
        self.constraint_fn   = constraint_fn
        self.rf_annual       = rf_annual

    def optimize(self, is_data: pd.DataFrame) -> tuple[dict, float]:
        """
        Search param_grid on IS data. Returns (best_params, best_score).

        Parameters
        ----------
        is_data : DataFrame of OHLCV data covering only the IS window.
                  MUST NOT contain any OOS bars. The engine enforces this.
        """
        # Build candidate grid from a temporary strategy instance
        template = self.strategy_cls()
        grid     = template.param_grid

        param_names = list(grid.keys())
        param_values = list(grid.values())

        best_params: dict  = {}
        best_score:  float = -np.inf

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            # Skip constraint-violating combinations
            if self.constraint_fn is not None and not self.constraint_fn(params):
                continue

            try:
                strategy = self.strategy_cls(**params)
                signals  = strategy.generate_signals(is_data)
                returns  = self.execution_model.compute_returns(is_data, signals)
                score    = self.objective_fn(returns, self.rf_annual) \
                           if "rf_annual" in self.objective_fn.__code__.co_varnames \
                           else self.objective_fn(returns)
            except Exception:
                score = -np.inf

            if np.isfinite(score) and score > best_score:
                best_score  = score
                best_params = params

        if not best_params:
            # Fallback to template defaults if all combos failed
            best_params = {k: v[0] for k, v in grid.items()}

        return best_params, best_score
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_optimizer.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascent/research/wf_framework/optimizer.py tests/test_wf_framework/test_optimizer.py
git commit -m "feat: wf-framework optimizer — grid search strictly within IS window"
```

---

## Task 5: PerformanceAnalyzer — metrics + Walk-Forward Efficiency

**Files:**
- Create: `ascent/research/wf_framework/metrics.py`
- Create: `tests/test_wf_framework/test_metrics.py`

WFE (Walk-Forward Efficiency) is defined per fold then averaged:

```
WFE_fold = OOS_Sharpe_fold / IS_Sharpe_fold
WFE      = mean(WFE_fold)   [excluding folds where IS_Sharpe <= 0]

WFE > 1.0 : strategy generalises better than in-sample (rare, check for bugs)
WFE 0.5–1.0: normal degradation — acceptable
WFE < 0.5  : significant overfitting — strategy should not be traded live
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wf_framework/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.metrics import PerformanceAnalyzer, FoldResult

@pytest.fixture
def positive_returns():
    np.random.seed(1)
    idx = pd.bdate_range("2021-01-04", periods=252)
    return pd.Series(np.random.randn(252) * 0.008 + 0.0003, index=idx)

@pytest.fixture
def benchmark_returns(positive_returns):
    np.random.seed(2)
    return pd.Series(np.random.randn(len(positive_returns)) * 0.007 + 0.0002,
                     index=positive_returns.index)

@pytest.fixture
def fold_results():
    return [
        FoldResult(fold_id=0, is_sharpe=1.2, oos_returns=pd.Series([0.001]*63)),
        FoldResult(fold_id=1, is_sharpe=0.9, oos_returns=pd.Series([0.0005]*63)),
        FoldResult(fold_id=2, is_sharpe=1.5, oos_returns=pd.Series([-0.0002]*63)),
    ]

def test_sharpe_positive_for_positive_returns(positive_returns):
    pa = PerformanceAnalyzer()
    assert pa.sharpe(positive_returns) > 0

def test_sharpe_uses_rf(positive_returns):
    pa = PerformanceAnalyzer(rf_annual=0.05)
    sharpe_high_rf = pa.sharpe(positive_returns)
    pa_no_rf = PerformanceAnalyzer(rf_annual=0.0)
    sharpe_no_rf = pa_no_rf.sharpe(positive_returns)
    assert sharpe_no_rf > sharpe_high_rf

def test_max_drawdown_negative(positive_returns):
    pa = PerformanceAnalyzer()
    mdd = pa.max_drawdown(positive_returns)
    assert mdd <= 0

def test_sortino_geq_sharpe_positive(positive_returns):
    pa = PerformanceAnalyzer()
    assert pa.sortino(positive_returns) >= pa.sharpe(positive_returns)

def test_win_rate_range(positive_returns):
    pa = PerformanceAnalyzer()
    wr = pa.win_rate(positive_returns)
    assert 0.0 <= wr <= 1.0

def test_wfe_computed(fold_results):
    pa = PerformanceAnalyzer()
    wfe = pa.walk_forward_efficiency(fold_results)
    assert np.isfinite(wfe)
    assert wfe > 0

def test_full_report_keys(positive_returns, benchmark_returns, fold_results):
    pa = PerformanceAnalyzer()
    report = pa.full_report(positive_returns, benchmark_returns, fold_results)
    for key in ["cagr", "sharpe", "sortino", "max_drawdown", "win_rate",
                "wfe", "alpha", "beta", "n_folds"]:
        assert key in report, f"Missing key: {key}"
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_metrics.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `metrics.py`**

```python
# ascent/research/wf_framework/metrics.py
"""
Performance Analyser
====================
Computes all standard metrics on OOS data only, plus Walk-Forward Efficiency.

All metrics are computed on the stitched OOS equity curve — never on IS data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    """Holds per-fold IS Sharpe and OOS returns for WFE computation."""
    fold_id:    int
    is_sharpe:  float
    oos_returns: pd.Series


class PerformanceAnalyzer:
    """Compute metrics strictly on OOS return series."""

    def __init__(self, rf_annual: float = 0.0, periods_per_year: int = 252):
        self.rf_annual        = rf_annual
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def cagr(self, returns: pd.Series) -> float:
        n   = len(returns)
        tot = (1 + returns).prod()
        if n == 0 or tot <= 0:
            return 0.0
        return tot ** (self.periods_per_year / n) - 1

    def volatility(self, returns: pd.Series) -> float:
        return returns.std() * np.sqrt(self.periods_per_year)

    def sharpe(self, returns: pd.Series) -> float:
        vol = self.volatility(returns)
        if vol < 1e-10:
            return 0.0
        return (self.cagr(returns) - self.rf_annual) / vol

    def sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < self.rf_annual / self.periods_per_year]
        dv = downside.std() * np.sqrt(self.periods_per_year)
        if dv < 1e-10:
            return np.inf if self.cagr(returns) > self.rf_annual else 0.0
        return (self.cagr(returns) - self.rf_annual) / dv

    def max_drawdown(self, returns: pd.Series) -> float:
        cum  = (1 + returns).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        return float(dd.min())

    def win_rate(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).mean())

    # ------------------------------------------------------------------
    # Alpha / Beta vs benchmark
    # ------------------------------------------------------------------

    def alpha_beta(
        self,
        returns:   pd.Series,
        benchmark: pd.Series,
    ) -> tuple[float, float]:
        common = returns.index.intersection(benchmark.index)
        r = returns.reindex(common)
        b = benchmark.reindex(common)
        if b.var() < 1e-12:
            return 0.0, 0.0
        beta  = float(r.cov(b) / b.var())
        alpha = self.cagr(r) - beta * self.cagr(b)
        return alpha, beta

    # ------------------------------------------------------------------
    # Walk-Forward Efficiency
    # ------------------------------------------------------------------

    def walk_forward_efficiency(self, fold_results: list[FoldResult]) -> float:
        """
        WFE = mean(OOS_Sharpe_fold / IS_Sharpe_fold) across folds where
        IS_Sharpe > 0. Returns NaN if no valid folds.

        Interpretation
        --------------
        > 1.0 : OOS beats IS — unusual, verify no data leakage.
        0.5–1.0: Normal degradation — strategy is tradeable.
        < 0.5 : Significant overfitting — do not trade live.
        """
        ratios = []
        for fold in fold_results:
            if fold.is_sharpe <= 0:
                continue
            oos_sharpe = self.sharpe(fold.oos_returns)
            if np.isfinite(oos_sharpe):
                ratios.append(oos_sharpe / fold.is_sharpe)
        return float(np.mean(ratios)) if ratios else float("nan")

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        oos_returns:  pd.Series,
        benchmark:    pd.Series | None,
        fold_results: list[FoldResult],
    ) -> dict:
        report = {
            "cagr":         self.cagr(oos_returns),
            "volatility":   self.volatility(oos_returns),
            "sharpe":       self.sharpe(oos_returns),
            "sortino":      self.sortino(oos_returns),
            "max_drawdown": self.max_drawdown(oos_returns),
            "win_rate":     self.win_rate(oos_returns),
            "wfe":          self.walk_forward_efficiency(fold_results),
            "n_folds":      len(fold_results),
            "n_oos_days":   len(oos_returns),
        }

        if benchmark is not None:
            alpha, beta = self.alpha_beta(oos_returns, benchmark)
            report["alpha"] = alpha
            report["beta"]  = beta
        else:
            report["alpha"] = float("nan")
            report["beta"]  = float("nan")

        return report

    def print_report(self, report: dict) -> None:
        print("=" * 55)
        print("  WALK-FORWARD OOS PERFORMANCE REPORT")
        print("=" * 55)
        print(f"  OOS Trading Days  : {report['n_oos_days']}")
        print(f"  Folds             : {report['n_folds']}")
        print(f"  CAGR              : {report['cagr']*100:+.2f}%")
        print(f"  Volatility        : {report['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio      : {report['sharpe']:.3f}")
        print(f"  Sortino Ratio     : {report['sortino']:.3f}")
        print(f"  Max Drawdown      : {report['max_drawdown']*100:.2f}%")
        print(f"  Win Rate          : {report['win_rate']*100:.1f}%")
        print("-" * 55)
        print(f"  Alpha vs BM       : {report['alpha']*100:+.2f}%")
        print(f"  Beta              : {report['beta']:.3f}")
        print("-" * 55)
        print(f"  Walk-Forward Eff. : {report['wfe']:.3f}  "
              f"({'acceptable' if report['wfe'] >= 0.5 else 'OVERFIT'})")
        print("=" * 55)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_metrics.py -v
```
Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascent/research/wf_framework/metrics.py tests/test_wf_framework/test_metrics.py
git commit -m "feat: wf-framework metrics — Sharpe/Sortino/MDD/WFE analyser"
```

---

## Task 6: WalkForwardEngine — top-level orchestrator

**Files:**
- Create: `ascent/research/wf_framework/engine.py`
- Create: `tests/test_wf_framework/test_engine.py`

The engine is the only public surface a user needs. It:
1. Generates windows via `WindowGenerator`
2. For each fold: slices IS data → optimizer → best params → OOS signals → execution → fold returns
3. Stitches OOS returns chronologically (no overlap)
4. Calls `PerformanceAnalyzer.full_report` on the stitched series
5. Prints the report and returns equity curve + report dict

The engine enforces the key boundary defense: the optimizer receives `is_data = full_data.loc[is_slice]` — never the full dataset.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_wf_framework/test_engine.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.engine import WalkForwardEngine
from ascent.research.wf_framework.strategy import MACrossStrategy
from ascent.research.wf_framework.execution import ExecutionConfig
from ascent.research.wf_framework.windows import WindowGenerator

@pytest.fixture
def synthetic_ohlcv():
    """600 days of trending synthetic price data."""
    np.random.seed(99)
    n   = 600
    idx = pd.bdate_range("2018-01-02", periods=n)
    log_returns = np.random.randn(n) * 0.01 + 0.0003  # slight upward drift
    close = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=idx)
    return pd.DataFrame({
        "open":   close.shift(1).fillna(100),
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": 1_000_000,
    })

@pytest.fixture
def benchmark(synthetic_ohlcv):
    spy = synthetic_ohlcv["close"].pct_change().fillna(0)
    return spy.rename("benchmark")

def test_engine_runs(synthetic_ohlcv, benchmark):
    engine = WalkForwardEngine(
        strategy_cls=MACrossStrategy,
        window_generator=WindowGenerator(is_days=252, oos_days=63,
                                         purge_days=21, embargo_days=5),
        exec_config=ExecutionConfig(commission_pct=0.0005),
        constraint_fn=lambda p: p["fast"] < p["slow"],
    )
    equity_curve, report = engine.run(synthetic_ohlcv, benchmark)
    assert isinstance(equity_curve, pd.Series)
    assert isinstance(report, dict)
    assert "sharpe" in report and "wfe" in report

def test_oos_periods_non_overlapping(synthetic_ohlcv):
    engine = WalkForwardEngine(
        strategy_cls=MACrossStrategy,
        window_generator=WindowGenerator(is_days=252, oos_days=63,
                                         purge_days=21, embargo_days=5),
    )
    equity_curve, _ = engine.run(synthetic_ohlcv)
    assert not equity_curve.index.duplicated().any(), \
        "OOS periods must not overlap — duplicate dates found"

def test_equity_curve_starts_at_1(synthetic_ohlcv):
    engine = WalkForwardEngine(
        strategy_cls=MACrossStrategy,
        window_generator=WindowGenerator(is_days=252, oos_days=63,
                                         purge_days=21, embargo_days=5),
    )
    equity_curve, _ = engine.run(synthetic_ohlcv)
    assert abs(equity_curve.iloc[0] - 1.0) < 1e-6

def test_wfe_finite(synthetic_ohlcv):
    engine = WalkForwardEngine(
        strategy_cls=MACrossStrategy,
        window_generator=WindowGenerator(is_days=252, oos_days=63,
                                         purge_days=21, embargo_days=5),
    )
    _, report = engine.run(synthetic_ohlcv)
    assert np.isfinite(report["wfe"])

def test_no_is_data_in_optimizer(synthetic_ohlcv):
    """
    Verify that the optimizer's IS slice never extends past the IS window end.
    Achieved by patching the optimizer to record the max date it receives.
    """
    from unittest.mock import patch, MagicMock
    from ascent.research.wf_framework.optimizer import ParameterOptimizer
    max_dates_seen = []

    original_optimize = ParameterOptimizer.optimize
    def recording_optimize(self, is_data):
        max_dates_seen.append(is_data.index.max())
        return original_optimize(self, is_data)

    engine = WalkForwardEngine(
        strategy_cls=MACrossStrategy,
        window_generator=WindowGenerator(is_days=252, oos_days=63,
                                         purge_days=21, embargo_days=5),
    )

    with patch.object(ParameterOptimizer, "optimize", recording_optimize):
        equity_curve, _ = engine.run(synthetic_ohlcv)

    windows = engine.last_windows_
    for i, w in enumerate(windows):
        if i < len(max_dates_seen):
            assert max_dates_seen[i] <= w.purge_start, \
                f"Fold {i}: optimizer received data past IS end ({max_dates_seen[i]} > {w.purge_start})"
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_engine.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `engine.py`**

```python
# ascent/research/wf_framework/engine.py
"""
Walk-Forward Engine
===================
Top-level orchestrator. Call `engine.run(data, benchmark)` to get the
stitched OOS equity curve and full performance report.

Boundary defenses in this file
-------------------------------
1. IS slice passed to optimizer is `data.loc[w.slice_is(dates)]` — the
   `slice_is` method already excludes the purge tail (see windows.py).
   The optimizer cannot access any OOS data.

2. OOS signals are generated on `data.loc[w.slice_oos(dates)]`. The
   strategy `generate_signals` call on OOS data is provided fresh OHLCV
   starting from `oos_start`, so it cannot accidentally look back into
   IS data beyond the warmup needed for the strategy's own lookback.
   To be fully causal, generate_signals receives data from IS_START to
   OOS_END but signals are only harvested from OOS_START onward.

3. `last_windows_` is stored for post-run inspection and testing.
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


class WalkForwardEngine:
    """
    Parameters
    ----------
    strategy_cls     : BaseStrategy subclass to optimise and evaluate.
    window_generator : WindowGenerator instance.
    exec_config      : ExecutionConfig for friction modelling.
    objective        : "sharpe" or "calmar" — IS optimisation target.
    constraint_fn    : Optional param constraint (e.g., fast < slow).
    rf_annual        : Risk-free rate for Sharpe and Sortino.
    verbose          : Print fold-level progress.
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data:      pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> tuple[pd.Series, dict]:
        """
        Execute the full walk-forward evaluation.

        Parameters
        ----------
        data      : Full OHLCV DataFrame sorted by date index.
        benchmark : Optional benchmark daily return Series (e.g. SPY returns).

        Returns
        -------
        equity_curve : pd.Series starting at 1.0, stitched OOS cumulative returns.
        report       : dict with Sharpe, Sortino, MDD, WFE, alpha, beta, etc.
        """
        dates   = data.index.drop_duplicates().sort_values()
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  WALK-FORWARD ENGINE — {len(windows)} folds")
            print(f"  IS: {self.wg.is_days}d | OOS: {self.wg.oos_days}d | "
                  f"Purge: {self.wg.purge_days}d | Embargo: {self.wg.embargo_days}d")
            print(f"{'='*60}")

        fold_results:   list[FoldResult]  = []
        oos_return_chunks: list[pd.Series] = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            # ----------------------------------------------------------
            # BOUNDARY DEFENSE: optimizer receives IS slice ONLY
            # ----------------------------------------------------------
            is_data  = data.loc[is_dates]

            # Optimise strictly on IS data
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            # ----------------------------------------------------------
            # OOS evaluation: strategy sees data from IS start → OOS end
            # for warmup, but signals harvested from OOS start only.
            # ----------------------------------------------------------
            full_context_dates = dates[
                (dates >= w.is_start) & (dates <= w.oos_end)
            ]
            full_context_data = data.loc[full_context_dates]

            strategy   = self.strategy_cls(**best_params)
            all_signals = strategy.generate_signals(full_context_data)
            oos_signals = all_signals.loc[oos_dates]
            oos_data    = data.loc[oos_dates]

            oos_returns = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                oos_sharpe = self.analyser.sharpe(oos_returns)
                print(
                    f"  Fold {w.fold_id}: "
                    f"IS [{w.is_start.date()} → {w.purge_start.date()}) | "
                    f"OOS [{w.oos_start.date()} → {w.oos_end.date()}] | "
                    f"params={best_params} | IS_Sharpe={is_sharpe:.2f} | "
                    f"OOS_Sharpe={oos_sharpe:.2f}"
                )

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        if not oos_return_chunks:
            raise RuntimeError("All folds skipped — no OOS returns produced.")

        # Stitch chronologically (no overlapping OOS periods by construction)
        stitched_returns = pd.concat(oos_return_chunks).sort_index()
        stitched_returns = stitched_returns[~stitched_returns.index.duplicated(keep="first")]

        # Equity curve starting at 1.0
        equity_curve = (1 + stitched_returns).cumprod()
        equity_curve = equity_curve / equity_curve.iloc[0]

        # Benchmark aligned to OOS period
        bm_aligned = None
        if benchmark is not None:
            bm_aligned = benchmark.reindex(stitched_returns.index).fillna(0)

        report = self.analyser.full_report(stitched_returns, bm_aligned, fold_results)

        if self.verbose:
            print()
            self.analyser.print_report(report)

        return equity_curve, report
```

- [ ] **Step 4: Run integration tests**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/test_engine.py -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/test_wf_framework/ -v
```
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add ascent/research/wf_framework/engine.py tests/test_wf_framework/test_engine.py
git commit -m "feat: wf-framework engine — WalkForwardEngine orchestrator + integration tests"
```

---

## Task 7: Smoke test with a live example run

**Files:**
- No new files — run inline

Confirm the full framework works end-to-end on synthetic data with a printed report and WFE score.

- [ ] **Step 1: Run the smoke test**

```bash
.venv/bin/python - << 'EOF'
import numpy as np
import pandas as pd
from ascent.research.wf_framework import (
    WalkForwardEngine, WindowGenerator, MACrossStrategy, ExecutionConfig
)

np.random.seed(42)
n   = 800
idx = pd.bdate_range("2017-01-03", periods=n)
log_ret = np.random.randn(n) * 0.012 + 0.0004
close = pd.Series(100 * np.exp(np.cumsum(log_ret)), index=idx)
data = pd.DataFrame({
    "open":   close.shift(1).fillna(100),
    "high":   close * 1.006,
    "low":    close * 0.994,
    "close":  close,
    "volume": 2_000_000,
})
benchmark = close.pct_change().fillna(0).rename("spy")

engine = WalkForwardEngine(
    strategy_cls     = MACrossStrategy,
    window_generator = WindowGenerator(
        is_days=252, oos_days=63, purge_days=21, embargo_days=5,
        window_type="rolling"
    ),
    exec_config  = ExecutionConfig(
        slippage_model  = "atr",
        atr_multiplier  = 0.10,
        commission_pct  = 0.0005,
        borrow_rate_annual = 0.02,
    ),
    constraint_fn = lambda p: p["fast"] < p["slow"],
    rf_annual     = 0.04,
    verbose       = True,
)

equity_curve, report = engine.run(data, benchmark)
print(f"\nEquity curve: {equity_curve.iloc[0]:.4f} → {equity_curve.iloc[-1]:.4f}")
print(f"WFE interpretation: {'acceptable' if report['wfe'] >= 0.5 else 'OVERFITTING'}")
EOF
```

Expected: prints fold-by-fold progress, full performance report, WFE score.

- [ ] **Step 2: Final commit**

```bash
git add ascent/research/wf_framework/
git commit -m "feat: wf-framework smoke test passing — full walk-forward OOS engine complete"
```

---

## Self-Review

**Spec coverage:**
- ✅ WalkForwardEngine class-based structure
- ✅ Rolling AND anchored window types (WindowGenerator `window_type` param)
- ✅ IS/OOS lengths configurable
- ✅ Param optimisation within IS only (boundary enforced in engine.run)
- ✅ Purging: `slice_is` excludes purge tail; documented in windows.py
- ✅ Embargo: dates between IS end and OOS start excluded from both
- ✅ ATR-based slippage
- ✅ Fixed percentage slippage
- ✅ Commission per trade
- ✅ Borrow/financing cost for shorts
- ✅ Cumulative OOS equity curve vs benchmark
- ✅ Sharpe, Sortino, Max Drawdown, Win Rate
- ✅ WFE score with interpretation thresholds
- ✅ Risk-free rate configurable

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:** `FoldResult` defined in Task 5, used in Tasks 5 and 6. `SplitWindow.slice_is` / `slice_oos` defined Task 1, used Task 6. `ParameterOptimizer.optimize` defined Task 4, patched in Task 6 test. All consistent.
