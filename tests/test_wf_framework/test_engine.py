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
    np.random.seed(99)
    n   = 600
    idx = pd.bdate_range("2018-01-02", periods=n)
    log_returns = np.random.randn(n) * 0.01 + 0.0003
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
    from unittest.mock import patch
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
        engine.run(synthetic_ohlcv)

    windows = engine.last_windows_
    for i, w in enumerate(windows):
        if i < len(max_dates_seen):
            assert max_dates_seen[i] < w.purge_start, \
                f"Fold {i}: optimizer received data past IS end"
