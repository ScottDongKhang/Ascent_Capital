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
        verbose          = True,
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


def test_optimizer_receives_is_data_only(multi_ohlcv):
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
