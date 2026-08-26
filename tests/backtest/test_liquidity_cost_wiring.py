# tests/backtest/test_liquidity_cost_wiring.py
"""
Call-site wiring test for the liquidity-scaled cost model.

A code review found that `liquidity_scaled_cost_model()` /
`BacktestEngine.run()`'s `volume=` parameter (both added in an earlier
session, together with `BacktestConfig.adv_lookback_days` /
`impact_floor_mult` / `impact_ceil_mult`) were completely dead in
production: neither of the two real `BacktestEngine.run()` call sites
(`ascent/main.py`'s daily pipeline, `ascent/research/walk_forward_runner.py`'s
canonical walk-forward) ever passed `volume=`, so the flat-cost fallback
always fired and the ADV-scaled cost model had only ever run inside its own
unit tests (`tests/backtest/test_liquidity_cost_model.py`).

This file does NOT re-test the cost math itself (that's already covered).
It only asserts that both production call sites now actually pass a
non-None `volume=` panel sourced from `FeatureBuilder.volume` — i.e. that
the wiring in this session's fix is real and didn't regress.

Both `run_pipeline()` (ascent/main.py) and `run_walk_forward_backtest()` /
`walk_forward_pipeline()` (ascent/research/walk_forward_runner.py) are too
heavy to execute end-to-end in a unit test (same rationale as
`TestWiredIntoPipeline` in tests/test_main_mrr_integration.py), so this
follows that file's established pattern: assert directly on source text
that the call sites are wired the way we mean them to be, plus one
executable regression against `BacktestEngine.run()` itself confirming a
plain (non-lagged, non-ADV) volume panel shaped like `close_prices` is
exactly what the engine expects from a `FeatureBuilder.volume` panel.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ascent.backtest.engine import BacktestEngine


def _root_src(relpath: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, relpath)) as f:
        return f.read()


class TestMainPyWiring:
    """ascent/main.py's production BacktestEngine.run() call site."""

    def test_volume_pulled_from_builder(self):
        src = _root_src("ascent/main.py")
        assert "volume   = builder.volume" in src or "volume = builder.volume" in src, (
            "ascent/main.py must pull a volume panel from the same "
            "FeatureBuilder instance used for close/open prices"
        )

    def test_engine_run_call_site_passes_volume(self):
        src = _root_src("ascent/main.py")
        idx = src.index("result = engine.run(")
        call_block = src[idx: idx + 400]
        assert "volume=volume" in call_block, (
            "ascent/main.py's engine.run() call must pass volume=volume — "
            "without this the liquidity-scaled cost model never engages in "
            "the daily production pipeline and silently falls back to flat "
            "costs"
        )


class TestWalkForwardRunnerWiring:
    """ascent/research/walk_forward_runner.py's canonical OOS backtest call site."""

    def test_volume_pulled_from_full_builder(self):
        src = _root_src("ascent/research/walk_forward_runner.py")
        assert "volume_full  = full_builder.volume" in src or "volume_full = full_builder.volume" in src, (
            "walk_forward_runner.py must pull a volume panel from the same "
            "full_builder instance used for close_full/open_full"
        )

    def test_engine_run_call_site_passes_volume(self):
        src = _root_src("ascent/research/walk_forward_runner.py")
        idx = src.index("result = engine.run(")
        call_block = src[idx: idx + 400]
        assert "volume=volume_full" in call_block, (
            "walk_forward_runner.py's engine.run() call must pass "
            "volume=volume_full — without this the canonical walk-forward "
            "never exercises the liquidity-scaled cost model and its "
            "Sharpe/CAGR numbers reflect flat costs only"
        )


class TestFeatureBuilderVolumeShapeMatchesEngineExpectation:
    """Executable regression: a plain FeatureBuilder-shaped volume panel
    (same index/columns as close_prices, raw share volume — NOT
    pre-lagged, NOT pre-multiplied into dollar ADV) is what
    BacktestEngine.run() expects it to derive ADV-dollar from internally.
    """

    def test_engine_computes_adv_and_diverges_from_flat_cost(self):
        dates = pd.bdate_range("2025-01-01", periods=40)
        syms = ["AAA", "BBB"]
        rng = np.random.default_rng(0)

        close = pd.DataFrame(
            {s: 100 * np.cumprod(1 + rng.normal(0.0, 0.01, len(dates))) for s in syms},
            index=dates,
        )
        open_ = close.shift(1).fillna(close.iloc[0])

        # BBB is far thinner than AAA -> its impact cost should scale up
        # once a large rebalance hits it, relative to the flat model.
        volume = pd.DataFrame(
            {"AAA": 5_000_000.0, "BBB": 5_000.0}, index=dates
        )

        weights = pd.DataFrame(0.0, index=dates, columns=syms)
        weights.loc[dates[5]:, "AAA"] = 0.5
        weights.loc[dates[5]:, "BBB"] = 0.5

        engine = BacktestEngine(
            initial_capital=1_000_000.0,
            spread_bps=5.0,
            impact_bps=5.0,
            rebalance_freq_days=5,
            execution_delay=1,
        )

        flat_result = engine.run(weights, close, open_, volume=None)
        adv_result = engine.run(weights, close, open_, volume=volume)

        assert not flat_result.costs.equals(adv_result.costs), (
            "supplying a volume panel with a thin symbol (BBB) must "
            "produce different per-rebalance costs than the flat model"
        )
