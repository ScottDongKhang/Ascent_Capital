# tests/backtest/test_liquidity_cost_model.py
"""
Liquidity-scaled impact cost model.

Before this, BacktestEngine charged a flat spread_bps + impact_bps on every
trade regardless of size or symbol liquidity — a $50k trade in AAPL and a
$50k trade in a thin small-cap were charged identically. This adds an
optional `volume` panel to BacktestEngine.run(); when supplied, impact cost
scales with sqrt(trade_notional / ADV_dollar) per symbol
(ascent/backtest/costs.py::liquidity_scaled_cost_model). When not supplied,
behavior must stay byte-identical to the old flat model.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.backtest.costs import flat_cost_model, liquidity_scaled_cost_model
from ascent.backtest.engine import BacktestEngine


def _synthetic_market(n_days: int = 60, n_syms: int = 2, seed: int = 0):
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_syms)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {s: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days)) for s in syms},
        index=dates,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    return close, open_, dates, syms


class TestLiquidityScaledCostModelUnit:
    """Unit tests directly against liquidity_scaled_cost_model()."""

    def test_large_trade_in_low_adv_symbol_costs_more_bps_than_small_trade(self):
        # Same symbol, same ADV; a much bigger trade should cost more bps.
        adv = pd.Series({"AAA": 1_000_000.0})  # $1M ADV — thin name
        portfolio_value = 10_000_000.0

        small_delta = pd.Series({"AAA": 0.01})   # $100k trade, 10% of ADV
        large_delta = pd.Series({"AAA": 0.30})   # $3M trade, 300% of ADV

        small_cost = liquidity_scaled_cost_model(
            small_delta, portfolio_value, adv, spread_bps=5.0, impact_bps=5.0,
        )
        large_cost = liquidity_scaled_cost_model(
            large_delta, portfolio_value, adv, spread_bps=5.0, impact_bps=5.0,
        )

        small_bps = small_cost / (small_delta.abs().sum() / 2) * 10_000
        large_bps = large_cost / (large_delta.abs().sum() / 2) * 10_000

        assert large_bps > small_bps, (
            f"large trade ({large_bps:.2f}bps) should cost more per unit "
            f"turnover than the small trade ({small_bps:.2f}bps) in the same "
            "low-ADV symbol"
        )

    def test_same_notional_costs_more_in_lower_adv_symbol(self):
        portfolio_value = 10_000_000.0
        delta = pd.Series({"THIN": 0.05, "LIQUID": 0.05})  # $500k each
        adv = pd.Series({"THIN": 500_000.0, "LIQUID": 50_000_000.0})

        cost = liquidity_scaled_cost_model(delta, portfolio_value, adv)

        thin_only = liquidity_scaled_cost_model(
            pd.Series({"THIN": 0.05}), portfolio_value, adv[["THIN"]],
        )
        liquid_only = liquidity_scaled_cost_model(
            pd.Series({"LIQUID": 0.05}), portfolio_value, adv[["LIQUID"]],
        )
        assert thin_only > liquid_only

    def test_flat_fallback_matches_flat_cost_model_when_adv_none(self):
        delta = pd.Series({"A": 0.1, "B": -0.05})
        cost = liquidity_scaled_cost_model(delta, 1_000_000.0, None, spread_bps=5.0, impact_bps=5.0)
        turnover = float(delta.abs().sum() / 2)
        expected = flat_cost_model(turnover, 10.0)
        assert cost == pytest.approx(expected)

    def test_flat_fallback_per_symbol_when_adv_missing_or_zero(self):
        delta = pd.Series({"KNOWN": 0.1, "NO_ADV": 0.1, "ZERO_ADV": 0.1})
        adv = pd.Series({"KNOWN": 5_000_000.0, "ZERO_ADV": 0.0})  # NO_ADV absent entirely

        cost = liquidity_scaled_cost_model(delta, 1_000_000.0, adv, spread_bps=5.0, impact_bps=5.0)

        # NO_ADV and ZERO_ADV should each fall back to flat (spread+impact)=10bps,
        # KNOWN gets some impact-scaled value. Just check the total is finite,
        # positive, and bounded above by an all-flat-at-ceiling estimate.
        assert np.isfinite(cost)
        assert cost > 0
        ceiling = flat_cost_model(float(delta.abs().sum() / 2), 5.0 + 5.0 * 10.0)
        assert cost <= ceiling

    def test_cost_is_bounded_and_never_negative_or_infinite(self):
        portfolio_value = 10_000_000.0
        delta = pd.Series({"HUGE_TRADE": 1.0, "ZERO_ADV": 0.5, "NEG_ADV": 0.5})
        adv = pd.Series({"HUGE_TRADE": 1.0, "ZERO_ADV": 0.0, "NEG_ADV": -100.0})

        cost = liquidity_scaled_cost_model(
            delta, portfolio_value, adv, spread_bps=5.0, impact_bps=5.0,
            impact_floor_mult=0.1, impact_ceil_mult=10.0,
        )
        assert np.isfinite(cost)
        assert cost >= 0
        # Upper bound: every symbol capped at spread + impact*ceil_mult bps.
        max_possible_bps = 5.0 + 5.0 * 10.0
        ceiling = flat_cost_model(float(delta.abs().sum() / 2), max_possible_bps)
        assert cost <= ceiling + 1e-12

    def test_zero_trade_costs_nothing(self):
        delta = pd.Series({"A": 0.0, "B": 0.0})
        adv = pd.Series({"A": 1_000_000.0, "B": 1_000_000.0})
        cost = liquidity_scaled_cost_model(delta, 1_000_000.0, adv)
        assert cost == pytest.approx(0.0)


class TestBacktestEngineVolumeIntegration:
    """Engine-level: volume=None must reproduce old flat-cost behavior exactly."""

    def test_no_volume_matches_pre_existing_flat_cost_behavior(self):
        close, open_, dates, syms = _synthetic_market()
        tw = pd.DataFrame(0.5, index=dates, columns=syms)  # fully invested

        eng = BacktestEngine(rebalance_freq_days=21, execution_delay=1,
                              spread_bps=5.0, impact_bps=5.0)
        res_no_volume = eng.run(tw, close, open_)  # no volume kwarg at all

        eng2 = BacktestEngine(rebalance_freq_days=21, execution_delay=1,
                               spread_bps=5.0, impact_bps=5.0)
        res_explicit_none = eng2.run(tw, close, open_, volume=None)

        pd.testing.assert_series_equal(res_no_volume.costs, res_explicit_none.costs)

        # And costs match flat_cost_model directly on the recorded turnover.
        for dt, turn in res_no_volume.turnover.items():
            if turn > 0:
                expected = flat_cost_model(turn, 10.0)
                assert res_no_volume.costs.loc[dt] == pytest.approx(expected)

    def test_low_volume_symbol_costs_more_than_flat_baseline_when_trade_is_large(self):
        close, open_, dates, syms = _synthetic_market(n_syms=2)
        tw = pd.DataFrame(0.5, index=dates, columns=syms)

        # Very thin volume relative to a $1M portfolio rebalancing into 100%
        # gross exposure — the impact term should dominate and push the
        # ADV-scaled cost above the flat baseline.
        volume = pd.DataFrame(1_000.0, index=dates, columns=syms)  # ~$100k ADV/day

        flat_eng = BacktestEngine(rebalance_freq_days=21, execution_delay=1,
                                   spread_bps=5.0, impact_bps=5.0,
                                   initial_capital=1_000_000.0)
        res_flat = flat_eng.run(tw, close, open_)

        scaled_eng = BacktestEngine(rebalance_freq_days=21, execution_delay=1,
                                     spread_bps=5.0, impact_bps=5.0,
                                     initial_capital=1_000_000.0)
        res_scaled = scaled_eng.run(tw, close, open_, volume=volume)

        first_cost_flat = res_flat.costs[res_flat.costs > 0].iloc[0]
        first_cost_scaled = res_scaled.costs[res_scaled.costs > 0].iloc[0]
        assert first_cost_scaled > first_cost_flat

    def test_engine_costs_stay_finite_with_ragged_volume_panel(self):
        """Volume panel missing a symbol / with zeros must not blow up or NaN."""
        close, open_, dates, syms = _synthetic_market(n_syms=3)
        tw = pd.DataFrame(1 / 3, index=dates, columns=syms)

        volume = pd.DataFrame(0.0, index=dates, columns=syms[:2])  # zero + missing col

        eng = BacktestEngine(rebalance_freq_days=10, execution_delay=1)
        res = eng.run(tw, close, open_, volume=volume)

        assert not res.costs.isna().any()
        assert np.isfinite(res.costs).all()
        assert (res.costs >= 0).all()
        assert not res.portfolio_returns.isna().any()
