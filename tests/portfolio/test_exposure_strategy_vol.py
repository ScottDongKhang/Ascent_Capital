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
