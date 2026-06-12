# tests/portfolio/test_exposure.py
"""
Exposure overlay tests — the research/production parity guarantee.

Both ascent/main.py and wf_framework/ascent_strategy.py delegate to
ascent/portfolio/exposure.py; these tests pin the shared math and verify the
WF strategy methods produce identical output to the shared functions.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.exposure import (
    ma_filter_scale,
    vol_target_scale,
    apply_exposure_overlays,
)


def _make_spy(n=400, daily_ret=0.0005, vol=0.0, seed=7, start="2024-01-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    rets = daily_ret + vol * rng.standard_normal(n)
    return pd.Series(100 * np.cumprod(1 + rets), index=dates)


class TestMaFilterScale:
    def test_uptrend_no_cut(self):
        spy = _make_spy(daily_ret=0.001)
        scale = ma_filter_scale(spy, spy.index[-30:])
        assert (scale == 1.0).all()

    def test_downtrend_cuts_without_vix(self):
        spy = _make_spy(n=300, daily_ret=0.001)
        # Crash: 25% drop over the last 60 days pushes price below 200MA
        crash = spy.copy()
        crash.iloc[-60:] = crash.iloc[-61] * np.linspace(1.0, 0.75, 60)
        scale = ma_filter_scale(crash, crash.index[-5:])
        assert (scale == 0.70).all()

    def test_vix_gate_blocks_cut_when_calm(self):
        spy = _make_spy(n=300, daily_ret=0.001)
        crash = spy.copy()
        crash.iloc[-60:] = crash.iloc[-61] * np.linspace(1.0, 0.75, 60)
        calm_vix = pd.Series(14.0, index=crash.index)
        scale = ma_filter_scale(crash, crash.index[-5:], vix_close=calm_vix)
        assert (scale == 1.0).all()

    def test_vix_gate_allows_cut_when_stressed(self):
        spy = _make_spy(n=300, daily_ret=0.001)
        crash = spy.copy()
        crash.iloc[-60:] = crash.iloc[-61] * np.linspace(1.0, 0.75, 60)
        hot_vix = pd.Series(32.0, index=crash.index)
        scale = ma_filter_scale(crash, crash.index[-5:], vix_close=hot_vix)
        assert (scale == 0.70).all()

    def test_insufficient_history_no_cut(self):
        spy = _make_spy(n=100, daily_ret=-0.002)  # < min_periods=150
        scale = ma_filter_scale(spy, spy.index[-5:])
        assert (scale == 1.0).all()


class TestVolTargetScale:
    def test_low_vol_full_exposure(self):
        spy = _make_spy(vol=0.004)  # ~6% annualized < 15% target
        scale = vol_target_scale(spy, spy.index[-10:])
        assert (scale == 1.0).all()  # capped at 1.0

    def test_high_vol_scales_down(self):
        spy = _make_spy(vol=0.03)  # ~48% annualized
        scale = vol_target_scale(spy, spy.index[-10:])
        assert (scale < 1.0).all()
        assert (scale >= 0.25).all()  # floor

    def test_extreme_vol_hits_floor(self):
        spy = _make_spy(vol=0.08)  # ~127% annualized → 0.15/1.27 < 0.25
        scale = vol_target_scale(spy, spy.index[-10:])
        assert (scale == 0.25).all()

    def test_causal_no_lookahead(self):
        """Scale at date d must not change when future prices change."""
        spy = _make_spy(vol=0.01)
        d = spy.index[-20]
        base = vol_target_scale(spy, pd.Index([d])).iloc[0]
        tampered = spy.copy()
        tampered.iloc[-10:] = tampered.iloc[-10:] * 3.0  # wild future move
        after = vol_target_scale(tampered, pd.Index([d])).iloc[0]
        assert base == pytest.approx(after)

    def test_short_history_neutral(self):
        spy = _make_spy(n=4)
        scale = vol_target_scale(spy, spy.index)
        assert (scale == 1.0).all()


class TestApplyExposureOverlays:
    def _weights(self, dates):
        w = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
        w["AAA"] = 0.6
        w["BBB"] = 0.4
        return w

    def test_rebalance_only_holds_scale_between_rebalances(self):
        spy = _make_spy(n=400, vol=0.02)
        dates = spy.index[-40:]
        w = self._weights(dates)
        # Single rebalance at the start: identical rows after row 0
        scaled, _ = apply_exposure_overlays(w, spy)
        # Scale frozen at first row → all rows identical
        assert (scaled.nunique() == 1).all()

    def test_disabled_vol_targeting(self):
        spy = _make_spy(n=400, vol=0.03)
        dates = spy.index[-10:]
        w = self._weights(dates)
        scaled, meta = apply_exposure_overlays(w, spy, vol_targeting_enabled=False)
        assert meta["mean_vol_scale"] == 1.0
        # Only the MA filter may act when vol targeting is off
        ma_only = w.mul(ma_filter_scale(spy, dates), axis=0)
        pd.testing.assert_frame_equal(scaled, ma_only)

    def test_empty_weights_passthrough(self):
        spy = _make_spy()
        w = pd.DataFrame()
        scaled, meta = apply_exposure_overlays(w, spy)
        assert scaled.empty


class TestResearchProductionParity:
    """The WF strategy methods must produce exactly the shared module's output."""

    def _long_data(self, spy):
        return pd.DataFrame({
            "symbol": "SPY",
            "date": spy.index,
            "close": spy.values,
        })

    def test_wf_vol_target_matches_shared(self):
        from ascent.research.wf_framework.ascent_strategy import AscentPortfolioStrategy

        spy = _make_spy(n=300, vol=0.025)
        rebal_dates = spy.index[-30::10]
        w = pd.DataFrame(0.1, index=rebal_dates, columns=["AAA"])

        strat = AscentPortfolioStrategy.__new__(AscentPortfolioStrategy)  # no __init__ side effects
        wf_out = strat._apply_vol_target(self._long_data(spy), w)

        expected_scale = vol_target_scale(spy, rebal_dates)
        expected = w.mul(expected_scale, axis=0)
        pd.testing.assert_frame_equal(wf_out, expected)

    def test_wf_ma_overlay_matches_shared(self, monkeypatch):
        from ascent.research.wf_framework import ascent_strategy as mod
        from ascent.research.wf_framework.ascent_strategy import AscentPortfolioStrategy

        spy = _make_spy(n=300, daily_ret=0.001)
        crash = spy.copy()
        crash.iloc[-60:] = crash.iloc[-61] * np.linspace(1.0, 0.75, 60)
        rebal_dates = crash.index[-5:]
        w = pd.DataFrame(0.1, index=rebal_dates, columns=["AAA"])

        # Force the no-VIX path so the test doesn't depend on the macro cache
        import ascent.portfolio.exposure as exposure_mod
        monkeypatch.setattr(exposure_mod, "load_vix_from_macro_cache", lambda: None)

        strat = AscentPortfolioStrategy.__new__(AscentPortfolioStrategy)
        wf_out = strat._apply_200ma_overlay(self._long_data(crash), w)

        expected_scale = ma_filter_scale(crash, rebal_dates, vix_close=None)
        expected = w.mul(expected_scale, axis=0)
        pd.testing.assert_frame_equal(wf_out, expected)
        assert np.allclose(wf_out["AAA"], 0.07)  # 0.1 × 0.70 — cut actually fired
