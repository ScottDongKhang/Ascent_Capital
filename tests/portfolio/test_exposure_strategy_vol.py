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
