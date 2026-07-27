# tests/portfolio/test_momentum_crash_overlay.py
"""
Momentum crash indicator — Daniel & Moskowitz (2016).

Momentum crashes cluster in panic states: a prolonged market decline
followed by a rebound. In that state, cut exposure.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.portfolio.exposure import (
    CRASH_MULTIPLIER,
    momentum_crash_scale,
)


def _series(values, start="2023-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _path(n_down: int, down_rate: float, n_up: int, up_rate: float):
    """A decline of n_down days then a rebound of n_up days."""
    vals = [100.0]
    for _ in range(n_down):
        vals.append(vals[-1] * (1 + down_rate))
    for _ in range(n_up):
        vals.append(vals[-1] * (1 + up_rate))
    return _series(vals)


class TestMomentumCrashScale:
    def test_bear_plus_rebound_cuts_exposure(self):
        """The crash state: 2y cumulative negative, recent window positive."""
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(0.5)

    def test_bear_without_rebound_does_not_cut(self):
        """Still falling is not the crash state — the 200MA cut owns that."""
        spy = _path(n_down=600, down_rate=-0.001, n_up=0, up_rate=0.0)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(1.0)

    def test_rebound_without_bear_does_not_cut(self):
        """A rally in an ongoing bull is not a crash state."""
        spy = _path(n_down=0, down_rate=0.0, n_up=600, up_rate=0.001)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), bear_lookback=504,
                                   rebound_lookback=21, multiplier=0.5)
        assert out.iloc[0] == pytest.approx(1.0)

    def test_calm_bull_is_never_cut(self):
        """Regression for the 2026-06/07 window: this must NOT fire."""
        rng = np.random.default_rng(4)
        spy = _series(100 * np.cumprod(
            1 + rng.normal(0.0004, 0.008, 700)))
        out = momentum_crash_scale(spy, spy.index[-30:], multiplier=0.5)
        assert (out == 1.0).all()

    def test_is_causal_future_data_does_not_leak(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[400]
        base = momentum_crash_scale(spy, pd.Index([d])).iloc[0]

        tampered = spy.copy()
        tampered.iloc[401:] = tampered.iloc[401:] * 5.0
        after = momentum_crash_scale(tampered, pd.Index([d])).iloc[0]
        assert base == pytest.approx(after)

    def test_insufficient_history_returns_one(self):
        spy = _series([100.0, 99.0, 101.0])
        out = momentum_crash_scale(spy, spy.index, bear_lookback=504)
        assert (out == 1.0).all()

    def test_multiplier_is_configurable(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        d = spy.index[-1]
        out = momentum_crash_scale(spy, pd.Index([d]), multiplier=0.25)
        assert out.iloc[0] == pytest.approx(0.25)

    def test_multiplier_one_is_a_noop(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        out = momentum_crash_scale(spy, spy.index[-5:], multiplier=1.0)
        assert (out == 1.0).all()

    def test_empty_dates_returns_empty(self):
        spy = _path(n_down=10, down_rate=-0.001, n_up=5, up_rate=0.002)
        assert momentum_crash_scale(spy, pd.Index([])).empty

    def test_duplicate_index_entries_are_tolerated(self):
        spy = _path(n_down=560, down_rate=-0.001, n_up=25, up_rate=0.004)
        dupd = pd.concat([spy, spy.iloc[-3:]]).sort_index()
        out = momentum_crash_scale(dupd, pd.Index([spy.index[-1]]))
        assert out.iloc[0] in (0.5, CRASH_MULTIPLIER)
