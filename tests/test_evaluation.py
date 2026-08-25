# tests/test_evaluation.py
"""
Tests for ascent/research/evaluation.py, focused on the wipeout-clamp bug:

annualized_return() used to clamp to 0.0 whenever the cumulative product of
(1+returns) fell to <= 0 (e.g. a single -100% return day). calmar_ratio()
then divided that clamped 0.0 by a near-total max_drawdown, producing
Calmar == 0.0 -- indistinguishable from (and ranking above) a genuinely flat
series that legitimately scores 0.0. This is a promotion-ranking bug: a
wipeout variant must score as the worst outcome, not a neutral one.
"""
import numpy as np
import pandas as pd
import pytest

from ascent.research.evaluation import (
    annualized_return,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def _series(values):
    idx = pd.bdate_range("2026-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_annualized_return_wipeout_is_negative_not_zero():
    """A -100% day must annualize to -1.0 (total loss), not clamp to 0.0."""
    returns = _series([0.01, 0.02, -1.0, 0.01, -0.005])
    result = annualized_return(returns)
    assert result == -1.0


def test_annualized_return_flat_series_is_zero():
    """A genuinely flat all-zero return series must still score exactly 0.0."""
    returns = _series([0.0] * 20)
    assert annualized_return(returns) == 0.0


def test_annualized_return_normal_series_is_unaffected():
    """Normal, non-wipeout returns compound as before (no total <= 0)."""
    returns = _series([0.001] * 252)
    result = annualized_return(returns)
    assert result == pytest.approx((1.001 ** 252) - 1)


def test_calmar_ratio_wipeout_scores_strongly_negative():
    """
    A variant with an otherwise-normal return series that includes one
    catastrophic -100% day must score Calmar strongly negative -- it must
    NOT be 0.0 (the old buggy behavior) and must NOT be positive.
    """
    np.random.seed(0)
    normal_days = list(np.random.normal(0.001, 0.005, 60))
    returns = _series(normal_days[:30] + [-1.0] + normal_days[30:])
    calmar = calmar_ratio(returns)
    assert calmar < -0.5, f"wipeout Calmar should be strongly negative, got {calmar}"


def test_calmar_ratio_flat_series_stays_zero():
    """A flat all-zero series (no drawdown, no return) must still be ~0.0."""
    returns = _series([0.0] * 50)
    assert calmar_ratio(returns) == 0.0


def test_calmar_ratio_ranks_wipeout_worse_than_modest_drawdown_worse_than_flat():
    """
    Ranking sanity check for the promotion path
    (max(results, key=lambda x: x['oos_calmar'])):
    wipeout < modest-drawdown < flat.

    Both non-flat series use a full 252-day window (periods_per_year=252,
    so the CAGR exponent is exactly 1 and doesn't amplify short-window
    noise) and an explicit day-0 zero-return to establish a real peak
    before the drawdown, so max_drawdown() has something to measure
    against.
    """
    flat = _series([0.0] * 252)

    # Modest, non-catastrophic drawdown: -15% dip, partial recovery to a
    # -5% total/annualized return. Real loss, but nowhere near a wipeout.
    modest_returns = [0.0, -0.15] + [0.0] * 249
    modest_returns.append(0.95 / 0.85 - 1)  # recover cum value to 0.95
    modest_dd = _series(modest_returns)
    assert len(modest_dd) == 252

    # Total wipeout on day 1: cumulative product hits exactly 0.
    wipeout_returns = [0.0, -1.0] + [0.0] * 250
    wipeout = _series(wipeout_returns)
    assert len(wipeout) == 252

    c_flat = calmar_ratio(flat)
    c_modest = calmar_ratio(modest_dd)
    c_wipeout = calmar_ratio(wipeout)

    assert c_flat == 0.0
    assert c_modest == pytest.approx(-0.05 / 0.15, rel=1e-6)
    assert c_wipeout == -1.0

    assert c_wipeout < c_modest, (
        f"wipeout ({c_wipeout}) must rank strictly worse than a modest "
        f"drawdown ({c_modest})"
    )
    assert c_modest < c_flat, (
        f"a real drawdown ({c_modest}) should rank worse than flat ({c_flat})"
    )
    assert c_wipeout < c_flat, (
        f"wipeout ({c_wipeout}) must rank strictly worse than flat ({c_flat})"
    )


def test_sharpe_ratio_wipeout_is_strongly_negative():
    """Sharpe must not silently clamp a wipeout to 0.0 either."""
    np.random.seed(2)
    normal_days = list(np.random.normal(0.001, 0.01, 40))
    returns = _series(normal_days[:20] + [-1.0] + normal_days[20:])
    sharpe = sharpe_ratio(returns)
    assert sharpe < 0


def test_sortino_ratio_wipeout_is_strongly_negative():
    """Sortino must not silently clamp a wipeout to 0.0 either."""
    np.random.seed(3)
    normal_days = list(np.random.normal(0.001, 0.01, 40))
    returns = _series(normal_days[:20] + [-1.0] + normal_days[20:])
    sortino = sortino_ratio(returns)
    assert sortino < 0
