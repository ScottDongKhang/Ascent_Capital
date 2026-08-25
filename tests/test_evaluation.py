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
    lo_adjusted_sharpe_ratio,
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


# --- Lo (2002) autocorrelation-adjusted Sharpe ------------------------------
#
# lo_adjusted_sharpe_ratio() is an ADDITIONAL metric next to sharpe_ratio(),
# not a replacement. These tests hand-verify the correction against an
# independently computed lag-1 sample autocorrelation (via np.corrcoef,
# a different code path than the function's own pd.Series.autocorr call),
# check that near-zero autocorrelation leaves the naive Sharpe effectively
# unchanged, and check the zero-variance edge case doesn't crash.


def test_lo_adjusted_sharpe_matches_hand_computed_known_autocorr():
    """
    Small, deterministic 5-point return series with q=2 (only lag-1 enters
    the correction: 2*(1-1/2)*rho_1 = rho_1). rho_1 is computed independently
    here via np.corrcoef on the lag-1-shifted pairs (NOT by calling the
    function's own autocorr machinery), giving a genuine hand-verified
    known autocorrelation coefficient to check the formula against:

        sharpe_corrected = sharpe_naive / sqrt(1 + 2*sum_{k=1}^{q-1}(1-k/q)*rho_k)
                          = sharpe_naive / sqrt(1 + rho_1)   [for q=2]
    """
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.00])
    q = 2

    naive = sharpe_ratio(returns, periods_per_year=252)

    x = returns.values[1:]
    y = returns.values[:-1]
    rho1 = np.corrcoef(x, y)[0, 1]
    expected = naive / np.sqrt(1 + rho1)

    result = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=q)
    assert result == pytest.approx(expected, rel=1e-9)


def test_lo_adjusted_sharpe_is_noop_under_zero_autocorrelation():
    """
    A return series specifically constructed to have ~zero lag-1..lag-(q-1)
    sample autocorrelation should leave the Lo correction close to a no-op:
    lo_adjusted_sharpe_ratio ~= sharpe_ratio.
    """
    np.random.seed(42)
    returns = _series(np.random.normal(0.0008, 0.01, 2000))
    q = 10

    naive = sharpe_ratio(returns, periods_per_year=252)
    adjusted = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=q)

    assert adjusted == pytest.approx(naive, rel=0.1)


def test_lo_adjusted_sharpe_positive_autocorrelation_shrinks_sharpe():
    """
    Sanity check on the strategy's actual mechanism: a rebalance-induced
    return series (same value repeated for a block of days, mimicking
    forward-filled weights between rebalances) has strong positive serial
    correlation, and the Lo-adjusted Sharpe must come out LOWER in magnitude
    than the naive Sharpe -- matching Lo (2002)'s finding that naive
    annualization overstates Sharpe under positive autocorrelation.
    """
    np.random.seed(7)
    block_rets = np.random.normal(0.002, 0.003, 60)
    # Forward-fill each "rebalance" return for 10 days, like the strategy's
    # actual rebalance_freq_days=10 cadence.
    returns = _series(np.repeat(block_rets, 10))

    naive = sharpe_ratio(returns, periods_per_year=252)
    adjusted = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=10)

    assert naive > 0
    assert abs(adjusted) < abs(naive)


def test_lo_adjusted_sharpe_zero_variance_does_not_crash():
    """
    Zero-variance returns must fall through to sharpe_ratio()'s own vol==0
    handling (0.0), not NaN/crash/divide-by-zero. Uses an all-zero series
    (exact float 0.0 std) rather than a constant nonzero series, since a
    constant nonzero series can pick up nonzero float-rounding noise in
    .std() -- a pre-existing property of sharpe_ratio()'s own `vol == 0`
    check that this function intentionally mirrors, not something this
    function is responsible for tightening.
    """
    returns = _series([0.0] * 30)
    result = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=10)
    assert result == 0.0


def test_lo_adjusted_sharpe_too_few_observations_falls_back_to_naive():
    """n <= q: not enough data to trust the correction -> return naive Sharpe."""
    returns = _series([0.01, -0.02, 0.015, 0.005, -0.01])
    naive = sharpe_ratio(returns, periods_per_year=252)
    result = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=10)
    assert result == naive


def test_lo_adjusted_sharpe_default_q_is_ten():
    """No explicit q -> defaults to 10 (documented rebalance-cadence proxy)."""
    np.random.seed(11)
    returns = _series(np.random.normal(0.001, 0.01, 100))
    result_default = lo_adjusted_sharpe_ratio(returns, periods_per_year=252)
    result_explicit = lo_adjusted_sharpe_ratio(returns, periods_per_year=252, q=10)
    assert result_default == result_explicit
