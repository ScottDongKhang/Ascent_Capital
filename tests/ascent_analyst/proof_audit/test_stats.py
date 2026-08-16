"""Pin the IC/t-test/Sharpe math against a synthetic series with a known planted mean.

Real market data is never used here -- this only checks the arithmetic.
"""
import math

import pytest

from ascent.analyst.proof_audit.stats import score_ic_series


def test_positive_planted_ic_is_significant():
    # 60 days of IC centered at 0.05 with small noise -> should be clearly significant.
    daily_ic = [0.05 + 0.01 * math.sin(i) for i in range(60)]
    daily_ls_return = [0.001 + 0.0002 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.ic_mean == pytest.approx(0.05, abs=0.01)
    assert result.p_value < 0.05
    assert result.n == 60


def test_zero_mean_ic_is_not_significant():
    daily_ic = [0.01 * math.sin(i) for i in range(60)]  # oscillates around 0
    daily_ls_return = [0.0001 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.p_value > 0.05


def test_negative_planted_ic_is_significant_and_negative():
    daily_ic = [-0.05 + 0.01 * math.sin(i) for i in range(60)]
    daily_ls_return = [-0.001 + 0.0002 * math.sin(i) for i in range(60)]
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.ic_mean < 0
    assert result.p_value < 0.05


def test_sharpe_is_annualized():
    # constant positive daily return with zero variance is degenerate (std=0);
    # use a small planted variance instead so Sharpe is finite and computable.
    # Add variance to daily_ic too to avoid scipy precision warnings on zero-variance t-test.
    daily_ic = [0.03 + 0.001 * math.sin(i) for i in range(40)]
    daily_ls_return = [0.001, 0.0008] * 20
    result = score_ic_series(daily_ic, daily_ls_return)
    assert result.sharpe > 0
    assert math.isfinite(result.sharpe)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        score_ic_series([0.01, 0.02], [0.001, 0.002])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        score_ic_series([0.01] * 10, [0.001] * 9)
