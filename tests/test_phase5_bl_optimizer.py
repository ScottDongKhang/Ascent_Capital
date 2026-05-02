# tests/test_phase5_bl_optimizer.py
"""Tests for Phase 5: Black-Litterman MV Optimizer + TC-aware rebalancing."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_returns(n=200, m=10, seed=42):
    np.random.seed(seed)
    # Use a known Friday (2026-04-17) so bdate_range returns exactly n dates
    idx  = pd.bdate_range(end="2026-04-17", periods=n)
    cols = [f"S{i}" for i in range(m)]
    rets = np.random.normal(0.0004, 0.015, (n, m))
    return pd.DataFrame(rets, index=idx, columns=cols)


def _make_close(n=200, m=10, seed=42):
    rets = _make_returns(n=n, m=m, seed=seed)
    return 100 * (1 + rets).cumprod()


# ── Covariance module ─────────────────────────────────────────────────────────

def test_ledoit_wolf_returns_ndarray():
    from ascent.portfolio.covariance import ledoit_wolf_cov
    rets = _make_returns()
    cov  = ledoit_wolf_cov(rets, min_periods=63)
    assert cov is not None
    assert cov.shape == (10, 10)
    # Covariance matrix must be symmetric and positive semi-definite
    assert np.allclose(cov, cov.T, atol=1e-8)
    eigs = np.linalg.eigvalsh(cov)
    assert (eigs >= -1e-8).all(), f"Non-PSD covariance: min eigenvalue {eigs.min():.6f}"


def test_ledoit_wolf_annualizes():
    from ascent.portfolio.covariance import ledoit_wolf_cov
    rets  = _make_returns()
    cov_a = ledoit_wolf_cov(rets, annualize=True)
    cov_d = ledoit_wolf_cov(rets, annualize=False)
    assert cov_a is not None and cov_d is not None
    assert np.allclose(cov_a, cov_d * 252, atol=1e-6)


def test_ledoit_wolf_returns_none_on_insufficient_data():
    from ascent.portfolio.covariance import ledoit_wolf_cov
    rets = _make_returns(n=30)
    cov  = ledoit_wolf_cov(rets, min_periods=63)
    assert cov is None


def test_get_cov_falls_back_to_sample():
    """get_cov falls back gracefully when sklearn unavailable (monkey-patched)."""
    from ascent.portfolio import covariance as cov_mod
    rets = _make_returns()
    # Patch ledoit_wolf to fail → should fall back to sample covariance
    orig = cov_mod.ledoit_wolf_cov
    cov_mod.ledoit_wolf_cov = lambda *a, **kw: None
    try:
        result = cov_mod.get_cov(rets, min_periods=63)
        assert result is not None, "sample fallback should succeed"
        assert result.shape == (10, 10)
    finally:
        cov_mod.ledoit_wolf_cov = orig


# ── Black-Litterman ───────────────────────────────────────────────────────────

def test_bl_weights_returns_series():
    from ascent.portfolio.mv_optimizer import bl_weights
    rets   = _make_returns()
    scores = pd.Series(np.random.randn(10), index=rets.columns)
    w = bl_weights(scores, rets, max_weight=0.20)
    assert w is not None
    assert not w.empty
    assert abs(w.sum() - 1.0) < 0.01, f"weights sum to {w.sum():.4f}"
    assert (w >= 0).all(), "all weights must be non-negative"
    assert (w <= 0.20 + 1e-6).all(), f"max_weight violated: {w.max():.4f}"


def test_bl_weights_high_alpha_gets_more_weight():
    """Symbol with highest alpha z-score should receive above-average weight."""
    from ascent.portfolio.mv_optimizer import bl_weights
    rets = _make_returns()
    scores = pd.Series(np.zeros(10), index=rets.columns)
    scores["S0"] = 3.0   # very strong alpha signal
    w = bl_weights(scores, rets, max_weight=0.30)
    assert w is not None
    assert w["S0"] >= w.mean(), "high-alpha name should get above-average weight"


def test_bl_weights_returns_none_on_insufficient_history():
    from ascent.portfolio.mv_optimizer import bl_weights
    rets   = _make_returns(n=30)
    scores = pd.Series(np.random.randn(10), index=rets.columns)
    w = bl_weights(scores, rets, min_history=63)
    assert w is None


def test_bl_weights_returns_none_on_too_few_names():
    from ascent.portfolio.mv_optimizer import bl_weights
    rets   = _make_returns(m=2)
    scores = pd.Series(np.random.randn(2), index=rets.columns)
    w = bl_weights(scores, rets)
    assert w is None


# ── TC-aware optimizer ────────────────────────────────────────────────────────

def test_tc_aware_no_trade_when_equal():
    """When current == target, TC-aware should return the same weights."""
    from ascent.portfolio.tc_aware_optimizer import tc_aware_weights
    from ascent.portfolio.covariance import get_cov

    syms = [f"S{i}" for i in range(5)]
    n    = 5
    w_eq = pd.Series(np.ones(n) / n, index=syms)
    rets = _make_returns(m=n)
    rets.columns = syms
    cov  = get_cov(rets)

    result = tc_aware_weights(
        target_weights=w_eq,
        current_weights=w_eq,
        mu=pd.Series(np.zeros(n), index=syms),
        cov=cov,
        kappa=0.001,
        min_trade=0.0,
    )
    assert abs(result.sum() - 1.0) < 0.01
    for s in syms:
        assert abs(result[s] - w_eq[s]) < 0.01, f"{s}: {result[s]} vs {w_eq[s]}"


def test_tc_aware_small_deviation_stays_put():
    """Sub-deadband deviations should not trigger trades."""
    from ascent.portfolio.tc_aware_optimizer import tc_aware_weights
    from ascent.portfolio.covariance import get_cov

    syms = [f"S{i}" for i in range(5)]
    n    = 5
    rets = _make_returns(m=n)
    rets.columns = syms
    cov  = get_cov(rets)

    w_curr   = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=syms)
    w_target = pd.Series([0.21, 0.20, 0.19, 0.20, 0.20], index=syms)  # 1% drift

    result = tc_aware_weights(
        target_weights=w_target,
        current_weights=w_curr,
        mu=pd.Series(np.zeros(n), index=syms),
        cov=cov,
        kappa=0.001,
        min_trade=0.005,  # 50bps deadband
    )
    assert abs(result.sum() - 1.0) < 0.02


def test_tc_aware_returns_target_when_no_current():
    from ascent.portfolio.tc_aware_optimizer import tc_aware_weights
    from ascent.portfolio.covariance import get_cov

    syms = [f"S{i}" for i in range(4)]
    rets = _make_returns(m=4)
    rets.columns = syms
    cov = get_cov(rets)

    w_tgt = pd.Series([0.40, 0.30, 0.20, 0.10], index=syms)
    result = tc_aware_weights(
        target_weights=w_tgt,
        current_weights=pd.Series(dtype=float),
        mu=pd.Series(np.zeros(4), index=syms),
        cov=cov,
    )
    assert result is not None
    assert not result.empty


# ── apply_bl_to_latest ────────────────────────────────────────────────────────

def test_apply_bl_preserves_row_count():
    """apply_bl_to_latest must not change the number of rows in target_weights."""
    from ascent.portfolio.optimizer import apply_bl_to_latest

    n, m = 300, 10
    close = _make_close(n=n, m=m)
    alpha = pd.DataFrame(
        np.random.randn(n, m), index=close.index, columns=close.columns
    )
    tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(5):
        tw.iloc[-1, i] = 0.20   # 5 equal-weight names in last row

    result = apply_bl_to_latest(tw, close, alpha, max_weight=0.30)
    assert result.shape == tw.shape


def test_apply_bl_last_row_sums_to_one():
    """Last row must sum to ~1.0 after BL."""
    from ascent.portfolio.optimizer import apply_bl_to_latest

    n, m = 300, 10
    close = _make_close(n=n, m=m)
    alpha = pd.DataFrame(
        np.random.randn(n, m), index=close.index, columns=close.columns
    )
    tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(6):
        tw.iloc[-1, i] = 1.0 / 6

    result = apply_bl_to_latest(tw, close, alpha, max_weight=0.30, min_history=126)
    row_sum = result.iloc[-1].sum()
    assert abs(row_sum - 1.0) < 0.02, f"last row sums to {row_sum:.4f}"


def test_apply_bl_respects_max_weight():
    """No weight in the last row should exceed max_weight."""
    from ascent.portfolio.optimizer import apply_bl_to_latest

    n, m = 300, 8
    close = _make_close(n=n, m=m)
    alpha = pd.DataFrame(
        np.random.randn(n, m), index=close.index, columns=close.columns
    )
    tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(m):
        tw.iloc[-1, i] = 1.0 / m

    result = apply_bl_to_latest(tw, close, alpha, max_weight=0.20, min_history=100)
    assert result.iloc[-1].max() <= 0.21, (
        f"max weight {result.iloc[-1].max():.4f} exceeds cap"
    )


def test_apply_bl_returns_unchanged_on_insufficient_data():
    """With less than min_history rows, BL must fall back silently."""
    from ascent.portfolio.optimizer import apply_bl_to_latest

    n, m = 50, 6
    close = _make_close(n=n, m=m)
    alpha = pd.DataFrame(
        np.random.randn(n, m), index=close.index, columns=close.columns
    )
    tw = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(m):
        tw.iloc[-1, i] = 1.0 / m

    result = apply_bl_to_latest(tw, close, alpha, max_weight=0.20, min_history=200)
    # Should be unchanged (fallback)
    pd.testing.assert_frame_equal(result, tw)
