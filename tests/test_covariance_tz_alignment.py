"""tests/test_covariance_tz_alignment.py

Regression tests for the W1-sizing covariance bug:
`build_factor_covariance_matrix` -> `compute_residual_variances` intersected a
tz-aware `prices_live` index (America/New_York) against a tz-naive
`factor_returns` index, producing an EMPTY intersection, feeding LedoitWolf 0
rows, and silently poisoning the covariance diagonal with NaN.

All fixtures are synthetic — no dependency on the live cache.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ascent.risk.covariance_model import (
    _to_naive_calendar_index,
    build_factor_covariance_matrix,
    compute_residual_variances,
)
from ascent.risk.factor_model import BETA_COLS


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_tz_aware_prices(n_syms=8, n_days=300, tz="America/New_York"):
    """Synthetic close prices with a tz-aware DatetimeIndex (mirrors prices_live)."""
    np.random.seed(11)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days, tz=tz)
    syms = [f"S{i:02d}" for i in range(n_syms)]
    data = 100 * np.cumprod(1 + np.random.normal(0, 0.015, (n_days, n_syms)), axis=0)
    return pd.DataFrame(data, index=idx, columns=syms)


def _make_naive_factor_returns(n_days=300):
    """Synthetic factor returns with a tz-naive DatetimeIndex (mirrors Fama-French)."""
    np.random.seed(13)
    idx = pd.bdate_range(end="2026-04-30", periods=n_days)  # tz-naive
    data = np.random.normal(0, 0.01, (n_days, 7))
    return pd.DataFrame(
        data, index=idx,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"],
    )


def _make_loadings(syms):
    np.random.seed(17)
    data = np.random.normal(0, 0.3, (len(syms), 6))
    return pd.DataFrame(data, index=syms, columns=BETA_COLS)


def _make_weights(syms):
    w = np.ones(len(syms)) / len(syms)
    return pd.Series(w, index=syms)


# ── _to_naive_calendar_index ─────────────────────────────────────────────────

def test_to_naive_calendar_index_strips_tz_and_matches_naive_day():
    """A tz-aware index and a tz-naive index for the same calendar days must
    become equal (and therefore intersect) after normalization."""
    aware = pd.bdate_range("2026-01-05", periods=10, tz="America/New_York")
    naive = pd.bdate_range("2026-01-05", periods=10)

    norm_aware = _to_naive_calendar_index(aware)
    norm_naive = _to_naive_calendar_index(naive)

    assert norm_aware.tz is None
    assert norm_naive.tz is None
    assert list(norm_aware) == list(norm_naive)


def test_to_naive_calendar_index_idempotent_on_already_naive():
    naive = pd.bdate_range("2026-01-05", periods=5)
    assert list(_to_naive_calendar_index(naive)) == list(naive)


# ── compute_residual_variances: tz alignment ────────────────────────────────

def test_residual_variances_align_tz_aware_prices_with_tz_naive_factors():
    """Reproduces the exact reported bug: tz-aware prices vs tz-naive factor
    returns must NOT produce an empty intersection / all-NaN residual variances."""
    syms = [f"S{i:02d}" for i in range(8)]
    prices = _make_tz_aware_prices(n_syms=8, n_days=300)
    factors = _make_naive_factor_returns(n_days=300)
    loadings = _make_loadings(syms)

    assert prices.index.tz is not None
    assert factors.index.tz is None
    # Sanity: the raw (un-normalized) intersection is empty — this is the bug.
    assert len(prices.index.intersection(factors.index)) == 0

    D = compute_residual_variances(prices, loadings, factors, lookback=252)

    assert D.shape == (8,)
    assert np.all(np.isfinite(D))
    assert np.all(D > 0)


# ── build_factor_covariance_matrix: end-to-end ───────────────────────────────

def _patch_prices_live(prices_df):
    """Patch the local import target inside covariance_model's function body."""
    return patch("ascent.data.store.parquet.has_data", return_value=True), \
        patch("ascent.data.store.parquet.load_parquet", return_value=prices_df)


def test_build_factor_covariance_matrix_normal_case_is_finite_and_psd():
    syms = [f"S{i:02d}" for i in range(8)]
    prices = _make_tz_aware_prices(n_syms=8, n_days=300)
    factors = _make_naive_factor_returns(n_days=300)
    loadings = _make_loadings(syms)
    weights = _make_weights(syms)

    p1, p2 = _patch_prices_live(prices)
    with p1, p2, \
         patch("ascent.risk.covariance_model.get_factor_loadings", return_value=loadings), \
         patch("ascent.risk.covariance_model.get_factor_returns", return_value=factors):
        result = build_factor_covariance_matrix(weights, as_of_date="2026-04-30")

    Sigma = result["full"]
    assert Sigma.shape == (8, 8)
    assert np.all(np.isfinite(Sigma))
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-8).all(), f"Not PSD: min eigenvalue {eigvals.min()}"
    # Not the degraded-fallback diagonal proxy — off-diagonal structure exists.
    assert not np.allclose(Sigma, np.eye(8) * 0.04)


def test_build_factor_covariance_matrix_empty_intersection_falls_back_clean():
    """Prices and factor returns share zero overlapping calendar days even
    after tz normalization (disjoint date ranges) — must degrade to the
    clean identity fallback, not silently return NaN."""
    syms = [f"S{i:02d}" for i in range(6)]
    # Prices window is entirely in the past relative to factor returns.
    prices = _make_tz_aware_prices(n_syms=6, n_days=50)
    prices.index = pd.bdate_range(end="2020-01-31", periods=50, tz="America/New_York")
    factors = _make_naive_factor_returns(n_days=300)  # ends 2026-04-30
    loadings = _make_loadings(syms)
    weights = _make_weights(syms)

    p1, p2 = _patch_prices_live(prices)
    with p1, p2, \
         patch("ascent.risk.covariance_model.get_factor_loadings", return_value=loadings), \
         patch("ascent.risk.covariance_model.get_factor_returns", return_value=factors):
        result = build_factor_covariance_matrix(weights, as_of_date="2026-04-30")

    Sigma = result["full"]
    assert np.all(np.isfinite(Sigma)), "Empty intersection must not poison Sigma with NaN"
    assert np.allclose(Sigma, np.eye(6) * 0.04), "Expected clean identity fallback"


def test_build_factor_covariance_matrix_guards_against_poisoned_residuals():
    """Even if compute_residual_variances somehow returns NaN (e.g. a future
    regression reintroduces bad alignment), the post-condition guard on
    build_factor_covariance_matrix must catch it and return the clean
    fallback rather than propagate NaN."""
    syms = [f"S{i:02d}" for i in range(5)]
    prices = _make_tz_aware_prices(n_syms=5, n_days=300)
    factors = _make_naive_factor_returns(n_days=300)
    loadings = _make_loadings(syms)
    weights = _make_weights(syms)

    poisoned = np.full(5, np.nan)

    p1, p2 = _patch_prices_live(prices)
    with p1, p2, \
         patch("ascent.risk.covariance_model.get_factor_loadings", return_value=loadings), \
         patch("ascent.risk.covariance_model.get_factor_returns", return_value=factors), \
         patch("ascent.risk.covariance_model.compute_residual_variances", return_value=poisoned):
        result = build_factor_covariance_matrix(weights, as_of_date="2026-04-30")

    Sigma = result["full"]
    assert np.all(np.isfinite(Sigma)), "Poisoned residual variances must not leak NaN into Sigma"
    assert np.allclose(Sigma, np.eye(5) * 0.04)


def test_build_factor_covariance_matrix_missing_factor_data_still_clean():
    """Pre-existing legitimate degradation path (factor data unavailable)
    must still work unchanged."""
    syms = [f"S{i:02d}" for i in range(4)]
    weights = _make_weights(syms)

    with patch("ascent.risk.covariance_model.get_factor_loadings", return_value=pd.DataFrame()), \
         patch("ascent.risk.covariance_model.get_factor_returns", return_value=pd.DataFrame()):
        result = build_factor_covariance_matrix(weights, as_of_date="2026-04-30")

    Sigma = result["full"]
    assert np.all(np.isfinite(Sigma))
    assert np.allclose(Sigma, np.eye(4) * 0.04)
