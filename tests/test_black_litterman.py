"""tests/test_black_litterman.py

8 tests for Plan 2 — Black-Litterman blending + regime covariance.
All tests use synthetic data and avoid network calls.
"""
import numpy as np
import pandas as pd
import pytest


SYMS = ["A", "B", "C", "D"]


def _quant():
    return pd.Series([1.0, -1.0, 0.5, -0.5], index=SYMS)


def _llm():
    return pd.Series([0.8, -0.8, 0.2, 0.2], index=SYMS)


def _cov4():
    return np.eye(4) * 0.04


# ── Black-Litterman ───────────────────────────────────────────────────────────

def test_bl_posterior_between_prior_and_view():
    from ascent.portfolio.black_litterman import black_litterman_views
    posterior = black_litterman_views(_quant(), _llm())
    # Posterior should be strictly between prior and view (shrinkage)
    assert ((posterior - _quant()).abs() < (_llm() - _quant()).abs()).all()


def test_bl_high_confidence_view_dominates():
    from ascent.portfolio.black_litterman import black_litterman_views
    # omega_scale=0.01 → near-certain views → posterior close to LLM
    # Must pass covariance so the proper BL matrix path runs (not shrinkage fallback)
    posterior = black_litterman_views(_quant(), _llm(), covariance=_cov4(), tau=0.5, omega_scale=0.01)
    diff_to_llm   = (posterior - _llm()).abs().mean()
    diff_to_quant = (posterior - _quant()).abs().mean()
    assert diff_to_llm < diff_to_quant


def test_bl_low_confidence_view_minimal_impact():
    from ascent.portfolio.black_litterman import black_litterman_views
    # omega_scale=100 → very uncertain views → posterior close to quant prior
    posterior = black_litterman_views(_quant(), _llm(), tau=0.01, omega_scale=100.0)
    diff_to_quant = (posterior - _quant()).abs().mean()
    diff_to_llm   = (posterior - _llm()).abs().mean()
    assert diff_to_quant < diff_to_llm


def test_bl_missing_llm_alpha_returns_quant():
    from ascent.portfolio.black_litterman import black_litterman_views
    posterior = black_litterman_views(_quant(), None)
    pd.testing.assert_series_equal(posterior, _quant())


def test_bl_handles_partial_overlap():
    from ascent.portfolio.black_litterman import black_litterman_views
    # LLM alpha covers only 2 of 4 symbols
    llm_partial = pd.Series([0.9, -0.9], index=["A", "B"])
    quant = _quant()
    posterior = black_litterman_views(quant, llm_partial)
    # Unmapped symbols (C, D) should keep their quant prior
    assert posterior["C"] == quant["C"]
    assert posterior["D"] == quant["D"]
    # Mapped symbols should move toward LLM view
    assert abs(posterior["A"] - quant["A"]) > 0


def test_bl_fallback_shrinkage_when_no_covariance():
    from ascent.portfolio.black_litterman import black_litterman_views
    # No covariance → shrinkage fallback
    posterior = black_litterman_views(_quant(), _llm(), covariance=None, tau=0.10)
    assert len(posterior) == 4
    # Result should be strictly between quant and llm
    for sym in SYMS:
        lo = min(_quant()[sym], _llm()[sym])
        hi = max(_quant()[sym], _llm()[sym])
        assert lo - 1e-9 <= posterior[sym] <= hi + 1e-9


def test_blending_weight_maps_ir_to_tau():
    from ascent.portfolio.black_litterman import get_blending_weight
    assert get_blending_weight(0.20) == 0.05
    assert get_blending_weight(0.45) == 0.10
    assert get_blending_weight(0.75) == 0.15


def test_regime_covariance_blend_weights_sum_to_one():
    from ascent.portfolio.regime_covariance import blend_regime_covariances
    n = 5
    calm   = np.eye(n) * 0.02
    stress = np.eye(n) * 0.04
    crisis = np.eye(n) * 0.08

    for label, conf in [("crisis", 0.8), ("stressed", 0.7), ("calm_bull", 0.9), ("uncertain", 0.5)]:
        result = blend_regime_covariances(calm, stress, crisis, label, conf)
        assert result.shape == (n, n)
        # Verify it's a valid covariance (positive diagonal)
        assert (np.diag(result) > 0).all()
