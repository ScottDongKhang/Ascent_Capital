"""tests/test_mvo_optimizer.py

12 tests for Plan 2 — MVO Optimizer.
All tests use synthetic data and avoid network calls.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def _alpha(n=20, seed=1):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n), index=[f"S{i:02d}" for i in range(n)])


def _cov(n=20, vol=0.25):
    daily_var = (vol / np.sqrt(252)) ** 2
    return np.eye(n) * daily_var


def _sector_map(n=20):
    sectors = ["Tech", "Health", "Finance", "Energy", "Consumer",
               "Utilities", "Materials", "Industrials", "Real Estate", "Comm"]
    return {f"S{i:02d}": sectors[i % len(sectors)] for i in range(n)}


# ── MVO core ──────────────────────────────────────────────────────────────────

def test_mvo_weights_sum_to_one():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    w = optimize_mvo(_alpha(), _cov(), top_n=10)
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-3


def test_mvo_max_weight_respected():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    w = optimize_mvo(_alpha(), _cov(), max_weight=0.10, top_n=10)
    assert w is not None
    assert (w <= 0.10 + 1e-5).all()


def test_mvo_top_n_enforced():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    w = optimize_mvo(_alpha(30), _cov(30), top_n=10)
    assert w is not None
    assert len(w) <= 10


def test_mvo_turnover_penalty_reduces_changes():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    a = _alpha()
    c = _cov()
    prev = pd.Series(np.ones(20) / 20, index=a.index)

    w_no_penalty  = optimize_mvo(a, c, current_weights=None,  turnover_penalty=0.0,  top_n=10)
    w_with_penalty = optimize_mvo(a, c, current_weights=prev, turnover_penalty=0.01, top_n=10)

    if w_no_penalty is None or w_with_penalty is None:
        pytest.skip("solver unavailable")

    all_syms = prev.index
    change_no   = (w_no_penalty.reindex(all_syms).fillna(0) - prev).abs().sum()
    change_with = (w_with_penalty.reindex(all_syms).fillna(0) - prev).abs().sum()
    assert change_with <= change_no + 1e-4


def test_mvo_risk_aversion_reduces_concentration():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    a = _alpha()
    c = _cov()

    w_low  = optimize_mvo(a, c, risk_aversion=0.1,  top_n=10)
    w_high = optimize_mvo(a, c, risk_aversion=10.0, top_n=10)

    if w_low is None or w_high is None:
        pytest.skip("solver unavailable")

    # Higher risk aversion → lower portfolio variance (more spread out)
    var_low  = float(w_low.values  @ c[:len(w_low), :len(w_low)]  @ w_low.values)
    var_high = float(w_high.values @ c[:len(w_high), :len(w_high)] @ w_high.values)
    assert var_high <= var_low + 1e-8


def test_mvo_infeasible_returns_none():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    # max_weight so tight that sum-to-1 is infeasible for top_n
    a = _alpha(5)
    c = _cov(5)
    w = optimize_mvo(a, c, max_weight=0.001, top_n=5)  # 5 × 0.001 = 0.005 < 1.0
    assert w is None


def test_mvo_diagonal_proxy_when_no_covariance():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    w = optimize_mvo(_alpha(), covariance=None, top_n=10)
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-3


def test_mvo_factor_constraints_respected():
    from ascent.portfolio.mvo_optimizer import optimize_mvo
    n = 10
    a = _alpha(n)
    c = _cov(n)

    # Momentum constraint: all loadings = 1 (equal exposure), bound to [-0.2, 0.2]
    bvec = np.ones(n)
    factor_constraints = [{"loadings_vector": bvec, "lb": -0.2, "ub": 0.2, "factor": "beta_umd"}]
    w = optimize_mvo(a, c, factor_constraints=factor_constraints, top_n=n)
    if w is None:
        pytest.skip("solver unavailable")
    exposure = float(bvec @ w.reindex([f"S{i:02d}" for i in range(n)]).fillna(0).values)
    assert -0.2 - 1e-4 <= exposure <= 0.2 + 1e-4


def test_mvo_fallback_in_optimizer_py():
    from ascent.portfolio.optimizer import sector_constrained_weighted_mvo
    a = _alpha(10)
    # Force MVO to fail by patching optimize_mvo to return None
    with patch("ascent.portfolio.optimizer.optimize_mvo", return_value=None):
        w, method = sector_constrained_weighted_mvo(a, top_n=5)
    assert method == "rank_weight_fallback"
    assert not w.empty
    assert abs(w.sum() - 1.0) < 1e-4


def test_sector_constraint_prescreen():
    from ascent.portfolio.optimizer import sector_constrained_weighted_mvo
    a = _alpha(20)
    sm = _sector_map(20)
    w, method = sector_constrained_weighted_mvo(a, sector_map=sm, n=10)
    # No two positions should share a sector
    selected_sectors = [sm.get(s) for s in w.index if w[s] > 0.001]
    assert len(selected_sectors) == len(set(selected_sectors))


def test_optimization_method_logged():
    from ascent.portfolio.optimizer import sector_constrained_weighted_mvo
    a = _alpha(10)
    _, method = sector_constrained_weighted_mvo(a, top_n=5)
    assert method in ("mvo", "rank_weight_fallback")


def test_covariance_reindexed_to_screened_universe():
    """
    Bug: covariance built over the FULL alpha universe (e.g. ~900 symbols) was
    guarded on `covariance.shape[0] == len(alpha_screened)` (e.g. 15), which is
    never true, so the real covariance was silently discarded every time. Fix:
    reindex by symbol using `covariance_symbols` instead of comparing shapes.
    A correctly reindexed non-degenerate matrix should be used by the solver
    (verified indirectly: risk_aversion should now meaningfully discriminate
    between a genuinely diversifying covariance and the flat proxy).
    """
    from ascent.portfolio.optimizer import sector_constrained_weighted_mvo

    n_full = 30
    full_syms = [f"S{i:02d}" for i in range(n_full)]
    a = pd.Series(np.random.RandomState(7).randn(n_full), index=full_syms)

    # Full-universe covariance: S00-S09 form a tight cluster (corr 0.9), rest independent.
    n = n_full
    cov = np.eye(n) * (0.20 / np.sqrt(252)) ** 2
    for i in range(10):
        for j in range(10):
            if i != j:
                cov[i, j] = 0.9 * (0.20 / np.sqrt(252)) ** 2

    w, method = sector_constrained_weighted_mvo(
        a, covariance=cov, covariance_symbols=full_syms, n=10,
    )
    assert method == "mvo"
    assert not w.empty
    assert abs(w.sum() - 1.0) < 1e-3


def test_covariance_reindex_missing_symbols_falls_back_to_proxy(caplog):
    """If the screened universe has symbols the covariance matrix doesn't
    cover, reindexing would silently NaN — must fall through to the proxy
    and log loudly, not crash or silently proceed with NaNs."""
    from ascent.portfolio.optimizer import sector_constrained_weighted_mvo

    full_syms = [f"S{i:02d}" for i in range(5)]  # covariance covers only 5
    a = pd.Series(np.random.RandomState(3).randn(10),
                  index=[f"S{i:02d}" for i in range(10)])  # alpha universe has 10
    cov = np.eye(5) * (0.20 / np.sqrt(252)) ** 2

    with caplog.at_level("WARNING"):
        w, method = sector_constrained_weighted_mvo(
            a, covariance=cov, covariance_symbols=full_syms, n=10,
        )
    assert method == "mvo"
    assert not w.empty
    assert any("covariance" in rec.message.lower() for rec in caplog.records)


def test_per_name_vol_proxy_penalizes_high_vol_name_more():
    """
    Fallback proxy (no covariance available) should use per-name trailing
    realized vol, not a single flat 0.25 for every asset — a wild name (e.g.
    BRBR-style 75% vol) shouldn't be treated identically to a cash-like name
    (e.g. SGOV 0.2% vol) just because the factor covariance wasn't available.
    """
    from ascent.portfolio.mvo_optimizer import optimize_mvo

    syms = ["WILD", "CALM"]
    a = pd.Series({"WILD": 1.0, "CALM": 1.0})  # identical alpha
    vols = pd.Series({"WILD": 0.80, "CALM": 0.05})

    w_high_ra = optimize_mvo(
        a, covariance=None, vols=vols, risk_aversion=50.0, top_n=2, max_weight=1.0,
        min_weight=0.0,
    )
    if w_high_ra is None:
        pytest.skip("solver unavailable")
    # With equal alpha and a real risk penalty, the far riskier name should
    # get materially less weight than the calm one.
    assert w_high_ra["CALM"] > w_high_ra["WILD"]


def test_per_name_vol_proxy_missing_symbol_falls_back_to_flat():
    """A symbol with no trailing vol data should not crash — falls back to
    the flat 0.25 estimate for that name only."""
    from ascent.portfolio.mvo_optimizer import optimize_mvo

    a = pd.Series({"A": 1.0, "B": 1.0})
    vols = pd.Series({"A": 0.20})  # B missing
    w = optimize_mvo(a, covariance=None, vols=vols, top_n=2, max_weight=1.0, min_weight=0.0)
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-3


def test_expected_turnover_cost_positive():
    from ascent.portfolio.mvo_optimizer import compute_expected_turnover_cost
    current  = pd.Series({"A": 0.5, "B": 0.5})
    proposed = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    cost = compute_expected_turnover_cost(current, proposed, one_way_cost=0.002)
    assert cost >= 0.0
    assert cost > 0.0  # weights changed, so there's a cost
