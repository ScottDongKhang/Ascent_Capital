"""ascent/risk/factor_constraints.py

Factor constraint generator for the MVO optimizer (Plan 2).

Produces linear constraints: lb <= w'B_factor <= ub
in a format consumable by cvxpy (Plan 2 wires these in).

This module is built and tested here; Plan 2 imports it.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ascent.risk.factor_model import get_factor_loadings, BETA_COLS

log = logging.getLogger(__name__)

# Default bounds per factor (in loading units ≈ σ of factor return)
_CALM_BULL_BOUNDS = {
    "beta_mkt": (-0.30, 0.30),
    "beta_smb": (-0.40, 0.40),
    "beta_hml": (-0.40, 0.40),
    "beta_rmw": (-0.40, 0.40),
    "beta_cma": (-0.40, 0.40),
    "beta_umd": (-0.50, 0.50),
}

_CRISIS_BOUNDS = {
    "beta_mkt": (-0.15, 0.15),
    "beta_smb": (-0.25, 0.25),
    "beta_hml": (-0.25, 0.25),
    "beta_rmw": (-0.25, 0.25),
    "beta_cma": (-0.25, 0.25),
    "beta_umd": (-0.20, 0.20),
}

_STRESSED_BOUNDS = {
    "beta_mkt": (-0.20, 0.20),
    "beta_smb": (-0.30, 0.30),
    "beta_hml": (-0.30, 0.30),
    "beta_rmw": (-0.30, 0.30),
    "beta_cma": (-0.30, 0.30),
    "beta_umd": (-0.35, 0.35),
}


def get_regime_factor_bounds(regime_label: str) -> dict:
    """
    Return factor bounds tightened for the current regime.
    Crisis is tightest; calm_bull is standard.
    """
    label = str(regime_label).lower()
    if "crisis" in label:
        return _CRISIS_BOUNDS
    if "stress" in label:
        return _STRESSED_BOUNDS
    return _CALM_BULL_BOUNDS


def build_factor_constraints(
    symbols: list[str],
    as_of_date,
    regime_label: str = "calm_bull",
) -> list[dict]:
    """
    Build linear factor constraint dicts for the MVO optimizer.

    Each dict represents one constraint: lb <= w'B_k <= ub for factor k.
    Format: {"loadings_vector": np.ndarray(N,), "lb": float, "ub": float, "factor": str}

    Plan 2 converts these into cvxpy constraints:
        cp.constraints += [lb <= loadings_vector @ w, loadings_vector @ w <= ub]

    Returns empty list when factor data is unavailable (optimizer ignores gracefully).
    """
    loadings = get_factor_loadings(as_of_date)
    if loadings.empty:
        log.warning("[FactorConstraints] No loadings for %s — skipping constraints", as_of_date)
        return []

    loadings = loadings.reindex(symbols).fillna(0)
    bounds = get_regime_factor_bounds(regime_label)

    constraints = []
    for i, col in enumerate(BETA_COLS):
        bvec = loadings[col].values  # N-length loading vector for this factor
        lb, ub = bounds.get(col, (-1.0, 1.0))
        constraints.append({
            "loadings_vector": bvec,
            "lb":              float(lb),
            "ub":              float(ub),
            "factor":          col,
        })

    log.debug("[FactorConstraints] Built %d constraints for regime=%s", len(constraints), regime_label)
    return constraints
