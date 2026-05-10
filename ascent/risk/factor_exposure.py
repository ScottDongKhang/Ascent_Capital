"""ascent/risk/factor_exposure.py

Portfolio-level factor tilt computation and dashboard export.

compute_portfolio_exposures(weights, as_of_date) -> pd.Series
  Returns w'B: one exposure per factor (approx. SD units of factor return).

check_factor_bounds(exposures, bounds) -> list[str]
  Soft bounds — returns list of violation strings, never blocks trades.

exposure_report(weights, as_of_date) -> dict
  Human-readable factor exposure dict; written to dashboard/factor_exposures.json.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ascent.risk.factor_model import get_factor_loadings, BETA_COLS

log = logging.getLogger(__name__)

EXPOSURE_PATH = Path("dashboard/factor_exposures.json")

_FACTOR_LABELS = {
    "beta_mkt": "Market",
    "beta_smb": "Size (SMB)",
    "beta_hml": "Value (HML)",
    "beta_rmw": "Profitability (RMW)",
    "beta_cma": "Investment (CMA)",
    "beta_umd": "Momentum (UMD)",
}

_DEFAULT_BOUNDS = {
    "beta_mkt": (-0.30, 0.30),
    "beta_smb": (-0.40, 0.40),
    "beta_hml": (-0.40, 0.40),
    "beta_rmw": (-0.40, 0.40),
    "beta_cma": (-0.40, 0.40),
    "beta_umd": (-0.50, 0.50),
}


def compute_portfolio_exposures(
    weights_series: pd.Series,
    as_of_date,
) -> pd.Series:
    """
    Portfolio factor exposures = w'B.

    Returns a Series indexed by BETA_COLS with exposure values in loading units
    (approximately standard deviations of factor return).
    Returns zero Series on any failure.
    """
    try:
        loadings = get_factor_loadings(as_of_date).reindex(weights_series.index)
        if loadings.empty:
            return pd.Series(0.0, index=BETA_COLS)

        w = weights_series.reindex(loadings.index).fillna(0).values
        B = loadings[BETA_COLS].fillna(0).values  # N × 6
        exposures = w @ B                          # (6,)
        return pd.Series(exposures, index=BETA_COLS)
    except Exception as exc:
        log.warning("[FactorExposure] Computation failed: %s", exc)
        return pd.Series(0.0, index=BETA_COLS)


def check_factor_bounds(
    exposures: pd.Series,
    bounds: Optional[dict] = None,
) -> list[str]:
    """
    Check exposures against soft bounds.
    Returns list of human-readable violation strings (empty = no violations).
    These are informational — they never block trades.
    """
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    violations = []
    for factor, (lo, hi) in bounds.items():
        val = float(exposures.get(factor, 0.0))
        if val < lo:
            violations.append(
                f"{_FACTOR_LABELS.get(factor, factor)}: {val:+.3f} < lower bound {lo:+.3f}"
            )
        elif val > hi:
            violations.append(
                f"{_FACTOR_LABELS.get(factor, factor)}: {val:+.3f} > upper bound {hi:+.3f}"
            )
    return violations


def exposure_report(
    weights_series: pd.Series,
    as_of_date,
) -> dict:
    """
    Full factor exposure report for a portfolio.

    Returns dict with:
      exposures          — {factor: value}
      dominant_factor    — factor with largest |exposure|
      concentration_score — Herfindahl of squared exposures
      violations         — list of bound violations
      as_of              — date string
    """
    exposures = compute_portfolio_exposures(weights_series, as_of_date)
    violations = check_factor_bounds(exposures)

    sq = exposures ** 2
    total_sq = sq.sum()
    hhi = float((sq / total_sq).pow(2).sum()) if total_sq > 1e-10 else 0.0
    dominant = str(exposures.abs().idxmax()) if not exposures.empty else ""

    if violations:
        log.info("[FactorExposure] Bound violations: %s", "; ".join(violations))

    return {
        "exposures":          {k: round(float(v), 4) for k, v in exposures.items()},
        "dominant_factor":    dominant,
        "concentration_score": round(hhi, 4),
        "violations":         violations,
        "as_of":              str(as_of_date),
    }


def export_factor_exposures(weights_series: pd.Series, as_of_date) -> None:
    """Write factor exposure report to dashboard/factor_exposures.json."""
    try:
        report = exposure_report(weights_series, as_of_date)
        report["timestamp"] = datetime.now().isoformat()
        EXPOSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPOSURE_PATH.write_text(json.dumps(report, indent=2))
        log.info("[FactorExposure] Exported to %s", EXPOSURE_PATH)
    except Exception as exc:
        log.warning("[FactorExposure] Export failed: %s", exc)


def format_exposure_context(weights_series: pd.Series, as_of_date) -> str:
    """
    Human-readable factor exposure summary for debate agent context.
    Returns empty string on failure.
    """
    try:
        report = exposure_report(weights_series, as_of_date)
        exposures = report["exposures"]
        lines = ["Portfolio factor exposures (vs. market-neutral = 0):"]
        for col in BETA_COLS:
            label = _FACTOR_LABELS.get(col, col)
            val = exposures.get(col, 0.0)
            direction = "long" if val > 0 else "short"
            lines.append(f"  {label}: {val:+.3f}σ ({direction} tilt)")
        if report["violations"]:
            lines.append("Bound breaches: " + "; ".join(report["violations"]))
        lines.append(f"Dominant factor: {_FACTOR_LABELS.get(report['dominant_factor'], report['dominant_factor'])}")
        return "\n".join(lines)
    except Exception as exc:
        log.warning("[FactorExposure] Context format failed: %s", exc)
        return ""
