"""
ascent/regime/integration.py
-----------------------------
Connects regime signals into Ascent's portfolio construction pipeline.

This module translates RegimeSignal objects into concrete portfolio-level
modifications:
  A. Gross exposure control  (risk_multiplier on position sizing)
  B. Sleeve reweighting      (alpha stack weight adjustments)
  C. Trade threshold control (signal threshold, rebalance band widening)
  D. Risk constraint changes (sector caps, max name weight)
  E. Covariance settings     (volatility half-life tightening in stress)

API contract:
  regime_adjust_portfolio(weights, signal, config) -> adjusted_weights
  regime_adjust_alpha_weights(base_weights, signal) -> adjusted_weights
  regime_get_constraints(signal, base_constraints) -> modified_constraints
  regime_covariance_halflife(signal, base_halflife) -> halflife_days

All functions are pure or nearly pure — they take inputs and return
outputs without side effects. The caller (stack.py or optimizer) decides
when to call them.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .types import RegimeLabel, RegimeSignal, REGIME_CONFIG_DEFAULTS

log = logging.getLogger(__name__)

# ── Base alpha sleeve weights (must sum to 1) ─────────────────────────────
_BASE_SLEEVE_WEIGHTS: Dict[str, float] = {
    "trend": 0.45,
    "statarb": 0.35,
    "meanrev": 0.10,
    "volatility": 0.10,
}

# ── A. Gross exposure control ─────────────────────────────────────────────

def regime_scale_weights(
    weights: pd.Series,
    signal: RegimeSignal,
) -> pd.Series:
    """
    Apply the regime risk multiplier to a portfolio weight vector.

    Parameters
    ----------
    weights : pd.Series
        Raw portfolio weights (may sum to < 1 already after optimizer).
    signal : RegimeSignal
        Current regime signal.

    Returns
    -------
    pd.Series
        Scaled weights. Long-only constraint enforced (no short positions).
    """
    if signal is None:
        return weights

    mult = signal.risk_multiplier
    scaled = weights * mult
    # Clamp to positive (long-only)
    scaled = scaled.clip(lower=0.0)

    if mult != 1.0:
        log.info(
            f"regime.integration: gross exposure scaled by {mult:.2f} "
            f"(regime={signal.label.value})"
        )
    return scaled


# ── B. Alpha sleeve reweighting ───────────────────────────────────────────

def regime_adjust_sleeve_weights(
    base_sleeve_weights: Optional[Dict[str, float]] = None,
    signal: Optional[RegimeSignal] = None,
    adjustment_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Return regime-adjusted alpha sleeve weights.

    Parameters
    ----------
    base_sleeve_weights : dict, optional
        Base sleeve weights. Defaults to _BASE_SLEEVE_WEIGHTS.
    signal : RegimeSignal, optional
        Current regime signal. If None, returns base weights.
    adjustment_scale : float
        Scale the adjustments (0 = no change, 1 = full regime adjustment).

    Returns
    -------
    dict
        Adjusted sleeve weights normalized to sum to 1.
    """
    base = dict(base_sleeve_weights or _BASE_SLEEVE_WEIGHTS)

    if signal is None or not signal.sleeve_adjustments:
        return base

    adjusted = {}
    for sleeve, base_w in base.items():
        delta = signal.sleeve_adjustments.get(sleeve, 0.0) * adjustment_scale
        adjusted[sleeve] = max(0.0, base_w + delta)

    # Renormalize
    total = sum(adjusted.values())
    if total <= 0:
        log.warning("regime.integration: sleeve weights summed to zero — using base")
        return base

    normalized = {k: v / total for k, v in adjusted.items()}
    log.debug(f"regime.integration: sleeve weights adjusted for {signal.label.value}: {normalized}")
    return normalized


# ── C. Signal threshold and rebalance band widening ───────────────────────

def regime_signal_threshold(
    base_threshold: float = 0.0,
    signal: Optional[RegimeSignal] = None,
) -> float:
    """
    Return a regime-adjusted signal threshold.
    In stressed / crisis regimes, raise the bar to reduce noise-driven trades.
    """
    if signal is None:
        return base_threshold

    label = signal.label
    bump: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 0.0,
        RegimeLabel.EUPHORIC.value: 0.05,
        RegimeLabel.STRESSED.value: 0.10,
        RegimeLabel.CRISIS.value: 0.15,
        RegimeLabel.UNCERTAIN.value: 0.08,
    }
    return base_threshold + bump.get(label.value, 0.0)


def regime_rebalance_band(
    base_band: float = 0.02,
    signal: Optional[RegimeSignal] = None,
) -> float:
    """
    Return a regime-adjusted rebalance tolerance band.
    Widen bands in stressed regimes to reduce excessive turnover.
    """
    if signal is None:
        return base_band

    multiplier: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 1.0,
        RegimeLabel.EUPHORIC.value: 1.2,
        RegimeLabel.STRESSED.value: 1.5,
        RegimeLabel.CRISIS.value: 2.0,
        RegimeLabel.UNCERTAIN.value: 1.3,
    }
    return base_band * multiplier.get(signal.label.value, 1.0)


# ── D. Risk constraints ───────────────────────────────────────────────────

def regime_max_weight(
    base_max_weight: float = 0.15,
    signal: Optional[RegimeSignal] = None,
) -> float:
    """
    Return maximum single-name weight given the current regime.
    Tighten in crisis to prevent concentration.
    """
    if signal is None:
        return base_max_weight

    overrides: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: base_max_weight,
        RegimeLabel.EUPHORIC.value: min(base_max_weight, 0.12),
        RegimeLabel.STRESSED.value: min(base_max_weight, 0.10),
        RegimeLabel.CRISIS.value: min(base_max_weight, 0.08),
        RegimeLabel.UNCERTAIN.value: min(base_max_weight, 0.12),
    }
    return overrides.get(signal.label.value, base_max_weight)


def regime_sector_cap(
    base_sector_cap: float = 1.0,  # e.g. 1 stock per sector = effectively no cap
    signal: Optional[RegimeSignal] = None,
) -> float:
    """
    Return max sector weight allocation. Returns fractional cap in [0, 1].
    In crisis, tighten to prevent sector concentration.
    """
    if signal is None:
        return base_sector_cap

    multiplier: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 1.00,
        RegimeLabel.EUPHORIC.value: 0.90,
        RegimeLabel.STRESSED.value: 0.80,
        RegimeLabel.CRISIS.value: 0.70,
        RegimeLabel.UNCERTAIN.value: 0.90,
    }
    return base_sector_cap * multiplier.get(signal.label.value, 1.0)


# ── E. Covariance / risk model settings ──────────────────────────────────

def regime_covariance_halflife(
    base_halflife: int = 63,
    signal: Optional[RegimeSignal] = None,
) -> int:
    """
    Return EWMA volatility half-life for covariance estimation.
    Use shorter half-life in stressed regimes so the model reacts faster
    to rising correlations and volatility.
    """
    if signal is None:
        return base_halflife

    multiplier: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 1.0,
        RegimeLabel.EUPHORIC.value: 0.8,
        RegimeLabel.STRESSED.value: 0.5,
        RegimeLabel.CRISIS.value: 0.3,
        RegimeLabel.UNCERTAIN.value: 0.7,
    }
    new_halflife = int(base_halflife * multiplier.get(signal.label.value, 1.0))
    return max(10, new_halflife)  # floor at 10 days


# ── Composite portfolio adjustment ────────────────────────────────────────

def apply_regime_to_portfolio(
    weights: pd.Series,
    signal: Optional[RegimeSignal],
    base_max_weight: float = 0.15,
) -> Tuple[pd.Series, Dict]:
    """
    Apply all regime adjustments to a portfolio weight vector in one call.

    Returns
    -------
    adjusted_weights : pd.Series
    metadata : dict
        Summary of what was applied (for logging / dashboard).
    """
    if signal is None:
        return weights, {"regime": "none", "adjustments": "none"}

    # Step 1: scale gross exposure
    w = regime_scale_weights(weights, signal)

    # Step 2: enforce per-name cap with iterative redistribution
    max_w = regime_max_weight(base_max_weight, signal)
    # Iterative clip-and-renormalize (converges in a few passes)
    target_sum = signal.risk_multiplier
    for _ in range(10):
        total = w.sum()
        if total > 1e-9:
            w = w / total * target_sum
        over_mask = w > max_w
        if not over_mask.any():
            break
        w = w.clip(upper=max_w)

    metadata = {
        "regime": signal.label.value,
        "risk_multiplier": signal.risk_multiplier,
        "max_weight_cap": max_w,
        "entropy": signal.entropy,
        "dwell_days": signal.dwell_days,
        "transition": signal.transition_flag,
    }
    return w, metadata


# ── Helper: build regime signal series from a pre-built signal DataFrame ──

def build_regime_series(signal_df: pd.DataFrame) -> pd.Series:
    """
    Given the output of RegimeDecisionEngine.process_to_frame(),
    return a simple pd.Series of RegimeLabel values indexed by date.
    """
    if "label" not in signal_df.columns:
        return pd.Series(dtype=str)
    return signal_df["label"].map(RegimeLabel.from_str)


def get_signal_for_date(
    signal_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Optional[RegimeSignal]:
    """
    Retrieve the most recent RegimeSignal available as of as_of_date
    from a pre-computed signal DataFrame.

    NOTE: uses only data up to as_of_date — no leakage.
    """
    available = signal_df.loc[signal_df.index.tz_localize(None) <= pd.Timestamp(as_of_date).tz_localize(None)]
    if available.empty:
        log.warning(f"regime.integration: no signal available as of {as_of_date.date()}")
        return None

    row = available.iloc[-1]
    k_cols = [c for c in signal_df.columns if c.startswith("prob_")]
    probs = np.array([row[c] for c in sorted(k_cols)]) if k_cols else np.array([1.0])

    # Reconstruct sleeve_adjustments from columns
    sleeve_cols = [c for c in signal_df.columns if c.startswith("sleeve_")]
    sleeve_adj = {c.replace("sleeve_", ""): float(row[c]) for c in sleeve_cols}

    return RegimeSignal(
        date=row.name,
        probs=probs,
        label=RegimeLabel.from_str(str(row["label"])),
        entropy=float(row.get("entropy", 0.5)),
        transition_flag=bool(row.get("transition_flag", False)),
        risk_multiplier=float(row.get("risk_multiplier", 1.0)),
        sleeve_adjustments=sleeve_adj,
        dwell_days=int(row.get("dwell_days", 0)),
    )
