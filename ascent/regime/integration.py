"""
ascent/regime/integration.py
-----------------------------
Connects regime signals into Ascent's portfolio construction pipeline.

This module translates RegimeSignal objects into concrete portfolio-level
modifications:
  C. Trade threshold control (signal threshold, rebalance band widening)
  D. Risk constraint changes (sector caps, max name weight)
  E. Covariance settings     (volatility half-life tightening in stress)

All functions are pure or nearly pure — they take inputs and return
outputs without side effects. The caller (stack.py or optimizer) decides
when to call them.

Note: gross exposure scaling (risk_multiplier) and alpha-sleeve reweighting
(sleeve_adjustments) used to live here as regime_scale_weights(),
regime_adjust_sleeve_weights(), and the composite apply_regime_to_portfolio().
All three were confirmed to have zero live callers (deleted 2026-08-16) —
regime_signal was already accepted-but-unused by build_alpha_stack(), and
apply_regime_to_portfolio was imported but never called in ascent/main.py.
regime_max_weight() below is a separate, still-live cap-tightening mechanism.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .types import RegimeLabel, RegimeSignal, REGIME_CONFIG_DEFAULTS

log = logging.getLogger(__name__)

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
