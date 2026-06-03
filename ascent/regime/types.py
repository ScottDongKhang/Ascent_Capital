"""
ascent/regime/types.py
----------------------
Stable data-contract types for the regime subsystem.

Everything the rest of the codebase consumes from the regime engine
lives here as dataclasses or named enumerations so imports stay clean
and callers never need to reach into model internals.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Regime labels ──────────────────────────────────────────────────────────
class RegimeLabel(str, enum.Enum):
    """
    Four-state vocabulary inferred from estimated state properties.
    Labels are descriptive, not prescriptive — the model assigns states
    and we relabel after fitting based on mean return / vol / drawdown.
    """
    CALM_BULL = "calm_bull"          # low vol, positive drift
    EUPHORIC = "euphoric"            # high momentum, elevated vol
    STRESSED = "stressed"            # negative drift, rising vol
    CRISIS = "crisis"                # large negative returns, high vol / drawdown
    UNCERTAIN = "uncertain"          # transition / high entropy

    @classmethod
    def from_str(cls, s: str) -> "RegimeLabel":
        for member in cls:
            if member.value == s:
                return member
        return cls.UNCERTAIN


# ── Per-date regime signal ─────────────────────────────────────────────────
@dataclass
class RegimeSignal:
    """
    The single object that portfolio construction consumes.
    All probability vectors are length K (number of regimes).
    """
    date: pd.Timestamp

    # Probability of each latent state
    probs: np.ndarray                # shape (K,), sums to ~1

    # Chosen label after hysteresis / smoothing
    label: RegimeLabel

    # Shannon entropy of probs — 0 = certain, log(K) = maximum uncertainty
    entropy: float

    # True on the day the smoothed regime label changes
    transition_flag: bool

    # Scalar in [0, 1] — multiplier applied to gross exposure
    risk_multiplier: float

    # Fractional adjustments to each alpha sleeve weight
    # e.g. {"trend": +0.10, "statarb": -0.05, "meanrev": 0.0, "volatility": 0.05}
    sleeve_adjustments: Dict[str, float] = field(default_factory=dict)

    # Days the current label has been active continuously
    dwell_days: int = 0

    @property
    def confidence(self) -> float:
        """Max probability across states — proxy for certainty."""
        return float(np.max(self.probs))

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "label": self.label.value,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "risk_multiplier": self.risk_multiplier,
            "transition_flag": self.transition_flag,
            "dwell_days": self.dwell_days,
            **{f"prob_{i}": float(self.probs[i]) for i in range(len(self.probs))},
            **{f"sleeve_{k}": v for k, v in self.sleeve_adjustments.items()},
        }


# ── Walk-forward model selection scorecard ────────────────────────────────
@dataclass
class RegimeScorecard:
    """Summary of model selection results over walk-forward windows."""
    n_regimes: int
    lookback_days: int
    aic_mean: float
    bic_mean: float
    loglike_mean: float
    regime_persistence_mean: float    # avg dwell time in days
    transition_stability: float       # std of transition matrix across windows
    oos_sharpe_impact: float          # portfolio Sharpe with vs without regime
    oos_drawdown_impact: float        # max drawdown improvement
    false_switch_rate: float          # switches per month
    tiny_state_penalty: float         # fraction of windows with a near-empty state

    @property
    def composite_score(self) -> float:
        """
        Higher is better. Combines information criteria with
        practical usefulness metrics.
        """
        score = (
            -0.20 * self.aic_mean / 1000          # lower AIC preferred
            + 0.20 * self.regime_persistence_mean  # longer dwell preferred
            - 0.15 * self.transition_stability     # stable transitions preferred
            + 0.25 * self.oos_sharpe_impact        # OOS Sharpe lift preferred
            - 0.10 * self.false_switch_rate        # fewer switches preferred
            - 0.10 * self.tiny_state_penalty       # avoid degenerate states
        )
        return score


# ── Break detection result ────────────────────────────────────────────────
@dataclass
class BreakResult:
    """Structural break detection output."""
    feature_name: str
    break_dates: List[pd.Timestamp]
    n_breaks: int
    confidence: float                  # normalized from ruptures cost ratio


# ── Regime config defaults (consumed by config.py addition) ──────────────
REGIME_CONFIG_DEFAULTS: Dict = {
    # Feature layer
    "regime_spy_ticker": "SPY",
    "regime_vix_ticker": "^VIX",      # fetched if available; graceful fallback
    "regime_feature_lookbacks": [5, 21, 63],

    # Model
    "regime_n_candidates": [2, 3, 4],  # evaluated in walk-forward selection
    "regime_training_min_days": 504,   # ~2 years minimum
    "regime_refit_every_days": 5,      # Phase 3b: frequent scheduled refit (was 63)

    # Walk-forward validation
    "regime_wf_train_days": 756,       # 3 years
    "regime_wf_test_days": 126,        # 6 months
    "regime_wf_step_days": 63,         # step every quarter

    # Hysteresis
    "regime_enter_threshold": 0.65,    # probability to enter new regime (fallback when severity unknown)
    "regime_exit_threshold": 0.45,     # probability to exit (hysteresis band)
    "regime_min_dwell_days": 5,        # must stay K days before switch counts
    "regime_entropy_uncertain_threshold": 0.85,  # fraction of max entropy
    # Asymmetric hysteresis — downgrade (→ worse regime) is faster to trigger
    # than upgrade (→ better regime). See decision.py _HysteresisStateMachine.
    "regime_downgrade_threshold": 0.40,   # prob needed to enter a worse regime
    "regime_upgrade_threshold":   0.70,   # prob needed to enter a better regime

    # Integration — exposure multipliers by regime label
    "regime_risk_multiplier": {
        "calm_bull": 1.00,
        "euphoric": 0.85,
        "stressed": 0.65,
        "crisis": 0.40,
        "uncertain": 0.75,
    },

    # Integration — sleeve adjustments by regime label (additive fractions)
    "regime_sleeve_adjustments": {
        "calm_bull":  {"trend": +0.10, "statarb": 0.00, "meanrev": -0.05, "volatility": -0.05},
        "euphoric":   {"trend": +0.05, "statarb": -0.05, "meanrev": -0.05, "volatility": +0.05},
        "stressed":   {"trend": -0.10, "statarb": +0.05, "meanrev": +0.05, "volatility": 0.00},
        "crisis":     {"trend": -0.15, "statarb": -0.05, "meanrev": 0.00, "volatility": +0.20},
        "uncertain":  {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
    },

    # Sector / weight tightening by regime
    "regime_max_weight_overrides": {
        "calm_bull": 0.15,
        "euphoric": 0.12,
        "stressed": 0.10,
        "crisis": 0.08,
        "uncertain": 0.12,
    },

    # Break detection
    "regime_break_min_size": 63,       # ruptures min_size parameter
    "regime_break_penalty_multiplier": 2.0,
}
