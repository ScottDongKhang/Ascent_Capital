"""
ascent/regime/decision.py
--------------------------
Converts raw regime probabilities into actionable state signals.

Layers applied in order:
  1. Entropy filter   — classify as UNCERTAIN when model is low-confidence
  2. Hysteresis       — require probability > enter_threshold for N days to switch
  3. Min dwell        — block transitions until current regime held >= min_dwell_days
  4. Output           — emit smoothed RegimeSignal per date

This layer runs AFTER model.predict_probs() and BEFORE integration.py.
All decisions are causal: only information up to date t is used.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import RegimeLabel, RegimeSignal

log = logging.getLogger(__name__)

# Severity ordering for asymmetric hysteresis.
# Higher = more defensive / risk-off. Used to decide which threshold to apply.
_REGIME_SEVERITY: Dict[str, int] = {
    "calm_bull": 0,
    "euphoric":  0,
    "uncertain": 1,
    "stressed":  2,
    "crisis":    3,
}


# ── Shannon entropy ────────────────────────────────────────────────────────

def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy normalized to [0, 1] range."""
    k = len(probs)
    if k <= 1:
        return 0.0
    p = np.clip(probs, 1e-12, 1.0)
    raw = -np.sum(p * np.log(p))
    max_entropy = np.log(k)
    return float(raw / max_entropy)


# ── Hysteresis state machine ───────────────────────────────────────────────

class _HysteresisStateMachine:
    """
    Tracks the current smoothed regime and applies hysteresis.

    Rules:
      • Current regime remains active until a candidate new regime's
        probability exceeds enter_threshold for min_dwell_days consecutive days.
      • Once active, a regime is held unless its probability drops below
        exit_threshold (and the exit has also been sustained briefly).
    """

    def __init__(
        self,
        initial_regime: int,
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.35,
        min_dwell_days: int = 3,
        entropy_uncertain_threshold: float = 0.90,
        severity: Optional[dict] = None,
        downgrade_threshold: Optional[float] = None,
        upgrade_threshold: Optional[float] = None,
    ):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_dwell_days = min_dwell_days
        self.entropy_threshold = entropy_uncertain_threshold
        self._severity = severity or {}
        self._downgrade_threshold = downgrade_threshold if downgrade_threshold is not None else enter_threshold
        self._upgrade_threshold = upgrade_threshold if upgrade_threshold is not None else enter_threshold

        self._current: int = initial_regime
        self._dwell: int = 0
        self._candidate: Optional[int] = None
        self._candidate_streak: int = 0

    def step(self, probs: np.ndarray) -> tuple:
        """
        Process one day's probability vector.
        Returns (regime_index, transition_flag, dwell_days).
        """
        k = len(probs)
        entropy = _entropy(probs)
        dominant = int(np.argmax(probs))
        dominant_prob = float(probs[dominant])

        # High entropy → UNCERTAIN mode (don't change underlying state)
        is_uncertain = entropy >= self.entropy_threshold

        self._dwell += 1
        transition_flag = False

        if is_uncertain:
            # Stay in current regime but signal uncertainty
            self._candidate = None
            self._candidate_streak = 0
            return self._current, transition_flag, self._dwell, True

        # Evaluate transition — use asymmetric threshold based on regime severity
        if dominant != self._current:
            current_sev = self._severity.get(self._current, -1)
            candidate_sev = self._severity.get(dominant, -1)
            if current_sev == -1 or candidate_sev == -1:
                threshold = self.enter_threshold           # unknown severity → fallback
            elif candidate_sev > current_sev:
                threshold = self._downgrade_threshold      # transitioning to worse regime
            else:
                threshold = self._upgrade_threshold        # transitioning to better regime

            if dominant_prob >= threshold:
                if self._candidate == dominant:
                    self._candidate_streak += 1
                else:
                    self._candidate = dominant
                    self._candidate_streak = 1

                if self._candidate_streak >= self.min_dwell_days:
                    self._current = dominant
                    self._dwell = 1
                    self._candidate = None
                    self._candidate_streak = 0
                    transition_flag = True
            else:
                self._candidate = None
                self._candidate_streak = 0

        return self._current, transition_flag, self._dwell, False


# ── Decision engine ────────────────────────────────────────────────────────

class RegimeDecisionEngine:
    """
    Converts a probability DataFrame (dates × K states) into a list
    of RegimeSignal objects.

    Parameters
    ----------
    state_labels : dict {int: RegimeLabel}
        Mapping from state index to human-readable label.
    enter_threshold : float
        Probability must exceed this to begin transition countdown.
    exit_threshold : float
        Not currently used but reserved for future exit-confirmation logic.
    min_dwell_days : int
        Days a candidate regime must dominate before transition is committed.
    entropy_uncertain_threshold : float
        Fraction of max entropy above which day is labeled UNCERTAIN.
    risk_multipliers : dict {str: float}
        Per-label gross exposure multipliers.
    sleeve_adjustments : dict {str: dict{str: float}}
        Per-label alpha sleeve weight adjustments.
    """

    def __init__(
        self,
        state_labels: Dict[int, RegimeLabel],
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.35,
        min_dwell_days: int = 3,
        entropy_uncertain_threshold: float = 0.90,
        downgrade_threshold: float = 0.40,
        upgrade_threshold: float = 0.70,
        risk_multipliers: Optional[Dict[str, float]] = None,
        sleeve_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.state_labels = state_labels
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_dwell_days = min_dwell_days
        self.entropy_threshold = entropy_uncertain_threshold
        self.downgrade_threshold = downgrade_threshold
        self.upgrade_threshold = upgrade_threshold

        # Defaults if not supplied
        self._risk_mult = risk_multipliers or {
            "calm_bull": 1.00,
            "euphoric": 0.85,
            "stressed": 0.65,
            "crisis": 0.40,
            "uncertain": 0.75,
        }
        self._sleeve_adj = sleeve_adjustments or {
            "calm_bull":  {"trend": +0.10, "statarb": 0.00, "meanrev": -0.05, "volatility": -0.05},
            "euphoric":   {"trend": +0.05, "statarb": -0.05, "meanrev": -0.05, "volatility": +0.05},
            "stressed":   {"trend": -0.10, "statarb": +0.05, "meanrev": +0.05, "volatility": 0.00},
            "crisis":     {"trend": -0.15, "statarb": -0.05, "meanrev": 0.00, "volatility": +0.20},
            "uncertain":  {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
        }

    def process(self, prob_df: pd.DataFrame) -> List[RegimeSignal]:
        """
        Convert probability DataFrame to a list of RegimeSignals.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Columns are state indices (int), index is DatetimeIndex.
            Rows must be in chronological order.
        """
        if prob_df.empty:
            return []

        n_states = prob_df.shape[1]
        # Build severity dict {state_idx: severity_int} for asymmetric hysteresis
        severity = {
            s_idx: _REGIME_SEVERITY.get(
                lbl.value if hasattr(lbl, "value") else str(lbl), -1
            )
            for s_idx, lbl in self.state_labels.items()
        }
        first_probs = prob_df.iloc[0].values
        initial_state = int(np.argmax(first_probs))
        machine = _HysteresisStateMachine(
            initial_regime=initial_state,
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
            min_dwell_days=self.min_dwell_days,
            entropy_uncertain_threshold=self.entropy_threshold,
            severity=severity,
            downgrade_threshold=self.downgrade_threshold,
            upgrade_threshold=self.upgrade_threshold,
        )

        signals: List[RegimeSignal] = []

        for i, (date, row) in enumerate(prob_df.iterrows()):
            probs = row.values.astype(float)
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum() + 1e-12

            state_idx, transition_flag, dwell_days, is_uncertain = machine.step(probs)

            # Determine label
            if is_uncertain:
                label = RegimeLabel.UNCERTAIN
            else:
                label = self.state_labels.get(state_idx, RegimeLabel.UNCERTAIN)

            entropy = _entropy(probs)
            risk_mult = self._risk_mult.get(label.value, 0.75)
            sleeve_adj = self._sleeve_adj.get(label.value, {})

            signals.append(RegimeSignal(
                date=date,
                probs=probs,
                label=label,
                entropy=entropy,
                transition_flag=transition_flag,
                risk_multiplier=risk_mult,
                sleeve_adjustments=sleeve_adj,
                dwell_days=dwell_days,
            ))

        n_transitions = sum(1 for s in signals if s.transition_flag)
        label_counts = {}
        for s in signals:
            label_counts[s.label.value] = label_counts.get(s.label.value, 0) + 1
        log.info(
            f"regime.decision: processed {len(signals)} days, "
            f"{n_transitions} transitions, label distribution: {label_counts}"
        )
        return signals

    def process_to_frame(self, prob_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: return signals as a DataFrame."""
        signals = self.process(prob_df)
        if not signals:
            return pd.DataFrame()
        rows = [s.to_dict() for s in signals]
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    def latest_signal(self, prob_df: pd.DataFrame) -> Optional[RegimeSignal]:
        """Return only the most recent signal (for live use)."""
        signals = self.process(prob_df)
        return signals[-1] if signals else None
