"""
ascent/config/types.py
Shared type definitions for the Ascent Capital platform.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class AgentOutput:
    """
    Standardized output emitted by every specialist agent after each EOD run.
    The orchestrator collects a list of these and makes capital allocation decisions.

    Fields:
        agent_id:       Unique string identifier, e.g. "us_equities", "macro",
                        "international", "alternatives"
        as_of_date:     The trading date this output corresponds to
        target_weights: Dict mapping symbol -> target weight (floats summing to ~1.0)
        regime_signal:  The agent's regime label for this date, e.g. "calm_bull",
                        "stressed", "crisis". None if regime engine did not produce a signal.
        alpha_scores:   Optional DataFrame of raw alpha scores per symbol
                        (for debugging/logging)
        skill_score:    Rolling 63-day OOS Sharpe ratio of this agent.
                        None if insufficient history.
                        Agents with negative skill_score get zero capital from the orchestrator.
        metadata:       Freeform dict for agent-specific diagnostics
    """
    agent_id: str
    as_of_date: date
    target_weights: Dict[str, float]
    regime_signal: Optional[str] = None
    alpha_scores: Optional[pd.DataFrame] = None
    skill_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not isinstance(self.target_weights, dict):
            raise TypeError("target_weights must be a dict")

    @property
    def n_positions(self) -> int:
        """Number of non-zero positions."""
        return sum(1 for w in self.target_weights.values() if w > 0)

    @property
    def total_weight(self) -> float:
        """Sum of all target weights (should be ~1.0)."""
        return sum(self.target_weights.values())

    def summary(self) -> str:
        """One-line summary for logging."""
        skill_str = f"{self.skill_score:.3f}" if self.skill_score is not None else "N/A"
        return (
            f"[AgentOutput] {self.agent_id} | {self.as_of_date} | "
            f"{self.n_positions} positions | regime={self.regime_signal} | "
            f"skill={skill_str}"
        )


