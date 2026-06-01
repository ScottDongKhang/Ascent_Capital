"""ascent/causal/compatibility.py

Gate 1: regime-causal mechanism compatibility.
Static dict lookup — no LLM, no external calls.
"""
from typing import List

# Mechanism types allowed per regime.
# macro_hedge is allowed in every regime — always preserve it.
_REGIME_ALLOWED: dict = {
    "calm_bull": {
        "momentum_catalyst",
        "supply_demand_inflection",
        "quality_defensive",
        "macro_hedge",
    },
    "stressed": {
        "quality_defensive",
        "macro_hedge",
        "mean_reversion",       # short-term reversion can work in stressed markets
    },
    "crisis": {
        "macro_hedge",
    },
    "neutral": {
        "momentum_catalyst",
        "quality_defensive",
        "supply_demand_inflection",
        "macro_hedge",
        "mean_reversion",
    },
    "uncertain": {
        "quality_defensive",
        "momentum_catalyst",
        "macro_hedge",
    },
}

# Conservative fallback for unknown regimes
_FALLBACK_ALLOWED = {"quality_defensive", "macro_hedge"}


def regime_compatible(mechanism_type: str, regime: str) -> bool:
    """
    Return True if a mechanism type is compatible with the current regime.

    Args:
        mechanism_type: one of the six mechanism types from dag_builder.py
        regime: current regime label (e.g. "calm_bull", "stressed", "crisis")

    Returns:
        True if the mechanism is allowed in this regime, False otherwise.
    """
    allowed = _REGIME_ALLOWED.get(regime, _FALLBACK_ALLOWED)
    return mechanism_type in allowed


def filter_mechanisms(mechanisms: List[dict], regime: str) -> List[dict]:
    """
    Filter a list of mechanism dicts to those compatible with the current regime.
    Each dict must have a 'mechanism_type' key.

    Returns:
        Filtered list (preserves order, no mutation of originals).
    """
    return [m for m in mechanisms if regime_compatible(m.get("mechanism_type", ""), regime)]
