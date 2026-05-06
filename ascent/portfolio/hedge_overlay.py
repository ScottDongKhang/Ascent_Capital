"""
ascent/portfolio/hedge_overlay.py

Regime-adaptive VIXY tail hedge overlay.

Computes a hedge weight from the current RegimeSignal, then scales all
non-VIXY positions down proportionally to make room. Weights always
sum to 1.0 after the overlay. No I/O — pure functions only.

Called by run_all_agents.py after orchestration, before writing
execution/merged_weights.json.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

from ascent.regime.types import RegimeLabel, RegimeSignal

# Base VIXY weights by regime label (before confidence scaling)
_BASE_HEDGE: Dict[RegimeLabel, float] = {
    RegimeLabel.CRISIS:    0.08,
    RegimeLabel.STRESSED:  0.04,
    RegimeLabel.EUPHORIC:  0.02,
    RegimeLabel.CALM_BULL: 0.00,
    RegimeLabel.UNCERTAIN: 0.00,
}


def compute_hedge_weight(label: RegimeLabel, confidence: float) -> float:
    """
    Return the target VIXY weight for the given regime label and confidence.

    Scales the base weight by confidence so the hedge grows gradually
    as the regime signal becomes more certain — no binary jump at threshold.

    Args:
        label:      RegimeLabel enum value
        confidence: RegimeSignal.confidence (max prob across states, 0–1)

    Returns:
        VIXY target weight, scaled by confidence from the base hedge table.
        With default base weights, maximum is 0.08 (crisis at full confidence).
    """
    base = _BASE_HEDGE.get(label, 0.0)
    return round(base * confidence, 4)


def apply_hedge_overlay(
    weights: Dict[str, float],
    regime_signal: Optional[Union[RegimeSignal, str]],
) -> Tuple[Dict[str, float], Dict]:
    """
    Apply tail hedge overlay to a portfolio weights dict.

    Removes any existing VIXY allocation, scales all remaining positions
    proportionally to `1 - hedge_weight`, then sets VIXY to `hedge_weight`.
    If hedge_weight < 0.005 (i.e. calm_bull or low-confidence stressed),
    returns the original weights unchanged.

    Args:
        weights:       {symbol: weight}, must sum to ~1.0
        regime_signal: Current RegimeSignal from regime engine, a plain regime
                       label string (from AgentOutput.regime_signal), or None.
                       When a string is passed, confidence defaults to 0.7.

    Returns:
        (hedged_weights, metadata) where hedged_weights sums to 1.0 and
        metadata contains hedge_weight, regime_label, confidence, vixy_before,
        vixy_after for logging.
    """
    # Resolve label and confidence from whatever caller passes in
    if regime_signal is None:
        _label: Optional[RegimeLabel] = None
        _confidence: float = 0.0
    elif isinstance(regime_signal, str):
        _label = RegimeLabel.from_str(regime_signal)
        _confidence = 0.7  # string carries no probability info; use moderate default
    else:
        _label = regime_signal.label
        _confidence = regime_signal.confidence

    vixy_before = weights.get("VIXY", 0.0)

    # Validate input weights sum to ~1.0
    total_in = sum(weights.values())
    if abs(total_in - 1.0) > 0.01:
        import warnings
        warnings.warn(
            f"apply_hedge_overlay: input weights sum to {total_in:.4f}, expected ~1.0. "
            "Normalising before overlay.",
            stacklevel=2,
        )
        weights = {k: v / total_in for k, v in weights.items()}
        vixy_before = weights.get("VIXY", 0.0)

    no_change_meta = {
        "hedge_weight": 0.0,
        "regime_label": _label.value if _label else "unknown",
        "confidence":   _confidence,
        "vixy_before":  vixy_before,
        "vixy_after":   vixy_before,
    }

    if _label is None:
        return dict(weights), no_change_meta

    hedge_weight = compute_hedge_weight(_label, _confidence)

    if hedge_weight < 0.005:
        return dict(weights), no_change_meta

    # Strip existing VIXY so we don't double-count
    non_vixy = {k: v for k, v in weights.items() if k != "VIXY"}
    total_non_vixy = sum(non_vixy.values())

    if total_non_vixy <= 0:
        return dict(weights), no_change_meta

    # Scale all non-VIXY positions to fill (1 - hedge_weight) of portfolio
    target_non_vixy = 1.0 - hedge_weight
    scale = target_non_vixy / total_non_vixy
    hedged = {sym: w * scale for sym, w in non_vixy.items()}
    hedged["VIXY"] = hedge_weight

    meta = {
        "hedge_weight": hedge_weight,
        "regime_label": _label.value,
        "confidence":   _confidence,
        "vixy_before":  vixy_before,
        "vixy_after":   hedge_weight,
    }
    return hedged, meta
