"""ascent/causal/velocity.py
Pure-Python mechanism velocity score.
No external imports — safe to call from any context.
"""


def mechanism_velocity_score(
    current_value: float,
    baseline_value: float,
    threshold_value: float,
) -> float:
    """
    Returns a velocity score in [0.0, 1.0] measuring progress from baseline
    toward threshold.

    velocity = (current - baseline) / (threshold - baseline)

    Returns 0.0 when the range is zero (baseline == threshold).
    Clamped to [0.0, 1.0].
    """
    denom = threshold_value - baseline_value
    if abs(denom) < 1e-10:
        return 0.0
    v = (current_value - baseline_value) / denom
    return max(0.0, min(1.0, v))
