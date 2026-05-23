import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a risk manager at an institutional quantitative fund. "
    "You have real position-level data: return since last rebalance, momentum, and alpha rank. "
    "Identify the SINGLE most dangerous assumption currently embedded in this portfolio — "
    "something the quant signals might not catch. Reference specific symbols and numbers. "
    "Think about: hidden correlations, regime fragility, crowding, event risk, liquidity, "
    "or thesis staleness. Be specific, quantitative, and under 120 words."
)


def generate_adversarial_challenge(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    regime: str = "unknown",
    position_health: Optional[Dict] = None,
) -> str:
    """
    Returns a single adversarial challenge string, or '' on failure.
    position_health: dict of {symbol: PositionHealth} from position_health.py
    """
    if not merged_weights:
        return ""

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return ""

    # Build health context — sorted worst first
    health_lines = []
    if position_health:
        from ascent.monitoring.position_health import format_health_summary
        health_lines = [
            "Position health (sorted by risk):",
            format_health_summary(position_health),
        ]
        deteriorating = [s for s, h in position_health.items() if h.flag == "DETERIORATING"]
        watching = [s for s, h in position_health.items() if h.flag == "WATCH"]
        if deteriorating:
            health_lines.append(f"DETERIORATING positions: {', '.join(deteriorating)}")
        if watching:
            health_lines.append(f"WATCH positions: {', '.join(watching)}")
    else:
        top = sorted(merged_weights.items(), key=lambda x: -x[1])[:8]
        health_lines = ["Positions: " + ", ".join(f"{s} ({w:.1%})" for s, w in top)]

    # Portfolio-level concentration context
    top5_weight = sum(sorted(merged_weights.values(), reverse=True)[:5])

    user_prompt = "\n".join([
        f"Date: {date}",
        f"Regime: {regime}",
        f"Portfolio: {len(merged_weights)} positions, top-5 concentration {top5_weight:.0%}",
        "",
    ] + health_lines + [
        "",
        "What is the single most dangerous assumption in this portfolio right now?",
        "Reference specific symbols and return/momentum numbers from the data above.",
    ])

    try:
        return generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=350,
            temperature=0.7,
        ).strip()
    except Exception as e:
        log.warning("[AdversarialDaily] Failed: %s", e)
        return ""
