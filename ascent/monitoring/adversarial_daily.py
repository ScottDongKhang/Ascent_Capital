import logging
from typing import Dict

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a risk manager at an institutional quantitative fund. "
    "Your job is to identify the SINGLE most dangerous assumption currently embedded "
    "in the portfolio — something the quant signals would not catch. "
    "Think about: hidden correlations, regime fragility, crowding, "
    "event risk, liquidity, or thesis staleness. "
    "Be specific, quantitative, and under 100 words."
)


def generate_adversarial_challenge(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    regime: str = "unknown",
) -> str:
    """
    Returns a single adversarial challenge string, or '' on failure.
    """
    if not merged_weights:
        return ""

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return ""

    top_positions = sorted(merged_weights.items(), key=lambda x: x[1], reverse=True)[:8]
    pos_str = ", ".join(f"{s} ({w:.1%})" for s, w in top_positions)

    user_prompt = (
        f"Date: {date}\n"
        f"Regime: {regime}\n"
        f"Top positions: {pos_str}\n"
        f"Total positions: {len(merged_weights)}\n\n"
        "What is the single most dangerous assumption in this portfolio right now?"
    )

    try:
        return generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.7,
        ).strip()
    except Exception as e:
        log.warning("[AdversarialDaily] Failed: %s", e)
        return ""
