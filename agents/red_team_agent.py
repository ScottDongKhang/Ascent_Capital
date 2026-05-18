# agents/red_team_agent.py
"""
Red team adversarial agent — attacks an AI PM portfolio proposal.

Uses SONNET_MODEL to generate the hardest possible bear critique of a proposed
portfolio and thesis. Never raises — returns empty string on any failure so that
the calling agent can fall through to its initial proposal.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ascent.llm.client import get_client, SONNET_MODEL

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a skeptical short-seller with unlimited capital and no position to protect. "
    "Your job is to find every reason a proposed portfolio could blow up. "
    "Be specific, quantitative where possible, and brutally concise. "
    "Do not offer constructive suggestions — only attack."
)


def run_red_team(portfolio: dict, thesis: dict, regime: str = "") -> str:
    """
    Given a proposed portfolio and thesis, return the hardest possible bear critique.

    Args:
        portfolio: Dict of {symbol: weight} from the AI PM's propose_portfolio call.
        thesis:    The investment thesis dict from the same call.
        regime:    Current market regime label (e.g. 'calm_bull', 'stressed', 'crisis').

    Returns:
        Plain-text critique string. Returns "" on any failure — never raises.
    """
    try:
        # Build a compact portfolio summary for the prompt
        top_positions = sorted(portfolio.items(), key=lambda x: -x[1])[:10]
        positions_str = "\n".join(
            f"  {sym}: {w:.1%}" for sym, w in top_positions
        ) if top_positions else "  (empty portfolio)"

        # Extract key thesis elements safely
        market_view = thesis.get("market_view", "not provided")
        key_risks = thesis.get("key_risks", [])
        what_could_be_wrong = thesis.get("what_could_be_wrong", "not provided")
        quant_overrides = thesis.get("quant_overrides", [])

        risks_str = (
            "\n".join(f"  - {r}" for r in key_risks)
            if key_risks
            else "  (none listed)"
        )
        overrides_str = (
            "\n".join(
                f"  - {o.get('symbol','?')}: {o.get('ai_action','?')} — {o.get('reason','?')}"
                for o in quant_overrides
            )
            if quant_overrides
            else "  (none)"
        )

        prompt = f"""You are reviewing the following portfolio proposal submitted by a portfolio manager.

CURRENT REGIME: {regime if regime else "unknown"}

PROPOSED PORTFOLIO:
{positions_str}

PM'S MARKET VIEW: {market_view}

PM'S STATED RISKS:
{risks_str}

PM'S WHAT COULD BE WRONG: {what_could_be_wrong}

PM'S QUANT OVERRIDES:
{overrides_str}

Your task:
1. For each of the top positions (up to 5), write one tight paragraph identifying the SINGLE worst-case scenario that could destroy that position. Be specific — cite sector dynamics, earnings risk, rate sensitivity, or valuation if relevant.
2. Identify ONE systemic risk the PM is ignoring — this should be a correlation, crowding, or macro tail risk that is NOT mentioned in the PM's risk list.
3. Call out any position that looks like narrative-driven momentum likely to mean-revert badly in a {regime if regime else "current"} regime environment.
4. End with a one-sentence "kill shot" — the single most dangerous thing about this portfolio as a whole.

Format: position-by-position paragraphs, then systemic risk section, then kill shot. Be concise — one paragraph max per position."""

        client = get_client()
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            system=_SYSTEM_PROMPT,
        )
        return response.content[0].text

    except Exception as exc:
        log.warning("[RedTeam] Red team critique failed: %s — skipping revision pass", exc)
        return ""
