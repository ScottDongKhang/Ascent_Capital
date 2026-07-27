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

from ascent.llm.client import get_client, SONNET_MODEL, extract_text

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a skeptical short-seller with unlimited capital and no position to protect. "
    "Your job is to find every reason a proposed portfolio could blow up. "
    "Be specific, quantitative where possible, and brutally concise. "
    "Do not offer constructive suggestions — only attack."
)


def run_red_team(
    portfolio: dict,
    thesis: dict,
    regime: str = "",
    quant_weights: dict | None = None,
) -> str:
    """
    Given a proposed portfolio and thesis, return the hardest possible bear critique.

    Args:
        portfolio:     Dict of {symbol: weight} from the AI PM's propose_portfolio call.
        thesis:        The investment thesis dict from the same call.
        regime:        Current market regime label (e.g. 'calm_bull', 'stressed', 'crisis').
        quant_weights: The pure quant baseline weights before AI PM overrides. When provided,
                       the red team attacks the AI PM vs quant deltas specifically.

    Returns:
        Plain-text critique string. Returns "" on any failure — never raises.
    """
    try:
        top_positions = sorted(portfolio.items(), key=lambda x: -x[1])[:10]
        positions_str = "\n".join(
            f"  {sym}: {w:.1%}" for sym, w in top_positions
        ) if top_positions else "  (empty portfolio)"

        market_view = thesis.get("market_view", "not provided")
        key_risks = thesis.get("key_risks", [])
        what_could_be_wrong = thesis.get("what_could_be_wrong", "not provided")
        quant_overrides = thesis.get("quant_overrides", [])
        pre_mortem = thesis.get("pre_mortem", "not provided")

        risks_str = (
            "\n".join(f"  - {r}" for r in key_risks) if key_risks else "  (none listed)"
        )
        overrides_str = (
            "\n".join(
                f"  - {o.get('symbol','?')} [{o.get('override_type','?')}]: "
                f"{o.get('ai_action','?')} — {o.get('reason','?')}"
                for o in quant_overrides
            )
            if quant_overrides else "  (none)"
        )

        # Build vs-quant delta section when quant baseline is available
        delta_section = ""
        if quant_weights:
            all_syms = set(list(portfolio.keys()) + list(quant_weights.keys()))
            deltas = []
            for sym in all_syms:
                ai_w  = portfolio.get(sym, 0.0)
                q_w   = quant_weights.get(sym, 0.0)
                diff  = ai_w - q_w
                if abs(diff) >= 0.02:
                    direction = "ADDED" if q_w == 0 else ("REMOVED" if ai_w == 0 else
                                ("INCREASED" if diff > 0 else "REDUCED"))
                    deltas.append((sym, q_w, ai_w, diff, direction))
            deltas.sort(key=lambda x: -abs(x[3]))

            if deltas:
                delta_lines = []
                for sym, q_w, ai_w, diff, direction in deltas[:8]:
                    delta_lines.append(
                        f"  {sym}: Quant={q_w:.1%} → AI PM={ai_w:.1%} ({direction}, delta={diff:+.1%})"
                    )
                delta_section = (
                    "\nAI PM vs QUANT BASELINE DELTAS (positions changed ≥2%):\n"
                    + "\n".join(delta_lines)
                    + "\n"
                )

        prompt = f"""You are reviewing a portfolio proposal from an AI portfolio manager who has overridden the pure quant baseline.

CURRENT REGIME: {regime if regime else "unknown"}

PROPOSED PORTFOLIO:
{positions_str}
{delta_section}
PM'S MARKET VIEW: {market_view}

PM'S PRE-MORTEM: {pre_mortem}

PM'S STATED RISKS:
{risks_str}

PM'S WHAT COULD BE WRONG: {what_could_be_wrong}

PM'S QUANT OVERRIDES:
{overrides_str}

Your task — be brutally specific, no soft language:

1. OVERRIDE ATTACKS: For each AI PM override vs the quant baseline (from the delta section above), state the worst-case outcome if the quant was RIGHT and the AI PM was WRONG. Quantify the cost: if the PM reduced SATS from 10% to 4% and SATS rallies 40%, that is a 2.4% alpha drag. Make the PM feel the cost of being wrong.

2. POSITION ATTACKS: For each of the top-5 positions, identify the single most dangerous scenario specific to that name — not a generic "could fall" but a specific catalyst (earnings miss, rate shock, regulatory risk, supply chain disruption).

3. SYSTEMIC BLIND SPOT: Identify ONE correlation or crowding risk the PM is ignoring that is NOT in their risk list. This must be specific — name the positions that correlate and the scenario that triggers it.

4. KILL SHOT: One sentence. The single most dangerous thing about this portfolio that the PM is rationalizing away.

Format: override attacks first, then position attacks, then blind spot, then kill shot. One paragraph max per item. No constructive suggestions — only attack."""

        client = get_client()
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,   # headroom: thinking shares this budget
            messages=[{"role": "user", "content": prompt}],
            system=_SYSTEM_PROMPT,
        )
        return extract_text(response)

    except Exception as exc:
        log.warning("[RedTeam] Red team critique failed: %s — skipping revision pass", exc)
        return ""
