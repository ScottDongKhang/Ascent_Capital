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
from ascent.llm.prompt_loader import get_prompt

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = get_prompt("red_team.system")


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

        from ascent.llm.prompt_loader import get_prompt_formatted
        prompt = get_prompt_formatted(
            "red_team.attack",
            regime=regime if regime else "unknown",
            positions_str=positions_str,
            delta_section=delta_section,
            market_view=market_view,
            pre_mortem=pre_mortem,
            risks_str=risks_str,
            what_could_be_wrong=what_could_be_wrong,
            overrides_str=overrides_str,
        )

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
