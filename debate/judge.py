"""
debate/judge.py
Synthesizes debate arguments into a structured verdict.

Returns a JSON verdict with:
    confidence:      float 0-1
    recommendation:  "proceed" | "reduce_size" | "halt_and_review"
    key_risks:       list of strings
    reasoning:       string
"""

import json
from ascent.llm.client import generate_structured, extended_thinking_completion, SONNET_MODEL as DEFAULT_MODEL
from debate.outcome_tracker import load_credibility_context, load_recent_verdict_outcomes


def run_judge(
    bull_argument: str,
    bear_argument: str,
    devils_argument: str,
    portfolio_state: dict,
    regime_arg: str = "",
    quant_check: str = "",
    round2_args: dict = None,
    disagreement_context: str = "",
) -> dict:
    """
    Synthesize all debate arguments into a single verdict.
    Now also receives regime specialist and quant sanity check.
    Defaults to 'reduce_size' on parsing failure — a broken judge should not greenlight execution.
    """
    round2_args = round2_args or {}

    # Quant issues are objective — if there are blocking issues, weight them heavily
    quant_block = quant_check and "✗" in quant_check

    # Load debater track records and judge's own recent history
    regime           = portfolio_state.get("us_regime")
    cred_context     = load_credibility_context(regime)
    recent_outcomes  = load_recent_verdict_outcomes(n=5)

    context = (
        f"Date: {portfolio_state.get('date', 'unknown')}\n"
        f"Regime: {portfolio_state.get('us_regime', 'unknown')}\n"
        f"Positions: {portfolio_state.get('n_positions', 0)}\n\n"
        f"BULL ARGUMENT:\n{bull_argument}\n\n"
        f"BEAR ARGUMENT:\n{bear_argument}\n\n"
        f"DEVIL'S ADVOCATE:\n{devils_argument}\n\n"
        + (f"REGIME SPECIALIST:\n{regime_arg}\n\n" if regime_arg else "")
        + (f"QUANT SANITY CHECK:\n{quant_check}\n\n" if quant_check else "")
        + ("NOTE: The quant sanity check found objective numerical issues (marked ✗). "
           "These are facts, not opinions — weight them heavily.\n"
           if quant_block else "")
        + (f"\n{cred_context}\n" if cred_context else "")
        + (f"\n{recent_outcomes}\n" if recent_outcomes else "")
        + (
            "\n\nROUND 2 — REBUTTALS (agents responding to each other):\n"
            + (f"BULL REBUTTAL:\n{round2_args['bull_rebuttal']}\n\n" if round2_args.get("bull_rebuttal") else "")
            + (f"BEAR REBUTTAL:\n{round2_args['bear_rebuttal']}\n\n" if round2_args.get("bear_rebuttal") else "")
            + (f"DEVIL'S ADVOCATE REBUTTAL:\n{round2_args['devils_advocate_rebuttal']}\n\n" if round2_args.get("devils_advocate_rebuttal") else "")
            + (f"REGIME SPECIALIST REBUTTAL:\n{round2_args['regime_specialist_rebuttal']}\n\n" if round2_args.get("regime_specialist_rebuttal") else "")
            if round2_args else ""
        )
    )

    system_prompt = (
        "You are the Portfolio Manager and final decision-maker at Ascent Capital. "
        "Synthesize ALL arguments — Round 1 (bull, bear, devil's advocate, regime specialist, "
        "quant sanity check) AND Round 2 rebuttals where agents engaged with each other — "
        "into a single verdict. Round 2 arguments are more focused; weight them accordingly.\n\n"
        "The regime specialist argues from historical regime patterns. "
        "The quant sanity check is objective math — if it flags issues, take them seriously.\n\n"
        "Respond with ONLY valid JSON (no markdown, no backticks, no text outside the JSON):\n"
        "{\n"
        '    "confidence": 0.0 to 1.0,\n'
        '    "recommendation": "proceed" or "reduce_size" or "halt_and_review",\n'
        '    "key_risks": ["risk 1", "risk 2", "risk 3"],\n'
        '    "reasoning": "one paragraph explaining your decision"\n'
        "}"
    )
    if disagreement_context:
        system_prompt += f"\n\n{disagreement_context}"

    try:
        raw = extended_thinking_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            model=DEFAULT_MODEL,
            max_tokens=5000,
            thinking_budget=3000,
        )

        # Extract the JSON object by brace-finding — robust against markdown fences,
        # trailing text, or any preamble the LLM adds before/after the JSON block.
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response (len={len(raw)})")
        raw = raw[start:end]

        verdict = json.loads(raw)

        # Validate required fields
        assert "confidence" in verdict
        assert "recommendation" in verdict
        assert verdict["recommendation"] in ("proceed", "reduce_size", "halt_and_review")

        return verdict

    except (json.JSONDecodeError, AssertionError, Exception) as e:
        print(f"[Judge] Failed to parse verdict ({e}) — defaulting to reduce_size (safe failure)")
        return {
            "confidence":      0.3,
            "recommendation":  "reduce_size",
            "key_risks":       [f"Judge parsing failed ({str(e)[:80]}) — safe default: reduce_size"],
            "reasoning":       "LLM output could not be parsed. Defaulting to reduce_size rather than proceed — a broken judge should not greenlight execution.",
        }
