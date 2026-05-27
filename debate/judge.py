"""
debate/judge.py
Synthesizes debate arguments into a structured verdict AND makes ONE position change.

Two outputs:
  1. verdict (proceed | reduce_size | halt_and_review) — gates execution as before
  2. position_changes — ONE specific weight adjustment with a falsifiable 10-day prediction

The position_changes output is the Adversarial Intelligence layer's key contribution.
Every change is logged by adversarial_authority.py and scored 10 days later.
"""

import json
from ascent.llm.client import generate_structured, extended_thinking_completion, SONNET_MODEL as DEFAULT_MODEL
from debate.outcome_tracker import load_credibility_context, load_recent_verdict_outcomes


def run_judge(
    bull_argument:    str,
    bear_argument:    str,
    devils_argument:  str,
    portfolio_state:  dict,
    regime_arg:       str = "",
    quant_check:      str = "",
    round2_args:      dict = None,
    disagreement_context: str = "",
    adversarial_context:  str = "",
    adversarial_engine:   dict = None,
) -> dict:
    """
    Synthesize all debate arguments into a verdict + ONE adversarial position change.

    Defaults to 'reduce_size' + no position_changes on parsing failure.
    Defaults to empty position_changes if no flag clears the authority threshold.
    """
    round2_args       = round2_args or {}
    adversarial_engine = adversarial_engine or {}
    quant_block       = quant_check and "✗" in quant_check

    regime           = portfolio_state.get("us_regime")
    cred_context     = load_credibility_context(regime)
    recent_outcomes  = load_recent_verdict_outcomes(n=5)

    # Load adversarial authority state for the prompt
    authority_context = ""
    try:
        from debate.adversarial_authority import format_authority_for_judge
        authority_context = format_authority_for_judge(regime or "unknown")
    except Exception:
        pass

    weights = portfolio_state.get("weights", {})

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
           "These are facts, not opinions — weight them heavily.\n\n" if quant_block else "")
        + (f"\n{cred_context}\n" if cred_context else "")
        + (f"\n{recent_outcomes}\n" if recent_outcomes else "")
        + (
            "\n\nROUND 2 — REBUTTALS:\n"
            + (f"BULL REBUTTAL:\n{round2_args['bull_rebuttal']}\n\n"
               if round2_args.get("bull_rebuttal") else "")
            + (f"BEAR REBUTTAL:\n{round2_args['bear_rebuttal']}\n\n"
               if round2_args.get("bear_rebuttal") else "")
            + (f"DEVIL'S ADVOCATE REBUTTAL:\n{round2_args['devils_advocate_rebuttal']}\n\n"
               if round2_args.get("devils_advocate_rebuttal") else "")
            + (f"REGIME SPECIALIST REBUTTAL:\n{round2_args['regime_specialist_rebuttal']}\n\n"
               if round2_args.get("regime_specialist_rebuttal") else "")
            if round2_args else ""
        )
        + (f"\n\n{adversarial_context}\n" if adversarial_context else "")
        + (f"\n\n{authority_context}\n" if authority_context else "")
    )

    top_flags   = adversarial_engine.get("top_flags", [])
    flags_json  = json.dumps(top_flags[:3], indent=2) if top_flags else "[]"

    system_prompt = (
        "You are the Portfolio Manager and final decision-maker at Ascent Capital. "
        "You synthesize two things:\n\n"
        "1. MACRO VERDICT: Is the overall portfolio direction correct? "
        "Options: 'proceed' | 'reduce_size' | 'halt_and_review'\n"
        "   - halt_and_review: ONLY for catastrophic risk (systemic event, fund-level error)\n"
        "   - reduce_size: broad portfolio risk warrants general trimming\n"
        "   - proceed: execute as proposed\n\n"
        "2. ONE ADVERSARIAL POSITION CHANGE: The single most important specific intervention. "
        "Pick from the top adversarial flags (ranked by priority). "
        "This is a FALSIFIABLE PREDICTION — state exactly what you expect to happen in 10 days.\n"
        "   Rules:\n"
        "   - ONE change only. Forced prioritization.\n"
        "   - Maximum weight change = what the authority level allows per intervention type\n"
        "   - SUSPENDED types may NOT be used\n"
        "   - Weight change must be ≥1% to be worth the intervention cost\n"
        "   - If no flag clears the bar, set position_changes to []\n"
        "   - Cannot increase any position (only reduce)\n"
        "   - Cannot drop below 1% (use 1% as floor)\n\n"
        f"Current positions for reference:\n"
        + "\n".join(f"  {s}: {w:.1%}" for s, w in
                    sorted(weights.items(), key=lambda x: -x[1])[:15])
        + f"\n\nTop adversarial flags:\n{flags_json}\n\n"
        "Respond with ONLY valid JSON (no markdown, no text outside the JSON):\n"
        "{\n"
        '    "confidence": 0.0 to 1.0,\n'
        '    "recommendation": "proceed" or "reduce_size" or "halt_and_review",\n'
        '    "key_risks": ["risk 1", "risk 2", "risk 3"],\n'
        '    "reasoning": "one paragraph",\n'
        '    "position_changes": [\n'
        '        {\n'
        '            "symbol": "TICKER",\n'
        '            "current_weight": 0.00,\n'
        '            "new_weight": 0.00,\n'
        '            "intervention_type": "adversarial_thesis|regime_sizing|coherence_risk|event_risk",\n'
        '            "reason": "specific reason referencing the data",\n'
        '            "prediction": "X underperforms/outperforms Y over next 10 trading days"\n'
        "        }\n"
        "    ]\n"
        "}"
    )
    if disagreement_context:
        system_prompt += f"\n\n{disagreement_context}"

    try:
        raw = extended_thinking_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ],
            model=DEFAULT_MODEL,
            max_tokens=5000,
            thinking_budget=3000,
        )

        start = raw.find("{")
        depth = 0
        end   = -1
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in response")

        verdict = json.loads(raw[start:end])

        assert "confidence" in verdict
        assert "recommendation" in verdict
        assert verdict["recommendation"] in ("proceed", "reduce_size", "halt_and_review")

        # Validate + clamp position_changes
        position_changes = verdict.get("position_changes", [])
        validated_changes = []
        for change in position_changes[:1]:  # enforce ONE change max
            sym     = change.get("symbol", "")
            new_w   = float(change.get("new_weight", 0))
            curr_w  = float(change.get("current_weight", weights.get(sym, 0)))
            itype   = change.get("intervention_type", "adversarial_thesis")

            # Authority check
            try:
                from debate.adversarial_authority import get_authority
                auth = get_authority(itype)
                if auth["suspended"]:
                    print(f"[Judge] Skipping {itype} intervention on {sym} — type suspended")
                    continue
                max_change = auth["allowed_change_pct"]
                # Clamp new_weight to not exceed max_change reduction
                min_allowed = max(0.01, curr_w - max_change)
                new_w = max(new_w, min_allowed)
            except Exception:
                pass

            if sym not in weights:
                continue
            if new_w >= weights[sym]:  # must be a reduction
                continue
            if abs(weights[sym] - new_w) < 0.005:  # <0.5pp change = not worth it
                continue

            change["current_weight"] = round(weights.get(sym, curr_w), 6)
            change["new_weight"]     = round(new_w, 6)
            validated_changes.append(change)

        verdict["position_changes"] = validated_changes

        return verdict

    except (json.JSONDecodeError, AssertionError, Exception) as e:
        print(f"[Judge] Failed to parse verdict ({e}) — defaulting to reduce_size (safe failure)")
        return {
            "confidence":      0.3,
            "recommendation":  "reduce_size",
            "key_risks":       [f"Judge parsing failed ({str(e)[:80]}) — safe default"],
            "reasoning":       "LLM output could not be parsed. Defaulting to reduce_size.",
            "position_changes": [],
        }
