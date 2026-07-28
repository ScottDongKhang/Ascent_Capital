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
from ascent.llm.client import generate_structured, extended_thinking_completion, DEFAULT_MODEL
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

    Defaults to 'proceed' (degraded=True) + no position_changes on parsing failure.
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
        "   - ONE change only, in EITHER direction (reduce OR conviction_press increase)\n"
        "   - For a conviction_press (increase): ONLY valid when ALL THREE are true:\n"
        "     (a) position is in quant top-quartile alpha rank\n"
        "     (b) bull argument explicitly cited clean crowding signal or positive tail asymmetry\n"
        "     (c) bear/devil's-advocate did NOT name this position as a specific concern\n"
        "   - For a reduction: cite the specific adversarial flag or risk\n"
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
        '            "intervention_type": "adversarial_thesis|regime_sizing|coherence_risk|event_risk|conviction_press",\n'
        '            "reason": "specific reason referencing the data",\n'
        '            "prediction": "X underperforms/outperforms Y over next 10 trading days"\n'
        "        }\n"
        "    ],\n"
        '    "reduction_pct": 0.10,\n'
        '    "protected_positions": [\n'
        '        {"symbol": "TICKER", "reason": "why this must NOT be cut now"}\n'
        "    ]\n"
        "}\n\n"
        "About the last two fields — they only matter when recommendation is\n"
        '"reduce_size", and they are how your reasoning actually reaches the book:\n'
        "  - reduction_pct: how much GROSS exposure to remove (0.02-0.25). The\n"
        "    remainder becomes cash. Omit for 0.10. This is a real de-grossing:\n"
        "    before 2026-07-28 a reduce_size verdict renormalized back to 1.0 and\n"
        "    so reduced nothing at all, three times in a row.\n"
        "  - protected_positions: names that must NOT be trimmed to fund the\n"
        "    reduction — hedges ahead of the catalyst they hedge, cash/T-bill\n"
        "    sleeves, anything load-bearing. State them here even if you also\n"
        "    explain them in `reasoning`: the execution layer reads ONLY this\n"
        "    field. On 2026-07-27 a verdict argued in prose that UUP and TLT must\n"
        "    not be cut 48h before FOMC, had nowhere to record it, and the\n"
        "    size-sorted fallback sold both.\n"
        "    The reduction is taken from everything else, so do not protect so\n"
        "    much that it cannot be funded."
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

        # Normalize protected_positions to a list of {symbol, reason} for names
        # actually held. The execution layer reads ONLY this field — prose in
        # `reasoning` is invisible to it (see ascent/execution/eod_runner.py
        # _verdict_protected_symbols). Unknown symbols are dropped so a
        # hallucinated ticker cannot shield nothing while shrinking the sleeve
        # available to fund the reduction.
        _prot_in = verdict.get("protected_positions")
        _prot_out = []
        if isinstance(_prot_in, list):
            for item in _prot_in:
                if isinstance(item, str):
                    sym, why = item, ""
                elif isinstance(item, dict):
                    sym, why = item.get("symbol"), str(item.get("reason", ""))
                else:
                    continue
                if not isinstance(sym, str) or not sym.strip():
                    continue
                sym = sym.strip().upper()
                if sym in weights:
                    _prot_out.append({"symbol": sym, "reason": why[:200]})
                else:
                    print(f"[Judge] Dropping protected_positions entry {sym} — not held")
        verdict["protected_positions"] = _prot_out

        # Clamp reduction_pct into the band the executor accepts, so an
        # out-of-range number is visible in the artifact rather than silently
        # replaced downstream.
        _red = verdict.get("reduction_pct")
        if isinstance(_red, (int, float)) and not isinstance(_red, bool) and _red > 0:
            verdict["reduction_pct"] = max(0.02, min(0.25, float(_red)))
        else:
            verdict.pop("reduction_pct", None)

        # Validate + clamp position_changes
        position_changes = verdict.get("position_changes", [])
        validated_changes = []
        for change in position_changes[:1]:  # enforce ONE change max
            sym     = change.get("symbol", "")
            new_w   = float(change.get("new_weight", 0))
            curr_w  = float(change.get("current_weight", weights.get(sym, 0)))
            itype   = change.get("intervention_type", "adversarial_thesis")

            if sym not in weights:
                continue

            # Authority check
            try:
                from debate.adversarial_authority import get_authority
                auth = get_authority(itype)
                if auth["suspended"]:
                    print(f"[Judge] Skipping {itype} intervention on {sym} — type suspended")
                    continue
                max_change = auth["allowed_change_pct"]
                # Clamp: new_w must stay within max_change of current in either direction
                new_w = max(curr_w - max_change, min(curr_w + max_change, new_w))
            except Exception:
                max_change = 0.02  # fallback: 2% max change

            # Drop trivially small changes (< 0.5pp)
            if abs(weights[sym] - new_w) < 0.005:
                continue

            # For conviction_press: enforce 10% max-weight hard cap
            MAX_WEIGHT = 0.10
            if itype == "conviction_press" and new_w > MAX_WEIGHT:
                new_w = MAX_WEIGHT
            if itype == "conviction_press" and new_w <= weights[sym]:
                continue  # a conviction_press must actually increase

            # For reductions: enforce 1% floor
            if itype != "conviction_press" and new_w < 0.01:
                new_w = 0.01
            if itype != "conviction_press" and new_w >= weights[sym]:
                continue  # a reduction must actually reduce

            change["current_weight"] = round(weights.get(sym, curr_w), 6)
            change["new_weight"]     = round(new_w, 6)
            validated_changes.append(change)

        verdict["position_changes"] = validated_changes

        return verdict

    except (json.JSONDecodeError, AssertionError, Exception) as e:
        print(f"[Judge] Failed to parse verdict ({e}) — defaulting to proceed (degraded)")
        return {
            "confidence":        0.0,
            "recommendation":    "proceed",
            "key_risks":         [f"judge_parse_failure: {str(e)[:80]}"],
            "reasoning":         "LLM output could not be parsed. Advisory layer degraded — proceeding without judge intervention.",
            "position_changes":  [],
            # Present-but-empty so downstream readers never see a missing key on
            # the degraded path. recommendation is "proceed" here, so neither is
            # consulted — but a later change to that default should not also
            # silently change de-grossing behaviour.
            "protected_positions": [],
            "degraded":          True,
        }
