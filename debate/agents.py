"""
debate/agents.py
Debate agent definitions for Ascent Capital.

Agents:
  - Bull analyst       — Druckenmiller persona: asymmetric risk-reward, momentum, aggressive when conviction high
  - Bear analyst       — Burry persona: downside first, FCF fragility, terse/data-driven, contrarian
  - Devil's advocate   — Taleb persona: convexity/concavity, fat tails, turkey problem, via negativa
  - Regime specialist  — LLM, Haiku
  - Quant sanity checker — pure Python, no LLM

Uses claude-sonnet-5 (DEBATE_MODEL) for bull/bear/devil.
Uses claude-haiku-4-5-20251001 (HAIKU_MODEL) for regime specialist.
Quant sanity checker uses no LLM.
"""

import logging

from ascent.llm.client import generate_structured, tool_completion, SONNET_MODEL as DEBATE_MODEL, HAIKU_MODEL
from debate.outcome_tracker import load_credibility_context

log = logging.getLogger(__name__)


def _get_agent_track_record(agent_name: str, regime: str) -> str:
    """Return formatted track record for injection into agent system prompts.
    Returns empty string when no data available (dormant until ~August 2026)."""
    try:
        from debate.outcome_tracker import CREDIBILITY_PATH
        import json as _json
        if not CREDIBILITY_PATH.exists():
            return ""
        cred       = _json.loads(CREDIBILITY_PATH.read_text())
        regime_key = str(regime).lower()
        accuracy   = cred.get("by_regime", {}).get(regime_key, {}).get(agent_name)
        n          = cred.get("sample_counts", {}).get(regime_key, {}).get(agent_name, 0)
        if accuracy is None or n < 10:
            return ""
        warning = (
            " Calibrate confidence DOWN — your historical accuracy here is below 50%."
            if accuracy < 0.50 else
            " Your track record here is solid."
        )
        return (
            f"\nYOUR TRACK RECORD in {regime.upper()} regime: "
            f"{accuracy:.0%} accuracy over {n} debates.{warning}"
        )
    except Exception as _exc:
        log.debug("[Debate] _get_agent_track_record(%s, %s) failed: %s", agent_name, regime, _exc)
        return ""


def _build_context(portfolio_state: dict) -> str:
    lines = [
        f"Date: {portfolio_state.get('date', 'unknown')}",
        f"Regime (equities): {portfolio_state.get('us_regime', 'unknown')}",
        f"Regime (macro): {portfolio_state.get('macro_regime', 'unknown')}",
        f"Total positions: {portfolio_state.get('n_positions', 0)}",
        f"Capital allocation: {portfolio_state.get('allocation', {})}",
        "",
        "Proposed portfolio weights:",
    ]
    for sym, w in sorted(portfolio_state.get("weights", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {sym}: {w:.2%}")

    catalyst_ctx = portfolio_state.get("catalyst_context", {})
    catalyst_text = catalyst_ctx.get("catalyst_text", "") if isinstance(catalyst_ctx, dict) else ""
    if catalyst_text and "no upcoming" not in catalyst_text.lower():
        lines.append("")
        lines.append("Upcoming catalysts:")
        lines.append(catalyst_text)

    memory_ctx = portfolio_state.get("memory_context", [])
    if not memory_ctx:
        # Actively query memory for relevant past verdicts
        try:
            from memory.r2r_interface import query_memory, format_memory_context
            regime_label = portfolio_state.get("us_regime", "unknown")
            top_positions = sorted(
                (portfolio_state.get("weights") or {}).items(),
                key=lambda x: -x[1]
            )[:3]
            top_str = ", ".join(f"{s}({w:.0%})" for s, w in top_positions)
            memory_ctx = query_memory(
                f"verdict outcome {regime_label} regime {top_str}",
                n=3,
            )
        except Exception:
            pass
    if memory_ctx:
        try:
            from memory.r2r_interface import format_memory_context
            memory_text = format_memory_context(memory_ctx)
            if "no relevant" not in memory_text.lower():
                lines.append("")
                lines.append(memory_text)
        except Exception:
            pass

    # Post-trade reflections — structured lessons from past outcomes in this regime
    regime_for_reflection = portfolio_state.get("us_regime", "unknown")
    try:
        from memory.reflection_agent import load_recent_reflections, format_reflections_for_context
        recent_reflections = load_recent_reflections(regime=regime_for_reflection, n=3)
        reflection_text = format_reflections_for_context(recent_reflections)
        if reflection_text:
            lines.append("")
            lines.append(reflection_text)
    except Exception:
        pass

    quant_ctx = portfolio_state.get("quant_context")
    if quant_ctx and isinstance(quant_ctx, dict):
        summary = quant_ctx.get("summary_text", "")
        if summary:
            lines.append("")
            lines.append(summary)

    return "\n".join(lines)


# ── Section builders — each extracts one topic from portfolio_state ───────────

def _section_weights(ps: dict) -> str:
    weights = ps.get("weights", {})
    if not weights:
        return ""
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    lines = ["Current positions (by weight):"]
    for sym, w in sorted_w[:15]:
        lines.append(f"  {sym}: {w:.1%}")
    if ps.get("n_positions"):
        lines.append(f"Total positions: {ps['n_positions']}")
    return "\n".join(lines)


def _section_momentum(ps: dict) -> str:
    meta = ps.get("metadata", {})
    alpha = meta.get("alpha_scores", {})
    if not alpha:
        return "(Momentum data not available — alpha_scores not in metadata)"
    mom_keys = [k for k in alpha if "mom" in k.lower() or "trend" in k.lower()]
    if not mom_keys:
        return "(No momentum signals found in alpha_scores)"
    lines = ["Momentum signals (cross-sectional z-scores):"]
    for k in sorted(mom_keys)[:5]:
        lines.append(f"  {k}: {alpha[k]}")
    return "\n".join(lines)


def _section_fundamental(ps: dict) -> str:
    meta = ps.get("metadata", {})
    alpha = meta.get("alpha_scores", {})
    fund_keys = [k for k in alpha if any(x in k.lower()
                 for x in ("fundamental", "earnings", "profitability", "accrual"))]
    if not fund_keys:
        return "(Fundamental/earnings signals not available in metadata)"
    lines = ["Fundamental signals:"]
    for k in sorted(fund_keys)[:5]:
        lines.append(f"  {k}: {alpha[k]}")
    return "\n".join(lines)


def _section_concentration(ps: dict) -> str:
    weights = ps.get("weights", {})
    if not weights:
        return ""
    vals = sorted(weights.values(), reverse=True)
    top1 = vals[0] if vals else 0
    top3 = sum(vals[:3]) if len(vals) >= 3 else sum(vals)
    lines = [
        "Concentration stats:",
        f"  Largest position: {top1:.1%}",
        f"  Top-3 combined:   {top3:.1%}",
        f"  N positions:      {len(weights)}",
    ]
    return "\n".join(lines)


def _section_regime(ps: dict) -> str:
    lines = []
    if ps.get("us_regime"):
        lines.append(f"US regime: {ps['us_regime']}")
    if ps.get("macro_regime"):
        lines.append(f"Macro regime: {ps['macro_regime']}")
    meta = ps.get("metadata", {})
    if meta.get("regime_entropy") is not None:
        lines.append(f"Regime entropy: {meta['regime_entropy']:.3f}")
    if meta.get("regime_confidence") is not None:
        lines.append(f"Regime confidence: {meta['regime_confidence']:.2f}")
    return "\n".join(lines) if lines else "(Regime data not available)"


def _section_macro(ps: dict) -> str:
    meta = ps.get("metadata", {})
    macro = meta.get("macro_snapshot", {})
    if not macro:
        return "(Macro snapshot not in metadata)"
    lines = ["Macro indicators:"]
    for k, v in list(macro.items())[:8]:
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _section_tail(ps: dict) -> str:
    meta = ps.get("metadata", {})
    mc = meta.get("monte_carlo", {})
    if not mc:
        return "(Monte Carlo data not in metadata)"
    lines = ["Monte Carlo tail scenarios:"]
    for k in ("p5", "p10", "p50", "p90", "p95"):
        if k in mc:
            lines.append(f"  {k}: {mc[k]:+.2%}")
    return "\n".join(lines)


def _section_allocation(ps: dict) -> str:
    alloc = ps.get("allocation", {})
    if not alloc:
        return ""
    lines = ["Capital allocation by agent:"]
    for agent_id, pct in alloc.items():
        lines.append(f"  {agent_id}: {pct:.1%}")
    return "\n".join(lines)


def _section_factor_exposures(ps: dict) -> str:
    """Factor exposure context — devil's advocate only."""
    try:
        import pandas as pd
        from datetime import date
        from ascent.risk.factor_exposure import format_exposure_context
        weights_raw = ps.get("weights", {})
        if not weights_raw:
            return ""
        weights = pd.Series({k: float(v) for k, v in weights_raw.items()})
        as_of = ps.get("as_of_date", date.today().isoformat())
        ctx = format_exposure_context(weights, as_of)
        return ctx if ctx else ""
    except Exception:
        return ""


def _section_altdata_positives(ps: dict) -> str:
    """
    Bull-only: alt data positives + positions the adversarial engine cannot find a strong short case for.
    Symbols with adversarial_score < 0.3 = hard to short = structural long quality.
    """
    adv = ps.get("adversarial_engine", {})
    adv_scores = adv.get("adversarial_scores", {})
    weights    = ps.get("weights", {})

    lines = []

    # Positions adversarial engine failed to find a strong bear case for
    hard_to_short = [
        (sym, data)
        for sym, data in adv_scores.items()
        if data.get("adversarial_score", 0.5) < 0.30 and sym in weights
    ]
    if hard_to_short:
        lines.append("Positions with weak adversarial case (structural quality — hard to short):")
        for sym, data in sorted(hard_to_short, key=lambda x: x[1]["adversarial_score"]):
            score = data["adversarial_score"]
            lines.append(f"  {sym} ({weights[sym]:.1%}): adversarial_score={score:.2f} — {data.get('short_thesis', 'no compelling short case')}")

    # Positive alt data signals
    import json as _j
    from pathlib import Path as _Path
    sec_path = _Path("data_cache/altdata_sec_detail.json")
    if sec_path.exists():
        try:
            detail = _j.loads(sec_path.read_text())
            pos_signals = []
            for sym in weights:
                if sym in detail:
                    tone = detail[sym].get("tone_score", 0) or 0
                    rev  = detail[sym].get("revenue_momentum", 0) or 0
                    if tone > 0.2 or rev > 0.3:
                        pos_signals.append((sym, tone, rev))
            if pos_signals:
                lines.append("\nPositive SEC signals (management tone + revenue momentum):")
                for sym, tone, rev in sorted(pos_signals, key=lambda x: -(x[1]+x[2]))[:5]:
                    lines.append(f"  {sym}: tone={tone:+.2f} revenue_mom={rev:+.2f}")
        except Exception:
            pass

    return "\n".join(lines) if lines else "(No strong alt-data positives or weak-short positions identified)"


def _section_adversarial_flags(ps: dict) -> str:
    """
    Bear-only: adversarial engine flags + positions that moved >15% since entry (event risk).
    Gives Bear evidence-based ammunition rather than generic risk arguments.
    """
    adv      = ps.get("adversarial_engine", {})
    scores   = adv.get("adversarial_scores", {})
    sizing   = adv.get("sizing_recommendations", {})
    top_flags = adv.get("top_flags", [])
    weights  = ps.get("weights", {})

    lines = []

    # Top adversarial flags
    if top_flags:
        lines.append("Adversarial Intelligence — top risk flags (pre-computed):")
        for flag in top_flags[:4]:
            sym  = flag["symbol"]
            lines.append(
                f"  {sym} ({flag['current_weight']:.1%}): {flag['reason'][:80]}"
            )
            if flag.get("suggested_weight"):
                lines.append(f"    → Suggested trim to {flag['suggested_weight']:.1%} "
                             f"[type: {flag['intervention_type']}]")

    # Event momentum positions (single-day >15% since entry = elevated revert risk)
    event_movers = [
        (sym, d)
        for sym, d in sizing.items()
        if d.get("position_type") == "event_momentum" and sym in weights
    ]
    if event_movers:
        lines.append("\nEvent-momentum positions (>15% single-day move in past 30d — mean-revert risk):")
        for sym, d in event_movers:
            lines.append(
                f"  {sym}: current={d['current_weight']:.1%}, "
                f"optimal={d['optimal_weight']:.1%}, deviation={d['deviation']:+.1%}"
            )

    # Oversized positions
    oversized = [(sym, d) for sym, d in sizing.items() if d.get("oversized") and sym in weights]
    if oversized:
        lines.append("\nOversized positions per regime-conditional rules:")
        for sym, d in oversized[:4]:
            lines.append(f"  {sym}: {d['recommendation'][:80]}")

    # Upcoming earnings (from catalyst context)
    cat = ps.get("catalyst_context", {})
    if isinstance(cat, dict) and cat.get("upcoming_events"):
        events = [e for e in cat["upcoming_events"] if e.get("days_away", 99) <= 7]
        if events:
            lines.append("\nEarnings within 7 days (binary event risk):")
            for e in events[:4]:
                lines.append(f"  {e.get('symbol', '?')}: {e.get('event_type', '?')} in {e.get('days_away', '?')} days")

    return "\n".join(lines) if lines else "(No adversarial flags — engine data not available)"


def _section_coherence(ps: dict) -> str:
    """
    Devil's Advocate only: portfolio coherence analysis from the adversarial engine.
    This is what the quant model and AI PM cannot see — emergent portfolio-level risk.
    """
    adv = ps.get("adversarial_engine", {})
    coh = adv.get("coherence", {})
    if not coh:
        return "(Portfolio coherence data not available)"

    n_pos  = coh.get("n_positions", 0)
    n_bets = coh.get("n_independent_bets", 0)
    dom    = coh.get("dominant_exposure", "unknown")
    flip   = coh.get("regime_flip_impact_estimate", "N/A")
    lw     = coh.get("largest_cluster_weight", 0)

    lines = [
        "Portfolio Coherence Analysis (what the quant model cannot see):",
        f"  {n_pos} positions → {n_bets} independent bets",
        f"  Dominant exposure: {dom} ({lw:.1%} of portfolio)",
        f"  Regime flip sensitivity: {flip}",
    ]

    clusters = coh.get("narrative_clusters", [])
    if clusters:
        lines.append("  Narrative clusters (each is one bet, not many):")
        for c in clusters[:5]:
            syms = ", ".join(p["symbol"] for p in c["positions"][:3])
            extra = f" +{len(c['positions'])-3}" if len(c["positions"]) > 3 else ""
            lines.append(f"    {c['name']}: {c['total_weight']:.1%} [{syms}{extra}]")

    em_w   = coh.get("em_weight", 0)
    comm_w = coh.get("commodity_weight", 0)
    if em_w > 0.15:
        lines.append(f"  WARNING: EM exposure {em_w:.1%} — single policy/currency event affects entire cluster")
    if comm_w > 0.15:
        lines.append(f"  WARNING: Commodity exposure {comm_w:.1%} — correlated to same supply/demand shock")

    return "\n".join(lines)


def _section_tail_asymmetry(ps: dict) -> str:
    """
    Taleb-specific: compute tail asymmetry from Monte Carlo percentiles.
    Tail ratio < 1.0 means the portfolio has more downside than upside — concave payoff.
    """
    meta = ps.get("metadata", {})
    mc = meta.get("monte_carlo", {})
    p5  = mc.get("p5")
    p50 = mc.get("p50")
    p95 = mc.get("p95")

    if p5 is None or p50 is None or p95 is None:
        return "(Monte Carlo tail data not available — cannot compute tail asymmetry)"

    upside   = p95 - p50
    downside = p50 - p5

    if downside == 0:
        return f"Monte Carlo: p5={p5:+.2%} p50={p50:+.2%} p95={p95:+.2%} — tail asymmetry incalculable (zero downside deviation)"

    tail_ratio = upside / abs(downside)
    if tail_ratio > 1.2:
        shape = "CONVEX — upside exceeds downside"
    elif tail_ratio > 0.8:
        shape = "SYMMETRIC — balanced tail exposure"
    else:
        shape = "CONCAVE — downside exceeds upside (fragile)"

    regime_entropy = meta.get("regime_entropy")
    entropy_str = f"  Regime entropy: {regime_entropy:.3f}" if regime_entropy is not None else ""
    turkey_warning = ""
    if regime_entropy is not None and regime_entropy < 0.5:
        turkey_warning = "\n  TURKEY PROBLEM: regime entropy extremely low — complacency risk is high"

    return (
        f"Tail asymmetry (Monte Carlo):\n"
        f"  Upside (p50→p95): +{upside:.2%}\n"
        f"  Downside (p5→p50): -{downside:.2%} (loss potential from median)\n"
        f"  Tail ratio: {tail_ratio:.2f} — {shape}"
        + (f"\n{entropy_str}" if entropy_str else "")
        + turkey_warning
    )


# ── Agent context assembler ────────────────────────────────────────────────────

_AGENT_SECTIONS = {
    "bull": [_section_weights, _section_momentum, _section_fundamental,
             _section_altdata_positives, _section_allocation],
    "bear": [_section_weights, _section_adversarial_flags, _section_concentration,
             _section_regime, _section_allocation],
    "devils_advocate": [_section_weights, _section_coherence, _section_regime,
                        _section_tail, _section_tail_asymmetry, _section_factor_exposures, _section_allocation],
    "regime_specialist": [_section_regime, _section_macro],
}


def _build_agent_context(portfolio_state: dict, agent_role: str) -> str:
    """
    Build agent-specific context — no agent sees the full portfolio state.
    Falls back to _build_context() for unknown roles.
    """
    builders = _AGENT_SECTIONS.get(agent_role)
    if builders is None:
        return _build_context(portfolio_state)

    date_str = portfolio_state.get("date", "unknown")
    sections = [f"[{agent_role.upper()} VIEW — {date_str}]"]
    for builder in builders:
        try:
            block = builder(portfolio_state)
            if block:
                sections.append(block)
        except Exception as _exc:
            log.debug("[Debate] Section builder %s failed: %s", builder.__name__, _exc)
    return "\n\n".join(sections)


_EVIDENCE_RULE = (
    " EVIDENCE RULE: Every number you cite (return %, position weight, scenario "
    "percentile, ratio) must appear in the Portfolio context you were given. Write "
    "[FROM CONTEXT] immediately after any number you quote from it. If you cannot "
    "find a number in the context, state the observation qualitatively rather than "
    "inventing a value."
)


def run_bull_agent(portfolio_state: dict) -> str:
    context      = _build_agent_context(portfolio_state, "bull")
    regime       = portfolio_state.get("us_regime", "unknown")
    cred_context = load_credibility_context(regime)
    track_record = _get_agent_track_record("bull", regime)
    user_prompt  = f"Portfolio context:\n{context}"
    if cred_context:
        user_prompt += f"\n\n{cred_context}"
    user_prompt += "\n\nMake the bull case for these trades."
    return generate_structured(
        system_prompt=(
            "You are the Druckenmiller-inspired Bull Analyst at Ascent Capital. "
            "Reason like Stanley Druckenmiller: hunt for ASYMMETRIC setups where upside "
            "significantly outweighs downside. Lead with momentum — which positions are "
            "accelerating and have regime tailwinds? Compute the implicit risk-reward: "
            "if the Monte Carlo p95 gain is 3x the p5 loss, say so explicitly. "
            "Be aggressive when conviction is high. Identify which positions the "
            "adversarial engine rated hard-to-short — these are your structural longs. "
            "Call out if the bear or devil tends to over-warn in this regime. "
            "Cut your argument the moment the thesis changes — don't defend stale positions. "
            "Keep under 200 words."
            f"{track_record}"
            f"{_EVIDENCE_RULE}"  # EVIDENCE RULE — see module-level _EVIDENCE_RULE constant
        ),
        user_prompt=user_prompt,
        model=DEBATE_MODEL,
        temperature=0.6,
        use_cache=True,
    )


def run_bear_agent(portfolio_state: dict) -> str:
    context      = _build_agent_context(portfolio_state, "bear")
    regime       = portfolio_state.get("us_regime", "unknown")
    cred_context = load_credibility_context(regime)
    track_record = _get_agent_track_record("bear", regime)
    user_prompt  = f"Portfolio context:\n{context}"
    if cred_context:
        user_prompt += f"\n\n{cred_context}"
    user_prompt += (
        "\n\nMake the bear case against these trades. "
        "Use the available tools to verify sector concentrations, VaR, and momentum "
        "before making quantitative claims."
    )
    try:
        from debate.agent_tools import DEBATE_TOOLS, execute_tool
        return tool_completion(
            system_prompt=(
                "You are the Burry-inspired Bear Analyst at Ascent Capital. "
                "Reason like Michael Burry: DOWNSIDE FIRST. Lead with the weakest "
                "link — which position has the worst quantitative support (highest "
                "adversarial score, weakest momentum, binary event risk)? "
                "What is the market missing about the downside? "
                "Cite hard numbers: adversarial scores, VaR, concentration stats, "
                "event risk within 7 days. Be terse and data-driven — no fluff. "
                "Contrarian edge: if the adversarial engine found no compelling bear "
                "case, say so rather than manufacturing one. Capital preservation "
                "comes before any trade. Use the provided tools to compute sector "
                "concentration and VaR BEFORE making quantitative claims. "
                "Calibrate from your historical accuracy in this regime. Keep under 200 words."
                f"{track_record}"
                f"{_EVIDENCE_RULE}"  # EVIDENCE RULE — see module-level _EVIDENCE_RULE constant
            ),
            user_prompt=user_prompt,
            tools=DEBATE_TOOLS,
            tool_executor=execute_tool,
            model=DEBATE_MODEL,
            max_tokens=800,
            max_tool_calls=2,
            use_cache=True,
        )
    except Exception as e:
        log.warning("[Bear] Tool completion failed (%s), falling back to generate_structured", e)
        return generate_structured(
            system_prompt=(
                "You are the Burry-inspired Bear Analyst at Ascent Capital. "
                "Lead with downside — weakest position, hardest number. Terse. Under 200 words."
                f"{_EVIDENCE_RULE}"  # EVIDENCE RULE — see module-level _EVIDENCE_RULE constant
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
            use_cache=True,
        )


def run_devils_advocate(portfolio_state: dict) -> str:
    context = _build_agent_context(portfolio_state, "devils_advocate")

    # Add scenario numbers if available
    scenario_summary = portfolio_state.get("scenario_summary", "")
    scenario_results = portfolio_state.get("scenario_results", [])

    scenario_context = ""
    if scenario_results:
        lines = ["\nWorst-case scenario analysis (Monte Carlo p5/p50):"]
        for s in scenario_results:
            lines.append(
                f"  {s['scenario']}: worst case {s['worst_case']:+.1f}% | "
                f"base case {s['base_case']:+.1f}% | "
                f"best case {s['best_case']:+.1f}%"
            )
        scenario_context = "\n".join(lines)

    # Inject weekend scenario plan (flagged scenarios only — highest-prob tail risks)
    weekend_scenario_context = ""
    try:
        import json as _json
        from pathlib import Path as _Path
        from datetime import date as _date, timedelta as _td
        sp = _Path("data_cache/scenario_plan.json")
        if sp.exists():
            sp_data = _json.loads(sp.read_text())
            cutoff = (_date.today() - _td(days=8)).isoformat()
            if sp_data.get("as_of", "") >= cutoff:
                flagged = [s for s in sp_data.get("scenarios", []) if s.get("flagged")]
                if flagged:
                    lines = ["\nWeekend adversarial scenario plan — FLAGGED (prob ≥40%):"]
                    for s in flagged:
                        prob = s.get("probability", 0)
                        impact = s.get("impact", {}).get("total_impact_pct", 0)
                        lines.append(
                            f"  {s['name']}: {prob:.0%} prob | portfolio impact {impact:+.1f}% | "
                            f"action: {s.get('pre_emptive_action', 'n/a')}"
                        )
                    weekend_scenario_context = "\n".join(lines)
    except Exception:
        pass

    regime       = portfolio_state.get("us_regime", "unknown")
    cred_context = load_credibility_context(regime)
    track_record = _get_agent_track_record("devils_advocate", regime)
    user_prompt  = f"Portfolio context:\n{context}"
    if scenario_context:
        user_prompt += f"\n{scenario_context}"
    if weekend_scenario_context:
        user_prompt += f"\n{weekend_scenario_context}"
    if cred_context:
        user_prompt += f"\n\n{cred_context}"
    user_prompt += "\n\nWhat is the most dangerous blind spot? Use the scenario numbers."

    # MiroFish crowd timing context
    _mirofish_context = ""
    mirofish_sent = portfolio_state.get("mirofish_sentiment")
    if isinstance(mirofish_sent, dict):
        alignment = mirofish_sent.get("alignment_score", 1.0)
        if alignment < 0.50:
            flags = mirofish_sent.get("warning_flags", [])
            crowd = mirofish_sent.get("overall_sentiment", "unknown")
            flags_str = "\n".join(f"  - {f}" for f in flags[:4]) if flags else "  (no specific flags)"
            _mirofish_context = (
                f"\n\n══ MIROFISH CROWD INTELLIGENCE (alignment={alignment:.2f} — USE THIS) ══\n"
                f"The crowd simulation returned '{crowd.upper()}' sentiment, "
                f"but the AI PM is proposing AMPLIFY at full weight.\n"
                f"Alignment score {alignment:.2f} < 0.50 means the crowd's reading diverges from the thesis.\n"
                f"CROWD WARNING FLAGS:\n{flags_str}\n"
                f"REQUIRED: Attack the AI PM's AMPLIFY picks on crowd timing grounds. "
                f"Why might the crowd be seeing something the AI PM missed? "
                f"Use these flags as specific evidence. This is a FALSIFIABLE argument — "
                f"if the crowd is wrong, say why. If they're right, say so explicitly."
            )

    _da_system_prompt = (
        "You are the Taleb-inspired Devil's Advocate at Ascent Capital. "
        "Reason like Nassim Taleb — your job is FRAGILITY DETECTION and TAIL RISK, "
        "not generic pessimism. Follow this checklist:\n"
        "1. CONVEXITY CHECK: Is this portfolio CONVEX (limited downside, large upside) "
        "or CONCAVE (large downside, limited upside)? Use the tail asymmetry ratio in "
        "your context — if tail_ratio < 1.0, the portfolio is concave and you must name it.\n"
        "2. TURKEY PROBLEM: Is regime entropy dangerously low? Low entropy = complacency. "
        "The turkey feels safe until Thanksgiving. If regime confidence is high AND vol is "
        "low, flag the 'calm before the storm' explicitly.\n"
        "3. VIA NEGATIVA: What single event — NOT priced in by the quant model — would "
        "most harm this portfolio? Fat tails come from the events the model doesn't see.\n"
        "4. COHERENCE FRAGILITY: Do the portfolio clusters all break together under the "
        "same tail event? If so, the 15 positions are really one bet.\n"
        "Use Taleb's vocabulary: antifragile, convexity, barbell, turkey problem, "
        "via negativa, fat tail, Lindy effect. Cite the Monte Carlo numbers. "
        "Be specific. Keep under 150 words."
        f"{track_record}"
        f"{_EVIDENCE_RULE}"  # EVIDENCE RULE — see module-level _EVIDENCE_RULE constant
        f"{_mirofish_context}"
    )
    try:
        from debate.agent_tools import DEBATE_TOOLS, execute_tool
        return tool_completion(
            system_prompt=_da_system_prompt,
            user_prompt=user_prompt,
            tools=DEBATE_TOOLS,
            tool_executor=execute_tool,
            model=DEBATE_MODEL,
            max_tokens=800,
            max_tool_calls=2,
            use_cache=True,
        )
    except Exception as e:
        log.warning("[DevilsAdvocate] Tool completion failed (%s), falling back to generate_structured", e)
        return generate_structured(
            system_prompt=_da_system_prompt,
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.65,
            use_cache=True,
        )


# ── Regime specialist ──────────────────────────────────────────────────────────

# Historical regime stats — what typically happens after each regime
REGIME_PLAYBOOK = {
    "calm_bull": {
        "typical_duration": "3-6 months",
        "typical_outcome":  "continued momentum, low drawdown risk",
        "watch_for":        "euphoria signals, breadth narrowing, vol compression",
        "posture":          "full deployment, favor momentum names",
    },
    "stressed": {
        "typical_duration": "4-12 weeks",
        "typical_outcome":  "elevated drawdowns, defensive outperformance",
        "watch_for":        "credit spreads, VIX term structure, breadth collapse",
        "posture":          "reduce equity, increase macro/defensive allocation",
    },
    "crisis": {
        "typical_duration": "2-8 weeks acute, 3-6 months recovery",
        "typical_outcome":  "sharp drawdowns, correlation spikes, liquidity gaps",
        "watch_for":        "fed intervention signals, credit market stabilization",
        "posture":          "maximum defensive, preserve capital, minimal equity",
    },
    "neutral": {
        "typical_duration": "2-4 weeks",
        "typical_outcome":  "mixed, no clear directional edge",
        "watch_for":        "regime transition signals in either direction",
        "posture":          "maintain current allocation, no strong conviction",
    },
    "uncertain": {
        "typical_duration": "1-3 weeks",
        "typical_outcome":  "elevated uncertainty, wait for clarity",
        "watch_for":        "regime confirmation signals",
        "posture":          "reduce size, wait for clearer regime",
    },
}


def run_regime_specialist(portfolio_state: dict) -> str:
    """
    Regime specialist — argues purely from the regime signal and its implications.
    Uses Haiku (cheaper, structured task).

    Unlike bull/bear who argue about specific positions, the regime specialist
    asks: given what the regime historically implies, is this portfolio construction
    appropriate? Is the sizing right for this regime? Are we overweight the wrong
    factors?
    """
    regime   = str(portfolio_state.get("us_regime", "unknown")).lower()
    playbook = REGIME_PLAYBOOK.get(regime, {})
    weights  = portfolio_state.get("weights", {})

    # Classify proposed weights into defensive vs risk-on
    defensive_syms = {"TLT", "IEF", "GLD", "BIL", "VIXY", "LQD", "UUP"}
    risk_on_syms   = {"QQQ", "XLK", "EEM", "VWO", "HYG", "SVXY", "PDBC"}

    defensive_w = sum(w for s, w in weights.items() if s in defensive_syms)
    risk_on_w   = sum(w for s, w in weights.items() if s in risk_on_syms)

    context = (
        f"Date: {portfolio_state.get('date', 'unknown')}\n"
        f"Current regime: {regime.upper()}\n"
        f"Macro regime: {portfolio_state.get('macro_regime', 'unknown')}\n\n"
        f"Regime playbook:\n"
        f"  Typical duration: {playbook.get('typical_duration', 'unknown')}\n"
        f"  Typical outcome: {playbook.get('typical_outcome', 'unknown')}\n"
        f"  Watch for: {playbook.get('watch_for', 'unknown')}\n"
        f"  Recommended posture: {playbook.get('posture', 'unknown')}\n\n"
        f"Portfolio risk profile:\n"
        f"  Defensive weight: {defensive_w:.1%}\n"
        f"  Risk-on weight: {risk_on_w:.1%}\n"
        f"  Remaining (mixed): {1 - defensive_w - risk_on_w:.1%}\n\n"
        f"Proposed weights:\n"
        + "\n".join(f"  {s}: {w:.2%}" for s, w in
                    sorted(weights.items(), key=lambda x: -x[1]))
    )

    track_record = _get_agent_track_record("regime_specialist", regime)
    try:
        return generate_structured(
            system_prompt=(
                "You are the Regime Specialist at Ascent Capital. You argue purely "
                "from the regime signal — not from individual stock opinions. "
                "Your job: given what this regime historically implies, is the "
                "proposed portfolio construction appropriate? Is sizing right? "
                "Are the right factors represented? "
                "Be specific about the regime label and what it demands. "
                f"Keep under 150 words."
                f"{track_record}"
            ),
            user_prompt=f"Regime context:\n{context}\n\nIs this portfolio right for this regime?",
            model=HAIKU_MODEL,
            temperature=0.4,
            use_cache=True,
        )
    except Exception as e:
        return f"Regime specialist failed: {e}"


# ── Quant sanity checker ───────────────────────────────────────────────────────

# Thresholds for sanity checks
MAX_SINGLE_WEIGHT     = 0.15   # single position above this is a flag (orchestrator cap is 10%, buffer for rounding)
MAX_SECTOR_WEIGHT     = 0.40   # single sector above this is a flag
MAX_TURNOVER_WARN     = 0.50   # turnover above 50% is a flag
MIN_POSITIONS         = 5      # fewer than this is a flag
MAX_POSITIONS         = 25     # more than this is a flag
WEIGHT_SUM_TOLERANCE  = 0.02   # weights should sum to 1 ± this

# Sector map — loaded from profiles.parquet, augmented with ETF buckets
def _load_quant_sector_map() -> dict:
    """
    Load sector map from profiles.parquet (source of truth for US equities)
    and augment with ETF/macro/international/alternatives buckets.
    """
    base = {}
    try:
        import pandas as pd
        from pathlib import Path
        profiles_path = Path("data_cache/profiles.parquet")
        if profiles_path.exists():
            df = pd.read_parquet(profiles_path)
            if "symbol" in df.columns and "sector" in df.columns:
                base = dict(zip(df["symbol"], df["sector"]))
    except Exception:
        pass

    # ETF/macro/international/alternatives — not in profiles.parquet
    etf_map = {
        "TLT": "rates", "IEF": "rates", "LQD": "rates", "BIL": "cash",
        "SHY": "rates", "TIP": "rates",
        "HYG": "credit", "JNK": "credit",
        "UUP": "fx",
        "GLD": "commodities", "IAU": "commodities",
        "PDBC": "commodities", "USO": "commodities",
        "DBA": "commodities", "DBB": "commodities",
        "VNQ": "reits", "IYR": "reits", "SCHH": "reits",
        "IFRA": "infrastructure",
        "VIXY": "volatility", "VXX": "volatility",
        "SVXY": "volatility", "SVOL": "volatility",
        "EEM": "em_equity", "VWO": "em_equity",
        "EWT": "em_equity", "EWZ": "em_equity",
        "AAXJ": "em_equity", "EWY": "em_equity",
        "INDA": "em_equity",
        "EWJ": "developed_intl", "EWG": "developed_intl",
        "EWU": "developed_intl", "EFA": "developed_intl",
        "QQQ": "us_tech", "XLK": "us_tech",
        "XLU": "us_defensive", "XLP": "us_defensive", "XLV": "us_defensive",
        "XLE": "us_energy", "XLF": "us_financials",
        "XLI": "us_industrials", "XLB": "us_materials",
    }
    base.update(etf_map)  # ETF map fills gaps, doesn't override profiles.parquet
    return base

_SECTOR_MAP_CACHE: dict = {}


def _get_sector_map() -> dict:
    """Lazy-load sector map on first call — avoids import-time parquet dependency."""
    global _SECTOR_MAP_CACHE
    if not _SECTOR_MAP_CACHE:
        _SECTOR_MAP_CACHE = _load_quant_sector_map()
    return _SECTOR_MAP_CACHE


def run_quant_sanity_check(portfolio_state: dict) -> str:
    """
    Pure Python quant sanity checker — no LLM, no API cost.

    Checks:
    1. Weight sum (should be ~1.0)
    2. Max single position weight
    3. Min/max position count
    4. Sector concentration
    5. Turnover estimate vs prior weights
    6. Regime-weight alignment (e.g. crisis regime + high risk-on = flag)

    Returns a plain text report of issues found.
    If no issues, returns a clean bill of health.
    """
    weights  = portfolio_state.get("weights", {})
    regime   = str(portfolio_state.get("us_regime", "unknown")).lower()
    issues   = []
    warnings = []

    if not weights:
        return "QUANT CHECK: No weights to check."

    # 1. Weight sum
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        issues.append(f"Weights sum to {total:.4f} — expected 1.0 ± {WEIGHT_SUM_TOLERANCE}")

    # 2. Max single position
    max_sym, max_w = max(weights.items(), key=lambda x: x[1])
    if max_w > MAX_SINGLE_WEIGHT:
        issues.append(f"{max_sym} weight {max_w:.1%} exceeds max {MAX_SINGLE_WEIGHT:.0%}")

    # 3. Position count
    n = len(weights)
    if n < MIN_POSITIONS:
        issues.append(f"Only {n} positions — portfolio is dangerously concentrated")
    elif n > MAX_POSITIONS:
        warnings.append(f"{n} positions — unusually high, check diversification logic")

    # 4. Sector concentration
    sector_weights: dict = {}
    sector_map = _get_sector_map()
    for sym, w in weights.items():
        sector = sector_map.get(sym, "unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w

    for sector, sw in sector_weights.items():
        if sw > MAX_SECTOR_WEIGHT:
            issues.append(f"Sector '{sector}' is {sw:.1%} of portfolio — above {MAX_SECTOR_WEIGHT:.0%} limit")

    # 5. Regime-weight alignment
    defensive_syms = {"TLT", "IEF", "GLD", "BIL", "VIXY", "LQD", "UUP", "XLU", "XLP", "XLV"}
    risk_on_syms   = {"QQQ", "XLK", "EEM", "VWO", "HYG", "SVXY", "PDBC", "XLE"}

    defensive_w = sum(w for s, w in weights.items() if s in defensive_syms)
    risk_on_w   = sum(w for s, w in weights.items() if s in risk_on_syms)

    if "crisis" in regime and risk_on_w > 0.30:
        issues.append(
            f"Crisis regime but {risk_on_w:.1%} in risk-on instruments — "
            f"misaligned with regime signal"
        )
    if "stressed" in regime and risk_on_w > 0.50:
        warnings.append(
            f"Stressed regime but {risk_on_w:.1%} in risk-on instruments — "
            f"consider reducing"
        )
    if "calm_bull" in regime and defensive_w > 0.50:
        warnings.append(
            f"Calm bull regime but {defensive_w:.1%} in defensive instruments — "
            f"may be leaving returns on the table"
        )

    # 6. Zero-weight positions (shouldn't be in the dict but check anyway)
    zeros = [s for s, w in weights.items() if w <= 0]
    if zeros:
        warnings.append(f"Zero-weight positions in dict: {zeros} — clean these up")

    # Build report
    lines = ["QUANT SANITY CHECK:"]
    if issues:
        lines.append(f"  ✗ {len(issues)} issue(s):")
        for i in issues:
            lines.append(f"    - {i}")
    if warnings:
        lines.append(f"  ⚠ {len(warnings)} warning(s):")
        for w in warnings:
            lines.append(f"    - {w}")
    if not issues and not warnings:
        lines.append(
            f"  ✓ Clean: {n} positions, sum={total:.4f}, "
            f"max={max_sym} {max_w:.1%}, "
            f"defensive={defensive_w:.1%}, risk-on={risk_on_w:.1%}"
        )

    return "\n".join(lines)


# ── Round 2: Rebuttal functions ────────────────────────────────────────────────

def _format_round1_for_rebuttal(round1_args: dict) -> str:
    """Format all Round 1 arguments into a compact block for Round 2 prompts."""
    parts = []
    labels = {
        "bull": "BULL (Round 1)",
        "bear": "BEAR (Round 1)",
        "devils_advocate": "DEVIL'S ADVOCATE (Round 1)",
        "regime_specialist": "REGIME SPECIALIST (Round 1)",
    }
    for key, label in labels.items():
        arg = round1_args.get(key, "")
        if arg and "failed" not in arg.lower()[:20]:
            parts.append(f"{label}:\n{arg}")
    return "\n\n".join(parts)


def run_bull_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """Round 2: Bull agent responds to all Round 1 arguments."""
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer, respond to the bear and devil's strongest points. "
        "What did they get wrong or overweight?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Druckenmiller-inspired Bull Analyst at Ascent Capital in Round 2. "
                "You have read all Round 1 arguments. Rebut the bear's and devil's "
                "strongest concerns. Is the bear manufacturing risk or citing real fragility? "
                "Is the devil's tail-risk argument already reflected in the Monte Carlo p5? "
                "Engage with their specific claims. Do not repeat your Round 1 argument. "
                "Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
        )
    except Exception as e:
        return f"Bull rebuttal failed: {e}"


def run_bear_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """Round 2: Bear agent responds to all Round 1 arguments."""
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer, respond to the bull and regime specialist's key points. "
        "What critical risk did they dismiss or underweight?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Burry-inspired Bear Analyst at Ascent Capital in Round 2. "
                "You have read all Round 1 arguments. What downside risk is the bull "
                "still glossing over? What number did they not address? "
                "If the regime specialist gave the portfolio a pass, did they account "
                "for the weakest position's fragility? Be terse. Cite a specific number. "
                "Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
        )
    except Exception as e:
        return f"Bear rebuttal failed: {e}"


def run_devils_advocate_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """Round 2: Devil's advocate finds the shared blind spot all Round 1 debaters are still making."""
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer: what dangerous assumption is EVERY Round 1 debater "
        "still making — including the bear?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Taleb-inspired Devil's Advocate at Ascent Capital in Round 2. "
                "You have read all Round 1 arguments. Find the shared FRAGILITY — "
                "the assumption that EVEN THE BEAR is making about normal conditions. "
                "What tail event would invalidate both the bull AND the bear's framing? "
                "Is the debate itself concave — arguing about the middle while ignoring the tails? "
                "Make it concrete. Use Taleb vocabulary. Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.7,
        )
    except Exception as e:
        return f"Devil's advocate rebuttal failed: {e}"


def run_regime_specialist_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """Round 2: Regime specialist checks if Round 1 adequately addressed regime risk."""
    regime = str(portfolio_state.get("us_regime", "unknown")).lower()
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Current regime: {regime.upper()}\n\n"
        f"Round 1 arguments:\n{round1_block}\n\n"
        "In 60 words or fewer: is the regime posture question settled by Round 1, "
        "or is there a regime-specific risk still unaddressed?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Regime Specialist at Ascent Capital in Round 2 of debate. "
                "You argue only from regime signal — not from individual stock opinions. "
                "Has Round 1 adequately addressed the regime-specific risk? "
                "If not, name what is still missing. Under 60 words."
            ),
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            temperature=0.4,
        )
    except Exception as e:
        return f"Regime specialist rebuttal failed: {e}"
