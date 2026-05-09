"""
debate/agents.py
Debate agent definitions for Ascent Capital.

Agents:
  - Bull analyst
  - Bear analyst
  - Devil's advocate
  - Regime specialist (LLM, Haiku)
  - Quant sanity checker (pure Python, no LLM)

Uses claude-opus-4-6 (DEBATE_MODEL) for bull/bear/devil.
Uses claude-haiku-4-5-20251001 (HAIKU_MODEL) for regime specialist.
Quant sanity checker uses no LLM.
"""

import logging

from ascent.llm.client import generate_structured, DEFAULT_MODEL as DEBATE_MODEL, HAIKU_MODEL
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


# ── Agent context assembler ────────────────────────────────────────────────────

_AGENT_SECTIONS = {
    "bull": [_section_weights, _section_momentum, _section_fundamental, _section_allocation],
    "bear": [_section_weights, _section_concentration, _section_regime, _section_allocation],
    "devils_advocate": [_section_weights, _section_regime, _section_tail, _section_allocation],
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
            "You are the Bull Analyst at Ascent Capital. Your job is to make "
            "the strongest case FOR executing the proposed trades as-is. Reference specific "
            "positions, the current regime, and momentum signals. Be specific and data-driven. "
            "You have been given historical accuracy data for each debater — use it to "
            f"understand where the bear and devil tend to over-warn. Keep your argument under 200 words."
            f"{track_record}"
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
    user_prompt += "\n\nMake the bear case against these trades."
    return generate_structured(
        system_prompt=(
            "You are the Bear Analyst at Ascent Capital. Your job is to argue "
            "for REDUCING risk or WAITING. Identify the weakest positions, concentration risks, "
            "regime fragility, or macro headwinds. Be specific. "
            "You have been given historical accuracy data — use it to calibrate how often "
            f"your past warnings were correct in this regime. Keep under 200 words."
            f"{track_record}"
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

    regime       = portfolio_state.get("us_regime", "unknown")
    cred_context = load_credibility_context(regime)
    track_record = _get_agent_track_record("devils_advocate", regime)
    user_prompt  = f"Portfolio context:\n{context}"
    if scenario_context:
        user_prompt += f"\n{scenario_context}"
    if cred_context:
        user_prompt += f"\n\n{cred_context}"
    user_prompt += "\n\nWhat is the most dangerous blind spot? Use the scenario numbers."

    return generate_structured(
        system_prompt=(
            "You are the Devil's Advocate at Ascent Capital. Your job is to "
            "find the SINGLE most dangerous assumption in the current portfolio construction. "
            "What could go catastrophically wrong that the quant signals would NOT catch? "
            "You have been given Monte Carlo scenario analysis showing worst-case portfolio "
            "impacts. Use these numbers to make a specific, quantified argument. "
            "You also have historical accuracy data — use it to understand when your "
            "past warnings were prescient vs. over-cautious. "
            "Think about: earnings surprises, geopolitical events, liquidity gaps, "
            f"correlation breakdowns. Be specific. Keep under 150 words."
            f"{track_record}"
        ),
        user_prompt=user_prompt,
        model=DEBATE_MODEL,
        temperature=0.7,
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
                "You are the Bull Analyst at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Rebut the bear's and devil's "
                "strongest concerns concisely. Do not repeat your Round 1 argument — "
                "engage directly with their specific claims. Under 75 words."
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
                "You are the Bear Analyst at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Rebut the bull's and regime "
                "specialist's strongest points. Identify the risk they are still not "
                "taking seriously. Under 75 words."
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
                "You are the Devil's Advocate at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Find the shared blind spot — "
                "the assumption that even the bear is making. Make it concrete and "
                "quantified if possible. Under 75 words."
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
