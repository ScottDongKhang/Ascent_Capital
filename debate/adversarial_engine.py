"""
debate/adversarial_engine.py

Adversarial Intelligence — Layers 1–3.

Layer 1: Adversarial Thesis Generation
  Generates the strongest possible SHORT thesis for each position via a single
  batched Haiku call. Scores 0–1 vs the long thesis. Score > 0.6 = flagged.

Layer 2: Regime-Conditional Optimal Sizing
  Classifies each position as event_momentum | trend | reversion | etf | unknown.
  Applies historically-grounded size targets per type per regime.
  Returns which positions are outside their optimal range.

Layer 3: Portfolio Coherence Engine
  Computes emergent portfolio properties that no per-position analysis captures:
  - Narrative clusters (how many independent bets the book actually contains)
  - Aggregate factor exposure (dominant risk across all names)
  - Regime sensitivity (estimated portfolio impact of a regime flip)

Called by debate_runner.py before agents run.
Output injected into portfolio_state for agent context and judge decision.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ADVERSARIAL_FLAG_THRESHOLD = 0.60   # short thesis score above this → flagged
SIZING_DEVIATION_PCT       = 0.02   # ≥2pp above max → oversized flag

# ── Regime-conditional sizing table ──────────────────────────────────────────
# (min_weight, optimal_weight, max_weight) per (regime, position_type)
_SIZING = {
    "calm_bull": {
        "event_momentum": (0.03, 0.050, 0.070),  # single-day spike → mean-revert risk
        "trend":          (0.05, 0.070, 0.100),  # momentum comfortable in bull
        "reversion":      (0.03, 0.050, 0.075),
        "etf":            (0.03, 0.060, 0.100),
        "unknown":        (0.04, 0.065, 0.100),
    },
    "stressed": {
        "event_momentum": (0.02, 0.040, 0.060),
        "trend":          (0.03, 0.050, 0.075),
        "reversion":      (0.02, 0.030, 0.050),
        "etf":            (0.03, 0.050, 0.080),
        "unknown":        (0.03, 0.045, 0.070),
    },
    "crisis": {
        "event_momentum": (0.02, 0.030, 0.050),
        "trend":          (0.02, 0.040, 0.060),
        "reversion":      (0.01, 0.025, 0.040),
        "etf":            (0.03, 0.050, 0.080),
        "unknown":        (0.02, 0.035, 0.060),
    },
    "neutral":   {t: (0.03, 0.055, 0.090) for t in
                  ["event_momentum", "trend", "reversion", "etf", "unknown"]},
    "uncertain": {t: (0.02, 0.045, 0.070) for t in
                  ["event_momentum", "trend", "reversion", "etf", "unknown"]},
}

# Known ETF/macro/alternatives tickers → "etf" position type
_ETF_SET = {
    "TLT", "IEF", "LQD", "BIL", "SHY", "TIP", "HYG", "UUP", "GLD", "PDBC",
    "USO", "DBA", "DBB", "VNQ", "IFRA", "VIXY", "EEM", "VWO", "EWT", "EWZ",
    "AAXJ", "EWY", "INDA", "EWJ", "EWG", "EWU", "EFA", "KMLM", "SGOV",
    "EWC", "EWY", "SVXY", "SVOL",
}

# Named narrative buckets for cross-asset clustering
_NARRATIVE_BUCKETS = {
    "em_equity":      {"EEM", "VWO", "EWT", "EWZ", "AAXJ", "EWY", "INDA"},
    "developed_intl": {"EWJ", "EWG", "EWU", "EFA", "EWC"},
    "us_rates":       {"TLT", "IEF", "BIL", "SHY", "TIP", "SGOV"},
    "us_credit":      {"HYG", "LQD"},
    "commodities":    {"GLD", "PDBC", "DBA", "DBB", "USO"},
    "fx_dollar":      {"UUP"},
    "volatility":     {"VIXY", "SVXY", "SVOL"},
    "real_assets":    {"VNQ", "IFRA"},
    "alts_trend":     {"KMLM"},
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_altdata(symbols: list) -> dict:
    """Load SEC tone + transcript sentiment from caches. Safe to call with any symbols."""
    result = {s: {} for s in symbols}

    sec_path = Path("data_cache/altdata_sec_detail.json")
    if sec_path.exists():
        try:
            detail = json.loads(sec_path.read_text())
            for sym in symbols:
                if sym in detail:
                    d = detail[sym]
                    result[sym]["sec_tone"]            = d.get("tone_score")
                    result[sym]["sec_risk_trend"]      = d.get("risk_trend")
                    result[sym]["sec_revenue_momentum"] = d.get("revenue_momentum")
        except Exception as e:
            log.debug("[AdvEngine] SEC detail load failed: %s", e)

    try:
        import pandas as pd
        tp = Path("data_cache/altdata_transcripts.parquet")
        if tp.exists():
            trans = pd.read_parquet(tp)
            for sym in symbols:
                if sym in trans.columns:
                    col = trans[sym].dropna()
                    if not col.empty:
                        result[sym]["transcript_sentiment"] = float(col.iloc[-1])
    except Exception as e:
        log.debug("[AdvEngine] Transcript load failed: %s", e)

    return result


def _load_recent_price_data(symbols: list, lookback_days: int = 30) -> dict:
    """
    Load recent price action per symbol from prices_live parquet.
    Returns {symbol: {single_day_max, cumulative_return, n_days}}
    """
    result = {}
    try:
        from ascent.data.store.parquet import load_parquet
        import pandas as pd

        prices_long = load_parquet("prices_live")
        prices_wide = prices_long.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )
        prices_wide.index = pd.to_datetime(prices_wide.index)
        cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
        recent = prices_wide[prices_wide.index >= cutoff]

        for sym in symbols:
            if sym not in recent.columns:
                continue
            col = recent[sym].dropna()
            if len(col) < 2:
                continue
            daily_rets = col.pct_change().dropna()
            result[sym] = {
                "single_day_max":     float(daily_rets.abs().max()),
                "cumulative_return":  float((col.iloc[-1] / col.iloc[0]) - 1),
                "n_days":             len(col),
            }
    except Exception as e:
        log.debug("[AdvEngine] Price load failed: %s", e)

    return result


# ── Layer 1: Adversarial Thesis ───────────────────────────────────────────────

def _generate_adversarial_theses(
    weights: dict,
    altdata: dict,
    price_data: dict,
    regime: str,
) -> dict:
    """
    One batched Haiku call to generate + score SHORT thesis for every position.
    Returns {symbol: {short_thesis, adversarial_score, flagged}}
    """
    from ascent.llm.client import generate_structured, HAIKU_MODEL

    lines = []
    for sym, weight in sorted(weights.items(), key=lambda x: -x[1]):
        alt   = altdata.get(sym, {})
        pdata = price_data.get(sym, {})

        parts = [f"{sym} (weight={weight:.1%})"]
        if pdata.get("cumulative_return") is not None:
            parts.append(f"30d={pdata['cumulative_return']:+.1%}")
        if pdata.get("single_day_max") is not None:
            parts.append(f"max_1d={pdata['single_day_max']:.1%}")
        if alt.get("sec_tone") is not None:
            parts.append(f"sec_tone={alt['sec_tone']:+.2f}")
        if alt.get("transcript_sentiment") is not None:
            parts.append(f"transcript={alt['transcript_sentiment']:+.2f}")
        if alt.get("sec_risk_trend") is not None:
            parts.append(f"risk_trend={alt['sec_risk_trend']:+.2f}")
        lines.append(" | ".join(parts))

    positions_text = "\n".join(lines)

    system_prompt = (
        "You are an adversarial risk analyst at a hedge fund. For each long position, "
        "generate the most compelling SHORT thesis a skilled short-seller would make.\n"
        "Score adversarial strength 0.0–1.0:\n"
        "  0.2 = trivial concern, 0.4 = legitimate worry, 0.6 = serious structural risk, "
        "0.8 = near-bulletproof short case\n"
        "Negative sec_tone = deteriorating fundamentals. Negative risk_trend = worsening risks.\n"
        "Large single-day move (>15%) = event momentum stock — mean-revert risk.\n\n"
        f"Regime: {regime}\n\n"
        "Return ONLY valid JSON (no markdown, no text outside JSON):\n"
        '{\"positions\": [{\"symbol\": \"X\", \"short_thesis\": \"<30 words>\", '
        '\"adversarial_score\": 0.0}]}'
    )

    try:
        raw = generate_structured(
            system_prompt=system_prompt,
            user_prompt=f"Portfolio positions:\n{positions_text}\n\nGenerate adversarial theses.",
            model=HAIKU_MODEL,
            temperature=0.3,
            use_cache=True,
        )

        # Depth-tracking JSON extractor (same pattern as sec_filings.py)
        start = raw.find("{")
        if start == -1:
            raise ValueError("no JSON in response")
        depth = 0
        end = -1
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError("unmatched braces")

        parsed     = json.loads(raw[start:end])
        positions  = parsed.get("positions", [])
        result     = {}
        for item in positions:
            sym   = item.get("symbol", "")
            score = min(1.0, max(0.0, float(item.get("adversarial_score", 0.5))))
            thesis = item.get("short_thesis", "")
            if sym in weights:
                result[sym] = {
                    "short_thesis":       thesis,
                    "adversarial_score":  round(score, 3),
                    "flagged":            score > ADVERSARIAL_FLAG_THRESHOLD,
                }

        # Fill any positions the LLM missed
        for sym in weights:
            if sym not in result:
                result[sym] = {"short_thesis": "", "adversarial_score": 0.5, "flagged": False}

        return result

    except Exception as e:
        log.warning("[AdvEngine] Adversarial thesis generation failed: %s", e)
        return {s: {"short_thesis": "", "adversarial_score": 0.5, "flagged": False}
                for s in weights}


# ── Layer 2: Regime-Conditional Sizing ───────────────────────────────────────

def _classify_position_type(symbol: str, price_data: dict) -> str:
    if symbol in _ETF_SET:
        return "etf"
    pdata         = price_data.get(symbol, {})
    single_day    = pdata.get("single_day_max", 0.0)
    cum_return    = pdata.get("cumulative_return", 0.0)

    if single_day > 0.15:
        return "event_momentum"
    if cum_return > 0.08:
        return "trend"
    if cum_return < -0.03:
        return "reversion"
    return "unknown"


def _compute_regime_sizing(weights: dict, price_data: dict, regime: str) -> dict:
    """
    For each position, determine if current weight is appropriate for this regime.
    Returns {symbol: {position_type, optimal_weight, deviation, oversized, recommendation}}
    """
    regime_clean = str(regime).lower()
    table        = _SIZING.get(regime_clean, _SIZING["neutral"])
    result       = {}

    for sym, weight in weights.items():
        pos_type           = _classify_position_type(sym, price_data)
        min_w, opt_w, max_w = table.get(pos_type, table["unknown"])
        deviation           = weight - opt_w
        oversized           = weight > max_w + SIZING_DEVIATION_PCT

        if weight > max_w:
            rec = (f"Trim {sym} to {opt_w:.1%} — currently {weight:.1%}, "
                   f"max for {pos_type} in {regime_clean} is {max_w:.1%}")
        elif weight < min_w:
            rec = f"{sym} below {pos_type} minimum ({min_w:.1%}) in {regime_clean}"
        else:
            rec = f"{sym} size OK [{min_w:.1%}–{max_w:.1%}] for {pos_type}"

        result[sym] = {
            "position_type":  pos_type,
            "optimal_weight": round(opt_w, 4),
            "min_weight":     round(min_w, 4),
            "max_weight":     round(max_w, 4),
            "current_weight": round(weight, 4),
            "deviation":      round(deviation, 4),
            "oversized":      oversized,
            "recommendation": rec,
        }

    return result


# ── Layer 3: Portfolio Coherence Engine ──────────────────────────────────────

def _compute_coherence(weights: dict, regime: str) -> dict:
    """
    Compute emergent portfolio properties that per-position analysis misses:
    - How many independent bets does the book actually contain?
    - What is the dominant risk factor across all names?
    - What happens to the whole book if regime flips tomorrow?
    """
    if not weights:
        return {"n_positions": 0, "n_independent_bets": 0, "narrative_clusters": [],
                "dominant_exposure": "unknown", "largest_cluster_weight": 0.0,
                "regime_flip_impact_estimate": "N/A"}

    # Load sector map from profiles.parquet
    sector_map = {}
    try:
        import pandas as pd
        p = Path("data_cache/profiles.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" in df.columns and "sector" in df.columns:
                sector_map = dict(zip(df["symbol"], df["sector"]))
    except Exception:
        pass

    # Assign each position to a narrative bucket
    sym_to_narrative = {}
    for bucket, members in _NARRATIVE_BUCKETS.items():
        for sym in members:
            if sym in weights:
                sym_to_narrative[sym] = bucket

    for sym in weights:
        if sym in sym_to_narrative:
            continue
        sector = str(sector_map.get(sym, "us_equity_other")).lower().replace(" ", "_")
        if not sector or sector in ("unknown", "nan", ""):
            sector = "us_equity_other"
        sym_to_narrative[sym] = sector

    # Aggregate cluster weights
    cluster_w: dict = {}
    for sym, w in weights.items():
        cluster = sym_to_narrative.get(sym, "unknown")
        cluster_w[cluster] = cluster_w.get(cluster, 0.0) + w

    sorted_clusters = sorted(cluster_w.items(), key=lambda x: -x[1])

    # Independent bets = clusters with >3% aggregate weight
    significant = [(c, w) for c, w in sorted_clusters if w > 0.03]
    n_bets        = len(significant)
    dominant      = sorted_clusters[0][0] if sorted_clusters else "unknown"
    largest_w     = sorted_clusters[0][1] if sorted_clusters else 0.0

    narrative_clusters = []
    for cluster, cw in sorted_clusters[:8]:
        members = [
            {"symbol": s, "weight": round(w, 4)}
            for s, w in weights.items()
            if sym_to_narrative.get(s) == cluster
        ]
        narrative_clusters.append({
            "name":         cluster,
            "total_weight": round(cw, 4),
            "positions":    sorted(members, key=lambda x: -x["weight"]),
        })

    # Regime flip impact: approximate based on equity/EM/commodity exposure
    eq_weight   = sum(w for s, w in weights.items()
                      if s not in _ETF_SET or sym_to_narrative.get(s, "").startswith("us_equity"))
    em_weight   = cluster_w.get("em_equity", 0.0)
    comm_weight = cluster_w.get("commodities", 0.0)
    rates_weight = cluster_w.get("us_rates", 0.0)

    regime_clean = str(regime).lower()
    if "calm_bull" in regime_clean:
        impact = -(eq_weight * 0.10 + em_weight * 0.15 + comm_weight * 0.08 - rates_weight * 0.02)
        flip_desc = f"calm_bull→stressed: est {impact:+.1%}"
    elif "stressed" in regime_clean:
        impact = -(eq_weight * 0.15 + em_weight * 0.20 + comm_weight * 0.10 - rates_weight * 0.03)
        flip_desc = f"stressed→crisis: est {impact:+.1%}"
    elif "crisis" in regime_clean:
        impact = eq_weight * 0.08 + em_weight * 0.10 + rates_weight * 0.01
        flip_desc = f"crisis→recovery: est {impact:+.1%}"
    else:
        impact = -(eq_weight * 0.08)
        flip_desc = f"regime flip: est {impact:+.1%}"

    return {
        "n_positions":               len(weights),
        "n_independent_bets":        n_bets,
        "narrative_clusters":        narrative_clusters,
        "dominant_exposure":         dominant,
        "largest_cluster_weight":    round(largest_w, 4),
        "regime_flip_impact_estimate": flip_desc,
        "em_weight":                 round(em_weight, 4),
        "commodity_weight":          round(comm_weight, 4),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_adversarial_engine(portfolio_state: dict) -> dict:
    """
    Run all three Adversarial Intelligence layers.

    Returns:
        {
          "adversarial_scores":     {symbol: {short_thesis, adversarial_score, flagged}},
          "sizing_recommendations": {symbol: {position_type, optimal_weight, deviation, oversized, recommendation}},
          "coherence":              {n_independent_bets, narrative_clusters, dominant_exposure, ...},
          "top_flags":              [{symbol, reason, intervention_type, current_weight,
                                      suggested_weight, priority_score}],
        }
    """
    weights = portfolio_state.get("weights", {})
    regime  = str(portfolio_state.get("us_regime", "unknown")).lower()

    if not weights:
        return {
            "adversarial_scores":     {},
            "sizing_recommendations": {},
            "coherence":              _compute_coherence({}, regime),
            "top_flags":              [],
        }

    print(f"[AdvEngine] Running adversarial analysis on {len(weights)} positions "
          f"(regime={regime})...")

    symbols    = list(weights.keys())
    altdata    = _load_altdata(symbols)
    price_data = _load_recent_price_data(symbols, lookback_days=30)

    print("[AdvEngine] Layer 1: Adversarial theses...")
    adv_scores = _generate_adversarial_theses(weights, altdata, price_data, regime)

    print("[AdvEngine] Layer 2: Regime-conditional sizing...")
    sizing = _compute_regime_sizing(weights, price_data, regime)

    print("[AdvEngine] Layer 3: Portfolio coherence...")
    coherence = _compute_coherence(weights, regime)

    # Build ranked flags for judge — combine adversarial + sizing signals
    flags = []
    seen  = set()

    for sym, data in adv_scores.items():
        if not data.get("flagged"):
            continue
        score    = data["adversarial_score"]
        opt_w    = sizing.get(sym, {}).get("optimal_weight", weights[sym] * 0.80)
        priority = score * weights.get(sym, 0) * 10  # weight by position size impact
        flags.append({
            "symbol":            sym,
            "reason":            f"adversarial_thesis: {data['short_thesis']}",
            "intervention_type": "adversarial_thesis",
            "current_weight":    weights[sym],
            "suggested_weight":  min(opt_w, weights[sym] * 0.80),  # trim at least 20%
            "adversarial_score": score,
            "priority_score":    round(priority, 3),
        })
        seen.add(sym)

    for sym, data in sizing.items():
        if not data.get("oversized") or sym in seen:
            continue
        priority = abs(data["deviation"]) * 6
        flags.append({
            "symbol":            sym,
            "reason":            f"regime_sizing: {data['recommendation']}",
            "intervention_type": "regime_sizing",
            "current_weight":    data["current_weight"],
            "suggested_weight":  data["optimal_weight"],
            "adversarial_score": adv_scores.get(sym, {}).get("adversarial_score", 0.5),
            "priority_score":    round(priority, 3),
        })
        seen.add(sym)

    flags.sort(key=lambda x: -x["priority_score"])

    n_flagged  = sum(1 for s in adv_scores.values() if s.get("flagged"))
    n_oversized = sum(1 for s in sizing.values() if s.get("oversized"))
    n_bets     = coherence.get("n_independent_bets", 0)
    n_pos      = coherence.get("n_positions", 0)

    print(f"[AdvEngine] Done — {n_flagged} adversarial flags, {n_oversized} sizing flags, "
          f"{n_bets}/{n_pos} independent bets")

    return {
        "adversarial_scores":     adv_scores,
        "sizing_recommendations": sizing,
        "coherence":              coherence,
        "top_flags":              flags[:5],
    }


def format_adversarial_context(engine_output: dict) -> str:
    """Format engine output as a text block for injection into judge / agent prompts."""
    if not engine_output:
        return ""

    lines = ["ADVERSARIAL INTELLIGENCE REPORT:"]

    # Coherence summary
    coh   = engine_output.get("coherence", {})
    n_pos = coh.get("n_positions", 0)
    n_bet = coh.get("n_independent_bets", 0)
    dom   = coh.get("dominant_exposure", "unknown")
    lw    = coh.get("largest_cluster_weight", 0)
    flip  = coh.get("regime_flip_impact_estimate", "N/A")

    lines.append(f"\nPortfolio Coherence:")
    lines.append(f"  {n_pos} positions = {n_bet} independent bets "
                 f"(dominant: {dom} at {lw:.1%})")
    lines.append(f"  Regime flip sensitivity: {flip}")

    clusters = coh.get("narrative_clusters", [])
    for c in clusters[:4]:
        syms = ", ".join(p["symbol"] for p in c["positions"][:4])
        extra = f" +{len(c['positions'])-4} more" if len(c["positions"]) > 4 else ""
        lines.append(f"  └ {c['name']}: {c['total_weight']:.1%} [{syms}{extra}]")

    # Top flags
    top_flags = engine_output.get("top_flags", [])
    if top_flags:
        lines.append(f"\nTop Adversarial Flags (ranked by priority):")
        for i, flag in enumerate(top_flags, 1):
            lines.append(
                f"  {i}. {flag['symbol']} ({flag['current_weight']:.1%}) — "
                f"{flag['reason'][:90]}"
            )
            if flag.get("suggested_weight"):
                lines.append(
                    f"     Suggested: {flag['suggested_weight']:.1%} | "
                    f"type: {flag['intervention_type']} | "
                    f"priority: {flag['priority_score']:.2f}"
                )

    return "\n".join(lines)
