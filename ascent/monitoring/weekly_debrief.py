"""
ascent/monitoring/weekly_debrief.py

Weekly post-mortem: synthesizes the week's trading activity into structured
findings that feed back into the AI PM's priors next week.

Reads: attribution_log, holdings_log, ai_pm_calibration, debate verdicts,
       decision_memory, skill_scores_log.

Writes: data_cache/weekly_debrief.json

Called by weekend_runner.py every weekend run.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

DEBRIEF_PATH   = Path("data_cache/weekly_debrief.json")
ATTRIBUTION_LOG = Path("logs/attribution_log.jsonl")
HOLDINGS_LOG    = Path("logs/holdings_log.jsonl")
CALIBRATION_LOG = Path("logs/ai_pm_calibration.jsonl")
DECISION_LOG    = Path("logs/decision_memory.jsonl")
DEBATE_DIR      = Path("outputs/debate_log")


def _load_jsonl(path: Path, days_back: int = 7) -> list:
    if not path.exists():
        return []
    cutoff = date.today() - timedelta(days=days_back)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            row_date = d.get("date") or d.get("timestamp", "")[:10]
            if row_date >= cutoff.isoformat():
                rows.append(d)
        except Exception:
            pass
    return rows


def _load_verdicts(days_back: int = 7) -> list:
    if not DEBATE_DIR.exists():
        return []
    cutoff = date.today() - timedelta(days=days_back)
    verdicts = []
    for f in sorted(DEBATE_DIR.glob("verdict_*.json")):
        date_str = f.stem.replace("verdict_", "")
        if date_str >= cutoff.isoformat():
            try:
                verdicts.append(json.loads(f.read_text()))
            except Exception:
                pass
    return verdicts


def _summarize_attribution(rows: list) -> dict:
    """Compute week-level alpha, top contributors, top drags."""
    if not rows:
        return {}

    # Deduplicate by date, keep last
    by_date = {}
    for r in rows:
        if r.get("n_positions", 0) >= 5:
            by_date[r["date"]] = r

    if not by_date:
        return {}

    port_cum = 1.0
    spy_cum  = 1.0
    all_contrib: dict = {}
    days = 0

    for r in sorted(by_date.values(), key=lambda x: x["date"]):
        port_cum *= (1 + r["portfolio_return"])
        if r.get("spy_return") is not None:
            spy_cum *= (1 + r["spy_return"])
        days += 1
        for p in r.get("top_contributors", []):
            sym = p["symbol"]
            all_contrib[sym] = all_contrib.get(sym, 0.0) + p.get("contribution", 0.0)
        for p in r.get("top_drags", []):
            sym = p["symbol"]
            all_contrib[sym] = all_contrib.get(sym, 0.0) + p.get("contribution", 0.0)

    sorted_contrib = sorted(all_contrib.items(), key=lambda x: -x[1])
    return {
        "trading_days": days,
        "portfolio_return_pct": round((port_cum - 1) * 100, 2),
        "spy_return_pct": round((spy_cum - 1) * 100, 2),
        "alpha_pct": round((port_cum - spy_cum) * 100, 2),
        "top_contributors": [{"symbol": s, "contribution_pct": round(c * 100, 3)}
                              for s, c in sorted_contrib[:5] if c > 0],
        "top_drags": [{"symbol": s, "contribution_pct": round(c * 100, 3)}
                      for s, c in sorted_contrib[-5:] if c < 0],
    }


def _summarize_overrides(calibration_rows: list, decision_rows: list) -> dict:
    """Summarize AI PM override behavior this week."""
    if not calibration_rows and not decision_rows:
        return {}

    overrides = [r for r in decision_rows
                 if r.get("realized_wedge") is not None]
    if not overrides:
        return {"n_overrides": len(decision_rows), "matured": 0}

    wins = sum(1 for r in overrides if r.get("realized_wedge", 0) > 0)
    avg_wedge = sum(r.get("realized_wedge", 0) for r in overrides) / len(overrides)

    by_type: dict = {}
    for r in overrides:
        t = r.get("override_type", "unknown")
        if t not in by_type:
            by_type[t] = {"wins": 0, "total": 0, "wedges": []}
        by_type[t]["total"] += 1
        by_type[t]["wedges"].append(r.get("realized_wedge", 0))
        if r.get("realized_wedge", 0) > 0:
            by_type[t]["wins"] += 1

    type_summary = {
        t: {
            "win_rate": round(v["wins"] / v["total"], 3),
            "avg_wedge_pct": round(sum(v["wedges"]) / len(v["wedges"]) * 100, 2),
            "n": v["total"],
        }
        for t, v in by_type.items()
    }

    return {
        "n_matured_overrides": len(overrides),
        "overall_win_rate": round(wins / len(overrides), 3),
        "avg_wedge_pct": round(avg_wedge * 100, 2),
        "by_type": type_summary,
    }


def _summarize_verdicts(verdicts: list) -> dict:
    if not verdicts:
        return {}
    outcomes = [v.get("verdict") for v in verdicts if v.get("verdict")]
    return {
        "n_debates": len(verdicts),
        "proceed": outcomes.count("proceed"),
        "reduce_size": outcomes.count("reduce_size"),
        "halt_and_review": outcomes.count("halt_and_review"),
    }


def _call_haiku_synthesis(
    attribution: dict,
    overrides: dict,
    verdicts: dict,
    regime: str,
) -> str:
    """Call Haiku to synthesize findings into a structured debrief narrative."""
    try:
        from ascent.llm.client import HAIKU_MODEL
        import anthropic
        client = anthropic.Anthropic()

        prompt = f"""You are the chief risk officer of a quantitative fund.
Review this week's performance and identify the 3 most important patterns.

REGIME: {regime}

ATTRIBUTION (week):
{json.dumps(attribution, indent=2)}

AI PM OVERRIDE PERFORMANCE:
{json.dumps(overrides, indent=2)}

DEBATE VERDICTS:
{json.dumps(verdicts, indent=2)}

Respond in JSON with exactly these fields:
{{
  "what_worked": ["<1-line observation>", ...],  // up to 3 items
  "what_didnt": ["<1-line observation>", ...],   // up to 3 items
  "systematic_bias": "<one sentence on biggest recurring pattern>",
  "watch_next_week": ["<symbol or theme>", ...], // up to 3 items
  "top_risk": "<the single biggest risk to this portfolio next week>"
}}"""

        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return text
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_weekly_debrief() -> dict:
    """
    Build weekly debrief and write to data_cache/weekly_debrief.json.
    Returns the debrief dict.
    """
    print("[WeeklyDebrief] Building weekly post-mortem...")

    attribution_rows = _load_jsonl(ATTRIBUTION_LOG, days_back=8)
    decision_rows    = _load_jsonl(DECISION_LOG, days_back=8)
    calibration_rows = _load_jsonl(CALIBRATION_LOG, days_back=8)
    verdicts         = _load_verdicts(days_back=8)

    attribution = _summarize_attribution(attribution_rows)
    overrides   = _summarize_overrides(calibration_rows, decision_rows)
    verdict_sum = _summarize_verdicts(verdicts)

    # Get current regime
    regime = "unknown"
    try:
        sig = json.loads(Path("dashboard/regime_signal.json").read_text())
        regime = sig.get("regime", sig.get("label", "unknown"))
    except Exception:
        pass

    # LLM synthesis
    synthesis_raw = _call_haiku_synthesis(attribution, overrides, verdict_sum, regime)
    try:
        synthesis = json.loads(synthesis_raw)
    except Exception:
        synthesis = {"raw": synthesis_raw}

    debrief = {
        "as_of": date.today().isoformat(),
        "regime": regime,
        "attribution": attribution,
        "override_performance": overrides,
        "debate_verdicts": verdict_sum,
        "synthesis": synthesis,
    }

    import tempfile, os
    tmp = Path(tempfile.mktemp(dir=DEBRIEF_PATH.parent, suffix=".tmp"))
    DEBRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(debrief, indent=2))
    os.replace(tmp, DEBRIEF_PATH)

    alpha = attribution.get("alpha_pct", "n/a")
    print(f"[WeeklyDebrief] Done. Week alpha: {alpha}% | "
          f"Top risk: {synthesis.get('top_risk', 'see file')}")
    return debrief
