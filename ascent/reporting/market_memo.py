"""
ascent/reporting/market_memo.py
Builds the 20in20 one-page market memo as JSON + Markdown.

Drop into ascent/reporting/market_memo.py
Create ascent/reporting/__init__.py if it doesn't exist.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


# ── Payload builder ──────────────────────────────────────────────────────────

def build_market_memo_payload(
    config: Any,                            # Config20in20
    regime: Any,                            # RegimeSummary
    themes_table: pd.DataFrame,
    relative_value_table: pd.DataFrame,
    comps_tables: Optional[Dict[str, pd.DataFrame]] = None,
    scenario_table: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Returns a JSON-serialisable dict.  No numpy types — all plain Python.
    """
    posture = regime.posture
    conf_pct = f"{regime.confidence * 100:.0f}%"

    # ── Headline ─────────────────────────────────────────────────────────────
    headline = (
        f"Posture: {posture.title()} "
        f"({regime.regime_label.replace('_', ' ').title()} regime, "
        f"{conf_pct} confidence)"
    )

    # ── Themes ────────────────────────────────────────────────────────────────
    leaders: List[dict] = []
    laggards: List[dict] = []
    if not themes_table.empty:
        score_col = _find_col(themes_table, ["score", "momentum_3m", "mom_63d", "return_3m"])
        theme_col  = _find_col(themes_table, ["theme", "Theme"])
        if score_col and theme_col:
            for _, row in themes_table.iterrows():
                entry = {
                    "theme": str(row[theme_col]),
                    "score": _safe_float(row.get(score_col)),
                }
                if entry["score"] is not None and entry["score"] >= 0:
                    leaders.append(entry)
                else:
                    laggards.append(entry)
            leaders  = sorted(leaders,  key=lambda x: x["score"] or 0, reverse=True)[:3]
            laggards = sorted(laggards, key=lambda x: x["score"] or 0)[:3]

    # ── Relative value ────────────────────────────────────────────────────────
    stretched: List[dict] = []
    depressed: List[dict] = []
    if not relative_value_table.empty:
        rv_col     = _find_col(relative_value_table, ["rv_z", "z_score", "zscore", "score"])
        ticker_col = _find_col(relative_value_table, ["ticker", "symbol", "Ticker"])
        theme_col2 = _find_col(relative_value_table, ["theme", "sector", "Theme"])
        if rv_col and ticker_col:
            for _, row in relative_value_table.iterrows():
                z = _safe_float(row.get(rv_col))
                if z is None:
                    continue
                entry = {
                    "ticker": str(row[ticker_col]),
                    "theme":  str(row.get(theme_col2, "")) if theme_col2 else "",
                    "rv_z":   round(z, 2),
                    "note":   "extended vs peers" if z > 0 else "washed out vs peers",
                }
                if z > 0.5:
                    stretched.append(entry)
                elif z < -0.5:
                    depressed.append(entry)
            stretched = sorted(stretched, key=lambda x: x["rv_z"], reverse=True)[:4]
            depressed = sorted(depressed, key=lambda x: x["rv_z"])[:4]

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenarios_out: List[dict] = []
    if scenario_table is not None and not scenario_table.empty:
        for _, row in scenario_table.iterrows():
            scenarios_out.append({
                "scenario":    str(row.get("scenario", row.get("name", "?"))),
                "pnl_est":     _safe_float(row.get("pnl_est", row.get("pnl", None))),
                "most_exposed": _parse_list(row.get("most_exposed", "")),
            })

    # ── Rule-based takeaways ──────────────────────────────────────────────────
    takeaways = _generate_takeaways(posture, regime)

    # ── Assemble ──────────────────────────────────────────────────────────────
    payload: Dict[str, Any] = {
        "asof":          config.asof,
        "horizon_days":  config.memo_horizon_days,
        "headline":      headline,
        "regime": {
            "label":           regime.regime_label,
            "posture":         posture,
            "confidence":      regime.confidence,
            "confidence_pct":  conf_pct,
            "risk_multiplier": regime.risk_multiplier,
            "days_in_regime":  regime.days_in_regime,
            "notes":           regime.notes,
        },
        "themes": {
            "leaders":  leaders,
            "laggards": laggards,
        },
        "relative_value": {
            "most_stretched": stretched,
            "most_depressed": depressed,
        },
        "scenarios":    scenarios_out,
        "takeaways":    takeaways,
        "disclaimer":   "For internal discussion only. Not investment advice.",
    }
    return payload


# ── Writers ──────────────────────────────────────────────────────────────────

def write_market_memo(
    payload: Dict[str, Any],
    outputs_dir: str,
    filename_stem: str,
) -> Dict[str, str]:
    """
    Writes <stem>.json and <stem>.md under outputs_dir.
    Returns {"json": path, "md": path}.
    """
    base = Path(outputs_dir)
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / f"{filename_stem}.json"
    md_path   = base / f"{filename_stem}.md"

    # JSON
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}


# ── Markdown renderer ────────────────────────────────────────────────────────

def _render_markdown(p: Dict[str, Any]) -> str:
    r = p["regime"]
    lines: List[str] = []
    a = lines.append

    a(f"# Ascent Intel — Market Memo")
    a(f"**{p['asof']}** | Horizon: {p['horizon_days']}d\n")
    a(f"## {p['headline']}\n")

    # Regime block
    a("### Market Regime")
    a(f"| Field | Value |")
    a(f"|---|---|")
    a(f"| Regime | `{r['label']}` |")
    a(f"| Posture | **{r['posture'].upper()}** |")
    a(f"| Confidence | {r['confidence_pct']} |")
    a(f"| Risk Multiplier | {r['risk_multiplier']:.0%} |")
    a(f"| Days in Regime | {r['days_in_regime']} |")
    a(f"\n> {r['notes']}\n")

    # Themes
    if p["themes"]["leaders"] or p["themes"]["laggards"]:
        a("### Theme Leadership")
        a("| Theme | Score | Direction |")
        a("|---|---|---|")
        for t in p["themes"]["leaders"]:
            s = f"{t['score']:+.2f}" if t["score"] is not None else "n/a"
            a(f"| {t['theme']} | {s} | 📈 Leading |")
        for t in p["themes"]["laggards"]:
            s = f"{t['score']:+.2f}" if t["score"] is not None else "n/a"
            a(f"| {t['theme']} | {s} | 📉 Lagging |")
        a("")

    # Relative value
    rv = p["relative_value"]
    if rv["most_stretched"] or rv["most_depressed"]:
        a("### Relative Value Monitor")
        if rv["most_stretched"]:
            a("**Extended vs peers:**")
            for x in rv["most_stretched"]:
                a(f"- `{x['ticker']}` (z={x['rv_z']:+.1f}) — {x['note']}")
        if rv["most_depressed"]:
            a("\n**Washed out vs peers:**")
            for x in rv["most_depressed"]:
                a(f"- `{x['ticker']}` (z={x['rv_z']:+.1f}) — {x['note']}")
        a("")

    # Scenarios
    if p["scenarios"]:
        a("### Scenario Watch")
        for s in p["scenarios"]:
            pnl = f"{s['pnl_est']:+.1%}" if s["pnl_est"] is not None else "n/a"
            exposed = ", ".join(s["most_exposed"]) if s["most_exposed"] else "—"
            a(f"- **{s['scenario']}** → est. P&L {pnl} | exposed: {exposed}")
        a("")

    # Takeaways
    if p["takeaways"]:
        a("### Key Takeaways")
        for t in p["takeaways"]:
            a(f"- {t}")
        a("")

    a(f"---")
    a(f"*{p['disclaimer']}*")

    return "\n".join(lines)


# ── Rule-based takeaways ──────────────────────────────────────────────────────

def _generate_takeaways(posture: str, regime: Any) -> List[str]:
    base = {
        "constructive": [
            "Trend environment intact — maintain full exposure and favour momentum names.",
            "Monitor for breadth deterioration as a leading warning signal.",
        ],
        "selective": [
            "Late-cycle signals present — reduce concentration and trim crowded positions.",
            "Favour quality and low-vol over high-beta; watch for regime deterioration.",
        ],
        "neutral": [
            "Mixed signals — maintain balanced exposure without large directional bets.",
            "Reassess in 5–7 days as regime clarity improves.",
        ],
        "defensive": [
            "Risk elevated — tighten position sizing and avoid high-beta concentration.",
            "Prioritise quality/defensive tilt until regime improves.",
        ],
        "crisis": [
            "Capital preservation mode — reduce gross exposure to minimum viable level.",
            "Do not add risk until regime stabilises; await at least 5 consecutive calm days.",
        ],
        "uncertain": [
            "Low regime confidence — avoid large conviction bets in either direction.",
            "Wait for clearer probability separation before adjusting posture.",
        ],
    }.get(posture, ["Monitor market conditions closely."])

    # Append risk multiplier context
    mult = regime.risk_multiplier
    if mult < 0.70:
        base.append(f"System risk multiplier at {mult:.0%} — portfolio sizing reduced accordingly.")

    return base


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return []
