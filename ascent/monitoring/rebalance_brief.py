import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
except ImportError:
    generate_structured = None  # type: ignore
    HAIKU_MODEL = "claude-haiku-4-5-20251001"

log = logging.getLogger(__name__)

_INTEL_DIR  = "data_cache/daily_intelligence"
_BRIEF_PATH = "data_cache/rebalance_brief.json"
_MAX_ENTRIES = 9

_SYSTEM = (
    "You are a senior portfolio manager synthesizing a pre-rebalance intelligence brief. "
    "You have been given up to 9 days of structured observations: conviction decay, signal health, "
    "regime trajectory, historical analogues, per-position thesis updates, daily adversarial "
    "challenges, and upcoming macro events. "
    "Write a 300-400 word briefing covering: (1) regime assessment and stability, "
    "(2) positions whose thesis has weakened most, (3) alpha signal environment, "
    "(4) key risks and upcoming catalysts, (5) what historical analogues suggest. "
    "Be specific, reference symbols by name, and use numbers where available. "
    "This brief will be the first thing the AI portfolio manager reads before making decisions."
)


def _load_entries(intel_dir: str) -> List[Dict]:
    d = Path(intel_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("*.json"))[-_MAX_ENTRIES:]
    entries = []
    for f in files:
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            continue
    return entries


def _extract_stale_positions(entries: List[Dict]) -> List[str]:
    if not entries:
        return []
    # Prefer position_health flags (richer signal) over conviction_decay rank delta
    health = entries[-1].get("position_health", {})
    if health:
        return [sym for sym, h in health.items() if h.get("flag") == "DETERIORATING"]
    # Fallback to conviction_decay rank drop
    latest = entries[-1].get("conviction_decay", {})
    stale = []
    for sym, data in latest.items():
        r_then = data.get("rank_at_rebalance")
        r_now  = data.get("rank_today")
        if r_then is not None and r_now is not None and (r_now - r_then) >= 10:
            stale.append(sym)
    return stale


def _extract_health_summary(entries: List[Dict]) -> str:
    """Return a compact health table for the rebalance brief prompt."""
    if not entries:
        return ""
    health = entries[-1].get("position_health", {})
    if not health:
        return ""
    lines = []
    order = {"DETERIORATING": 0, "WATCH": 1, "OK": 2}
    sorted_items = sorted(health.items(), key=lambda kv: (order.get(kv[1].get("flag", "OK"), 3),
                                                           kv[1].get("return_since_rebalance") or 0))
    for sym, h in sorted_items[:12]:
        flag = h.get("flag", "OK")
        ret  = h.get("return_since_rebalance")
        mom  = h.get("momentum_252d")
        rank = h.get("alpha_rank_pct")
        ret_str  = f"{ret:+.1%}" if ret is not None else "n/a"
        mom_str  = f"{mom:+.0%}" if mom is not None else "n/a"
        rank_str = f"{rank:.0%}" if rank is not None else "n/a"
        lines.append(f"  {sym} [{flag}] ret={ret_str} mom={mom_str} rank={rank_str}")
    return "Position health (worst first):\n" + "\n".join(lines)


def _extract_weakening_sleeves(entries: List[Dict]) -> List[str]:
    if not entries:
        return []
    latest = entries[-1].get("signal_health", {})
    return [s for s, d in latest.items() if d.get("status") in ("weakening", "deteriorating")]


def _write_brief(data: Dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, p)
    except Exception as e:
        log.error("[RebalanceBrief] Write failed: %s", e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def generate_rebalance_brief(
    date: str,
    intel_dir: str = _INTEL_DIR,
    brief_path: str = _BRIEF_PATH,
) -> Dict[str, Any]:
    """
    Synthesizes daily intelligence entries into a structured rebalance briefing.
    Writes to brief_path atomically. Returns empty brief on failure or no entries.
    """
    empty = {
        "date": date, "synthesis": "", "stale_positions": [],
        "weakening_sleeves": [], "top_macro_risks": [],
        "analogue_signal": "", "adversarial_themes": [], "n_entries": 0,
    }

    entries = _load_entries(intel_dir)
    if not entries:
        log.warning("[RebalanceBrief] No intelligence entries in %s", intel_dir)
        _write_brief(empty, brief_path)
        return empty

    stale_positions   = _extract_stale_positions(entries)
    weakening_sleeves = _extract_weakening_sleeves(entries)

    adversarial_themes = list({
        e.get("adversarial_challenge", "")
        for e in entries if e.get("adversarial_challenge")
    })

    top_macro_risks = [
        f"{ev.get('event')} ({ev.get('days_away')}d, sensitivity {ev.get('sensitivity')})"
        for ev in entries[-1].get("macro_events", [])[:3]
    ]

    all_outcomes = [
        a.get("outcome_21d") for e in entries
        for a in e.get("historical_analogues", [])
        if a.get("outcome_21d") is not None
    ]
    analogue_signal = (
        f"{len(all_outcomes)} analogues; median 21d outcome "
        f"{sorted(all_outcomes)[len(all_outcomes) // 2]:+.1%}"
        if all_outcomes else "Insufficient historical analogues"
    )

    health_summary = _extract_health_summary(entries)

    summary_lines = [
        f"Period: last {len(entries)} trading days ending {date}",
        f"Deteriorating positions: {stale_positions or 'none'}",
        f"Weakening alpha sleeves: {weakening_sleeves or 'none'}",
        f"Regime trajectory: {json.dumps(entries[-1].get('regime_trajectory', {}))}",
        f"Analogue signal: {analogue_signal}",
        f"Top macro risks: {top_macro_risks}",
    ]
    if health_summary:
        summary_lines += ["", health_summary]
    summary_lines += [
        "",
        "Daily adversarial challenges (last 3):",
    ] + [f"  - {t}" for t in adversarial_themes[-3:]] + [
        "Latest position thesis updates:",
    ] + [
        f"  {sym}: {thesis[:100]}"
        for sym, thesis in list(entries[-1].get("position_theses", {}).items())[:8]
    ]

    synthesis = ""
    try:
        if generate_structured is None:
            raise ImportError("LLM client unavailable")
        synthesis = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt="\n".join(summary_lines),
            model=HAIKU_MODEL,
            max_tokens=700,
            temperature=0.3,
            use_cache=True,
        ).strip()
    except Exception as e:
        log.warning("[RebalanceBrief] Haiku synthesis failed: %s", e)

    # Latest health flags for the AI PM tool
    latest_health = entries[-1].get("position_health", {}) if entries else {}
    deteriorating = [s for s, h in latest_health.items() if h.get("flag") == "DETERIORATING"]
    watching      = [s for s, h in latest_health.items() if h.get("flag") == "WATCH"]

    result = {
        "date":               date,
        "synthesis":          synthesis,
        "stale_positions":    stale_positions,
        "deteriorating":      deteriorating,
        "watching":           watching,
        "weakening_sleeves":  weakening_sleeves,
        "top_macro_risks":    top_macro_risks,
        "analogue_signal":    analogue_signal,
        "adversarial_themes": adversarial_themes[-3:],
        "n_entries":          len(entries),
    }
    _write_brief(result, brief_path)
    return result
