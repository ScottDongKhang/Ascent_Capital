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
    latest = entries[-1].get("conviction_decay", {})
    stale = []
    for sym, data in latest.items():
        r_then = data.get("rank_at_rebalance")
        r_now  = data.get("rank_today")
        if r_then is not None and r_now is not None and (r_now - r_then) >= 10:
            stale.append(sym)
    return stale


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

    summary_lines = [
        f"Period: last {len(entries)} trading days ending {date}",
        f"Stale positions (rank dropped ≥10): {stale_positions or 'none'}",
        f"Weakening alpha sleeves: {weakening_sleeves or 'none'}",
        f"Regime trajectory: {json.dumps(entries[-1].get('regime_trajectory', {}))}",
        f"Analogue signal: {analogue_signal}",
        f"Top macro risks: {top_macro_risks}",
        "Daily adversarial challenges (last 3):",
    ] + [f"  - {t}" for t in adversarial_themes[-3:]] + [
        "Latest position thesis updates:",
    ] + [
        f"  {sym}: {thesis[:80]}"
        for sym, thesis in list(entries[-1].get("position_theses", {}).items())[:6]
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

    result = {
        "date":               date,
        "synthesis":          synthesis,
        "stale_positions":    stale_positions,
        "weakening_sleeves":  weakening_sleeves,
        "top_macro_risks":    top_macro_risks,
        "analogue_signal":    analogue_signal,
        "adversarial_themes": adversarial_themes[-3:],
        "n_entries":          len(entries),
    }
    _write_brief(result, brief_path)
    return result
