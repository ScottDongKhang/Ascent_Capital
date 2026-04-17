"""
ascent/reporting/debrief.py
Post-rebalance debrief generator.

Runs 14 days after each rebalance. Reads what the debate said,
what actually happened to the portfolio, and writes a one-paragraph
lesson. That lesson is stored and fed into future debates as memory.

Uses Haiku — cheap, runs automatically, no human needed.

Called by:
    debate/debate_runner.py at the start of each debate session
    (score_pending_verdicts already runs there — debrief piggybacks)

Output:
    outputs/debate_log/debrief_YYYY-MM-DD.json
    {
        "date": "...",
        "verdict_recommendation": "...",
        "nav_change": 0.012,
        "lesson": "one paragraph",
        "lesson_tags": ["regime", "timing", "concentration"]
    }
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from ascent.llm.client import generate_structured, HAIKU_MODEL
DEBATE_LOG_DIR = Path("outputs/debate_log")
EOD_LOG_PATH   = Path("logs/eod_log.jsonl")
DEBRIEF_WINDOW = 14  # days after rebalance to write debrief


def _load_nav_series() -> dict:
    nav = {}
    if not EOD_LOG_PATH.exists():
        return nav
    for line in EOD_LOG_PATH.read_text().splitlines():
        try:
            e = json.loads(line)
            d, pv = e.get("run_date"), e.get("portfolio_value")
            if d and pv:
                nav[d] = float(pv)
        except Exception:
            pass
    return nav


def _load_recent_lessons(n: int = 3) -> str:
    """Pull last N debrief lessons to give the LLM continuity."""
    files = sorted(DEBATE_LOG_DIR.glob("debrief_*.json"), reverse=True)[:n]
    lessons = []
    for f in files:
        try:
            rec = json.loads(f.read_text())
            d   = rec.get("date", "?")
            l   = rec.get("lesson", "")
            if l:
                lessons.append(f"[{d}] {l}")
        except Exception:
            pass
    if not lessons:
        return ""
    return "RECENT LESSONS:\n" + "\n".join(lessons)


def write_debrief(verdict_record: dict, nav_change: float) -> Optional[dict]:
    """
    Generate and save a debrief for a scored verdict.

    Args:
        verdict_record: Full verdict JSON record (already scored)
        nav_change:     Portfolio NAV change over the 14-day window

    Returns:
        Debrief dict, or None on failure
    """
    vdate      = verdict_record.get("date", "unknown")
    verdict    = verdict_record.get("verdict", {})
    rec_str    = verdict.get("recommendation", "unknown")
    confidence = verdict.get("confidence", 0)
    key_risks  = verdict.get("key_risks", [])
    reasoning  = verdict.get("reasoning", "")
    regime     = verdict_record.get("portfolio_state", {}).get("us_regime", "unknown")
    arguments  = verdict_record.get("arguments", {})

    recent_lessons = _load_recent_lessons()

    nav_str    = f"{'+' if nav_change >= 0 else ''}{nav_change:.2%}"
    correct    = verdict_record.get("outcome_score", 0.5)
    correctness = {1.0: "CORRECT", 0.5: "PARTIALLY CORRECT", 0.0: "WRONG"}.get(correct, "UNCLEAR")

    user_prompt = f"""
Rebalance date: {vdate}
Regime: {regime}
Debate verdict: {rec_str.upper()} (confidence {confidence:.0%})
Outcome ({DEBRIEF_WINDOW} days): portfolio {nav_str} — verdict was {correctness}

Key risks the debate flagged:
{chr(10).join(f'- {r}' for r in key_risks)}

Judge reasoning:
{reasoning[:300]}

Bull argument summary:
{arguments.get('bull', '')[:200]}

Bear argument summary:
{arguments.get('bear', '')[:200]}

{recent_lessons}

Write one paragraph (3-5 sentences) summarizing what this debate got right or wrong,
why, and what the system should remember for next time. Be specific about the regime,
the recommendation, and the outcome. No fluff.
"""

    try:
        lesson = generate_structured(
            system_prompt=(
                "You are the institutional memory of Ascent Capital. "
                "Write concise, specific post-rebalance lessons that will help "
                "future debate sessions make better decisions. "
                "Focus on what was learned, not what happened. "
                "Be direct. No preamble."
            ),
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.4,
        )
    except Exception as e:
        print(f"[Debrief] LLM call failed: {e}")
        return None

    # Tag the lesson with relevant themes
    lesson_lower = lesson.lower()
    tags = []
    if any(w in lesson_lower for w in ["regime", "bull", "bear", "stressed", "crisis"]):
        tags.append("regime")
    if any(w in lesson_lower for w in ["timing", "wait", "early", "late"]):
        tags.append("timing")
    if any(w in lesson_lower for w in ["concentration", "position", "size", "weight"]):
        tags.append("concentration")
    if any(w in lesson_lower for w in ["macro", "rates", "dollar", "commodity"]):
        tags.append("macro")
    if any(w in lesson_lower for w in ["confident", "confidence", "overconfident"]):
        tags.append("confidence")

    debrief = {
        "date":                   vdate,
        "verdict_recommendation": rec_str,
        "verdict_confidence":     confidence,
        "regime":                 regime,
        "nav_change":             round(nav_change, 6),
        "outcome_correctness":    correctness,
        "lesson":                 lesson,
        "lesson_tags":            tags,
    }

    out_path = DEBATE_LOG_DIR / f"debrief_{vdate}.json"
    DEBATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(debrief, f, indent=2)

    print(f"[Debrief] Written: {out_path}")
    print(f"[Debrief] Lesson: {lesson[:120]}...")
    return debrief


def run_pending_debriefs() -> int:
    """
    Check all scored verdicts that don't yet have a debrief. Write one.
    Returns count of debriefs written.
    Called at the start of each debate session alongside score_pending_verdicts().
    """
    if not DEBATE_LOG_DIR.exists():
        return 0

    nav = _load_nav_series()
    written = 0

    for vf in sorted(DEBATE_LOG_DIR.glob("verdict_*.json")):
        try:
            rec = json.loads(vf.read_text())
        except Exception:
            continue

        # Only process scored verdicts
        if not rec.get("outcome_scored"):
            continue

        vdate = rec.get("date", "")
        debrief_path = DEBATE_LOG_DIR / f"debrief_{vdate}.json"
        if debrief_path.exists():
            continue  # already written

        nav_change = rec.get("outcome_nav_change")
        if nav_change is None:
            continue

        result = write_debrief(rec, nav_change)
        if result:
            written += 1

    return written


def load_debrief_context(n: int = 3) -> str:
    """
    Load last N debrief lessons for injection into debate agent prompts.
    Called by debate/agents.py alongside historical context.
    """
    files = sorted(DEBATE_LOG_DIR.glob("debrief_*.json"), reverse=True)[:n]
    if not files:
        return ""

    lines = ["POST-REBALANCE LESSONS (what the system learned):"]
    for f in files:
        try:
            rec = json.loads(f.read_text())
            d      = rec.get("date", "?")
            rec_v  = rec.get("verdict_recommendation", "?")
            chg    = rec.get("nav_change", 0)
            cor    = rec.get("outcome_correctness", "?")
            lesson = rec.get("lesson", "")
            tags   = rec.get("lesson_tags", [])
            chg_str = f"{'+' if chg >= 0 else ''}{chg:.1%}"
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"\n[{d}] {rec_v.upper()} → {chg_str} ({cor}){tag_str}")
            lines.append(f"  {lesson[:200]}")
        except Exception:
            continue

    return "\n".join(lines) if len(lines) > 1 else ""
