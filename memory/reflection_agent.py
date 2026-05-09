"""
memory/reflection_agent.py

FinMem-style post-trade reflection.

After each verdict is scored (14 days post-decision), Haiku reads the
outcome and writes a structured lesson: what went wrong, what the losing
side missed, and how future agents should calibrate confidence in this
regime. Lessons are stored in memory/reflections.jsonl and injected into
future debate contexts via _build_context() in debate/agents.py.

Source: FinMem (Wang et al., 2023) — arxiv.org/abs/2311.13743
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DEBATE_LOG_DIR  = Path("outputs/debate_log")
REFLECTIONS_PATH = Path("memory/reflections.jsonl")

_SYSTEM_PROMPT = (
    "You are a senior portfolio risk manager conducting a post-trade review. "
    "You will be shown a debate verdict and its actual outcome 14 days later. "
    "Your job: write a concise structured lesson so future debate teams make fewer errors. "
    "Respond ONLY with valid JSON matching the specified format. No other text."
)

_USER_TEMPLATE = """Post-trade review:

Regime at decision time: {regime}
Debate verdict: {recommendation}
Outcome 14 days later: portfolio moved {nav_change:+.1%} → verdict was {correct_str}

Bull argued (summary): {bull_summary}
Bear argued (summary): {bear_summary}
Devil's Advocate argued (summary): {devil_summary}

Write a structured lesson in this exact JSON format:
{{"lesson": "one sentence — what future teams should watch for in this regime",
  "key_error": "one sentence — what the {losing_side} side got wrong",
  "confidence_calibration": "UP, DOWN, or HOLD — how much to trust the {wrong_agent} agent in {regime} regime going forward",
  "regime": "{regime}"}}"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.2,
            use_cache=True,
        )
    except Exception as exc:
        log.warning("[Reflection] LLM call failed: %s", exc)
        return None


def _load_reflected_dates() -> set:
    """Return the set of verdict dates already reflected on (to avoid re-processing)."""
    if not REFLECTIONS_PATH.exists():
        return set()
    dates = set()
    try:
        for line in REFLECTIONS_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "verdict_date" in row:
                    dates.add(row["verdict_date"])
    except Exception:
        pass
    return dates


def reflect_on_verdict(verdict_path: Path) -> Optional[Dict]:
    """
    Reflect on a single scored verdict. Returns reflection dict or None if skipped.

    Skips if outcome_scored is not True, or if LLM call fails.
    """
    try:
        data = json.loads(verdict_path.read_text())
    except Exception as exc:
        log.warning("[Reflection] Cannot read %s: %s", verdict_path, exc)
        return None

    if not data.get("outcome_scored"):
        return None

    regime         = str(data.get("portfolio_state", {}).get("us_regime", "unknown")).lower()
    recommendation = data.get("verdict", {}).get("recommendation", "proceed")
    nav_change     = float(data.get("outcome_nav_change", 0.0))
    date_str       = data.get("date", "unknown")

    correct = (
        (recommendation == "proceed"         and nav_change >= 0) or
        (recommendation == "reduce_size"     and nav_change < -0.005) or
        (recommendation == "halt_and_review" and nav_change < -0.01)
    )
    correct_str = "CORRECT" if correct else "INCORRECT"
    losing_side = "bull" if nav_change < -0.01 else ("bear" if nav_change > 0.01 else "neither")
    wrong_agent = losing_side if losing_side != "neither" else "bull"

    args          = data.get("arguments", {})
    bull_summary  = str(args.get("bull",            ""))[:150]
    bear_summary  = str(args.get("bear",            ""))[:150]
    devil_summary = str(args.get("devils_advocate", ""))[:150]

    user_prompt = _USER_TEMPLATE.format(
        regime=regime, recommendation=recommendation,
        nav_change=nav_change, correct_str=correct_str,
        bull_summary=bull_summary, bear_summary=bear_summary,
        devil_summary=devil_summary,
        losing_side=losing_side, wrong_agent=wrong_agent,
    )

    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        parsed["date"]         = str(date_str)
        parsed["verdict_date"] = str(date_str)
        parsed["nav_change"]   = round(nav_change, 4)
        parsed["correct"]      = correct
        parsed.setdefault("regime", regime)
        return parsed
    except Exception as exc:
        log.warning("[Reflection] JSON parse failed: %s", exc)
        return None


def reflect_on_new_outcomes() -> int:
    """
    Reflect on all newly-scored verdicts not yet reflected on.
    Appends structured lessons to memory/reflections.jsonl.
    Returns count of new reflections written.
    """
    if not DEBATE_LOG_DIR.exists():
        return 0

    already_reflected = _load_reflected_dates()
    count = 0
    REFLECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for vf in sorted(DEBATE_LOG_DIR.glob("verdict_*.json")):
        try:
            data = json.loads(vf.read_text())
        except Exception:
            continue

        if not data.get("outcome_scored"):
            continue

        date_str = data.get("date", "")
        if date_str in already_reflected:
            continue

        reflection = reflect_on_verdict(vf)
        if reflection is None:
            continue

        with open(REFLECTIONS_PATH, "a") as f:
            f.write(json.dumps(reflection) + "\n")
        count += 1
        log.info("[Reflection] Wrote lesson for %s (regime=%s, correct=%s)",
                 date_str, reflection.get("regime"), reflection.get("correct"))

    return count


def load_recent_reflections(regime: Optional[str] = None, n: int = 3) -> List[Dict]:
    """Load N most recent reflections, optionally filtered by regime."""
    if not REFLECTIONS_PATH.exists():
        return []

    rows = []
    try:
        for line in REFLECTIONS_PATH.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []

    if regime:
        regime_lower = str(regime).lower()
        rows = [r for r in rows if str(r.get("regime", "")).lower() == regime_lower]

    rows.sort(key=lambda r: r.get("date", ""), reverse=True)
    return rows[:n]


def format_reflections_for_context(reflections: List[Dict]) -> str:
    """Format recent reflections as a concise LLM-readable block for debate context injection."""
    if not reflections:
        return ""

    lines = [f"Post-trade lessons — {len(reflections)} recent outcome(s) in this regime:"]
    for i, r in enumerate(reflections, 1):
        correct_str = "CORRECT" if r.get("correct") else "INCORRECT"
        calib = r.get("confidence_calibration", "HOLD")
        lines.append(
            f"\n[{i}] {r.get('date', 'unknown')} | Verdict was {correct_str} | "
            f"Calibrate {r.get('wrong_agent_type', 'bull')} {calib}"
        )
        if r.get("lesson"):
            lines.append(f"    Lesson: {r['lesson']}")
        if r.get("key_error"):
            lines.append(f"    Key error: {r['key_error']}")

    return "\n".join(lines)
