"""
ascent/reporting/blind_spot_detector.py

Reads all past debate verdicts and debriefs, identifies systematic
patterns of error, and writes a structured blind spot map.

This map is injected into every future debate as permanent context:
"Known blind spots: X, Y, Z. Agents must specifically address these."

Runs:
- At the start of each debate session (in debate_runner.py)
- Requires at least 3 verdicts to produce meaningful output

Output:
    outputs/debate_log/blind_spots.json
    {
        "updated": "2026-04-15",
        "n_verdicts_analyzed": 5,
        "patterns": [
            {
                "pattern": "EM contagion consistently underestimated",
                "frequency": 3,
                "first_seen": "2026-04-04",
                "last_seen": "2026-04-06",
                "severity": "high",
                "recommendation": "Devil's advocate must quantify compound EM+commodity scenario"
            }
        ],
        "summary": "one paragraph plain English summary of blind spots"
    }
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DEBATE_LOG_DIR   = Path("outputs/debate_log")
BLIND_SPOTS_PATH = Path("outputs/debate_log/blind_spots.json")
MIN_VERDICTS     = 3  # minimum verdicts needed before pattern detection is meaningful


def _load_all_verdicts() -> list:
    """Load all verdict JSON files sorted by date."""
    if not DEBATE_LOG_DIR.exists():
        return []
    files = sorted(DEBATE_LOG_DIR.glob("verdict_*.json"))
    verdicts = []
    for f in files:
        try:
            v = json.loads(f.read_text())
            verdicts.append(v)
        except Exception:
            pass
    return verdicts


def _load_all_debriefs() -> list:
    """Load all debrief JSON files."""
    if not DEBATE_LOG_DIR.exists():
        return []
    files = sorted(DEBATE_LOG_DIR.glob("debrief_*.json"))
    debriefs = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            debriefs.append(d)
        except Exception:
            pass
    return debriefs


def _build_analysis_payload(verdicts: list, debriefs: list) -> str:
    """Build a compact text payload for the LLM to analyze."""
    lines = []
    lines.append(f"DEBATE HISTORY ({len(verdicts)} sessions):\n")

    for v in verdicts:
        vdate = v.get("date", "?")
        rec = v.get("verdict", {}).get("recommendation", "?")
        conf = v.get("verdict", {}).get("confidence", "?")
        risks = v.get("verdict", {}).get("key_risks", [])
        reasoning = v.get("verdict", {}).get("reasoning", "")[:300]
        regime = v.get("portfolio_state", {}).get("us_regime", "?")
        scored = v.get("outcome_scored", False)
        nav_change = v.get("outcome_nav_change", None)

        lines.append(f"--- {vdate} | {rec.upper()} | confidence={conf} | regime={regime} ---")
        for r in risks:
            lines.append(f"  RISK: {r}")
        lines.append(f"  REASONING: {reasoning}...")
        if scored and nav_change is not None:
            lines.append(f"  OUTCOME: NAV {nav_change:+.2%} over 14 days")
        lines.append("")

    if debriefs:
        lines.append(f"\nPOST-REBALANCE LESSONS ({len(debriefs)} debriefs):\n")
        for d in debriefs:
            ddate = d.get("date", "?")
            lesson = d.get("lesson", "")[:200]
            tags = d.get("lesson_tags", [])
            lines.append(f"--- {ddate} | tags={tags} ---")
            lines.append(f"  {lesson}")
            lines.append("")

    return "\n".join(lines)


def detect_blind_spots(force: bool = False) -> Optional[dict]:
    """
    Analyze all past verdicts and debriefs to identify systematic blind spots.

    Args:
        force: If True, regenerate even if blind_spots.json already exists today.

    Returns:
        The blind spot map dict, or None if insufficient data.
    """
    # Check if we already ran today
    if not force and BLIND_SPOTS_PATH.exists():
        try:
            existing = json.loads(BLIND_SPOTS_PATH.read_text())
            if existing.get("updated") == date.today().isoformat():
                return existing
        except Exception:
            pass

    verdicts = _load_all_verdicts()
    debriefs = _load_all_debriefs()

    if len(verdicts) < MIN_VERDICTS:
        print(f"[BlindSpot] Only {len(verdicts)} verdicts — need {MIN_VERDICTS} minimum. Skipping.")
        return None

    print(f"[BlindSpot] Analyzing {len(verdicts)} verdicts + {len(debriefs)} debriefs...")

    payload = _build_analysis_payload(verdicts, debriefs)

    from ascent.llm.client import generate_structured, HAIKU_MODEL

    system_prompt = (
        "You are a meta-analyst reviewing an AI investment system's debate history. "
        "Your job is to identify SYSTEMATIC patterns of error — things the system "
        "keeps getting wrong across multiple sessions. "
        "Focus on: recurring risk factors that keep appearing, assumptions that are "
        "never challenged, scenarios that are consistently underestimated, and "
        "positions or exposures that are flagged repeatedly without resolution. "
        "Output ONLY valid JSON with this exact structure: "
        '{"patterns": [{"pattern": "...", "frequency": N, "first_seen": "YYYY-MM-DD", '
        '"last_seen": "YYYY-MM-DD", "severity": "high|medium|low", '
        '"recommendation": "specific instruction for future debate agents"}], '
        '"summary": "one paragraph plain English"} '
        "No markdown, no explanation, just the JSON."
    )

    user_prompt = (
        f"Analyze this debate history and identify systematic blind spots:\n\n{payload}\n\n"
        "What patterns of error appear repeatedly? What does this system keep missing? "
        "What assumptions are never challenged? Be specific and actionable."
    )

    try:
        raw = generate_structured(
            system_prompt, user_prompt,
            model=HAIKU_MODEL,
            max_tokens=3000,
            temperature=0.2,
        )
        raw = raw.strip()
        # Strip markdown fences robustly
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        raw = raw.strip()

        # Increase max_tokens to avoid truncation
        result = json.loads(raw)

        # Add metadata
        result["updated"] = date.today().isoformat()
        result["n_verdicts_analyzed"] = len(verdicts)
        result["n_debriefs_analyzed"] = len(debriefs)

        BLIND_SPOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BLIND_SPOTS_PATH.write_text(json.dumps(result, indent=2))

        n_patterns = len(result.get("patterns", []))
        print(f"[BlindSpot] Found {n_patterns} systematic pattern(s) — written to {BLIND_SPOTS_PATH}")
        for p in result.get("patterns", []):
            print(f"  [{p.get('severity','?').upper()}] {p.get('pattern','?')} (seen {p.get('frequency','?')}x)")

        return result

    except Exception as e:
        print(f"[BlindSpot] Detection failed: {e}")
        return None


def load_blind_spot_context() -> str:
    """
    Returns a formatted string for injection into debate agent prompts.
    Called by agents.py and judge.py.
    """
    if not BLIND_SPOTS_PATH.exists():
        return ""

    try:
        data = json.loads(BLIND_SPOTS_PATH.read_text())
        patterns = data.get("patterns", [])
        if not patterns:
            return ""

        lines = ["\n=== KNOWN SYSTEM BLIND SPOTS (address these explicitly) ==="]
        for p in patterns:
            sev = p.get("severity", "?").upper()
            pattern = p.get("pattern", "?")
            freq = p.get("frequency", "?")
            rec = p.get("recommendation", "")
            lines.append(f"[{sev}] {pattern} (flagged {freq}x)")
            if rec:
                lines.append(f"  → {rec}")
        lines.append("=== END BLIND SPOTS ===\n")
        return "\n".join(lines)

    except Exception:
        return ""


if __name__ == "__main__":
    result = detect_blind_spots(force=True)
    if result:
        print("\n--- SUMMARY ---")
        print(result.get("summary", ""))
