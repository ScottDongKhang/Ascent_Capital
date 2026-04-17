"""
ascent/reporting/ic_brief_generator.py
IC Brief generator.

Reads outputs/20in20/memo_YYYY-MM-DD.json and generates a structured
Investment Committee brief as Markdown.

V1: Template-based (no Claude API). When API key is available,
switch _generate_with_claude() from stub to real implementation.

Usage:
    python3 -m ascent.reporting.ic_brief_generator                    # today
    python3 -m ascent.reporting.ic_brief_generator --date 2026-04-01  # specific date
    python3 -m ascent.reporting.ic_brief_generator --claude           # use Claude API (requires key)
"""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

try:
    from ascent.llm.client import generate_structured as _llm_generate, HAIKU_MODEL
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    HAIKU_MODEL = "claude-haiku-4-5-20251001"


MEMO_DIR   = Path("outputs/20in20/memos")
OUTPUT_DIR = Path("outputs/20in20")


# ── Memo loading ───────────────────────────────────────────────────────────────

def _load_memo(target_date: date) -> dict:
    """Load the Intel memo JSON for a given date. Falls back to latest."""
    memo_path = MEMO_DIR / f"market_memo_{target_date.isoformat()}.json"

    if not memo_path.exists():
        memo_files = sorted(MEMO_DIR.glob("market_memo_*.json"), reverse=True)
        if memo_files:
            memo_path = memo_files[0]
            print(f"[ICBrief] No memo for {target_date}, using latest: {memo_path.name}")
        else:
            print(f"[ICBrief] No memo files found in {MEMO_DIR}")
            return {}

    try:
        with open(memo_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[ICBrief] Failed to load {memo_path}: {e}")
        return {}


# ── Template generation ────────────────────────────────────────────────────────

def _generate_local(memo: dict, target_date: date) -> str:
    """
    Template-based IC brief. No API call.
    Defensively extracts sections from the memo JSON using .get().
    """
    headline     = memo.get("headline", "Market Update")
    regime_raw   = memo.get("regime_summary", {})
    regime_label = regime_raw.get("label", "unknown") if isinstance(regime_raw, dict) else str(regime_raw)
    regime_note  = regime_raw.get("note", "") if isinstance(regime_raw, dict) else ""

    leaders   = memo.get("theme_leaders", [])
    laggards  = memo.get("theme_laggards", [])
    stretched = memo.get("rv_stretched", [])
    depressed = memo.get("rv_depressed", [])
    scenarios = memo.get("scenario_watch", [])
    takeaways = memo.get("key_takeaways", [])

    def _item_str(item):
        if isinstance(item, dict):
            name = item.get("theme") or item.get("name") or "Unknown"
            note = item.get("note") or item.get("description") or ""
            return f"{name}: {note}" if note else name
        return str(item)

    lines = [
        f"# IC Brief — {target_date.isoformat()}",
        "",
        "## Headline",
        headline,
        "",
        "## Regime Summary",
        f"**Current regime**: {regime_label}",
    ]
    if regime_note:
        lines.append(regime_note)
    lines.append("")

    # Risks — from laggards + scenarios
    lines.append("## Top 3 Risks")
    risks = [f"- {_item_str(x)}" for x in laggards[:2]]
    if scenarios:
        risks.append(f"- Scenario watch: {_item_str(scenarios[0])}")
    if not risks:
        risks = ["- Insufficient data to identify specific risks"]
    lines.extend(risks[:3])
    lines.append("")

    # Opportunities — from leaders + depressed RV
    lines.append("## Top 3 Opportunities")
    opps = [f"- {_item_str(x)}" for x in leaders[:2]]
    if depressed:
        opps.append(f"- Relative value: {_item_str(depressed[0])} appears depressed")
    if not opps:
        opps = ["- Insufficient data to identify specific opportunities"]
    lines.extend(opps[:3])
    lines.append("")

    # Scenarios to monitor
    lines.append("## Scenarios to Monitor")
    if scenarios:
        for sc in scenarios[:3]:
            lines.append(f"- {_item_str(sc)}")
    else:
        lines.append("- No active scenario watches")
    lines.append("")

    # Recommended posture
    lines.append("## Recommended Posture")
    posture_map = {
        "calm_bull": "Maintain full allocation. Favor momentum and growth.",
        "euphoric":  "Trim overweights. Raise cash buffer. Watch for reversal.",
        "stressed":  "Reduce equity exposure. Increase macro / defensive allocation.",
        "crisis":    "Defensive posture. Preserve capital. Hold until regime stabilizes.",
        "neutral":   "Maintain current allocation with heightened monitoring.",
    }
    lines.append(posture_map.get(regime_label.lower(), "Maintain current allocation with heightened monitoring."))
    lines.append("")

    # Key takeaways
    if takeaways:
        lines.append("## Key Takeaways")
        for t in takeaways:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — template mode (Claude API not configured)*")

    return "\n".join(lines)


# ── Claude API stub ────────────────────────────────────────────────────────────

def _generate_with_llm(memo: dict, target_date: date) -> str:
    """
    Generate IC brief using OpenRouter LLM (Qwen 3 14B, free tier).
    Falls back to template mode if API call fails or key is missing.
    """
    if not _LLM_AVAILABLE:
        print("[ICBrief] LLM client not available — falling back to template")
        return _generate_local(memo, target_date)

    system_prompt = """You are the Chief Investment Officer of Ascent Capital, \
an AI-native quantitative investment platform. You write concise, structured \
Investment Committee briefs. You are direct, data-driven, and avoid vague language.

Output format (markdown):
# IC Brief — [DATE]
## Headline
[One sentence summary of market state]
## Regime Summary
[Current regime label and what it means for positioning]
## Top 3 Risks
- [Risk 1 with specific data point]
- [Risk 2]
- [Risk 3]
## Top 3 Opportunities
- [Opportunity 1 with specific data point]
- [Opportunity 2]
- [Opportunity 3]
## Scenarios to Monitor
- [Scenario 1: trigger condition and portfolio impact]
- [Scenario 2]
## Recommended Posture
[Specific recommendation: maintain/reduce/increase exposure, which agents to favor]
## Key Takeaways
- [Takeaway 1]
- [Takeaway 2]
---
*Generated by Ascent Capital IC Brief Generator*"""

    import json as _json
    user_prompt = (
        f"Generate an IC brief for {target_date.isoformat()} "
        f"based on this Intel memo data:\n\n"
        f"{_json.dumps(memo, indent=2, default=str)}\n\n"
        f"Be specific. Reference actual symbols, regime labels, and data from the memo. "
        f"Keep it under 500 words."
    )

    try:
        brief = _llm_generate(system_prompt, user_prompt, model=HAIKU_MODEL, max_tokens=1500)
        print("[ICBrief] Generated via Claude Haiku")
        return brief
    except Exception as e:
        print(f"[ICBrief] LLM generation failed ({e}) — falling back to template")
        return _generate_local(memo, target_date)


def _generate_with_claude(memo: dict, target_date: date) -> str:
    """Legacy stub — redirects to _generate_with_llm."""
    return _generate_with_llm(memo, target_date)


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_ic_brief(target_date: date = None, use_llm: bool = True, use_claude: bool = None) -> str:
    """
    Generate an IC brief for the given date.

    Args:
        target_date: Date for the brief. Defaults to today.
        use_llm:     If True (default), use OpenRouter LLM. Falls back to template on failure.
        use_claude:  Legacy alias for use_llm (ignored if use_llm is set explicitly).

    Returns:
        Markdown string of the IC brief.
    """
    target_date = target_date or date.today()

    memo = _load_memo(target_date)
    if not memo:
        print(f"[ICBrief] No memo data for {target_date} — cannot generate brief")
        return ""

    if use_llm:
        brief = _generate_with_llm(memo, target_date)
    else:
        brief = _generate_local(memo, target_date)

    output_path = OUTPUT_DIR / f"ic_brief_{target_date.isoformat()}.md"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(brief)

    print(f"[ICBrief] Written to {output_path}")
    return brief


if __name__ == "__main__":
    target_date = None
    if "--date" in sys.argv:
        idx         = sys.argv.index("--date")
        target_date = date.fromisoformat(sys.argv[idx + 1])

    use_llm = "--template" not in sys.argv  # default is LLM mode now
    generate_ic_brief(target_date=target_date, use_llm=use_llm)
