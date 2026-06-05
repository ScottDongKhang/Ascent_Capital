# ascent/strategy/ai_pm_learning.py
"""
AI PM continuous learning system. Three components:

1. daily_intelligence_brief()  — Sonnet daily analysis of held positions + macro + thesis health
   Called every non-rebalance day after _log_holdings(). ~$0.03/day = ~$7/year.

2. run_post_mortem()  — Sonnet deep analysis 21 days after a rebalance, when outcomes are known.
   Called automatically when 21+ days have elapsed since the last logged rebalance.
   ~$0.04/rebalance × 26 = ~$1/year.

3. update_pattern_memory()  — extracts patterns from post-mortems into a growing JSON playbook.
   Called after each post-mortem. Haiku. ~$0.01/rebalance × 26 = ~$0.26/year.

Together these give the AI PM ~270 structured learning sessions per year (vs 26 currently),
building pattern memory equivalent to years of Wall Street experience compressed into months.
"""
from __future__ import annotations
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_REPO           = Path(__file__).resolve().parent.parent.parent
PATTERN_MEMORY  = _REPO / "data_cache" / "ai_pm_pattern_memory.json"
POSTMORTEM_LOG  = _REPO / "logs" / "ai_pm_postmortems.jsonl"
BRIEF_LOG       = _REPO / "logs" / "ai_pm_daily_briefs.jsonl"


# ── 1. Daily Intelligence Brief ────────────────────────────────────────────────

def daily_intelligence_brief(
    today: date,
    positions: list,
    price_returns: dict,
    feedback: dict,
    macro_context: Optional[dict] = None,
) -> str:
    """
    Sonnet daily brief: thesis health per position + macro read + risk flags.
    Returns the brief text. Logs to ai_pm_daily_briefs.jsonl.
    """
    from ascent.llm.client import SONNET_MODEL
    import anthropic
    client = anthropic.Anthropic()

    level = feedback.get("level", 0)
    title = feedback.get("title", "Analyst")

    # Load pattern memory for context
    patterns = _load_pattern_memory()
    pattern_context = ""
    if patterns.get("avoid"):
        pattern_context = "\nPATTERNS YOU HAVE LEARNED TO AVOID:\n" + \
            "\n".join(f"  • {p}" for p in patterns["avoid"][-3:])
    if patterns.get("work"):
        pattern_context += "\nPATTERNS THAT WORK FOR YOU:\n" + \
            "\n".join(f"  • {p}" for p in patterns["work"][-3:])

    # Build position table with today's moves
    pos_lines = []
    for p in sorted(positions, key=lambda x: -abs(price_returns.get(x.get("symbol",""), 0))):
        sym = p.get("symbol", "")
        w   = p.get("weight", 0)
        ret = price_returns.get(sym)
        ret_str = f"{ret:+.2%}" if ret is not None else "N/A"
        pos_lines.append(f"  {sym:<6} {w:.1%}  {ret_str}")
    pos_table = "\n".join(pos_lines)

    # Macro context from FRED cache
    macro_str = ""
    if macro_context:
        macro_str = f"\nMACRO TODAY: SPY {macro_context.get('spy',0):+.2%} | VIX {macro_context.get('vix','?')} | 10Y {macro_context.get('t10y','?')} | HY spread {macro_context.get('hy_spread','?')}"

    # Load last thesis to check health
    last_thesis = _load_last_thesis()
    thesis_context = ""
    if last_thesis:
        thesis_context = f"\nYOUR LAST REBALANCE THESIS (check if still intact):\n{last_thesis[:400]}"

    worst = feedback.get("worst_call_10d") or {}
    worst_str = f"{worst.get('symbol')} ({worst.get('alpha',0):+.1%} over 10d)" if worst.get("symbol") else "none"

    prompt = f"""You are an experienced portfolio manager reviewing your book. Today is {today.isoformat()}.
Level: {title} | Worst recent call: {worst_str}{macro_str}{pattern_context}{thesis_context}

HELD POSITIONS (symbol | weight | today's return):
{pos_table}

Give a sharp, experienced daily review in three sections:

**THESIS HEALTH** — for each position that moved >1% today: is the original thesis intact, strengthening, weakening, or broken? Be specific about why. If you don't know the specific reason for a move, say so honestly rather than guessing.

**BIGGEST RISK RIGHT NOW** — one specific risk to the portfolio that wasn't there a week ago. Could be a position showing technical weakness, a macro shift, or a catalyst approaching.

**WHAT TO WATCH BEFORE NEXT REBALANCE** — one or two names that need a decision at the next rebalance and what you'd need to see to act.

Be direct, specific, and critical. Reference the actual return numbers. A Wall Street vet doesn't hedge every sentence."""

    resp = client.messages.create(
        model=SONNET_MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    brief_text = resp.content[0].text if resp.content else ""

    BRIEF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIEF_LOG, "a") as f:
        f.write(json.dumps({
            "date":          today.isoformat(),
            "level":         level,
            "price_returns": price_returns,
            "brief":         brief_text,
        }) + "\n")

    return brief_text


# ── 2. Post-Rebalance Post-Mortem ──────────────────────────────────────────────

def run_post_mortem(today: date, feedback: dict) -> Optional[str]:
    """
    21 days after a rebalance, run a deep post-mortem on what worked and why.
    Returns post-mortem text, or None if not enough time has elapsed.
    """
    decisions = _load_decisions()
    if not decisions:
        return None

    # Find the most recent decision that hasn't had a post-mortem yet
    # and is at least 21 days old
    postmortems_done = set()
    if POSTMORTEM_LOG.exists():
        for line in POSTMORTEM_LOG.read_text().splitlines():
            try:
                postmortems_done.add(json.loads(line).get("rebalance_date"))
            except Exception:
                pass

    target = None
    for dec in reversed(decisions):
        dec_date = dec.get("date", "")
        if dec_date in postmortems_done:
            continue
        if dec.get("overrides_applied") and \
           (today - date.fromisoformat(dec_date)).days >= 21:
            target = dec
            break

    if not target:
        return None

    # Get outcomes from feedback
    scored = [s for s in feedback.get("last_5_decisions", [])
              if s.get("date", "") >= target["date"]]
    if not scored:
        return None

    from ascent.llm.client import SONNET_MODEL
    import anthropic
    client = anthropic.Anthropic()

    patterns = _load_pattern_memory()

    outcomes_text = "\n".join(
        f"  {s['symbol']:6s} {s['type']:8s}  AI={s['ai_w']:.1%} vs Quant={s['quant_w']:.1%}"
        f"  10d={s.get('outcome_10d',0):+.3%}  21d={s.get('outcome_21d',0):+.3%}"
        f"  → {(s.get('verdict','?') or '?').upper()}"
        for s in scored
    )

    prompt = f"""You are doing a post-mortem on your rebalance from {target['date']} (21+ days ago).

YOUR OVERRIDES AT THE TIME:
{outcomes_text}

YOUR THESIS SUMMARY:
{target.get('thesis_summary', 'N/A')}

OVERALL: Hit rate {feedback.get('hit_rate_21d',0):.0%} | Profit factor {feedback.get('profit_factor',0):.2f}x | Fade rate {feedback.get('fade_rate',0):.0%}

EXISTING PATTERNS YOU TRACK:
Avoid: {patterns.get('avoid', [])}
Work: {patterns.get('work', [])}

Do a honest, self-critical post-mortem in three parts:

**WHAT WORKED AND WHY** — for each win: was the thesis right? Was it the right reason? Would you make the same call again?

**WHAT FAILED AND WHY** — for each loss/fade: what did you miss? Was it timing, wrong thesis, wrong catalyst, or bad regime fit? Be specific, not generic.

**ONE RULE TO ADD TO YOUR PLAYBOOK** — based on this rebalance, what's one specific, actionable pattern to remember? Format: "When [condition], [action] because [reason]." Make it concrete enough to apply next time."""

    resp = client.messages.create(
        model=SONNET_MODEL, max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    mortem_text = resp.content[0].text if resp.content else ""

    POSTMORTEM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(POSTMORTEM_LOG, "a") as f:
        f.write(json.dumps({
            "date":            today.isoformat(),
            "rebalance_date":  target["date"],
            "post_mortem":     mortem_text,
        }) + "\n")

    log.info("[AIPMLearning] Post-mortem written for rebalance %s", target["date"])
    return mortem_text


# ── 3. Pattern Memory ──────────────────────────────────────────────────────────

def update_pattern_memory(post_mortem_text: str, today: date) -> None:
    """
    Extract the 'one rule' from the post-mortem and add it to the pattern memory.
    Uses Haiku — cheap extraction, not generation.
    """
    from ascent.llm.client import HAIKU_MODEL
    import anthropic
    client = anthropic.Anthropic()

    patterns = _load_pattern_memory()

    prompt = f"""Extract the key learnings from this post-mortem into structured rules.

POST-MORTEM:
{post_mortem_text}

Return ONLY a JSON object with two arrays:
{{
  "avoid": ["one specific pattern to avoid, formatted as: When X, avoid Y because Z"],
  "work":  ["one specific pattern that works, formatted as: When X, do Y because Z"]
}}

Each entry must be actionable and specific. If no clear pattern emerged, return empty arrays.
Return ONLY the JSON, no other text."""

    resp = client.messages.create(
        model=HAIKU_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        extracted = json.loads(resp.content[0].text.strip())
        if extracted.get("avoid"):
            patterns.setdefault("avoid", []).extend(extracted["avoid"])
            patterns["avoid"] = patterns["avoid"][-20:]  # keep last 20
        if extracted.get("work"):
            patterns.setdefault("work", []).extend(extracted["work"])
            patterns["work"] = patterns["work"][-20:]

        patterns["last_updated"] = today.isoformat()
        patterns["total_postmortems"] = patterns.get("total_postmortems", 0) + 1
        _save_pattern_memory(patterns)
        log.info("[AIPMLearning] Pattern memory updated: %d avoid, %d work patterns",
                 len(patterns.get("avoid", [])), len(patterns.get("work", [])))
    except Exception as e:
        log.warning("[AIPMLearning] Pattern extraction failed: %s", e)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_pattern_memory() -> dict:
    if PATTERN_MEMORY.exists():
        try:
            return json.loads(PATTERN_MEMORY.read_text())
        except Exception:
            pass
    return {"avoid": [], "work": [], "total_postmortems": 0}


def _save_pattern_memory(patterns: dict) -> None:
    PATTERN_MEMORY.parent.mkdir(parents=True, exist_ok=True)
    PATTERN_MEMORY.write_text(json.dumps(patterns, indent=2))


def _load_decisions() -> list:
    path = _REPO / "logs" / "ai_pm_decision_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return sorted(rows, key=lambda x: x.get("date", ""))


def _load_last_thesis() -> str:
    """Load the most recent AI PM thesis summary for context."""
    try:
        theses = sorted(Path(_REPO / "outputs" / "ai_pm_theses").glob("*.json"))
        if theses:
            t = json.loads(theses[-1].read_text())
            mv = t.get("market_view", "")
            ov = t.get("quant_overrides", [])
            ov_str = ", ".join(f"{o.get('symbol')} ({o.get('type')})" for o in ov[:3])
            return f"Market view: {mv[:200]}\nOverrides: {ov_str}"
    except Exception:
        pass
    return ""


def get_pattern_summary() -> str:
    """Return a formatted summary of pattern memory for injection into prompts."""
    patterns = _load_pattern_memory()
    if not patterns.get("avoid") and not patterns.get("work"):
        return ""
    lines = [f"AI PM PATTERN MEMORY ({patterns.get('total_postmortems',0)} post-mortems):"]
    if patterns.get("avoid"):
        lines.append("AVOID:")
        lines.extend(f"  • {p}" for p in patterns["avoid"][-5:])
    if patterns.get("work"):
        lines.append("WORKS:")
        lines.extend(f"  • {p}" for p in patterns["work"][-5:])
    return "\n".join(lines)
