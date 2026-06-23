## MISSION

Ascent Capital is an AI-native hedge fund where the AI PM is not an overlay — it IS the
primary alpha source. The quant stack produces raw signal; the AI PM synthesizes and
overrides. The system is underperforming: +8.8% live (Apr 1–Jun 3) vs SPY +15.9%.

Target: OOS Sharpe > 0.65 (from a verified-clean **0.41**; the old 0.483 came from a
corrupted price cache), excess CAGR > 5% vs SPY (from a verified +1.0pp), close live gap.

You have two files pre-pasted below. Read ONE additional file, then write full specs.

---

## EMBEDDED DIAGNOSTICS — these are already confirmed, do not re-derive

**Finding 1 — AI PM has never learned anything**
`data_cache/ai_pm_pattern_memory.json` is empty/invalid. Zero post-mortems have run.
The learning system (`ai_pm_learning.py`, pasted below) requires a decision log entry
with `overrides_applied` AND 21 days elapsed. Either no override-producing rebalance
has occurred, OR the decision log write is broken. Either way: the AI PM is running
with zero institutional memory after 5 weeks live.

**Finding 2 — AI PM produced 7 consecutive zero daily returns at launch**
From `earned_authority.json`, first 7 entries of `ai_returns_21d` = [0,0,0,0,0,0,0].
Quant returns same period = [+0.012, +0.014, +0.005, +0.017, +0.011, -0.025, +0.011].
The AI PM was making zero contribution for the first week while the quant ran.
Either: (a) weight modifications are being zeroed by a gate/guard, (b) ai_weight=0.05
is applied as a multiplicative scalar making changes negligible, or (c) Phase 2 calls
`propose_portfolio` but the result is not applied downstream.

**Finding 3 — Judge verdict history (all 3 rebalances ever run)**
- Apr 15: `reduce_size`, confidence=0.88, 0 position changes. Liberation Day rebalance.
  High-confidence reduce_size = system over-hedged the exact recovery. SPY then rallied.
  This is the primary driver of the +8.8% vs +15.9% live gap.
- May 27: `proceed`, confidence=0.62, 1 position change (1pp trim).
- Jun 10: `proceed`, confidence=0.65, 1 position change (PK: 7%→6%).

The judge can ONLY reduce positions (hardcoded in `judge.py`, pasted below:
`if new_w >= weights[sym]: continue`). On parse failure it defaults to `reduce_size`.
Combined: the system is structurally short alpha. It trims when uncertain, never adds
when confident.

---

## FILE TO READ

Read `agents/ai_pm_agent.py` in full. Focus on:

1. How `ai_weight` (0.05) is mathematically applied to weight modifications — is it a
   cap on change magnitude, a scalar multiplier, or a portfolio allocation fraction?
   What weight change does it produce on a typical ±2pp override recommendation?

2. Whether `_apply_recency_gate_python()` or any other guard can zero out modifications
   silently, producing the 7 zero-return days.

3. The Phase 2 prompt construction — does it QUOTE the prethesis text from
   `ai_prethesis_latest.json` and require the AI PM to follow it or explicitly override
   with a reason? Or is the prethesis loose "context" that Phase 2 can ignore?

4. Whether `run_ai_pm()` writes `overrides_applied=True` to the decision log. If this
   field is missing or False, `run_post_mortem()` will never trigger (it filters on this
   field), and pattern memory will never be populated — ever.

5. Whether tool failures in Phase 2 are silent. If MiroFish / options flow / COT return
   None and the prompt doesn't flag missing data, Phase 2 reasons from less information
   than it believes it has.

---

## THREE LENSES — tag every finding with one or more

**[MACRO-PM]** Druckenmiller / discretionary macro:
The Apr 15 rebalance had Liberation Day tariff risk visible in public data. Did the
prethesis produce a DIRECTIONAL, FALSIFIABLE view ("tariff fear is priced, hold
cyclicals, fade defensives") or a hedge ("elevated uncertainty, reduce exposure")?
The latter is not a thesis — it's risk management dressed as analysis. A real macro PM
has a view and sizes for asymmetry. Does this system produce views or hedges?

**[QUANT]** AQR / systematic:
Seven zero-return days = seven days of zero IC contribution from the AI PM. The 5%
authority should mean the AI PM shifts final weights by up to ±5% net. If actual
modifications are <0.1%, the AI PM is IC-neutral noise on top of the quant. Confirm
whether the math produces meaningful weight changes or trivial ones.

**[POD]** Citadel / Millennium multi-manager:
A pod risk officer reduces AND sizes up into high-conviction setups. The current
judge can only reduce. Combined with a `reduce_size` parse-failure default, the system
has a systematic negative bias — it hedges uncertainty but cannot press winners.
Is this the single largest structural alpha leak?

---

## FOUR DIAGNOSTIC QUESTIONS — answer each explicitly before writing specs

1. **Zero-return root cause**: Trace `run_ai_pm()` → weight modification → final applied
   weights. Find the exact line where 5% authority produces the weight delta. If
   `ai_weight=0.05` is a scalar on the CHANGE (e.g., `delta = proposed_delta * 0.05`),
   a ±2pp recommendation becomes ±0.1pp — invisible. Show the actual math.

2. **Prethesis binding**: Is the prethesis text injected into Phase 2 with a hard
   instruction ("your prethesis was X — either follow it or state why you are overriding
   it and what changed")? Or is it ambient context the model can ignore? Cite the prompt.

3. **Decision log write path**: Find where `run_ai_pm()` writes to `ai_pm_decision_log.jsonl`.
   Does that write include `overrides_applied: True` when overrides were made? If not,
   the post-mortem trigger in `ai_pm_learning.py` will never fire — confirm.

4. **Judge structural bias**: The judge defaults to `reduce_size` on failure AND can
   never add weight. Estimate: across the three rebalances, what was the net weight
   change from judge interventions (sum of all position_changes deltas)? Is it always
   negative (always reducing exposure)?

---

## PRE-PASTED FILE 1: ascent/strategy/ai_pm_learning.py

```python
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
```

---

## PRE-PASTED FILE 2: debate/judge.py

```python
"""
debate/judge.py
Synthesizes debate arguments into a structured verdict AND makes ONE position change.

Two outputs:
  1. verdict (proceed | reduce_size | halt_and_review) — gates execution as before
  2. position_changes — ONE specific weight adjustment with a falsifiable 10-day prediction

The position_changes output is the Adversarial Intelligence layer's key contribution.
Every change is logged by adversarial_authority.py and scored 10 days later.
"""

import json
from ascent.llm.client import generate_structured, extended_thinking_completion, DEFAULT_MODEL
from debate.outcome_tracker import load_credibility_context, load_recent_verdict_outcomes


def run_judge(
    bull_argument:    str,
    bear_argument:    str,
    devils_argument:  str,
    portfolio_state:  dict,
    regime_arg:       str = "",
    quant_check:      str = "",
    round2_args:      dict = None,
    disagreement_context: str = "",
    adversarial_context:  str = "",
    adversarial_engine:   dict = None,
) -> dict:
    """
    Synthesize all debate arguments into a verdict + ONE adversarial position change.

    Defaults to 'reduce_size' + no position_changes on parsing failure.
    Defaults to empty position_changes if no flag clears the authority threshold.
    """
    round2_args       = round2_args or {}
    adversarial_engine = adversarial_engine or {}
    quant_block       = quant_check and "✗" in quant_check

    regime           = portfolio_state.get("us_regime")
    cred_context     = load_credibility_context(regime)
    recent_outcomes  = load_recent_verdict_outcomes(n=5)

    # Load adversarial authority state for the prompt
    authority_context = ""
    try:
        from debate.adversarial_authority import format_authority_for_judge
        authority_context = format_authority_for_judge(regime or "unknown")
    except Exception:
        pass

    weights = portfolio_state.get("weights", {})

    context = (
        f"Date: {portfolio_state.get('date', 'unknown')}\n"
        f"Regime: {portfolio_state.get('us_regime', 'unknown')}\n"
        f"Positions: {portfolio_state.get('n_positions', 0)}\n\n"
        f"BULL ARGUMENT:\n{bull_argument}\n\n"
        f"BEAR ARGUMENT:\n{bear_argument}\n\n"
        f"DEVIL'S ADVOCATE:\n{devils_argument}\n\n"
        + (f"REGIME SPECIALIST:\n{regime_arg}\n\n" if regime_arg else "")
        + (f"QUANT SANITY CHECK:\n{quant_check}\n\n" if quant_check else "")
        + ("NOTE: The quant sanity check found objective numerical issues (marked ✗). "
           "These are facts, not opinions — weight them heavily.\n\n" if quant_block else "")
        + (f"\n{cred_context}\n" if cred_context else "")
        + (f"\n{recent_outcomes}\n" if recent_outcomes else "")
        + (
            "\n\nROUND 2 — REBUTTALS:\n"
            + (f"BULL REBUTTAL:\n{round2_args['bull_rebuttal']}\n\n"
               if round2_args.get("bull_rebuttal") else "")
            + (f"BEAR REBUTTAL:\n{round2_args['bear_rebuttal']}\n\n"
               if round2_args.get("bear_rebuttal") else "")
            + (f"DEVIL'S ADVOCATE REBUTTAL:\n{round2_args['devils_advocate_rebuttal']}\n\n"
               if round2_args.get("devils_advocate_rebuttal") else "")
            + (f"REGIME SPECIALIST REBUTTAL:\n{round2_args['regime_specialist_rebuttal']}\n\n"
               if round2_args.get("regime_specialist_rebuttal") else "")
            if round2_args else ""
        )
        + (f"\n\n{adversarial_context}\n" if adversarial_context else "")
        + (f"\n\n{authority_context}\n" if authority_context else "")
    )

    top_flags   = adversarial_engine.get("top_flags", [])
    flags_json  = json.dumps(top_flags[:3], indent=2) if top_flags else "[]"

    system_prompt = (
        "You are the Portfolio Manager and final decision-maker at Ascent Capital. "
        "You synthesize two things:\n\n"
        "1. MACRO VERDICT: Is the overall portfolio direction correct? "
        "Options: 'proceed' | 'reduce_size' | 'halt_and_review'\n"
        "   - halt_and_review: ONLY for catastrophic risk (systemic event, fund-level error)\n"
        "   - reduce_size: broad portfolio risk warrants general trimming\n"
        "   - proceed: execute as proposed\n\n"
        "2. ONE ADVERSARIAL POSITION CHANGE: The single most important specific intervention. "
        "Pick from the top adversarial flags (ranked by priority). "
        "This is a FALSIFIABLE PREDICTION — state exactly what you expect to happen in 10 days.\n"
        "   Rules:\n"
        "   - ONE change only. Forced prioritization.\n"
        "   - Maximum weight change = what the authority level allows per intervention type\n"
        "   - SUSPENDED types may NOT be used\n"
        "   - Weight change must be ≥1% to be worth the intervention cost\n"
        "   - If no flag clears the bar, set position_changes to []\n"
        "   - Cannot increase any position (only reduce)\n"
        "   - Cannot drop below 1% (use 1% as floor)\n\n"
        f"Current positions for reference:\n"
        + "\n".join(f"  {s}: {w:.1%}" for s, w in
                    sorted(weights.items(), key=lambda x: -x[1])[:15])
        + f"\n\nTop adversarial flags:\n{flags_json}\n\n"
        "Respond with ONLY valid JSON (no markdown, no text outside the JSON):\n"
        "{\n"
        '    "confidence": 0.0 to 1.0,\n'
        '    "recommendation": "proceed" or "reduce_size" or "halt_and_review",\n'
        '    "key_risks": ["risk 1", "risk 2", "risk 3"],\n'
        '    "reasoning": "one paragraph",\n'
        '    "position_changes": [\n'
        '        {\n'
        '            "symbol": "TICKER",\n'
        '            "current_weight": 0.00,\n'
        '            "new_weight": 0.00,\n'
        '            "intervention_type": "adversarial_thesis|regime_sizing|coherence_risk|event_risk",\n'
        '            "reason": "specific reason referencing the data",\n'
        '            "prediction": "X underperforms/outperforms Y over next 10 trading days"\n'
        "        }\n"
        "    ]\n"
        "}"
    )
    if disagreement_context:
        system_prompt += f"\n\n{disagreement_context}"

    try:
        raw = extended_thinking_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": context},
            ],
            model=DEFAULT_MODEL,
            max_tokens=5000,
            thinking_budget=3000,
        )

        start = raw.find("{")
        depth = 0
        end   = -1
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in response")

        verdict = json.loads(raw[start:end])

        assert "confidence" in verdict
        assert "recommendation" in verdict
        assert verdict["recommendation"] in ("proceed", "reduce_size", "halt_and_review")

        # Validate + clamp position_changes
        position_changes = verdict.get("position_changes", [])
        validated_changes = []
        for change in position_changes[:1]:  # enforce ONE change max
            sym     = change.get("symbol", "")
            new_w   = float(change.get("new_weight", 0))
            curr_w  = float(change.get("current_weight", weights.get(sym, 0)))
            itype   = change.get("intervention_type", "adversarial_thesis")

            # Authority check
            try:
                from debate.adversarial_authority import get_authority
                auth = get_authority(itype)
                if auth["suspended"]:
                    print(f"[Judge] Skipping {itype} intervention on {sym} — type suspended")
                    continue
                max_change = auth["allowed_change_pct"]
                # Clamp new_weight to not exceed max_change reduction
                min_allowed = max(0.01, curr_w - max_change)
                new_w = max(new_w, min_allowed)
            except Exception:
                pass

            if sym not in weights:
                continue
            if new_w >= weights[sym]:  # must be a reduction
                continue
            if abs(weights[sym] - new_w) < 0.005:  # <0.5pp change = not worth it
                continue

            change["current_weight"] = round(weights.get(sym, curr_w), 6)
            change["new_weight"]     = round(new_w, 6)
            validated_changes.append(change)

        verdict["position_changes"] = validated_changes

        return verdict

    except (json.JSONDecodeError, AssertionError, Exception) as e:
        print(f"[Judge] Failed to parse verdict ({e}) — defaulting to reduce_size (safe failure)")
        return {
            "confidence":      0.3,
            "recommendation":  "reduce_size",
            "key_risks":       [f"Judge parsing failed ({str(e)[:80]}) — safe default"],
            "reasoning":       "LLM output could not be parsed. Defaulting to reduce_size.",
            "position_changes": [],
        }
```

---

## OUTPUT FORMAT

### SECTION 1 — PUNCH LIST

Group by severity: CRITICAL → HIGH → MEDIUM
Format per item: `[file:line] — finding — [MACRO-PM / QUANT / POD] — one-line alpha impact`

### SECTION 2 — FULL SPEC PER FINDING

For every item in the punch list:

```
## [Title]
Severity: CRITICAL/HIGH/MEDIUM  |  Lens: [MACRO-PM / QUANT / POD]
Problem: what is broken and why it costs alpha
Root cause: specific function + line number
Fix: exact code change — what to add, remove, or change
Files: [file.py:function_name()]
Success metric: what confirms this worked
Estimated impact: effect on Sharpe / alpha / live P&L
```

Be specific. Cite file paths and function names. No generic advice.
The goal is to make the AI PM the genuine alpha edge it is supposed to be.
