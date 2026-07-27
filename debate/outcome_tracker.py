"""
debate/outcome_tracker.py
Scores past debate verdicts against what actually happened.
Tracks per-agent win rates by regime so the judge can learn.

Called two ways:
1. Automatically after each EOD run (score_pending_verdicts)
2. By agents.py / judge.py to pull credibility context

Scoring logic:
- PROCEED was correct   → portfolio flat or up in 14 days
- REDUCE_SIZE correct   → portfolio down in 14 days (hedging paid)
- HALT_AND_REVIEW correct → portfolio down ≥1% in 14 days
- Partial credit for directional accuracy
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

DEBATE_LOG_DIR  = Path("outputs/debate_log")
EOD_LOG_PATH    = Path("logs/eod_log.jsonl")
CREDIBILITY_PATH = Path("outputs/debate_log/agent_credibility.json")
OUTCOME_WINDOW_DAYS = 14  # days after verdict to measure outcome

# Minimum scored samples required before a track-record figure is shown to the
# judge as an accuracy percentage rather than an explicit "insufficient data"
# note. Same threshold as get_agent_regime_accuracy's min_samples convention
# below — with fewer scored outcomes, the estimate is noise (see W4 finding:
# a single 2026-04-15 record was presented to the judge as a track record and
# suppressed a trim).
MIN_SAMPLES_FOR_TRACK_RECORD = 10

_REGIME_LABEL_PREFIX = "regimelabel."


def _normalize_regime_key(regime) -> str:
    """
    Normalize a regime label for use as a credibility-bucket key.

    Strips a leading "RegimeLabel." (from `str(RegimeSignal.enum_member)`)
    and lowercases, so "RegimeLabel.STRESSED" and "stressed" land in the same
    bucket instead of splitting sample counts across two keys.
    """
    s = str(regime).strip().lower()
    if s.startswith(_REGIME_LABEL_PREFIX):
        s = s[len(_REGIME_LABEL_PREFIX):]
    return s


# ── Credibility loader ───────────────────────────────────────────────────────

def _load_credibility() -> dict:
    """Load agent_credibility.json. Returns empty dict on any failure."""
    if CREDIBILITY_PATH.exists():
        try:
            return json.loads(CREDIBILITY_PATH.read_text())
        except Exception:
            pass
    return {}


def get_agent_regime_accuracy(agent_name: str, regime: str,
                               min_samples: int = 10) -> Optional[float]:
    """
    Return per-agent accuracy in a specific regime, or None if insufficient data.

    Args:
        agent_name:  "bull", "bear", "devil", or "regime_specialist"
        regime:      Regime label string, e.g. "stressed", "calm_bull"
        min_samples: Minimum number of scored debates required to return a value.
                     Set to 10 — with fewer debates the accuracy estimate is noise.
                     Expect this to return None for all agents until ~August 2026.

    Returns:
        Float accuracy in [0, 1], or None if no data / too few samples.
    """
    cred = _load_credibility()
    regime_key = _normalize_regime_key(regime)
    accuracy   = cred.get("by_regime", {}).get(regime_key, {}).get(agent_name)
    n_samples  = cred.get("sample_counts", {}).get(regime_key, {}).get(agent_name, 0)
    if accuracy is None or n_samples < min_samples:
        return None
    return float(accuracy)


# ── NAV loader ────────────────────────────────────────────────────────────────

def _load_nav_series() -> dict:
    """Returns {date_str: equity} from holdings_log.jsonl. Falls back to eod_log."""
    nav = {}
    # Primary: holdings_log (real Alpaca equity)
    holdings_path = Path("logs/holdings_log.jsonl")
    if holdings_path.exists():
        for line in holdings_path.read_text().splitlines():
            try:
                e = json.loads(line)
                d = e.get("date")
                pv = e.get("equity")
                if d and pv:
                    nav[d] = float(pv)
            except Exception:
                pass
        if nav:
            return nav
    # Legacy fallback: eod_log
    legacy_path = Path("logs/eod_log.jsonl")
    if legacy_path.exists():
        for line in legacy_path.read_text().splitlines():
            try:
                e = json.loads(line)
                d = e.get("date") or e.get("run_date")
                pv = e.get("nav") or e.get("portfolio_value")
                if d and pv:
                    nav[d] = float(pv)
            except Exception:
                pass
    return nav


# ── Outcome scorer ────────────────────────────────────────────────────────────

def _score_verdict(recommendation: str, nav_change_pct: float) -> float:
    """
    Returns a correctness score 0.0–1.0.
    
    Logic:
      PROCEED        → correct if nav_change >= 0, partial if -1% to 0%, wrong if < -1%
      REDUCE_SIZE    → correct if nav_change < -0.5%, partial if -0.5% to +0.5%, wrong if > +0.5%
      HALT_AND_REVIEW → correct if nav_change < -1%, partial if -1% to 0%, wrong if > 0%
    """
    rec = recommendation.lower()
    c = nav_change_pct

    if rec == "proceed":
        if c >= 0:      return 1.0
        if c >= -0.01:  return 0.5
        return 0.0

    if rec == "reduce_size":
        if c < -0.005:  return 1.0
        if c < 0.005:   return 0.5
        return 0.0

    if rec == "halt_and_review":
        if c < -0.01:   return 1.0
        if c <= 0:      return 0.5
        return 0.0

    return 0.5  # unknown recommendation


def score_pending_verdicts() -> int:
    """
    Scan all verdict files. For any that are OUTCOME_WINDOW_DAYS old
    and don't yet have an outcome score, compute and write it back.
    Returns count of verdicts scored.
    """
    if not DEBATE_LOG_DIR.exists():
        return 0

    nav = _load_nav_series()
    scored = 0

    for vf in sorted(DEBATE_LOG_DIR.glob("verdict_*.json")):
        try:
            rec = json.loads(vf.read_text())
        except Exception:
            continue

        # Skip if already scored
        if rec.get("outcome_scored"):
            continue

        vdate_str = rec.get("date", "")
        try:
            vdate = date.fromisoformat(vdate_str)
        except Exception:
            continue

        # Not old enough yet
        if date.today() - vdate < timedelta(days=OUTCOME_WINDOW_DAYS):
            continue

        # Find NAV at verdict date and OUTCOME_WINDOW_DAYS later
        verdict_nav = nav.get(vdate_str)
        future_dates = [(vdate + timedelta(days=i)).isoformat()
                        for i in range(1, OUTCOME_WINDOW_DAYS + 2)]
        future_navs  = [(d, nav[d]) for d in future_dates if d in nav]

        if not verdict_nav or not future_navs:
            continue

        final_date, final_nav = future_navs[-1]
        nav_change = (final_nav - verdict_nav) / verdict_nav

        recommendation = rec.get("verdict", {}).get("recommendation", "proceed")
        score = _score_verdict(recommendation, nav_change)

        # Write outcome back into the verdict file
        rec["outcome_scored"]    = True
        rec["outcome_nav_change"] = round(nav_change, 6)
        rec["outcome_window_end"] = final_date
        rec["outcome_score"]     = score  # 0.0, 0.5, or 1.0

        # Per-agent scores (crude: all agents share the verdict score)
        # Refined: if bear/devil raised risks that materialized, they get bonus
        rec["agent_scores"] = _score_individual_agents(rec, nav_change)

        with open(vf, "w") as f:
            json.dump(rec, f, indent=2, default=str)

        scored += 1

    if scored:
        _rebuild_credibility()

    return scored


def _score_individual_agents(record: dict, nav_change: float) -> dict:
    """
    Score each agent individually.
    Bear and devil get bonus credit when nav drops and they flagged risk.
    Bull gets bonus credit when nav rises and they argued for it.
    """
    recommendation = record.get("verdict", {}).get("recommendation", "proceed")
    base = _score_verdict(recommendation, nav_change)

    args = record.get("arguments", {})
    bull_text  = (args.get("bull") or "").lower()
    bear_text  = (args.get("bear") or "").lower()
    devil_text = (args.get("devils_advocate") or "").lower()

    def _risk_keywords_present(text):
        keywords = ["risk", "concern", "warning", "danger", "caution",
                    "avoid", "reduce", "downside", "volatile", "uncertain"]
        return sum(1 for k in keywords if k in text)

    bear_risk_count  = _risk_keywords_present(bear_text)
    devil_risk_count = _risk_keywords_present(devil_text)
    bull_conf_count  = _risk_keywords_present(bull_text)  # bull flagging own risks = humility

    # Bear/devil rewarded more when they warned and nav dropped
    bear_bonus  = min(0.2, bear_risk_count  * 0.02) if nav_change < -0.005 else 0
    devil_bonus = min(0.2, devil_risk_count * 0.02) if nav_change < -0.005 else 0
    bull_bonus  = 0.1 if nav_change > 0.005 else 0

    return {
        "bull":            round(min(1.0, base + bull_bonus),  3),
        "bear":            round(min(1.0, base + bear_bonus),  3),
        "devils_advocate": round(min(1.0, base + devil_bonus), 3),
    }


# ── Credibility tracker ───────────────────────────────────────────────────────

def _rebuild_credibility():
    """
    Recompute per-agent win rates by regime from all scored verdicts.
    Writes to outputs/debate_log/agent_credibility.json.
    """
    if not DEBATE_LOG_DIR.exists():
        return

    # Structure: {regime: {agent: [scores]}}
    from collections import defaultdict
    buckets: dict = defaultdict(lambda: defaultdict(list))
    overall: dict = defaultdict(list)

    for vf in DEBATE_LOG_DIR.glob("verdict_*.json"):
        try:
            rec = json.loads(vf.read_text())
        except Exception:
            continue

        if not rec.get("outcome_scored"):
            continue

        # Normalize so "RegimeLabel.STRESSED" and "stressed" share one bucket
        # instead of silently splitting sample counts (see W4 finding).
        regime = _normalize_regime_key(rec.get("portfolio_state", {}).get("us_regime", "unknown"))
        agent_scores = rec.get("agent_scores", {})

        for agent, score in agent_scores.items():
            buckets[regime][agent].append(score)
            overall[agent].append(score)

    credibility = {"by_regime": {}, "overall": {}, "sample_counts": {},
                   "overall_sample_counts": {}}

    for regime, agents in buckets.items():
        credibility["by_regime"][regime] = {}
        credibility["sample_counts"][regime] = {}
        for agent, scores in agents.items():
            if len(scores) > 0:
                credibility["by_regime"][regime][agent] = round(sum(scores) / len(scores), 3)
                credibility["sample_counts"][regime][agent] = len(scores)

    for agent, scores in overall.items():
        if len(scores) > 0:
            credibility["overall"][agent] = round(sum(scores) / len(scores), 3)
            credibility["overall_sample_counts"][agent] = len(scores)

    DEBATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CREDIBILITY_PATH, "w") as f:
        json.dump(credibility, f, indent=2)


# ── Context loader (called by agents.py and judge.py) ────────────────────────

def load_credibility_context(regime: Optional[str] = None,
                              min_samples: int = MIN_SAMPLES_FOR_TRACK_RECORD) -> str:
    """
    Returns a plain-English credibility summary for injection into agent prompts.
    E.g. "In stressed regime: bear agent accuracy 78% (9 debates), bull 44%, devil 67%"

    Below `min_samples` scored debates, the accuracy figure is withheld and
    replaced with an explicit insufficient-data note — with n<10 the estimate
    is noise (see MIN_SAMPLES_FOR_TRACK_RECORD).
    """
    if not CREDIBILITY_PATH.exists():
        return ""

    try:
        cred = json.loads(CREDIBILITY_PATH.read_text())
    except Exception:
        return ""

    lines = ["AGENT TRACK RECORD (historical accuracy):"]

    # Regime-specific if available
    if regime:
        regime_key = _normalize_regime_key(regime)
        by_regime  = cred.get("by_regime", {})

        # Exact match on the normalized key first (deterministic). Fall back
        # to a normalized substring match only if no exact key exists, and
        # break ties deterministically (sorted) rather than on dict iteration
        # order.
        matched = regime_key if regime_key in by_regime else None
        if matched is None:
            candidates = sorted(
                k for k in by_regime
                if k in regime_key or regime_key in k
            )
            matched = candidates[0] if candidates else None

        if matched:
            regime_data   = by_regime[matched]
            sample_counts = cred.get("sample_counts", {}).get(matched, {})
            lines.append(f"\nIn {matched} regime:")
            for agent in ["bull", "bear", "devils_advocate"]:
                acc   = regime_data.get(agent)
                n     = sample_counts.get(agent, 0)
                label = agent.replace("_", " ").title()
                if acc is None:
                    lines.append(f"  {label}: no data yet for this regime")
                elif n < min_samples:
                    lines.append(
                        f"  {label}: n={n}, insufficient to infer a track record "
                        f"(min {min_samples} required)"
                    )
                else:
                    pct = f"{acc:.0%}"
                    lines.append(f"  {label}: {pct} accuracy ({n} debates)")

    # Overall always shown
    overall       = cred.get("overall", {})
    overall_n_map = cred.get("overall_sample_counts", {})
    if overall:
        lines.append("\nOverall (all regimes):")
        for agent in ["bull", "bear", "devils_advocate"]:
            acc   = overall.get(agent)
            n     = overall_n_map.get(agent, 0)
            label = agent.replace("_", " ").title()
            if acc is None:
                continue
            if n < min_samples:
                lines.append(
                    f"  {label}: n={n}, insufficient to infer a track record "
                    f"(min {min_samples} required)"
                )
            else:
                lines.append(f"  {label}: {acc:.0%}")

    return "\n".join(lines) if len(lines) > 1 else ""


def load_recent_verdict_outcomes(n: int = 5,
                                  min_samples: int = MIN_SAMPLES_FOR_TRACK_RECORD) -> str:
    """
    Returns a summary of the last N scored verdicts with their outcomes.
    For injection into the judge prompt.

    Below `min_samples` total scored verdicts, individual outcomes are
    withheld and replaced with an explicit insufficient-data note. This is
    the guard for the exact failure mode found in the W4 audit: a single
    scored verdict (2026-04-15, a whole-portfolio-NAV outcome unrelated to
    any specific position) was presented to the judge as "JUDGE'S OWN TRACK
    RECORD" and used to justify suppressing an unrelated single-name trim.
    """
    if not DEBATE_LOG_DIR.exists():
        return ""

    verdict_files = sorted(DEBATE_LOG_DIR.glob("verdict_*.json"), reverse=True)
    scored_recs = []
    for vf in verdict_files:
        try:
            rec = json.loads(vf.read_text())
        except Exception:
            continue
        if rec.get("outcome_scored"):
            scored_recs.append(rec)

    total_scored = len(scored_recs)
    if total_scored < min_samples:
        return (
            f"JUDGE'S OWN TRACK RECORD: n={total_scored}, insufficient to infer a "
            f"track record (min {min_samples} required). Treat any individual past "
            f"outcome as anecdote, not evidence — do not let it override the current "
            f"analysis."
        )

    lines = ["JUDGE'S OWN TRACK RECORD (recent scored decisions):"]
    found = 0

    for rec in scored_recs:
        if found >= n:
            break

        vdate   = rec.get("date", "?")
        verdict = rec.get("verdict", {})
        rec_str = verdict.get("recommendation", "?")
        conf    = verdict.get("confidence", 0)
        score   = rec.get("outcome_score", "?")
        chg     = rec.get("outcome_nav_change", 0)
        regime  = rec.get("portfolio_state", {}).get("us_regime", "?")

        chg_str = f"{'+' if chg >= 0 else ''}{chg:.1%}" if isinstance(chg, float) else "?"
        score_label = {1.0: "CORRECT", 0.5: "PARTIAL", 0.0: "WRONG"}.get(score, "?")

        lines.append(
            f"  [{vdate}] {rec_str.upper()} (conf {conf:.0%}, regime {regime}) "
            f"→ {chg_str} → {score_label}"
        )
        found += 1

    return "\n".join(lines) if found else ""
