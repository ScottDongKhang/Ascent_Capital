# ascent/strategy/ai_pm_perf_feedback.py
"""
Daily Python-computed learning brief. Zero LLM cost.

Reads decision_log + counterfactual_daily + earned_authority.
Writes data_cache/ai_pm_perf_feedback.json every day after _log_holdings().

The AI PM reads this file before every rebalance-day Phase 2 run.
It is the primary mechanism by which the AI PM learns from its past decisions.

Handles signed weights for short positions (long-short mode).
"""
from __future__ import annotations
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO         = Path(__file__).resolve().parent.parent.parent
FEEDBACK_PATH = _REPO / "data_cache" / "ai_pm_perf_feedback.json"
DECISION_LOG  = _REPO / "logs" / "ai_pm_decision_log.jsonl"
DAILY_LOG     = _REPO / "logs" / "counterfactual_daily.jsonl"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _confidence(n: int) -> str:
    if n < 5:  return "low"
    if n < 15: return "medium"
    return "high"


def _incremental_alpha(ai_w: float, quant_w: float, stock_return: float) -> float:
    """
    (ai_weight - quant_weight) * return — the AI PM's true contribution.
    Works correctly for shorts: if AI PM shorts -4% vs quant -2% and stock falls -8%,
    incremental = (-0.04 - (-0.02)) * (-0.08) = (-0.02) * (-0.08) = +0.0016 (positive alpha).
    """
    return (ai_w - quant_w) * stock_return


def _is_fade(outcome_10d: Optional[float], outcome_21d: Optional[float]) -> bool:
    """Fade: positive at 10d but negative at 21d — short-term momentum mistaken for alpha."""
    if outcome_10d is None or outcome_21d is None:
        return False
    return outcome_10d > 0 and outcome_21d < 0


def _sortino(returns: List[float]) -> float:
    """Annualised Sortino ratio. Returns 0 if n < 5."""
    if len(returns) < 5:
        return 0.0
    mean = sum(returns) / len(returns)
    neg  = [r for r in returns if r < 0]
    if not neg:
        return mean * math.sqrt(252) * 100
    dd = math.sqrt(sum(r ** 2 for r in neg) / len(returns))
    return 0.0 if dd == 0 else mean / dd * math.sqrt(252)


def _fetch_price_return(symbol: str, as_of: str, days_forward: int) -> Optional[float]:
    """Fetch stock return from as_of + days_forward trading days. Returns None on failure."""
    try:
        import yfinance as yf
        start = date.fromisoformat(as_of).isoformat()
        end   = (date.fromisoformat(as_of) + timedelta(days=days_forward + 10)).isoformat()
        df    = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].squeeze().dropna()
        if len(closes) < 2:
            return None
        target_idx = min(days_forward, len(closes) - 1)
        return float((closes.iloc[target_idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as e:
        log.debug("[PerfFeedback] Price fetch %s: %s", symbol, e)
        return None


def _load_decisions() -> List[dict]:
    if not DECISION_LOG.exists():
        return []
    rows = []
    for line in DECISION_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return sorted(rows, key=lambda x: x.get("date", ""))


def _load_daily_records() -> List[dict]:
    if not DAILY_LOG.exists():
        return []
    rows = []
    for line in DAILY_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _load_authority_state() -> dict:
    from ascent.strategy.earned_authority import get_state
    return get_state()


# ── Score pending decisions ────────────────────────────────────────────────────

def _score_decisions(decisions: List[dict]) -> List[dict]:
    """Score each override decision at 5d/10d/21d/63d horizons where enough time has passed."""
    today = date.today()
    scored = []

    for dec in decisions:
        try:
            dec_date = date.fromisoformat(dec["date"])
        except Exception:
            continue
        days_since = (today - dec_date).days

        for ov in dec.get("overrides_applied", []):
            sym    = ov.get("symbol", "")
            ai_w   = ov.get("ai_w", 0.0)
            quant_w = ov.get("quant_w", 0.0)
            ov_type = ov.get("type", "amplify")

            record: dict = {
                "date":        dec["date"],
                "symbol":      sym,
                "type":        ov_type,
                "ai_w":        ai_w,
                "quant_w":     quant_w,
                "is_short":    ai_w < 0,
                "outcome_5d":  None,
                "outcome_10d": None,
                "outcome_21d": None,
                "outcome_63d": None,
                "verdict":     None,
                "fade":        False,
                "early":       False,
            }

            for horizon, min_days, key in [
                (5,  5,  "outcome_5d"),
                (10, 10, "outcome_10d"),
                (21, 21, "outcome_21d"),
                (63, 63, "outcome_63d"),
            ]:
                if days_since >= min_days:
                    raw = _fetch_price_return(sym, dec["date"], horizon)
                    if raw is not None:
                        record[key] = round(_incremental_alpha(ai_w, quant_w, raw), 6)
                    else:
                        record[key] = 0.0  # orphaned (halted/delisted) → score 0, counts toward n_evaluated

            # Classify verdict
            r10 = record["outcome_10d"]
            r21 = record["outcome_21d"]
            r63 = record["outcome_63d"]

            if r10 is not None:
                if _is_fade(r10, r21):
                    record["verdict"] = "fade"
                    record["fade"]    = True
                elif r10 < 0 and r63 is not None and r63 > 0:
                    record["verdict"] = "early"
                    record["early"]   = True
                elif r10 >= 0:
                    record["verdict"] = "win"
                else:
                    record["verdict"] = "miss"

            scored.append(record)

    return scored


# ── Main compute ───────────────────────────────────────────────────────────────

def compute_feedback() -> dict:
    """Compute and write the daily learning brief. Returns the feedback dict."""
    today     = date.today()
    state     = _load_authority_state()
    level     = state.get("level", 0)
    decisions = _load_decisions()
    daily     = _load_daily_records()

    scored    = _score_decisions(decisions)
    evaluated = [s for s in scored if s.get("outcome_10d") is not None]
    pending   = [s for s in scored if s.get("outcome_10d") is None]

    # ── Sortino on Track D vs Track A★ ──────────────────────────────────────
    d_rets  = [r.get("track_d_return", 0.0)     for r in daily[-21:]]
    as_rets = [r.get("track_astar_return", 0.0) for r in daily[-21:]]
    sortino_d  = _sortino(d_rets)
    sortino_as = _sortino(as_rets)
    n_days = len(d_rets)

    # ── Hit rate, profit factor, fade rate ──────────────────────────────────
    wins   = [s for s in evaluated if s.get("verdict") == "win"]
    misses = [s for s in evaluated if s.get("verdict") == "miss"]
    fades  = [s for s in evaluated if s.get("verdict") == "fade"]

    hit_rate      = len(wins) / len(evaluated) if evaluated else 0.0
    gross_wins    = sum(abs(s["outcome_10d"]) for s in wins)
    gross_losses  = sum(abs(s["outcome_10d"]) for s in misses)
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (1.5 if gross_wins > 0 else 1.0)
    fade_rate     = len(fades) / len(evaluated) if evaluated else 0.0

    # ── Override type breakdown ─────────────────────────────────────────────
    def _type_stats(ov_type: str):
        sub = [s for s in evaluated if s.get("type", "").lower() == ov_type and s.get("outcome_10d") is not None]
        avg = sum(s["outcome_10d"] for s in sub) / len(sub) if sub else 0.0
        return round(avg, 6), len(sub)

    amp_avg, amp_n  = _type_stats("amplify")
    red_avg, red_n  = _type_stats("reduce")
    new_avg, new_n  = _type_stats("new")
    sht_avg, sht_n  = _type_stats("short")

    best  = max(evaluated, key=lambda x: x.get("outcome_10d", 0), default=None)
    worst = min(evaluated, key=lambda x: x.get("outcome_10d", 0), default=None)

    # ── Promotion gates ──────────────────────────────────────────────────────
    _PROMO_CFG = {
        (1, 2): {"sortino_edge": 0.20, "hit_rate": 0.52, "profit_factor": 1.2, "min_decisions": 5},
        (2, 3): {"sortino_edge": 0.30, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 8},
        (3, 4): {"sortino_edge": 0.40, "hit_rate": 0.55, "profit_factor": 1.3, "min_decisions": 10},
        (4, 5): {"sortino_edge": 0.50, "hit_rate": 0.58, "profit_factor": 1.4, "min_decisions": 15},
    }
    cfg  = _PROMO_CFG.get((level, level + 1), {})
    edge = sortino_d - sortino_as

    promotion_gates: dict = {}
    if cfg:
        promotion_gates = {
            "sortino_edge":  {"pass": edge > cfg["sortino_edge"],             "value": round(edge, 3),         "threshold": cfg["sortino_edge"]},
            "hit_rate":      {"pass": hit_rate >= cfg["hit_rate"],            "value": round(hit_rate, 3),     "threshold": cfg["hit_rate"]},
            "profit_factor": {"pass": profit_factor > cfg["profit_factor"],   "value": round(profit_factor, 3),"threshold": cfg["profit_factor"]},
            "min_decisions": {"pass": len(evaluated) >= cfg["min_decisions"], "value": len(evaluated),         "threshold": cfg["min_decisions"]},
            "fade_rate":     {"pass": fade_rate <= 0.30,                      "value": round(fade_rate, 3),    "threshold": 0.30},
            "regime_gate":   {"pass": True,  "value": "not yet evaluated"},
            "cooldown":      {"pass": not state.get("in_cooldown", False), "value": "active" if state.get("in_cooldown") else "clear"},
        }

    # ── Cooldown info ────────────────────────────────────────────────────────
    cooldown_until = state.get("cooldown_until")
    cooldown_days_remaining = 0
    if cooldown_until:
        try:
            delta = (date.fromisoformat(cooldown_until) - today).days
            cooldown_days_remaining = max(0, delta)
        except Exception:
            pass

    days_stuck = state.get("days_stuck", 0)

    feedback = {
        "as_of":                     today.isoformat(),
        "level":                     level,
        "title":                     state.get("title", "Shadow"),
        "ai_weight":                 state.get("ai_weight", 0.0),
        "days_at_level":             state.get("days_at_level", 0),
        "in_cooldown":               state.get("in_cooldown", False),
        "cooldown_days_remaining":   cooldown_days_remaining,
        "days_stuck":                days_stuck,
        "stuck_alert":               days_stuck >= 63,
        "sortino_21d_d":             round(sortino_d, 3),
        "sortino_21d_astar":         round(sortino_as, 3),
        "sortino_edge":              round(edge, 3),
        "sortino_n_days":            n_days,
        "hit_rate_21d":              round(hit_rate, 3),
        "profit_factor":             round(profit_factor, 3),
        "fade_rate":                 round(fade_rate, 3),
        "override_win_rate":         round(hit_rate, 3),
        "amplify_avg_alpha_10d":     amp_avg,
        "amplify_n":                 amp_n,
        "amplify_confidence":        _confidence(amp_n),
        "reduce_avg_alpha_10d":      red_avg,
        "reduce_n":                  red_n,
        "reduce_ban_active":         (red_n >= 5 and red_avg < 0),
        "short_avg_alpha_10d":       sht_avg,
        "short_n":                   sht_n,
        "short_confidence":          _confidence(sht_n),
        "new_position_avg_alpha_10d": new_avg,
        "new_position_n":            new_n,
        "n_decisions_evaluated":     len(evaluated),
        "n_decisions_pending":       len(pending),
        "last_5_decisions":          scored[-5:],
        "best_call_10d":             {"symbol": best["symbol"], "type": best["type"], "alpha": best["outcome_10d"], "n_basis": len(evaluated)} if best else None,
        "worst_call_10d":            {"symbol": worst["symbol"], "type": worst["type"], "alpha": worst["outcome_10d"], "n_basis": len(evaluated)} if worst else None,
        "promotion_gates":           promotion_gates,
        "phase1_accuracy":           {"regime_accuracy_rate": None, "sleeve_prior_value": None, "note": "scored after 10 trading days"},
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(feedback, indent=2))
    log.info("[PerfFeedback] Written: Level %d, edge %.3f, n_eval=%d, stuck=%s",
             level, edge, len(evaluated), feedback["stuck_alert"])

    if feedback["stuck_alert"]:
        print(f"[AIPMAuthority] WARNING: AI PM at Level {level} for 63+ days — review promotion gates.")

    return feedback
