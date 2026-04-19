"""
ascent/monitoring/counterfactual_tracker.py

Tracks the counterfactual: what would pure quant have done vs. what debate did?
For every debate session, logs both sets of weights. After 10 days, scores both.

Log format (logs/counterfactual_log.jsonl):
  {"type": "quant_snapshot", "date": "2026-04-29", "weights": {...}}
  {"type": "debate_snapshot", "date": "2026-04-29", "weights": {...}}
  {"type": "outcome", "date": "2026-04-29", "quant_10d": 0.023,
   "debate_10d": 0.031, "ai_added_value": true}
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

LOG_PATH = Path("logs/counterfactual_log.jsonl")
OUTCOME_WINDOW = 10  # calendar days


def snapshot_quant_weights(weights: Dict[str, float], run_date: date) -> None:
    """Call BEFORE debate runs — locks the pure quant portfolio."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type":    "quant_snapshot",
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Counterfactual] Quant snapshot saved: {len(weights)} positions")


def snapshot_debate_weights(weights: Dict[str, float], run_date: date) -> None:
    """Call AFTER debate adjusts weights — locks the AI-augmented portfolio."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type":    "debate_snapshot",
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Counterfactual] Debate snapshot saved: {len(weights)} positions")


def _compute_portfolio_return(
    weights: Dict[str, float],
    start_prices: Dict[str, float],
    end_prices: Dict[str, float],
) -> float:
    """Compute weighted portfolio return given start and end prices."""
    total = 0.0
    weight_used = 0.0
    for sym, w in weights.items():
        if sym in start_prices and sym in end_prices and start_prices[sym] > 0:
            ret = (end_prices[sym] - start_prices[sym]) / start_prices[sym]
            total += w * ret
            weight_used += w
    if weight_used > 0:
        return total / weight_used
    return 0.0


def score_pending_counterfactuals(
    prices_override: Optional[Dict] = None,
    as_of_date: Optional[date] = None,
) -> int:
    """
    Score unscored counterfactuals where OUTCOME_WINDOW days have passed.
    Returns count of verdicts scored.

    Args:
        prices_override: For testing — {symbol: [price_list]}
        as_of_date:      For testing — treat this as today
    """
    if not LOG_PATH.exists():
        return 0

    today = as_of_date or date.today()
    lines = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]

    quant_snaps  = {l["date"]: l["weights"] for l in lines if l["type"] == "quant_snapshot"}
    debate_snaps = {l["date"]: l["weights"] for l in lines if l["type"] == "debate_snapshot"}
    scored_dates = {l["date"] for l in lines if l["type"] == "outcome"}

    scored = 0
    for d_str, quant_w in quant_snaps.items():
        if d_str in scored_dates:
            continue
        if d_str not in debate_snaps:
            continue

        snap_date = date.fromisoformat(d_str)
        if (today - snap_date).days < OUTCOME_WINDOW:
            continue

        try:
            if prices_override:
                all_syms = set(quant_w) | set(debate_snaps[d_str])
                start_p  = {s: prices_override[s][0]  for s in all_syms if s in prices_override}
                end_p    = {s: prices_override[s][-1] for s in all_syms if s in prices_override}
            else:
                from ascent.data.store.parquet import load_parquet
                price_df = load_parquet("prices_live")
                if price_df is None or price_df.empty:
                    continue
                idx = price_df.index
                start_row = price_df.loc[idx <= str(snap_date)].iloc[-1] if len(price_df.loc[idx <= str(snap_date)]) > 0 else None
                end_row   = price_df.loc[idx <= str(today)].iloc[-1] if len(price_df.loc[idx <= str(today)]) > 0 else None
                if start_row is None or end_row is None:
                    continue
                start_p = start_row.dropna().to_dict()
                end_p   = end_row.dropna().to_dict()

            debate_w   = debate_snaps[d_str]
            quant_ret  = _compute_portfolio_return(quant_w,  start_p, end_p)
            debate_ret = _compute_portfolio_return(debate_w, start_p, end_p)
            ai_added   = debate_ret > quant_ret

            outcome = {
                "type":           "outcome",
                "date":           d_str,
                "outcome_date":   today.isoformat(),
                "quant_10d":      round(quant_ret, 6),
                "debate_10d":     round(debate_ret, 6),
                "ai_edge":        round(debate_ret - quant_ret, 6),
                "ai_added_value": ai_added,
            }
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(outcome) + "\n")

            direction = "AI beat quant" if ai_added else "Quant beat AI"
            print(f"[Counterfactual] {d_str}: quant={quant_ret:.2%} debate={debate_ret:.2%} {direction}")
            scored += 1

        except Exception as e:
            print(f"[Counterfactual] Scoring {d_str} failed: {type(e).__name__}: {e}")

    return scored


def get_ai_win_rate(regime_filter: Optional[str] = None) -> Dict:
    """
    Compute AI win rate from all scored counterfactuals.
    Returns {"win_rate": float, "avg_edge": float, "n_samples": int}
    """
    if not LOG_PATH.exists():
        return {"win_rate": 0.0, "avg_edge": 0.0, "n_samples": 0}

    outcomes = []
    for l in LOG_PATH.read_text().splitlines():
        if not l.strip():
            continue
        try:
            rec = json.loads(l)
            if rec.get("type") == "outcome":
                outcomes.append(rec)
        except Exception:
            pass

    if regime_filter:
        outcomes = [o for o in outcomes if o.get("regime") == regime_filter]

    if not outcomes:
        return {"win_rate": 0.0, "avg_edge": 0.0, "n_samples": 0}

    wins     = sum(1 for o in outcomes if o.get("ai_added_value"))
    avg_edge = sum(o.get("ai_edge", 0) for o in outcomes) / len(outcomes)

    return {
        "win_rate":  round(wins / len(outcomes), 3),
        "avg_edge":  round(avg_edge, 4),
        "n_samples": len(outcomes),
    }
