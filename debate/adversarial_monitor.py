"""
debate/adversarial_monitor.py

Non-rebalance day lightweight scan for early rebalance triggers.

Checks current portfolio for three classes of developing risk:
  1. Earnings in <7 days on a held position (binary event — Adversarial Intelligence cannot size-correct post-event)
  2. SEC tone deteriorating significantly since last scan (risk_trend < -0.5)
  3. Position moved >20% since entry (event momentum triggering resize alert)

Output: writes to data_cache/adversarial_monitor.json (checked by daily_intelligence.py)
Prints console alerts for conditions requiring attention.
Does NOT make weight changes — flags only. All changes happen on rebalance days.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

MONITOR_PATH = Path("data_cache/adversarial_monitor.json")
ALERT_THRESHOLD_EARNINGS_DAYS = 7
ALERT_THRESHOLD_SEC_RISK     = -0.50
ALERT_THRESHOLD_PRICE_MOVE   = 0.20


def _load_portfolio_weights() -> dict:
    p = Path("execution/merged_weights.json")
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        w = data.get("weights", data)
        return {s: float(v) for s, v in w.items()
                if isinstance(v, (int, float)) and v > 0}
    except Exception:
        return {}


def _check_earnings_risk(symbols: list) -> list:
    """Identify portfolio positions with earnings in <7 days."""
    alerts = []
    try:
        from ascent.reporting.catalyst_scanner import scan_catalysts
        result = scan_catalysts(symbols)
        for event in result.get("upcoming_events", []):
            days = event.get("days_away", 99)
            if days <= ALERT_THRESHOLD_EARNINGS_DAYS:
                alerts.append({
                    "type":       "earnings_risk",
                    "symbol":     event.get("symbol", ""),
                    "event":      event.get("event_type", "earnings"),
                    "days_away":  days,
                    "message":    f"{event.get('symbol', '')} has {event.get('event_type', 'event')} in {days} days — binary risk on held position",
                })
    except Exception as e:
        pass
    return alerts


def _check_sec_deterioration(symbols: list) -> list:
    """Identify positions with worsening SEC signals since last report."""
    alerts = []
    sec_path = Path("data_cache/altdata_sec_detail.json")
    if not sec_path.exists():
        return []
    try:
        detail = json.loads(sec_path.read_text())
        for sym in symbols:
            if sym not in detail:
                continue
            risk_trend = detail[sym].get("risk_trend") or 0
            tone       = detail[sym].get("tone_score") or 0
            if risk_trend < ALERT_THRESHOLD_SEC_RISK or tone < -0.4:
                alerts.append({
                    "type":       "sec_deterioration",
                    "symbol":     sym,
                    "risk_trend": round(risk_trend, 3),
                    "tone":       round(tone, 3),
                    "message":    f"{sym} SEC signals deteriorating (risk_trend={risk_trend:.2f}, tone={tone:.2f})",
                })
    except Exception:
        pass
    return alerts


def _check_price_moves(weights: dict) -> list:
    """Identify positions up >20% since portfolio construction (event momentum risk)."""
    alerts = []
    symbols = list(weights.keys())
    if not symbols:
        return []

    try:
        from ascent.data.store.parquet import load_parquet
        import pandas as pd

        prices_long = load_parquet("prices_live")
        prices_wide = prices_long.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )
        prices_wide.index = pd.to_datetime(prices_wide.index)

        # Use last 21 trading days as proxy for "since last rebalance"
        cutoff = pd.Timestamp(date.today() - timedelta(days=30))
        recent = prices_wide[prices_wide.index >= cutoff]

        for sym in symbols:
            if sym not in recent.columns:
                continue
            col = recent[sym].dropna()
            if len(col) < 5:
                continue
            move = (col.iloc[-1] / col.iloc[0]) - 1
            if abs(move) > ALERT_THRESHOLD_PRICE_MOVE:
                direction = "up" if move > 0 else "down"
                alerts.append({
                    "type":    "large_price_move",
                    "symbol":  sym,
                    "move":    round(float(move), 4),
                    "message": f"{sym} moved {move:+.1%} in last 30d — event momentum, check adversarial sizing on next rebalance",
                })
    except Exception:
        pass
    return alerts


def run_adversarial_monitor() -> dict:
    """
    Run the non-rebalance day scan. Returns a summary dict.
    Writes to data_cache/adversarial_monitor.json.
    """
    today   = date.today()
    weights = _load_portfolio_weights()
    symbols = list(weights.keys())

    if not symbols:
        result = {"date": today.isoformat(), "alerts": [], "n_alerts": 0}
        MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        MONITOR_PATH.write_text(json.dumps(result, indent=2))
        return result

    all_alerts = []

    earnings_alerts = _check_earnings_risk(symbols)
    all_alerts.extend(earnings_alerts)

    sec_alerts = _check_sec_deterioration(symbols)
    all_alerts.extend(sec_alerts)

    price_alerts = _check_price_moves(weights)
    all_alerts.extend(price_alerts)

    result = {
        "date":     today.isoformat(),
        "alerts":   all_alerts,
        "n_alerts": len(all_alerts),
        "summary":  {
            "earnings_risk":   len(earnings_alerts),
            "sec_deterioration": len(sec_alerts),
            "large_price_moves": len(price_alerts),
        },
    }

    import os, tempfile
    MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=MONITOR_PATH.parent, suffix=".tmp"))
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, MONITOR_PATH)

    if all_alerts:
        print(f"[AdvMonitor] {len(all_alerts)} alerts:")
        for alert in all_alerts:
            print(f"  ⚠ {alert['message']}")
    else:
        print(f"[AdvMonitor] Clean — no adversarial alerts today")

    return result
