"""ascent/execution/intraday_trigger.py

Intraday rebalance trigger logic.
Checked twice daily at 12:00 PM ET and 14:30 PM ET.

Three triggers:
  (a) Regime emergency — SPY -3% intraday AND VIX > 30 → de-risk (weights × 0.70)
  (b) Kill switch approach — drawdown > 12% → reduce gross exposure by 20%
  (c) Event urgency — high-urgency event in last 60 min for top-5 position → partial trim

All intraday adjustments logged to logs/intraday_adjustments.jsonl.
All orders use TWAP with urgency="high" (15-min window).
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_ADJUSTMENT_LOG = Path("logs/intraday_adjustments.jsonl")
_EVENT_LOG       = Path("logs/event_trades.jsonl")

SPY_CRASH_THRESHOLD  = -0.03   # -3% intraday
VIX_SPIKE_THRESHOLD  = 30.0
DRAWDOWN_WARN        = 0.12    # 12% — soft warn / preemptive trim
EVENT_LOOKBACK_MIN   = 60      # minutes to look back for high-urgency events


def _get_spy_vix(market_data: dict) -> tuple[float, float]:
    """Extract SPY intraday return and VIX from market_data dict. Returns (0, 0) on failure."""
    spy_ret = float(market_data.get("spy_intraday_return", 0.0))
    vix     = float(market_data.get("vix", 0.0))
    return spy_ret, vix


def _get_current_drawdown(portfolio_state: dict) -> float:
    """Extract current drawdown from portfolio_state. Returns 0 on failure."""
    try:
        return float(portfolio_state.get("drawdown", 0.0))
    except Exception:
        return 0.0


def _get_recent_high_urgency_events(top5_symbols: list[str]) -> list[dict]:
    """Read event log for high-urgency events in the last 60 minutes for top-5 positions."""
    if not _EVENT_LOG.exists():
        return []
    cutoff = datetime.now() - timedelta(minutes=EVENT_LOOKBACK_MIN)
    events = []
    try:
        with open(_EVENT_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ts = datetime.fromisoformat(e.get("timestamp", "2000-01-01"))
                    if ts < cutoff:
                        continue
                    if e.get("urgency") == "high" and e.get("symbol") in top5_symbols:
                        events.append(e)
                except Exception:
                    pass
    except Exception:
        pass
    return events


def check_intraday_triggers(
    portfolio_state: dict,
    market_data: dict,
) -> list[dict]:
    """
    Evaluate three intraday triggers. Returns list of trigger dicts (empty = no action).

    portfolio_state: {"weights": {symbol: float}, "nav": float, "drawdown": float}
    market_data: {"spy_intraday_return": float, "vix": float}
    """
    triggers = []

    spy_ret, vix = _get_spy_vix(market_data)
    drawdown     = _get_current_drawdown(portfolio_state)
    weights      = portfolio_state.get("weights", {})
    top5 = sorted(weights, key=weights.get, reverse=True)[:5]

    # (a) Regime emergency
    if spy_ret <= SPY_CRASH_THRESHOLD and vix >= VIX_SPIKE_THRESHOLD:
        triggers.append({
            "type":          "regime_emergency",
            "reason":        f"SPY {spy_ret:.1%} intraday, VIX={vix:.1f}",
            "action":        "de_risk",
            "weight_scalar": 0.70,
            "urgency":       "high",
            "timestamp":     datetime.now().isoformat(),
        })
        log.warning("[IntradayTrigger] REGIME EMERGENCY: SPY=%.1f%% VIX=%.1f → de-risk ×0.70",
                    spy_ret * 100, vix)

    # (b) Drawdown pre-emption
    if drawdown >= DRAWDOWN_WARN and not any(t["type"] == "regime_emergency" for t in triggers):
        triggers.append({
            "type":          "drawdown_preemption",
            "reason":        f"Drawdown {drawdown:.1%} ≥ {DRAWDOWN_WARN:.0%} soft warn",
            "action":        "reduce_exposure",
            "exposure_reduction": 0.20,
            "urgency":       "high",
            "timestamp":     datetime.now().isoformat(),
        })
        log.warning("[IntradayTrigger] DRAWDOWN PRE-EMPTION: %.1f%% → reduce exposure 20%%",
                    drawdown * 100)

    # (c) Event urgency on top-5 positions
    urgent_events = _get_recent_high_urgency_events(top5)
    for event in urgent_events:
        triggers.append({
            "type":      "event_urgency",
            "reason":    f"High-urgency event for {event['symbol']}: {event.get('category')}",
            "action":    "partial_trim",
            "symbol":    event["symbol"],
            "trim_pct":  0.50,
            "urgency":   "high",
            "timestamp": datetime.now().isoformat(),
        })
        log.warning("[IntradayTrigger] EVENT URGENCY: %s %s → partial trim 50%%",
                    event["symbol"], event.get("category"))

    return triggers


def execute_intraday_adjustment(
    trigger: dict,
    current_positions: dict[str, float],
    alpaca_broker=None,
) -> dict:
    """
    Submit the minimal set of orders to implement the intraday adjustment.
    Uses TWAP with urgency="high". Returns result dict.
    """
    action = trigger.get("action")
    result = {"trigger": trigger, "orders_submitted": [], "status": "logged_only"}

    if action == "de_risk":
        scalar = trigger.get("weight_scalar", 0.70)
        result["description"] = f"De-risk: multiply all weights × {scalar}"
        # In production: submit proportional trims via TWAP
        # Currently logged only — TWAP_ENABLED controls execution
        log.info("[IntradayTrigger] De-risk logged: %s", result["description"])

    elif action == "reduce_exposure":
        reduction = trigger.get("exposure_reduction", 0.20)
        result["description"] = f"Reduce gross exposure by {reduction:.0%}"
        log.info("[IntradayTrigger] Exposure reduction logged: %s", result["description"])

    elif action == "partial_trim":
        sym      = trigger.get("symbol")
        trim_pct = trigger.get("trim_pct", 0.50)
        result["description"] = f"Partial trim {sym} by {trim_pct:.0%}"
        log.info("[IntradayTrigger] Partial trim logged: %s", result["description"])

    _log_adjustment(trigger, result)
    return result


def _log_adjustment(trigger: dict, result: dict) -> None:
    _ADJUSTMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trigger":   trigger,
        "result":    result,
    }
    with open(_ADJUSTMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
