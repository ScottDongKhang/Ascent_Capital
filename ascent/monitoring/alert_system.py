"""
ascent/monitoring/alert_system.py
Drawdown, factor-breach, and sleeve-IC-degradation alerts.

Alerts are written to logs/alerts.jsonl.
Optional push via ntfy.sh: set NTFY_TOPIC in .env.
Deduplication: same alert type per symbol is suppressed for 4 hours.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ALERTS_LOG      = Path("logs/alerts.jsonl")
ALERT_STATE_FILE = Path("logs/alert_state.json")
_DEDUP_HOURS    = 4
_DRAWDOWN_WARN  = 0.05   # 5%
_DRAWDOWN_CRIT  = 0.10   # 10%
_FACTOR_BREACH  = 0.6    # ±0.6σ
_IC_FLOOR       = -0.01  # sleeve IC below this → warning

# "liveness" alerts (pipeline staleness / missed rebalances) come from
# scripts/heartbeat_check.py, which runs OUTSIDE the pipeline it monitors
# on its own launchd interval schedule. That script is stdlib-only and does
# not import this module in the common case (so it can't be broken by the
# same failure it watches) — but when `ascent` IS importable it calls
# send_alert() directly so liveness alerts share the same log/dedup file as
# every other alert type. Nothing here changes existing callers/behavior.
VALID_ALERT_TYPES = {
    "drawdown", "factor_breach", "sleeve_ic_degradation", "liveness", "system_alive",
}


def _load_alert_state() -> Dict[str, str]:
    """Load {alert_key: last_sent_iso} deduplication map."""
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_alert_state(state: Dict[str, str]) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_duplicate(alert_key: str, state: Dict[str, str]) -> bool:
    last_str = state.get(alert_key)
    if not last_str:
        return False
    try:
        last_dt = datetime.fromisoformat(last_str)
        return (datetime.utcnow() - last_dt) < timedelta(hours=_DEDUP_HOURS)
    except Exception:
        return False


def check_alerts(
    portfolio_state: Optional[Dict[str, Any]] = None,
    live_nav: Optional[Any] = None,
    factor_exposures: Optional[Dict[str, float]] = None,
    sleeve_ic: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Evaluate all alert categories. Returns list of new alert dicts.

    Args:
        portfolio_state: dict with optional "drawdown" key
        live_nav:        LiveNAV instance (has compute_intraday_drawdown())
        factor_exposures: {factor_name: exposure_z_score}
        sleeve_ic:        {sleeve_name: trailing_21d_ic}
    """
    alerts = []
    state  = _load_alert_state()
    now    = datetime.utcnow()

    # Drawdown alert
    drawdown = None
    if live_nav is not None:
        try:
            drawdown = abs(live_nav.compute_intraday_drawdown())
        except Exception:
            pass
    elif portfolio_state:
        drawdown = abs(portfolio_state.get("drawdown", 0.0))

    if drawdown is not None:
        if drawdown >= _DRAWDOWN_CRIT:
            key = "drawdown:critical"
            if not _is_duplicate(key, state):
                alerts.append({
                    "type":     "drawdown",
                    "severity": "CRITICAL",
                    "message":  f"Intraday drawdown {drawdown:.1%} exceeds {_DRAWDOWN_CRIT:.0%} threshold",
                    "value":    round(drawdown, 4),
                    "timestamp": now.isoformat(),
                })
                state[key] = now.isoformat()
        elif drawdown >= _DRAWDOWN_WARN:
            key = "drawdown:warning"
            if not _is_duplicate(key, state):
                alerts.append({
                    "type":     "drawdown",
                    "severity": "WARNING",
                    "message":  f"Intraday drawdown {drawdown:.1%} exceeds {_DRAWDOWN_WARN:.0%} threshold",
                    "value":    round(drawdown, 4),
                    "timestamp": now.isoformat(),
                })
                state[key] = now.isoformat()

    # Factor exposure breach
    if factor_exposures:
        for factor, exposure in factor_exposures.items():
            if abs(exposure) > _FACTOR_BREACH:
                key = f"factor:{factor}"
                if not _is_duplicate(key, state):
                    alerts.append({
                        "type":     "factor_breach",
                        "severity": "WARNING",
                        "factor":   factor,
                        "message":  f"Factor '{factor}' exposure {exposure:.2f}σ exceeds ±{_FACTOR_BREACH}σ",
                        "value":    round(exposure, 4),
                        "timestamp": now.isoformat(),
                    })
                    state[key] = now.isoformat()

    # Sleeve IC degradation
    if sleeve_ic:
        for sleeve, ic in sleeve_ic.items():
            if ic is not None and ic < _IC_FLOOR:
                key = f"sleeve_ic:{sleeve}"
                if not _is_duplicate(key, state):
                    alerts.append({
                        "type":     "sleeve_ic_degradation",
                        "severity": "WARNING",
                        "sleeve":   sleeve,
                        "message":  f"Sleeve '{sleeve}' 21-day IC {ic:.4f} below floor {_IC_FLOOR}",
                        "value":    round(ic, 4),
                        "timestamp": now.isoformat(),
                    })
                    state[key] = now.isoformat()

    if alerts:
        _save_alert_state(state)
        for alert in alerts:
            send_alert(alert)

    return alerts


def send_alert(alert_dict: dict) -> None:
    """
    Write alert to logs/alerts.jsonl and log at appropriate level.
    Optional: POST to ntfy.sh if NTFY_TOPIC is set.
    """
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG, "a") as f:
        f.write(json.dumps(alert_dict) + "\n")

    msg = f"[Alert:{alert_dict.get('type','?')}] {alert_dict.get('message','')}"
    if alert_dict.get("severity") == "CRITICAL":
        log.critical(msg)
    else:
        log.warning(msg)

    topic = os.environ.get("NTFY_TOPIC", "")
    if topic:
        try:
            import requests
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=alert_dict.get("message", "Ascent alert"),
                headers={
                    "Title": f"Ascent {alert_dict.get('severity', 'ALERT')}",
                    "Priority": "high" if alert_dict.get("severity") == "CRITICAL" else "default",
                },
                timeout=5,
            )
        except Exception:
            pass


def send_system_alive_ping(last_run: Optional[str] = None, nav: Optional[float] = None,
                            nav_prior: Optional[float] = None) -> dict:
    """Positive daily "system alive" notification.

    An alert channel that only ever fires on failure is indistinguishable
    from a broken one — you cannot tell "no alerts because everything is
    fine" from "no alerts because the alerting path itself died" (which is
    exactly what happened during the 27-day outage: NTFY_TOPIC was unset,
    logs/alerts.jsonl was never created, and nothing told the user).

    This writes a low-severity "INFO" entry to the same alerts.jsonl log
    (so there is always at least one entry per day proving the channel is
    live) and sends it through the same ntfy path as send_alert(). Callers
    are expected to invoke this once per day, independent of whether any
    failure alert fired.
    """
    if nav is not None and nav_prior is not None and nav_prior:
        nav_note = f"NAV {nav:,.2f} ({(nav / nav_prior - 1):+.2%} vs prior)"
    elif nav is not None:
        nav_note = f"NAV {nav:,.2f}"
    else:
        nav_note = "NAV unchanged"
    message = f"System alive. Last run {last_run or 'unknown'}. {nav_note}."
    alert = {
        "type": "system_alive",
        "severity": "INFO",
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    send_alert(alert)
    return alert
