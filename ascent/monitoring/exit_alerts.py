"""
ascent/monitoring/exit_alerts.py
Position exit alert monitor.

When a held position drops below a configurable threshold intraday,
this module fetches recent price context and asks Haiku:
"Is this noise or signal?"

Writes a one-line verdict to logs/exit_alerts.jsonl.
Advisory only — no auto-execution, no orders.
You review it. System learns what kinds of moves matter.

Uses Haiku — fast, cheap, can run multiple times per day.

Called by:
    eod_runner.py after positions are loaded
    (or standalone for testing)

Output appended to:
    logs/exit_alerts.jsonl
    {"date": "...", "symbol": "MRK", "drop_pct": -0.032,
     "verdict": "SIGNAL", "reason": "one sentence", "action": "review"}
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from ascent.llm.client import generate_structured, HAIKU_MODEL
ALERT_LOG_PATH   = Path("logs/exit_alerts.jsonl")

# Thresholds
INTRADAY_DROP_ALERT  = -0.025   # -2.5% intraday triggers alert
TRAILING_DROP_ALERT  = -0.05    # -5% from recent high triggers alert
MAX_ALERTS_PER_RUN   = 5        # don't spam if market is crashing broadly


def _get_current_prices() -> Dict[str, float]:
    """Pull current intraday prices from Alpaca."""
    try:
        from ascent.execution.alpaca_broker import get_positions
        positions_df = get_positions()
        if positions_df.empty:
            return {}
        if "current_price" in positions_df.columns:
            return dict(zip(positions_df["symbol"], positions_df["current_price"]))
        return {}
    except Exception:
        return {}


def _get_previous_closes(symbols: List[str]) -> Dict[str, float]:
    """Get previous close prices from parquet cache."""
    try:
        import pandas as pd
        cache = Path("data_cache/prices_live.parquet")
        if not cache.exists():
            return {}
        df = pd.read_parquet(cache)
        if df.empty:
            return {}
        # Expect wide format: index=date, columns=symbols
        last_close = df.iloc[-1]
        return {sym: float(last_close[sym]) for sym in symbols
                if sym in last_close.index and not pd.isna(last_close[sym])}
    except Exception:
        return {}


def _already_alerted_today(symbol: str) -> bool:
    """Don't fire more than one alert per symbol per day."""
    if not ALERT_LOG_PATH.exists():
        return False
    today = date.today().isoformat()
    for line in ALERT_LOG_PATH.read_text().splitlines():
        try:
            rec = json.loads(line)
            if rec.get("date") == today and rec.get("symbol") == symbol:
                return True
        except Exception:
            pass
    return False


def _get_recent_holdings_context(symbol: str) -> str:
    """Pull how long we've held this symbol and at what weight."""
    try:
        holdings_path = Path("ascent_holdings_ledger.csv")
        if not holdings_path.exists():
            return ""
        import pandas as pd
        df = pd.read_csv(holdings_path)
        if "symbol" not in df.columns:
            return ""
        sym_rows = df[df["symbol"] == symbol].tail(5)
        if sym_rows.empty:
            return ""
        first_date = sym_rows.iloc[0].get("date", "?")
        avg_weight = sym_rows.get("weight", sym_rows.get("target_weight",
                     sym_rows.iloc[:, -1])).mean()
        return f"Held since {first_date}, avg weight {avg_weight:.1%}"
    except Exception:
        return ""


def _write_alert(alert: dict):
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(alert, default=str) + "\n")


def _classify_move(
    symbol: str,
    drop_pct: float,
    current_price: float,
    prev_close: float,
    holding_context: str,
) -> dict:
    """Ask Haiku: is this drop noise or signal?"""

    user_prompt = f"""
Symbol: {symbol}
Today's move: {drop_pct:+.1%} (from ${prev_close:.2f} to ${current_price:.2f})
Position context: {holding_context or 'no history available'}
Date: {date.today().isoformat()}

Is this move NOISE (normal volatility, hold) or SIGNAL (something has changed, review position)?

Respond in exactly this format:
VERDICT: NOISE or SIGNAL
REASON: one sentence explaining why
ACTION: hold / review / flag_for_exit
"""

    try:
        raw = generate_structured(
            system_prompt=(
                "You are a risk monitor at Ascent Capital. "
                "You assess whether intraday price drops in held positions "
                "represent normal volatility (noise) or a meaningful change "
                "in the investment thesis (signal). "
                "Be concise and specific. Do not overreact to normal market moves."
            ),
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=100,
            temperature=0.2,
        )

        # Parse the structured response
        verdict = "NOISE"
        reason  = "normal market volatility"
        action  = "hold"

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[-1].strip().upper()
                if "SIGNAL" in v:
                    verdict = "SIGNAL"
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[-1].strip()
            elif line.startswith("ACTION:"):
                action = line.split(":", 1)[-1].strip().lower()

        return {"verdict": verdict, "reason": reason, "action": action}

    except Exception as e:
        return {"verdict": "NOISE", "reason": f"classification failed: {e}", "action": "hold"}


def run_exit_alerts(current_weights: Optional[Dict[str, float]] = None) -> List[dict]:
    """
    Check all held positions for significant intraday drops.
    Write alerts to log. Return list of SIGNAL alerts.

    Args:
        current_weights: Optional {symbol: weight} of current holdings.
                         If None, pulled from Alpaca positions.

    Returns:
        List of alert dicts for SIGNAL verdicts only.
    """
    current_prices = _get_current_prices()
    if not current_prices:
        print("[ExitAlerts] No current prices available — skipping")
        return []

    symbols = list(current_prices.keys())
    prev_closes = _get_previous_closes(symbols)

    alerts_fired = 0
    signal_alerts = []

    for sym in symbols:
        if alerts_fired >= MAX_ALERTS_PER_RUN:
            print(f"[ExitAlerts] Max alerts ({MAX_ALERTS_PER_RUN}) reached — stopping")
            break

        if _already_alerted_today(sym):
            continue

        current = current_prices.get(sym)
        prev    = prev_closes.get(sym)
        if not current or not prev or prev == 0:
            continue

        drop_pct = (current - prev) / prev
        if drop_pct > INTRADAY_DROP_ALERT:  # not dropping enough
            continue

        # This position is down significantly — classify it
        holding_ctx = _get_recent_holdings_context(sym)
        result      = _classify_move(sym, drop_pct, current, prev, holding_ctx)

        alert = {
            "date":       date.today().isoformat(),
            "timestamp":  datetime.now().isoformat(),
            "symbol":     sym,
            "drop_pct":   round(drop_pct, 4),
            "current_price": current,
            "prev_close":    prev,
            "verdict":    result["verdict"],
            "reason":     result["reason"],
            "action":     result["action"],
            "weight":     (current_weights or {}).get(sym, 0.0),
        }

        _write_alert(alert)
        alerts_fired += 1

        verdict_tag = "⚠️  SIGNAL" if result["verdict"] == "SIGNAL" else "   noise"
        print(f"[ExitAlerts] {verdict_tag} | {sym} {drop_pct:+.1%} | {result['reason']}")

        if result["verdict"] == "SIGNAL":
            signal_alerts.append(alert)

    if not signal_alerts and alerts_fired > 0:
        print(f"[ExitAlerts] {alerts_fired} drop(s) checked — all noise")

    return signal_alerts


def load_recent_alerts(n: int = 10) -> List[dict]:
    """Load last N alerts from log. Used for dashboard display."""
    if not ALERT_LOG_PATH.exists():
        return []
    lines = ALERT_LOG_PATH.read_text().splitlines()
    alerts = []
    for line in reversed(lines):
        try:
            alerts.append(json.loads(line))
            if len(alerts) >= n:
                break
        except Exception:
            pass
    return alerts


if __name__ == "__main__":
    print("[ExitAlerts] Running manual check...")
    signals = run_exit_alerts()
    if signals:
        print(f"\n{len(signals)} SIGNAL alert(s):")
        for a in signals:
            print(f"  {a['symbol']} {a['drop_pct']:+.1%} — {a['reason']}")
    else:
        print("No signal alerts.")
