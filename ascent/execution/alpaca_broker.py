"""
ascent/execution/alpaca_broker.py
Alpaca paper trading API wrapper.
"""
import os
import time
import requests
import pandas as pd
from typing import Optional

_ORDER_RETRY = 3
_ORDER_DELAY = 0.4  # seconds between order submissions (Alpaca rate limit ~200 req/min)


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_KEY      = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET   = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_SECRET_KEY", "")


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type":        "application/json",
    }


def get_account() -> dict:
    """Return account info including portfolio value and buying power."""
    r = requests.get(f"{ALPACA_BASE_URL}/account", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_positions() -> pd.DataFrame:
    """
    Return current positions as a DataFrame with columns:
    symbol, qty, market_value, current_price, weight
    Weights are computed as fraction of total portfolio value.
    """
    r = requests.get(f"{ALPACA_BASE_URL}/positions", headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()

    if not data:
        return pd.DataFrame(columns=["symbol", "qty", "market_value", "current_price", "weight"])

    rows = []
    for p in data:
        rows.append({
            "symbol":        p["symbol"],
            "qty":           float(p["qty"]),
            "market_value":  float(p["market_value"]),
            "current_price": float(p["current_price"]),
        })

    df = pd.DataFrame(rows)
    total_mv = df["market_value"].sum()
    df["weight"] = df["market_value"] / total_mv if total_mv > 0 else 0.0
    return df


def get_portfolio_value() -> float:
    """Return total portfolio equity value in dollars."""
    acct = get_account()
    return float(acct["equity"])


def get_portfolio_history(period: str = "6M") -> dict:
    """Return {date_iso: settled_day_return} from Alpaca's published 1D portfolio-history
    bars. These bars settle only after ~17:00 PT — after the daily run — so the series
    ends at the previous trading day. This is the authoritative source for Track B,
    unlike the same-day (equity − last_equity)/last_equity estimate which reads 0.0
    before the account is marked. Returns {} on any failure (caller skips backfill)."""
    from datetime import datetime
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/account/portfolio/history",
            headers=_headers(),
            params={"period": period, "timeframe": "1D", "extended_hours": "false"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        out, prev_eq = {}, None
        for ts, eq in zip(data.get("timestamp", []), data.get("equity", [])):
            if eq is None or eq == 0:
                prev_eq = eq
                continue
            d = datetime.fromtimestamp(ts).date().isoformat()
            if prev_eq and prev_eq > 0:
                out[d] = round(eq / prev_eq - 1, 6)
            prev_eq = eq
        return out
    except Exception as exc:
        print(f"[Alpaca] portfolio history fetch failed: {exc}")
        return {}


def submit_order(symbol: str, qty: float, side: str, order_type: str = "market") -> dict:
    """
    Submit a paper order to Alpaca. Retries up to 3× on rate-limit or timeout.
    side: 'buy' or 'sell'
    qty: number of shares (fractional supported)
    """
    payload = {
        "symbol":        symbol,
        "qty":           round(qty, 6),
        "side":          side,
        "type":          order_type,
        "time_in_force": "day",
    }
    for attempt in range(_ORDER_RETRY):
        try:
            r = requests.post(
                f"{ALPACA_BASE_URL}/orders", headers=_headers(), json=payload, timeout=15
            )
            if r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(_ORDER_DELAY)
            return r.json()
        except requests.Timeout:
            if attempt < _ORDER_RETRY - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"submit_order({symbol}) failed after {_ORDER_RETRY} attempts")


def close_position(symbol: str) -> dict:
    """
    Close an entire position by symbol.
    Use this for full liquidations instead of submit_order(qty=...) — avoids
    403 errors caused by rounding mismatch between estimated and actual share count.
    Uses DELETE /v2/positions/{symbol} which Alpaca handles as "sell all".
    """
    for attempt in range(_ORDER_RETRY):
        try:
            r = requests.delete(
                f"{ALPACA_BASE_URL}/positions/{symbol}", headers=_headers(), timeout=15
            )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(_ORDER_DELAY)
            return r.json()
        except requests.Timeout:
            if attempt < _ORDER_RETRY - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"close_position({symbol}) failed after {_ORDER_RETRY} attempts")


def cancel_all_orders() -> None:
    """Cancel all open orders. Retries on rate-limit."""
    for attempt in range(_ORDER_RETRY):
        try:
            r = requests.delete(f"{ALPACA_BASE_URL}/orders", headers=_headers(), timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return
        except requests.Timeout:
            if attempt < _ORDER_RETRY - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def get_open_orders() -> list:
    """Return list of open orders."""
    r = requests.get(f"{ALPACA_BASE_URL}/orders?status=open", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def is_market_open() -> bool:
    """Check if the market is currently open."""
    r = requests.get(f"{ALPACA_BASE_URL}/clock", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json().get("is_open", False)
