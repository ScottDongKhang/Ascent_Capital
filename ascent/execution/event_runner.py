"""ascent/execution/event_runner.py

Event trade sizing, limit order submission, and IC tracking.

EVENT_TRADING_ENABLED = False by default — must be manually set to True
after 30-day paper-mode validation period.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ⚠ Kill switch — must be manually enabled after paper-mode validation
EVENT_TRADING_ENABLED = False

EVENT_LOG = Path("logs/event_trades.jsonl")
MAX_EVENT_PCT = 0.005   # 0.5% NAV hard cap per event trade
MIN_TRADE_USD = 500.0   # minimum trade size (fractional-share threshold)

_URGENCY_MULTIPLIER = {"high": 1.0, "medium": 0.6, "low": 0.3}


def compute_event_size(
    conviction: float,
    urgency: str,
    nav: float,
    max_event_pct: float = MAX_EVENT_PCT,
) -> float:
    """
    Dollar size for an event trade.
    Base = conviction × max_event_pct × NAV.
    Urgency multiplier applied. Hard cap at max_event_pct × NAV. Min $500.
    Returns 0.0 if below minimum (skip trade).
    """
    mult  = _URGENCY_MULTIPLIER.get(str(urgency).lower(), 0.3)
    size  = float(conviction) * mult * max_event_pct * float(nav)
    size  = min(size, max_event_pct * float(nav))
    if size < MIN_TRADE_USD:
        return 0.0
    return round(size, 2)


def submit_event_trade(
    symbol: str,
    direction: str,
    size_dollars: float,
    rationale: str,
    event_id: str,
    nav: float = 100_000.0,
) -> dict:
    """
    Submit a limit order for an event trade.
    When EVENT_TRADING_ENABLED=False, logs the decision but submits no order.
    Returns fill dict or {"status": "disabled"} / {"status": "expired"}.
    """
    if not EVENT_TRADING_ENABLED:
        log.info("[EventRunner] EVENT_TRADING_ENABLED=False — logging decision only: %s %s $%.0f",
                 direction, symbol, size_dollars)
        return {"status": "disabled", "symbol": symbol, "size_dollars": size_dollars}

    # Large event trade (> 1% NAV) goes through approval gate with 10-min timeout
    if size_dollars > nav * 0.01:
        log.info("[EventRunner] %s: size $%.0f > 1%% NAV — routing to approval gate", symbol, size_dollars)
        return _submit_via_approval(symbol, direction, size_dollars, rationale, event_id)

    return _submit_limit_order(symbol, direction, size_dollars, rationale, event_id)


def _submit_limit_order(symbol: str, direction: str, size_dollars: float, rationale: str, event_id: str) -> dict:
    """Place a 5-minute limit order at bid+1tick (buy) or ask-1tick (sell)."""
    try:
        import alpaca_trade_api as tradeapi
        from ascent.config.settings import APIKeys
        keys = APIKeys.from_env()
        api  = tradeapi.REST(keys.alpaca_key, keys.alpaca_secret, base_url="https://paper-api.alpaca.markets")

        quote = api.get_latest_quote(symbol)
        if direction == "buy":
            limit_price = round(float(quote.ap) + 0.01, 2)   # ask + 1 tick
        else:
            limit_price = round(float(quote.bp) - 0.01, 2)   # bid - 1 tick

        shares = max(1, int(size_dollars / limit_price))
        side   = "buy" if direction == "buy" else "sell"

        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side=side,
            type="limit",
            time_in_force="day",
            limit_price=str(limit_price),
            client_order_id=event_id[:36],
        )
        return {
            "status":      "submitted",
            "order_id":    order.id,
            "symbol":      symbol,
            "shares":      shares,
            "limit_price": limit_price,
            "fill_price":  None,
        }
    except Exception as e:
        log.warning("[EventRunner] limit order failed for %s: %s", symbol, e)
        return {"status": "error", "symbol": symbol, "error": str(e)}


def _submit_via_approval(symbol: str, direction: str, size_dollars: float, rationale: str, event_id: str) -> dict:
    """Route large event trade through approval gate (10-min timeout)."""
    try:
        approval_path = Path("execution/pending_approvals.json")
        existing: list = []
        if approval_path.exists():
            try:
                existing = json.loads(approval_path.read_text())
            except Exception:
                existing = []
        existing.append({
            "event_id":    event_id,
            "symbol":      symbol,
            "direction":   direction,
            "size_dollars": size_dollars,
            "rationale":   rationale,
            "type":        "event_trade",
            "timeout_min": 10,
            "created_at":  datetime.now().isoformat(),
        })
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(json.dumps(existing))
        return {"status": "pending_approval", "symbol": symbol, "size_dollars": size_dollars}
    except Exception as e:
        log.warning("[EventRunner] approval gate failed: %s", e)
        return {"status": "error", "symbol": symbol, "error": str(e)}


def log_event_trade(
    event_id: str,
    classification: dict,
    size: float,
    fill: dict,
    nav: float,
    source: str = "edgar",
) -> None:
    """Append event trade decision to logs/event_trades.jsonl."""
    entry = {
        "event_id":      event_id,
        "timestamp":     datetime.now().isoformat(),
        "source":        source,
        "symbol":        classification.get("symbol", ""),
        "direction":     classification.get("direction", "neutral"),
        "conviction":    classification.get("conviction", 0.0),
        "urgency":       classification.get("urgency", "low"),
        "category":      classification.get("category", "unknown"),
        "rationale":     classification.get("rationale", ""),
        "size_dollars":  size,
        "fill_price":    fill.get("fill_price"),
        "nav_at_decision": nav,
        "status":        fill.get("status", "unknown"),
    }
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def compute_event_ic(lookback_days: int = 20) -> dict:
    """
    Compute Spearman IC between event trade direction and forward returns.
    Reads logs/event_trades.jsonl; fetches current prices via yfinance.
    Returns {"ic_5d", "ic_10d", "ic_20d", "n_trades"}.
    """
    if not EVENT_LOG.exists():
        return {"ic_5d": None, "ic_10d": None, "ic_20d": None, "n_trades": 0}

    trades: list[dict] = []
    cutoff = date.today()
    with open(EVENT_LOG) as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("direction") in ("buy", "sell") and t.get("symbol"):
                    trades.append(t)
            except Exception:
                pass

    if not trades:
        return {"ic_5d": None, "ic_10d": None, "ic_20d": None, "n_trades": 0}

    symbols = list({t["symbol"] for t in trades})
    try:
        import yfinance as yf
        from scipy.stats import spearmanr
        prices = yf.download(symbols, period=f"{lookback_days + 5}d", auto_adjust=True, progress=False)
        if prices.empty:
            return {"ic_5d": None, "ic_10d": None, "ic_20d": None, "n_trades": len(trades)}

        if isinstance(prices.columns, __import__("pandas").MultiIndex):
            closes = prices["Close"]
        else:
            closes = prices

        results: dict = {"ic_5d": None, "ic_10d": None, "ic_20d": None, "n_trades": len(trades)}
        for horizon_days, key in [(5, "ic_5d"), (10, "ic_10d"), (20, "ic_20d")]:
            directions, fwd_rets = [], []
            for t in trades:
                sym = t["symbol"]
                if sym not in closes.columns:
                    continue
                try:
                    ts   = __import__("pandas").Timestamp(t["timestamp"][:10])
                    if ts not in closes.index:
                        idx = closes.index.searchsorted(ts)
                        if idx >= len(closes.index):
                            continue
                        ts = closes.index[idx]
                    future_idx = closes.index.searchsorted(ts) + horizon_days
                    if future_idx >= len(closes):
                        continue
                    entry_px  = float(closes[sym].loc[ts])
                    future_px = float(closes[sym].iloc[future_idx])
                    fwd_ret   = (future_px - entry_px) / entry_px
                    direction_sign = 1 if t["direction"] == "buy" else -1
                    directions.append(direction_sign)
                    fwd_rets.append(fwd_ret)
                except Exception:
                    pass
            if len(directions) >= 5:
                ic, _ = spearmanr(directions, fwd_rets)
                results[key] = round(float(ic), 4)
        return results
    except Exception as e:
        log.warning("[EventRunner] IC computation failed: %s", e)
        return {"ic_5d": None, "ic_10d": None, "ic_20d": None, "n_trades": len(trades)}


def make_event_id() -> str:
    return str(uuid.uuid4())
