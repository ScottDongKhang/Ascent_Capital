"""
ascent/data/streaming/alpaca_stream.py
Alpaca WebSocket 1-minute bar consumer.

Writes bars to TimescaleDB prices_1m and maintains an in-memory
last-price cache for intraday NAV computation.

Stream URL: wss://stream.data.alpaca.markets/v2/iex  (free plan)
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# In-memory last-price cache: {symbol: last_close}
_price_cache: Dict[str, float] = {}
_cache_lock = threading.Lock()

_stream_thread: Optional[threading.Thread] = None
_reconnect_delays = [1, 2, 4, 8, 16, 30, 60]


def get_live_price(symbol: str) -> Optional[float]:
    """
    Return last streamed close price for symbol.
    Returns None if not yet received (callers fall back to daily close).
    """
    with _cache_lock:
        return _price_cache.get(symbol)


def _update_cache(symbol: str, price: float) -> None:
    with _cache_lock:
        _price_cache[symbol] = price


def default_bar_callback(bar: dict) -> None:
    """
    Called for every 1-minute bar received.
    Updates in-memory cache and writes to TimescaleDB prices_1m.
    """
    symbol = bar.get("S") or bar.get("symbol", "")
    close  = float(bar.get("c") or bar.get("close", 0))
    if not symbol or close == 0:
        return

    _update_cache(symbol, close)

    try:
        import pandas as pd
        from ascent.data.store.timescale import write_prices

        ts = bar.get("t") or bar.get("timestamp") or datetime.utcnow().isoformat()
        df = pd.DataFrame([{
            "time":   pd.Timestamp(ts, tz="UTC"),
            "symbol": symbol,
            "open":   float(bar.get("o") or bar.get("open", close)),
            "high":   float(bar.get("h") or bar.get("high", close)),
            "low":    float(bar.get("l") or bar.get("low", close)),
            "close":  close,
            "volume": int(bar.get("v") or bar.get("volume", 0)),
        }])
        write_prices(df, table="prices_1m")
    except Exception as exc:
        log.debug("[Stream] TimescaleDB write failed for %s: %s", symbol, exc)


def _run_stream(symbols: List[str], on_bar: Callable) -> None:
    """
    Internal loop: connect, subscribe, stream. Reconnects on failure.
    """
    api_key    = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not api_secret:
        log.warning("[Stream] ALPACA_KEY / ALPACA_SECRET not set — stream disabled")
        return

    retry_idx = 0
    while True:
        try:
            import websocket
            import json

            ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            ws_app = websocket.WebSocketApp(
                ws_url,
                on_open=lambda ws: _on_open(ws, api_key, api_secret, symbols),
                on_message=lambda ws, msg: _on_message(ws, msg, on_bar),
                on_error=lambda ws, err: log.warning("[Stream] WebSocket error: %s", err),
                on_close=lambda ws, c, m: log.info("[Stream] WebSocket closed: %s %s", c, m),
            )
            log.info("[Stream] Connecting to Alpaca WebSocket...")
            ws_app.run_forever(ping_interval=30, ping_timeout=10)

        except ImportError:
            log.warning("[Stream] websocket-client not installed — live streaming disabled. pip install websocket-client")
            return
        except Exception as exc:
            log.warning("[Stream] Connection error: %s", exc)

        delay = _reconnect_delays[min(retry_idx, len(_reconnect_delays) - 1)]
        log.info("[Stream] Reconnecting in %ss...", delay)
        time.sleep(delay)
        retry_idx += 1


def _on_open(ws, api_key: str, api_secret: str, symbols: List[str]) -> None:
    import json
    ws.send(json.dumps({"action": "auth", "key": api_key, "secret": api_secret}))
    # Alpaca sends auth confirmation; subscribe on next on_message
    ws._symbols_to_subscribe = symbols
    ws._subscribed = False


def _on_message(ws, msg: str, on_bar: Callable) -> None:
    import json
    try:
        data = json.loads(msg)
    except Exception:
        return

    if not isinstance(data, list):
        data = [data]

    for item in data:
        t = item.get("T") or item.get("type", "")
        if t == "success" and item.get("msg") == "authenticated":
            syms = getattr(ws, "_symbols_to_subscribe", [])
            if syms and not getattr(ws, "_subscribed", False):
                ws.send(json.dumps({
                    "action": "subscribe",
                    "bars": syms,
                }))
                ws._subscribed = True
                log.info("[Stream] Subscribed to %d symbols", len(syms))
        elif t == "b":
            on_bar(item)


def start_stream(
    symbols: List[str],
    on_bar_callback: Optional[Callable] = None,
) -> threading.Thread:
    """
    Start the WebSocket streaming thread (daemon).
    Returns the thread object.
    """
    global _stream_thread

    if _stream_thread is not None and _stream_thread.is_alive():
        log.info("[Stream] Stream thread already running.")
        return _stream_thread

    callback = on_bar_callback or default_bar_callback
    t = threading.Thread(
        target=_run_stream,
        args=(symbols, callback),
        name="alpaca-stream",
        daemon=True,
    )
    t.start()
    _stream_thread = t
    log.info("[Stream] Streaming thread started (%d symbols)", len(symbols))
    return t
