"""
ascent/integrations/stocktwits.py

StockTwits public sentiment — pre-fetched, zero hallucination.

Fetches user-labeled Bullish/Bearish tag counts from StockTwits public API
(no auth required). Returns structured sentiment per ticker.

Rate limit: ~200 req/hour free tier. With 15-symbol universe = 15 req/run.
Add 0.2s delay between requests to stay polite.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

import requests

log = logging.getLogger(__name__)

_BASE_URL     = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_TIMEOUT      = 5   # seconds
_DELAY        = 0.2  # seconds between requests
_MIN_LABELED  = 5   # below this → stale signal


def _fetch_symbol(symbol: str) -> dict:
    """Fetch last 30 messages for symbol. Returns raw API dict or {}."""
    url = _BASE_URL.format(symbol=symbol.upper())
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.debug("[StockTwits] %s returned HTTP %d", symbol, resp.status_code)
    except Exception as exc:
        log.debug("[StockTwits] %s fetch failed: %s", symbol, exc)
    return {}


def _parse_messages(messages: list) -> tuple[int, int, int]:
    """Return (bullish_count, bearish_count, total_count)."""
    bullish = bearish = 0
    for msg in messages:
        sentiment = (msg.get("entities") or {}).get("sentiment") or {}
        basic = sentiment.get("basic", "")
        if basic == "Bullish":
            bullish += 1
        elif basic == "Bearish":
            bearish += 1
    return bullish, bearish, len(messages)


def _band(ratio: float) -> str:
    if ratio >= 0.75:  return "strongly_bullish"
    if ratio >= 0.55:  return "bullish"
    if ratio >= 0.45:  return "neutral"
    if ratio >= 0.25:  return "bearish"
    return "strongly_bearish"


def _empty_entry(stale: bool = True) -> dict:
    return {"bullish": 0, "bearish": 0, "n_labeled": 0, "n_total": 0,
            "ratio": 0.5, "band": "neutral", "stale": stale}


def get_sentiment(symbols: list[str], max_messages: int = 30) -> Dict[str, dict]:
    """
    Fetch Bullish/Bearish label counts from StockTwits for each symbol.

    Returns:
        {
          "CAT": {"bullish": 18, "bearish": 5, "n_labeled": 23, "n_total": 30,
                  "ratio": 0.78, "band": "bullish", "stale": False},
          ...
        }
    stale=True when n_labeled < 5 (low signal — do not rely on it).
    """
    results: Dict[str, dict] = {}
    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(_DELAY)
        sym_upper = sym.upper()
        raw = _fetch_symbol(sym_upper)
        messages = raw.get("messages", [])
        bullish, bearish, total = _parse_messages(messages)
        n_labeled = bullish + bearish

        if n_labeled < _MIN_LABELED:
            entry = _empty_entry(stale=True)
            entry.update(bullish=bullish, bearish=bearish,
                         n_labeled=n_labeled, n_total=total)
            results[sym_upper] = entry
            log.debug("[StockTwits] %s: stale (n_labeled=%d)", sym_upper, n_labeled)
            continue

        ratio = bullish / n_labeled
        results[sym_upper] = {
            "bullish":   bullish,
            "bearish":   bearish,
            "n_labeled": n_labeled,
            "n_total":   total,
            "ratio":     round(ratio, 3),
            "band":      _band(ratio),
            "stale":     False,
        }
        log.debug("[StockTwits] %s: %.0f%% bullish (%d/%d labeled)",
                  sym_upper, ratio * 100, bullish, n_labeled)

    return results


def format_sentiment_block(sentiment: Dict[str, dict]) -> str:
    """
    Format sentiment dict as a concise verified block for LLM prompt injection.
    Skips stale entries. Returns empty string if nothing to show.
    """
    lines = []
    for sym, data in sorted(sentiment.items()):
        if data.get("stale"):
            continue
        n    = data["n_labeled"]
        tot  = data["n_total"]
        band = data["band"]
        pct  = round(data["ratio"] * 100)
        lines.append(f"  {sym}: {pct}% bullish ({n} labeled / {tot} total) — {band}")

    if not lines:
        return ""

    return (
        "══ CROWD SENTIMENT (StockTwits, user-labeled) ══\n"
        + "\n".join(lines)
        + "\n══════════════════════════════════════════════\n"
    )
