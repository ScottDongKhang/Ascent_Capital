"""ascent/data/ingest/capitol_trades.py

Congressional trade disclosure fetcher.
House: https://efts.house.gov/LATEST/search-index
Senate: https://efts.senate.gov/LATEST/search-index

Disclosure lag is 30-45 days; conviction is capped at 0.4 accordingly.
"""
from __future__ import annotations
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SEEN_PATH = Path("data_cache/seen_disclosures.json")
_HEADERS  = {"User-Agent": "AscentCapital research@ascentcap.io"}


def _get_json(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        import requests
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("[CapTrades] GET %s failed: %s", url, e)
        return None


def _load_seen() -> set:
    if SEEN_PATH.exists():
        try:
            return set(json.loads(SEEN_PATH.read_text()))
        except Exception:
            return set()
    return set()


def _save_seen(seen: set) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(list(seen)))


def _parse_house_hit(hit: dict) -> Optional[dict]:
    src = hit.get("_source", {})
    symbol = src.get("ticker", "")
    if not symbol:
        return None
    tx_type = str(src.get("transaction_type_str", "")).lower()
    filed   = src.get("filed_at_date") or src.get("transaction_date") or ""
    try:
        filed_date = date.fromisoformat(str(filed)[:10])
    except Exception:
        filed_date = date.today()
    return {
        "senator":          src.get("representative_name") or src.get("senator_name") or "unknown",
        "symbol":           symbol.upper(),
        "transaction_type": tx_type,
        "amount_range":     src.get("amount") or src.get("amount_range") or "unknown",
        "filed_date":       filed_date,
        "transaction_date": filed_date,
        "source":           "house",
    }


def _dedup_key(d: dict) -> str:
    return f"{d['senator']}|{d['symbol']}|{d['transaction_type']}|{d['filed_date']}"


def fetch_recent_disclosures(lookback_hours: int = 4, universe_symbols: Optional[set] = None) -> list[dict]:
    """
    Fetch recent congressional disclosures from House and Senate eFD APIs.
    Filters to universe_symbols if provided. Deduplicates against seen cache.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today     = date.today().isoformat()

    house_url  = (
        f"https://efts.house.gov/LATEST/search-index"
        f"?q=&dateRange=custom&startdt={yesterday}&enddt={today}&limit=50"
    )
    senate_url = (
        f"https://efts.senate.gov/LATEST/search-index"
        f"?q=&dateRange=custom&startdt={yesterday}&enddt={today}&limit=50"
    )

    seen = _load_seen()
    results: list[dict] = []

    for url in (house_url, senate_url):
        data = _get_json(url)
        if not data:
            continue
        hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
        for hit in hits:
            d = _parse_house_hit(hit)
            if d is None:
                continue
            if universe_symbols and d["symbol"] not in universe_symbols:
                continue
            key = _dedup_key(d)
            if key in seen:
                continue
            seen.add(key)
            results.append(d)

    if results:
        _save_seen(seen)
    return results
