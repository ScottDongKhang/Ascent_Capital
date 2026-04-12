#!/usr/bin/env python3
"""
Quick API key check for Polygon, FRED, and Alpaca.
Run from project root: .venv/bin/python check_apis.py
Does not print or log any key values.
"""
from __future__ import annotations

import os
import sys

# Load .env before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Add project to path so we can import ascent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests


def check_polygon(key: str) -> tuple[bool, str]:
    if not key or key.strip() == "" or key == "your_key":
        return False, "key not set or placeholder"
    url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05"
    r = requests.get(url, params={"apiKey": key, "limit": 5}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        n = len(data.get("results") or [])
        return True, f"OK ({n} bars)"
    if r.status_code == 401:
        return False, "401 Unauthorized (invalid or inactive key)"
    return False, f"{r.status_code} {r.reason}"


def check_fred(key: str) -> tuple[bool, str]:
    if not key or key.strip() == "" or key == "your_key":
        return False, "key not set or placeholder"
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(
        url,
        params={
            "series_id": "DFF",
            "api_key": key,
            "file_type": "json",
            "observation_start": "2024-01-01",
            "observation_end": "2024-01-05",
        },
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        n = len(data.get("observations") or [])
        return True, f"OK ({n} observations)"
    if r.status_code == 400 and "bad api_key" in (r.text or "").lower():
        return False, "invalid key"
    if r.status_code == 403:
        return False, "403 Forbidden (invalid key or quota)"
    return False, f"{r.status_code} {r.reason}"


def check_alpaca(key_id: str, secret: str, base_url: str) -> tuple[bool, str]:
    if not key_id or not secret:
        return False, "key or secret not set"
    if key_id == "your_key" or secret == "your_secret":
        return False, "placeholder key/secret"
    url = (base_url or "https://paper-api.alpaca.markets").rstrip("/") + "/v2/account"
    r = requests.get(
        url,
        headers={
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        },
        timeout=15,
    )
    if r.status_code == 200:
        return True, "OK (paper account)"
    if r.status_code == 401:
        return False, "401 Unauthorized (invalid key/secret)"
    return False, f"{r.status_code} {r.reason}"


def main():
    from ascent.config.settings import get_config

    cfg = get_config()
    k = cfg.keys

    print("API key check (no keys printed)\n" + "-" * 50)

    ok, msg = check_polygon(k.polygon)
    print(f"Polygon:  {'OK' if ok else 'FAIL'} — {msg}")

    ok, msg = check_fred(k.fred)
    print(f"FRED:     {'OK' if ok else 'FAIL'} — {msg}")

    ok, msg = check_alpaca(k.alpaca_key, k.alpaca_secret, getattr(cfg, "alpaca_base_url", None) or os.getenv("ALPACA_BASE_URL", ""))
    print(f"Alpaca:   {'OK' if ok else 'FAIL'} — {msg}")

    print("-" * 50)


if __name__ == "__main__":
    main()
