"""
Exa news search integration — fetch live news headlines for equity symbols.
Uses the Exa neural search API (https://exa.ai). Requires EXA_API_KEY env var.
Free tier: 1,000 searches/month. Ascent uses ~60/month at current scale.
"""
from __future__ import annotations

import os
import time

import logging

import requests

logger = logging.getLogger(__name__)


def fetch_news(
    symbols: list[str],
    max_per_symbol: int = 2,
) -> dict[str, list[str]]:
    """
    Fetch live news summaries for each symbol via Exa search.

    Returns {symbol: [summary, ...]} — empty list for any symbol that fails.
    Delays 0.2s between requests to stay within free tier rate limits.
    Never raises — failures are logged and return empty list for that symbol.
    """
    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        logger.warning("[ExaNews] EXA_API_KEY not set — skipping news fetch")
        return {sym: [] for sym in symbols}

    headers = {"x-api-key": api_key, "content-type": "application/json"}
    results: dict[str, list[str]] = {}

    for sym in symbols:
        payload = {
            "query": f"{sym} stock news catalyst today",
            "type": "auto",
            "numResults": max_per_symbol,
            "contents": {
                "summary": {
                    "schema": {
                        "type": "object",
                        "required": ["answer"],
                        "additionalProperties": False,
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Key news headline or catalyst from this article",
                            }
                        },
                    }
                }
            },
        }
        try:
            resp = requests.post(
                "https://api.exa.ai/search",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            summaries = [
                r["summary"]["answer"]
                for r in data.get("results", [])
                if r.get("summary", {}).get("answer")
            ]
            results[sym] = summaries[:max_per_symbol]
        except Exception as exc:
            logger.warning("[ExaNews] Failed for %s: %s", sym, exc)
            results[sym] = []

        time.sleep(0.2)

    return results
