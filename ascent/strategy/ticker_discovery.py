"""
Ticker discovery — surfaces one compelling candidate from current-holdings news.
Uses HAIKU_MODEL for cost efficiency (classifier task, not judgment task).
Candidate must appear in or derive from real Exa-fetched news — not hallucinated.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ascent.llm.client import HAIKU_MODEL, chat_completion

logger = logging.getLogger(__name__)

_DISCOVERY_SYSTEM = """You are a catalyst scanner for an equity portfolio.
You receive live news headlines from current holdings and identify ONE compelling ticker
NOT in the current portfolio that appears in or is directly related to the news provided.

Rules:
- The candidate MUST appear in or be directly derivable from the news text given.
- Do not invent tickers or use outside knowledge not in the news.
- If no compelling candidate is present in the news, set conviction_score below 0.75.
- Return valid JSON only. No markdown fences, no extra text.

JSON format:
{"symbol": "TICKER", "conviction_score": 0.0, "catalyst_snippet": "...", "rationale": "..."}
"""


@dataclass
class DiscoveryResult:
    symbol: str
    conviction_score: float
    catalyst_snippet: str
    rationale: str


def run_discovery(
    news_context: dict[str, list[str]],
    existing_universe: list[str],
) -> DiscoveryResult | None:
    """
    Given Exa news headlines for current holdings, identify ONE ticker candidate
    not in existing_universe. Returns None if conviction < 0.75 or no candidate found.
    Never raises.
    """
    if not news_context or not any(v for v in news_context.values()):
        return None

    news_lines = [
        f"  {sym}: {h}"
        for sym, headlines in news_context.items()
        for h in headlines
    ]
    if not news_lines:
        return None

    user_prompt = (
        f"Current portfolio symbols (do NOT suggest these): {', '.join(existing_universe)}\n\n"
        "Live news from current holdings:\n"
        + "\n".join(news_lines)
        + "\n\nIdentify ONE new ticker candidate with highest conviction. "
        "Must appear in or relate directly to the news above."
    )

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": _DISCOVERY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.2,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]).rstrip("`").strip()
        if not raw:
            return None
        data = json.loads(raw)
        symbol = str(data.get("symbol", "")).upper().strip()
        conviction = float(data.get("conviction_score", 0.0))

        if not symbol or symbol in existing_universe:
            return None
        if conviction < 0.75:
            return None

        return DiscoveryResult(
            symbol=symbol,
            conviction_score=conviction,
            catalyst_snippet=str(data.get("catalyst_snippet", ""))[:200],
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("[TickerDiscovery] run_discovery failed: %s", exc)
        return None
