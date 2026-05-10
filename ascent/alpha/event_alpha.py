"""ascent/alpha/event_alpha.py

Event classifier — translates filing text and market anomalies into structured signals.
Uses Haiku for 8-K classification; rule-based for congressional trades and options.
"""
from __future__ import annotations
import logging
from typing import Optional
from datetime import date

log = logging.getLogger(__name__)

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
except Exception:
    generate_structured = None  # type: ignore[assignment]
    HAIKU_MODEL = "claude-haiku-4-5-20251001"

_NEUTRAL = {"direction": "neutral", "conviction": 0.0, "urgency": "low", "category": "unknown", "rationale": ""}

_CLASSIFY_SYSTEM = """You are a financial event classifier. Classify 8-K filings by their likely price impact.

Rules:
- Only classify on hard facts: revenue miss/beat vs expectation, guidance cut/raise, material adverse events, regulatory approval/rejection.
- Output "neutral" when ambiguous, routine, or administrative.
- Never speculate on management intent.

Output JSON with these fields:
  direction: "buy" | "sell" | "neutral"
  conviction: float 0.0-1.0
  urgency: "high" | "medium" | "low"
  category: string (e.g. "earnings_beat", "guidance_cut", "product_approval")
  rationale: one sentence

Examples:
1. "Revenue beat by 12%, raised full-year guidance" → {"direction":"buy","conviction":0.8,"urgency":"high","category":"earnings_beat","rationale":"Revenue and guidance both beat expectations."}
2. "Lowered FY guidance by 15% citing macro headwinds" → {"direction":"sell","conviction":0.7,"urgency":"high","category":"guidance_cut","rationale":"Material guidance reduction signals demand weakness."}
3. "Changes to board compensation structure" → {"direction":"neutral","conviction":0.0,"urgency":"low","category":"administrative","rationale":"Routine administrative filing with no earnings impact."}
4. "Class action lawsuit filed alleging securities fraud" → {"direction":"sell","conviction":0.6,"urgency":"medium","category":"legal_risk","rationale":"Material litigation risk, outcome uncertain."}
5. "FDA approves new drug application for XYZ" → {"direction":"buy","conviction":0.9,"urgency":"high","category":"product_approval","rationale":"Regulatory approval unlocks significant revenue opportunity."}
"""


def classify_event(filing_text: str, company_name: str, filing_type: str) -> dict:
    """Classify an 8-K filing using Haiku. Returns neutral on any failure."""
    try:
        if generate_structured is None:
            return _NEUTRAL.copy()
        prompt = f"Company: {company_name}\nFiling type: {filing_type}\n\nFiling text:\n{filing_text[:3000]}"
        schema = {
            "type": "object",
            "properties": {
                "direction":  {"type": "string", "enum": ["buy", "sell", "neutral"]},
                "conviction": {"type": "number"},
                "urgency":    {"type": "string", "enum": ["high", "medium", "low"]},
                "category":   {"type": "string"},
                "rationale":  {"type": "string"},
            },
            "required": ["direction", "conviction", "urgency", "category", "rationale"],
        }
        result = generate_structured(
            prompt=prompt,
            system=_CLASSIFY_SYSTEM,
            schema=schema,
            model=HAIKU_MODEL,
        )
        if not isinstance(result, dict) or "direction" not in result:
            return _NEUTRAL.copy()
        result["conviction"] = float(max(0.0, min(1.0, result.get("conviction", 0.0))))
        return result
    except Exception as e:
        log.warning("[EventAlpha] classify_event failed: %s", e)
        return _NEUTRAL.copy()


def classify_congressional_trade(disclosure: dict) -> dict:
    """
    Rule-based classifier for congressional stock disclosures.
    Always conviction=0.4 (30-45 day lag means the trade is not fresh).
    """
    try:
        tx_type = str(disclosure.get("transaction_type", "")).lower()
        symbol  = str(disclosure.get("symbol", ""))
        senator = str(disclosure.get("senator", "unknown"))
        if not symbol:
            return _NEUTRAL.copy()
        direction = "buy" if "purchase" in tx_type else "sell" if "sale" in tx_type else "neutral"
        if direction == "neutral":
            return _NEUTRAL.copy()
        return {
            "symbol":    symbol,
            "direction": direction,
            "conviction": 0.4,
            "urgency":   "low",
            "category":  "congressional_trade",
            "rationale": f"{senator} disclosed a {tx_type}; 30-45 day disclosure lag.",
        }
    except Exception as e:
        log.warning("[EventAlpha] classify_congressional_trade failed: %s", e)
        return _NEUTRAL.copy()


def classify_options_anomaly(symbol: str, iv_zscore: float, put_call_ratio_zscore: float) -> dict:
    """
    Rule-based options anomaly classifier. No LLM needed.

    IV z-score > 2.5 AND put/call z-score > 2.0 → sell, conviction 0.6, urgency high
    IV z-score > 2.5 only → sell, conviction 0.4, urgency medium
    put/call z-score < -2.0 (unusual calls) → buy, conviction 0.5, urgency medium
    """
    try:
        if iv_zscore > 2.5 and put_call_ratio_zscore > 2.0:
            return {
                "symbol": symbol, "direction": "sell", "conviction": 0.6,
                "urgency": "high", "category": "options_iv_spike",
                "rationale": f"IV z={iv_zscore:.1f} and put/call z={put_call_ratio_zscore:.1f} both elevated.",
            }
        if iv_zscore > 2.5:
            return {
                "symbol": symbol, "direction": "sell", "conviction": 0.4,
                "urgency": "medium", "category": "options_iv_elevated",
                "rationale": f"IV z={iv_zscore:.1f} elevated; no directional put/call confirmation.",
            }
        if put_call_ratio_zscore < -2.0:
            return {
                "symbol": symbol, "direction": "buy", "conviction": 0.5,
                "urgency": "medium", "category": "options_call_buying",
                "rationale": f"Unusual call buying: put/call z={put_call_ratio_zscore:.1f}.",
            }
        return _NEUTRAL.copy()
    except Exception as e:
        log.warning("[EventAlpha] classify_options_anomaly failed: %s", e)
        return _NEUTRAL.copy()
