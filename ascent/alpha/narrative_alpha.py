"""
ascent/alpha/narrative_alpha.py
Narrative shift alpha — detects quarter-over-quarter narrative changes in LLM fundamental analyses.

Signal: +1 (positive narrative flip), -1 (negative flip), 0 (stable).
Cross-sectionally z-scored before return.
Uses Haiku to compare two cached LLM analyses — cheap, fast, factual comparison.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NARRATIVE_CACHE_PATH = Path("data_cache/narrative_shift_cache.json")
LLM_FUND_CACHE_PATH  = Path("data_cache/llm_fundamental_cache.json")

# Lazy import — falls back gracefully if ascent.llm not available in tests
try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL as _HAIKU_MODEL
except ImportError:
    generate_structured = None  # type: ignore[assignment]
    _HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = (
    "You are a financial analyst comparing two quarterly investment narrative summaries. "
    "Reason ONLY from the two summaries provided — do not use any outside knowledge "
    "about the company, sector, or market conditions from your training data. "
    "Your entire analysis must be grounded in the direction, confidence, and trend text given. "
    "Respond only with valid JSON. No other text."
)

_NARRATIVE_SHIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "shift":  {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["shift", "reason"],
    "additionalProperties": False,
}


def _load_narrative_cache() -> dict:
    if NARRATIVE_CACHE_PATH.exists():
        try:
            return json.loads(NARRATIVE_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_narrative_cache(cache: dict) -> None:
    NARRATIVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NARRATIVE_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _get_symbol_analyses(symbol: str, fund_cache: dict) -> list[dict]:
    """
    Return last 2 analyses for symbol, sorted newest-first.
    Each entry: {date, analysis}.
    Cache keys are "{symbol}_{quarter_end_date}".
    """
    prefix = f"{symbol}_"
    entries = []
    for key, analysis in fund_cache.items():
        if key.startswith(prefix):
            date_str = key[len(prefix):]
            entries.append({"date": date_str, "analysis": analysis})
    # Sort newest-first
    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def _compute_shift(symbol: str, current: dict, prior: dict) -> float:
    """
    Call Haiku to compare current vs prior analysis. Returns float in [-1, 1].
    Caches result to NARRATIVE_CACHE_PATH keyed by "{symbol}_{current_date}_{prior_date}".
    Returns 0.0 on any failure.

    Note: current and prior are analysis dicts with keys: direction, confidence, key_trend, uncertainty.
    """
    # Build cache key from content rather than dates since this fn receives analysis dicts directly
    current_dir = current.get("direction", "?")
    current_conf = current.get("confidence", 0.0)
    current_trend = current.get("key_trend", "")
    prior_dir = prior.get("direction", "?")
    prior_conf = prior.get("confidence", 0.0)
    prior_trend = prior.get("key_trend", "")

    # Use a stable cache key derived from the content
    import hashlib
    content_str = f"{symbol}|{prior_dir}|{prior_conf}|{prior_trend}|{current_dir}|{current_conf}|{current_trend}"
    cache_key = f"{symbol}_{hashlib.md5(content_str.encode()).hexdigest()[:12]}"

    # Check cache
    narr_cache = _load_narrative_cache()
    if cache_key in narr_cache:
        cached = narr_cache[cache_key]
        if isinstance(cached, (int, float)):
            return float(cached)
        if isinstance(cached, dict):
            return float(cached.get("shift", 0.0))

    # Build prompt — only key_trend and direction+confidence, not full entry
    user_prompt = (
        f"Prior quarter (Q-1): direction={prior_dir}, confidence={prior_conf:.2f}, "
        f"trend=\"{prior_trend}\"\n"
        f"Current quarter (Q0): direction={current_dir}, confidence={current_conf:.2f}, "
        f"trend=\"{current_trend}\"\n\n"
        f"Has the investment narrative shifted? Respond ONLY with valid JSON:\n"
        f"{{\"shift\": <float -1 to 1>, \"reason\": \"<one sentence>\"}}\n"
        f"shift > 0: narrative improved (bullish signal). "
        f"shift < 0: narrative deteriorated (bearish signal). shift = 0: stable."
    )

    try:
        if generate_structured is None:
            log.warning("[NarrativeAlpha] LLM client unavailable for %s", symbol)
            return 0.0
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=_HAIKU_MODEL,
            max_tokens=128,
            temperature=0.1,
            use_cache=False,
            json_schema=_NARRATIVE_SHIFT_SCHEMA,
        )
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("[NarrativeAlpha] No JSON in response for %s", symbol)
            return 0.0
        parsed = json.loads(raw[start:end])
        shift = float(parsed.get("shift", 0.0))
        # Clamp to [-1, 1]
        shift = max(-1.0, min(1.0, shift))
        # Cache with reason for debugging
        narr_cache[cache_key] = {"shift": shift, "reason": parsed.get("reason", "")}
        _save_narrative_cache(narr_cache)
        return shift
    except Exception as exc:
        log.warning("[NarrativeAlpha] Haiku call failed for %s: %s", symbol, exc)
        return 0.0


def build_narrative_alpha(
    symbols: list[str],
    as_of_date: Optional[str] = None,
) -> pd.Series:
    """
    Returns cross-sectionally z-scored narrative shift scores indexed by symbol.
    Symbols with no prior quarter data get 0.
    Symbols where LLM call fails get 0.
    Result is always a complete Series covering all input symbols.
    """
    if not symbols:
        return pd.Series(dtype=float)

    # Load fundamental cache once
    if not LLM_FUND_CACHE_PATH.exists():
        log.debug("[NarrativeAlpha] LLM fundamental cache not found — returning zeros")
        return pd.Series(0.0, index=symbols)

    try:
        fund_cache = json.loads(LLM_FUND_CACHE_PATH.read_text())
    except Exception as exc:
        log.warning("[NarrativeAlpha] Failed to load LLM fundamental cache: %s", exc)
        return pd.Series(0.0, index=symbols)

    raw_scores: dict[str, float] = {}
    for symbol in symbols:
        analyses = _get_symbol_analyses(symbol, fund_cache)
        if len(analyses) < 2:
            raw_scores[symbol] = 0.0
            continue
        current = analyses[0]["analysis"]
        prior   = analyses[1]["analysis"]
        shift   = _compute_shift(symbol, current, prior)
        raw_scores[symbol] = shift

    scores = pd.Series(raw_scores, dtype=float)

    # If all scores are zero (no data), return without z-scoring
    if scores.abs().sum() < 1e-8:
        return scores

    std = scores.std()
    if std < 1e-8:
        return pd.Series(0.0, index=scores.index)

    z = (scores - scores.mean()) / std
    return z.clip(-3, 3)
