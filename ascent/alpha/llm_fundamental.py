"""
ascent/alpha/llm_fundamental.py

LLM-based fundamental alpha signal using Chicago Booth 6-step CoT.

Sends anonymized financial ratios to Claude Haiku. Caches results by
(symbol, quarter_end_date) so the LLM is only called when new
fundamental data is available. Returns a cross-sectional z-score Series.

Source: Kim, Muhn, Nikolaev (2024). arxiv.org/abs/2407.17866
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_PATH = Path("data_cache/llm_fundamental_cache.json")

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
    from ascent.llm.prompt_loader import get_prompt as _get_prompt
except ImportError:  # allow test environments without the full LLM stack
    generate_structured = None  # type: ignore[assignment]
    HAIKU_MODEL = "claude-haiku-4-5-20251001"
    _get_prompt = lambda key, **kw: "[PROMPT UNAVAILABLE]"  # noqa: E731

_SYSTEM_PROMPT = _get_prompt("alpha.llm_fundamental.system")

_LLM_FUNDAMENTAL_SCHEMA = {
    "type": "object",
    "properties": {
        "direction":       {"type": "string", "enum": ["UP", "DOWN", "NEUTRAL"]},
        "confidence":      {"type": "number"},
        "key_trend":       {"type": "string"},
        "uncertainty":     {"type": "string"},
        "quoted_evidence": {
            "type": "string",
            "description": (
                "A direct quote of one or two specific numbers from the provided "
                "metrics table that most support your forecast direction. "
                "Example: 'Q0 gross_profitability=0.412, Q-1=0.389 (+0.023)'. "
                "If no supporting number exists, write 'no clear numerical support'."
            ),
        },
    },
    "required": ["direction", "confidence", "key_trend", "uncertainty", "quoted_evidence"],
    "additionalProperties": False,
}

_USER_TEMPLATE = """Analyze these quarterly financial metrics for an anonymous company.

Financial Data (Q-3 = three quarters ago, Q0 = most recent quarter):
{metrics_table}

Step 1: Identify 3 key trends in revenue growth, gross margin, and asset base (cite specific numbers from the table above).
Step 2: Compute: (a) gross margin change Q-3→Q0, (b) accruals ratio trend, (c) asset growth rate Q-3→Q0.
Step 3: Interpret each economically — improving, stable, or deteriorating, and why.
Step 4: Identify any inflection points in the last 2 quarters.
Step 5: Forecast next-quarter earnings direction. State confidence (0.0–1.0) and primary reason.
Step 6: State the single most important uncertainty in your forecast.

Respond ONLY with a JSON object matching the provided schema. The quoted_evidence field must contain actual numbers copied from the table above."""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _format_metrics_table(ratios: dict) -> str:
    quarters = sorted(ratios.keys())
    metrics  = ["gross_profitability", "accruals", "asset_growth"]
    lines    = ["Quarter | " + " | ".join(metrics), "-" * 60]
    for q in quarters:
        vals = ratios.get(q, {})
        row  = [q] + [f"{vals[m]:.3f}" if m in vals else "N/A" for m in metrics]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _call_llm(symbol: str, metrics_table: str) -> Optional[dict]:
    try:
        user_prompt = _USER_TEMPLATE.format(metrics_table=metrics_table)
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=512,
            temperature=0.2,
            use_cache=True,
            json_schema=_LLM_FUNDAMENTAL_SCHEMA,
        )
        # Structured outputs guarantee valid JSON; parse defensively anyway
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("[LLM Fundamental] No JSON found in response for %s", symbol)
            return None
        parsed = json.loads(raw[start:end])
        direction  = parsed.get("direction", "").upper()
        confidence = float(parsed.get("confidence", 0.0))
        if direction not in ("UP", "DOWN", "NEUTRAL"):
            log.warning("[LLM Fundamental] Invalid direction '%s' for %s", direction, symbol)
            return None
        if not (0.0 <= confidence <= 1.0):
            log.warning("[LLM Fundamental] Confidence out of range %.3f for %s", confidence, symbol)
            return None
        return {
            "direction":       direction,
            "confidence":      confidence,
            "key_trend":       parsed.get("key_trend", ""),
            "uncertainty":     parsed.get("uncertainty", ""),
            "quoted_evidence": parsed.get("quoted_evidence", ""),
        }
    except Exception as exc:
        log.warning("[LLM Fundamental] Call failed for %s: %s", symbol, exc)
        return None


def llm_fundamental_alpha(
    fundamentals_df: Optional[pd.DataFrame],
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Generate LLM-based fundamental alpha scores.

    Args:
        fundamentals_df: DataFrame with columns [symbol, date, gross_profitability,
                         accruals, asset_growth]. One row per (symbol, quarter).
        as_of_date:      Point-in-time cutoff. Defaults to today. Quarters within
                         45 calendar days of this date are excluded (filing lag).

    Returns:
        Cross-sectional z-score Series indexed by symbol. Empty if no valid data.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return pd.Series(dtype=float)

    required  = {"gross_profitability", "accruals", "asset_growth"}
    available = required.intersection(fundamentals_df.columns)
    if len(available) < 2:
        log.warning("[LLM Fundamental] Missing columns. Have: %s", list(fundamentals_df.columns))
        return pd.Series(dtype=float)

    as_of   = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
    cutoff  = as_of - pd.Timedelta(days=45)
    fund    = fundamentals_df.copy()
    fund["date"] = pd.to_datetime(fund["date"])
    fund    = fund[fund["date"] <= cutoff]

    if fund.empty:
        return pd.Series(dtype=float)

    cache       = _load_cache()
    raw_scores: Dict[str, float] = {}
    cache_dirty = False

    for symbol, grp in fund.groupby("symbol"):
        grp = grp.sort_values("date").tail(4)
        if len(grp) < 2:
            continue

        last_date = str(grp["date"].iloc[-1].date())
        cache_key = f"{symbol}_{last_date}"

        if cache_key in cache:
            result = cache[cache_key]
        else:
            ratios = {}
            for i, (_, row) in enumerate(grp.iterrows()):
                label = f"Q-{len(grp)-1-i}" if i < len(grp) - 1 else "Q0"
                ratios[label] = {
                    col: round(float(row[col]), 4)
                    for col in available
                    if pd.notna(row.get(col))
                }
            result = _call_llm(symbol, _format_metrics_table(ratios))
            if result is not None:
                cache[cache_key] = result
                cache_dirty = True

        if result is not None:
            sign = 1.0 if result["direction"] == "UP" else (
                   -1.0 if result["direction"] == "DOWN" else 0.0)
            raw_scores[symbol] = sign * result["confidence"]

    if cache_dirty:
        _save_cache(cache)

    # Log signals for IC tracking
    if raw_scores:
        import json as _json
        _sig_path = Path("logs/llm_fundamental_signals.jsonl")
        _sig_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_sig_path, "a") as _f:
            _f.write(_json.dumps({
                "date": str(as_of.date()),
                "n_symbols": len(raw_scores),
                "scores": {k: round(v, 4) for k, v in raw_scores.items()},
            }) + "\n")

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    if len(scores) < 2:
        return pd.Series(0.0, index=scores.index)
    std    = scores.std()
    if std < 1e-8:
        return pd.Series(0.0, index=scores.index)
    return (scores - scores.mean()) / std
