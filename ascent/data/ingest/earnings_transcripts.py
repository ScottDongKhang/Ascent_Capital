"""ascent/data/ingest/earnings_transcripts.py

Earnings call transcript pipeline.
Primary source: EDGAR 8-K Item 2.02 (Results of Operations).
1-business-day lag. Forward-fills 63 days.
"""
from __future__ import annotations
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_PATH = Path("data_cache/altdata_transcripts.parquet")

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
except Exception:
    generate_structured = None  # type: ignore[assignment]
    HAIKU_MODEL = "claude-haiku-4-5-20251001"

_QA_PATTERNS = [
    r"QUESTION\s+AND\s+ANSWER",
    r"Q\s*&\s*A\s+SESSION",
    r"QUESTIONS?\s+FROM\s+ANALYST",
    r"OPERATOR[:\s]+.*?(QUESTION|FIRST QUESTION)",
]
_CLASSIFY_SYSTEM = """Analyze this earnings call transcript extract and return JSON with:
- tone: float -1.0 (defensive/cautious) to 1.0 (confident/optimistic)
- defensiveness: float 0.0 (direct answers) to 1.0 (heavily hedged)
- forward_confidence: float -1.0 (pessimistic about future) to 1.0 (optimistic)
- quantitative_ratio: float 0.0 (all qualitative) to 1.0 (specific numbers throughout)

Base on hard evidence in the text. Use 0.0 as default when ambiguous."""


def extract_qa_section(transcript_text: str) -> tuple[str, str]:
    """Split transcript into prepared remarks and Q&A section."""
    upper = transcript_text.upper()
    qa_start = -1
    for pat in _QA_PATTERNS:
        m = re.search(pat, upper)
        if m:
            qa_start = m.start()
            break

    if qa_start == -1:
        return transcript_text[:3000], ""

    prepared = transcript_text[:qa_start][:3000]
    qa = transcript_text[qa_start:][:3000]
    return prepared, qa


def classify_transcript_signal(prepared_remarks: str, qa_section: str, symbol: str) -> dict:
    """Classify transcript via Haiku. Returns zeros on failure."""
    _neutral = {"tone": 0.0, "defensiveness": 0.0, "forward_confidence": 0.0, "quantitative_ratio": 0.0}
    try:
        if generate_structured is None:
            return _neutral.copy()
        schema = {
            "type": "object",
            "properties": {
                "tone":               {"type": "number"},
                "defensiveness":      {"type": "number"},
                "forward_confidence": {"type": "number"},
                "quantitative_ratio": {"type": "number"},
            },
            "required": ["tone", "defensiveness", "forward_confidence", "quantitative_ratio"],
        }
        text = f"PREPARED REMARKS:\n{prepared_remarks}\n\nQ&A:\n{qa_section}"
        result = generate_structured(
            prompt=f"Company: {symbol}\n\n{text}",
            system=_CLASSIFY_SYSTEM,
            schema=schema,
            model=HAIKU_MODEL,
        )
        if not isinstance(result, dict):
            return _neutral.copy()
        out = {}
        for k, lo, hi in [("tone", -1, 1), ("defensiveness", 0, 1),
                           ("forward_confidence", -1, 1), ("quantitative_ratio", 0, 1)]:
            out[k] = float(max(lo, min(hi, result.get(k, 0.0))))
        return out
    except Exception as e:
        log.warning("[Transcripts] classify failed for %s: %s", symbol, e)
        return _neutral.copy()


def _compute_combined_signal(sig: dict) -> float:
    """Average tone + forward_confidence, subtract defensiveness. Range: -2 to 2."""
    return (sig.get("tone", 0.0) + sig.get("forward_confidence", 0.0)
            - sig.get("defensiveness", 0.0))


def build_transcript_signal_panel(
    records: list[dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build dates × symbols panel from pre-classified transcript records.
    records: list of {"symbol": str, "earnings_date": date, "transcript_text": str}
    1-business-day lag enforced. Forward-fills 63 days.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        try:
            sym = rec["symbol"]
            e_date = pd.Timestamp(rec["earnings_date"])
            signal_date = e_date + pd.offsets.BDay(1)
            text = rec.get("transcript_text", "")
            prepared, qa = extract_qa_section(text)
            sig = classify_transcript_signal(prepared, qa, sym)
            combined = _compute_combined_signal(sig)
            rows.append({"date": signal_date, "symbol": sym, "signal": combined})
        except Exception as e:
            log.warning("[Transcripts] record failed: %s", e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="symbol", values="signal", aggfunc="last")

    if start_date and end_date:
        date_range = pd.bdate_range(start=start_date, end=end_date)
        wide = wide.reindex(date_range)

    wide = wide.ffill(limit=63)
    wide.index.name = "date"
    return wide


def update_transcript_signals(records: list[dict]) -> pd.DataFrame:
    """Incremental update from new earnings records."""
    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    panel = build_transcript_signal_panel(records)
    if panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, panel]).groupby(level=0).last()
    else:
        combined = panel

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CACHE_PATH)
    log.info("[Transcripts] Updated transcript signal panel: %s", combined.shape)
    return combined


def load_transcript_signals() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        try:
            return pd.read_parquet(_CACHE_PATH)
        except Exception as e:
            log.warning("[Transcripts] load failed: %s", e)
    return pd.DataFrame()
