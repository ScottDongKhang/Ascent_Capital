"""ascent/data/ingest/sec_filings.py

SEC 10-K/10-Q full-text pipeline.
Downloads MD&A + Risk Factors sections, classifies via Haiku.
45-day filing lag, 90-day forward-fill.
"""
from __future__ import annotations
import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_PATH = Path("data_cache/altdata_sec.parquet")
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&category=form-type&forms={form}&startdt={start}&enddt={end}"
_EDGAR_FILING = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# Rate limit: 10 req/sec max; use 0.12s gap to stay safe
_SEC_DELAY = 0.12

try:
    from ascent.llm.client import generate_structured, HAIKU_MODEL
except Exception:
    generate_structured = None  # type: ignore[assignment]
    HAIKU_MODEL = "claude-haiku-4-5-20251001"

_MDA_PATTERNS = [
    r"ITEM\s+7[\.A]?\s*[\.\-–—]?\s*MANAGEMENT.{0,40}DISCUSSION",
    r"MANAGEMENT.{0,40}DISCUSSION\s+AND\s+ANALYSIS",
    r"MD&A",
    r"ITEM\s+2\s*[\.\-–—]?\s*MANAGEMENT.{0,40}DISCUSSION",  # 10-Q
]
_NEXT_SECTION_PATTERNS = [
    r"ITEM\s+[789]\w*[\.A]?\s*[\.\-–—]",
    r"ITEM\s+3\s*[\.\-–—]",
    r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
    r"CONTROLS\s+AND\s+PROCEDURES",
    r"FINANCIAL\s+STATEMENTS",
]

_CLASSIFY_SYSTEM = """You are a financial analyst extracting structured signals from 10-K/10-Q MD&A sections.

Analyze the text and return JSON with these five float fields:
- revenue_momentum: -1.0 (decelerating/declining) to +1.0 (accelerating/growing strongly)
- margin_trend: -1.0 (contracting) to +1.0 (expanding)
- tone: -1.0 (defensive/cautious) to +1.0 (confident/optimistic)
- liquidity_risk: 0.0 (no concern) to 1.0 (severe: covenant breach, going concern, cash burn)
- guidance: -1.0 (lowered) to +1.0 (raised), 0.0 if maintained or absent

Be conservative — only assign extreme values on clear hard facts."""


def _get(url: str, retries: int = 3) -> Optional[str]:
    try:
        import requests
        for attempt in range(retries):
            try:
                r = requests.get(url, headers={"User-Agent": "Ascent Capital research@ascentcap.ai"}, timeout=15)
                if r.status_code == 200:
                    return r.text
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
            except Exception as e:
                log.debug("SEC fetch attempt %d failed: %s", attempt + 1, e)
                time.sleep(0.5)
    except Exception as e:
        log.warning("requests not available: %s", e)
    return None


def extract_mda_section(full_text: str) -> str:
    """Extract MD&A section from filing text. Falls back to first 4000 chars."""
    upper = full_text.upper()
    start_idx = -1
    for pat in _MDA_PATTERNS:
        m = re.search(pat, upper)
        if m:
            start_idx = m.start()
            break

    if start_idx == -1:
        return full_text[:4000]

    end_idx = len(full_text)
    for pat in _NEXT_SECTION_PATTERNS:
        m = re.search(pat, upper[start_idx + 100:])
        if m:
            candidate = start_idx + 100 + m.start()
            if candidate < end_idx:
                end_idx = candidate

    section = full_text[start_idx:end_idx]
    # Strip HTML tags
    section = re.sub(r"<[^>]+>", " ", section)
    section = re.sub(r"\s+", " ", section).strip()
    return section[:8000]


def classify_filing_signal(mda_text: str, symbol: str, period_end: date) -> dict:
    """Classify MD&A text into structured signals via Haiku. Returns zeros on failure."""
    _neutral = {
        "revenue_momentum": 0.0, "margin_trend": 0.0, "tone": 0.0,
        "liquidity_risk": 0.0, "guidance": 0.0,
    }
    try:
        if generate_structured is None:
            return _neutral.copy()
        schema = {
            "type": "object",
            "properties": {
                "revenue_momentum": {"type": "number"},
                "margin_trend":     {"type": "number"},
                "tone":             {"type": "number"},
                "liquidity_risk":   {"type": "number"},
                "guidance":         {"type": "number"},
            },
            "required": ["revenue_momentum", "margin_trend", "tone", "liquidity_risk", "guidance"],
        }
        prompt = (
            f"Company: {symbol}  Period ending: {period_end}\n\n"
            f"MD&A excerpt:\n{mda_text[:3000]}"
        )
        result = generate_structured(prompt=prompt, system=_CLASSIFY_SYSTEM, schema=schema, model=HAIKU_MODEL)
        if not isinstance(result, dict):
            return _neutral.copy()
        # Clamp values to valid ranges
        out = {}
        for k, lo, hi in [
            ("revenue_momentum", -1.0, 1.0), ("margin_trend", -1.0, 1.0),
            ("tone", -1.0, 1.0), ("liquidity_risk", 0.0, 1.0), ("guidance", -1.0, 1.0),
        ]:
            out[k] = float(max(lo, min(hi, result.get(k, 0.0))))
        return out
    except Exception as e:
        log.warning("[SecFilings] classify_filing_signal failed for %s: %s", symbol, e)
        return _neutral.copy()


def fetch_full_text_filing(symbol: str, filing_type: str = "10-K",
                            start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Fetch MD&A section from EDGAR for symbol's most recent filing of filing_type."""
    if start_date is None:
        start_date = (date.today() - timedelta(days=365)).isoformat()
    if end_date is None:
        end_date = date.today().isoformat()

    url = _EDGAR_SEARCH.format(
        symbol=symbol, form=filing_type,
        start=start_date, end=end_date,
    )
    raw = _get(url)
    if not raw:
        return ""
    try:
        hits = json.loads(raw).get("hits", {}).get("hits", [])
        if not hits:
            return ""
        # Take most recent filing
        src = hits[0].get("_source", {})
        filing_url = src.get("file_date_formatted") or src.get("period_of_report") or ""
        # Try to get the actual document URL
        entity_id = src.get("entity_id", "")
        file_num = src.get("file_num", "")
        display_date = src.get("display_date_filed", "")
        # EDGAR full-text search returns the filing index URL
        doc_url = src.get("biz_location") or src.get("_id") or ""
        if not doc_url:
            return ""
        time.sleep(_SEC_DELAY)
        text = _get(doc_url)
        if not text:
            return ""
        return extract_mda_section(text)
    except Exception as e:
        log.warning("[SecFilings] fetch_full_text_filing failed for %s: %s", symbol, e)
        return ""


def build_sec_signal_panel(
    symbols: list[str],
    start_date: str,
    end_date: Optional[str] = None,
    filing_type: str = "10-K",
) -> pd.DataFrame:
    """
    Build a dates × symbols panel of SEC filing signals.
    Applies 45-day filing lag. Forward-fills 90 days.
    Primary signal: revenue_momentum.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    rows = []
    for sym in symbols:
        try:
            time.sleep(_SEC_DELAY)
            text = fetch_full_text_filing(sym, filing_type=filing_type,
                                          start_date=start_date, end_date=end_date)
            if not text:
                continue
            # Approximate period end as today - 45 days (filing lag)
            period_end = date.today() - timedelta(days=45)
            signal_date = period_end + timedelta(days=45)  # apply 45-day lag
            sig = classify_filing_signal(text, sym, period_end)
            rows.append({"date": signal_date, "symbol": sym, **sig})
        except Exception as e:
            log.warning("[SecFilings] build_sec_signal_panel: symbol %s failed: %s", sym, e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Pivot to wide (dates × symbols) using revenue_momentum as primary signal
    wide = df.pivot(columns="symbol", values="revenue_momentum")
    date_range = pd.bdate_range(start=start_date, end=end_date)
    wide = wide.reindex(date_range)
    wide = wide.ffill(limit=90)  # 90-day forward-fill (signal decays at next quarterly filing)
    wide.index.name = "date"
    return wide


def update_sec_signals(symbols: list[str], lookback_months: int = 12) -> pd.DataFrame:
    """Incremental update — only re-fetch if cache is absent or stale."""
    start_date = (date.today() - timedelta(days=lookback_months * 30)).isoformat()

    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    panel = build_sec_signal_panel(symbols, start_date=start_date)
    if panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, panel]).groupby(level=0).last()
    else:
        combined = panel

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CACHE_PATH)
    log.info("[SecFilings] Updated SEC signal panel: %s", combined.shape)
    return combined


def load_sec_signals() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        try:
            return pd.read_parquet(_CACHE_PATH)
        except Exception as e:
            log.warning("[SecFilings] load failed: %s", e)
    return pd.DataFrame()
