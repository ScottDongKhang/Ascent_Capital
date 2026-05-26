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
_DETAIL_PATH = Path("data_cache/altdata_sec_detail.json")
_FRESHNESS_DAYS = 90  # re-fetch only if cache older than this
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

_RISK_PATTERNS = [
    r"ITEM\s+1A[\.A]?\s*[\.\-–—]?\s*RISK\s+FACTORS",
    r"RISK\s+FACTORS",
]
_RISK_NEXT_SECTION = [
    r"ITEM\s+1B[\.\s]",
    r"ITEM\s+2[\.\s]",
    r"UNRESOLVED\s+STAFF\s+COMMENTS",
    r"PROPERTIES",
]


def extract_risk_factors_section(full_text: str) -> str:
    """Extract Item 1A Risk Factors section. Returns empty string if absent."""
    upper = full_text.upper()
    start_idx = -1
    for pat in _RISK_PATTERNS:
        m = re.search(pat, upper)
        if m:
            start_idx = m.start()
            break
    if start_idx == -1:
        return ""
    end_idx = len(full_text)
    for pat in _RISK_NEXT_SECTION:
        m = re.search(pat, upper[start_idx + 100:])
        if m:
            candidate = start_idx + 100 + m.start()
            if candidate < end_idx:
                end_idx = candidate
    section = full_text[start_idx:end_idx]
    section = re.sub(r"<[^>]+>", " ", section)
    section = re.sub(r"\s+", " ", section).strip()
    return section[:4000]


_CLASSIFY_SYSTEM = """You are a financial analyst extracting structured signals from 10-K/10-Q filings.

Analyze the MD&A and Risk Factors text and return JSON with these seven float fields:
- revenue_momentum: -1.0 (decelerating/declining) to +1.0 (accelerating/growing strongly)
- margin_trend: -1.0 (contracting) to +1.0 (expanding)
- tone: -1.0 (defensive/cautious) to +1.0 (confident/optimistic)
- liquidity_risk: 0.0 (no concern) to 1.0 (severe: covenant breach, going concern, cash burn)
- guidance: -1.0 (lowered) to +1.0 (raised), 0.0 if maintained or absent
- risk_trend: -1.0 (significant new risks appearing) to +1.0 (existing risks diminishing or resolved)
- guidance_specificity: 0.0 (vague qualitative language) to 1.0 (specific numerical targets)

Be conservative — only assign extreme values on clear hard facts."""

_YOY_SYSTEM = """You are comparing two consecutive quarterly SEC filings for the same company.
Return JSON with one field: yoy_improvement (-1.0 = significantly worse, 0.0 = unchanged, +1.0 = significantly better).
Base your assessment on changes in tone, revenue trajectory, margin direction, and guidance."""


def _pick_primary_doc(index_html: str, cik_int: int, adsh_nodashes: str) -> str:
    """
    Find the primary 10-K/10-Q document URL from an EDGAR filing index page.
    Skips exhibits (filenames containing 'ex', 'exhibit', or extra suffixes).
    Returns full URL or empty string.
    """
    prefix = f"/Archives/edgar/data/{cik_int}/{adsh_nodashes}/"
    all_links = re.findall(
        rf'href="({re.escape(prefix)}[^"]+\.htm)"',
        index_html, re.IGNORECASE
    )
    _EXHIBIT_PAT = re.compile(
        r"(^ex\d|^exhibit|x\d{2}kxex|x\d{2}qxex|xex\d|_ex\d)", re.IGNORECASE
    )
    for link in all_links:
        filename = link.split("/")[-1]
        if not _EXHIBIT_PAT.search(filename):
            return "https://www.sec.gov" + link
    # Fallback: first link in the filing directory
    return "https://www.sec.gov" + all_links[0] if all_links else ""


def _parse_json_response(raw: str) -> dict:
    """Extract JSON object from LLM response, handling markdown code blocks and trailing text."""
    if isinstance(raw, dict):
        return raw
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in LLM response")
    return json.loads(m.group())


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
    """Extract MD&A section from filing text. Falls back to first 4000 chars.
    Skips TOC entries (< 500 chars) to find the actual section body."""
    upper = full_text.upper()

    for pat in _MDA_PATTERNS:
        for m in re.finditer(pat, upper):
            start_idx = m.start()
            end_idx = len(full_text)
            for epat in _NEXT_SECTION_PATTERNS:
                em = re.search(epat, upper[start_idx + 100:])
                if em:
                    candidate = start_idx + 100 + em.start()
                    if candidate < end_idx:
                        end_idx = candidate
            if end_idx - start_idx < 500:
                continue  # TOC entry — keep searching for actual section
            section = full_text[start_idx:end_idx]
            section = re.sub(r"<[^>]+>", " ", section)
            section = re.sub(r"\s+", " ", section).strip()
            return section[:8000]

    return full_text[:4000]


def classify_filing_signal(
    mda_text: str,
    symbol: str,
    period_end: date,
    risk_factors_text: str = "",
    prev_signals: Optional[dict] = None,
) -> dict:
    """Classify MD&A + Risk Factors into 7 structured signals via Haiku. Returns zeros on failure."""
    _neutral = {
        "revenue_momentum": 0.0, "margin_trend": 0.0, "tone": 0.0,
        "liquidity_risk": 0.0, "guidance": 0.0,
        "risk_trend": 0.0, "guidance_specificity": 0.0,
    }
    try:
        if generate_structured is None:
            return _neutral.copy()
        schema = {
            "type": "object",
            "properties": {
                "revenue_momentum":     {"type": "number"},
                "margin_trend":         {"type": "number"},
                "tone":                 {"type": "number"},
                "liquidity_risk":       {"type": "number"},
                "guidance":             {"type": "number"},
                "risk_trend":           {"type": "number"},
                "guidance_specificity": {"type": "number"},
            },
            "required": ["revenue_momentum", "margin_trend", "tone",
                         "liquidity_risk", "guidance", "risk_trend", "guidance_specificity"],
        }
        prompt = (
            f"Company: {symbol}  Period ending: {period_end}\n\n"
            f"MD&A:\n{mda_text[:2500]}"
        )
        if risk_factors_text:
            prompt += f"\n\nRISK FACTORS:\n{risk_factors_text[:2000]}"

        raw = generate_structured(
            system_prompt=_CLASSIFY_SYSTEM,
            user_prompt=prompt,
            model=HAIKU_MODEL,
        )
        result = _parse_json_response(raw)
        if not isinstance(result, dict):
            return _neutral.copy()

        ranges = [
            ("revenue_momentum", -1.0, 1.0), ("margin_trend", -1.0, 1.0),
            ("tone", -1.0, 1.0), ("liquidity_risk", 0.0, 1.0),
            ("guidance", -1.0, 1.0), ("risk_trend", -1.0, 1.0),
            ("guidance_specificity", 0.0, 1.0),
        ]
        out = {k: float(max(lo, min(hi, result.get(k, 0.0)))) for k, lo, hi in ranges}

        # YoY comparison if previous quarter signals provided
        if prev_signals:
            try:
                prev_str = ", ".join(f"{k}={v:.2f}" for k, v in prev_signals.items())
                curr_str = ", ".join(f"{k}={v:.2f}" for k, v in out.items())
                yoy_prompt = f"Previous quarter: {prev_str}\nCurrent quarter: {curr_str}"
                yoy_raw = generate_structured(
                    system_prompt=_YOY_SYSTEM,
                    user_prompt=yoy_prompt,
                    model=HAIKU_MODEL,
                )
                yoy = _parse_json_response(yoy_raw)
                if isinstance(yoy, dict):
                    out["yoy_improvement"] = float(max(-1.0, min(1.0, yoy.get("yoy_improvement", 0.0))))
            except Exception:
                out["yoy_improvement"] = 0.0

        return out
    except Exception as e:
        log.warning("[SecFilings] classify_filing_signal failed for %s: %s", symbol, e)
        return _neutral.copy()


_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
_ticker_to_cik: dict = {}


def _get_cik_for_symbol(symbol: str) -> str:
    """Return zero-padded 10-digit CIK for a ticker symbol, or '' if not found."""
    global _ticker_to_cik
    if not _ticker_to_cik:
        raw = _get(_TICKERS_URL)
        if raw:
            try:
                data = json.loads(raw)
                _ticker_to_cik = {
                    v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                    for v in data.values()
                }
            except Exception:
                pass
    return _ticker_to_cik.get(symbol.upper(), "")


def fetch_full_text_filing(symbol: str, filing_type: str = "10-K",
                            start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Fetch MD&A section from EDGAR for symbol's most recent filing of filing_type."""
    if start_date is None:
        start_date = (date.today() - timedelta(days=365)).isoformat()
    if end_date is None:
        end_date = date.today().isoformat()

    try:
        cik = _get_cik_for_symbol(symbol)
        if not cik:
            log.debug("[SecFilings] CIK not found for %s", symbol)
            return ""

        time.sleep(_SEC_DELAY)
        sub_raw = _get(_SUBMISSIONS_URL.format(cik=cik))
        if not sub_raw:
            return ""
        sub = json.loads(sub_raw)
        recent = sub.get("filings", {}).get("recent", {})
        forms     = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        accepted_forms = {filing_type, filing_type + "/A"}

        for i, form in enumerate(forms):
            if form not in accepted_forms:
                continue
            fd = filing_dates[i]
            if fd < start_date or fd > end_date:
                continue
            adsh_nodashes = accessions[i].replace("-", "")
            cik_int = int(cik)
            primary = primary_docs[i]
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{adsh_nodashes}/{primary}"
            )
            time.sleep(_SEC_DELAY)
            text = _get(doc_url)
            if text:
                # Strip HTML tags before section extraction (modern iXBRL filings
                # have tags interspersed in text, breaking pattern matching)
                clean = re.sub(r"<[^>]+>", " ", text)
                clean = re.sub(r"\s+", " ", clean)
                return extract_mda_section(clean)

        return ""
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
    Primary signal: revenue_momentum. All 5 signals persisted to detail JSON.
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
            period_end = date.today() - timedelta(days=45)
            # Roll back to last business day (filing signal must land on a trading day)
            raw_signal_date = pd.Timestamp(period_end + timedelta(days=45))
            while raw_signal_date.dayofweek >= 5:
                raw_signal_date -= pd.Timedelta(days=1)
            signal_date = raw_signal_date
            risk_text = extract_risk_factors_section(text)
            prev_sig = load_sec_detail(sym) or None
            sig = classify_filing_signal(text, sym, period_end,
                                         risk_factors_text=risk_text,
                                         prev_signals=prev_sig if prev_sig else None)
            _update_detail_cache(sym, sig, period_end)          # persist all signals
            rows.append({"date": signal_date, "symbol": sym, **sig})
        except Exception as e:
            log.warning("[SecFilings] build_sec_signal_panel: symbol %s failed: %s", sym, e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    wide = df.pivot(columns="symbol", values="revenue_momentum")
    date_range = pd.bdate_range(start=start_date, end=end_date)
    wide = wide.reindex(date_range)
    wide = wide.ffill(limit=90)
    wide.index.name = "date"
    return wide


def update_sec_signals(symbols: list[str], lookback_months: int = 12) -> pd.DataFrame:
    """Incremental update — fetch only symbols missing from cache or with stale data."""
    start_date = (date.today() - timedelta(days=lookback_months * 30)).isoformat()

    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    # Determine which symbols need fetching: absent from cache or all-NaN
    if not existing.empty:
        last_date = existing.index[-1]
        age_days = (pd.Timestamp.today() - last_date).days
        cached_cols = set(existing.columns)
        # Symbol is "fresh" if it's in cache, has ≥1 non-null value, and cache age < 90 days
        fresh = {s for s in symbols
                 if s in cached_cols and existing[s].notna().any() and age_days < _FRESHNESS_DAYS}
        to_fetch = [s for s in symbols if s not in fresh]
    else:
        to_fetch = list(symbols)

    if not to_fetch:
        log.info("[SecFilings] All %d symbols fresh — skipping EDGAR fetch", len(symbols))
        return existing

    log.info("[SecFilings] Fetching %d symbols from EDGAR: %s", len(to_fetch), to_fetch)
    panel = build_sec_signal_panel(to_fetch, start_date=start_date)
    if panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, panel], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
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


def _update_detail_cache(symbol: str, signals: dict, period_end: date) -> None:
    """Persist full signal dict for a symbol to altdata_sec_detail.json."""
    detail: dict = {}
    if _DETAIL_PATH.exists():
        try:
            detail = json.loads(_DETAIL_PATH.read_text())
        except Exception:
            pass
    detail[symbol] = {
        **signals,
        "period_end": period_end.isoformat(),
        "as_of": date.today().isoformat(),
    }
    _DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DETAIL_PATH.write_text(json.dumps(detail, indent=2))


def load_sec_detail(symbol: str) -> dict:
    """Return full signal dict for symbol from altdata_sec_detail.json, or {}."""
    if _DETAIL_PATH.exists():
        try:
            return json.loads(_DETAIL_PATH.read_text()).get(symbol, {})
        except Exception:
            pass
    return {}
