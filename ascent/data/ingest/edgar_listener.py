"""ascent/data/ingest/edgar_listener.py

EDGAR RSS polling with deduplication.
Fetches 8-K, 8-K/A, and 10-Q filings filed within the last N minutes.
"""
from __future__ import annotations
import difflib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

log = logging.getLogger(__name__)

CIK_MAP_PATH   = Path("data_cache/cik_to_symbol.json")
SEEN_PATH      = Path("data_cache/seen_filings.json")
EDGAR_BASE     = "https://www.sec.gov"
ATOM_NS        = "http://www.w3.org/2005/Atom"
_RATE_DELAY    = 0.11  # 10 req/s SEC fair-use limit


def _get(url: str, timeout: int = 10) -> Optional[str]:
    try:
        import requests
        resp = requests.get(url, headers={"User-Agent": "AscentCapital research@ascentcap.io"}, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log.warning("[EDGAR] GET %s failed: %s", url, e)
        return None


def poll_edgar_rss(filing_types: tuple = ("8-K", "8-K/A")) -> list[dict]:
    """Fetch recent filings from EDGAR Atom RSS for each filing type."""
    results = []
    for ft in filing_types:
        url = (
            f"{EDGAR_BASE}/cgi-bin/browse-edgar"
            f"?action=getcurrent&type={quote(ft)}&dateb=&owner=include&count=40&output=atom"
        )
        text = _get(url)
        time.sleep(_RATE_DELAY)
        if not text:
            continue
        try:
            root = ET.fromstring(text)
            ns = {"a": ATOM_NS}
            for entry in root.findall("a:entry", ns):
                title    = entry.findtext("a:title", default="", namespaces=ns)
                link_el  = entry.find("a:link", ns)
                link     = link_el.get("href", "") if link_el is not None else ""
                updated  = entry.findtext("a:updated", default="", namespaces=ns)
                # Extract CIK and company name from title: "CompanyName (CIK) (8-K)"
                cik_match = re.search(r"\((\d+)\)", title)
                cik = cik_match.group(1) if cik_match else ""
                company_name = re.sub(r"\s*\(\d+\)\s*\([^)]+\)", "", title).strip()
                try:
                    filed_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                except Exception:
                    filed_at = datetime.now(timezone.utc)
                results.append({
                    "cik": cik,
                    "company_name": company_name,
                    "filing_type": ft,
                    "filed_at": filed_at,
                    "url": link,
                })
        except Exception as e:
            log.warning("[EDGAR] Parse failed for %s: %s", ft, e)
    return results


def extract_8k_text(filing_url: str) -> str:
    """Fetch filing index, find primary doc, return first 4000 chars of clean text."""
    idx_text = _get(filing_url)
    time.sleep(_RATE_DELAY)
    if not idx_text:
        return ""
    # Find primary document href
    doc_match = re.search(r'href="([^"]+\.(?:htm|txt))"', idx_text, re.IGNORECASE)
    if not doc_match:
        return ""
    doc_url = EDGAR_BASE + doc_match.group(1) if doc_match.group(1).startswith("/") else doc_match.group(1)
    doc_text = _get(doc_url)
    time.sleep(_RATE_DELAY)
    if not doc_text:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", doc_text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:4000]


def _load_cik_map() -> dict:
    if CIK_MAP_PATH.exists():
        try:
            return json.loads(CIK_MAP_PATH.read_text())
        except Exception:
            return {}
    return {}


def is_universe_company(company_name: str, cik: str) -> bool:
    """Check if the filing company is in the Ascent universe."""
    cik_map = _load_cik_map()
    # CIK match (reliable)
    if cik and cik in cik_map:
        return True
    # Fuzzy name match
    known_names = list(cik_map.values()) if cik_map else []
    for name in known_names:
        ratio = difflib.SequenceMatcher(None, company_name.lower(), str(name).lower()).ratio()
        if ratio >= 0.85:
            return True
    return False


def build_cik_map(symbols: list[str]) -> dict:
    """Fetch CIK for each symbol and save to data_cache/cik_to_symbol.json."""
    cik_map: dict = {}
    for sym in symbols:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{sym}%22&dateRange=custom&startdt=2020-01-01"
        text = _get(url)
        time.sleep(_RATE_DELAY)
        if not text:
            continue
        try:
            data = json.loads(text)
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                entity_id = hits[0].get("_source", {}).get("entity_id", "")
                if entity_id:
                    cik_map[str(entity_id)] = sym
        except Exception:
            pass
    CIK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CIK_MAP_PATH.write_text(json.dumps(cik_map))
    return cik_map


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


def get_recent_filings(lookback_minutes: int = 10) -> list[dict]:
    """Return new filings filed within lookback_minutes, not yet seen."""
    all_filings = poll_edgar_rss()
    seen = _load_seen()
    now = datetime.now(timezone.utc)
    cutoff_seconds = lookback_minutes * 60

    new_filings = []
    for f in all_filings:
        if f["url"] in seen:
            continue
        age_seconds = (now - f["filed_at"]).total_seconds()
        if age_seconds <= cutoff_seconds:
            new_filings.append(f)
            seen.add(f["url"])

    if new_filings:
        _save_seen(seen)
    return new_filings


def mark_seen(filing_url: str) -> None:
    """Mark a filing URL as processed."""
    seen = _load_seen()
    seen.add(filing_url)
    _save_seen(seen)
