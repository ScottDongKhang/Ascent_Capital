"""
memory/r2r_interface.py
R2R memory interface for Ascent Capital.

query_memory(query, n) — semantic search over past verdict history.
ingest_verdict(verdict_path) — add a new verdict to memory.
format_memory_context(results) — format results for LLM prompt injection.

Uses R2R HTTP API if R2R_API_KEY is set in the environment.
Falls back to local keyword search over outputs/debate_log/*.json otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_DEFAULT_DEBATE_LOG_DIR = Path("outputs/debate_log")
R2R_API_KEY = os.environ.get("R2R_API_KEY", "")
R2R_BASE_URL = os.environ.get("R2R_BASE_URL", "https://api.r2r.ai")


# ── Local keyword search (BM25-style scoring) ────────────────────────────────

def _score_verdict(verdict_data: dict, query_tokens: set) -> float:
    """
    Score a verdict dict against a query token set.
    Counts token overlaps in regime, weights keys, reasoning text.
    Higher = more relevant.
    """
    score = 0.0
    try:
        ps = verdict_data.get("portfolio_state", {})
        v = verdict_data.get("verdict", {})

        text_blob = " ".join([
            str(ps.get("us_regime", "")),
            str(ps.get("macro_regime", "")),
            " ".join(ps.get("weights", {}).keys()),
            str(v.get("recommendation", "")),
            str(v.get("reasoning", "")),
            " ".join(v.get("key_risks", [])),
        ]).lower()

        tokens_in_doc = set(text_blob.split())
        score = len(query_tokens & tokens_in_doc)
    except Exception:
        pass
    return score


def _local_search(
    query: str,
    debate_log_dir: Path = _DEFAULT_DEBATE_LOG_DIR,
    n: int = 3,
) -> List[dict]:
    """
    Search past verdicts by keyword overlap. No external dependencies.

    Returns list of dicts with keys: date, recommendation, reasoning, regime.
    Sorted by score descending; limited to top-n with score > 0.
    """
    if not debate_log_dir.exists():
        return []

    query_tokens = set(query.lower().split())
    scored = []

    for verdict_file in debate_log_dir.glob("verdict_*.json"):
        try:
            data = json.loads(verdict_file.read_text())
        except Exception:
            continue

        score = _score_verdict(data, query_tokens)
        if score > 0:
            ps = data.get("portfolio_state", {})
            v = data.get("verdict", {})
            scored.append((score, {
                "date":           data.get("date", "unknown"),
                "recommendation": v.get("recommendation", "unknown"),
                "reasoning":      v.get("reasoning", "")[:300],
                "regime":         ps.get("us_regime", "unknown"),
                "key_risks":      v.get("key_risks", [])[:3],
            }))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:n]]


# ── R2R HTTP API path ─────────────────────────────────────────────────────────

def _r2r_search(query: str, n: int = 3) -> List[dict]:
    """
    Query R2R API for semantically similar past verdicts.
    Raises on failure — caller handles fallback.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {R2R_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_settings": {"search_limit": n},
    }
    resp = requests.post(
        f"{R2R_BASE_URL}/v2/search",
        json=payload,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for hit in data.get("results", []):
        meta = hit.get("metadata", {})
        results.append({
            "date":           meta.get("date", "unknown"),
            "recommendation": meta.get("recommendation", "unknown"),
            "reasoning":      hit.get("text", "")[:300],
            "regime":         meta.get("regime", "unknown"),
            "key_risks":      meta.get("key_risks", []),
        })
    return results


def _r2r_ingest(document_text: str, metadata: dict) -> bool:
    """
    Ingest a document into R2R. Returns True on success.
    Raises on failure — caller handles.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {R2R_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "documents": [{"text": document_text, "metadata": metadata}]
    }
    resp = requests.post(
        f"{R2R_BASE_URL}/v2/ingest_documents",
        json=payload,
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return True


# ── Public interface ──────────────────────────────────────────────────────────

def query_memory(
    query: str,
    n: int = 3,
    debate_log_dir: Path = _DEFAULT_DEBATE_LOG_DIR,
) -> List[dict]:
    """
    Search memory for past situations similar to `query`.

    Uses R2R API if R2R_API_KEY is set, local keyword search otherwise.
    Falls back to local search if R2R call fails.

    Returns:
        List of result dicts: {date, recommendation, reasoning, regime, key_risks}
        Empty list if nothing relevant found.
    """
    if R2R_API_KEY:
        try:
            return _r2r_search(query, n=n)
        except Exception as e:
            log.warning(f"[Memory] R2R search failed ({e}), falling back to local search")

    return _local_search(query, debate_log_dir=debate_log_dir, n=n)


def ingest_verdict(verdict_path: Path) -> None:
    """
    Ingest a verdict JSON into memory (R2R or no-op if no API key).

    Called by debate_runner.py after a verdict is written.
    Non-fatal — logs warning on any failure.
    """
    if not R2R_API_KEY:
        return  # local search reads files directly, no explicit ingestion needed

    try:
        data = json.loads(verdict_path.read_text())
        ps = data.get("portfolio_state", {})
        v = data.get("verdict", {})

        text = (
            f"Date: {data.get('date', 'unknown')}\n"
            f"Regime: {ps.get('us_regime', 'unknown')}\n"
            f"Recommendation: {v.get('recommendation', 'unknown')}\n"
            f"Reasoning: {v.get('reasoning', '')}\n"
            f"Key risks: {', '.join(v.get('key_risks', []))}\n"
            f"Positions: {', '.join(ps.get('weights', {}).keys())}\n"
        )
        metadata = {
            "date": data.get("date"),
            "recommendation": v.get("recommendation"),
            "regime": ps.get("us_regime"),
            "key_risks": v.get("key_risks", []),
        }
        _r2r_ingest(text, metadata)
        log.info(f"[Memory] Ingested verdict {verdict_path.name} into R2R")
    except Exception as e:
        log.warning(f"[Memory] Failed to ingest {verdict_path.name} into R2R: {e}")


def format_memory_context(results: List[dict]) -> str:
    """
    Format memory query results as a concise LLM-readable block.

    Returns:
        Multi-line string suitable for injection into a debate agent prompt.
    """
    if not results:
        return "No relevant historical situations found in memory."

    lines = [f"Historical memory — {len(results)} similar past situation(s):"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[{i}] {r['date']} | Regime: {r['regime']} | Verdict: {r['recommendation']}"
        )
        if r.get("reasoning"):
            lines.append(f"    Reasoning: {r['reasoning'][:200]}")
        if r.get("key_risks"):
            lines.append(f"    Key risks: {', '.join(r['key_risks'][:3])}")

    return "\n".join(lines)
