"""
memory/ticker_memory.py

Per-ticker AI PM outcome memory.

Records every AI PM override per symbol with rationale snippet.
Scores outcomes at 10d/21d via yfinance (incremental alpha = (ai_w - quant_w) * return).
Injects per-ticker history into Phase 2 prompt when a symbol is being considered.

Zero LLM cost — pure Python + yfinance.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
TICKER_MEMORY_PATH = _REPO / "memory" / "ticker_memory.jsonl"


# ── Write ──────────────────────────────────────────────────────────────────────

def record_decision(
    symbol: str,
    date_str: str,
    ai_w: float,
    quant_w: float,
    decision_type: str,
    rationale_snippet: str,
) -> None:
    """Append one override decision to ticker_memory.jsonl. Never raises."""
    try:
        entry = {
            "symbol":            symbol.upper(),
            "date":              date_str,
            "ai_w":              round(float(ai_w), 6),
            "quant_w":           round(float(quant_w), 6),
            "type":              decision_type,
            "rationale":         str(rationale_snippet)[:200],
            "scored":            False,
            "outcome_10d":       None,
            "outcome_21d":       None,
            "verdict":           None,
        }
        TICKER_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TICKER_MEMORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.debug("[TickerMemory] record_decision %s failed: %s", symbol, exc)


# ── Score ──────────────────────────────────────────────────────────────────────

def _fetch_return(symbol: str, from_date: str, horizon: int) -> Optional[float]:
    """Fetch stock return from from_date + horizon trading days. Returns None on failure."""
    try:
        import yfinance as yf
        start = date.fromisoformat(from_date)
        end   = (start + timedelta(days=horizon + 15)).isoformat()
        df    = yf.download(symbol, start=from_date, end=end,
                            auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].squeeze().dropna()
        idx = min(horizon, len(closes) - 1)
        return float((closes.iloc[idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[TickerMemory] _fetch_return %s: %s", symbol, exc)
        return None


def _classify_verdict(r10: Optional[float], r21: Optional[float]) -> Optional[str]:
    if r10 is None:
        return None
    if r10 == 0.0:
        return None  # no incremental alpha — no opinion
    if r10 > 0 and r21 is not None and r21 < 0:
        return "fade"
    if r10 < 0 and r21 is not None and r21 > 0:
        return "early"
    return "win" if r10 >= 0 else "miss"


def score_outcomes(today: date) -> int:
    """
    For each unscored entry 10+ days old: fetch returns, compute incremental alpha,
    classify verdict, rewrite the file. Returns count of newly scored entries.
    """
    if not TICKER_MEMORY_PATH.exists():
        return 0

    entries: List[dict] = []
    try:
        for line in TICKER_MEMORY_PATH.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    except Exception as exc:
        log.warning("[TickerMemory] Could not read: %s", exc)
        return 0

    scored_count = 0
    for entry in entries:
        if entry.get("scored"):
            continue
        try:
            dec_date   = date.fromisoformat(entry["date"])
        except Exception:
            continue
        days_since = (today - dec_date).days
        if days_since < 10:
            continue

        sym   = entry["symbol"]
        ai_w  = entry["ai_w"]
        qw    = entry["quant_w"]
        r10 = r21 = None

        raw10 = _fetch_return(sym, entry["date"], 10)
        if raw10 is not None:
            r10 = round((ai_w - qw) * raw10, 6)

        if days_since >= 21:
            raw21 = _fetch_return(sym, entry["date"], 21)
            if raw21 is not None:
                r21 = round((ai_w - qw) * raw21, 6)

        if r10 is None:
            continue  # yfinance failed — retry next run, do not mark scored

        entry["outcome_10d"] = r10
        entry["outcome_21d"] = r21
        entry["verdict"]     = _classify_verdict(r10, r21)
        entry["scored"]      = True
        scored_count        += 1
        log.info("[TickerMemory] Scored %s @ %s: 10d=%s 21d=%s → %s",
                 sym, entry["date"], r10, r21, entry["verdict"])

    if scored_count:
        tmp = TICKER_MEMORY_PATH.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        tmp.rename(TICKER_MEMORY_PATH)

    return scored_count


# ── Read / Inject ──────────────────────────────────────────────────────────────

def _load_entries() -> List[dict]:
    if not TICKER_MEMORY_PATH.exists():
        return []
    rows = []
    try:
        for line in TICKER_MEMORY_PATH.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def get_ticker_context(symbol: str, n: int = 3) -> str:
    """
    Return formatted string of last N AI PM decisions on this ticker with outcomes.
    Empty string if no history or all entries are unscored.
    """
    sym    = symbol.upper()
    rows   = [e for e in _load_entries() if e.get("symbol") == sym]
    scored = [e for e in rows if e.get("scored")]
    recent = sorted(scored, key=lambda e: e.get("date", ""), reverse=True)[:n]

    if not recent:
        return ""

    lines = [f"AI PM HISTORY — {sym} (last {len(recent)} call(s)):"]
    for e in reversed(recent):  # chronological order
        r10_str  = f"{e['outcome_10d']:+.3%}" if e.get("outcome_10d") is not None else "pending"
        r21_str  = f"{e['outcome_21d']:+.3%}" if e.get("outcome_21d") is not None else "pending"
        verdict  = (e.get("verdict") or "?").upper()
        lines.append(
            f"  {e['date']} {e['type']:8s} ai={e['ai_w']:.1%} vs q={e['quant_w']:.1%}"
            f" → 10d={r10_str} 21d={r21_str} [{verdict}]"
            f"\n    rationale: {e.get('rationale','')[:120]}"
        )
    return "\n".join(lines)


def get_cross_ticker_lessons(n: int = 3) -> str:
    """
    Return last N scored decisions (any ticker) as a cross-asset learning block.
    Empty string if no scored history.
    """
    scored = [e for e in _load_entries() if e.get("scored")]
    recent = sorted(scored, key=lambda e: e.get("date", ""), reverse=True)[:n]
    if not recent:
        return ""
    lines = [f"CROSS-TICKER AI PM LESSONS (last {len(recent)} scored calls):"]
    for e in recent:
        verdict = (e.get("verdict") or "?").upper()
        r10     = f"{e['outcome_10d']:+.3%}" if e.get("outcome_10d") is not None else "?"
        lines.append(
            f"  {e['date']} {e['symbol']:5s} {e['type']:8s}"
            f" 10d={r10} [{verdict}]: {e.get('rationale','')[:100]}"
        )
    return "\n".join(lines)
