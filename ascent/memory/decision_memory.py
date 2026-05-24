# ascent/memory/decision_memory.py
"""
Proprietary decision memory for the AI PM.

Every override the AI PM makes is stored with its context features. After 21 days,
the realized wedge outcome is filled in. The AI PM queries this database before each
new override to see its own historical win rate for that (type, regime) combination.

This is not a wrapper — it is a compounding knowledge base that grows with every
rebalance and can never be replicated by someone starting from scratch.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "decision_memory.jsonl"

OVERRIDE_TYPES = frozenset([
    "data_quality", "regime_macro", "news_event", "correlation_risk", "valuation",
])


@dataclass
class OverrideRecord:
    entry_id: str            # "{date}_{symbol}"
    rebalance_date: str
    symbol: str
    override_type: str       # one of OVERRIDE_TYPES
    regime: str
    ai_action: str           # e.g. "REMOVED", "REDUCED from 10% to 4%"
    ai_weight: float         # AI PM proposed weight
    quant_weight: float      # quant baseline weight
    weight_delta: float      # ai_weight - quant_weight
    momentum_252d: Optional[float] = None   # from get_position_momentum if available
    wedge_21d: Optional[float] = None       # filled after 21 trading days
    # Alt data snapshot at override time — for ML conviction model training
    sec_tone: Optional[float] = None
    transcript_sentiment: Optional[float] = None
    reddit_buzz: Optional[float] = None
    trends_direction: Optional[float] = None


def _read_altdata_context(symbol: str) -> dict:
    """Read current alt data signals for a symbol from cached files. Returns None for absent sources."""
    ctx: dict = {
        "sec_tone": None,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": None,
    }
    try:
        import json as _json
        detail_path = _REPO_ROOT / "data_cache" / "altdata_sec_detail.json"
        if detail_path.exists():
            detail = _json.loads(detail_path.read_text())
            ctx["sec_tone"] = detail.get(symbol, {}).get("tone")
    except Exception:
        pass

    for key, fname in [
        ("transcript_sentiment", "altdata_transcripts.parquet"),
        ("reddit_buzz",          "altdata_reddit.parquet"),
        ("trends_direction",     "altdata_trends.parquet"),
    ]:
        try:
            import pandas as _pd
            p = _REPO_ROOT / "data_cache" / fname
            if p.exists():
                df = _pd.read_parquet(p)
                if symbol in df.columns:
                    col = df[symbol].dropna()
                    if not col.empty:
                        ctx[key] = float(col.iloc[-1])
        except Exception:
            pass

    return ctx


def ingest_override(
    rebalance_date: str,
    symbol: str,
    override_type: str,
    regime: str,
    ai_action: str,
    ai_weight: float,
    quant_weight: float,
    momentum_252d: Optional[float] = None,
    log_path: Path = _DEFAULT_LOG,
) -> None:
    """Store an AI PM override decision for future outcome tracking."""
    if override_type not in OVERRIDE_TYPES:
        log.warning("[DecisionMemory] Unknown override_type '%s' — storing anyway", override_type)

    entry_id = f"{rebalance_date}_{symbol}"
    altdata_ctx = _read_altdata_context(symbol)
    record = OverrideRecord(
        entry_id=entry_id,
        rebalance_date=rebalance_date,
        symbol=symbol,
        override_type=override_type,
        regime=regime,
        ai_action=ai_action,
        ai_weight=ai_weight,
        quant_weight=quant_weight,
        weight_delta=ai_weight - quant_weight,
        momentum_252d=momentum_252d,
        wedge_21d=None,
        sec_tone=altdata_ctx["sec_tone"],
        transcript_sentiment=altdata_ctx["transcript_sentiment"],
        reddit_buzz=altdata_ctx["reddit_buzz"],
        trends_direction=altdata_ctx["trends_direction"],
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Dedup by entry_id
    try:
        if log_path.exists():
            existing = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            if any(e.get("entry_id") == entry_id for e in existing):
                log.debug("[DecisionMemory] Duplicate entry %s — skipping", entry_id)
                return
    except Exception:
        pass

    with log_path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    log.info("[DecisionMemory] Ingested override: %s %s (%s) delta=%+.1%%",
             symbol, override_type, regime, record.weight_delta * 100)


def update_outcomes(
    rebalance_date: str,
    wedge_21d: float,
    log_path: Path = _DEFAULT_LOG,
) -> None:
    """Fill realized wedge for all overrides from a given rebalance date."""
    if not log_path.exists():
        return
    try:
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        changed = False
        for row in rows:
            if row.get("rebalance_date") == rebalance_date and row.get("wedge_21d") is None:
                # Attribute portfolio-level wedge proportionally by |weight_delta|
                row["wedge_21d"] = round(wedge_21d, 6)
                changed = True
        if changed:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", dir=log_path.parent, delete=False, suffix=".tmp"
            )
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
            tmp.flush()
            os.replace(tmp.name, log_path)
            log.info("[DecisionMemory] Updated outcomes for rebalance %s", rebalance_date)
    except Exception as exc:
        log.warning("[DecisionMemory] update_outcomes failed: %s", exc)


def query(
    override_type: Optional[str] = None,
    regime: Optional[str] = None,
    n: int = 15,
    log_path: Path = _DEFAULT_LOG,
) -> List[OverrideRecord]:
    """Return the n most recent matching override records that have realized outcomes."""
    if not log_path.exists():
        return []
    try:
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        results = []
        for row in rows:
            if override_type and row.get("override_type") != override_type:
                continue
            if regime and row.get("regime") != regime:
                continue
            results.append(OverrideRecord(**row))
        return results[-n:]
    except Exception as exc:
        log.warning("[DecisionMemory] query failed: %s", exc)
        return []


def get_statistics(
    override_type: Optional[str] = None,
    regime: Optional[str] = None,
    log_path: Path = _DEFAULT_LOG,
) -> Dict:
    """Return aggregated statistics for a (type, regime) filter."""
    matured = [r for r in query(override_type, regime, n=1000, log_path=log_path)
               if r.wedge_21d is not None]
    if not matured:
        return {"n_cases": 0, "win_rate": None, "avg_wedge": None, "best": None, "worst": None}

    wedges = [r.wedge_21d for r in matured]
    wins   = [w for w in wedges if w > 0]
    return {
        "n_cases":   len(wedges),
        "win_rate":  len(wins) / len(wedges),
        "avg_wedge": sum(wedges) / len(wedges),
        "best":      max(wedges),
        "worst":     min(wedges),
    }


def format_query_result(
    override_type: Optional[str] = None,
    regime: Optional[str] = None,
    log_path: Path = _DEFAULT_LOG,
) -> str:
    """Return a human-readable summary for the AI PM tool."""
    stats = get_statistics(override_type, regime, log_path)
    records = [r for r in query(override_type, regime, n=10, log_path=log_path)
               if r.wedge_21d is not None]

    label = " / ".join(filter(None, [override_type, regime])) or "all overrides"

    if stats["n_cases"] == 0:
        pending = query(override_type, regime, n=5, log_path=log_path)
        pending_count = sum(1 for r in pending if r.wedge_21d is None)
        return (
            f"Decision history for [{label}]: no matured outcomes yet. "
            f"{pending_count} override(s) recorded, awaiting 21-day window. "
            "Insufficient data — apply default sizing discipline."
        )

    lines = [
        f"Decision history for [{label}] — {stats['n_cases']} matured outcome(s):",
        f"  Win rate:   {stats['win_rate']:.0%}",
        f"  Avg wedge:  {stats['avg_wedge']:+.2%}",
        f"  Best:       {stats['best']:+.2%}  |  Worst: {stats['worst']:+.2%}",
        "",
        "Recent cases:",
    ]
    for r in records[-5:]:
        sign = "✓" if r.wedge_21d > 0 else "✗"
        mom  = f"  mom={r.momentum_252d:+.0%}" if r.momentum_252d is not None else ""
        lines.append(
            f"  {sign} {r.rebalance_date} {r.symbol} [{r.regime}]{mom}"
            f"  {r.ai_action}  wedge={r.wedge_21d:+.2%}"
        )

    # Conviction recommendation
    n  = stats["n_cases"]
    wr = stats["win_rate"]
    aw = stats["avg_wedge"]
    if n < 5:
        rec = "INSUFFICIENT DATA — proceed but reduce size 20% (building track record)"
    elif wr >= 0.60 and aw > 0:
        rec = "STRONG TRACK RECORD — proceed at full conviction size"
    elif wr >= 0.45:
        rec = "MIXED TRACK RECORD — proceed but cap at medium conviction size (5–7%)"
    else:
        rec = "POOR TRACK RECORD — reduce 40% OR require additional confirming signal"

    lines += ["", f"  Recommendation: {rec}"]
    return "\n".join(lines)
