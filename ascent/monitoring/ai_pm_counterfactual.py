# ascent/monitoring/ai_pm_counterfactual.py
"""
Four-track counterfactual engine for AI PM authority evaluation.

Track A★ — Pure Quant (no Phase 1 sleeve priors, no Phase 2 blend)
Track A  — Quant + Phase 1 priors (before Phase 2 blend)
Track B  — Actual portfolio (Alpaca last_equity — source of truth)
Track C  — SPY benchmark
Track D  — Pure AI PM at 100% weight, normalized (diagnostic only — never executes)

Primary metric at Levels 1–2: Track D vs Track A★ (Track B too diluted at 5% weight)
Primary metric at Levels 3–5: Track B vs Track A★ (meaningful at 30–75% weight)

Handles signed weights (longs + shorts) for long-short mode.
"""
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

_REPO          = Path(__file__).resolve().parent.parent.parent
QUANT_STAR_LOG = _REPO / "logs" / "counterfactual_quant_star_snapshots.jsonl"
QUANT_LOG      = _REPO / "logs" / "counterfactual_quant_snapshots.jsonl"
AI_PM_LOG      = _REPO / "logs" / "counterfactual_ai_snapshots.jsonl"
DAILY_LOG      = _REPO / "logs" / "counterfactual_daily.jsonl"


# ── Snapshot functions (rebalance days) ────────────────────────────────────────

def _idempotent_write(path: Path, date_str: str, record: dict) -> bool:
    """Write only if no entry for date_str already exists. Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                if json.loads(line).get("date") == date_str:
                    log.debug("[Counterfactual] Snapshot for %s already exists in %s — skipping", date_str, path.name)
                    return False
            except Exception:
                pass
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def snapshot_quant_star(run_date: date, weights: Dict[str, float]) -> None:
    """Track A★: quant with default regime weights, ZERO Phase 1 influence.
    Call this BEFORE Phase 1 runs on rebalance days."""
    written = _idempotent_write(QUANT_STAR_LOG, run_date.isoformat(), {
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    })
    if written:
        log.info("[Counterfactual] Track A★ snapshot: %d positions", len(weights))


def snapshot_quant(run_date: date, weights: Dict[str, float]) -> None:
    """Track A: quant after Phase 1 sleeve priors, before Phase 2 blend.
    Call this AFTER Phase 1 runs, BEFORE authority_blend() on rebalance days."""
    written = _idempotent_write(QUANT_LOG, run_date.isoformat(), {
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    })
    if written:
        log.info("[Counterfactual] Track A snapshot: %d positions", len(weights))


def snapshot_ai_pm(run_date: date, weights: Dict[str, float]) -> None:
    """Track D: AI PM proposed portfolio, normalized to sum=1.0 (longs only for normalisation).
    Handles signed weights — shorts preserved, longs renormalized.
    Call this AFTER Phase 2 completes on rebalance days."""
    longs  = {k: v for k, v in weights.items() if v > 0}
    shorts = {k: v for k, v in weights.items() if v < 0}

    # Normalize longs to sum=1.0
    long_total = sum(longs.values())
    if long_total > 0:
        longs = {k: v / long_total for k, v in longs.items()}

    normalized = {**longs, **shorts}
    written = _idempotent_write(AI_PM_LOG, run_date.isoformat(), {
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in normalized.items()},
    })
    if written:
        log.info("[Counterfactual] Track D snapshot: %d longs, %d shorts",
                 len(longs), len(shorts))


# ── Daily scoring ──────────────────────────────────────────────────────────────

def _portfolio_return(weights: Dict[str, float], prices: Dict[str, dict]) -> float:
    """Compute weighted portfolio return. Handles signed weights (longs + shorts)."""
    ret = 0.0
    for sym, w in weights.items():
        p = prices.get(sym)
        if p and p.get("prev", 0) > 0:
            price_ret = (p["curr"] - p["prev"]) / p["prev"]
            ret += w * price_ret  # short (negative w) × positive return = negative contribution
    return ret


def score_daily(
    run_date: date,
    quant_star_weights: Optional[Dict[str, float]],
    quant_weights: Optional[Dict[str, float]],
    ai_pm_weights: Optional[Dict[str, float]],
    track_b_return: float,
    spy_return: float,
    prices: Dict[str, dict],
) -> dict:
    """Compute all five track daily returns and append to DAILY_LOG."""
    astar_ret = _portfolio_return(quant_star_weights, prices) if quant_star_weights else None
    a_ret     = _portfolio_return(quant_weights, prices)      if quant_weights     else None
    d_ret     = _portfolio_return(ai_pm_weights, prices)      if ai_pm_weights     else None

    record = {
        "date":               run_date.isoformat(),
        "track_astar_return": round(astar_ret, 6) if astar_ret is not None else None,
        "track_a_return":     round(a_ret, 6)     if a_ret     is not None else None,
        "track_b_return":     round(float(track_b_return), 6),
        "track_c_return":     round(float(spy_return), 6),
        "track_d_return":     round(d_ret, 6)     if d_ret     is not None else None,
    }

    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DAILY_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_last_snapshot(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        try:
            last = json.loads(line)
        except Exception:
            pass
    return last


def load_snapshots() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Returns (quant_star_weights, quant_weights, ai_pm_weights) from last rebalance."""
    def _w(snap):
        return snap.get("weights", {}) if snap else {}
    return (
        _w(_load_last_snapshot(QUANT_STAR_LOG)),
        _w(_load_last_snapshot(QUANT_LOG)),
        _w(_load_last_snapshot(AI_PM_LOG)),
    )


def load_daily_records() -> list:
    if not DAILY_LOG.exists():
        return []
    rows = []
    for line in DAILY_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def get_cumulative_returns() -> dict:
    """Compute cumulative returns for all tracks since AI PM went live."""
    records = load_daily_records()
    if not records:
        return {}

    def cumret(key):
        v = 1.0
        for r in records:
            v *= (1 + r.get(key, 0.0))
        return round((v - 1) * 100, 3)

    return {
        "n_days":      len(records),
        "start_date":  records[0]["date"],
        "end_date":    records[-1]["date"],
        "track_astar": cumret("track_astar_return"),
        "track_a":     cumret("track_a_return"),
        "track_b":     cumret("track_b_return"),
        "track_c":     cumret("track_c_return"),
        "track_d":     cumret("track_d_return"),
    }


def print_cumulative_report() -> None:
    c = get_cumulative_returns()
    if not c:
        print("[Counterfactual] No data yet")
        return
    print(f"[Counterfactual] Since AI PM live ({c['start_date']} → {c['end_date']}, {c['n_days']} days):")
    print(f"  Track A★ (Pure Quant):    {c['track_astar']:+.2f}%")
    print(f"  Track A  (Quant+P1):      {c['track_a']:+.2f}%")
    print(f"  Track B  (Actual):        {c['track_b']:+.2f}%")
    print(f"  Track C  (SPY):           {c['track_c']:+.2f}%")
    print(f"  Track D  (Pure AI PM):    {c['track_d']:+.2f}%")
    print(f"  AI value add  (B−A★): {c['track_b'] - c['track_astar']:+.2f}pp vs pure quant")
    print(f"  AI signal     (D−A★): {c['track_d'] - c['track_astar']:+.2f}pp — full authority estimate")
