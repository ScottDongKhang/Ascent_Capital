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


def snapshot_ai_pm(run_date: date, weights: Dict[str, float], force_sealed: bool = False) -> None:
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
        "date":         run_date.isoformat(),
        "weights":      {k: round(v, 6) for k, v in normalized.items()},
        "force_sealed": force_sealed,
    })
    if written:
        log.info("[Counterfactual] Track D snapshot: %d longs, %d shorts",
                 len(longs), len(shorts))


# ── Daily scoring ──────────────────────────────────────────────────────────────

def _portfolio_return(weights: Dict[str, float], prices: Dict[str, dict]) -> Optional[float]:
    """Compute weighted portfolio return. Handles signed weights (longs + shorts).

    Returns None when NONE of the weighted symbols could be priced — this
    distinguishes "no price data this day" (e.g. the daily yfinance fetch failed)
    from a genuinely flat 0.0 return. Returning 0.0 in the no-data case freezes
    the track at its last cumulative value while other tracks keep accruing,
    which silently corrupts every cross-track comparison.
    """
    import math
    ret = 0.0
    priced = 0
    for sym, w in weights.items():
        p = prices.get(sym)
        if not p:
            continue
        prev, curr = p.get("prev", 0), p.get("curr", float("nan"))
        # Skip NaN prices: yfinance period='5d' returns a trailing all-NaN row
        # for today's unpublished bar; a single NaN must not poison the track.
        if prev is None or curr is None or prev <= 0 or math.isnan(prev) or math.isnan(curr):
            continue
        price_ret = (curr - prev) / prev
        ret += w * price_ret  # short (negative w) × positive return = negative contribution
        priced += 1
    if priced == 0:
        return None
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

    _upsert_daily(record)
    return record


def backfill_track_b(history: Dict[str, float]) -> int:
    """Overwrite track_b_return with Alpaca's SETTLED daily return for every date the
    published 1D portfolio-history bar covers.

    The value the runner writes at run time is unreliable: it derives Track B from
    (equity − last_equity)/last_equity, but Alpaca often hasn't marked the account
    intraday, so equity == last_equity → a fake 0.0 (and the June-18 repair nulled
    those to None). Alpaca's 1D portfolio-history bar only settles after ~17:00 PT —
    after our daily run — and is the source of truth. Each run we replay it over the
    log so every settled day carries the real number; today's unsettled row is not in
    `history`, so it is left as-is and self-heals on the next run.

    Returns the number of rows changed (idempotent: re-running with the same history
    is a no-op)."""
    if not history or not DAILY_LOG.exists():
        return 0
    rows, changed = [], 0
    for line in DAILY_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("date")
        if d in history and r.get("track_b_return") != history[d]:
            r["track_b_return"] = history[d]
            changed += 1
        rows.append(r)
    if changed:
        with open(DAILY_LOG, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return changed


def _load_snapshot_asof(path: Path, on_date: str) -> Dict[str, float]:
    """Return the weights from the latest snapshot in `path` dated on/before `on_date`.

    A★/A/D are frozen at each rebalance and held until the next one, so the snapshot
    that applies to a given calendar day is the most recent one with date <= that day —
    the same convention load_snapshots() uses for "today" (it just takes the last)."""
    if not path.exists():
        return {}
    best = None
    for line in path.read_text().splitlines():
        try:
            snap = json.loads(line)
        except Exception:
            continue
        if snap.get("date", "") <= on_date:
            if best is None or snap["date"] >= best["date"]:
                best = snap
    return best.get("weights", {}) if best else {}


def _return_from_closes(weights: Dict[str, float], on_date: str,
                        closes: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Fixed-weight daily return for `weights` on `on_date`, priced from historical
    closes {symbol: {date: close}}. prev = the most recent close strictly before
    on_date. Mirrors _portfolio_return: returns None when no symbol can be priced
    (so a missing-data day records None, never a fabricated 0.0)."""
    import math
    ret, priced = 0.0, 0
    for sym, w in weights.items():
        series = closes.get(sym)
        if not series:
            continue
        curr = series.get(on_date)
        prevs = [d for d in series if d < on_date]
        if curr is None or not prevs:
            continue
        prev = series[max(prevs)]
        if prev is None or prev <= 0 or math.isnan(prev) or math.isnan(curr):
            continue
        ret += w * (curr - prev) / prev
        priced += 1
    return ret if priced else None


def backfill_astar_d(closes: Dict[str, Dict[str, float]]) -> int:
    """Heal Track A★ / A / D rows that recorded None — the missing analog of
    backfill_track_b. Track B self-heals from Alpaca's settled bars; A★/A/D never did,
    so rows left None by the pre-June-19 freeze/NaN-poisoning bugs stayed None forever,
    starving the AI PM evaluation window. This recomputes each null track for every day
    from the as-of snapshot weights + historical closes {symbol: {date: close}}.

    Idempotent and conservative: only fills rows currently None (never overwrites a
    real computed value). Returns the number of (row, track) cells filled."""
    if not closes or not DAILY_LOG.exists():
        return 0
    track_to_path = {
        "track_astar_return": QUANT_STAR_LOG,
        "track_a_return":     QUANT_LOG,
        "track_d_return":     AI_PM_LOG,
    }
    rows, changed = [], 0
    for line in DAILY_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("date")
        for track, path in track_to_path.items():
            if d and r.get(track) is None:
                w = _load_snapshot_asof(path, d)
                if not w:
                    continue
                val = _return_from_closes(w, d, closes)
                if val is not None:
                    r[track] = round(val, 6)
                    changed += 1
        rows.append(r)
    if changed:
        with open(DAILY_LOG, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return changed


def _upsert_daily(record: dict) -> None:
    """Write one row per date (last-wins). Reruns on the same day REPLACE the
    prior row instead of appending — otherwise get_cumulative_returns multiplies
    every rerun into the product (the June-10 overnight rerun wrote ~9 rows)."""
    DAILY_LOG.parent.mkdir(parents=True, exist_ok=True)
    date_str = record["date"]
    rows = []
    if DAILY_LOG.exists():
        for line in DAILY_LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("date") != date_str:
                rows.append(r)
    rows.append(record)
    with open(DAILY_LOG, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


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

def _cumret_over(records: list, key: str) -> Optional[float]:
    """Cumulative return over the days where `key` is non-None. None days are
    skipped, not treated as 0.0. Returns None if the track has no valid day."""
    days = [r for r in records if r.get(key) is not None]
    if not days:
        return None
    v = 1.0
    for r in days:
        v *= (1 + r[key])
    return round((v - 1) * 100, 3)


def _common_window(records: list, key_a: str, key_b: str) -> list:
    return [r for r in records if r.get(key_a) is not None and r.get(key_b) is not None]


def _common_window_diff(records: list, key_a: str, key_b: str) -> Optional[float]:
    """Difference (cum A − cum B) computed ONLY over days where BOTH tracks have
    data. Comparing two tracks cumulated over disjoint windows is meaningless —
    this is what produced the fictional −11.6pp 'AI PM cost' (A★ frozen on
    2026-06-04 while B kept accruing through 2026-06-18). Callers should also
    read the matching n_common_* count: a 1–2 day overlap is noise, not signal."""
    common = _common_window(records, key_a, key_b)
    if not common:
        return None
    va = vb = 1.0
    for r in common:
        va *= (1 + r[key_a])
        vb *= (1 + r[key_b])
    return round((va - vb) * 100, 3)


def get_cumulative_returns() -> dict:
    """Compute cumulative returns for all tracks since AI PM went live.

    Each track is cumulated only over its own non-None days; cross-track
    comparisons (B−A★, D−A★) use only the common window where both have data.
    """
    records = load_daily_records()
    if not records:
        return {}

    return {
        "n_days":      len(records),
        "start_date":  records[0]["date"],
        "end_date":    records[-1]["date"],
        "track_astar": _cumret_over(records, "track_astar_return"),
        "track_a":     _cumret_over(records, "track_a_return"),
        "track_b":     _cumret_over(records, "track_b_return"),
        "track_c":     _cumret_over(records, "track_c_return"),
        "track_d":     _cumret_over(records, "track_d_return"),
        # Honest apples-to-apples comparisons (common window only):
        "ai_value_add_b_vs_astar": _common_window_diff(records, "track_b_return", "track_astar_return"),
        "ai_signal_d_vs_astar":    _common_window_diff(records, "track_d_return", "track_astar_return"),
        "n_common_b_astar": len(_common_window(records, "track_b_return", "track_astar_return")),
        "n_common_d_astar": len(_common_window(records, "track_d_return", "track_astar_return")),
    }


def print_cumulative_report() -> None:
    c = get_cumulative_returns()
    if not c:
        print("[Counterfactual] No data yet")
        return
    def _f(v):
        return f"{v:+.2f}%" if v is not None else "  n/a"
    def _fp(v):
        return f"{v:+.2f}pp" if v is not None else "n/a (no common window)"
    print(f"[Counterfactual] Since AI PM live ({c['start_date']} → {c['end_date']}, {c['n_days']} days):")
    print(f"  Track A★ (Pure Quant):    {_f(c['track_astar'])}")
    print(f"  Track A  (Quant+P1):      {_f(c['track_a'])}")
    print(f"  Track B  (Actual):        {_f(c['track_b'])}")
    print(f"  Track C  (SPY):           {_f(c['track_c'])}")
    print(f"  Track D  (Pure AI PM):    {_f(c['track_d'])}")
    print(f"  AI value add  (B−A★): {_fp(c['ai_value_add_b_vs_astar'])} vs pure quant ({c['n_common_b_astar']} common days)")
    print(f"  AI signal     (D−A★): {_fp(c['ai_signal_d_vs_astar'])} — full authority estimate ({c['n_common_d_astar']} common days)")
