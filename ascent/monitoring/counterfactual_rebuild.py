"""Rebuild logs/counterfactual_daily.jsonl from clean sources.

Why this exists (audit 2026-07-27). The live log had three defects that made every
figure derived from it unusable:

1. **Missing days.** 45 rows for 78 trading days. `backfill_track_b` could only
   mutate rows that already existed, never insert, so any day the pipeline did
   not run was permanently absent — including a 19-day outage hole. `_cumret_over`
   chains straight across gaps as if those days never happened, which inflated
   every cumulative figure 2.5-4x. Track C claimed SPY +16.63% over a window in
   which SPY returned +5.31%.

2. **Track B keyed one day late.** `datetime.fromtimestamp(ts)` with no tz, on
   Alpaca's UTC epochs, rendered in host-local time (UTC+7): a 16:00 ET close
   became the next calendar day. Same-day corr(B, A*) was -0.005 against +0.60 at
   lag 1 — impossible for two books sharing 95% of their holdings, and the source
   of the headline "AI PM cost".

3. **A market-holiday row.** 2026-06-19 (Juneteenth) carried
   track_b +1.53% / track_c +1.04% on a day the market was shut.

(1) and (2) are fixed at source, but the existing rows were written under the old
behaviour, so the history itself has to be re-derived.

Sources, all independent of the broken log:
    Track B    Alpaca `get_portfolio_history()` — SETTLED 1D bars, now
               market-dated. Verified 2026-07-28: 86 dates, 17 Mondays,
               0 Saturdays, 0 non-trading days (the bug produced 1 and 14).
    A* / A / D  the snapshot files, priced from prices_live.parquet
    Track C    SPY closes from prices_live.parquet

Deliberate limits, so nobody reads more into the output than it supports:
  - A track with no snapshot yet is None, never 0.0. A fabricated zero freezes a
    track at its last cumulative value while the others accrue.
  - Alpaca's equity series is total-return (dividends reinvested); A*/A/D are
    priced from prices_live closes, which on the production cache are split-only.
    So B is not perfectly comparable to A*/D on dividend-paying names. This
    asymmetry is pre-existing — score_daily uses the same two sources — and is
    NOT silently corrected here.
  - Only the mechanical reconstruction is performed. Nothing is interpolated
    across days where a source is absent.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ascent.monitoring.ai_pm_counterfactual import (
    DAILY_LOG,
    _delta_portfolio_return,
    _portfolio_return,
)

_REPO = Path(__file__).resolve().parent.parent.parent

ASTAR_SNAPS = _REPO / "logs" / "counterfactual_quant_star_snapshots.jsonl"
A_SNAPS     = _REPO / "logs" / "counterfactual_quant_snapshots.jsonl"
D_SNAPS     = _REPO / "logs" / "counterfactual_ai_snapshots.jsonl"

_TRACK_KEYS = (
    "track_astar_return", "track_a_return", "track_b_return",
    "track_c_return", "track_d_return",
)


# ── sources ───────────────────────────────────────────────────────────────────

def load_snapshots(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except Exception:
            continue
        if s.get("date") and isinstance(s.get("weights"), dict):
            out.append({"date": s["date"], "weights": s["weights"]})
    out.sort(key=lambda s: s["date"])
    return out


def snapshot_asof(snaps: List[dict], on_date: str) -> Optional[Dict[str, float]]:
    """Weights in force on `on_date`.

    A*/A/D are frozen at each rebalance and held until the next, so the snapshot
    that applies is the most recent one dated on or before the day. Returns None
    when no snapshot exists yet — that track simply did not exist.
    """
    best = None
    for s in snaps:
        if s["date"] <= on_date and (best is None or s["date"] >= best["date"]):
            best = s
    return dict(best["weights"]) if best else None


def load_closes(symbols=None) -> Dict[str, Dict[str, float]]:
    """{symbol: {date: close}} from prices_live, deduplicated.

    prices_live is known to carry duplicate (symbol, day) rows from historic
    multi-source blending, and its `date` column can be tz-aware, tz-naive, or an
    object column mixing both. Keying is delegated to the store's own
    `_calendar_day_key`, which handles all three shapes and applies the
    evening-rollover rule (a bar stamped >=17:00 local belongs to the NEXT
    calendar day). Using anything else here would silently disagree with the
    dates the live pipeline reads. Last value wins.
    """
    from ascent.data.store.parquet import _calendar_day_key, load_parquet

    df = load_parquet("prices_live")
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["_d"] = _calendar_day_key(df["date"]).astype(str).str.slice(0, 10)
    if symbols:
        df = df[df["symbol"].isin(set(symbols))]
    df = df.drop_duplicates(subset=["symbol", "_d"], keep="last")
    out: Dict[str, Dict[str, float]] = {}
    for sym, d, c in zip(df["symbol"], df["_d"], df["close"]):
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        out.setdefault(str(sym), {})[d] = v
    return out


def build_price_map(closes: Dict[str, Dict[str, float]], day: str,
                    prev_day: str) -> Dict[str, dict]:
    """{symbol: {"prev": close, "curr": close}} for the two days, in the shape
    `_portfolio_return` expects. Symbols missing either close are omitted rather
    than defaulted, so they contribute nothing instead of a fake 0.0."""
    pm = {}
    for sym, series in closes.items():
        p, c = series.get(prev_day), series.get(day)
        if p is None or c is None:
            continue
        pm[sym] = {"prev": p, "curr": c}
    return pm


def trading_days_between(start: str, end: str) -> List[str]:
    """NYSE trading days in [start, end], from the stdlib holiday calendar in
    scripts/heartbeat_check.py (pandas_market_calendars is not installed)."""
    sys.path.insert(0, str(_REPO / "scripts"))
    from heartbeat_check import _holiday_set, is_trading_day  # noqa: E402

    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    hol = _holiday_set(a.year - 1, b.year + 1)
    out, cur = [], a
    while cur <= b:
        if is_trading_day(cur, hol):
            out.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return out


# ── rebuild ───────────────────────────────────────────────────────────────────

def rebuild_rows(trading_days: List[str],
                 closes: Dict[str, Dict[str, float]],
                 astar_snaps: List[dict],
                 a_snaps: List[dict],
                 d_snaps: List[dict],
                 track_b: Dict[str, float],
                 spy_symbol: str = "SPY") -> List[dict]:
    """One row per trading day, each value traceable to a source.

    Days on which no track can be computed are dropped rather than emitted as a
    row of nulls — an all-null row is noise that makes gaps look like data.
    """
    rows: List[dict] = []
    for i, day in enumerate(trading_days):
        prev_day = trading_days[i - 1] if i > 0 else None
        pm = build_price_map(closes, day, prev_day) if prev_day else {}

        astar_w = snapshot_asof(astar_snaps, day)
        a_w     = snapshot_asof(a_snaps, day)
        d_w     = snapshot_asof(d_snaps, day)

        astar = _portfolio_return(astar_w, pm) if (astar_w and pm) else None
        a_ret = _portfolio_return(a_w, pm)     if (a_w and pm)     else None
        d_ret = _portfolio_return(d_w, pm)     if (d_w and pm)     else None
        delta = (_delta_portfolio_return(d_w, astar_w, pm)
                 if (d_w and astar_w and pm) else None)

        spy = pm.get(spy_symbol)
        c_ret = ((spy["curr"] - spy["prev"]) / spy["prev"]) if spy and spy["prev"] else None

        b_ret = track_b.get(day)

        row = {
            "date":               day,
            "track_astar_return": round(astar, 6) if astar is not None else None,
            "track_a_return":     round(a_ret, 6) if a_ret is not None else None,
            "track_b_return":     round(float(b_ret), 6) if b_ret is not None else None,
            "track_c_return":     round(c_ret, 6) if c_ret is not None else None,
            "track_d_return":     round(d_ret, 6) if d_ret is not None else None,
            "track_delta_return": round(delta, 6) if delta is not None else None,
            "source":             "rebuild",
        }
        if all(row[k] is None for k in _TRACK_KEYS):
            continue
        rows.append(row)
    return rows


def backup_existing(log_path: Path = None) -> Optional[Path]:
    """Copy the current log aside before overwriting. Returns the backup path."""
    log_path = log_path or DAILY_LOG
    if not log_path.exists():
        return None
    stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    bak = log_path.with_suffix(f".pre_rebuild.{stamp}.bak.jsonl")
    shutil.copy2(log_path, bak)
    return bak


def write_rows(rows: List[dict], log_path: Path = None) -> None:
    log_path = log_path or DAILY_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
