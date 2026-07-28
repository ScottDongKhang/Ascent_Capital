#!/usr/bin/env python3
"""Rebuild logs/counterfactual_daily.jsonl from clean sources.

    .venv/bin/python scripts/rebuild_counterfactual_log.py            # dry run
    .venv/bin/python scripts/rebuild_counterfactual_log.py --write    # commit it

Dry run by default: it prints the comparison and writes nothing. `--write` backs
the current log up to `.pre_rebuild.<utc-stamp>.bak.jsonl` first.

See ascent/monitoring/counterfactual_rebuild.py for why this is needed and what
the rebuild deliberately does NOT do.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ascent.monitoring.ai_pm_counterfactual import DAILY_LOG  # noqa: E402
from ascent.monitoring.counterfactual_rebuild import (  # noqa: E402
    ASTAR_SNAPS, A_SNAPS, D_SNAPS, backup_existing, load_closes, load_snapshots,
    rebuild_rows, trading_days_between, write_rows,
)

_TRACKS = ("track_astar_return", "track_a_return", "track_b_return",
           "track_c_return", "track_d_return")


def _cumret(rows, key):
    """Compound the non-null values of one track. Gaps are skipped, which is
    exactly the behaviour that made the old log's figures wrong — reported here
    only so the before/after comparison is like-for-like."""
    p, n = 1.0, 0
    for r in rows:
        v = r.get(key)
        if v is not None:
            p *= (1.0 + float(v))
            n += 1
    return (p - 1.0) * 100.0, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite the log (backs up first)")
    ap.add_argument("--period", default="6M", help="Alpaca portfolio-history period")
    args = ap.parse_args()

    old = []
    if DAILY_LOG.exists():
        for line in DAILY_LOG.read_text().splitlines():
            if line.strip():
                try:
                    old.append(json.loads(line))
                except Exception:
                    pass

    print("[Rebuild] Reading settled Track B from Alpaca...")
    from ascent.execution.alpaca_broker import get_portfolio_history
    track_b = get_portfolio_history(period=args.period) or {}
    if not track_b:
        print("[Rebuild] ABORT: Alpaca returned no history — Track B is the one "
              "series that cannot be reconstructed from local files.")
        return 1
    print(f"[Rebuild]   {len(track_b)} settled days "
          f"({min(track_b)} -> {max(track_b)})")

    astar_snaps = load_snapshots(ASTAR_SNAPS)
    a_snaps     = load_snapshots(A_SNAPS)
    d_snaps     = load_snapshots(D_SNAPS)
    print(f"[Rebuild] Snapshots: A*={len(astar_snaps)} A={len(a_snaps)} D={len(d_snaps)}")

    start, end = min(track_b), max(track_b)
    days = trading_days_between(start, end)
    print(f"[Rebuild] Trading days in window: {len(days)}")

    print("[Rebuild] Loading closes from prices_live...")
    needed = {"SPY"}
    for snaps in (astar_snaps, a_snaps, d_snaps):
        for s in snaps:
            needed.update(s["weights"])
    closes = load_closes(symbols=needed)
    print(f"[Rebuild]   {len(closes)} symbols priced (of {len(needed)} referenced)")

    rows = rebuild_rows(days, closes, astar_snaps, a_snaps, d_snaps, track_b)

    # ── comparison ────────────────────────────────────────────────────────────
    print(f"\n{'':<22}{'OLD':>12}{'REBUILT':>12}")
    print(f"{'rows':<22}{len(old):>12}{len(rows):>12}")
    for k in _TRACKS:
        o_n = sum(1 for r in old if r.get(k) is not None)
        n_n = sum(1 for r in rows if r.get(k) is not None)
        print(f"{k + ' (non-null)':<22}{o_n:>12}{n_n:>12}")
    print()
    for k in _TRACKS:
        o_c, o_n = _cumret(old, k)
        n_c, n_n = _cumret(rows, k)
        print(f"{k + ' cum%':<22}{o_c:>11.2f}%{n_c:>11.2f}%   (n {o_n} -> {n_n})")

    # ── integrity assertions ─────────────────────────────────────────────────
    ds = [r["date"] for r in rows]
    problems = []
    if len(ds) != len(set(ds)):
        problems.append("duplicate dates")
    if ds != sorted(ds):
        problems.append("not date-ordered")
    weekend = [d for d in ds if dt.date.fromisoformat(d).weekday() >= 5]
    if weekend:
        problems.append(f"weekend rows: {weekend}")
    allowed = set(days)
    stray = [d for d in ds if d not in allowed]
    if stray:
        problems.append(f"non-trading days: {stray}")
    print("\n[Rebuild] Integrity:", "; ".join(problems) if problems else "clean")
    if problems:
        print("[Rebuild] ABORT: refusing to write a log that fails its own checks.")
        return 1

    if not args.write:
        print("\n[Rebuild] DRY RUN — nothing written. Re-run with --write to commit.")
        return 0

    bak = backup_existing()
    write_rows(rows)
    print(f"\n[Rebuild] Backed up -> {bak}")
    print(f"[Rebuild] Wrote {len(rows)} rows -> {DAILY_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
