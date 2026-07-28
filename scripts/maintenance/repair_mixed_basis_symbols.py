#!/usr/bin/env python
"""
Repair the 4 symbols in data_cache/prices_live whose rows carry TWO different
adjustment bases for the same trading day (CRWD, MIDD, SPGI, HON).

Diagnosis (2026-07-28): after keying rows to their true trading day, 1,419
(symbol, day) groups hold conflicting `close` values. All of them belong to
4 symbols, ~355 days each. The ratios are not noise — CRWD's is exactly 4.0
(its 4:1 split) and HON/SPGI/MIDD sit at 1.05-1.24 (cumulative dividend
adjustment). So one row is adjusted and the other is not: the documented
yahoo / yfinance_hub blending hazard. A blind dedup would keep whichever row
happened to sort last, silently choosing the wrong basis for some symbols.

Method follows the 2026-06-24 KLAC precedent exactly: fresh-fetch each symbol
on the SPLIT-ONLY basis (auto_adjust=False, which still adjusts splits but not
dividends — the basis production `prices_live` is on) and replace that symbol's
rows in place, leaving all other symbols untouched. No total-return swap, so no
discontinuity against the other 934 symbols.

DRY RUN by default; pass --apply to write. Writes a verified backup first.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

WORKTREE = Path("/Users/scott/IdeaProjects/ascent-capital/.claude/worktrees/risk-mgmt")
sys.path.insert(0, str(WORKTREE))
from ascent.data.store.parquet import _calendar_day_key  # noqa: E402

CACHE = Path("/Users/scott/IdeaProjects/ascent-capital/data_cache/prices_live.parquet")
SYMBOLS = ["CRWD", "MIDD", "SPGI", "HON"]


def _conflicts(df: pd.DataFrame) -> pd.Series:
    g = df.groupby(["symbol", "_day"])["close"]
    spread = (g.max() - g.min()).abs()
    return spread[spread > 1e-6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import yfinance as yf

    df = pd.read_parquet(CACHE)
    df["_day"] = _calendar_day_key(df["date"])
    n0 = len(df)
    before = _conflicts(df)
    print(f"rows={n0:,}  conflicting groups={len(before):,}  "
          f"symbols affected={before.reset_index()['symbol'].nunique()}")

    tz = df["date"].dt.tz
    lo = df["_day"].min().strftime("%Y-%m-%d")
    hi = (df["_day"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"refetch window: {lo} -> {hi} (split-only, auto_adjust=False)\n")

    fixed = {}
    for sym in SYMBOLS:
        raw = yf.download(sym, start=lo, end=hi, auto_adjust=False,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            print(f"  {sym}: FETCH EMPTY — skipping (left as-is)")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower).reset_index()
        raw = raw.rename(columns={"index": "date", "Date": "date"})
        raw["symbol"] = sym
        raw["source"] = "yfinance_split_only_repair"
        # sanity: no implausible day-over-day jumps left
        r = raw.sort_values("date")["close"].pct_change().abs()
        jumps = int((r > 0.5).sum())
        print(f"  {sym}: {len(raw):,} rows  "
              f"${raw['close'].min():.2f}-${raw['close'].max():.2f}  "
              f"jumps>50%={jumps}")
        fixed[sym] = raw

    if not fixed:
        print("nothing fetched; aborting")
        return 1

    keep = df[~df["symbol"].isin(fixed)].drop(columns=["_day"])
    new = pd.concat(list(fixed.values()), ignore_index=True)
    # match the cache's stored tz + column set
    new["date"] = pd.to_datetime(new["date"])
    if new["date"].dt.tz is None:
        new["date"] = new["date"].dt.tz_localize(tz)
    else:
        new["date"] = new["date"].dt.tz_convert(tz)
    for c in keep.columns:
        if c not in new.columns:
            new[c] = pd.NA
    new = new[keep.columns]

    out = pd.concat([keep, new], ignore_index=True)
    out["_day"] = _calendar_day_key(out["date"])
    after = _conflicts(out)
    print(f"\nrows: {n0:,} -> {len(out):,}")
    print(f"conflicting groups: {len(before):,} -> {len(after):,}")
    if len(after):
        print("  REMAINING:", after.reset_index()["symbol"].unique()[:10])
    for sym in SYMBOLS:
        b = int((df["symbol"] == sym).sum()); a = int((out["symbol"] == sym).sum())
        print(f"  {sym:5s} {b:6,} -> {a:6,} rows")
    out = out.drop(columns=["_day"])

    if not args.apply:
        print("\nDRY RUN — nothing written.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = CACHE.with_suffix("").as_posix() + f".pre_basis_repair.{stamp}.bak.parquet"
    shutil.copy2(CACHE, bak)
    assert len(pd.read_parquet(bak)) == n0, "backup verification failed"
    print(f"\nbackup OK: {bak}")
    out.to_parquet(CACHE, index=False)
    print(f"WROTE {CACHE} rows={len(pd.read_parquet(CACHE)):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
