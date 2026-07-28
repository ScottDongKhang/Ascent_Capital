#!/usr/bin/env python
"""
One-time collapse of the re-accumulated duplicates in data_cache/prices_live.

Run ONLY after the parquet dedup fix (first-write dedup + evening-stamp
rollover) has landed, since it reuses that module's `_calendar_day_key` as the
single source of truth for what "same trading day" means.

Safety:
  * writes a timestamped .bak.parquet first and verifies it is readable
  * refuses to write if the collapse would drop a distinct (symbol, trading
    day) key, i.e. it may only remove redundant rows, never coverage
  * refuses to write if any surviving group had conflicting `close` values
    (a real ambiguity that a human must adjudicate, not a script)
  * DRY RUN by default; pass --apply to write

Precedent for the .bak naming: data_cache/prices_live.pre_klac_fix.*.bak.parquet
and prices_live.pre_dedup.*.bak.parquet from earlier repairs.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Import the FIXED _calendar_day_key from the branch worktree (main does not
# carry the fix yet), but operate on an explicitly-passed cache path — the
# module's own DATA_DIR is derived from __file__ and would otherwise point at
# the worktree's copy of data_cache rather than production's.
WORKTREE = Path("/Users/scott/IdeaProjects/ascent-capital/.claude/worktrees/risk-mgmt")
sys.path.insert(0, str(WORKTREE))

from ascent.data.store.parquet import _calendar_day_key  # noqa: E402

DEFAULT_PATH = ("/Users/scott/IdeaProjects/ascent-capital/data_cache/"
                "prices_live.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write")
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    path = Path(args.path)
    print(f"cache: {path}")
    df = pd.read_parquet(path)
    n0 = len(df)

    key = pd.DataFrame({"symbol": df["symbol"],
                        "day": _calendar_day_key(df["date"])})
    dup_mask = key.duplicated(keep="last")
    keys_before = key.drop_duplicates().shape[0]

    print(f"rows before        : {n0:,}")
    print(f"duplicate rows     : {int(dup_mask.sum()):,}")
    print(f"distinct (sym,day) : {keys_before:,}")
    print(f"symbols            : {df['symbol'].nunique()}")

    # --- ambiguity check: do any collapsing groups disagree on close? ---
    chk = df.assign(_day=key["day"])
    grp = chk.groupby(["symbol", "_day"])["close"]
    spread = (grp.max() - grp.min()).abs()
    ambiguous = spread[spread > 1e-6]
    print(f"groups w/ conflicting close: {len(ambiguous):,}")
    if len(ambiguous):
        print("  WORST 5 (symbol, day, close spread):")
        print(ambiguous.sort_values(ascending=False).head().to_string())

    out = df[~dup_mask].reset_index(drop=True)
    keys_after = pd.DataFrame({
        "symbol": out["symbol"],
        "day": _calendar_day_key(out["date"])}).drop_duplicates().shape[0]

    print(f"rows after         : {len(out):,}  (removed {n0 - len(out):,})")
    print(f"distinct keys after: {keys_after:,}")

    if keys_after != keys_before:
        print(f"ABORT: key coverage changed {keys_before} -> {keys_after}")
        return 1
    if len(ambiguous):
        print("ABORT: conflicting close values inside collapsing groups — "
              "a human must adjudicate which row is authoritative.")
        return 1
    # spot-check a few well-known symbols keep full history
    for sym in ("SPY", "AAPL", "KLAC", "PSTG"):
        if sym in set(df["symbol"]):
            b = int((df["symbol"] == sym).sum())
            a = int((out["symbol"] == sym).sum())
            print(f"  {sym:6s} {b:6,} -> {a:6,} rows")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix("").as_posix() + f".pre_dedup2.{stamp}.bak.parquet"
    shutil.copy2(path, bak)
    probe = pd.read_parquet(bak)
    assert len(probe) == n0, "backup verification failed"
    print(f"\nbackup OK: {bak}  ({len(probe):,} rows)")

    out.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    print(f"WROTE {path}  rows={len(back):,}")
    k = pd.DataFrame({"s": back["symbol"],
                      "d": _calendar_day_key(back["date"])})
    print(f"remaining duplicates: {int(k.duplicated().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
