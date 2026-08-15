#!/usr/bin/env python
"""
Repair `data_cache/prices_live.parquet` phantom-row data loss.

Background: `outputs/wf_results/phantom-row-diagnosis-2026-08-15.md` (Task 1,
read-only) found that `prices_live` holds two populations of rows for the
same trading days — "real" rows (date time-of-day == 00:00:00) and "phantom"
rows (time-of-day != 00:00:00, typically 19:00/20:00, the yfinance_hub
late-fetch stamp). Critically, 99.94% of phantom (symbol, normalized-date)
cells have NO matching real row anywhere in the cache: 49 symbols have their
ENTIRE price history only as phantom rows, and 6 more have their pre-2026
backfill only as phantom rows. A naive "drop all non-midnight rows" rule
would permanently destroy that data.

This script instead MERGES the two populations per (symbol, trading day):
  - if a real (00:00:00) row exists for that cell, keep it (its values win)
  - if only a phantom row exists for that cell, keep IT, but rewrite its
    `date` field to midnight on its trading day so it becomes a normal row
    going forward

Merge key: `_calendar_day_key` (`ascent/data/store/parquet.py`), NOT a plain
`dt.normalize()`. This deviates from the plain-normalize wording in the task
brief -- deliberately, based on evidence gathered while implementing this
script (see the repair log / task report for the full trace):

  - Plain `dt.normalize()` as the merge key leaves `pivot_prices(...).shape`
    at ~2,008 rows, not the ~1,650 real NYSE trading days the brief itself
    requires this script to assert. 358 of those extra "dates" are phantom
    rows whose literal `date` field lands on a non-trading day (307 of the
    358 are Sundays) -- an artifact of the yfinance_hub evening-fetch stamp,
    not real market data.
  - Concretely: some phantom rows sharing a literal calendar date with a REAL
    row have DIFFERENT close prices for that date (e.g. BCO 2026-07-23: real
    close 121.09, phantom-stamped-20:00 close 122.00). Per the documented
    rollover rule (`_EVENING_ROLLOVER_HOUR` / `_calendar_day_key`'s
    docstring), an evening (>=17:00) stamp belongs to the NEXT trading day,
    not its own nominal date -- so that phantom row is really 2026-07-24's
    close, not a duplicate of 2026-07-23. Merging via plain normalize would
    have treated it as a duplicate of the real 07-23 row and silently
    discarded a distinct trading day's only price data -- the exact kind of
    loss Task 1's diagnosis warned about, just introduced fresh by this
    script instead of inherited from history.
  - `_calendar_day_key` is not a new mechanism: it is the same function
    `save_parquet`'s own dedup and `reconcile_numbers.py`'s data-integrity
    check already use for this exact "same nominal date, different intraday
    stamp" problem. Using it here (imported, not modified -- CLAUDE.md's "do
    not touch `_EVENING_ROLLOVER_HOUR`/`_calendar_day_key`" constraint is
    about not editing its rollover semantics, not about never calling it)
    keeps this repair consistent with the rest of the codebase and collapses
    the cache to exactly 1,650 dates with zero leftover phantom-only dates.

The repaired frame is written back through `save_parquet(df, "prices_live")`
(not a raw `to_parquet`) so it goes through the existing dedup path.

Safety: backs up the current cache file to
`data_cache/prices_live.parquet.pre_phantom_repair_bak` first, and refuses to
run (raises) if that backup already exists, so a prior backup is never
silently clobbered.

Usage:
  .venv/bin/python scripts/maintenance/repair_prices_live_phantom_rows.py
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ascent.data.store.parquet import (  # noqa: E402
    DATA_DIR,
    _calendar_day_key,
    load_parquet,
    save_parquet,
)
from ascent.data.normalize.prices import pivot_prices  # noqa: E402

CACHE_NAME = "prices_live"
CACHE_PATH = DATA_DIR / f"{CACHE_NAME}.parquet"
BACKUP_PATH = DATA_DIR / f"{CACHE_NAME}.parquet.pre_phantom_repair_bak"

LOG_PATH = REPO_ROOT / "outputs/wf_results/phantom-row-repair-2026-08-15.log"


def _log(lines: list[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def main() -> int:
    log_lines: list[str] = []

    # ---- Step 1: backup, fail loudly if a DIFFERENT one already exists -----
    if not CACHE_PATH.exists():
        raise SystemExit(f"Cache file not found: {CACHE_PATH}")

    if BACKUP_PATH.exists():
        # Idempotent-resume guard, not a silent overwrite: this repair run
        # unlinks CACHE_PATH after backing it up (see step 4), so a *correct*
        # re-run after an earlier failure will legitimately find its own
        # backup still present with the cache restored to match it byte-for-
        # byte. That is safe to resume from. Any OTHER backup content (a
        # different prior repair, or a hand-placed file) is not touched --
        # fail loudly instead of clobbering it.
        if filecmp.cmp(CACHE_PATH, BACKUP_PATH, shallow=False):
            _log(
                log_lines,
                f"Backup already exists at {BACKUP_PATH} and is byte-identical "
                f"to the current {CACHE_PATH} -- resuming from a prior "
                f"backup-then-interrupted run rather than re-copying.",
            )
        else:
            raise SystemExit(
                f"Refusing to overwrite existing backup at {BACKUP_PATH}: it "
                f"differs from the current {CACHE_PATH}. A prior repair "
                "backup with different content is already present -- "
                "investigate before re-running this script."
            )
    else:
        shutil.copy2(CACHE_PATH, BACKUP_PATH)
        _log(log_lines, f"Backed up {CACHE_PATH} -> {BACKUP_PATH}")

    # ---- Step 2: load, normalize date as merge key --------------------------
    df_before = load_parquet(CACHE_NAME)
    n_before = len(df_before)
    before_pivot_shape = pivot_prices(df_before, "close").shape
    _log(log_lines, f"Loaded {CACHE_NAME}: {n_before:,} rows")
    _log(log_lines, f"Before-repair pivot_prices(close).shape = {before_pivot_shape}")

    df = df_before.copy()
    # is_real distinguishes the two POPULATIONS (time-of-day check, unrelated
    # to the merge key below). day_key is the actual merge/grouping key --
    # see the module docstring for why this is `_calendar_day_key`
    # (rollover-aware trading day) and not a plain `dt.normalize()`.
    plain_day = pd.to_datetime(df["date"]).dt.normalize()
    is_real = df["date"] == plain_day
    day_key = _calendar_day_key(df["date"])
    df["_day"] = day_key
    df["_is_real"] = is_real

    n_real = int(is_real.sum())
    n_phantom = int((~is_real).sum())
    _log(log_lines, f"Real (00:00:00) rows: {n_real:,}; phantom rows: {n_phantom:,}")

    # ---- Step 3: merge per (symbol, trading day) -------------------------
    # Stable sort so that within each (symbol, _day) group, a real row (True)
    # always sorts AFTER any phantom row (False) -- drop_duplicates(keep="last")
    # then keeps the real row's values whenever one exists for that cell, and
    # falls back to the (last) phantom row's values otherwise, per Task 1's
    # finding that a phantom-only cell must not be dropped.
    df = df.sort_values(["symbol", "_day", "_is_real"], kind="mergesort")
    n_cells_before_merge = df.drop_duplicates(subset=["symbol", "_day"]).shape[0]
    merged = df.drop_duplicates(subset=["symbol", "_day"], keep="last").copy()

    n_dropped_as_true_dupe = len(df) - len(merged)
    # sanity: how many merged rows are phantom-only survivors (their kept row
    # was NOT a real row, i.e. no real row existed for that cell)
    n_phantom_only_survivors = int((~merged["_is_real"]).sum())
    _log(
        log_lines,
        f"Merged to {len(merged):,} distinct (symbol, normalized-date) cells "
        f"({n_dropped_as_true_dupe:,} rows collapsed away)",
    )
    _log(
        log_lines,
        f"Of those, {n_phantom_only_survivors:,} cells had NO real row and were "
        f"kept from their phantom row (date rewritten to midnight)",
    )

    # `_calendar_day_key` strips tz (see its docstring) to compare across
    # mixed tz-aware/naive/object inputs, but the on-disk `date` column is
    # tz-aware (America/New_York). Re-localize the rewritten dates so the
    # column keeps a single consistent dtype instead of silently degrading
    # to tz-naive/object for the phantom-derived rows only.
    new_dates = merged["_day"]
    if getattr(df_before["date"].dtype, "tz", None) is not None:
        tz = df_before["date"].dtype.tz
        new_dates = new_dates.dt.tz_localize(tz)
    merged["date"] = new_dates
    merged = merged.drop(columns=["_day", "_is_real"])

    # ---- Step 4: write back through save_parquet -----------------------------
    # save_parquet(), if the cache file already exists, reads it back and
    # concats it with the incoming frame before deduping -- and its own
    # internal dedup key is `_calendar_day_key` (evening-rollover-aware), not
    # the plain `dt.normalize()` key this script's merge used. The untouched
    # on-disk phantom rows (still at 19:00/20:00, which DOES roll over) would
    # then get a different dedup key than our already-repaired midnight rows
    # for the same trading day and both would survive, bloating the cache
    # instead of collapsing it. The backup above is the safety net, so it is
    # safe to remove the pre-repair file here and let save_parquet write the
    # fully-merged `merged` frame fresh (its internal dedup still runs, as a
    # no-op safety check, since every row in `merged` is now midnight-stamped
    # and rollover never fires for hour==0).
    CACHE_PATH.unlink()
    save_parquet(merged, CACHE_NAME)

    # ---- Step 5: verify post-repair -------------------------------------------
    df_after = load_parquet(CACHE_NAME)
    n_after = len(df_after)
    after_pivot_shape = pivot_prices(df_after, "close").shape
    _log(log_lines, f"After-repair rows: {n_after:,}")
    _log(log_lines, f"After-repair pivot_prices(close).shape = {after_pivot_shape}")

    after_rows_dim = after_pivot_shape[0]
    if not (1600 <= after_rows_dim <= 1700):
        raise SystemExit(
            f"Post-repair pivot row count {after_rows_dim} is outside the "
            f"expected ~1,650 range -- refusing to declare success. "
            f"Backup preserved at {BACKUP_PATH}."
        )
    assert after_rows_dim < before_pivot_shape[0], (
        "Post-repair pivot should have strictly fewer rows than before-repair pivot "
        f"(before={before_pivot_shape[0]}, after={after_rows_dim})"
    )
    _log(log_lines, "Post-repair pivot shape assertion PASSED (in [1600, 1700], < before)")

    # Check no non-midnight rows remain.
    day_after = pd.to_datetime(df_after["date"]).dt.normalize()
    n_non_midnight_after = int((df_after["date"] != day_after).sum())
    _log(log_lines, f"Non-midnight rows remaining after repair: {n_non_midnight_after:,}")
    if n_non_midnight_after != 0:
        raise SystemExit(
            f"{n_non_midnight_after} non-midnight rows remain after repair -- merge logic bug."
        )

    # ---- Step 6: reconcile_numbers.py data-integrity check ---------------------
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import importlib

        reconcile = importlib.import_module("reconcile_numbers")
        integrity = reconcile.data_integrity()
        _log(log_lines, f"reconcile_numbers.data_integrity(): {integrity}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        _log(log_lines, f"reconcile_numbers.data_integrity() failed (non-fatal): {exc!r}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    _log(log_lines, f"Log written to {LOG_PATH}")

    print(f"\nBEFORE: rows={n_before:,} pivot_shape={before_pivot_shape}")
    print(f"AFTER:  rows={n_after:,} pivot_shape={after_pivot_shape}")
    print(f"Backup: {BACKUP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
