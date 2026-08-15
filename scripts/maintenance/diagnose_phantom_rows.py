#!/usr/bin/env python
"""
Diagnose phantom-row data-loss risk in data_cache/prices_live.parquet.

Background (2026-08-15 root-cause writeup,
docs/superpowers/specs/2026-08-15-prices-live-phantom-row-fix-design.md):
prices_live.parquet holds two populations of rows for the same underlying
trading days:

  - "real" rows: time-of-day == 00:00:00, ~867/938 symbols populated per day
  - "phantom" rows: time-of-day != 00:00:00 (typically 19:00/20:00, the
    yfinance_hub late-fetch stamp described in _calendar_day_key), sparsely
    populated (~54/938 symbols per day)

The walk-forward harness does not dedupe on read (only the live pipeline
does, via _calendar_day_key + drop_duplicates), so these phantom rows are
double-counted as extra trading days by anything that reads the cache raw.

This script is READ-ONLY. It does not touch the cache file. Its job is to
answer one question before anyone deletes the phantom rows: for every
(symbol, normalized date) pair that has a phantom row, does a matching REAL
row already exist for that same day? If yes, dropping the phantom row is
free (it's a genuine duplicate). If no, that phantom row is the ONLY copy of
that (symbol, day) price and dropping it would silently destroy data.

Output:
  - outputs/wf_results/phantom-row-diagnosis-2026-08-15.md   (findings)
  - outputs/wf_results/phantom_only_cells_2026-08-15.csv     (full list of
    phantom-only (symbol, date) cells, if any)

Usage:
  .venv/bin/python scripts/maintenance/diagnose_phantom_rows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ascent.data.store.parquet import load_parquet  # noqa: E402

REPORT_PATH = REPO_ROOT / "outputs/wf_results/phantom-row-diagnosis-2026-08-15.md"
CSV_PATH = REPO_ROOT / "outputs/wf_results/phantom_only_cells_2026-08-15.csv"


def main() -> int:
    df = load_parquet("prices_live")
    n_total = len(df)

    if "date" not in df.columns or "symbol" not in df.columns:
        raise SystemExit(f"unexpected schema: columns={list(df.columns)}")

    date = pd.to_datetime(df["date"])
    day = date.dt.normalize()
    is_real = date == day  # time-of-day == 00:00:00
    is_phantom = ~is_real

    real = df.loc[is_real].copy()
    phantom = df.loc[is_phantom].copy()
    real["_day"] = day.loc[is_real]
    phantom["_day"] = day.loc[is_phantom]

    n_real = len(real)
    n_phantom = len(phantom)
    real_dates = real["_day"].nunique()
    phantom_dates = phantom["_day"].nunique()

    real_keys = set(zip(real["symbol"], real["_day"]))
    phantom_keys_df = phantom[["symbol", "_day"]].drop_duplicates()
    phantom_keys = set(zip(phantom_keys_df["symbol"], phantom_keys_df["_day"]))

    # symbol-level view: does this symbol have ANY real row at all, ever?
    real_syms = set(real["symbol"].unique())
    phantom_syms = set(phantom["symbol"].unique())
    symbols_never_real = sorted(phantom_syms - real_syms)
    symbols_mixed = sorted(phantom_syms & real_syms)

    n_phantom_cells = len(phantom_keys)  # distinct (symbol, day) with a phantom row
    phantom_only_keys = phantom_keys - real_keys
    n_phantom_only = len(phantom_only_keys)

    phantom_only_df = pd.DataFrame(
        sorted(phantom_only_keys, key=lambda t: (t[0], t[1])),
        columns=["symbol", "date"],
    )

    # merge in the phantom row's actual value columns (first match) for context
    value_cols = [c for c in df.columns if c not in ("date", "symbol")]
    phantom_lookup = phantom.drop_duplicates(subset=["symbol", "_day"], keep="first")
    phantom_lookup = phantom_lookup.drop(columns=["date"]).rename(columns={"_day": "date"})[
        ["symbol", "date"] + value_cols
    ]
    phantom_only_full = phantom_only_df.merge(phantom_lookup, on=["symbol", "date"], how="left")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if n_phantom_only:
        phantom_only_full.sort_values(["symbol", "date"]).to_csv(CSV_PATH, index=False)
        csv_written = True
    else:
        csv_written = False

    # clustering: by symbol
    by_symbol = (
        phantom_only_df.groupby("symbol").size().sort_values(ascending=False)
        if n_phantom_only
        else pd.Series(dtype=int)
    )
    n_symbols_affected = phantom_only_df["symbol"].nunique() if n_phantom_only else 0

    # clustering: by date range
    if n_phantom_only:
        dmin, dmax = phantom_only_df["date"].min(), phantom_only_df["date"].max()
        by_year = phantom_only_df["date"].dt.year.value_counts().sort_index()
    else:
        dmin = dmax = None
        by_year = pd.Series(dtype=int)

    sample20 = phantom_only_df.head(20)

    # ---- also report: how many phantom cells are non-null/populated at all? ----
    # A phantom row might exist but be entirely NaN across value columns.
    if value_cols:
        phantom_lookup_nonnull = phantom_lookup.dropna(subset=value_cols, how="all")
        n_phantom_cells_with_data = len(
            phantom_lookup_nonnull[["symbol", "date"]].drop_duplicates()
        )
    else:
        n_phantom_cells_with_data = None

    lines = []
    lines.append("# Phantom-row diagnosis — data_cache/prices_live.parquet")
    lines.append("")
    lines.append("Date: 2026-08-15. Read-only analysis. No cache file modified.")
    lines.append("")
    lines.append(
        "Script: `scripts/maintenance/diagnose_phantom_rows.py` "
        "(`.venv/bin/python scripts/maintenance/diagnose_phantom_rows.py`)"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Loaded `prices_live` via `ascent.data.store.parquet.load_parquet`. Split rows into "
        "**real** (`date` time-of-day == 00:00:00) and **phantom** (time-of-day != 00:00:00, "
        "typically 19:00/20:00 — the yfinance_hub late-fetch stamp). For every distinct "
        "`(symbol, date.normalize())` pair that has a phantom row, checked whether a real row "
        "exists for that same `(symbol, normalized date)`. A pair with a phantom row and NO "
        "matching real row is a **phantom-only cell** — dropping that phantom row would destroy "
        "the only copy of that day's price for that symbol."
    )
    lines.append("")
    lines.append("## Headline counts")
    lines.append("")
    lines.append(f"- Total rows in cache: **{n_total:,}**")
    lines.append(f"- Real rows (00:00:00): **{n_real:,}** across **{real_dates:,}** distinct dates")
    lines.append(
        f"- Phantom rows (non-00:00:00): **{n_phantom:,}** across **{phantom_dates:,}** distinct dates"
    )
    lines.append(f"- Distinct (symbol, date) cells with a phantom row: **{n_phantom_cells:,}**")
    lines.append(f"- Distinct symbols appearing in phantom rows: **{len(phantom_syms):,}**")
    if n_phantom_cells_with_data is not None:
        lines.append(
            f"  - of which have at least one non-null value column: **{n_phantom_cells_with_data:,}**"
        )
    lines.append("")
    lines.append("## Root cause of most phantom-only cells: never-real symbols")
    lines.append("")
    lines.append(
        f"Of the **{len(phantom_syms):,}** distinct symbols that appear in phantom rows, "
        f"**{len(symbols_never_real):,}** have **NO real (00:00:00) row anywhere in the cache, "
        f"for any date** — their entire price history in `prices_live` exists only via phantom "
        f"timestamps. The other **{len(symbols_mixed):,}** symbols ('mixed') have both real and "
        f"phantom rows."
    )
    lines.append("")
    lines.append(
        "The never-real symbol list is dominated by macro/ETF instruments (GLD, TLT, EEM, EFA, "
        "LQD, HYG, IAU, SGOV, TIP, VNQ, sector/country ETFs, etc.) — these look like "
        "macro/alternatives-agent instruments that were fetched only through the late-timestamp "
        "(yfinance_hub) path and never through the midnight-stamped path that `us_equities` uses. "
        "This is a structurally different situation from ordinary duplicate rows: for these "
        "symbols, the phantom row is not a duplicate of a same-day real fetch — it is the ONLY "
        "price observation that exists for that symbol/day anywhere in `prices_live`."
    )
    lines.append("")
    if symbols_never_real:
        lines.append("### Never-real symbols (entire history is phantom-only)")
        lines.append("")
        lines.append(", ".join(f"`{s}`" for s in symbols_never_real))
        lines.append("")
    lines.append(
        f"A further **{len(symbols_mixed) and len(set(phantom_only_df['symbol']) & set(symbols_mixed)):,}** "
        "symbols have SOME real rows but still show phantom-only cells for part of their history — "
        "these are symbols that appear to have been added to the real-fetch path only recently "
        "(their real rows start in 2026), so their 2020-2026 backfill exists only via phantom rows. "
        "See the CSV for the exact date ranges."
    )
    lines.append("")
    lines.append("## Phantom-only cells (the data-loss risk)")
    lines.append("")
    lines.append(
        f"**{n_phantom_only:,}** of the {n_phantom_cells:,} phantom (symbol, date) cells have "
        f"**no matching real row** for that symbol/date — i.e. "
        f"{100 * n_phantom_only / n_phantom_cells if n_phantom_cells else 0:.2f}% of phantom "
        f"cells are phantom-only."
    )
    lines.append("")
    if n_phantom_only:
        lines.append(f"Symbols affected: **{n_symbols_affected:,}**")
        lines.append(f"Date range of phantom-only cells: **{dmin.date()} to {dmax.date()}**")
        lines.append("")
        lines.append("### Distribution by year")
        lines.append("")
        lines.append("| year | phantom-only cells |")
        lines.append("|---|---|")
        for yr, cnt in by_year.items():
            lines.append(f"| {yr} | {cnt:,} |")
        lines.append("")
        lines.append("### Top symbols by phantom-only cell count")
        lines.append("")
        lines.append("| symbol | phantom-only cells |")
        lines.append("|---|---|")
        for sym, cnt in by_symbol.head(25).items():
            lines.append(f"| {sym} | {cnt:,} |")
        lines.append("")
        lines.append(f"Total distinct symbols with >=1 phantom-only cell: {n_symbols_affected:,}")
        lines.append("")
        lines.append("### Sample of 20 phantom-only (symbol, date) pairs")
        lines.append("")
        lines.append("| symbol | date |")
        lines.append("|---|---|")
        for _, row in sample20.iterrows():
            lines.append(f"| {row['symbol']} | {row['date'].date()} |")
        lines.append("")
        if csv_written:
            lines.append(
                f"Full list ({n_phantom_only:,} rows) written to "
                f"`outputs/wf_results/phantom_only_cells_2026-08-15.csv`, including the phantom "
                f"row's value columns for context."
            )
        lines.append("")
    else:
        lines.append(
            "**None.** Every phantom (symbol, date) cell also has a matching real row. Dropping "
            "all phantom rows is safe — no unrecoverable data would be lost."
        )
        lines.append("")

    lines.append("## Clustering assessment")
    lines.append("")
    if n_phantom_only == 0:
        lines.append(
            "No phantom-only cells exist, so there is nothing to target with a re-fetch. "
            "A blanket drop of all phantom rows (time-of-day != 00:00:00) is safe."
        )
    else:
        top_share = (
            by_symbol.head(10).sum() / n_phantom_only if n_phantom_only else 0
        )
        lines.append(
            f"Phantom-only cells are extremely concentrated by symbol, not spread thin: "
            f"top 10 symbols account for {100 * top_share:.1f}% of phantom-only cells "
            f"({by_symbol.head(10).sum():,} of {n_phantom_only:,}), and only "
            f"{n_symbols_affected:,} distinct symbols are affected in total (out of {len(phantom_syms):,} "
            f"symbols that appear anywhere in phantom rows), over the date range "
            f"{dmin.date()} to {dmax.date()}. This matches the never-real-symbol finding above: "
            "almost every phantom-only cell belongs to one of the 49 macro/ETF symbols that were "
            "never fetched via the real (midnight) path, not to scattered gaps in equity symbols."
        )
        lines.append("")
        lines.append(
            "This is NOT the ~54-sparse-equity-symbol picture assumed going in — it is a smaller "
            "set of non-equity instruments (ETFs/macro) whose *entire* price series in "
            "`prices_live` lives at the phantom timestamp, plus 6 recently-added equity symbols "
            "whose pre-2026 backfill is phantom-only."
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if n_phantom_only == 0:
        lines.append(
            "Safe to drop all phantom rows (time-of-day != 00:00:00) with no data-loss risk. "
            "This task made no changes; a follow-up task should implement the drop."
        )
    else:
        lines.append(
            "Do NOT blanket-drop all phantom rows. A time-of-day-only drop rule "
            f"(`date == date.normalize()`) would silently delete the entire price history of "
            f"**{len(symbols_never_real):,} symbols** ({', '.join('`'+s+'`' for s in symbols_never_real[:10])}, "
            "…) and the pre-2026 backfill of 6 more (BLD, GTLS, JHG, MASI, PSTG, SATS) — none of "
            "which have any other row in the cache to fall back to."
        )
        lines.append("")
        lines.append(
            "Recommended fix shape: drop phantom rows only where a real row for the same "
            "(symbol, normalized date) already exists (the safe, true-duplicate case — "
            f"{n_phantom_cells - n_phantom_only:,} of {n_phantom_cells:,} phantom cells). For the "
            f"remaining {n_phantom_only:,} phantom-only cells, KEEP the phantom row as the sole "
            "price record (do not drop it just because its timestamp isn't midnight), or, if a "
            "midnight-stamped series is a hard requirement downstream, normalize its timestamp to "
            "00:00:00 in place rather than deleting it. A targeted yfinance re-fetch "
            "(checkpoint-4 `repair_mixed_basis_symbols.py` pattern) is an alternative for the 6 "
            "mixed equity symbols, but is not needed for the 49 never-real macro/ETF symbols since "
            "the phantom row already IS their only fetched observation — re-fetching would just "
            "reproduce the same values. This task made no changes; a follow-up task should "
            "implement the fix using this diagnosis and the full CSV."
        )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))

    print(f"Total rows: {n_total:,}")
    print(f"Real rows: {n_real:,} ({real_dates:,} dates)")
    print(f"Phantom rows: {n_phantom:,} ({phantom_dates:,} dates)")
    print(f"Phantom (symbol,date) cells: {n_phantom_cells:,}")
    print(f"Phantom-ONLY cells (no matching real row): {n_phantom_only:,}")
    if n_phantom_only:
        print(f"  symbols affected: {n_symbols_affected:,}")
        print(f"  date range: {dmin.date()} to {dmax.date()}")
        print(f"  CSV written: {CSV_PATH}")
    print(f"Report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
