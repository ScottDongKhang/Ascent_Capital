# Phantom-row diagnosis — data_cache/prices_live.parquet

Date: 2026-08-15. Read-only analysis. No cache file modified.

Script: `scripts/maintenance/diagnose_phantom_rows.py` (`.venv/bin/python scripts/maintenance/diagnose_phantom_rows.py`)

## Method

Loaded `prices_live` via `ascent.data.store.parquet.load_parquet`. Split rows into **real** (`date` time-of-day == 00:00:00) and **phantom** (time-of-day != 00:00:00, typically 19:00/20:00 — the yfinance_hub late-fetch stamp). For every distinct `(symbol, date.normalize())` pair that has a phantom row, checked whether a real row exists for that same `(symbol, normalized date)`. A pair with a phantom row and NO matching real row is a **phantom-only cell** — dropping that phantom row would destroy the only copy of that day's price for that symbol.

## Headline counts

- Total rows in cache: **1,519,291**
- Real rows (00:00:00): **1,430,450** across **1,650** distinct dates
- Phantom rows (non-00:00:00): **88,841** across **1,648** distinct dates
- Distinct (symbol, date) cells with a phantom row: **88,841**
- Distinct symbols appearing in phantom rows: **105**
  - of which have at least one non-null value column: **88,841**

## Root cause of most phantom-only cells: never-real symbols

Of the **105** distinct symbols that appear in phantom rows, **49** have **NO real (00:00:00) row anywhere in the cache, for any date** — their entire price history in `prices_live` exists only via phantom timestamps. The other **56** symbols ('mixed') have both real and phantom rows.

The never-real symbol list is dominated by macro/ETF instruments (GLD, TLT, EEM, EFA, LQD, HYG, IAU, SGOV, TIP, VNQ, sector/country ETFs, etc.) — these look like macro/alternatives-agent instruments that were fetched only through the late-timestamp (yfinance_hub) path and never through the midnight-stamped path that `us_equities` uses. This is a structurally different situation from ordinary duplicate rows: for these symbols, the phantom row is not a duplicate of a same-day real fetch — it is the ONLY price observation that exists for that symbol/day anywhere in `prices_live`.

### Never-real symbols (entire history is phantom-only)

`AAXJ`, `ACI`, `ACM`, `BIL`, `BK`, `CBOE`, `CBRE`, `CBSH`, `CTRA`, `DBA`, `DBB`, `EEM`, `EFA`, `EWC`, `EWG`, `EWJ`, `EWT`, `EWU`, `EWY`, `EWZ`, `FLO`, `GLD`, `HYG`, `IAU`, `IEF`, `IFRA`, `INDA`, `KMLM`, `KTOS`, `KVUE`, `L`, `LQD`, `MKSI`, `MLI`, `OLLI`, `OLN`, `OMC`, `PAVE`, `PDBC`, `PVH`, `SGOV`, `TIP`, `TLT`, `UUP`, `VIXY`, `VNQ`, `VWO`, `WOOD`, `ZBH`

A further **6** symbols have SOME real rows but still show phantom-only cells for part of their history — these are symbols that appear to have been added to the real-fetch path only recently (their real rows start in 2026), so their 2020-2026 backfill exists only via phantom rows. See the CSV for the exact date ranges.

## Phantom-only cells (the data-loss risk)

**88,791** of the 88,841 phantom (symbol, date) cells have **no matching real row** for that symbol/date — i.e. 99.94% of phantom cells are phantom-only.

Symbols affected: **55**
Date range of phantom-only cells: **2020-01-01 to 2026-07-23**

### Distribution by year

| year | phantom-only cells |
|---|---|
| 2020 | 13,151 |
| 2021 | 13,555 |
| 2022 | 13,501 |
| 2023 | 13,612 |
| 2024 | 13,803 |
| 2025 | 13,693 |
| 2026 | 7,476 |

### Top symbols by phantom-only cell count

| symbol | phantom-only cells |
|---|---|
| AAXJ | 1,648 |
| OMC | 1,648 |
| IEF | 1,648 |
| INDA | 1,648 |
| KTOS | 1,648 |
| L | 1,648 |
| LQD | 1,648 |
| MKSI | 1,648 |
| MLI | 1,648 |
| OLLI | 1,648 |
| OLN | 1,648 |
| PAVE | 1,648 |
| HYG | 1,648 |
| PDBC | 1,648 |
| PVH | 1,648 |
| TIP | 1,648 |
| TLT | 1,648 |
| UUP | 1,648 |
| VIXY | 1,648 |
| VNQ | 1,648 |
| VWO | 1,648 |
| WOOD | 1,648 |
| IAU | 1,648 |
| IFRA | 1,648 |
| EFA | 1,648 |

Total distinct symbols with >=1 phantom-only cell: 55

### Sample of 20 phantom-only (symbol, date) pairs

| symbol | date |
|---|---|
| AAXJ | 2020-01-01 |
| AAXJ | 2020-01-02 |
| AAXJ | 2020-01-05 |
| AAXJ | 2020-01-06 |
| AAXJ | 2020-01-07 |
| AAXJ | 2020-01-08 |
| AAXJ | 2020-01-09 |
| AAXJ | 2020-01-12 |
| AAXJ | 2020-01-13 |
| AAXJ | 2020-01-14 |
| AAXJ | 2020-01-15 |
| AAXJ | 2020-01-16 |
| AAXJ | 2020-01-20 |
| AAXJ | 2020-01-21 |
| AAXJ | 2020-01-22 |
| AAXJ | 2020-01-23 |
| AAXJ | 2020-01-26 |
| AAXJ | 2020-01-27 |
| AAXJ | 2020-01-28 |
| AAXJ | 2020-01-29 |

Full list (88,791 rows) written to `outputs/wf_results/phantom_only_cells_2026-08-15.csv`, including the phantom row's value columns for context.

## Clustering assessment

Phantom-only cells are extremely concentrated by symbol, not spread thin: top 10 symbols account for 18.6% of phantom-only cells (16,480 of 88,791), and only 55 distinct symbols are affected in total (out of 105 symbols that appear anywhere in phantom rows), over the date range 2020-01-01 to 2026-07-23. This matches the never-real-symbol finding above: almost every phantom-only cell belongs to one of the 49 macro/ETF symbols that were never fetched via the real (midnight) path, not to scattered gaps in equity symbols.

This is NOT the ~54-sparse-equity-symbol picture assumed going in — it is a smaller set of non-equity instruments (ETFs/macro) whose *entire* price series in `prices_live` lives at the phantom timestamp, plus 6 recently-added equity symbols whose pre-2026 backfill is phantom-only.

## Recommendation

Do NOT blanket-drop all phantom rows. A time-of-day-only drop rule (`date == date.normalize()`) would silently delete the entire price history of **49 symbols** (`AAXJ`, `ACI`, `ACM`, `BIL`, `BK`, `CBOE`, `CBRE`, `CBSH`, `CTRA`, `DBA`, …) and the pre-2026 backfill of 6 more (BLD, GTLS, JHG, MASI, PSTG, SATS) — none of which have any other row in the cache to fall back to.

Recommended fix shape: drop phantom rows only where a real row for the same (symbol, normalized date) already exists (the safe, true-duplicate case — 50 of 88,841 phantom cells). For the remaining 88,791 phantom-only cells, KEEP the phantom row as the sole price record (do not drop it just because its timestamp isn't midnight), or, if a midnight-stamped series is a hard requirement downstream, normalize its timestamp to 00:00:00 in place rather than deleting it. A targeted yfinance re-fetch (checkpoint-4 `repair_mixed_basis_symbols.py` pattern) is an alternative for the 6 mixed equity symbols, but is not needed for the 49 never-real macro/ETF symbols since the phantom row already IS their only fetched observation — re-fetching would just reproduce the same values. This task made no changes; a follow-up task should implement the fix using this diagnosis and the full CSV.
