# Production Bugfixes — Design Spec

**Sub-project 1c** — fixes two real production bugs discovered during the proof-audit extension
(sub-project 1b, merged 2026-08-12) that were left unmeasured (not proven negative) in the
scorecard: `earnings`/`analyst` sleeves and all three specialist agents
(`macro_agent`/`international_agent`/`alternatives_agent`). Unlike sub-projects 1/1b, this one
**touches shared production code** — `ascent/features/feature_defs.py` and
`ascent/data/store/parquet.py` — since both bugs live there, not in the proof-audit tool.

**Context:** live paper trading (`com.ascentcapital.eod`/`.heartbeat`) has been paused since
2026-08-12 for the duration of the strip-down/rebuild effort, so there is no live execution risk
from these changes today — but the fixed code will be there when trading resumes, so correctness
matters as much as if it were live.

## Bug 1: `signal_date` not normalized to midnight

`ascent/features/feature_defs.py`'s `build_earnings_panel` (~line 299), `build_analyst_panel`
(~line 356), and `build_insider_panel` (~line 455) all derive `signal_date` via `tz_localize(None)`
without a following `.dt.normalize()` — unlike their siblings `build_options_panel` (line 413) and
`build_short_panel` (line 505), which already do `.dt.normalize().dt.tz_localize(None)`. Price
data is reindexed against a midnight-normalized index, so a `signal_date` carrying a nonzero
time-of-day never matches, silently producing all-NaN panels.

**Fix:** add `.dt.normalize()` to the same call chain in all three functions, matching the
already-correct sibling pattern exactly. No semantic loss — earnings/analyst/insider signals are
daily-granularity events.

**Test coverage:** `tests/test_earnings_alpha.py`, `tests/test_analyst_alpha.py` exist but
construct `signal_date` as already-midnight timestamps, so they wouldn't have caught this — add
one test per function injecting a non-midnight timestamp (e.g. `14:30:00`) and asserting the
panel still aligns. `build_insider_panel` has no dedicated test file (only indirect coverage via
`tests/test_phase6_signals.py`) — add one there or create a focused test file, matching whichever
convention the existing insider tests (if any beyond phase6) already use.

## Bug 2: `save_parquet` drops the index unconditionally

`ascent/data/store/parquet.py::save_parquet` (line ~143) calls `df.to_parquet(path,
index=False)` with no branch on index type. `load_parquet` (line ~147) does a plain
`pd.read_parquet(path)` with no index-restoration logic — this isn't unexercised code, it
genuinely doesn't exist. Three wide-format caches (`prices_macro`, `prices_international`,
`prices_alternatives` — date lives only in the index) are corrupted by every save: confirmed
`index_columns=[]` in parquet metadata, `RangeIndex` on reload, implausible row counts
(176k/151k/150k for 9-13 symbols). Already documented as a CLAUDE.md gotcha.

**Fix:** `save_parquet` branches on `isinstance(df.index, pd.DatetimeIndex)` — write `index=True`
with an explicit `index_label` in that case, `index=False` otherwise (preserving current
behavior for every other cache). `load_parquet` gains matching restore logic: if the persisted
index column is present, `set_index()` it back on read; otherwise behaves exactly as today.

**Call-site audit (confirmed by research, 7 sites beyond the 3 broken ones):**
`ascent/data/hub.py` (`prices_live`, `macro_live`), `ascent/data/ingest/{fundamentals,insider,
analyst,options,earnings,supplementary}.py` — all long-format with a default `RangeIndex` or
explicit `date`/`symbol` columns. None rely on the index being dropped; the fix's branch leaves
all of them byte-identical to current behavior.

**Test coverage:** add a round-trip test to whatever test file covers `ascent/data/store/
parquet.py` (find or create it) — save a DataFrame with a `DatetimeIndex`, reload, assert the
index survives; save a DataFrame with a `RangeIndex` (current-behavior case), reload, assert
still index-free, matching today's behavior exactly (regression guard).

## Cache repair (not a code fix — a data operation)

`_fetch_macro_prices`/`_fetch_international_prices`/`_fetch_alternatives_prices`
(`agents/macro_agent.py`, `international_agent.py`, `alternatives_agent.py`) all call
`yf.download(symbols, start="2020-01-01", ...)` — a full historical re-fetch, not
incremental/append-only, confirmed by research. Once both code fixes land, force-refresh the
three corrupted caches via a real network call to Yahoo Finance (confirmed reachable in this
session) — this is the only way to recover the lost date information; the corrupted parquet
files themselves cannot be repaired after the fact, since the index data is gone from disk, not
just misread.

## Verification

After both fixes and the cache repair: re-run `PYTHONPATH=. .venv/bin/python
scripts/run_proof_audit.py` and report the updated scorecard. Expect `earnings`/`analyst` to
move off `INSUFFICIENT_DATA` (real KEEP/CUT or a legitimate sparse-data reason), and
`macro_agent`/`international_agent`/`alternatives_agent` to be scored against their real,
distinct universes for the first time in this effort.

## Explicitly out of scope

- `insider`'s existing `INSUFFICIENT_DATA`/CUT status (already scoring in sub-project 1b) — Bug
  1's `build_insider_panel` fix may change its numbers, but no new insider-specific work here.
- Any other `save_parquet` call site not confirmed broken by the research audit above.
- Sub-project 2 (target architecture) — this only extends the evidentiary scorecard it will read.
- Resuming `com.ascentcapital.eod`/`.heartbeat` — stays paused.
