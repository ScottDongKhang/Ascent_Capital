# Disk hygiene pass — 2026-08-15

Archival only. Nothing deleted; nothing under `ascent/`, `agents/`, `debate/`, `analyst/`, or
any `.py` file was touched. No git commits (both directories are gitignored).

## Task 1 — `data_cache/` backup cruft

### Inventory (files/dirs over 5MB, from `du -sh data_cache/* | sort -rh`, before cleanup)

| Path | Size | Verdict |
|---|---|---|
| `_corrupt_backup_20260622-222216/` | 184M | ARCHIVED — pre-repair full copy from the 2026-06-22 corrupt-data incident |
| `prices_live.pre_klac_fix.20260624-160508.bak.parquet` | 183M | ARCHIVED — pre-repair backup, KLAC ×10 fix (`eb4b923`) |
| `prices_live.pre_dedup.20260624-174644.bak.parquet` | 118M | ARCHIVED — pre-repair backup, 321k dup-row collapse (`4c276b6`) |
| `prices_live_clean_refetch.parquet` | 81M | **LEFT ALONE** — actively used by research scripts (see below), not a repair backup despite the name |
| `prices_live.pre_dedup2.20260728-162756.bak.parquet` | 68M | ARCHIVED — pre-repair backup, `collapse_prices_live.py` dedup2 pass |
| `prices_live.pre_basis_repair.20260728-162726.bak.parquet` | 68M | ARCHIVED — pre-repair backup, `repair_mixed_basis_symbols.py` |
| `prices_live.parquet.pre_phantom_repair_bak` | 66M | ARCHIVED — pre-repair backup, phantom-row fix (`9fd74ea`/`9f145fc`/`0d1496e`/`e36442d`) |
| `prices_live.parquet` | 66M | **LEFT ALONE** — active live cache, mtime today (2026-08-15 12:03), newest write in the directory |
| `factor_loadings.parquet` | 56M | **LEFT ALONE** — not a backup, no `.bak`/`.pre_*`/`_corrupt_backup` naming |
| `prices_simulated.parquet` | 9.9M | **LEFT ALONE** — legitimate named cache variant per CLAUDE.md provenance rule |
| `prices_live_fallback_simulated.parquet` | 9.6M | **LEFT ALONE** — same |

Plus 4 tiny (~1KB each) `earned_authority.json.bak-*` generation backups, all predating the
current `earned_authority.json` (mtime 2026-08-13) — archived for consistency even though under
the 5MB threshold, since they unambiguously match the backup pattern.

### Method

For every `.bak`/`.pre_*repair*`/`_corrupt_backup*` candidate: read `CLAUDE.md`'s Data/caching
section, cross-referenced `git log --oneline | grep -iE "phantom|dedup|klac|basis_repair|corrupt"`
against each file's mtime (`stat -f "%Sm"`), and confirmed the current `prices_live.parquet`
postdates every candidate and was independently re-validated by the 2026-08 phantom-row fix
commit chain (walk-forward re-validated in `e36442d`, recurrence-guard follow-up in `0d1496e`).

**Important catch**: `prices_live_clean_refetch.parquet` looks backup-shaped but is not one —
`grep -rn "clean_refetch" --include="*.py" .` shows it is actively read by
`scripts/research/wf_overlay_comparison.py` (as `WF_CACHE` env default) and all three
`scripts/edge_tests/edge_test*.py` files. Left untouched.

### Actions taken

Created `data_cache/_archive_2026-08-15/`, moved 6 large backup files/dirs + 4 small
`earned_authority.json.bak-*` files into it (plain `mv`, filenames preserved), wrote
`data_cache/_archive_2026-08-15/MANIFEST.md` documenting each file's original size, what it
backs up, and why it's safe. Full detail in that manifest.

### Sizes

- `data_cache/` total: **922M before -> 922M after** (unchanged — `mv` within the same
  filesystem doesn't change the parent directory's total; nothing was deleted).
- Isolated, now-recoverable space in `data_cache/_archive_2026-08-15/`: **686M**
  (includes the 6 large backups plus the 4 small `earned_authority.json.bak-*` files).
- Active (non-archive) `data_cache/` content: **~236M** (922M − 686M).
- The archive directory is now a single, clearly-labeled unit that can be moved to cold
  storage or deleted in a later, separate pass without re-deriving which files were safe.

## Task 2 — `logs/llm_fundamental_signals.jsonl`

### Confirmation of consumer

`grep -rn "llm_fundamental_signals" --include="*.py" .` — only one production call site:
`ascent/alpha/llm_fundamental.py:230`, inside `llm_fundamental_alpha()`, which appends one line
whenever it computes non-empty scores. That function is invoked from `ascent/alpha/stack.py:281-286`,
gated by `alpha_weights.get("llm_fundamental", 0) > 0`. `DEFAULT_ALPHA_WEIGHTS` (both
`ascent/alpha/stack.py` and `ascent/research/self_improve.py`) is `{"meanrev": 0.50, "statarb":
0.50}` — no `llm_fundamental` key — and `data_cache/active_alpha_config.json` (which could
override) does not currently exist. **Confirmed: the sleeve does not run in the production
daily pipeline.**

Note: this `llm_fundamental` sleeve is a **different** sleeve from the one CLAUDE.md integrity
constraint #7 calls "disabled" — that constraint refers to `ascent/alpha/fundamental.py`'s
`fundamental_alpha()` (gated separately at `stack.py:258`). Both are zero-weighted today by the
same default-weights mechanism, but they are separate code paths; worth not conflating in
future doc edits.

### Live-write status — flagged finding, not a simple "dormant" case

At capture time (2026-08-15 ~17:54 PT) the file's mtime was **2 minutes old**, despite the
production weight being 0. Cause: a concurrent `pytest tests/ -q` run (PID 86020, confirmed via
`ps aux`, exited by the time of archiving) was exercising tests that call
`llm_fundamental_alpha()` directly with nonzero weight (e.g. `tests/test_llm_fundamental_alpha.py`,
`tests/alpha/test_stack_weights.py`), and the log path in `ascent/alpha/llm_fundamental.py:230`
is hardcoded to the real production path rather than a mockable/injected one — so **test runs
pollute the production log**, separately from whatever the scheduled daily pipeline does. This
means the underlying LLM calls do still fire sometimes (real API cost), just not from the
scheduled daily run — a distinct, separate concern from the sleeve's zero production weight.
Confirmed no writer was active before archiving (re-checked `ps aux` and `lsof` — clean).

### Action taken

Moved `logs/llm_fundamental_signals.jsonl` (30MB, 18,006 lines) to
`logs/_archive_2026-08-15/llm_fundamental_signals.jsonl.gz` (`gzip -9`, not deleted — kept for
possible future re-measurement if the sleeve is revisited). Wrote
`logs/_archive_2026-08-15/MANIFEST.md` with full detail including the test-pollution finding.
Compression: 30MB -> 163KB (~99.5% reduction). If tests append to the original path again,
Python's `open(path, "a")` will silently recreate a small file there — expected behavior, not a
failure of this cleanup.

### Sizes

- `logs/` total: **33MB before -> 3.4MB after**.
- `logs/_archive_2026-08-15/`: 168KB (compressed archive).

## Uncertain / left alone out of caution

None found beyond what's listed above as intentionally left alone. Everything moved had an
unambiguous, verifiable current/superseding counterpart.
