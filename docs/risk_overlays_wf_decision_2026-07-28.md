# Risk overlays — walk-forward comparison and enable decisions (2026-07-28)

Decision record for the two remaining 2026-07-27 risk plans. **Both treatments were
rejected. Both overlays stay disabled / defaulted to legacy behaviour, so live behaviour
is unchanged by this work.**

Plans: `docs/superpowers/plans/2026-07-27-strategy-own-vol-targeting.md`,
`docs/superpowers/plans/2026-07-27-momentum-crash-indicator.md`.

Kept as a standalone file (not folded into `CLAUDE.md` or `docs/session_log_archive.md`)
because both of those had uncommitted work-in-progress on `main` at the time, including a
policy rewrite moving performance numbers out of `CLAUDE.md`.

---

## What was implemented

**Plan 3 — strategy-own volatility targeting** (Barroso & Santa-Clara 2015; Moreira & Muir
2017). Tasks 1-4. `realized_vol_scale()` made generic over any return series;
`vol_target_scale()` retained as a bit-identical SPY wrapper; causal
`strategy_return_proxy()` (`r(t) = Σ w(t-1)·ret(t)`); `vol_reference` + `close` added to
`apply_exposure_overlays()` with fail-open fallback to `"spy"`.

**Plan 4 — momentum-crash indicator** (Daniel & Moskowitz 2016). Tasks 1-3.
`momentum_crash_scale()` fires ×0.50 when the trailing ~504d SPY return < 0 **and** the
trailing 21d return > 0; composed as a third multiplicative overlay; wired through both
production and research with an `inspect.getsource` parity guard so the two cannot silently
diverge.

### A bug the plan did not anticipate

The WF `_apply_vol_target` receives weights indexed at **rebalance dates only**, and
`strategy_return_proxy` reindexes prices onto the weights index. The plan's literal diff
therefore computes ~10-day returns and annualizes them by `sqrt(252)` — a ~3× volatility
overstatement that pins the scale near the 0.25 floor. Measured on a 15%-vol book:
**mean scale 0.466 (wrong) vs 0.95 (correct)**. The book would have been silently ~halved
by a pure grid artifact. Fixed by forward-filling the held book onto the daily close panel
first. Guarded by `tests/portfolio/test_exposure_strategy_vol.py::TestProxyNeedsADailyGrid`,
one of whose tests asserts the hazard still exists on a sparse grid so the guard cannot be
silently removed. Production was never affected (it passes a daily index).

---

## The walk-forward comparison

Three configurations run in **one process** so prices load once and nothing but the overlay
config differs. Cache `prices_live_clean_refetch` (936 symbols, 0 duplicates),
**21 folds, 1134 OOS days**. ~92 min total.

Artifacts: `outputs/wf_results/wf_report_{cashfix,stratvol,crashoverlay}_2026-07-27.json`
(+ matching `wf_equity_*.csv`, `wf_folds_*.json`).

| metric | baseline (`spy`) | stratvol (`strategy`) | crashoverlay (on) |
|---|---|---|---|
| Sharpe | 0.0856 | **−0.2143** | 0.0856 |
| CAGR | 3.91% | 0.34% | 3.91% |
| volatility | 12.76% | 12.96% | 12.76% |
| max drawdown | −21.42% | **−28.04%** | −21.42% |
| beta | 0.533 | 0.587 | 0.533 |
| regression alpha | −3.04% | −6.90% | −3.04% |
| win rate | 52.20% | 51.68% | 52.20% |
| WFE | 1.209 | 0.172 | 1.209 |
| vol scale mean | 0.8819 | 0.9247 | 0.8819 |
| vol scale min | 0.2500 | 0.2843 | 0.2500 |
| % dates at vol floor | 0.31% | 0.00% | 0.31% |
| crash cut dates | 0 | 0 | **0** |

> **⚠ These figures are EXPERIMENT-LOCAL. They do not supersede, confirm, or refute the
> canonical VERIFIED walk-forward number.** The canonical run zeroed the
> `llm_fundamental` and `narrative` sleeves; this comparison used the default "current"
> sleeve config with both loaded. Absolute Sharpe/CAGR are therefore **not comparable** to
> the canonical artifact — only the three columns are comparable *to each other*. Do not
> cite these as the system's walk-forward performance. `CURRENT_VERIFIED_NUMBERS.md`
> remains the only citable source.

---

## Decision — Plan 3: do NOT switch `vol_target_reference` to `"strategy"`

Gate required Sharpe +0.05 **and** improved max drawdown. Actual: **Sharpe −0.30 and
drawdown 6.6pp worse**. Rejected.

**Why, and it is instructive.** The book's own realized volatility (~12.8%) sits *below*
the 15% target, so scaling by its own vol **de-risks less** than scaling by SPY's: mean
scale 0.925 vs 0.882, min 0.284 vs 0.250, and 0% vs 0.31% of dates at the floor. The
overlay levers toward 1.0 and therefore carries *more* exposure into drawdowns.

Barroso & Santa-Clara's mechanism presumes the factor's own volatility **exceeds** the
target — true for a standalone long/short momentum factor, false for this book, which is
already defensive (~22% non-equity sleeves plus the 200MA cut). The premise does not hold,
so the result is not a surprise in hindsight, and the machinery stays in the tree behind a
config flag rather than being deleted.

## Decision — Plan 4: do NOT enable. Two independent reasons; the second is decisive

**(a) The firing audit is one macro episode, not many events.** On full history the crash
state occurs on 64/1395 OOS dates (4.6%) — above the plan's >5 threshold — but every one
falls inside a single 9-month window (post-2022-bear recovery, **2023-04-11 → 2024-01-08**)
across 15 contiguous runs, largest 14 days (22% of firings). That is one macro episode
fragmented, so the plan's "improvement is not driven by a single episode" condition is
unsatisfiable on this OOS window. The timing is economically *correct* — it fires in
exactly the post-bear rebound Daniel & Moskowitz describe.

**(b) The overlay is STRUCTURALLY INERT inside this walk-forward harness.** Each fold hands
the strategy only `is_days=252 + oos_days=63` ≈ **320 trading days**, while
`momentum_crash_scale` requires `bear_lookback + 1 = 505` rows strictly before each decision
date. It therefore returns 1.0 unconditionally: `crash_cut_n = 0` across 8,656 evaluations,
and the crashoverlay column is byte-identical to baseline.

**The zero-firing result is NOT evidence that the overlay is harmless — it is evidence that
this harness cannot test it.** Validating a 504-day-lookback rule requires a
`WindowGenerator` with a materially longer IS window, or passing full-history context to the
strategy for the lookback while keeping fold isolation for the signal. Recorded as an open
item rather than papered over.

---

## Production data-integrity regression found along the way

The first WF attempt burned 76 minutes and died in `engine._finalise` with
`cannot reindex on an axis with duplicate labels`. Cause: production `prices_live` had
re-accumulated **322,868 duplicate (symbol, trading-day) rows** since the 2026-06-24
collapse, inflating the date axis to 47 overlapping folds whose stitched OOS index carried
duplicate labels. `ascent/main.py` dedupes on read so **live trading was shielded**; the WF
framework does not, which is why only research broke.

**Two root causes, both fixed (TDD):**

1. `save_parquet` ran its dedup block only inside `if path.exists()`, so the first write for
   a cache name persisted with **zero** dedup, and every later append only compared new rows
   against that already-corrupt baseline. Dedup now always runs.
2. `_calendar_day_key` normalized in the stored timezone, so a bar stamped 19:00/20:00 NY —
   which is really the **next** trading day's close; the cache holds a `2020-01-01 19:00`
   row and 2020-01-01 is a market holiday — keyed to the prior day and never collided with
   the same trading day fetched at 00:00. Bars at/after 17:00 local now roll the key forward
   one day.

Verified empirically before accepting the fix: **PSTG is the only symbol carrying both stamp
types, and all 351 rolled matches have byte-identical closes**, proving the rollover creates
exactly the right collisions and that `keep="last"` discards nothing.

### Repair COMPLETED (2026-07-28)

Both steps ran against production `data_cache/prices_live.parquet`, each writing a verified
backup first. Scripts kept at `scripts/maintenance/`.

1. **Basis repair** (`repair_mixed_basis_symbols.py --apply`): re-fetched CRWD, MIDD, SPGI,
   HON on the split-only basis, 1,648 rows each, **0 implausible jumps**. Conflicting groups
   **1,419 → 0**. Backup `prices_live.pre_basis_repair.20260728-162726.bak.parquet`.
   (CRWD's cached max was a bogus $782 versus a true split-adjusted $210.73 — confirming the
   cache held rows unadjusted for its 4:1 split.)
2. **Duplicate collapse** (`collapse_prices_live.py --apply`): **1,839,056 → 1,517,608 rows**
   (321,448 removed) with key coverage identical (1,517,608 → 1,517,608), so only redundant
   rows went, never coverage. Backup `prices_live.pre_dedup2.20260728-162756.bak.parquet`.

Post-repair verification: 938 symbols, 1,648 trading days, **0 duplicates**, 0 null closes,
and the WF fold axis back to **22 folds (was 47)** — the original crash condition is gone.

**Still open, deliberately not changed: `CHRD`.** It carries a 257× jump on 2020-11-20 —
the documented irreparable source-side ticker-reuse history (Chord Energy reusing Oasis
Petroleum's ticker), which is why the 2026-06-22 clean re-fetch dropped it (936 vs 938
symbols). Dropping a universe symbol has portfolio consequences, unlike a pure duplicate
collapse, so it was left for a deliberate decision. Live risk is low: the artifact sits in
2020, outside today's 252-day momentum window, and the 2026-06-24 IC winsorization already
prevents one symbol from gating a sleeve. It WILL distort any backtest spanning 2020.

### Why the collapse was blocked until the re-fetch

The collapse script's safety guard **refused to write**: 1,419 (symbol, trading-day) groups
hold **conflicting close values** — not duplicates but two different **adjustment bases**.
Affected symbols and ratios:

| symbol | max/min ratio | reading |
|---|---|---|
| CRWD | exactly **4.0** | its 4:1 split — one row split-adjusted, one not |
| HON, SPGI, MIDD | 1.05 – 1.24 | cumulative dividend adjustment (TR vs split-only) |

~355 days each. A blind `keep="last"` would have kept the split-adjusted row for one symbol
and the unadjusted row for another — silently mixing bases. Repair follows the 2026-06-24
KLAC precedent: fresh-fetch those 4 symbols on the **split-only** (`auto_adjust=False`)
basis that production `prices_live` is on, and replace them in place. No total-return swap,
so no discontinuity against the other 934 symbols.

---

## Open items

- Validating the crash overlay needs a longer-IS `WindowGenerator`; it cannot be tested as
  currently configured.
- `CHRD` still holds an irreparable 257× ticker-reuse artifact in 2020 (see above). Decide
  whether to drop it from the universe as the clean re-fetch did.
- `feature/risk-management` has `main` merged into it and is ready to merge out. It was
  deliberately NOT merged into `main` while `main` carried uncommitted WIP.
