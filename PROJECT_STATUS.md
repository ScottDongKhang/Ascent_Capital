# Ascent Capital — Project Status & Architecture

**Written 2026-07-28.** Produced by six parallel read-only audits of the source tree, not from
documentation or memory. Every claim traces to a file, line, or artifact that was actually read.
Nothing in the repo was modified to produce this.

**Read this section first if you read nothing else.**

---

## 0. At a glance

| Question | Answer |
|---|---|
| Is it running? | Yes, but it went dark for 19 trading days (2026-06-30 → 2026-07-24) and missed 2 scheduled rebalances. Recovered 2026-07-27. |
| Live equity | **$104,640.21** (Alpaca paper), 22 positions |
| Live return | **+3.79%** since 2026-04-01 vs **SPY +12.67%** → **behind by ~8.9pp** |
| Backtest (the citable number) | **Sharpe 0.41, CAGR +10.3%, MaxDD −32.9%, beta 0.73** — with `WFE −0.65 (overfit)` disclosed, and two newly-found qualifications (§2.1) |
| AI layer | Level 1 "Analyst", 5% budget. **Structurally unable to be promoted** (return buffers empty, `n_decisions_evaluated = 0`) |
| Codebase | ~39k LoC in `ascent/`, ~20.5k LoC of tests (1,202 tests), 1,469 Python files |
| Next rebalance | **2026-08-05** |
| Biggest immediate risk | **FOMC 2026-07-29 (tomorrow)** with the hedge leg accidentally cut ~1.9pp below what the judge ordered |

**One-sentence summary.** The quant construction machinery is genuinely well built and honestly
measured; what is weak is the *plumbing around it* — data freshness, a large amount of implemented
risk machinery that is wired to nothing, an alerting layer that cannot fire, and a
measurement layer that can't currently grade the AI it exists to grade.

---

## 1. Urgent — things I would act on this week

Ordered by consequence, not by effort.

### 1.1 Live Alpaca credentials are committed on a public GitHub repo

`scripts/backfill_holdings_log.py:16-17` contains a hardcoded key/secret pair, present on
`origin/main` and on the `public-showcase` branch. I confirmed `gh api repos/ScottDongKhang/Ascent_Capital`
returns `public`. A **second, different** pair is in history in `scripts/run_eod.sh` across at
least 5 commits.

`run_eod.sh:36-37` already carries the note "previously hardcoded here — rotate those paper-trading
credentials" — so this remediation was started and **missed `backfill_holdings_log.py`**.

Paper account, so the blast radius is bounded. Still: rotate both pairs, and rewrite history.

### 1.2 The book goes into tomorrow's FOMC with less hedge than the judge ordered

On 2026-07-27 the judge issued `reduce_size` and said, verbatim:

> "UUP and TLT rank higher on priority, but both are the hedge leg and cutting insurance
> 48 hours before the catalyst it exists to hedge is poor timing — the bull wins that
> specific exchange."

What actually executed (comparing `verdict_2026-07-27.json → portfolio_state.weights` against
the executed target in `logs/eod_log.jsonl`):

| Symbol | Pre | Executed | Δ | vs authority cap (1.0pp) |
|---|---|---|---|---|
| UUP | 8.70% | 6.79% | **−1.91pp** | ~2x, and **not in `position_changes` at all** |
| TLT | 8.25% | 6.33% | **−1.92pp** | ~2x, **not in `position_changes`** |
| BIL | 5.43% | 3.48% | **−1.94pp** | −35.8%, the largest proportional cut in the book |
| IFRA | 7.12% | 5.19% | −1.93pp | never argued about by anyone |
| VNQ | 7.34% | 4.40% | **−2.93pp** | ~3x (judge's 1.0pp + ~1.93pp of fallback) |
| All 18 others | — | — | **+16.6% to +17.0% uniformly** | renormalized back to gross 1.0 |

**Net size reduction from a `reduce_size` verdict: zero.** The five largest positions were
force-trimmed by a near-constant ~1.93pp and everything else was scaled up ~17% to renormalize.
The EM names (EEM/EWT/EWJ/EFA) that the regime specialist argued should be "near-zero" were scaled
**up**.

**Good news:** this bug was diagnosed and fixed *while these audits were running*. The current
`ascent/execution/eod_runner.py` has `REDUCE_SIZE_GROSS_TARGET = 0.90`, a `protected` frozenset
derived from the verdict, and both call sites now pass both arguments (verified at `:836` and
`:922`). Its docstring enumerates the four defects, including that the old forced trim "sold
UUP/TLT/BIL — the exact FOMC hedge the verdict protected."

**What is still open:** the *current book* still reflects the bad trim. Decide before tomorrow's
FOMC whether to restore the hedge leg manually, or accept the position as-is.

### 1.3 An unsourced Sharpe of 0.52 is live in investor-facing output

`ascent/reporting/investor_letter.py:35-36`:
```python
WF_SHARPE = 0.52
WF_PERIOD  = "Jan 2020–Apr 2026"
```
Emitted at `:586` as "Walk-forward OOS Sharpe ({WF_PERIOD}): {WF_SHARPE}".

The canonical artifact says **0.4118**. **0.52 matches no `wf_report_*.json` in the repo** (nearest
is 0.4835, from a superseded corrupted-cache run). The period string is also wrong — the real OOS
window is 2021-01-08 → 2026-01-14.

This is the same class of defect as the retired `0.518`. It survives because
`verify_docs.py::no_unsourced_sharpe` iterates a **hardcoded 6-file list**
(`scripts/verify_docs.py:546-548`) that does not include this file. The guard passes 24/24 while
the wrong number ships.

### 1.4 The scheduler still has the outage's root cause

`logs/launchd_stderr.log` shows the cause of the 19-day silence verbatim — dozens of:
```
/bin/bash: /Users/kdong/Downloads/ascent capital v2 up to phase 5.1/scripts/run_eod.sh: Operation not permitted
```
A stale plist pointed at a previous machine's install path. That is now fixed (both plists are
byte-identical to the repo copies and loaded).

**But the deeper cause is not fixed.** `com.ascentcapital.heartbeat.plist:18-25` explicitly
switched to `StartInterval` because `StartCalendarInterval` jobs are *silently skipped* when the
Mac is asleep at the wall-clock moment, with no catch-up. **`com.ascentcapital.eod.plist:41` still
uses `StartCalendarInterval`** — so the watchdog survives a sleeping laptop and the job that
actually trades does not.

### 1.5 Three alert paths cannot fire

| Path | Why it's dead |
|---|---|
| Drawdown / factor breach / IC decay | `run_all_agents.py:1178` calls `_check_alerts()` **with no arguments**, inside a bare `except: pass`. Every threshold in `alert_system.py:22-25` derives from optional args that are `None`. Can only ever return `[]`. |
| Any ntfy delivery | `NTFY_TOPIC` is **absent from `.env`** (verified). Pipeline-internal alerts are write-to-logfile only. |
| Proof-of-life ping | `send_system_alive_ping()` — written specifically because "an alert channel that only ever fires on failure is indistinguishable from a broken one" — has **no production caller**. The stated fix for the silent outage is not installed. |

`logs/alerts.jsonl` corroborates: it contains only `liveness` entries, nothing else.

The one path that *does* work is `heartbeat_check.send_direct_alert()`, which always fires a local
macOS notification — effective only while someone is at the Mac.

### 1.6 The cost model / high-impact order block never runs

`ascent/execution/eod_runner.py:989-990`:
```python
_required_cost_keys = {"dollar_volume"}
features_arg = _cost_features if (_cost_features and _required_cost_keys.issubset(_cost_features)) else None
```
But `extract_cost_features` returns keys **`dollar_vol_21d`** and **`vol_21d`** — never
`dollar_volume` (`cost_model.py:10-11`, and I verified the return shape directly). The subset test
can never pass, so `features_arg` is always `None`, so `apply_cost_filter` — which removes
`HIGH_IMPACT` orders above the ADV participation cap — **never executes**.

The log line at `:985` still prints "Cost features loaded: N symbols", so the failure is invisible.

Two consequences ride on the same dead argument: `order_engine.py:105` gates the TWAP routing block
on `features` being truthy, so **TWAP is doubly non-functional** — and even if reached, that block
calls `should_use_twap()` and then only `log.info(...)`; it never calls `execute_twap`. Flipping
`TWAP_ENABLED` would change nothing.

### 1.7 Three regime sources disagree, and the stale ones drive behaviour

| Source | Label | As-of |
|---|---|---|
| `dashboard/regime_labels.csv` (engine truth) | **`stressed`** (59 consecutive days) | 2026-07-23 |
| `dashboard/regime_signal.json` | **`calm_bull`** | 2026-06-22 — **5 weeks stale** |
| `data_cache/ai_regime_assessment.json` | **`calm_bull`** | 2026-06-24 — **5 weeks stale** |
| The 07-27 debate path | `stressed` | 2026-07-27 |

The stale JSON is not inert: `run_all_agents.py:1454` reads its Opus-escalation trigger from it,
and four investor-facing reporting modules read it **with no freshness check at all**
(`investor_report.py:137`, `investor_letter.py:31,253`, `weekly_debrief.py:225`,
`scenario_planner.py:224`). A stale `calm_bull` is being published while the engine reads
`stressed`.

---

## 2. Performance

Three systems are kept separate. Conflating them is how a synthetic number got published once
before.

### 2.1 SYSTEM 1 — the quant backtest (the citable number)

Source: `outputs/wf_results/wf_report_clean_2026-06-22.json`, on a freshly re-fetched
single-source cache.

| Metric (OOS) | Value | Independently reproduced? |
|---|---|---|
| Sharpe | **0.41** | ✅ 0.4174 recomputed from `wf_equity_clean_2026-06-22.csv` at rf=4% |
| CAGR | **+10.3%** | ✅ |
| Excess CAGR vs SPY | **+1.0pp** | ✅ ~+1.1pp |
| Regression alpha | **+2.24%/yr** | — |
| Max drawdown | **−32.9%** | ✅ exact |
| Beta | **0.73** | ✅ |
| Window | 2021-01-08 → 2026-01-14, 21 folds, 1,134 OOS days | ✅ |
| **Walk-forward efficiency** | **−0.65 — OVERFIT, must be disclosed** | — |

**Three caveats that must travel with this number:**

1. **WFE −0.65** means the in-sample parameter optimizer adds *no* out-of-sample value. Fixed
   params would do as well. The Sharpe above is the realistic post-overfit figure.
2. **Sortino: every one of the 16 artifacts on disk is wrong.** `PerformanceAnalyzer.sortino`
   annualized both the numerator and the downside deviation, dividing results by √252. The stored
   `0.042` should be **≈0.67** (independently confirmed: 0.6748). The fix is now in
   `wf_framework/metrics.py` with two regression tests and a guard. Do not mechanically rescale the
   *older* artifacts — they came from a different code generation and ×√252 gives implausible
   values there.
3. **The framework that produced this number never calls `get_universe_on_date`.** `wf_framework`
   trades whatever symbols exist in the cache for the fold, so today's survivors are used for 2021
   folds. The headline Sharpe is computed on a **survivorship-selected universe** (the artifact's
   own `n_symbols: 936`). The *older* `walk_forward_runner.py` does this correctly with
   `strict=True, sp500_only=True`; the one that ran does not. This is the largest look-ahead
   exposure in the repo and it is not currently disclosed.

There has been **no walk-forward run in July at all** — newest artifact mtime is 2026-06-25.

#### ⚠️ 3 of the 21 folds produced no out-of-sample returns at all

This is new, and I verified the arithmetic directly. The artifact reports `n_folds: 21` and
`n_oos_days: 1134`. But 21 folds × 63 OOS days = **1,323 expected bars**. The shortfall is
**189 bars = exactly 3 × 63 = 3 whole dead folds.**

Reading `wf_folds_2026-06-04.json` (the fold-level file; the canonical clean run wrote only a
report and an equity CSV, but reports the *identical* 21 folds / 1,134 days, so the same folds are
dead):

| Fold | OOS window | IS Sharpe | OOS Sharpe |
|---|---|---|---|
| 16 | 2025-01-14 → 2025-04-14 | 6.69 | −2.46 |
| **17** | **2025-04-15 → 2025-07-16** | 1.17 | **0.0** |
| **18** | **2025-07-17 → 2025-10-14** | 1.29 | **0.0** |
| 19 | 2025-10-15 → 2026-01-14 | 2.79 | −0.19 |
| **20** | **2026-01-15 → 2026-04-16** | 2.59 | **0.0** |

Exactly folds 17, 18, and 20 — **the most recent end of the sample.** Their IS Sharpes are healthy
(1.17 / 1.29 / 2.59), so these are not folds that were skipped for lack of data; they trained fine
and then produced a flat, empty OOS series.

Two consequences:

1. **The headline Sharpe 0.41 is computed on 18 effective folds, not 21** — and two of the three
   dead ones (Apr–Oct 2025) fall *inside* the stated OOS window of 2021-01-08 → 2026-01-14. Roughly
   six months of the most recent period contributes nothing while being counted in the denominator
   of `n_folds`.
2. **The dead folds flatter the WFE.** `metrics.py:110-130` skips folds only when
   `is_sharpe <= 0`. These have positive IS Sharpe and OOS Sharpe of exactly 0.0, so each
   contributes a ratio of `0.0` to the mean — pulling a −0.65 WFE *toward zero*, i.e. making the
   overfit metric look better than it is.

#### Root cause: a cache-key collision between the optimizer and the engine

**Diagnosed and confirmed 2026-07-28.** It is not a data problem. It is a stale-cache bug.

`AscentPortfolioStrategy` memoises features and alpha in **class-level** dicts
(`ascent_strategy.py:67-68`) keyed by:

```python
data_key = (all_dates[0], as_of_date, data["symbol"].nunique())   # ascent_strategy.py:159
```

The engine calls `clear_cache()` once per fold and then runs the optimizer, which sweeps all 243
grid combos via `generate_signals(is_data)` — **with no `pit_boundary`** (`optimizer.py:105`), so
`as_of_date` falls back to the last date in the data it was given, i.e. the last IS date. The engine
then makes its real call, `generate_signals(full_context_data, pit_boundary=is_dates[-1])`
(`engine.py:177`), where `pit_boundary` is *also* the last IS date.

Both calls therefore produce:

| | `all_dates[0]` | `as_of_date` | `n_symbols` |
|---|---|---|---|
| Optimizer (IS only) | `is_start` | last IS date | symbols in the **IS window** |
| Engine (IS→OOS context) | `is_start` | last IS date | symbols in the **full context** |

The first two components are identical by construction. **The only thing distinguishing the two
cache entries is the symbol count** — and the winning params are, by definition, one of the combos
the optimizer already cached. So when the universe does not change between the IS window and the
full context, the engine's call is a cache hit on the **IS-only alpha panel**. That panel's index
ends at the last IS date, and the final line of `generate_signals` reindexes onto it:

```python
weights_ffilled = weights_at_rebal.reindex(alpha_dates).ffill().fillna(0.0)   # :347-352
```

So the returned weights contain no OOS rows at all → `oos_signals` is empty → `compute_returns`
hits its `len(common_dates) == 0` branch and returns an **empty Series** (`execution.py:112-113`) →
`analyser.sharpe([])` returns exactly `0.0` (`metrics.py:56-57`) → the fold contributes nothing to
the stitched curve. That is the precise signature in the artifact: `oos_sharpe`, `oos_cagr`, and
`wfe` all exactly `0.0`.

**Evidence — three independent confirmations:**

1. **The prediction holds across all 21 folds with zero mismatches.** Collision occurs iff
   `n_syms_IS == n_syms_FULL`:

   | Fold | `n_syms` IS | `n_syms` full | Collide? | Artifact |
   |---|---|---|---|---|
   | 0–16 | 896 → 933 | always **larger** | no | live |
   | **17** | 934 | **934** | **YES** | **DEAD** |
   | **18** | 934 | **934** | **YES** | **DEAD** |
   | 19 | 934 | 936 | no | live |
   | **20** | 936 | **936** | **YES** | **DEAD** |

   3/3 dead folds collide; 18/18 live folds do not.

2. **Fold 17 runs fine in isolation** — `clear_cache()` then the engine's call alone yields signals
   of shape `(252, 934)` indexed to 2025-07-16 and a full 63 OOS bars.

3. **Toggling the single variable reproduces the death.** Inserting one optimizer-style
   `generate_signals(is_data)` before the same call yields shape `(231, 934)` — exactly the IS bar
   count — indexed to 2025-03-07, the last IS date, with **0 OOS rows**.

**Why it only started firing recently:** the universe grew steadily from 896 to 936 symbols
(2020→2024), so the IS and full-context counts always differed and the collision never happened.
Once the universe plateaued in 2024, the counts started matching and folds began silently dying.
**This is a latent bug that activated with time, not a recent regression** — and it will get *worse*
as the universe stabilises further.

**Severity note:** this is a look-ahead-*safe* failure — it uses less data, not more, so it did not
inflate results by leaking the future. But it silently deletes folds from a published number, and
the zeros flatter the WFE.

**The fix** is to make the cache key unambiguous rather than to clear more aggressively. The
narrowest correct change is to include the data window's end date in the key, so an IS-only call and
an IS→OOS call can never share an entry:

```python
data_key = (all_dates[0], all_dates[-1], as_of_date, data["symbol"].nunique())
```

`all_dates[-1]` is the last date *present in the passed data* (last IS date for the optimizer,
`oos_end` for the engine), which is exactly the axis the two calls differ on. This keeps the
optimizer's 243-combo memoisation working — the whole point of the cache — while making the engine's
call a guaranteed miss. Add a regression test asserting
`n_oos_days == n_folds × oos_days` for a run with a static universe.

**Expected impact:** re-running restores 189 OOS bars (~9 months, Apr–Oct 2025 and Jan–Apr 2026) and
will change the published Sharpe, CAGR, and WFE. The direction is unknown — the three restored folds
had healthy IS Sharpes (1.17 / 1.29 / 2.59) but fold 16 and 19 either side of them were negative
(−2.46, −0.19), so do not assume it improves. **The number should be re-run and re-published before
it is cited again.**

### 2.2 SYSTEM 1+2 — the live paper book

| Metric | Value | Source |
|---|---|---|
| Window | 2026-03-23 → 2026-07-27 (75 unique dates from 84 rows) | `logs/holdings_log.jsonl` |
| Current equity | **$104,640.21** | last row |
| Total return | **+4.64%** (full) / **+3.79%** (from 2026-04-01) | equity-based |
| SPY, same window | **+12.67%** | `prices_live.parquet`, deduped on read |
| **Book vs SPY** | **−8.88pp** | derived |
| Peak | $112,869.60 on 2026-06-02 | |
| Max drawdown | **≥ −7.29%** — a *lower bound*, not a measurement | 19 trading days between peak and trough were never logged |
| Positions | 22 | |
| Sharpe | **NOT COMPUTABLE** | see below |

The ~9pp lag over four months is directionally consistent with the design (beta 0.73, defensive
non-equity sleeves, a 200MA cut, a vol-target overlay — all of which cost beta in an equity-only
bull). But it is real and large and should not be softened.

**Why no live Sharpe.** 75 unique sessions; 5 rows with a fake `day_return` of exactly 0.0
(the Alpaca late-settlement artifact); 11 duplicate rows (2026-06-10 appears **9 times**, with an
equity spread of 2.6% between them); and a 19-day hole. The `day_return` column chains to
**−12.24% raw / +0.37% deduped** against an equity-based **+3.79%** — three different answers from
one file. The column is broken. Do not reintroduce a Sharpe here.

**A published diagnosis needs retracting.** `CURRENT_VERIFIED_NUMBERS.md` currently explains the
two disagreeing SPY figures by saying the log's `spy_return` column "understates the index."
Deduped, that column chains to **+13.66%**, which *agrees* with the price cache (+12.67%). The real
cause is the 9 duplicated 2026-06-10 rows each carrying `spy_return: −0.0158`. Fix
`reconcile_numbers.py` to dedupe `holdings_log` on read (it already dedupes `prices_live`), then
retract that paragraph.

**Attribution over the outage window** (reconstructed from weights × price moves, **not** measured
P&L — `logs/attribution_log.jsonl` is 81% synthetic QQQ/SPY test rows written 2026-07-28 at 15:22
and needs purging):

| Symbol | Weight @06-29 | Move to 07-23 | Contribution |
|---|---|---|---|
| MU | 7.56% | **−20.22%** | **−1.53pp** |
| EWY | 7.14% | **−19.29%** | **−1.38pp** |
| KNF | 7.37% | −6.68% | −0.49pp |
| (10 others) | | | −1.64pp |
| BAX, AMCR, VNQ, DBB, UUP | | positive | +0.79pp |
| **Total reconstructed** | | | **−4.25pp** vs actual −3.19% |

**Two names, MU and EWY, account for 2.9pp of the 4.25pp reconstructed loss.** Both were AI-PM-sized
at the 2026-06-24 rebalance (`ai_pm_proposed: MU 0.08`; calibration log records MU `realized_21d:
−4.88%` against a thesis of "+8–14%"). Nobody was watching for 19 days.

### 2.3 SYSTEM 2 — the AI layer

**Current state:** Level 1 "Analyst", `ai_weight 0.05`, `days_at_level 19`, `days_stuck 19`,
`auto_revert_count 0`.

**Promotion gates (L1→L2): failing 4 of 7.**

| Gate | Value | Threshold | Pass |
|---|---|---|---|
| `sortino_edge` | 0.016 | ≥ 0.20 | ❌ |
| `hit_rate` | 0.0 | ≥ 0.52 | ❌ |
| `profit_factor` | 1.0 | ≥ 1.20 | ❌ |
| `min_decisions` | **0** | ≥ 5 | ❌ |
| `fade_rate` | 0.0 | ≤ 0.30 | ✅ |
| `regime_gate` | not yet evaluated | — | ✅ (default-pass) |
| `cooldown` | clear | — | ✅ |

**The critical reading: three of those four failures are failures to *measure*, not measured
underperformance.**

- All four return buffers in `earned_authority.json` are **empty lists** (verified directly). The
  promotion check requires `len(d_buf) >= window`, so promotion is currently **structurally
  impossible** regardless of performance.
- `n_decisions_evaluated = 0` while the decision log holds 9 rows and the calibration log holds 24
  rows with `realized_21d` populated. The scorer is not consuming its own inputs.
- `sortino_21d_d = 11.292` and `sortino_21d_astar = 11.276`. Annualized Sortinos of 11 are not
  physical — same `PerformanceAnalyzer` family as the WF bug. The *difference* may survive
  rescaling, but comparing 0.016 to a 0.20 threshold is comparing quantities on an unverified scale.

**The counterfactual tracks are a broken measurement, not a negative result.** This matters because
"the AI layer costs 6pp" has been treated as a finding.

| Comparison | Value | Days |
|---|---|---|
| D − A★ (pure AI vs pure quant) | −6.11pp | 24 |
| B − A★ (actual vs pure quant) | −2.01pp (or −7.82pp on a different overlap) | 24 / 38 |
| A − A★ | **+0.00pp** | 35 |

Four independent reasons not to cite any of it:

1. **`track_a_return` is byte-identical to `track_astar_return` on 35 of 35 overlapping days.** The
   A-vs-A★ distinction is degenerate.
2. **Chaining non-contiguous days is not a return.** 45 rows across a 79-trading-day span with a
   19-day hole. Track B chains to +16.03% while the actual account went +4.64%. B − C says the book
   *beat* SPY by +10.81pp; the equity-vs-cache measurement says it **lost** by 8.88pp. The file and
   the account disagree by ~20pp on the same question.
3. **One day moves the headline 1.35pp.** Leave-one-out: dropping 2026-05-26 → −4.75pp; dropping
   2026-06-23 → −7.44pp.
4. **The baseline contains the thing it's grading.** See §4.4 — Track A★ ("pure quant") is
   snapshotted from a pipeline that has *already consumed the AI PM's own Phase 1 priors*.
   `ascent/utils/freshness.py` documents this in the code itself.

Saying "the AI layer costs 6pp" is as unfounded as saying it adds alpha. **The honest framing right
now is the governance discipline, not the results** — the gates are correctly refusing promotion,
and the system is refusing to promote itself even though the reason it can't is partly a bug.

**Also note:** `logs/ai_pm_decision_log.jsonl` last entry is 2026-06-24 and records
`"phase2_model": "claude-opus-4-6"` — a generation behind the `claude-opus-5` constant now in
`ascent/llm/client.py`.

---

## 3. Architecture — end to end

This is the part to re-read when you come back to the project cold.

### 3.1 The 30-second version

```
rebalance_calendar.csv ──┐
                         ▼
              run_all_agents.py  (daily entrypoint, ~127KB, the conductor)
                         │
   ┌─────────────────────┼──────────────────────────────────────┐
   │  GATES: weekend? outage? already ran? sector coverage?      │
   │         regime stale? → refit                              │
   └─────────────────────┼──────────────────────────────────────┘
                         ▼
        4 QUANT AGENTS, in parallel (ThreadPoolExecutor)
        us_equities │ macro │ international │ alternatives
        └─ us_equities calls ascent/main.py:run_pipeline
              data → features → alpha stack → optimizer → overlays → backtest
                         ▼
        ORCHESTRATOR  orchestrator/central_intelligence.py
        regime-based capital allocation + skill weighting
        + 8 sequential guards (EM cap, intl cap, position cap,
          correlation guard, thesis coherence x2, crisis veto, macro divergence)
                         ▼
        HEDGE OVERLAY (VIXY sized by regime)
                         ▼
   ┌──────────── REBALANCE DAY ONLY ─────────────────────────────┐
   │  AI PM Phase 1  "pre-thesis"   → Sonnet 5,  16 read-only tools│
   │  AI PM Phase 2  "synthesis"    → Opus 5,    30 tools          │
   │  RED TEAM critique → Sonnet 5 → optional AI PM revision       │
   │  validate_pm_proposal → authority_blend (5% budget at L1)     │
   └───────────────────────────┬───────────────────────────────────┘
                               ▼
        write execution/merged_weights.json
                               ▼
   ┌──────────── REBALANCE DAY ONLY ─────────────────────────────┐
   │  HALT CHECK (execution/halt_state.json)                       │
   │  DEBATE  debate/debate_runner.py                              │
   │    adversarial engine (Haiku) → Round 1 (4 agents)            │
   │    → Round 2 rebuttals → JUDGE (Opus 5, extended thinking)    │
   │  apply EXACTLY ONE judge position change, authority-capped     │
   └───────────────────────────┬───────────────────────────────────┘
                               ▼
        eod_runner.run_eod_with_weights → Alpaca paper orders
                               ▼
        log holdings → generate_performance_page.py --push → GitHub Pages
```

### 3.2 Important correction to the project's own framing

**The four "specialist agents" contain zero LLM calls.**

`agents/us_equities_agent.py`, `macro_agent.py`, `international_agent.py`,
`alternatives_agent.py` import only `ascent.config`, `ascent.data.store.parquet`,
`ascent.portfolio.optimizer`, `pandas`, `numpy`, `yfinance`. **Nothing from `ascent.llm.client`.**
They are momentum/rank quant sleeves wrapped in an `AgentOutput` dataclass.

The AI-native layer proper is: **AI PM (2 phases) → debate (5 roles + judge) → adversarial engine →
red team → memory/reflection.** Describing the system as "4 AI agents" overstates it; describing it
as "4 quant sleeves under an LLM portfolio manager with an adversarial review board" is accurate.

### 3.3 The daily run, in order

**Gates first, before anything expensive** (`run_all_agents.py`):
1. `market_today()` — `America/New_York` calendar date (`ascent/utils/market_time.py:45`). The host
   is UTC+7, so this matters.
2. **Weekend branch** (`:626`) → `run_weekend()` and return.
3. **Catch-up guard** (`:647`) — refuses to run if >3 trading days were missed, unless `--catch-up`.
   Fails *closed*, including on any read failure.
4. **Same-session guard** (`:694`) — skips if `eod_log.jsonl` already has today. Fails open.
5. **Sector-coverage gate** (`:702`) — raises if `profiles` coverage < 80%.
6. **Regime staleness** (`:707`) — refits if `regime_signal.json` is >5 days old.

**Then bookkeeping** (both day types): score pending interventions, fill 21-day outcomes, start the
event-agent thread (market hours only), run the data hub, OpenBB/CFTC/Fama-French ingests, alt-data
collection, factor loadings.

**Then the four agents in parallel** (`:862`), each returning an `AgentOutput`.

**Then rebalance determination** (`:891`) — exact string match of `today.isoformat()` against
`rebalance_calendar.csv`.

**Then sequential PnL → skill → orchestrator** (`:952-1170`, explicitly commented as
must-not-parallelize, because skill scores are read from the PnL log written one step earlier).

**Then overlays, then the branch.**

### 3.4 `ascent/main.py:run_pipeline` — the quant core

Returns a **10-tuple** (`:915`):
```
(result, regime_engine, spy_wide, univ_wide, vix_series,
 target_weights, price_df, macro_df, price_cache_name, _alpha_breakdown)
```
Unpacked *positionally* by `agents/us_equities_agent.py:39-49`.

Steps:
1. **Data** — `load_or_fetch_prices` / `load_or_fetch_macro`. Cache name carries provenance:
   `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated`
   (live-fetch failure).
2. **Regime (1.5)** — fast path reads `regime_signal.json` if ≤5 days old; otherwise a full
   `RegimeEngine.fit`. Then blends `ai_regime_assessment.json` if fresh.
3. **Features** — `FeatureBuilder.compute_features()` → momentum, volatility, volume, mean
   reversion, trend, macro (broadcast), fundamentals, event/alt panels. Targets at horizons
   `[1,5,21]`.
4. **Alpha** — `build_alpha_stack(...)`, 15 sleeves, each cross-sectionally z-normalized before
   weighting.
5. **Portfolio** — factor covariance → MVO on the latest date only → rank-weighting for history →
   cluster cap → risk-budget cap → exposure overlays (200MA cut, vol targeting).
6. **Backtest** — `BacktestEngine` with spread 5bps + impact 5bps, execution delay 1 day.

### 3.5 The alpha stack (`ascent/alpha/stack.py:16-32`)

Sums to 1.00. Key set matches `self_improve.py` (guard-enforced).

| Sleeve | Weight | What it computes |
|---|---|---|
| `trend` | **0.41** | Weighted z-blend: mom_63d .30, mom_skip1m .20, macd .20, sma_cross .15, mom_21d .15, mom_126d .10 |
| `statarb` | **0.15** | Sector-residual reversal, 5/10/21d at .50/.30/.20, divided by *lagged* vol |
| `ml` | **0.10** | XGBoost, CPCV C(6,2)=15 folds, purge 5 / embargo 5 |
| `meanrev` | 0.05 | −zscore_20d, −(rsi−50)/50, −(bb_pct−0.5) |
| `volatility` | 0.05 | −vol_trend / vol_of_vol, CS-normalized |
| `earnings` | 0.05 | PEAD, momentum-neutralized by OLS residual |
| `analyst` | 0.05 | Rolling 63d net revisions |
| `llm_fundamental` | 0.03 | Haiku 6-step CoT on anonymized ratios |
| `narrative` | 0.03 | Haiku quarter-over-quarter shift ∈ {−1,0,+1} |
| `options_flow` | 0.02 | iv_skew ± put/call |
| `insider` | 0.02 | Rolling 63d net purchases |
| `short_interest` | 0.02 | short_pct_float |
| `earnings_tone` | 0.02 | Reindexes an offline weekly transcript-tone panel |
| `fundamental` | **0.00** | **disabled** — measured IC −0.015, IC-t −4.75 |
| `altdata` | **0.00** | gated off |

Then a **distressed filter zeroes any name with `mom_252d < −0.65`** (`:457`).

**Which weights actually run.** Resolution order is config `by_regime` → meta-learner →
`DEFAULT_ALPHA_WEIGHTS_BY_REGIME` → config `global` → flat default. I traced the on-disk state:
`active_alpha_config.json` has no `by_regime` key, and the meta-learner returns `None` (only
`calm_bull`, 7 of 15 sleeves, `min_n=1 < _MIN_OBS_TRUST=3`). **So
`DEFAULT_ALPHA_WEIGHTS_BY_REGIME` (`stack.py:36-63`) is what runs.**

⚠️ **That dict restores `fundamental = 0.08` in `stressed` and `crisis`** — re-enabling the sleeve
that integrity constraint 7 disables. `verify_docs.py` only checks the *flat* dict, so the guard
passes. The current regime is `stressed`. Worth a decision.

Then `_get_gated_weights` zeroes any sleeve whose rolling 5-date mean IC < −0.005 and donates the
freed weight to `trend`.

### 3.6 Portfolio construction and the guards that are real

**Objective** (`mvo_optimizer.py:117`, cvxpy CLARABEL → SCS fallback):
```
max  wᵀα − λ·wᵀΣw − κ·‖w − w_prev‖₁    s.t.  Σw = 1,  w ≥ 0,  w ≤ max_weight
```
λ = 1.0, κ = 0.002, `max_weight` = 0.10.

**Verified genuinely working:**
- `_water_fill_cap` (`optimizer.py:153-200`) is a real iterative water-filler — freeze over-cap,
  redistribute pro-rata (50 iters), then a 10-iteration clamp-and-renorm tail, with an
  infeasibility branch. *Note:* the docstring claims a "post-condition check" and there is **no
  assert** — it's a convergence loop.
- **Sector coverage <80% degrades safely and cannot collapse to one name.** `_normalize_sector`
  buckets each unknown symbol *individually* (`__unknown_{sym}__`), which is precisely what
  prevents the collapse.
- **Execution delay is honest** — no valid delayed signal means stay in cash, not trade the same
  bar (`backtest/engine.py:93-100`).
- **Vol-target scaling is strictly causal**; statarb divides by *lagged* vol.
- **No full-sample scaler anywhere in `feature_defs.py`** — every op is rolling or per-date
  cross-sectional.
- **ML sleeve guards genuinely disable the sleeve** (empty frame) on `n_converged < 10` or
  `p5 fold IC < −0.05`. Feature-name mismatch invalidates the cache, preventing the XGBoost shape
  crash.

**Key parameters:**

| Parameter | Value |
|---|---|
| Universe | 901 symbols (`config/us_equity_universe.json`) |
| `top_n` / `max_weight` / `min_weight` | 15 / **0.10** / 0.02 |
| Rebalance freq / execution delay | 10 days / 1 day |
| Costs | 5.0 spread + 5.0 impact = 10 bps |
| Vol target / lookback / floor / cap | 0.15 / 21d / 0.25 / 1.00 |
| 200MA cut | ×0.70, window 200, **requires VIX > 20** |
| Cluster cap / corr threshold | 0.20 / 0.70 |
| Distressed filter | `mom_252d < −0.65` |
| Correlation guard | cap 0.70 abs, 63d lookback, **halves the smaller position** |
| PM risk validator | pos 0.15, sector 0.40, min 5 positions |

### 3.7 The orchestrator merge

**Capital allocation** by regime: `calm_bull` 0.70/0.10/0.12/0.08 → `stressed` 0.45/0.25/0.10/0.20
→ `crisis` 0.30/0.30/0.05/0.35 (us_equities/macro/intl/alts). Agents with skill data get
`0.5 × (skill/Σskill) + 0.5 × base`; skill scores are **discarded if >1 day stale**.

Merge: `merged[sym] = Σ_agents w × alloc`, drop <0.5%, renormalize.

**Eight guards, in execution order:**
1. EM+commodity+gold+managed-futures aggregate cap **20%**
2. International equity aggregate cap **15%**
3. Per-position cap **10%** (single-pass water-fill)
4. **Correlation guard** — 0.70 absolute over 63 days, cross-agent pairs only, halves the smaller
   position. *This is the documented PDBC↔KMLM behaviour — confirmed as designed, not a bug.*
   Returns `[]` on <63 rows of data, i.e. **fails open silently**. Uses `abs(corr)`, so a −0.85
   hedge pair reads as a violation.
5. Thesis coherence L1 — 5 hardcoded pairs (UUP/PDBC, UUP/GLD, UUP/USO, VIXY/SVXY, TLT/HYG) at 4%
   → halve the smaller
6. Thesis coherence L2 — 6 factor-bucket conflicts at 6% → cut the smaller side 40%
7. Crisis veto — only when us_equities regime is `crisis`: merged = 0.60×macro + 0.40×merged
8. Macro-equity divergence — macro stressed while equity isn't → scale ×**0.90**, deliberately
   leaving Σ<1.0 as implicit cash

### 3.8 The AI PM, in detail

**Phase 1 — pre-thesis** (`run_ai_pm_prethesis`, `agents/ai_pm_agent.py:2240`)
- **Sonnet 5**, `max_tokens=6000`, 16 read-only tools, hard research budget of **7 calls** — the
  8th must be `propose_prethesis`.
- Output requires `macro_view`, `high_conviction_names`, and a `directional_stance` that itself
  requires `direction`, `thesis`, `upside_case_pct`, `downside_case_pct`, **`falsifier`**, and
  `horizon_days`. Forcing a falsifier at schema level is good design.
- Force-seal fallback bypasses the tool loop with `tool_choice={"type":"tool",...}`.

**Phase 2 — synthesis** (`run_ai_pm`, `:2421`)
- **Opus 5** when the smart trigger fires (regime is crisis, regime changed, ≥4 conviction names, or
  day-0 post-promotion), else Sonnet 5. **The trigger reads the stale `regime_signal.json`** — see
  §1.7.
- **30 tools**, ending in `propose_portfolio`. `quant_outputs` are pre-cached so `run_quant_agent`
  does not re-run the pipeline.
- **Prompt discipline is unusually well specified.** Declared edges: text/narrative, crowding,
  coherence. Declared **non**-edges: valuation, "data uncertainty", reducing extended names. Max
  **2** reductions per rebalance, each requiring all three of: crowding score ≥4 OVERCROWDED, a
  confirming text signal (`sec_tone < −0.3` or `transcript_sentiment < −0.3`), and a passing
  conviction gate.
- **Code-enforced submission gates** (`_tool_propose_portfolio:1729`) reject the proposal if
  performance feedback wasn't acknowledged, or if a Phase-1 thesis was sealed and
  `prethesis_disposition ∉ {FOLLOWED, OVERRIDDEN}`. Conviction inflation >40% is downgraded
  high→medium. Rejection text is fed back for up to 2 retries.
- **Red team** (Sonnet 5) critiques, then a second Opus turn may resubmit.

**How it lands:** `validate_pm_proposal` (max pos 15%, max sector 40%, min 5 positions, no
distressed) → `authority_blend` at the 5% Level-1 budget. On validator failure: quant 100%.

**⚠️ Phase 1 runs AFTER the quant agents and the orchestrator merge** (`:1354`, vs agents at
`:862`). The in-code comments at `:1312-1314` and `:1399` claiming otherwise are **stale and
misleading**. Consequence: `ai_regime_assessment.json` and `ai_prethesis_latest.json` are consumed
by the **next** run, not the current one.

### 3.9 Earned autonomy — two unrelated systems

**A. The AI PM career ladder** (`ascent/strategy/earned_authority.py`)

| Level | Title | `ai_weight` (one-way TE budget) |
|---|---|---|
| 0 | Shadow | 0.00 |
| 1 | **Analyst ← current** | **0.05** |
| 2 | Associate | 0.15 |
| 3 | Manager | 0.30 |
| 4 | Director | 0.50 |
| 5 | CEO | 0.75 |

Hard cap 0.80. Promotion requires **all six** gates simultaneously. Demotion is aggressive and
checked *before* promotion: a single day of Track D − A★ ≤ −0.099 drops straight to **Level 0** with
a 5-day cooldown.

**Nobody human promotes.** `update_authority` is called automatically from `run_all_agents.py:1778`.
There is no human-approval path in the module.

**B. Adversarial (judge) authority** (`debate/adversarial_authority.py`)

| Tier | Condition | `allowed_change_pct` |
|---|---|---|
| high | win_rate ≥ 0.70, n ≥ 30 | 4.0pp |
| medium | win_rate ≥ 0.50, n ≥ 30 | 2.0pp |
| **low** | **n < 30** | **1.0pp ← current** |
| suspended | win_rate < 0.40, n ≥ 30 | 0.0pp |

`adversarial_thesis` currently has **win_rate 0.778 with n=9** — good, but n<30 keeps it at 1.0pp.
Scoring excludes ±1pp-vs-SPY ties from the denominator, which is honest.

### 3.10 The debate

`debate_runner.run_debate` — score pending verdicts → run debriefs → detect blind spots → query
memory → Monte Carlo scenarios → scan catalysts → build quant context → **adversarial engine** →
**Round 1** (bull, bear, devil's advocate, regime specialist, quant sanity check) → **Round 2**
rebuttals → disagreement scorer → **judge** → write verdict.

**Exactly 2 rounds.** Models: bull/bear/devil = Sonnet 5; regime specialist = Haiku; quant sanity =
pure Python; judge = **Opus 5 via `extended_thinking_completion`**. Only the bear has tools
(4 of them, `max_tool_calls=2`).

**The judge caps itself before the runner sees anything** (`judge.py:167-215`): truncate
`position_changes` to `[:1]`, skip if the type is suspended, clamp to ±`allowed_change_pct`, drop
if |Δ| < 0.5pp, cap `conviction_press` at 10%, floor reductions at 1%. On parse failure it returns
`proceed`, confidence 0.0, `degraded: True`, no changes.

**The write-isolation invariant HOLDS — verified empirically.** I had an exhaustive grep run over
every write in `debate/*.py` (`open(...,"w")`, `write_text`, `json.dump`, `.write(`, `to_parquet`,
`save_parquet`). The complete result set is verdict files, `agent_credibility.json`,
`adversarial_authority.json`, `adversarial_interventions.jsonl`, `adversarial_monitor.json`, and
`execution/halt_state.json` (a flag file, not weights). **Nothing under `debate/` writes to
`merged_weights.json`, any parquet cache, or any alpha/portfolio artifact.** The judge's change
reaches the book only because `run_all_agents.py` — outside the package — reads
`verdict["position_changes"][0]` and rewrites the file itself.

**Two things worth knowing:**
- Every agent call is wrapped in `try/except` producing `"<X> failed: {e}"` strings that **flow into
  the judge prompt as if they were arguments**.
- `disagreement_scorer.interpret_disagreement` labels are **inverted** relative to the score
  definition — the docstring says 1.0 = max disagreement, but `score < 0.30` returns
  `"genuine_disagreement"`. Low impact (advisory only, and the judge is told to ignore it for
  direction) but the wrong label is written into the prompt.

### 3.11 Execution

**Alpaca, paper only, raw REST via `requests`** — no SDK on the main path.
`ALPACA_BASE_URL` defaults to `https://paper-api.alpaca.markets/v2`; there is **no live-endpoint
code path anywhere**. Credentials are captured **at import time**.

`run_eod_with_weights` (`eod_runner.py`) is the only thing that submits orders:
calendar re-gate → verdict handling → portfolio value + positions → `compute_orders` (drops
|delta| < 0.5%, sells before buys) → **kill switch** (8% warn / 15% hard stop, latched) →
`dry_run` short-circuit → `cancel_all_orders()` → per-order `close_position` (for full exits, to
dodge qty-rounding 403s) or `submit_order`.

**What is NOT in force despite appearances:**

| Claimed / expected control | Reality |
|---|---|
| Cost model / high-impact order block | Unreachable — key mismatch (§1.6) |
| TWAP routing | Logs only, never calls `execute_twap` (§1.6) |
| Large-trade human approval | `approval_server.py` is 313 lines with **no non-test caller**, while `eod_runner.py:625-626` documents approval as active. **That docstring is false.** |
| Market-open check before submitting | `is_market_open()` has exactly one caller in the whole repo — `pre_rebalance_checklist.py:127`. Neither execution path calls it. |
| Debate gate | `debate_gate.should_run_debate()` is a hardcoded `return True` |
| Kill switch on unexpected errors | **Fails open.** `eod_runner.py:850-851` swallows any non-`KillSwitchTriggered` exception and **continues to trade**. A malformed `kill_switch_state.json` or an unreadable eod log silently degrades to no drawdown protection. |

`intraday_trigger.execute_intraday_adjustment` is the same shape as TWAP — every branch is a
`log.info` with `"status": "logged_only"`, and the comment claims `TWAP_ENABLED` controls execution
when it does not. **Any risk plan that assumes "we can just enable TWAP" is wrong** — the wiring
does not exist.

### 3.12 The three off-calendar paths

Not everything happens on the calendar. Three mechanisms can trade between rebalances:

1. **Discovery mini-rebalance** — `ticker_discovery.run_discovery` finds a candidate from news;
   `_insert_candidate_weights` gives it `1/(n+1)` and trims everything else pro-rata. **Add-only** —
   no re-ranking, no agent re-run. Guarded by a 5-trading-day cooldown, suppressed within 3 trading
   days of a scheduled rebalance, and protected by an explicit assertion that aborts if the insert
   would fully exit any held name or add more than one symbol. Bases on the **live Alpaca book**,
   not the recomputed target — a comment documents the 2026-06-30 incident where using the target
   caused 27 orders and 7 full exits.
2. **Falsifier trim** — if a registered falsifier fires, one 25% trim to cash, floor 4%.
3. **IC-decay early trigger** — ⚠️ **this one cannot actually execute.** `run_all_agents.py:912`
   sets `is_rebalance = True`, but `eod_runner.py:646-661` re-reads the calendar independently and
   `force` is not passed on the main path. Full agents + AI PM Phase 1/2 + debate all run, and
   nothing trades.

### 3.13 Scheduling and publication

Two launchd agents, both loaded, both byte-identical to the repo copies:
- `com.ascentcapital.eod` → `scripts/run_eod.sh`, **Tue–Sat 09:00 machine-local (UTC+7)**. Local
  Tue 09:00 → Mon 19:00 PDT → **Monday's US session**; local Sat covers Friday. Monday is
  deliberately excluded. 09:00 rather than 08:00 buys margin over Alpaca's ~17:00 PT 1D-bar
  settlement, which is what makes Track B read real numbers.
- `com.ascentcapital.heartbeat` → `heartbeat_check.py --quiet`, **every 6h, `RunAtLoad=true`**.

**Publication has no CI.** There is **no `.github/workflows/`**. `generate_performance_page.py
--push` does `git add docs/index.html README.md` → commit → `git push` from whatever branch is
checked out, with no refspec. GitHub Pages serves `main:/docs` (confirmed live via the API). So a
laptop that doesn't run leaves stale numbers **published** — exactly the failure mode CLAUDE.md
warns about. The dashboard push also fires on halted and `halt_and_review` days, because it sits
outside the failure-guarded block.

---

## 4. Implemented but wired to nothing

This is the single most surprising category, and it matters because reading the config gives a very
different impression of the risk controls than reading the call graph.

### 4.1 Risk machinery that never executes

| Component | Status |
|---|---|
| **Regime `risk_multiplier`** (calm 1.00 → crisis 0.40) | Computed, attached to the signal, printed, **dashboard-published — and never multiplied into any weight anywhere.** The only code that would, `apply_regime_to_portfolio` (along with `regime_scale_weights` and `regime_adjust_sleeve_weights`), was confirmed to have zero live callers and was **deleted 2026-08-16** — not merely present-but-uncalled anymore. Gross exposure comes solely from the 200MA cut and vol targeting. |
| **Regime per-name max weight** | Reachable only via `sector_constrained_weighted(regime_signal=...)`, and `main.py:772` passes `None`. The MVO path *accepts* `regime_label` and never reads it. The live per-name cap is a constant 0.10, never tightened. |
| **Config sleeve adjustments** | `types.py` defines nonzero deltas; `settings.py:207-213` overrides **every entry to 0.00**. Confirmed: `sleeve_*` columns are `0.0` on every row of `regime_labels.csv`. |
| **Factor risk constraints** (`ascent.risk.factor_constraints.build_factor_constraints`) | Was fully implemented with regime-tightened bounds; `main.py:752` always passed `factor_constraints=None`. Confirmed zero live callers and **deleted 2026-08-16** — the `factor_constraints` parameter itself stays live in `mvo_optimizer.py`/`optimizer.py` for a future builder, but there is no code left to populate it. |
| **`ai_pm_guardrails.apply_guardrails`** | The per-level `max_change` / `max_new` / `max_te` / short-selling gate table had zero non-test callers and was **deleted 2026-08-16**, along with its private helpers (`_rolling_corr`, `_apply_tracking_error_cap`) and `_LEVEL_CONFIG`. Production uses only the validator + blend. |
| **`as_of_join` / `as_of_merge`** | Both **dead**, zero callers outside the guard that checks they're *defined*. Real PIT slicing is ad hoc. CLAUDE.md says "always use" these. |
| `portfolio/regime_covariance.py`, `risk/covariance.py::shrinkage_covariance`, `risk/regime.py::classify_regime` | Zero callers |
| `should_refit`, `check_and_run_emergency_refit`, `update_particle_filter` | Zero call sites — the only live refit is the 5-day staleness check |
| `self_improve` | Hard-gated off (`SELF_MODIFY_ENABLED = False`); `run_self_improve` early-returns |

**Net effect on the current book:** the regime label is `stressed`, and that produces **no exposure
cut whatsoever**. The entropy penalty fires on essentially every row, and `_apply_vix_confirmation`
restores stressed days to a multiplier of exactly 1.0 when VIX < 20 — which is moot anyway, since
nothing consumes the multiplier.

What the label *does* still drive: the VIXY hedge sizing, the Opus escalation trigger, the debate
gate, posture reporting, the factor-discovery and altdata IC gates, covariance half-life, and the
sleeve dict via `DEFAULT_ALPHA_WEIGHTS_BY_REGIME`.

### 4.2 Data freshness

**Five alpha sleeves carrying 0.17 combined weight are feeding on frozen caches.** The *fetchers*
for `analyst.py`, `earnings.py`, `fundamentals.py`, `insider.py`, plus `short_interest.py`, are
imported nowhere in the live path — but the caches are still read unconditionally every run
(`main.py:550-602`) with **no staleness gate** (`validate_cache` is never applied to them).

On-disk mtimes: `fundamentals` Apr 19, `earnings` Apr 20, `analyst_revisions` /
`insider_transactions` / `short_interest` May 3.

Also dead: `polygon.py`, `options.py`, `reddit_sentiment.py`. And `tiingo.py` reads
`TIINGO_API_KEY` while the live OpenBB path uses `TIINGO_TOKEN`.

### 4.3 The duplicate-row problem is worse than documented

`data_cache/prices_live.parquet` **right now**: 1,840,476 rows, **322,517 duplicate
`(symbol, calendar-day)` pairs (17.5%), across 928 symbols.** Time components 00:00 / 19:00 / 20:00.
Three source generations blended (`yfinance_hub`, `yahoo_hub`, `yahoo`).

The live pipeline defends itself with `~index.duplicated(keep="last")`. **The walk-forward framework
does not.** `scripts/run_ascent_wf.py:223` calls `load_parquet("prices_live")` and never
`normalize_prices` — which holds the only `drop_duplicates`.

**The consequence is specific and serious:** `WindowGenerator` counts *timestamps*, not trading
days. With 1,240 of 1,999 calendar days appearing twice, a 252-bar IS window spans ~126 calendar
days and **the 21-bar purge gap collapses to ~10 — shorter than the 21-day label horizon it exists
to cover.** The canonical artifact ran on the clean staging file and is unaffected. **Any future run
against `prices_live` is not.**

Provenance enforcement is also weaker than stated: it's **naming-convention only, write-side,
prices-only**. No reader anywhere checks provenance, the `source` column is excluded from the dedup
identity, and **macro has no fallback isolation at all** — FRED-failure simulated macro is written
straight into `macro_live` under the live name.

### 4.4 The AI PM has three uncapped channels

This is the finding with the most measurement consequence. Phase 1's output reaches the book through
paths that **bypass the earned-authority budget entirely**:

| Channel | Where | Bound |
|---|---|---|
| Conviction names floored to the 20th percentile of alpha | `ascent/main.py:674-682` | **none from earned authority** |
| Avoid-list names zeroed across the whole alpha panel | `ascent/main.py:685-689` | **none from earned authority** |
| Regime label + risk multiplier blend | `regime/engine.py:484-503` | blend α ≤ 0.30 |

`ascent/utils/freshness.py` documents this in the code itself: *"Because these two channels bypass
the earned-authority budget entirely, a month-old opinion was steering position sizing with no cap
and no expiry."* `AI_PRIOR_MAX_AGE_DAYS = 14` now bounds staleness (failing closed on
missing/unparseable/future dates) but does not remove the channel.

**And the measurement contamination that follows:** Track A★ — the "pure quant" baseline used to
grade the AI PM — is snapshotted from a pipeline that has **already consumed the AI PM's own
priors**. The baseline contains the thing it's measuring. This is an independent reason (on top of
the four in §2.3) not to cite the counterfactual.

### 4.5 Look-ahead exposures not currently disclosed

- **`wf_framework` never calls `get_universe_on_date`** — survivorship-selected universe (§2.1).
- **`wf_framework` fits no regime model at all** — `regime_signal=None`, `_estimate_regime` is dead,
  so the regime trend caps never fire in the backtest.
- **`filtered_probs` calls `score_samples`**, which returns forward–**backward** *smoothed*
  posteriors conditioned on the whole sequence. The method name and the "causal" docstring are wrong
  for every historical row of the label cache.
- **The production regime fit is full-sample** (`main.py:495-504`), and `save_for_intel` writes every
  historical label to `regime_labels.csv` — those in-sample labels are consumed by the
  factor-discovery gate and the altdata IC validator.
- **ML sleeve `purge_days=5` against a 21-day label** — leaks ~16 days of label overlap at all 15
  splits. And a 21-business-day fast path means CPCV (and therefore the sleeve's OOS reliability
  guards) is **skipped on most days**.
- **Two WFE definitions write into the same artifact family**, and the primary one *drops folds
  conditional on their in-sample outcome* (`if fold.is_sharpe <= 0: continue`).
- **`FIXED_PARAMS`** is documented as the median of IS-selected params across all 21 folds — chosen
  with knowledge of every fold, then fed back as a single-element grid.
- **Google Trends** divides the whole 12-month series by its own `max()` and overwrites the entire
  column history on each update — every historical value is scaled by a future maximum.
- **SEC filings** fabricate the vintage as `signal_date = (today − 45d) + 45d ≡ today`, while
  `methodology_index.py:151` advertises a derived `lag_days: 45`.

---

## 5. What the system decided recently, and why

### 5.1 Decision inventory

| Date | Type | Rec | Conf | Outcome |
|---|---|---|---|---|
| 2026-04-15 | **scheduled** | reduce_size | 0.88 | **WRONG** (+5.63% missed) |
| 2026-05-27 | **scheduled** | proceed | 0.62 | partially correct |
| 2026-06-10 | **scheduled** | proceed | 0.62 | **CORRECT** |
| 2026-06-15 | discovery (NVDA) | proceed | 0.62 | WRONG (−1.13%) |
| 2026-06-22 | discovery (INDY) | proceed | 0.62 | WRONG (−1.48%) |
| 2026-06-24 | **scheduled** | proceed | 0.58 | partially correct (−0.43%) |
| 2026-06-29 | discovery (SCHH) | proceed | 0.55 | **not scored** |
| 2026-07-08 | **scheduled** | — | — | **MISSED (outage)** |
| 2026-07-22 | **scheduled** | — | — | **MISSED (outage)** |
| 2026-07-27 | discovery (PDBC) + catch-up | **reduce_size** | 0.54 | **not scored** |

### 5.2 The 2026-07-27 decision, in the judge's own words

Confidence 0.54, disagreement 0.7914. This was the first move off `proceed` since April 15.

On what it accepted:
> "the devil's advocate and regime specialist independently converge on the same structural point
> that the bull never rebuts — the correlation assumption. The bull's rebuttal ('zero
> high-correlation pairs') is exactly the turkey argument: unconditional correlations measured in a
> low-vol tape (9.4% annualized) do not survive a stressed-to-crisis transition, and the model's own
> regime-flip estimate (-7.1%) dwarfs its p5 scenario (-1.5%). That gap is the honest signal here."

On what it chose:
> "VNQ is the cleanest intervention: highest adversarial score (0.68), the bull explicitly declined
> to defend it ('won't defend those on vibes alone') ... That is four-way agreement against one
> position with no defender — trimming into strength rather than into a hedge. Authority is capped at
> 1% per intervention (adversarial_thesis unproven, n=9), so VNQ goes 7.3% → 6.3%, well short of the
> 5.0% suggestion but the largest move earned authority permits."

**What it explicitly declined:** cutting UUP or TLT (the quoted hedge-leg reasoning in §1.2), the
regime specialist's full de-risking plan ("Increase TLT/SGOV to 20%+, cut real estate/EM by 50% ...
Rebalance toward 45%+ defensive"), and taking VNQ to the suggested 5.0%.

**Then the execution layer did the opposite** — see §1.2.

### 5.3 What the system knows about its own failures

`blind_spots.json` (updated 2026-07-27, 11 verdicts / 7 debriefs analyzed) tracks 10 patterns,
**6 rated HIGH**. Its own summary:

> "The core failure is separating 'regime continuation probability' from 'position structure
> safety' — the system conflates a correct macro call with permission to hold an unhedged,
> concentrated, correlated portfolio. ... Until the system enforces hard limits on concentration,
> beta, correlation, and defensive allocation independent of regime base rates, it will continue to
> hold unhedged concave portfolios through regime transitions."

Read that alongside §4.1: the hard limits it's asking for are **implemented and wired to nothing.**

Highest-frequency patterns: recency bias (the April 15 miss cited 4 times as justification for
ignoring warnings — and visible verbatim in the 06-29 and 06-24 verdicts); defensive allocation
misclassified (19.4% claimed vs 7.5% verified, an 11.9pp gap appearing across multiple sessions);
EM/cyclical concentration flagged 6 times with no material reduction ever; beta 0.928 against a
0.300 bound.

**The blind-spot text was injected into the 07-27 debate and the judge partially complied** — it did
*not* invoke the April 15 miss, and it moved off `proceed`. Then blind spots 3 and 6 were re-committed
by the execution layer, which scaled EM **up** ~17%.

**Agent credibility: nobody is trusted.** bull 0.371, bear 0.369, devil 0.374, all n=7 — below the
`MIN_SAMPLES_FOR_TRACK_RECORD = 10` display threshold. The judge used this in both directions on
07-27, accepting the bull's framing that "no single agent has an earned track record."

⚠️ **`logs/adversarial_interventions.jsonl` has no row for 06-29 (TLT) or 07-27 (VNQ).** Last entry
is 2026-06-24. Both recent interventions were never registered for outcome scoring, so the n counter
that gates authority promotion **is not advancing**. (Commit `56f598c fix(debate): apply and record
the judge's position change on the discovery path too` appears to address this — worth confirming on
the next discovery day.)

⚠️ **On 2026-06-29 the judge's change never landed at all** — every one of the 22 weights is
byte-identical between the verdict's `portfolio_state.weights` and the executed target. The intended
TLT 8.735% → 7.7% trim was a complete no-op. So across two consecutive interventions: one didn't
apply, and the next applied ~3x too much to the wrong names.

### 5.4 The phantom `merged_weights.json`

`execution/merged_weights.json` (dated 2026-07-27, `generated_at 15:33:46`) **does not describe the
live book.** It holds 17 names including AMD, ASH, CROX, DINO, DOC, DVA, MNST, STT — **none** of
which are in `holdings_log.jsonl`. It has VNQ at **9.72%**, *above* the pre-trade 7.34% and the
opposite of the judge's trim.

Its source is the second merge of the day (`multi_agent_run.jsonl`, 15:34:00), in which
**`agents.us_equities.n_positions == 937`** versus 10 on every prior run. That is a broken agent
output. It was written *after* orders were submitted at 15:33:05, so it was not executed — but it is
now the on-disk "current target," and anything downstream that reads that file gets a phantom
portfolio. This is a real hazard: §3.12 notes the discovery path bases on the live book, but other
consumers read this file.

---

## 6. Repo and test health

**Tests: 1,202 collected. Subsets pass cleanly (146 passed, 0 failed). The full suite is not usable
as a gate.**
- `pytest-timeout` is **not installed** (`--timeout` is an unrecognized argument).
- `pyproject.toml:15` sets `addopts = "-v --tb=short"` — no `-n`, no parallelization.
- Attempts ran >40 min wall without completing; `tests/test_wf_framework` is the bottleneck at
  ~200s **per test**.
- All 7 warnings are `datetime.utcnow()` deprecations.

**`tests/test_wf_framework/test_ascent_engine.py` fails — but not for the reason it first
appears.** I reproduced `test_engine_runs_with_portfolio_strategy` failing twice (200s, 206s). An
audit initially read this as "the engine that produces the published WF figures is broken." **That
overstates it, and I checked:** on real data the same engine produces healthy, varied fold results
(OOS Sharpes of 3.12, 2.15, 0.05, −0.36, −2.10, …) and the canonical 1,134-day curve. The test
fixture is 30 pure random-walk symbols (`np.random.randn * 0.012`, no drift, no cross-sectional
structure) with no sector metadata and no usable vol data — the logs show sector coverage failing,
the risk-budget cap exempting all 30 names, and every fold reporting `OOS_Sh=0.00`. That is a
degenerate-fixture failure, not a production defect.

It still matters, for two reasons. First, **this is a broken test gate on the engine that generates
the published numbers**, so a real regression there would not be caught. Second, the failure mode
exposes a real (if small) defect in production code.

**The empty-OOS guard in the engine is insufficient.** The confirmed traceback is:
```
ascent/research/wf_framework/engine.py:212
E   IndexError: single positional indexer is out-of-bounds
```
`engine.py:205` guards with `if not oos_return_chunks: raise RuntimeError("All folds skipped — no
OOS returns produced.")` — but that only checks the *list* is non-empty. Here the list contains
Series that are themselves empty, so the guard passes, `pd.concat` yields an empty frame, and
`equity_curve.iloc[0]` at `:212` raises a bare `IndexError` instead of the intended clear error.

That is the same code path as the three dead folds in §2.1. Today 3 of 21 folds come back empty and
the run survives; if a data problem emptied *all* of them, the engine would die with an opaque
pandas `IndexError` rather than the diagnostic message it was written to produce. The fix is to
guard on `stitched.empty` rather than on `not oos_return_chunks`.

So: fix the guard (one line, real), fix the fixture (so the gate works again), and separately find
out why real folds go empty (§2.1, the actual priority).

**Stale bytecode makes tracebacks unreadable.** `tests/__pycache__/*.pyc` carried `co_filename`
from a previous checkout (`/Users/scott/Downloads/ascent capital v2 up to phase 5.1/`), so pytest
printed `???` instead of source lines. `-p no:cacheprovider` does not clear it. Deleting
`tests/__pycache__` fixes it — I did that during verification.

**`scripts/verify_docs.py`: 24 passed, 0 failed.** The guard is genuinely effective — it catches
model constants, the Claude-5 parameter rules, `content[0].text` Haiku-only sites, alpha key-set
agreement, all four kill switches being `False`, the single-position-change rule, the 10-tuple, the
Sortino fix, and that every path named in CLAUDE.md resolves.

**Two weaknesses in the guard itself:**
- `no_unsourced_sharpe` iterates a **hardcoded 6-file list** that excludes `investor_letter.py`,
  where `WF_SHARPE = 0.52` lives (§1.3).
- `fundamental_disabled` uses `next(k for k in weights if "fundamental" in k)` — it picks
  `fundamental` only because it happens to precede `llm_fundamental` in the dict. Reordering would
  silently retarget it at `llm_fundamental = 0.03`. It also doesn't inspect
  `DEFAULT_ALPHA_WEIGHTS_BY_REGIME`, where `stressed`/`crisis` restore `fundamental = 0.08` (§3.5).

**⚠️ The guards and the map are uncommitted.** `scripts/verify_docs.py`,
`scripts/reconcile_numbers.py`, and `docs/REPO_MAP.md` are all **untracked**, yet CLAUDE.md treats
them as the enforcement layer. A fresh clone or any CI has no guard at all. Commit them.

**Git hygiene:**
- 35 of the last 60 commits are `chore: update performance dashboard` — bot commits dominate and
  make human changes hard to find.
- `.gitignore` covers `.env` correctly (never committed). But `logs/` was added to the ignore list
  *after* ~20 log files were already tracked, so `launchd_stderr.log`, `run_eod.log`, and
  `logs/snapshots/*` still get committed to a **public** repo.
- Untracked and should be ignored: `graphify-out/` (42M), `logs.bak-2026-06-19-rerun/` (2.1M),
  `.idea/`.
- `.worktrees/` holds **5 stale directories not registered with git** (`git worktree list` doesn't
  show them), each with its own `.venv`, some symlinked to Python 3.9. 15 local branches, 3 on
  origin.

**A note on this audit's own reliability.** The working tree changed *while these audits ran* —
`HEAD` advanced several times and `eod_runner.py` was rewritten mid-session by a concurrent agent
session committing to `main`. I re-verified the highest-stakes findings against the current tree
before writing this. Two findings from the audits are now **stale and fixed**: a `NameError` in
`_enforce_reduce_size` (the name is gone; both call sites now pass `target_gross` and `protected`),
and the judge's change not being recorded on the discovery path (commit `56f598c`). Everything else
in §1 I confirmed directly.

---

## 7. Honest scorecard

### Genuinely good

- **Weight capping, sector degradation, cost modelling, execution delay, and feature causality are
  all correctly implemented.** The water-fill cap is a real algorithm. The unknown-sector bucketing
  genuinely prevents single-name collapse. There is no full-sample scaler anywhere in the feature
  layer.
- **The ML sleeve's OOS reliability guards actually disable the sleeve** rather than warn.
- **The AI PM's prompt discipline is unusually rigorous** — declared non-edges, a schema-required
  falsifier, code-enforced submission gates that reject the proposal and feed the rejection back.
- **The debate write-isolation invariant holds**, verified by exhaustive grep rather than assumed.
- **The authority gates are refusing promotion** and have never been loosened.
- **The guard script is real and effective** where it reaches.
- **`heartbeat_check.py` is well engineered** — stdlib-only *by design* so it cannot be killed by
  the failure it watches, with hand-rolled market holidays including Gregorian-Easter Good Friday
  rather than a dependency that was never installed.
- **The project's documentation culture is genuinely unusual.** The `reduce_size` docstring
  enumerates four defects it found in its own predecessor. `freshness.py` documents the
  measurement contamination it introduces. That habit is why this audit could find as much as it did.

### Genuinely weak

1. **Measurement.** The live book has no defensible daily return series, the counterfactual is
   broken four ways, the attribution log is 81% synthetic, and the AI layer cannot be graded because
   its return buffers are empty and its scorer reads none of its 33 available input rows.
2. **A large gap between configured and executed risk controls.** Regime multipliers, regime max
   weight, factor constraints, the guardrails table, and both PIT helpers all read as active in
   config and are inert in the call graph.
3. **Data provenance and freshness.** 322k duplicate rows, five sleeves on 2-3 month old caches with
   no staleness gate, macro simulated data written under the live name.
4. **Alerting.** Three of four alert paths cannot fire; the proof-of-life ping written to solve the
   silent-outage problem is not installed.
5. **One unretracted wrong number is publishing right now** (`WF_SHARPE = 0.52`), in a file the
   guard doesn't scan — and the *correct* number has its own undisclosed defect: 3 of its 21 folds
   contributed no out-of-sample returns.
6. **Operational single point of failure.** No CI; publication depends on one laptop's `git push`;
   the trading job still uses the scheduler mode that silently skips when the Mac sleeps.

### What I would honestly claim externally today

> A modular quant research platform with a walk-forward-validated OOS backtest — Sharpe 0.41,
> CAGR +10.3%, beta 0.73, max drawdown −32.9% over 21 rolling folds and 1,134 out-of-sample days —
> with a negative walk-forward efficiency of −0.65 disclosed as overfit, and a survivorship
> qualification on the universe. Live on Alpaca paper since March 2026: +3.8%, against SPY +12.7%,
> with a one-month monitoring outage in the middle. An LLM portfolio-manager layer runs on top at a
> 5% tracking-error budget, has not earned promotion, and does not yet have a measurement
> infrastructure capable of establishing whether it adds value in either direction.

That is a weaker claim than the repo's prose in places, and a more defensible one.

---

## 8. Suggested order of work

**Before tomorrow (FOMC 2026-07-29):**
1. Decide whether to restore the UUP/TLT/BIL hedge leg that the fallback cut against the judge's
   explicit instruction (§1.2).
2. Rotate both Alpaca credential pairs and rewrite history (§1.1).

**This week:**
3. Delete `WF_SHARPE = 0.52` from `investor_letter.py`; route it through
   `reporting/verified_numbers.py`; widen the guard's file list (§1.3).
4. Switch `com.ascentcapital.eod.plist` to `StartInterval`, matching the heartbeat (§1.4).
5. Pass the real arguments to `check_alerts()`, set `NTFY_TOPIC`, and wire
   `send_system_alive_ping()` (§1.5).
6. Fix the cost-model key mismatch — one-line change, `{"dollar_volume"}` → `{"dollar_vol_21d"}`
   (§1.6).
7. Refresh or delete `regime_signal.json` and `ai_regime_assessment.json`; add a freshness gate to
   the four reporting consumers (§1.7).
8. **Commit `verify_docs.py`, `reconcile_numbers.py`, and `REPO_MAP.md`** (§6).
9. Purge the 251 synthetic rows from `attribution_log.jsonl` before they corrupt future work (§2.2).
10. Investigate the phantom `merged_weights.json` and the 937-position agent output (§5.4).

**Before the 2026-08-05 rebalance:**
11. **Fix the cache-key collision that kills folds 17, 18, and 20, then re-run and re-publish the
    backtest** (§2.1). Root cause is diagnosed and confirmed; the fix is one line in
    `ascent_strategy.py:159` (add `all_dates[-1]` to `data_key`) plus a regression test asserting
    `n_oos_days == n_folds × oos_days`. This is the highest-value item in this list — everything
    else is plumbing, this one is the published number. Expect Sharpe/CAGR/WFE to move; direction
    unknown.
12. Rebuild the authority return buffers and fix `n_decisions_evaluated = 0`, so the AI layer becomes
    measurable at all (§2.3).
13. Dedupe `prices_live` and make the WF framework dedupe on read — until then no fresh backtest on
    that cache is trustworthy (§4.3). Likely related to item 11.
14. Decide on `fundamental = 0.08` in the `stressed`/`crisis` weight dicts, given the current regime
    is `stressed` and integrity constraint 7 says the sleeve is disabled (§3.5).
15. Fix the `test_ascent_engine.py` fixture so the WF engine has a working test gate again (§6),
    change the `engine.py:205` guard to test `stitched.empty` instead of `not oos_return_chunks`,
    and delete `tests/__pycache__` so tracebacks are readable.

**Bigger decisions, not urgent:**
16. Either wire up the regime risk multiplier and factor constraints, or delete them and stop
    publishing a `risk_multiplier` that does nothing (§4.1).
17. Add `get_universe_on_date` to `wf_framework`, then re-run the canonical backtest — the headline
    Sharpe currently carries an undisclosed survivorship qualification (§2.1).
18. Make the test suite usable as a gate: install `pytest-timeout`, add `-n auto`, mark the slow
    WF tests (§6).
19. Extend the guard's coverage to the dead-wiring class of defect. `verify_docs.py` verifies the
    four kill switches are `False` but not that the code behind them works; nothing checks that
    `check_alerts()` is called with arguments, that `NTFY_TOPIC` is set, that the cost filter is
    reachable, or that `n_oos_days == n_folds × oos_days`. Every §1 finding is invisible to it.
