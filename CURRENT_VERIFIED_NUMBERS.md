# CURRENT_VERIFIED_NUMBERS.md — single source of truth

**This is the only file to quote performance numbers from.** Every figure is tagged with
(a) which system it describes, (b) the exact source artifact/log, (c) the date last
verified. If a number elsewhere (README, CLAUDE.md, dashboard, outreach) disagrees with
this file, **this file wins** and the other is stale.

Two systems are kept strictly separate:
- **SYSTEM 1 — quant engine** (alpha sleeves, regime, optimization, walk-forward backtest). Multi-year.
- **SYSTEM 2 — AI-native layer** (debate, AI PM, earned authority, counterfactual A★/A/B/C/D). Live since 2026-06-04.

**Sections 2 and 4 are machine-generated.** Regenerate them before quoting anything:

```
.venv/bin/python scripts/reconcile_numbers.py --write   # recompute from artifacts
.venv/bin/python scripts/reconcile_numbers.py --check    # fail if stale
.venv/bin/python scripts/verify_docs.py                  # check docs against code
```

They were hand-maintained until 2026-07-28, went five weeks stale, and lost to CLAUDE.md on
this file's own tiebreak rule. Anything inside the generated markers is overwritten; put
judgment and caveats outside them.

---

## 1. Quant engine — Walk-Forward OOS backtest (SYSTEM 1)

**STATUS: ✅ VERIFIED (2026-06-22) on a freshly re-fetched clean price cache.**

| Metric (OOS) | Value | Source | Verified |
|---|---|---|---|
| Sharpe ratio | **0.41** (engine 0.412; independent recompute 0.417) | `wf_report_clean_2026-06-22.json` | 2026-06-22 |
| CAGR | **+10.3%** (engine 0.1030; independent recompute +10.4%) | same | 2026-06-22 |
| Excess CAGR vs SPY | **+1.0pp** (strategy 10.42% − SPY 9.41%, identical window) | `wf_equity_clean_2026-06-22.csv` | 2026-06-22 |
| Regression alpha vs SPY | **+2.24%/yr** (annualized intercept) | `wf_report_clean_2026-06-22.json` | 2026-06-22 |
| Max drawdown | **−32.9%** | same | 2026-06-22 |
| Beta vs SPY | **0.73** | same | 2026-06-22 |
| Win rate | **50.2%** | same | 2026-06-22 |
| OOS window | **2021-01-08 → 2026-01-14** (1134 OOS days, 21 rolling folds) | same | 2026-06-22 |
| Walk-forward efficiency | **−0.65 (OVERFIT — disclose)** | same | 2026-06-22 |

**How this number was produced (and why it's now trustworthy):**
- Price cache re-fetched fresh from a **single source** (yfinance, `auto_adjust=True` → consistently split+dividend adjusted OHLC), 936 symbols, 2020-01-02 → 2026-06-22, into `data_cache/prices_live_clean_refetch.parquet`. Verified clean: **0 duplicate (symbol,date) rows, 0 implausible jumps** (worst remaining single-day moves are all real events — GME meme days, CAR +108% Nov-2021 squeeze, SHC +99.7% Jan-2023 settlement, LUMN +93% Aug-2024). Only **1 symbol dropped** (CHRD — irreparable source-side ticker-reuse history), vs the corrupted run's 12.
- LLM alpha sleeves **explicitly zeroed** (`alpha_overrides={"llm_fundamental":0.0,"narrative":0.0}`; runner logged `skipped=[... 'llm_fundamental' ...]` every fold). WF code path is pure quant — no `ai_pm`/`debate`/`counterfactual`/`anthropic` references in `ascent/research/wf_framework/` (grep-verified).
- **Same window and fold structure (1134 OOS days, 21 folds, beta 0.733) as the canonical `wf_report_2026-06-04.json`** → a true apples-to-apples comparison; the only thing that changed is data cleanliness.

**Methodology notes / caveats — read before citing:**
- **Walk-forward efficiency is −0.65:** the in-sample param optimizer adds no OOS value (fixed params would do as well). The Sharpe above is the realistic *post-overfit* OOS figure, but the overfit flag must be disclosed alongside it.
- This is a **modest** edge (Sharpe ~0.4) and a **thin** +1pp/yr excess return over SPY at defensive beta (0.73). A single backtest, **not a live track record**.
- **Sortino: the stored `0.042` in this artifact is wrong; the correct value for this run is ≈0.67.** Root cause found and fixed on 2026-07-28: `PerformanceAnalyzer.sortino` annualized both the numerator and the downside deviation, dividing every result by √252 (`0.042 × √252 = 0.666`). Fixed in `ascent/research/wf_framework/metrics.py`, pinned by two regression tests in `tests/test_wf_framework/test_metrics.py`, and guarded by `verify_docs.py::sortino_annualized_once`. **Artifacts written before 2026-07-28 still carry the wrong field** — multiply by √252, or re-run. Newly written artifacts are correct.
- I switched `close` to **total-return adjusted** (the original used split-only close). This adds dividends ~equally to strategy and SPY → it mainly lifts absolute CAGR and leaves Sharpe/alpha ~unchanged. Even so, the clean CAGR (10.3%) is **below** the corrupted run's 12.61% — confirming the corruption was inflating returns by more than the dividend boost.
- **Supersedes** "Sharpe 0.483 / CAGR +12.61%" (corrupted cache, inflated) and the contaminated −0.14 repaired run. Prior values −23.4% DD and +2.54% alpha matched no artifact and are dead.

**Production cache note:** the clean data was written to a *staging* file (`prices_live_clean_refetch.parquet`), **not** swapped into the live `prices_live.parquet`, because the clean re-fetch uses total-return-adjusted close which would change live momentum signals right before the June 24 rebalance. The corrupted cache is backed up (`data_cache/_corrupt_backup_20260622-222216/`) and the live pipeline dedupes on read. Swapping production is a separate, deliberate decision (flagged for the user).

---
<!-- BEGIN GENERATED live-book: reconcile_numbers.py -->

*Regenerated by `scripts/reconcile_numbers.py`. Do not hand-edit between the markers; edits are overwritten. Last regenerated: 2026-07-28.*

## 2. Live paper-trading book (SYSTEM 1 + SYSTEM 2 blended actual account)

| Metric | Value | Method | Source |
|---|---|---|---|
| Window | 2026-04-01 -> 2026-07-27 (77 logged rows) | rows at/after 2026-04-01 | `logs/holdings_log.jsonl` |
| Current equity | $104,640.21 | last logged row | `logs/holdings_log.jsonl` |
| Total return | +3.79% | equity 100,816 (2026-04-01) -> 104,640 | `logs/holdings_log.jsonl` |
| SPY, same window | +12.67% | close-to-close, 78 bars, 2026-04-01 -> 2026-07-23 | `data_cache/prices_live.parquet` |
| Book vs SPY | -8.88% | difference of the two above | derived |
| SPY per the log's own column | +4.67% | cumulative `spy_return` over 76 rows | `logs/holdings_log.jsonl` |
| Max drawdown | -7.29% | equity-based, peak 112,870 (2026-06-02) -> trough (2026-07-27) | `logs/holdings_log.jsonl` |
| Open positions | 22 | last logged row | `logs/holdings_log.jsonl` |
| Annualized Sharpe | NOT COMPUTABLE | see note below | - |

**Sharpe is deliberately absent.** At 77 sessions the standard error swamps the estimate, and 4 of those rows carry a `day_return` of exactly 0.0 — the Alpaca late-settlement artifact, not flat days. Any Sharpe computed from that column is meaningless. Do not reintroduce one here.

**The two SPY figures disagree** (+12.67% from the price cache vs +4.67% from the log column). The cache figure is the one to quote: the log's `spy_return` column inherits the same missing-day problem as `day_return`, so its cumulative product understates the index. Both are shown so the gap stays visible.


### Walk-forward artifact cross-check

Section 1 above must match `outputs/wf_results/wf_report_clean_2026-06-22.json`:

    Sharpe 0.41, CAGR +10.3%, max DD -32.9%, beta 0.73 (OOS 2021-01-08 -> 2026-01-14, 21 folds, 1134 days) [outputs/wf_results/wf_report_clean_2026-06-22.json]

- WFE is negative: the in-sample optimizer adds no out-of-sample value. Disclose as overfit.
- LLM-driven sleeves were zeroed for this run.

<!-- END GENERATED live-book -->

---

## 3. AI-native layer (SYSTEM 2)

Live since **2026-06-04**. **No multi-year track record exists for this layer.**

**STATUS: ✅ counterfactual log REBUILT 2026-07-28.** The figures below supersede everything
previously reported here and in `CLAUDE.md`. The old log was unusable — see "What was wrong"
below — and the numbers it produced (`B−A★ = −7.82pp`, `A★ +23.59%`, `SPY +16.63%`) must not
be quoted again from any source.

| Fact | Value | Source | Verified |
|---|---|---|---|
| Authority level | **Level 1 (Analyst), 5% budget**, 19 days at level | `data_cache/earned_authority.json` | 2026-07-28 |
| Completed **scheduled** rebalances participated in (Phase 2) | **2** — 2026-06-10, 2026-06-24 | `logs/ai_pm_decision_log.jsonl` (2 distinct dates) | 2026-07-28 |
| (07-08 and 07-22 scheduled rebalances **never ran** — 19-trading-day outage) | — | `logs/liveness.json`, `logs/eod_log.jsonl` | 2026-07-28 |
| (06-15, 06-22, 06-29 order submissions were off-calendar **discovery mini-rebalances** — no Phase 2) | — | `eod_log.jsonl` `trigger: discovery` | 2026-07-28 |
| Actual book vs pure-quant (B−A★) | **−5.92pp over 70 common days, t = −1.24** (not significant) | `logs/counterfactual_daily.jsonl` | 2026-07-28 |
| Pure-AI-PM vs pure-quant (D−A★) | **−3.04pp over 47 common days, t = −0.94** (not significant) | `logs/counterfactual_daily.jsonl` | 2026-07-28 |
| Promotion gates (L1→L2) | **failing** (n_decisions_evaluated 0, hit_rate 0, profit_factor 1.0, sortino_edge 0.016 vs +0.20 required) | `data_cache/ai_pm_perf_feedback.json` | 2026-07-28 |

### Cumulative track returns, 2026-03-24 → 2026-07-27 (86 trading days)

| Track | Value | Note |
|---|---|---|
| A★ pure quant | **+9.54%** | snapshot weights priced on `prices_live` |
| A quant + Phase-1 priors | **+0.73%** | only 31 non-null days (2 snapshots exist) |
| B actual book | **+4.70%** | settled Alpaca 1D bars |
| C SPY | **+13.13%** | `prices_live` closes |
| D pure AI PM | **−2.12%** | 47 non-null days |

**SPY beats pure quant over this window** (+13.13% vs +9.54%). The old log claimed the
reverse by ~7pp. This is consistent with the documented structural position — ~22% defensive
non-equity sleeves, the 200MA cut and the 15% vol-target overlay all cost beta in an
equity-only bull — but it is no longer masked by an inflated quant figure. Section 2's
independent book-vs-SPY figure (−8.88% over its own window) points the same way.

### Reconciliation — three independent checks, all passing

| Check | Result |
|---|---|
| Track C chained daily vs SPY point-to-point | **+13.13% = +13.13%**, exact |
| Track B chained vs raw Alpaca equity endpoints | **+4.70%**, 100,000.00 → 104,695.25, exact |
| Track B vs §2's `holdings_log` book return, same window (04-01→07-27) | **+3.77% vs +3.79%** — 0.02pp, different sources |

A chained series equals point-to-point only when there are no gaps and no misalignment,
which is precisely the check the old `+16.63%` SPY figure failed.

**Alignment:** same-day `corr(B, A★)` is now **+0.936**, lag-1 **−0.118**. Before the rebuild
those read −0.005 and +0.60 — two books sharing 95% of their holdings appearing uncorrelated
same-day and correlated one day late. That shift *was* the −7.82pp headline.

### What was wrong with the old log

1. **45 rows for 78 trading days.** The backfills could only mutate existing rows, never
   insert, so any day the pipeline did not run was permanently absent — including a 19-day
   outage hole. `_cumret_over` chains across gaps as though those days never happened, which
   inflated every cumulative figure 2.5–4×.
2. **Track B keyed one day late** — `datetime.fromtimestamp(ts)` with no timezone on Alpaca's
   UTC epochs, rendered in host-local time (UTC+7), so a 16:00 ET close became the next
   calendar day. Friday's bar became Saturday and Monday was unreachable (14 Saturdays,
   1 Monday on the published page).
3. **A 2026-06-19 row** carrying `track_b +1.53%` / `track_c +1.04%` on Juneteenth, with the
   market shut.

Rebuilt by `scripts/rebuild_counterfactual_log.py` (logic in
`ascent/monitoring/counterfactual_rebuild.py`, 17 tests; dry-run by default). 86 rows, all 86
expected trading days present, no duplicates, no weekends, no holidays. Previous log backed up
to `logs/counterfactual_daily.pre_rebuild.20260728-114026.bak.jsonl`.

Caveats:
- **Still not statistically significant.** t = −1.24 (B−A★) and −0.94 (D−A★). More
  importantly the daily observations are not independent evidence: weights are frozen between
  rebalances, and all of it descends from **2** Phase-2 decisions. Effective n is closer to 2
  than to 70.
- **A★/A/D cannot be reconciled against an external source.** They are reconstructions from
  snapshot weights priced on `prices_live`, using the same method `score_daily` uses live. Only
  B and C have outside confirmation.
- **One asymmetry, disclosed not corrected:** Alpaca equity is total-return, while production
  `prices_live` closes are split-only, so B is not perfectly comparable to A★/D on
  dividend-paying names. Pre-existing — `score_daily` reads the same two sources.
- **Track A is thin.** Only 2 quant snapshots exist, so its 31 non-null days and +0.73% carry
  much less weight than the other tracks.
- Authority gate thresholds are **held constant** (`ai_pm_perf_feedback.py`, never loosened);
  the system is correctly refusing promotion. As of 2026-07-28 `n_decisions_evaluated` is
  reachable for the first time (overrides are now derived from weight deltas rather than
  self-reported), so the gates can begin to accumulate real samples.
- Net: the AI layer remains **value-neutral-to-negative on a sample too small to judge**. The
  honest framing is still the governance discipline, not results — but note the transmission
  defects found on 2026-07-27 mean the layer's *judgment* has never been cleanly measured.
  See `docs/AI_PM_DIAGNOSIS_2026-07-27.md`.

---
<!-- BEGIN GENERATED data-integrity: reconcile_numbers.py -->

*Regenerated by `scripts/reconcile_numbers.py`. Do not hand-edit between the markers; edits are overwritten. Last regenerated: 2026-07-28.*

## 4. Data integrity status

`data_cache/prices_live.parquet` as measured now:

| Property | Value |
|---|---|
| Rows | 1,517,608 |
| Distinct symbols | 938 |
| Date range | 2020-01-02 -> 2026-07-24 |
| Duplicate (symbol, market-calendar-day) rows | **0** (0.0% of rows), across 0 symbols |
| Source generations present | `yfinance_hub` 1,502,745, `yahoo_hub` 7,920, `yfinance_split_only_repair` 6,592, `yahoo` 351 |

**No duplicates.** Any backtest run on this cache is clean on that axis.

More than one source generation is blended in this cache. Mixed adjustment bases (split-only vs total-return) across generations produce fake jumps, which is what drove a 70% alpha sleeve to zero once already.

<!-- END GENERATED data-integrity -->

---

## 5. Outstanding before any performance claim is made externally
1. ~~Re-fetch `prices_live` clean → re-run WF → get the real OOS number.~~ **DONE 2026-06-22** — Sharpe 0.41 / CAGR +10.3% verified (§1). Staging cache only; production cache not yet swapped (see §1 production note).
2. ~~WF-results header "CPCV C(6,2) = 15 folds" line~~ **CORRECTED** — the WF backtest is rolling 252d-IS/63d-OOS, **21 folds** (README updated). Still unverified: the *ML-sleeve internal* CV line in the alpha-stack table (`ML (XGBoost/CPCV) C(6,2)=15`) — that refers to a different thing (the ML sleeve's own cross-validation) and has not been checked against the code.
3. (Optional production step) Decide whether to swap the clean staging cache into live `prices_live.parquet` — changes live momentum signals to total-return-adjusted close; do deliberately, not before June 24.
4. ~~Decide whether the AI layer's −6.5pp counterfactual warrants action.~~ **SUPERSEDED 2026-07-28** — that figure was a measurement artifact. The rebuilt number is −5.92pp/70d at t = −1.24, from 2 Phase-2 decisions: still not actionable, but for a different reason (no signal yet, rather than a bad signal). Do **not** change the 5% authority budget on this evidence; re-ask after 8–10 cleanly scored rebalances, which is now possible for the first time. See §3 and `docs/AI_PM_DIAGNOSIS_2026-07-27.md`.
5. **Re-run the counterfactual rebuild once a few correct daily runs have accumulated.** The rebuild is reproducible (`scripts/rebuild_counterfactual_log.py`, dry-run by default) and idempotent on unchanged sources, so it can be re-applied to extend the series rather than being a one-off repair.
6. **Track A needs more snapshots.** Only 2 exist, so the "quant + Phase-1 priors" track is the weakest of the five and its +0.73% should not be leaned on.
