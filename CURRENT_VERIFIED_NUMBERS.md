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

Live since **2026-06-04** (~2.5 weeks). **No multi-year track record exists for this layer.**

| Fact | Value | Source | Verified |
|---|---|---|---|
| Authority level | **Level 1 (Analyst), 5% budget** | `data_cache/earned_authority.json` | 2026-06-22 |
| Completed **scheduled** rebalances participated in | **1 of 1** (June 10 only; next June 24) | `rebalance_calendar.csv` + `ai_pm_decision_log.jsonl` + `counterfactual_ai_snapshots.jsonl` | 2026-06-22 |
| (June 15 & 22 order submissions were off-calendar **discovery mini-rebalances** — AI PM did not enter Phase 2; not bugs) | — | `eod_log.jsonl` `trigger: discovery` | 2026-06-22 |
| Pure-AI-PM vs pure-quant (D−A★) | **−6.52pp over 23 common days** (stable) | `logs/counterfactual_daily.jsonl` | 2026-06-22 |
| Actual book vs pure-quant (B−A★) | **−6.57pp over 32 common days** (⚠️ UNSTABLE) | `logs/counterfactual_daily.jsonl` | 2026-06-22 |
| Promotion gates (L1→L2) | **failing 4 of 7** (sortino_edge −3.73 vs +0.20; hit_rate 0; profit_factor 1.0; decisions_evaluated 0) | `data_cache/ai_pm_perf_feedback.json` | 2026-06-22 |

Caveats:
- **B−A★ is not settled.** It was −5.27pp/31d in the logged file; running the standard
  daily backfill once filled a verified-correct April-16 cell and moved it to −6.57pp/32d
  (applied to the log 2026-06-22, backup `counterfactual_daily.jsonl.bak-astar-fill-*`).
  It keeps drifting as the heal reaches older rows and as n grows (n≈32, a single ±3% day
  moves it ~0.3–0.5pp). Treat as directional, low confidence.
- All counterfactual numbers dated before ~2026-06-20 are unreliable (pre self-heal repair).
- Authority gate thresholds are **held constant** (`ai_pm_perf_feedback.py:240–243`, one commit
  since creation, never loosened); the system is correctly refusing promotion.
- Net: the AI layer is currently **value-neutral-to-negative on a tiny sample**. Strongest
  honest framing is the *governance discipline*, not results.

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
4. Decide whether the AI layer's −6.5pp counterfactual at June 24 (next scheduled rebalance, n grows) warrants action.
