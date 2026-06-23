# CURRENT_VERIFIED_NUMBERS.md — single source of truth

**This is the only file to quote performance numbers from.** Every figure is tagged with
(a) which system it describes, (b) the exact source artifact/log, (c) the date last
verified. If a number elsewhere (README, CLAUDE.md, dashboard, outreach) disagrees with
this file, **this file wins** and the other is stale.

Two systems are kept strictly separate:
- **SYSTEM 1 — quant engine** (alpha sleeves, regime, optimization, walk-forward backtest). Multi-year.
- **SYSTEM 2 — AI-native layer** (debate, AI PM, earned authority, counterfactual A★/A/B/C/D). Live since 2026-06-04.

Last reconciled: **2026-06-22**.

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
- The engine's reported **Sortino (0.042) is miscomputed** (known bug; the old artifact also showed a wrong 0.050). Independent downside-deviation Sortino ≈ **0.68**. Do not cite the engine Sortino field.
- I switched `close` to **total-return adjusted** (the original used split-only close). This adds dividends ~equally to strategy and SPY → it mainly lifts absolute CAGR and leaves Sharpe/alpha ~unchanged. Even so, the clean CAGR (10.3%) is **below** the corrupted run's 12.61% — confirming the corruption was inflating returns by more than the dividend boost.
- **Supersedes** "Sharpe 0.483 / CAGR +12.61%" (corrupted cache, inflated) and the contaminated −0.14 repaired run. Prior values −23.4% DD and +2.54% alpha matched no artifact and are dead.

**Production cache note:** the clean data was written to a *staging* file (`prices_live_clean_refetch.parquet`), **not** swapped into the live `prices_live.parquet`, because the clean re-fetch uses total-return-adjusted close which would change live momentum signals right before the June 24 rebalance. The corrupted cache is backed up (`data_cache/_corrupt_backup_20260622-222216/`) and the live pipeline dedupes on read. Swapping production is a separate, deliberate decision (flagged for the user).

---

## 2. Live paper-trading book (SYSTEM 1 + 5% SYSTEM 2, blended actual account)

Alpaca paper, live since **2026-04-01**. Window 2026-04-01 → 2026-06-22 (~62 trading days).

| Metric | Value | Source | Confidence |
|---|---|---|---|
| Current equity | **$109,710** | `logs/holdings_log.jsonl` (last row) | high |
| Total return (equity-based, since Apr 1) | **+8.82%** | `holdings_log` equity 100,815→109,710 | high (verified) |
| SPY, same window | ≈ **+13.2%** | `holdings_log` `spy_return` cumulative | medium |
| Book vs SPY | ≈ **−4.4 to −5.1pp** (book trails the index) | derived; README states −5.08% | medium |
| Max drawdown | ≈ **−5.7% to −6.6%** (method-dependent) | equity-based −5.67%; dashboard −6.58% | medium |
| Annualized Sharpe | **NOT reliably verifiable** (dashboard shows 1.794; raw daily-return calc gives 0.38) | `generate_performance_page.py` vs `holdings_log` | LOW |

Caveats: the `holdings_log` **daily-return** series is unreliable (Alpaca intraday-settlement
issue documented in CLAUDE.md), so the dashboard Sharpe (1.794) cannot be independently
reproduced and the daily-return cumulative is wrong — only the **equity-based total return
(+8.82%)** is solid. At ~62 days, **no Sharpe here is statistically significant.** The ~5pp
lag vs SPY is structural (defensive non-equity sleeves + 200MA/vol-target overlays in an
equity bull), not an AI-layer effect.

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

## 4. Data integrity status

| Item | Status |
|---|---|
| `prices_live` cache | **CORRUPTED** — ~59% duplicate rows (two sources `yahoo`+`yfinance_hub` the tz-keyed dedup never collapsed) + 10×-type wrong values in 12 symbols (CAR, CHRD, CYTK, FLR, GME, KLAC, LUMN, MTDR, OVV, PR, SATS, SHC). Spans 2020–2026. |
| Live pipeline exposure | **Low** — `ascent/main.py` dedupes on read (`[~s.index.duplicated(keep="last")]`, lines 393/745/789); WF framework does **not**, which is why the backtest blew up (CAGR 10¹³%) but live did not. |
| Ingest bug | **FIXED (code)** 2026-06-22 — `ascent/data/store/parquet.py` now normalizes the `date` dedup key to calendar day so future writes stop accumulating dups (`tests/test_parquet_store_dedup.py`, 3 tests). **Does NOT clean existing rows or re-fetch** — that is the outstanding remediation. |
| Counterfactual log | Repaired + April-16 cell filled 2026-06-22 (backups in `logs/`). Self-heals each run but B−A★ still drifting (§3). |

---

## 5. Outstanding before any performance claim is made externally
1. ~~Re-fetch `prices_live` clean → re-run WF → get the real OOS number.~~ **DONE 2026-06-22** — Sharpe 0.41 / CAGR +10.3% verified (§1). Staging cache only; production cache not yet swapped (see §1 production note).
2. ~~WF-results header "CPCV C(6,2) = 15 folds" line~~ **CORRECTED** — the WF backtest is rolling 252d-IS/63d-OOS, **21 folds** (README updated). Still unverified: the *ML-sleeve internal* CV line in the alpha-stack table (`ML (XGBoost/CPCV) C(6,2)=15`) — that refers to a different thing (the ML sleeve's own cross-validation) and has not been checked against the code.
3. (Optional production step) Decide whether to swap the clean staging cache into live `prices_live.parquet` — changes live momentum signals to total-return-adjusted close; do deliberately, not before June 24.
4. Decide whether the AI layer's −6.5pp counterfactual at June 24 (next scheduled rebalance, n grows) warrants action.
