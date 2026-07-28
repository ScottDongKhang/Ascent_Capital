# Ascent Capital — Strategy Methodology

**Version:** 1.0 | **As of:** 2026-05-10 | **Inception:** April 1, 2026

---

## Executive Summary

Ascent Capital is an AI-native systematic multi-factor equity strategy that combines academic quant factors with real-time event intelligence and an LLM-powered risk circuit breaker. The system runs on Alpaca Markets paper trading (transitioning to live capital) and covers four asset class universes managed by independent specialist agents: US equities (S&P 500 + S&P 400, 901 symbols), macro (12 ETFs), international (12 ETFs), and alternatives (7 ETFs).

**Live since:** April 1, 2026  
**Walk-forward OOS record:** Sharpe 0.41, CAGR +10.3%, max drawdown −32.9%, beta 0.73, regression alpha +2.24%/yr vs SPY (OOS 2021-01-08 → 2026-01-14, 1,134 days, 21 rolling folds).
Walk-forward efficiency is −0.65, meaning the in-sample optimizer adds no out-of-sample value; this is disclosed rather than presented as a strength.
Source artifact: `outputs/wf_results/wf_report_clean_2026-06-22.json`. See `CURRENT_VERIFIED_NUMBERS.md` for the authoritative set and its caveats.
*(A prior version of this line reported CAGR +12.35% / Sharpe 0.518 for a Jan 2020 – Apr 2026 window. Those figures matched no artifact in the repository and are withdrawn.)*  
**Operator:** Scott Dong  
**Platform:** Alpaca Markets (paper → live migration in progress)

---

## Data Sources

| Source | What | Lag | Point-in-Time |
|--------|------|-----|---------------|
| Yahoo Finance | Daily OHLCV, 901 US equity symbols | 0 days | Historical closes only; no forward-fill across rebalance dates |
| FRED | 10+ macro series (yield curve, CPI, unemployment, oil) | 1 day | Series vintage as of fetch date |
| EDGAR 10-K/10-Q | MD&A full text, Haiku 5-axis classifier | 45 days | Filing date + 45-day buffer; no look-ahead |
| EDGAR 8-K | Earnings transcripts, event classifier | 1 day | Filing timestamp; 1-business-day execution lag |
| Capitol Trades | Congressional stock trades | 2 days | eFD disclosure timestamp |
| Reddit (PRAW) | Mention counts across 4 subreddits | 0 days | Daily snapshot |
| Google Trends | Search velocity via pytrends | 1 day | Weekly refresh, 1-day lag |
| Alpaca Options | IV z-scores, put/call ratios | 0 days | Kill-switched off pending validation |
| Alpaca WebSocket | 1-minute bars (streaming) | Real-time | Plan 6 streaming infrastructure |

**Failure handling:** If any data source fails, the affected alpha sleeve is zeroed for that rebalance. Simulated fallback data is never used under a live cache name — cache naming convention enforces data provenance.

---

## Alpha Sources

### 1. Trend (41% weight)
**Basis:** Jegadeesh & Titman (1993), Asness et al. (2014) momentum premium.  
**Signal:** Cross-sectional 12M-1M momentum (`mom_252d - mom_21d`). Skip-last-month variant at 20% sub-weight reduces short-term reversal contamination.  
**IC validation:** Positive Spearman IC across 6-year walk-forward on 901-symbol universe.  
**Distressed filter:** Names with `mom_252d < -0.65` (down >65% YoY) receive zero weight.

### 2. Stat-Arb (15% weight)
**Basis:** Industry-adjusted returns (Grundy & Martin 2001); sector residual mean reversion.  
**Signal:** Cross-sectional z-score of residual from sector mean; requires `data_cache/profiles.parquet` (93% sector coverage).

### 3. ML Sleeve — XGBoost (10% weight)
**Basis:** Gu, Kelly, Xiu (2020) machine learning for asset pricing.  
**Method:** CPCV C(6,2)=15 folds, 5-day purge, 5-day embargo. 6 features selected by IC/IR: `mom_skip1m, zscore_20d, high_52w_pct, mom_126d, vol_63d, earnings_surprise`. XGBoost with L1/L2 regularization.  
**Guard:** p5 IC Sharpe across folds > -0.05; sleeve disabled if criterion fails.  
**Retraining:** Every 21 trading days; cache fingerprint auto-invalidates on feature set changes.

### 4. Mean Reversion (5% weight)
**Basis:** Short-term reversal (Lehmann 1990).  
**Signal:** 20-day z-score of returns. Long recent underperformers, short recent outperformers.

### 5. Volatility (5% weight)
**Basis:** Ang et al. (2006) idiosyncratic volatility puzzle; low-volatility anomaly.  
**Signal:** `-(vol_trend_10d / vol_of_vol_21d)`. Long names with declining AND stable volatility — orthogonal to momentum.

### 6. Fundamental (5% weight)
**Basis:** Novy-Marx (2013) gross profitability, Sloan (1996) accruals anomaly, Cooper et al. (2008) asset growth.  
**Signal:** Cross-sectional blend of gross profitability, accruals (negative), asset growth (negative). 45-day filing lag. No momentum signal contamination (52-week high removed).

### 7. Earnings / PEAD (5% weight)
**Basis:** Bernard & Thomas (1989) post-earnings announcement drift.  
**Signal:** EPS surprise z-score, OLS momentum-beta residual (removes momentum contamination). 1-business-day announcement lag.

### 8. Analyst Revision (5% weight)
**Basis:** Jegadeesh et al. (2004) analyst revision momentum.  
**Signal:** Sparse; zero-filled if cache absent. Active when analyst data cache populated.

### 9. LLM Fundamental (3% weight)
**Basis:** Anthropic Claude Haiku (claude-haiku-4-5-20251001) 6-step Chain-of-Thought following Chicago Booth framework.  
**Signal:** Per-quarter assessment of earnings quality, moat strength, management signals, balance sheet risk. Cached by `(symbol, quarter_end)`; 45-day filing lag.  
**IC tracking:** Per-quarter IC logged to `logs/llm_fundamental_signals.jsonl`.

### 10. Options Flow (2% weight)
**Basis:** Pan & Poteshman (2006) informed options trading.  
**Signal:** IV z-score and put/call ratio z-score. Sparse — kill-switched off pending Alpaca options access.

### 11. Insider Transactions (2% weight)
**Basis:** Seyhun (1986) insider trading predictability.  
**Signal:** Net insider purchase score. Sparse; zero-filled if cache absent.

### 12. Short Interest (2% weight)
**Basis:** Dechow et al. (2001) short sellers as informed traders.  
**Signal:** Short squeeze indicator: high short interest + recent momentum. Sparse.

### 13. Alternative Data (0% — gated)
**Basis:** SEC full-text (10-K/10-Q MD&A), earnings transcripts, Reddit sentiment, Google Trends.  
**Gate:** IC_mean ≥ 0.015, IC_IR ≥ 0.60 (Harvey FDR), IC_min_regime > 0.010, n ≥ 20 observations. No source yet validated. Proposals written to `outputs/altdata_proposals/` for human review before deployment.

---

## Regime System

**Model:** Hidden Markov Model with K=2–4 states (best via walk-forward cross-validation).  
**States:** `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`.  
**Features:** Realized volatility, VIX term structure, credit spreads (HYG/LQD), yield curve slope (TLT/IEF), SPY momentum.  
**Hysteresis:** Enter threshold 0.55, exit 0.35, minimum dwell 3 days, entropy > 0.90 → `uncertain`.  
**Particle filter:** 500-particle SIR resampling; reinitializes on batch refit.  
**Emergency refit triggers:** SPY -3%+VIX>30, SPY 200MA cross, SPY/TLT correlation flip, regime break z-score > 3.5. Scheduled refit every 5 days.

**Regime effects:**
- Sleeve weight mix adjusted by regime (crisis → higher trend/vol allocation)
- Max position size: crisis → 8%, calm_bull → 15%
- SPY 200MA overlay: exposure ×0.70 when SPY < 200MA
- Orchestrator capital allocation: crisis → 60% macro + 40% blended
- VIXY hedge overlay: sized 0–8% by regime × confidence

---

## Portfolio Construction

### Primary Path: MVO with Black-Litterman (Plan 2)
1. **Sector pre-screening:** Rank alpha scores within each sector; select top name per sector (max_per_sector=1).
2. **Black-Litterman blending:** Quant prior (rank-weight) blended with LLM views (alpha scores). Blending weight `τ` by IC IR: IR < 0.30 → τ=0.05, IR < 0.60 → τ=0.10, else τ=0.15.
3. **MVO optimization:** `maximize w'α - λ(w'Σw) - κ‖w-w₀‖₁` via cvxpy (CLARABEL solver, SCS fallback). Σ from Barra-style factor risk model (FF5+UMD + Ledoit-Wolf shrinkage).
4. **Factor constraints:** Regime-conditional bounds on Mkt-Rf, SMB, HML exposures (crisis tightest).
5. **Fallback:** If MVO infeasible → rank-weight fallback.

### Rank-Weight Fallback (`sector_constrained_weighted`)
Water-fill iterative cap enforcement (≤50 iterations): `max_weight=0.10`, redistribute from capped names to uncapped until all weights below cap. Hard clamp + renormalize. Post-condition: weights sum to 1.0 ± tolerance.

**Sector coverage guard:** If sector labels cover < 80% of candidates, skip sector caps and log warning — never collapse to single name.

---

## Risk Management

| Control | Threshold | Action |
|---------|-----------|--------|
| Kill switch — soft | -8% drawdown | Log + continue |
| Kill switch — hard | -15% drawdown | Abort run, halt orders |
| Alternatives kill | -12% drawdown | Halt alternatives agent |
| Approval gate | >2% NAV per order | Write to pending_approvals.json, wait |
| SPY 200MA overlay | SPY < 200MA | Multiply weights × 0.70 |
| EM+Commodity cap | >20% aggregate | Hard cap after blending |
| Factor exposure bounds | Regime-conditional ±σ | cvxpy constraint in MVO |
| TWAP execution | >5% ADV per trade | Route to equal-spaced child orders (kill-switched off) |
| Market impact check | >10% ADV | Block order; >5% → warn |
| Intraday trigger | SPY -3%+VIX>30 or DD≥12% | De-risk ×0.70 or −20% exposure |

**VaR/CVaR:** Computed daily from factor model; reported in operator dashboard.

---

## Execution

**Broker:** Alpaca Markets (paper trading → live migration in progress).  
**TWAP Executor:** Kill-switched off (`TWAP_ENABLED=False`). When enabled: Almgren-Chriss optimal window sizing (`T* = √(participation / (2η·σ²))`), equal-spaced child limit orders, 5% ADV gate.  
**Implementation Shortfall:** IS decomposition (delay cost / market impact / opportunity cost in bps) logged to `logs/slippage_log.jsonl`.  
**Capacity Model:** Per-sleeve max AUM before signal decay from market impact. Informational only; logged weekly to `logs/capacity_log.jsonl`.  
**Almgren-Chriss parameters:** η = 0.142 (temporary impact coefficient), σ = daily vol.

---

## Debate Layer (Advisory Circuit Breaker)

Runs on every rebalance day before order submission. **Advisory only — never writes to alpha, portfolio, or execution modules directly.**

**Sequence:** Score past verdicts → post-trade debrief → blind spot detection → catalyst scan → Monte Carlo simulation → multi-agent debate → judge verdict.

| Agent | Model | Private Context |
|-------|-------|-----------------|
| Bull | claude-opus-4-6 | Portfolio weights, alpha scores, factor exposures |
| Bear | claude-opus-4-6 | Drawdown history, risk metrics, IS decomposition |
| Devil's Advocate | claude-opus-4-6 | Monte Carlo tail percentiles, regime breaks |
| Regime Specialist | claude-haiku-4-5-20251001 | Regime features, historical regime transitions |
| Quant Sanity | Pure Python | Weight sum, concentration, turnover |

Round 2 rebuttals: bull/bear/devil respond to each other before judge synthesizes.

**Verdicts:** `proceed` → execute normally | `reduce_size` → Haiku adjusts weights | `halt_and_review` → persist to halt_state.json.

**Disagreement scorer:** TF-IDF cosine similarity across agent reasoning traces. Score written to verdict JSON. Used to validate that private context subsets produce meaningfully different agent perspectives.

**Debate gate:** Only runs when conditions merit it (entropy >0.70, top position >12%, VaR <-3.5%, or catalyst event). Saves ~3–5 minutes on quiet days.

---

## Self-Improvement Loop

**Schedule:** Weekly (Sunday 6 AM).  
**Method:** Generate 5 sleeve-weight variants by perturbing weights via LLM-guided hypothesis generation (Haiku) and random perturbation. Evaluate each via real multi-fold expanding-window OOS (`run_lightweight_oos()`).  
**Promotion gate:** Edge > 0.05 Sharpe over baseline → 30-day shadow monitoring period → auto-promotion to `active_alpha_config.json`.  
**Current state:** `SELF_MODIFY_ENABLED = False`. Gate condition: positive OOS Sharpe for 30 consecutive trading days on flat config (~July 2026).  
**Per-regime variants:** Separate alpha weight configs for calm_bull / stressed / crisis.

**Factor Discovery (monthly, first Sunday):**  
Two paths: (A) PySR symbolic regression on pre-computed feature panel; (B) Haiku proposes JSON template parameters for 5 template families (no code injection).  
**IC gate:** IC_mean ≥ 0.015, IC_IR ≥ 0.60 (Harvey FDR), IC_min_regime > 0.010. Accepted proposals written to `outputs/factor_proposals/` for human review before deployment.

---

## Performance Attribution

**Daily:** `ascent/monitoring/attribution.py` computes:
- **Factor-explained P&L:** β(factor loadings) × factor returns
- **Idiosyncratic P&L:** residual after factor attribution
- Written to `logs/attribution_log.jsonl`

**IS Decomposition:**
- **Delay cost:** (arrival_price - decision_price) / decision_price × 10,000 × direction
- **Market impact:** (avg_fill_price - arrival_price) / decision_price × 10,000 × direction
- **Opportunity cost:** cost of unfilled shares

**Slippage IC feedback:** Weekly Spearman IC comparison of gross vs net-of-slippage alpha scores. Drag coefficient written to `active_alpha_config.json`. Active when MIN_FILLS=50 (~July 2026).

---

## Compliance and Audit

**Audit trail:** Every decision — signal generation, portfolio construction, debate verdict, order submission, fill, kill switch — is recorded in `logs/audit_trail.jsonl` with a SHA-256 hash chain. Any tampering with historical entries invalidates all subsequent hashes.

**Monthly integrity check:** `scripts/verify_audit_trail.py` verifies the full chain. Results logged to `logs/audit_integrity.jsonl`. Failures trigger CRITICAL alert but do not halt trading.

**Record retention:** All logs are version-controlled in git (gitignored from remote for privacy; local backup required). Audit trail is append-only.

**No regulatory registration:** This strategy is operated on the operator's own capital. No outside capital is accepted at this time. Before accepting outside capital, securities counsel must advise on applicable exemptions under the Investment Advisers Act of 1940.

---

*This document is version-controlled. Update after any material strategy change. Do not distribute to third parties without legal review of risk disclosures.*
