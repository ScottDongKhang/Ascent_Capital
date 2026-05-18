# Ascent Capital — AI-Native Quantitative Fund

![Tests](https://img.shields.io/badge/tests-492%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![AI PM](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)
![Debate](https://img.shields.io/badge/debate-Sonnet%204.6-purple)

**An AI portfolio manager that earns trading authority by proving itself in live markets — then attacks its own ideas before submitting.**

The AI PM starts at 0% allocation. It shadow-trades alongside the quant system, building a live return history. After 21 days of sustained Sharpe edge, it earns 25% weight. It can reach 75%. If its drawdown exceeds the quant baseline by 5 percentage points, it auto-reverts and starts over. The quant system always has a 20% floor.

Before each rebalance, a separate red-team agent attacks the AI PM's proposed portfolio with the hardest possible bear case — worst-case scenario per position, crowding risks, narrative-momentum traps. The AI PM must defend or revise. It also queries an episodic memory of past regime periods to calibrate expectations, and checks its own historical conviction-vs-outcome IC to know when to trust itself.

This is a full production system — not a backtest notebook.

---

## What Makes This Genuinely Different

Most quant repos stop at: data → backtest → publish Sharpe. This one doesn't.

### 1. Adversarial Self-Play Before Every Submission
After the AI PM proposes a portfolio, a dedicated red-team agent (Sonnet) independently attacks it — per-position worst-case scenarios, systemic risk blind spots, narrative-momentum traps. The AI PM then revises or defends before final submission. No fund of any size does this systematically.

### 2. AI PM That Earns Its Authority
The AI PM doesn't start with capital. It shadow-trades for 21 days, building a live Sharpe record. It earns allocation in phases (0% → 25% → 50% → 75%). Underperformance auto-reverts it. Every override of the quant system requires explicit reasoning referencing the signal data — recorded immutably in a SHA-256 hash-chain audit trail.

### 3. Regime-Aware Episodic Memory
Every run logs the current regime + portfolio weights. 21 days later, realized returns are filled in. The AI PM can query: *"What actually happened the last 5 times we were in a stressed regime?"* This is a continuously growing, queryable record of the system's own experience.

### 4. Self-Aware Calibration
The AI PM's conviction levels (high = quant override, medium = own rationale, low = quant agreement) are tracked against realized 21-day returns. Spearman IC between conviction and outcome is computed across all rebalances. The AI PM can check its own calibration before making high-conviction calls. If IC < 0.05, it knows to discount its own confidence.

### 5. Narrative Alpha — Thesis Shift Detection
An LLM compares current vs prior quarter fundamental analyses per name and flags when the investment narrative has shifted. Narrative flips (e.g., "strong execution" → "margin pressure") precede analyst consensus revisions. This signal is zero-weighted until the cache matures — built correctly from day one.

### 6. Multi-Agent Debate With Quantified Tail Risk
Before every rebalance: blind spot detection → catalyst scan → Monte Carlo simulation → Bull / Bear / Devil's Advocate (with actual p5–p95 return numbers) / Regime Specialist / Quant Sanity check. Round 2 cross-rebuttals. TF-IDF disagreement scoring tracked longitudinally. Verdict gates execution: proceed / reduce_size / halt_and_review.

### 7. Honest Walk-Forward OOS
Per-fold `get_universe_on_date()` — survivorship bias eliminated. Fold-local regime fit — no look-ahead from full-sample regime. CPCV (C(6,2)=15 folds) for the ML sleeve. Walk-forward OOS: CAGR +12.35%, Sharpe +0.518, alpha +0.68% vs SPY (Jan 2020 – Apr 2026).

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_agents.py                                               # daily pipeline (launchd 1:45 PM ET)
python -m pytest --tb=short -q                                        # 492 passed, 1 skipped
streamlit run demo_app.py                                             # investor demo (dark/gold, live LLM debate)
streamlit run ascent/dashboard/live_dashboard.py --server.port 8502  # operator dashboard
```

---

## Walk-Forward OOS Record

> Per-fold universe filter · fold-local regime fit · CPCV for ML sleeve · no look-ahead

| Metric | Value |
|--------|------:|
| Period | Jan 2020 – Apr 2026 |
| CAGR | +12.35% |
| Sharpe | +0.518 |
| Alpha vs SPY | +0.68% |
| Max Drawdown | −23.4% |

---

## Live Track Record

| Date | Event |
|------|-------|
| Apr 1, 2026 | Live paper trading begins (Alpaca) |
| Apr 15, 2026 | First rebalance: REDUCE_SIZE verdict, 27 orders |
| May 5, 2026 | Full portfolio rotation: 40 orders |
| May 7, 2026 | NAV $104,815 · Regime: calm_bull |
| May 16, 2026 | AI PM Agent live · adversarial red team · episodic memory · calibration tracking |
| May 17, 2026 | Narrative alpha sleeve added · 492 tests |

**Current holdings (May 2026):** KMLM, IFRA, AMKR, FIX, WDC, VICR, VRT, VAL, WCC, DBB, CNC, WFRD, PDBC, STLD, DBA, CHRD, EWY, EWC, IRM, MUSA + VIXY hedge.

---

## System Architecture

```
python run_all_agents.py  (daily, 1:45 PM ET via launchd)
│
├─ Factor data update (FF5+UMD rolling OLS) · intraday trigger check
│
├─ 4 Specialist Quant Agents (parallel, AgentScope)
│   ├── US Equities  ── 901 symbols (S&P 500 + 400) · 13-sleeve alpha stack
│   ├── Macro        ── 12 ETFs · trend-only · regime-sized exposure
│   ├── International── 12 ETFs · EM-aware (UUP > 50MA → 20% penalty)
│   └── Alternatives ── 7 ETFs · VIXY hedge · kill switch at 12% drawdown
│
├─ Orchestrator
│   ├── Regime base allocations (calm_bull / stressed / crisis)
│   ├── Skill-score blend · conviction bonus (≥2 agents share name → +15%)
│   ├── 63-day cross-agent correlation guard (cap 0.70)
│   ├── Thesis coherence check (12 factor buckets, 6 contradiction pairs → 40% reduction)
│   └── EM+commodity hard cap (20%)
│
├─ AI PM Agent  (Claude Opus 4.6 · 16-tool research loop)
│   ├── Phase 1: Market context (regime state, macro indicators)
│   ├── Phase 2: Quant baseline (all 4 agents via precomputed cache — no re-runs)
│   ├── Phase 3: Signal research (≤6 of 10 tools: SEC, transcripts, attribution,
│   │           earnings, factor exposures, VaR, sector concentration, momentum,
│   │           narrative shift, calibration report)
│   ├── Phase 4: propose_portfolio (weights + full investment memo)
│   │
│   ├── Red Team Pass (Sonnet): attacks proposal → per-position bear case + kill shot
│   ├── Revision Pass (Opus): AI PM defends or revises after red team critique
│   │
│   ├── Episodic memory: query past regime outcomes before proposing
│   ├── Calibration tracking: conviction (high/medium/agreed) vs realized IC
│   └── Pre-blend validator (hard limits: position ≤15%, sector ≤40%, ≥5 names, no shorts)
│
├─ Earned Authority Blend
│   ├── Phase 0: ai_weight=0.0 (shadow) → Phase 3: ai_weight=0.75 (max)
│   ├── Hard cap: 0.80 · quant always ≥20%
│   └── Auto-revert: AI 21d drawdown > quant + 5pp → reset to Phase 0
│
├─ Debate Layer  (rebalance days · advisory only)
│   ├── Score past verdicts → debrief → blind spot detection → catalyst scan
│   ├── Monte Carlo simulation (Mirofish, p5–p95)
│   ├── Round 1: Bull (Sonnet) · Bear (Sonnet) · Devil's Advocate (Sonnet)
│   │           Regime Specialist (Haiku) · Quant Sanity (pure Python)
│   ├── Round 2: Cross-rebuttals · TF-IDF disagreement score
│   └── Judge (Sonnet) → proceed / reduce_size / halt_and_review
│
└─ Execution
    ├── Kill switch: SOFT_WARN 8% · HARD_STOP 15% (alt: 12%)
    ├── Almgren-Chriss cost model (block >10% ADV, warn >5%)
    ├── TWAP executor (kill-switched pending paper validation)
    ├── Alpaca paper trading · IS decomposition · slippage tracking
    └── SHA-256 hash-chain audit trail · GIPS TWR performance reporting
```

---

## Alpha Stack (13 Sleeves)

| Sleeve | Weight | Signal |
|--------|-------:|--------|
| Trend | 41% | Cross-sectional momentum; skip-last-month `mom_252d − mom_21d` at 0.20 sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion (sector z-score of residual returns) |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds · 5-day purge+embargo · 6 features by IC/IR · p5 guard > −0.05 |
| Mean Reversion | 5% | Short-term reversal (z-score 20d) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day lag; momentum-neutral |
| Earnings PEAD | 5% | EPS surprise z-score · OLS momentum-beta residual · 1-bday lag |
| Analyst | 5% | Revision signal; sparse — zero-filled when cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Claude Haiku; cached by (symbol, quarter) |
| Options Flow | 2% | IV-adjusted sentiment; sparse |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Narrative Alpha | 0% | Quarter-over-quarter thesis shift detection; 0% until cache matures |

Weights are regime-adaptive via `active_alpha_config.json`, updated weekly by the self-improve loop. Distressed filter zeroes alpha for `mom_252d < −0.65`.

---

## Portfolio Construction

```
Alpha scores (13 sleeves, cross-sectionally z-scored, regime-weighted)
  → Distressed filter (zero names with mom_252d < −0.65)
  → Black-Litterman  (quant prior + LLM views; tau scales with IC IR)
  → cvxpy MVO        maximize: w'α − λ(w'Σw) − κ‖w − w_prev‖₁
                     Σ = B·F·B' + D  (FF5+UMD factor covariance + idiosyncratic residuals)
                     CLARABEL solver → SCS fallback → rank-weight fallback
  → Sector constraints  (max 1 per sector · fallback: skip caps if coverage < 80%)
  → SPY 200MA overlay   (×0.70 when SPY < 200MA)
  → VIXY hedge overlay  (0–8% by regime × HMM confidence)
```

---

## Regime System

HMM K=2–4 (walk-forward CV selects K). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3d. Entropy > 0.90 → `uncertain`. Particle filter: 500 particles SIR, reinitializes on batch refit.

Emergency refit triggers: SPY −3% + VIX > 30, 200MA cross, SPY/TLT correlation flip, break z-score > 3.5, or > 5 days stale. Leading indicators: credit spread (HYG/LQD), yield curve slope (TLT/IEF).

Regime propagates through every layer: sleeve weights, max position size, orchestrator capital allocation, debate context, AI PM episodic memory queries, VIXY hedge sizing.

---

## Key Systems

**Self-improve** — Sunday 6AM. Generates 5 sleeve-weight variants by LLM-guided hypothesis + random perturbation. Each variant scored via real multi-fold OOS (5-day purge+embargo). Shadow period 30 days, auto-promoted by `shadow_promoter.py`. `SELF_MODIFY_ENABLED=False` until +OOS Sharpe for 30 consecutive trading days.

**Factor discovery** — Monthly (first Sunday). PySR symbolic regression (Path A) + Haiku JSON template proposals (Path B). Both gated by Harvey FDR correction (IC_mean ≥ 0.015, IC_IR ≥ 0.60, positive in every observed regime). Proposals written to `outputs/factor_proposals/` — human review required, nothing auto-deploys.

**Compliance** — SHA-256 hash-chain audit trail (`compliance/audit_trail.py`). Every order, override, regime change, and config modification is chained. `scripts/verify_audit_trail.py` exits 0/1 (CI-ready). Monthly GIPS TWR performance reports. Full methodology doc + risk disclosures in `docs/`.

**R2R Semantic Memory** — R2R HTTP interface with BM25 fallback. IC briefs, self-improve results, and past verdicts are ingested for semantic retrieval. `R2R_API_KEY` not yet configured; BM25 active.

---

## Institutional Roadmap

| Component | Status | What It Adds |
|-----------|--------|--------------|
| Factor Risk Model | ✅ | FF5+UMD rolling OLS, Ledoit-Wolf Σ, factor P&L attribution |
| Portfolio Construction | ✅ | cvxpy MVO, Black-Litterman, regime covariance |
| Event-Driven Architecture | ✅ | EDGAR 8-K, Capitol Trades, options anomaly (`EVENT_TRADING_ENABLED=False`) |
| Alternative Data Pipeline | ✅ | SEC 10-K/Q, transcripts, Reddit, Google Trends; IC gate |
| Execution Excellence | ✅ | TWAP, IS decomposition, capacity model, intraday triggers (kill-switched) |
| Real-Time Infrastructure | ✅ | TimescaleDB, Alpaca WebSocket (901 symbols), live dashboard, monthly PDF |
| Live Track Record | ✅ | SHA-256 audit trail, GIPS TWR, risk disclosures, methodology doc |
| AI PM Agent | ✅ | Earned autonomy (0→75%), 16-tool loop, investment thesis audit trail |
| Adversarial Self-Play | ✅ | Red team attacks proposal before submission; AI PM revises or defends |
| Episodic Memory | ✅ | Per-regime outcome log; AI PM queries past periods before proposing |
| Calibration Tracking | ✅ | Conviction-vs-realized IC; AI PM sees its own hit rate |
| Narrative Alpha | ✅ | Quarter-over-quarter thesis shift detection (0% until cache matures) |

**Operational next steps (not code):** Deploy TimescaleDB (Docker), configure WebSocket (`ALPACA_KEY`), transfer real capital (~May–June 2026). YC-ready at April 2027 (12-month live track record).

---

## Environment Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Required `.env`:

```
ANTHROPIC_API_KEY=...   # Claude Opus (AI PM) + Sonnet (debate/red team) + Haiku (classifiers)
ALPACA_KEY=...
ALPACA_SECRET=...
FRED_API_KEY=...
```

Optional:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
TIMESCALEDB_URL=postgresql://postgres:ascent@localhost:5432/ascent
NTFY_TOPIC=...          # push alerts (drawdown, factor breach, IC degradation)
R2R_API_KEY=...         # semantic memory (BM25 fallback active without it)
```

---

## Repository Layout

```
ascent/
  config/       settings.py, types.py (AgentOutput), us_equity_universe.json
  data/
    ingest/     yahoo, fred, fundamentals, earnings, edgar_listener,
                capitol_trades, options_scanner, sec_filings,
                earnings_transcripts, reddit_sentiment, google_trends
    store/      parquet_store, point_in_time, timescale.py, schema.sql
    streaming/  alpaca_stream.py (WebSocket IEX, 901 symbols)
    validate/   altdata_validator.py (IC gate)
  features/     build_features, feature_defs, targets
  alpha/        trend, meanrev, statarb, ml_sleeve, fundamental, earnings,
                llm_fundamental, narrative_alpha, event_alpha, altdata_alpha, stack
  portfolio/    mvo_optimizer, black_litterman, regime_covariance,
                optimizer, hedge_overlay
  backtest/     engine, costs
  research/     walk_forward_runner, walk_forward_lightweight, cpcv,
                self_improve, shadow_promoter, factor_proposer,
                factor_discovery/ (PySR + LLM templates + Harvey FDR gate)
  regime/       engine, model, features, decision, particle_filter, breaks
  risk/         factor_data, factor_model, covariance_model,
                factor_exposure, factor_constraints, pm_risk_validator
  reporting/    market_memo, ic_brief_generator, blind_spot_detector,
                catalyst_scanner, debrief, investor_report
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch,
                twap_executor, implementation_shortfall,
                capacity_model, intraday_trigger, event_runner, debate_gate
  monitoring/   skill_tracker, forward_pnl_tracker, attribution,
                slippage_ic_feedback, live_nav, alert_system
  llm/          client.py (Claude Opus/Sonnet/Haiku routing, retry 3×)
  dashboard/    export_dashboard_data, live_dashboard.py (port 8502)
  strategy/     earned_authority.py, thesis_formatter.py, calibration_tracker.py

agents/         us_equities, macro, international, alternatives,
                event_agent, ai_pm_agent, red_team_agent
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker,
                disagreement_scorer, agent_tools
memory/         r2r_interface (BM25 fallback), reflection_agent, regime_memory
simulation/     mirofish_interface
compliance/     audit_trail, performance_report, risk_disclosure, methodology_index
docs/           methodology.md, risk_disclosures.md, superpowers/plans/, superpowers/specs/
scripts/        setup_timescaledb.sh, verify_audit_trail.py, evaluate_hedge.py

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl,
                active_alpha_config.json, factor_returns.parquet,
                llm_fundamental_cache.json, narrative_shift_cache.json,
                earned_authority.json, ai_pm_shadow_returns.jsonl
logs/           eod_log, slippage_log, attribution_log, audit_trail.jsonl,
                regime_episodes.jsonl, ai_pm_calibration.jsonl, alerts.jsonl
outputs/        debate_log/, investor_reports/, factor_proposals/, ai_pm_theses/

ascent/main.py        core pipeline entrypoint
run_all_agents.py     daily runner (launchd 1:45 PM ET)
demo_app.py           Streamlit investor demo (dark/gold, live LLM debate)
```

---

*Built by Scott Dong · Live paper trading since April 1, 2026 · Alpaca paper · Mac Air M5*
