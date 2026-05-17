# Ascent Capital — AI-Native Quantitative Fund

![Tests](https://img.shields.io/badge/tests-465%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![Model](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)

**An AI portfolio manager that earns trading authority through its live track record.**

It starts at 0% weight — shadow trading alongside the quant system. After 21 days of proven Sharpe edge, it earns 25% allocation. It can reach 75%. If its drawdown exceeds the quant baseline by 5 percentage points, it auto-reverts to 0% and starts over. The quant system always has a 20% floor.

This isn't a backtest notebook. It's a full production stack: 13-sleeve alpha, factor risk model, MVO portfolio construction, multi-agent LLM debate, live Alpaca execution, GIPS-compliant compliance, and a hash-chain audit trail — running daily since April 2026.

---

## What Makes This Different

**Most quant repos:** Backtest → optimize → publish Sharpe. Look-ahead bias optional.

**This system:**

- **AI PM with earned autonomy** — Claude Opus runs a 14-tool research loop each rebalance: reads macro data, runs all 4 quant agents as tool calls, pulls SEC filings/transcripts/attribution history, then either agrees with the quant signals or overrides them with explicit reasoning. It earns its authority; it doesn't start with it.
- **Honest OOS** — Walk-forward with per-fold `get_universe_on_date()`, regime fitted on training slice only, CPCV for ML sleeve. No look-ahead.
- **Multi-agent debate before every trade** — Bull / Bear / Devil's Advocate (with live Monte Carlo tail numbers) / Judge. Round 2 cross-rebuttals. Disagreement tracked longitudinally as a monitoring signal.
- **Institutional infrastructure** — Barra-style factor risk model (FF5+UMD), Black-Litterman + cvxpy MVO, TimescaleDB, Alpaca WebSocket (901 symbols), SHA-256 audit trail, monthly GIPS performance reports.
- **Self-improving** — Weekly OOS evaluation of weight variants, 30-day shadow periods, auto-promotion. Gate: must post positive OOS Sharpe for 30 consecutive trading days before self-modification is enabled.

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_agents.py                                               # daily pipeline (also via launchd 1:45 PM ET)
python -m pytest --tb=short -q                                        # 465 passed, 1 skipped
streamlit run demo_app.py                                             # investor demo (dark/gold, live LLM debate)
streamlit run ascent/dashboard/live_dashboard.py --server.port 8502  # operator dashboard
```

---

## Walk-Forward OOS Record

> Per-fold universe filter, fold-local regime fit, no look-ahead. Honest.

| Metric | Value |
|--------|------:|
| Period | Jan 2020 – Apr 2026 |
| CAGR | +12.35% |
| Sharpe | +0.518 |
| Alpha vs SPY | +0.68% |
| Max Drawdown | −23.4% |

---

## AI PM Agent: Earned Autonomy

The core idea: let the AI prove itself before giving it real capital.

```
Phase 0  ai_weight = 0.0   Shadow only. Builds 21-day return history.
Phase 1  ai_weight = 0.25  After 21 days: AI Sharpe > quant + 0.05
Phase 2  ai_weight = 0.50  Another 21 days of sustained edge
Phase 3  ai_weight = 0.75  Sustained (operational max)
Cap      ai_weight = 0.80  Hard cap. Quant always has 20% floor.

Auto-revert: AI 21d drawdown > quant + 5pp → reset to Phase 0, count increments
```

Each rebalance, the AI PM runs a structured 4-phase research loop (max 14 tool calls):

1. **Market context** — regime state, macro data
2. **Quant baseline** — runs all 4 specialist agents as tool calls, reads their output
3. **Signal research** — SEC filings, earnings transcripts, attribution history, factor exposures, VaR, momentum — picks ≤6 of 9 available tools
4. **Proposal** — `propose_portfolio(weights, thesis)` with full investment memo

Every quant override requires explicit reasoning referencing the signal data. The thesis is saved to `outputs/ai_pm_theses/` and written to the audit trail.

Pre-blend hard-limit validator (pure math, never LLM) rejects proposals with: position > 15%, sector > 40%, fewer than 5 names, negative weights, or names in distressed filter.

---

## Live Track Record

| Date | Event |
|------|-------|
| Apr 1, 2026 | Live paper trading begins (Alpaca) |
| Apr 15, 2026 | First rebalance: REDUCE_SIZE verdict, 27 orders |
| May 5, 2026 | Full portfolio rotation: 40 orders, full liquidation of prior holdings |
| May 7, 2026 | NAV $104,815 · Regime: calm_bull |
| May 16, 2026 | AI PM Agent live · 465 tests · All 7 institutional plans complete |

**Current holdings (May 2026):** KMLM, IFRA, AMKR, FIX, WDC, VICR, VRT, VAL, WCC, DBB, CNC, WFRD, PDBC, STLD, DBA, CHRD, EWY, EWC, IRM, MUSA + VIXY hedge.

---

## System Architecture

```
python run_all_agents.py  (daily, 1:45 PM ET via launchd)
│
├─ Step 0: Factor data update (FF5+UMD) · intraday trigger check
│
├─ 4 Specialist Agents (parallel)
│   ├── US Equities  ── 901 symbols (S&P 500 + 400), 13-sleeve alpha stack
│   ├── Macro        ── 12 ETFs, trend-only, regime-sized
│   ├── International── 12 ETFs, EM-aware (UUP > 50MA → 20% EM penalty)
│   └── Alternatives ── 7 ETFs, VIXY hedge, kill switch at 12% drawdown
│
├─ Orchestrator
│   ├── Regime base allocations (calm_bull / stressed / crisis)
│   ├── Skill-score blend · conviction bonus · correlation guard
│   └── EM+commodity hard cap (20%)
│
├─ AI PM Agent  (Claude Opus 4.6, tool-use loop)       ← NEW
│   ├── 14-tool research loop (macro → quant → signals → propose)
│   ├── Pre-blend risk validator (hard limits, pure math)
│   ├── Earned authority blend (ai_weight 0–80%)
│   └── Investment thesis saved to audit trail
│
├─ Debate Layer  (rebalance days only — advisory)
│   ├── Blind spot detection · catalyst scan · Monte Carlo (p5–p95)
│   ├── Round 1: Bull / Bear / Devil's Advocate (Sonnet) · Regime Specialist (Haiku) · Quant Sanity (Python)
│   ├── Round 2: Cross-rebuttals · TF-IDF disagreement score
│   └── Judge (Sonnet) → proceed / reduce_size / halt_and_review
│
└─ Execution
    ├── Approval gate (>2% NAV) · TWAP executor (kill-switched)
    ├── Alpaca paper trading
    └── IS decomposition · slippage tracking · SHA-256 audit trail
```

---

## Alpha Stack (13 Sleeves)

| Sleeve | Weight | Signal |
|--------|-------:|--------|
| Trend | 41% | Cross-sectional momentum; skip-last-month `mom_252d − mom_21d` at 0.20 sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, 5-day purge+embargo; 6 features by IC/IR; p5 guard > −0.05 |
| Mean Reversion | 5% | Short-term reversal (z-score 20d) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day lag; momentum-neutral |
| Earnings PEAD | 5% | EPS surprise z-score; OLS momentum-beta residual; 1-bday lag |
| Analyst | 5% | Revision signal; sparse — zero-filled when absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Claude Haiku; cached by (symbol, quarter) |
| Options Flow | 2% | IV-adjusted sentiment; sparse |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Alt Data | 0% | SEC 10-K, transcripts, Reddit, Google Trends — 0% until IC gate passed |

Distressed filter zeroes alpha for `mom_252d < −0.65`. Weights are regime-adaptive, updated weekly by the self-improve loop.

---

## Portfolio Construction

```
Alpha scores
  → Black-Litterman  (quant prior + LLM views; tau scales with IC IR)
  → cvxpy MVO        maximize: w'α − λ(w'Σw) − κ‖w − w_prev‖₁
                     Σ = B·F·B' + D  (FF5+UMD factor covariance + idiosyncratic residuals)
                     CLARABEL solver → SCS fallback → rank-weight fallback
  → Sector constraints  (max 1 per sector, skip caps if coverage < 80%)
  → SPY 200MA overlay   (×0.70 when SPY < 200MA)
  → VIXY hedge overlay  (0–8% by regime × confidence)
```

---

## Key Systems

**Regime** — HMM K=2–4 (walk-forward CV selects K). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Particle filter (500 SIR). Emergency refit on SPY −3%+VIX>30, 200MA cross, corr flip, break z>3.5, or >5 days stale. Leading indicators: credit spread (HYG/LQD), yield curve slope (TLT/IEF).

**Debate** — Advisory only. Sequence: score past verdicts → debrief → blind spot detection → catalyst scan → Monte Carlo → Round 1 → Round 2 cross-rebuttals → Judge. Verdict gates execution: `proceed` / `reduce_size` (Haiku adjusts weights) / `halt_and_review`.

**Self-improve** — Sunday 6AM. LLM-guided hypothesis generation + random perturbation → 5 variants → real multi-fold OOS (3–4 folds, 5-day purge+embargo) → shadow 30 days → auto-promote. `SELF_MODIFY_ENABLED=False` until +OOS Sharpe for 30 consecutive trading days.

**Factor discovery** — Monthly (first Sunday). PySR symbolic regression + Haiku JSON template proposals. Both gated by Harvey FDR (IC_mean ≥ 0.015, IC_IR ≥ 0.60, positive in every observed regime). Proposals written to `outputs/factor_proposals/` — human review required, nothing auto-deploys.

**Compliance** — SHA-256 hash-chain audit trail. Monthly `scripts/verify_audit_trail.py` integrity check (exits 0/1). GIPS TWR performance reporting. Full methodology doc + risk disclosures in `docs/`.

---

## Institutional Roadmap

| Plan | Status | What It Adds |
|------|--------|--------------|
| 1 — Factor Risk Model | ✅ | FF5+UMD rolling OLS, Ledoit-Wolf Σ, factor P&L attribution |
| 2 — Portfolio Construction | ✅ | cvxpy MVO, Black-Litterman, regime covariance |
| 3 — Event-Driven Architecture | ✅ | EDGAR 8-K, Capitol Trades, options anomaly (`EVENT_TRADING_ENABLED=False`) |
| 4 — Alternative Data Pipeline | ✅ | SEC 10-K/Q, transcripts, Reddit, Google Trends; IC gate |
| 5 — Execution Excellence | ✅ | TWAP, IS decomposition, capacity model, intraday triggers (kill-switched) |
| 6 — Real-Time Infrastructure | ✅ | TimescaleDB, Alpaca WebSocket, live dashboard (port 8502), monthly PDF reports |
| 7 — Live Track Record | ✅ | SHA-256 audit trail, GIPS TWR, risk disclosures, methodology doc |
| AI PM Agent | ✅ | Earned autonomy model, 14-tool research loop, investment thesis audit trail |

**Operational next steps (not code):** Deploy TimescaleDB (Docker), configure WebSocket (`ALPACA_KEY`), transfer real capital (~May–June 2026). YC-ready at April 2027 (12-month live track record).

---

## Environment Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Required `.env` keys:

```
ANTHROPIC_API_KEY=...   # Claude Opus (AI PM) + Sonnet (debate) + Haiku (classifiers)
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
```

TimescaleDB (optional, Plan 6): `bash scripts/setup_timescaledb.sh`

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
                llm_fundamental, event_alpha, altdata_alpha, stack
  portfolio/    mvo_optimizer, black_litterman, regime_covariance,
                optimizer, hedge_overlay
  backtest/     engine, costs
  research/     walk_forward_runner, walk_forward_lightweight, cpcv,
                self_improve, shadow_promoter, factor_proposer,
                factor_discovery/ (PySR + LLM templates + Harvey FDR gate)
  regime/       engine, model, features, decision, particle_filter, breaks
  risk/         factor_data, factor_model, covariance_model,
                factor_exposure, factor_constraints
  reporting/    market_memo, ic_brief_generator, blind_spot_detector,
                catalyst_scanner, debrief, investor_report
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch,
                twap_executor, implementation_shortfall,
                capacity_model, intraday_trigger, event_runner, debate_gate
  monitoring/   skill_tracker, forward_pnl_tracker, attribution,
                slippage_ic_feedback, live_nav, alert_system
  llm/          client.py (Claude Opus/Sonnet/Haiku routing, retry 3×)
  dashboard/    export_dashboard_data, live_dashboard.py (port 8502)
  strategy/     earned_authority.py, thesis_formatter.py

agents/         us_equities, macro, international, alternatives, event_agent, ai_pm_agent
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker,
                disagreement_scorer, agent_tools
memory/         r2r_interface (BM25 fallback), reflection_agent
simulation/     mirofish_interface
compliance/     audit_trail, performance_report, risk_disclosure, methodology_index
docs/           methodology.md, risk_disclosures.md, superpowers/plans/, superpowers/specs/
scripts/        setup_timescaledb.sh, verify_audit_trail.py, evaluate_hedge.py

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl,
                active_alpha_config.json, factor_returns.parquet,
                earned_authority.json, ai_pm_shadow_returns.jsonl
dashboard/      regime_signal.json, factor_exposures.json, methodology_index.json
outputs/        debate_log/, investor_reports/, factor_proposals/,
                altdata_proposals/, ai_pm_theses/
logs/           eod_log, slippage_log, attribution_log, event_trades,
                audit_trail.jsonl, alerts.jsonl

ascent/main.py        core pipeline entrypoint
run_all_agents.py     daily runner (launchd 1:45 PM ET)
demo_app.py           Streamlit investor demo (dark/gold, live LLM debate)
```

---

*Built by Scott Dong · Live paper trading since April 1, 2026 · Alpaca · Mac Air M5*
