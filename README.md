# Ascent Capital

![Tests](https://img.shields.io/badge/tests-716%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![AI PM](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)
![Alpha](https://img.shields.io/badge/alpha%20sleeves-14-orange)
![OOS Sharpe](https://img.shields.io/badge/OOS%20Sharpe-0.518-green)

### [📊 Live Performance Dashboard →](https://scottdongkhang.github.io/Ascent_Capital)
> Equity curve vs SPY · Current holdings · Full rebalance debate history · Updates automatically after every daily run.

**An AI-native quantitative investment fund built from scratch.** Not a backtest. Not a demo. A live system running on Alpaca paper trading that ingests market data every night, runs multi-agent quantitative analysis, debates its own trades through an adversarial LLM layer, and earns the right to manage capital by demonstrating a measurable edge in live markets.

---

## Live Track Record

> Paper trading since April 1, 2026 · AI PM shadow period began May 19, 2026 · **[Full dashboard →](https://scottdongkhang.github.io/Ascent_Capital)**

<!-- LIVE_STATS_START -->
| Metric | Value |
|--------|-------|
| Current NAV | $110,251 |
| Total Return | +9.36% |
| Alpha vs SPY | -6.09% |
| Sharpe (Ann.) | 2.863 |
| Max Drawdown | -4.32% |
| Days Live | 41 |
| Open Positions | 17 |
| Last Updated | 2026-06-01 |
<!-- LIVE_STATS_END -->

*Sharpe standard error over ~40 days is ~2.8 — not statistically significant yet.*

| Date | Event |
|------|-------|
| Apr 1, 2026 | Paper trading live · 29 orders · 9 initial positions |
| Apr 15, 2026 | Rebalance #1 · `REDUCE_SIZE` debate verdict · 27 orders |
| May 5, 2026 | Rebalance #2 · Full rotation · 40 orders · NAV $104,815 |
| May 19, 2026 | Rebalance #3 · 30 orders · NAV $103,790 · AI PM shadow begins |
| May 27, 2026 | Rebalance #4 · AI PM two-phase architecture live |
| May 30, 2026 | AI-Native Learning System wired |
| May 31, 2026 | Anti-hallucination hardening |
| Jun 1, 2026 | Causal Intelligence (Phases A–D) + monthly investor letter |

---

## Walk-Forward Out-of-Sample Results

> Survivorship bias eliminated · fold-local regime fit · CPCV C(6,2) = 15 folds · no look-ahead

| Metric | Value | Period |
|--------|------:|--------|
| CAGR | +12.35% | Jan 2020 – Apr 2026 |
| Sharpe Ratio | +0.518 | |
| Alpha vs SPY | +0.68% | |
| Max Drawdown | −23.4% | |

These figures reflect the quant system only. The AI PM, debate layer, and self-improve loop require live feedback and are not in this backtest.

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_agents.py            # daily pipeline (auto-detects weekend mode on Sat/Sun)
python run_all_agents.py --dry-run  # full logic, no orders submitted
python -m pytest --tb=short -q     # 716 passed, 1 skipped
streamlit run demo_app.py           # interactive investor demo
streamlit run ascent/dashboard/live_dashboard.py --server.port 8502
```

Required environment variables:

```env
ANTHROPIC_API_KEY=...    # Claude Opus/Sonnet/Haiku
ALPACA_KEY=...           # paper trading
ALPACA_SECRET=...
FRED_API_KEY=...         # macroeconomic data
```

> **Note (public showcase):** LLM system prompts and reasoning models are loaded from
> `private_prompts.yaml` (gitignored). Alpha signal logic and regime-conditional weights
> are redacted. Run `python scripts/generate_mock_models.py` to generate placeholder
> model artifacts before running the pipeline.

---

## System Architecture

```
python run_all_agents.py  ·  daily  ·  1:45 PM ET via launchd
│
├── Weekend Intelligence Pipeline (Sat/Sun — 11 jobs)
│   ├── Alt-data sweep (901 symbols)
│   ├── ML model retraining + hyperparameter search
│   ├── Factor discovery proposals → human review
│   ├── Self-improve loop (variant configs → shadow promotion)
│   ├── Conviction gate retrain
│   ├── Adversarial calibration (T+14d outcome scoring)
│   ├── AI PM deep research (Opus · full universe)
│   ├── Weekly debrief (attribution + AI PM performance)
│   ├── Adversarial scenario planning
│   └── Memory ingestion (BM25)
│
├── Daily outcome fills (regime memory · calibration · alpha wedge)
│
├── Earned authority update (every day)
│
├── Data Hub (Yahoo Finance · FRED · alt-data)
│
├── 4 Specialist Agents (parallel, ThreadPoolExecutor)
│   ├── US Equities    901 symbols · 14-sleeve alpha stack
│   ├── Macro          12 ETFs · regime-sized
│   ├── International  12 ETFs
│   └── Alternatives   7 ETFs · kill switch at 12%
│
├── Orchestrator
│   ├── Regime-based capital allocation
│   ├── Skill-weighted blending
│   ├── Correlation + coherence guards
│   └── Position caps + crisis override
│
│ ── NON-REBALANCE ────────────────────────────────────────────────────
│
├── Daily Intelligence (9 monitors → rebalance brief)
│
│ ── REBALANCE ────────────────────────────────────────────────────────
│
├── AI PM Agent (two-phase, Claude)
│   ├── Phase 1 — Pre-thesis (Sonnet · runs BEFORE quant agents)
│   └── Phase 2 — Synthesis (Opus · runs AFTER quant validation)
│
├── AI-Native Learning System
│   ├── Bayesian Meta-Learner (sleeve weights → empirical posteriors)
│   ├── AI Calibration (market character prediction tracking)
│   └── AI Regime Blend (HMM × AI assessment, α=0.05→0.30)
│
├── Adversarial Intelligence (3 layers · 1 falsifiable change/rebalance)
│   ├── Layer 1: Short thesis per position (scored 0–1)
│   ├── Layer 2: Regime-conditional sizing check
│   └── Layer 3: Narrative coherence / independent bet count
│
├── Debate (5 agents · 2 rounds · Monte Carlo injection)
│   ├── Bull · Bear · Devil's Advocate · Regime Specialist · Quant Sanity
│   └── Judge → verdict: proceed / reduce_size / halt_and_review
│
└── Execution
    ├── Kill switch: 8% warn · 15% halt (alternatives: 12%)
    ├── Almgren-Chriss cost model (blocks > 10% ADV)
    ├── Alpaca paper trading: retry ×3, 0.4s inter-order delay
    └── SHA-256 hash-chain audit trail · GIPS TWR monthly reports
```

---

## Alpha Stack — 14 Sleeves

| Sleeve | Category | Status |
|--------|----------|--------|
| Trend | Price momentum | ✅ live |
| Stat-arb | Sector-relative mean reversion | ✅ live |
| ML (XGBoost/CPCV) | Machine learning | ✅ live |
| Mean Reversion | Short-term reversal | ✅ live |
| Volatility | Vol regime | ✅ live |
| Fundamental | Accounting quality | ✅ live |
| Earnings (PEAD) | Post-earnings drift | ✅ live |
| Analyst | Revision momentum | ✅ live |
| LLM Fundamental | Haiku chain-of-thought | ✅ live |
| Narrative Alpha | Quarter-over-quarter shift | ✅ live |
| Options Flow | Market microstructure | ✅ live (sparse) |
| Insider | Transaction signal | ✅ live (sparse) |
| Short Interest | Squeeze potential | ✅ live (sparse) |
| Alt Data | IC-gated | 🔧 0% until IC gate passes |

*Regime-conditional weights, signal construction details, and IC thresholds are proprietary and not published in this repository. See `ascent/alpha/stack.py` for the equal-weight placeholder baseline.*

---

## Repository Layout

```
ascent/
  config/        settings · types (AgentOutput) · universe (901 symbols)
  data/ingest/   yahoo · fred · sec_filings · earnings_transcripts · google_trends
  features/      build_features · feature_defs (redacted)
  alpha/         14 sleeves (redacted) + stack combiner + meta_learner
  portfolio/     mvo_optimizer · black_litterman · regime_covariance
  research/      walk_forward_runner · cpcv · self_improve · factor_discovery/
  regime/        hmm engine · particle_filter · breaks · posture
  risk/          factor_model · covariance_model · pm_risk_validator
  causal/        dag_builder · compatibility · tracker · velocity
  execution/     eod_runner · alpaca_broker · kill_switch · slippage_tracker
  monitoring/    daily_intelligence · weekend_runner · weekly_debrief
  strategy/      earned_authority · calibration_tracker · conviction_gate
  llm/           client.py · prompt_loader.py

agents/          us_equities_agent · macro_agent · international_agent
                 alternatives_agent · ai_pm_agent (redacted) · red_team_agent (redacted)
debate/          adversarial_engine · debate_runner · agents (redacted) · judge (redacted)
orchestrator/    central_intelligence.py
compliance/      audit_trail (SHA-256) · performance_report (GIPS TWR)
scripts/         generate_mock_models.py · generate_performance_page.py
```

---

## Build Status

| Component | Status | Notes |
|-----------|--------|-------|
| 14-sleeve alpha stack | ✅ live | Regime-conditional weights · Bayesian meta-learner · IC gate |
| MVO + Black-Litterman | ✅ live | CLARABEL → SCS → rank-weight fallback |
| Factor risk model | ✅ live | FF5+UMD rolling OLS · Ledoit-Wolf shrinkage |
| 4 specialist agents | ✅ live | US equities · macro · international · alternatives |
| Multi-agent orchestration | ✅ live | Skill blend · correlation guard · coherence · position cap |
| AI PM — two-phase | ✅ shadow | Sonnet pre-thesis + Opus synthesis · Phase 0 (0% weight) |
| Adversarial self-play | ✅ live | Sonnet red team before every AI PM submission |
| Adversarial Intelligence | ✅ live | 3-layer engine · ONE falsifiable change per rebalance |
| Debate layer | ✅ live | 5 agents · 2 rounds · Monte Carlo injection |
| AI-Native Learning System | ✅ live | Bayesian meta-learner · AI calibration · AI regime blend |
| Decision memory | ✅ live | Compounding from May 19 · ML gate at n≥30 |
| Anti-hallucination hardening | ✅ live | Structured outputs on all LLM sleeves and debate agents |
| Causal Intelligence (A–D) | ✅ live | PC DAG · causal gates · early-exit tracker · debate attacks |
| Monthly investor letter | ✅ live | Auto-generated on first trading day of each month |
| Weekend intelligence pipeline | ✅ live | 11 jobs · debrief · scenarios · AI PM research |
| Self-improve loop | ✅ built | Disabled pending 30 consecutive days of positive Sharpe |
| Factor discovery | ✅ built | PySR + LLM proposals · Harvey FDR gate · human review |
| 130/30 long-short | ✅ built | Disabled pending ≥30 paper rebalances |
| TimescaleDB / WebSocket | 🔧 pending | |
| Real capital deployment | 🔧 pending | Targeting April 2027 |

---

## Integrity Constraints

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only
2. **No simulated data under live cache names** — fallback always labeled explicitly
3. **Max-weight hard cap** — `_water_fill_cap()` with post-condition check
4. **Sector constraint with coverage fallback** — < 80% valid labels → skip caps + warn
5. **Failed folds must be visible** — no silent zero-weight fallback in walk-forward
6. **Debate is advisory only** — never writes to alpha, portfolio, or execution modules
7. **Alpha sleeve registry** — `DEFAULT_ALPHA_WEIGHTS` in `stack.py` and `self_improve.py` must stay in sync

---

*Built by Scott Dong · Paper trading since April 1, 2026 · Target: 12-month live track record by April 2027*
