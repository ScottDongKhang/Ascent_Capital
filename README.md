# Ascent Capital

![Tests](https://img.shields.io/badge/tests-569%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![AI PM](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)

A full-stack quantitative fund built from scratch — data ingestion through live execution. 13-sleeve alpha stack, MVO/Black-Litterman portfolio construction, regime-adaptive orchestration across 4 specialist agents, and an AI portfolio manager that earns trading authority by proving itself in live markets before it touches capital.

Live on Alpaca paper trading since April 1, 2026. 3 rebalances completed. AI PM in shadow period.

---

## What This Actually Is

Most quant repos are backtests with a nice README. This one runs daily at 1:45 PM ET via launchd, submits real orders to Alpaca, tracks live slippage, and logs every decision to a SHA-256 hash-chained audit trail.

The system has three layers that most quantitative frameworks don't:

**1. An AI PM that learns from its own mistakes.**
Every override the AI PM makes — "I'm removing this name the quant likes, here's why" — is stored in a decision memory database with the context: override type, regime, weights, momentum. 21 days later, the realized return fills in automatically. Before making any future override of the same type in the same regime, the AI PM queries its own historical win rate. If it's been losing on valuation overrides in calm_bull markets, the conviction gate reduces its position size or blocks the override entirely. This is a compounding database — nobody starting from scratch has it.

**2. Adversarial self-play before every submission.**
After the AI PM proposes a portfolio, a separate red team agent attacks it. Not a critique of the overall thesis — a specific attack on each override the AI PM made versus the quant baseline: "the quant wanted SATS at 6.5%. You removed it. If the quant is right, what does that cost?" The AI PM then revises or defends with a 6-tool research pass. The final portfolio is the one that survived adversarial pressure.

**3. An AI PM that earns its authority.**
The AI PM starts at 0% allocation. It shadow-trades alongside the quant system, logging returns to `data_cache/ai_pm_shadow_returns.jsonl`. After 21 rebalance days with a sustained Sharpe edge of > 0.05 over the quant baseline, it earns 25% authority. It can reach 75%. The quant system always has at least 20% weight. If the AI PM's 21-day drawdown exceeds the quant's by 5 percentage points, it auto-reverts to Phase 0 and starts over.

---

## Live Track Record

52 days of paper trading is not statistically meaningful. Sharpe standard error after 52 days is ≈ 2.8 — any number you see here has a confidence interval wide enough to drive a truck through. We know this and track it explicitly.

What is meaningful: the system runs correctly, the orders go through, slippage is tracked, and the audit trail is clean.

| Date | Event |
|------|-------|
| Apr 1, 2026 | Live paper trading begins (Alpaca) · 29 orders |
| Apr 15, 2026 | Rebalance #1 · REDUCE_SIZE verdict from debate · 27 orders |
| May 5, 2026 | Rebalance #2 · Full rotation · 40 orders · NAV $104,815 |
| May 7, 2026 | SPY −0.31% · Portfolio −2.53% · Beta 1.26 working against us |
| May 19, 2026 | Rebalance #3 · 30 orders · NAV $103,790 · 18 positions · AI PM shadow begins |
| May 22, 2026 | Decision memory + conviction gate live · AI PM learns from outcomes |
| May 23, 2026 | Position health monitor · intelligence grounded in real data |

**Current portfolio (May 2026):** EWY 10.9%, PDBC 6.9%, CBOE/CHRD/HUM/SATS/SNDK/STRL/VICR/VRT/WDC ~6.5% each, DBB 3.8%, EWT/EEM 3.4% each, EWC 3.3%, DBA 3.1%, BIL 2.8%, KMLM 2.4%, UUP 1.6%. 18 positions. NAV ~$103,790.

**Skill scores** become meaningful after 63 days of live data (June 3, 2026). Until then the orchestrator uses equal skill weights.

---

## Walk-Forward OOS

> Per-fold universe filter (survivorship bias eliminated) · fold-local regime fit · CPCV C(6,2)=15 folds for ML sleeve · no look-ahead

| Metric | Value | Period |
|--------|------:|--------|
| CAGR | +12.35% | Jan 2020 – Apr 2026 |
| Sharpe | +0.518 | |
| Alpha vs SPY | +0.68% | |
| Max Drawdown | −23.4% | |

The OOS record reflects the 13-sleeve quant system only — AI PM, debate layer, and self-improve loop are not in this backtest because they require live market feedback to operate. The live track record will eventually be the honest benchmark.

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Daily pipeline (also runs via launchd at 1:45 PM ET)
python run_all_agents.py

# Dry run — all logic, no orders submitted
python run_all_agents.py --dry-run

# Tests
python -m pytest --tb=short -q   # 569 passed, 1 skipped

# Interactive demo
streamlit run demo_app.py

# Operator dashboard (live NAV, positions, regime signal)
streamlit run ascent/dashboard/live_dashboard.py --server.port 8502
```

Required `.env`:
```
ANTHROPIC_API_KEY=...
ALPACA_KEY=...
ALPACA_SECRET=...
FRED_API_KEY=...
```

---

## System Architecture

```
python run_all_agents.py  ·  daily  ·  1:45 PM ET via launchd
│
├── Outcome fills (best-effort, silent)
│   ├── Regime memory: realized returns for episodes ≥21 days old
│   ├── Calibration: conviction-vs-realized IC for old predictions
│   └── Alpha wedge: 21d price fetch → fill AI PM vs quant wedge
│       → propagate wedge to decision memory (per-override outcomes)
│
├── Earned authority update (runs every day, before split)
│   └── AI PM Sharpe edge ≥0.05 for 21 rebalances → advance phase
│       AI PM drawdown > quant + 5pp → revert to Phase 0
│
├── Data hub  (one parallel fetch, all universes)
│   ├── Yahoo: 901 US equities + macro + international + alternatives ETFs
│   ├── FRED: macro series (cached fallback on outage)
│   └── Sector profiles for portfolio construction
│
├── 4 Specialist Agents  (parallel, ThreadPoolExecutor)
│   ├── US Equities   901 symbols · 13-sleeve alpha → MVO/BL → sector constraints
│   ├── Macro         12 ETFs · trend-only · regime-sized (crisis: top 3 / 40%)
│   ├── International 12 ETFs · EM penalty when USD > 50MA
│   └── Alternatives  7 ETFs · trend 80% + low-vol 20% · kill switch at 12%
│
├── Orchestrator
│   ├── Base by regime: calm_bull US60/mac15/intl15/alt10
│   │                   stressed US45/mac25/intl10/alt20
│   │                   crisis   US30/mac30/intl5/alt35
│   ├── Skill blend: 50% rolling Sharpe + 50% base (negative Sharpe → zero)
│   ├── Conviction bonus: +15% when ≥2 agents hold the same name
│   ├── 63-day cross-agent correlation guard (cap 0.70 → halve smaller)
│   ├── Thesis coherence (12 factor buckets · 6 contradiction pairs → 40% cut)
│   ├── EM + commodity hard cap: 20%
│   └── Crisis veto: merged = 0.60×macro + 0.40×merged
│
│ ── NON-REBALANCE PATH ─────────────────────────────────────────────────────
│
├── Forward PnL · Skill scores · Counterfactual scoring
│
├── Daily Intelligence  (9 days accumulate before each rebalance)
│   ├── Position health  (pure Python, runs first — no LLM)
│   │   └── Per-position: return since rebalance · 252d/21d momentum
│   │       alpha rank percentile in universe · flag OK/WATCH/DETERIORATING
│   ├── Conviction decay: alpha rank drift per held position since entry
│   ├── Signal health: per-sleeve IC trend vs rebalance baseline
│   ├── Regime trajectory: stability score · stress trend from HMM series
│   ├── Historical analogues: match past regime fingerprints → realized outcomes
│   ├── Position thesis: Haiku checks if buy thesis holds (with real health data)
│   ├── Adversarial challenge: Haiku finds most dangerous assumption (with real data)
│   └── Macro calendar: FOMC/CPI/NFP/earnings exposure per position
│
│ ── REBALANCE PATH ─────────────────────────────────────────────────────────
│
├── Pre-rebalance checklist (blocks on kill switch / stale prices / sector gaps)
│
├── Rebalance brief: Haiku synthesizes 9 days of intelligence
│   └── Deteriorating positions · weakening sleeves · adversarial themes
│       → data_cache/rebalance_brief.json  (AI PM's first read)
│
├── AI PM Agent  (Claude Opus 4.6 · 22 tools · rebalance days only)
│   │
│   ├── Phase 1 — Context
│   │   └── get_rebalance_brief · regime state · macro · regime memory
│   │
│   ├── Phase 2 — Quant baseline
│   │   └── All 4 agents (precomputed cache, no re-runs) · momentum screen
│   │       Flag names with 252d momentum >200% as [EXTENDED]
│   │
│   ├── Phase 3 — Signal research + override decisions
│   │   ├── Up to 6 of 10 signal tools (SEC, transcripts, attribution,
│   │   │   earnings, VaR, factor exposure, sector, momentum, narrative, news)
│   │   ├── query_decision_history(override_type, regime)
│   │   │   → historical win rate + avg wedge + recent cases
│   │   └── check_override_conviction(override_type, regime)
│   │       → gate result + size multiplier (1.0 / 0.75 / 0.85 / block)
│   │
│   ├── Phase 4 — Submit
│   │   └── Write PRE-MORTEM · coherence check · propose_portfolio
│   │
│   ├── Red Team  (Sonnet · attacks AI PM vs quant deltas)
│   │   ├── Per-position: worst case if quant was right
│   │   └── Systemic kill shot: one scenario that takes down multiple positions
│   │
│   └── Revision pass (Opus · 6 tool calls · revise or defend)
│
├── Decision memory ingestion
│   └── Each override → logs/decision_memory.jsonl
│       (rebalance_date, symbol, type, regime, weights, momentum, wedge=None)
│
├── Earned authority blend
│   ├── Phase 0: ai_weight=0.0 (shadow mode — currently active)
│   ├── Phase 1: ai_weight=0.25 (after 21 rebalances with Sharpe edge)
│   ├── Phase 2: ai_weight=0.50
│   └── Phase 3: ai_weight=0.75 (hard cap 0.80 · quant always ≥20%)
│
├── Debate  (rebalance days · advisory · never writes to alpha/portfolio/execution)
│   ├── Score past verdicts · debrief · blind spot detection · catalyst scan
│   ├── Monte Carlo simulation (p5–p95 quantified tail risk)
│   ├── Round 1: Bull · Bear · Devil's Advocate · Regime Specialist · Quant Sanity
│   ├── Round 2: Cross-rebuttals · TF-IDF disagreement scoring
│   └── Judge → proceed / reduce_size / halt_and_review
│
└── Execution
    ├── Kill switch: SOFT_WARN 8% · HARD_STOP 15% (alt: 12%)
    ├── Almgren-Chriss cost model (block >10% ADV · warn >5%)
    ├── Alpaca paper trading · slippage tracking · IS decomposition
    └── SHA-256 hash-chain audit trail · GIPS TWR monthly reports
```

---

## Alpha Stack (14 Sleeves)

| Sleeve | Weight | Signal |
|--------|-------:|--------|
| Trend | 38% | Cross-sectional momentum; `mom_252d − mom_21d` skip-month at 20% sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion |
| ML (XGBoost) | 10% | CPCV 15 folds · 5-day purge+embargo · 12 features · p5 IC Sharpe guard |
| Mean Reversion | 5% | Short-term reversal (z-score 20d) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day lag; momentum-neutral |
| Earnings PEAD | 5% | EPS surprise z-score · OLS momentum-beta residual · 1-bday lag |
| Analyst | 5% | Revision signal; zero-filled when cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Haiku; cached by (symbol, quarter) |
| Narrative Alpha | 3% | Quarter-over-quarter thesis shift detection; returns zeros while cache matures |
| Options Flow | 2% | IV-adjusted sentiment; sparse |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Alt Data | 0% | SEC 10-K, transcripts, Reddit, Google Trends; 0% until IC gate passes |

Weights are regime-adaptive via `active_alpha_config.json`. Distressed filter zeroes names with `mom_252d < −0.65`. Regime tightens max position size: crisis → 8%, calm_bull → 15%.

---

## Portfolio Construction

```
Alpha scores (cross-sectionally z-scored, regime-weighted blend)
  → Distressed filter
  → Black-Litterman  (quant prior + LLM views; tau scales with IC IR)
  → cvxpy MVO        maximize: w'α − λ(w'Σw) − κ‖w − w_prev‖₁
                     Σ = B·F·B' + D  (FF5+UMD factor covariance)
                     CLARABEL → SCS → rank-weight fallback
  → Sector constraints  (max 1 per sector · skip if coverage < 80%)
  → SPY 200MA overlay   (×0.70 when SPY < 200MA)
  → VIXY hedge          (0–8% by regime × HMM confidence)
```

---

## The Decision Memory System

Every AI PM override is stored at submission time:

```python
{
  "entry_id": "2026-05-19_SATS",
  "rebalance_date": "2026-05-19",
  "symbol": "SATS",
  "override_type": "data_quality",   # data_quality / regime_macro / news_event /
  "regime": "calm_bull",             # correlation_risk / valuation
  "ai_action": "REMOVED",
  "ai_weight": 0.0,
  "quant_weight": 0.065,
  "weight_delta": -0.065,
  "momentum_252d": 31.96,            # 3196% — merger artifact, caught correctly
  "wedge_21d": null                  # filled automatically 21 days later
}
```

21 days later, the daily outcome-fill loop fetches realized prices, computes the AI PM vs quant wedge, and fills `wedge_21d`. The conviction gate then reads this database before every future override of the same type in the same regime:

- **< 5 cases**: proceed with 15% size reduction (building track record)
- **Win rate ≥ 60% + positive avg wedge**: full conviction size
- **Win rate ≥ 50%**: proceed at 75% size
- **Win rate < 35%, n ≥ 8**: blocked
- `data_quality`, `news_event`, `correlation_risk`: always approved — structural AI edge

The gate is ML-ready: when n ≥ 30 cases, it trains a logistic regression on the override features (type, regime, momentum, weight delta) and uses model probability instead of the heuristic thresholds. The interface to the AI PM never changes.

---

## Regime System

HMM K=2–4 (walk-forward CV selects K per regime). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3 days. Entropy > 0.90 → `uncertain` override.

Emergency refit triggers: SPY −3% + VIX > 30, 200MA cross, SPY/TLT correlation flip, break z-score > 3.5. Particle filter: 500-particle SIR with reinitialization on batch refit.

Regime propagates through: sleeve weights, position size caps, orchestrator capital allocation, debate context, AI PM episodic memory queries, VIXY hedge sizing, kill switch thresholds.

---

## Status Table

| Component | Status | Notes |
|-----------|--------|-------|
| 13-sleeve alpha stack | ✅ live | Regime-adaptive weights |
| MVO + Black-Litterman | ✅ live | CLARABEL/SCS/rank-weight fallback chain |
| 4 specialist agents | ✅ live | US equities, macro, international, alternatives |
| Multi-agent orchestration | ✅ live | Skill blend, correlation guard, coherence check |
| Debate layer | ✅ live | 5 agents, 2 rounds, Monte Carlo tail risk |
| AI PM agent | ✅ shadow | Phase 0 (0% weight) since May 19, 2026 |
| Adversarial self-play | ✅ live | Red team attacks before every submission |
| Decision memory | ✅ live | Compounding from May 19 — meaningful at ~rebalance 10 |
| Conviction gate | ✅ live | Rules-based now · logistic regression at n ≥ 30 |
| Earned authority | ✅ live | Phase 0 → 25% after 21 rebalances with Sharpe edge |
| Non-rebalance intelligence | ✅ live | 7 monitors daily · position health · rebalance brief |
| Factor risk model | ✅ live | FF5+UMD rolling OLS · Ledoit-Wolf covariance |
| Calibration tracking | ✅ live | Conviction IC vs realized 21d returns |
| Episodic regime memory | ✅ live | Per-regime outcome history, queried by AI PM |
| Self-improve loop | ✅ built | `SELF_MODIFY_ENABLED=False` until +Sharpe 30 days |
| Factor discovery | ✅ built | PySR + LLM proposals · Harvey FDR gate · human review |
| 130/30 long-short | ✅ built | `LONG_SHORT_ENABLED=False` until ≥30 rebalances |
| TWAP / event trading | ✅ built | Kill-switched pending paper validation |
| TimescaleDB | 🔧 pending | Docker deploy |
| WebSocket stream | 🔧 pending | ALPACA_KEY config |
| Real capital | 🔧 pending | ~May–June 2026 |
| 12-month live track record | 📅 April 2027 | YC-ready milestone |

---

## Repository Layout

```
ascent/
  config/        settings, types (AgentOutput), universe
  data/
    ingest/      yahoo, fred, fundamentals, earnings, edgar_listener,
                 capitol_trades, options_scanner, sec_filings,
                 earnings_transcripts, reddit_sentiment, google_trends
    store/       parquet, point_in_time, timescale, schema
    streaming/   alpaca_stream (WebSocket, 901 symbols)
    validate/    altdata_validator (IC gate)
  features/      build_features, feature_defs, targets
  alpha/         trend, meanrev, statarb, ml_sleeve, fundamental, earnings,
                 llm_fundamental, narrative_alpha, event_alpha, altdata_alpha, stack
  portfolio/     mvo_optimizer, black_litterman, regime_covariance,
                 optimizer, hedge_overlay, long_short
  research/      walk_forward_runner, cpcv, self_improve, shadow_promoter,
                 factor_proposer, factor_discovery/
  regime/        engine, model, features, decision, particle_filter, breaks
  risk/          factor_data, factor_model, covariance_model,
                 factor_exposure, pm_risk_validator
  reporting/     market_memo, ic_brief_generator, blind_spot_detector,
                 catalyst_scanner, debrief, investor_report
  execution/     eod_runner, alpaca_broker, order_engine, kill_switch,
                 cost_model, slippage_tracker, twap_executor,
                 implementation_shortfall, capacity_model, debate_gate
  monitoring/    skill_tracker, forward_pnl_tracker, attribution,
                 position_health, conviction_tracker, signal_health,
                 regime_trajectory, analogue_search, position_thesis,
                 adversarial_daily, macro_calendar, daily_intelligence,
                 rebalance_brief, rebalance_trigger, alert_system
  llm/           client.py  (Opus/Sonnet/Haiku routing · retry · cost log)
  strategy/      earned_authority, calibration_tracker, conviction_gate
  memory/        decision_memory, regime_memory, r2r_interface

agents/          us_equities_agent, macro_agent, international_agent,
                 alternatives_agent, ai_pm_agent, red_team_agent, event_agent
orchestrator/    central_intelligence.py
debate/          debate_runner, agents, judge, outcome_tracker, disagreement_scorer
memory/          r2r_interface (BM25 fallback), reflection_agent, regime_memory
compliance/      audit_trail (SHA-256 hash chain), performance_report,
                 risk_disclosure, methodology_index
docs/            methodology.md, risk_disclosures.md

run_all_agents.py       daily runner (launchd 1:45 PM ET)
demo_app.py             Streamlit investor demo
```

---

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```env
# Required
ANTHROPIC_API_KEY=...
ALPACA_KEY=...
ALPACA_SECRET=...
FRED_API_KEY=...

# Optional
TIMESCALEDB_URL=postgresql://postgres:ascent@localhost:5432/ascent
NTFY_TOPIC=...        # push alerts (drawdown, IC degradation)
R2R_API_KEY=...       # semantic memory (BM25 active without it)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

---

*Built by Scott Dong · Paper trading since April 1, 2026 · Target: 12-month live track record by April 2027*
