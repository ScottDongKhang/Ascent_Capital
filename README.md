# Ascent Capital

![Tests](https://img.shields.io/badge/tests-627%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![AI PM](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)
![Alpha Stack](https://img.shields.io/badge/alpha%20sleeves-14-orange)

**A full-stack AI-native quantitative fund built from scratch.** Data ingestion → 14-sleeve alpha → MVO/Black-Litterman portfolio construction → regime-adaptive multi-agent orchestration → AI portfolio manager that earns trading authority by proving itself in live markets → adversarial risk committee before every trade.

Runs daily at 1:45 PM ET via launchd. Orders go to Alpaca. Every decision is logged, hashed, and auditable.

---

## What makes this different

Most quant repos are backtests dressed up as systems. This one has four things that don't exist elsewhere:

**1. An AI PM that learns from its own mistakes — permanently.**
Every override the AI PM makes is stored with full context: override type, regime, weights, momentum. 21 days later, the realized return fills in automatically. Before any future override of the same type in the same regime, the AI PM queries its own historical win rate. If it's been losing valuation calls in calm_bull markets, the conviction gate reduces its size or blocks the override. This is a compounding proprietary database — nobody starting from scratch has it, and it gets more accurate every week.

**2. A risk committee that earns its authority by being right.**
Before every rebalance, an adversarial engine runs three layers of analysis the quant model structurally cannot see: (1) strongest possible short thesis per position via a batched Haiku call, scored 0–1; (2) regime-conditional sizing check — is each position the right size for this regime and position type; (3) narrative clustering — how many truly independent bets does the book contain, and what happens to the whole thing if the regime flips? The engine makes ONE prioritized weight change per rebalance, with a falsifiable 10-day prediction. That prediction is tracked. If an intervention type falls below 40% accuracy after 30 outcomes, it loses authority and gets suspended. This is not a veto layer — it's a calibrated specialist that gets stronger the longer it runs.

**2b. Adversarial self-play before the AI PM commits.**
After the AI PM proposes a portfolio, a separate Sonnet red team agent attacks it — not a general critique, a specific challenge to each delta vs the quant baseline: *"The quant wanted SATS at 6.5%. You removed it. If the quant is right, what does that cost?"* The AI PM then revises or defends with a 6-tool research pass. Only the portfolio that survives adversarial pressure gets submitted.

**3. An AI PM that earns its authority incrementally.**
The AI PM starts at 0% allocation and shadow-trades alongside the quant system. After 21 rebalance days with a Sharpe edge > 0.05, it earns 25% authority. It can reach 75%. If its drawdown exceeds the quant's by 5 percentage points, it auto-reverts to Phase 0. The quant system always retains at least 20%. This is not a toggle — it's a continuous trust protocol.

**4. A weekend brain that feeds the weekday.**
Every weekend, the system runs a closed intelligence loop: full alt-data sweep (901 symbols), weekly post-mortem on what worked, adversarial scenario planning with probability-weighted stress tests, ML GridSearch retraining, and an AI PM deep research session across the full universe. Monday morning, the AI PM reads all of it before touching capital. The system gets smarter every week, automatically, for ~$1.50 in API costs.

---

## Live Track Record

> 627 tests · Paper trading since April 1, 2026 · AI PM in shadow period since May 19

52 days of paper trading is not statistically significant — Sharpe standard error after 52 days is ≈ 2.8. We track this honestly, which is why authority is earned over 21 rebalances, not calendar time.

| Date | Event |
|------|-------|
| Apr 1, 2026 | Live paper trading begins · 29 orders |
| Apr 15, 2026 | Rebalance #1 · REDUCE_SIZE debate verdict · 27 orders |
| May 5, 2026 | Rebalance #2 · Full rotation · 40 orders · NAV $104,815 |
| May 19, 2026 | Rebalance #3 · 30 orders · NAV $103,790 · 18 positions · AI PM shadow begins |
| May 22, 2026 | Decision memory + conviction gate live |
| May 23, 2026 | Position health monitor grounded in live return data |
| May 25, 2026 | Weekend intelligence pipeline + bidirectional weekday ↔ weekend wiring |
| May 26, 2026 | Adversarial Intelligence live — 3-layer risk committee, earned authority, ONE change per rebalance |

**Current portfolio (May 2026):** EWY 10.9%, PDBC 6.9%, CBOE/CHRD/HUM/SATS/SNDK/STRL/VICR/VRT/WDC ~6.5% each, DBB 3.8%, EWT/EEM/EWC ~3.4% each, DBA 3.1%, BIL 2.8%, KMLM 2.4%, UUP 1.6%. 18 positions.

---

## Walk-Forward OOS

> Survivorship bias eliminated · fold-local regime fit · CPCV C(6,2)=15 folds · no look-ahead

| Metric | Value | Period |
|--------|------:|--------|
| CAGR | +12.35% | Jan 2020 – Apr 2026 |
| Sharpe | +0.518 | |
| Alpha vs SPY | +0.68% | |
| Max Drawdown | −23.4% | |

The OOS record is the quant system only. The AI PM, debate layer, and self-improve loop require live feedback and are not in this backtest.

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_agents.py            # daily pipeline (or weekend mode on Sat/Sun)
python run_all_agents.py --dry-run  # all logic, no orders
python -m pytest --tb=short -q     # 627 passed, 1 skipped
streamlit run demo_app.py           # investor demo
streamlit run ascent/dashboard/live_dashboard.py --server.port 8502
```

```env
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
├── Weekend detection (Sat/Sun → 10-job intelligence pipeline, once per ISO week)
│   ├── Alt-data full sweep (901 symbols: SEC 10-K/Q · transcripts · Reddit · Trends)
│   ├── LLM fundamental cache refresh (stale symbols only)
│   ├── ML GridSearch retrain (18 hyperparameter combos → best model on Monday)
│   ├── Factor discovery (PySR + LLM proposals → human review gate)
│   ├── Self-improve (20 variants → shadow promotion if Sharpe edge > 0.05)
│   ├── Conviction gate retrain (logistic regression on matured overrides)
│   ├── AI PM deep research (Opus · full 901-symbol universe · no trade output)
│   ├── Weekly debrief (Haiku: what worked · what didn't · systematic bias)
│   ├── Adversarial scenario plan (6 stress tests · LLM probability assessment)
│   └── Memory ingestion (BM25 semantic memory)
│
├── Outcome fills (best-effort, every day)
│   ├── Regime memory: realized returns for episodes ≥21 days old
│   ├── Calibration: conviction-vs-realized IC
│   └── Alpha wedge: 21d price fetch → AI PM vs quant delta per override
│
├── Earned authority update (every day, before rebalance/non-rebalance split)
│   └── Sharpe edge ≥0.05 for 21 rebalances → advance phase
│       Drawdown > quant + 5pp → revert to Phase 0
│
├── Data hub (one parallel fetch, all universes)
│   ├── Yahoo: 901 US equities + macro + international + alternatives
│   ├── FRED: macro series (cached fallback on outage)
│   └── Alt-data: SEC signals · transcripts · Reddit · Google Trends
│
├── 4 Specialist Agents (parallel, ThreadPoolExecutor)
│   ├── US Equities   901 symbols · 14-sleeve alpha → MVO/BL → sector constraints
│   ├── Macro         12 ETFs · trend-only · regime-sized (crisis: top 3 / 40%)
│   ├── International 12 ETFs · EM penalty when USD > 50MA
│   └── Alternatives  7 ETFs · trend 80% + low-vol 20% · kill switch at 12%
│
├── Orchestrator
│   ├── Base by regime: calm_bull US70/mac10/intl12/alt8
│   │                   stressed US45/mac25/intl10/alt20
│   │                   crisis   US30/mac30/intl5/alt35
│   ├── Skill blend · conviction bonus · correlation guard (63d, cap 0.70)
│   ├── Thesis coherence (12 factor buckets · 6 contradiction pairs → 40% cut)
│   ├── EM + commodity hard cap: 20% · post-blend position cap: 10%
│   └── Crisis veto: merged = 0.60×macro + 0.40×merged
│
│ ── NON-REBALANCE ───────────────────────────────────────────────────────────
│
├── Daily Intelligence (9 days accumulate → fed into Monday's rebalance brief)
│   ├── Position health  (pure Python · return/momentum/rank/flag per position)
│   ├── Conviction decay · Signal health · Regime trajectory
│   ├── Historical analogues · Position thesis · Adversarial challenge
│   └── Macro calendar (FOMC/CPI/NFP/earnings exposure per position)
│
│ ── REBALANCE ───────────────────────────────────────────────────────────────
│
├── Rebalance brief (Haiku synthesizes 9 days + weekend intel → AI PM's first read)
│
├── AI PM Agent (Claude Opus 4.6 · 24 tools · rebalance days only)
│   ├── Phase 1: get_rebalance_brief · get_scenario_plan · get_weekend_research
│   │           · regime state · macro · regime memory · alpha wedge
│   ├── Phase 2: All 4 agents (precomputed cache) · momentum screen
│   │           Flag 252d momentum >200% as [EXTENDED]
│   ├── Phase 3: Up to 6 signal tools · query_decision_history · check_override_conviction
│   ├── Phase 4: PRE-MORTEM · coherence check · propose_portfolio
│   ├── Red Team: Sonnet attacks per-position + systemic kill shot
│   └── Revision: Opus defends or revises (6 tools)
│
├── Decision memory ingestion (each override → logs/decision_memory.jsonl)
│   └── Alt-data context auto-filled: sec_tone · transcript_sentiment ·
│       reddit_buzz · trends_direction
│
├── Earned authority blend (Phase 0→1→2→3 · hard cap 0.80)
│
├── Adversarial Intelligence (fires every rebalance · 3 layers · 1 falsifiable weight change)
│   ├── Layer 1: Short thesis per position (batched Haiku · score 0–1 · >0.6 = flagged)
│   ├── Layer 2: Regime-conditional sizing (event_momentum/trend/reversion/etf/unknown × 5 regimes)
│   ├── Layer 3: Narrative coherence (cluster → independent bets · regime flip sensitivity)
│   ├── Asymmetric agents: bull sees altdata positives · bear sees flags · devil sees coherence
│   ├── Judge → ONE position change + falsifiable 10-day prediction + traditional verdict
│   └── Earned authority: win_rate >70%→4% · >50%→2% · <40% after 30 scored → suspended
│
├── Debate (advisory continuation · 5 agents · 2 rounds · Monte Carlo + weekend scenarios)
│   └── Judge → proceed / reduce_size / halt_and_review
│
└── Execution
    ├── Kill switch: SOFT_WARN 8% · HARD_STOP 15%
    ├── Almgren-Chriss cost model · Alpaca paper trading
    └── SHA-256 hash-chain audit trail · GIPS TWR monthly reports
```

---

## Alpha Stack (14 Sleeves)

| Sleeve | Weight | Signal |
|--------|-------:|--------|
| Trend | 38% | Cross-sectional momentum; `mom_252d − mom_21d` skip-month at 20% sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion |
| ML (XGBoost/CPCV) | 10% | 15 folds · 5-day purge+embargo · 12 features · weekend GridSearch |
| Mean Reversion | 5% | Short-term reversal (z-score 20d) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day lag; momentum-neutral |
| Earnings PEAD | 5% | EPS surprise z-score · OLS momentum-beta residual · 1-bday lag |
| Analyst | 5% | Revision signal; zero-filled when cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Haiku; cached by (symbol, quarter) |
| Narrative Alpha | 3% | Quarter-over-quarter thesis shift detection via Haiku |
| Options Flow | 2% | IV-adjusted sentiment; sparse |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Alt Data | 0% | SEC 10-K · transcripts · Reddit · Google Trends; 0% until IC gate passes |

Weights are regime-adaptive via `active_alpha_config.json`. Distressed filter zeroes names with `mom_252d < −0.65`.

---

## The Decision Memory System

```python
{
  "entry_id":              "2026-05-19_SATS",
  "override_type":         "data_quality",   # data_quality / regime_macro /
  "regime":                "calm_bull",        # news_event / correlation_risk / valuation
  "ai_action":             "REMOVED",
  "ai_weight":             0.0,
  "quant_weight":          0.065,
  "momentum_252d":         31.96,             # 3196% = merger artifact, correctly caught
  "sec_tone":              -0.3,              # auto-filled from alt-data caches
  "transcript_sentiment":  null,
  "wedge_21d":             null               # filled automatically 21 days later
}
```

The conviction gate reads this before every override:

| Cases | Win Rate | Result |
|-------|----------|--------|
| < 5 | any | Proceed, −15% size (building track record) |
| ≥ 5 | ≥ 60% + positive wedge | Full size |
| ≥ 5 | ≥ 50% | 75% size |
| ≥ 8 | < 35% | Blocked |
| any | — | `data_quality`, `news_event`, `correlation_risk` always approved |

At n ≥ 30 matured cases, a logistic regression trains automatically on override features. The AI PM interface never changes.

---

## Weekend Intelligence Pipeline

`run_all_agents.py` auto-detects Saturday/Sunday and branches into an 11-job intelligence run. Second run the same weekend: expensive once-per-weekend jobs are skipped.

| Job | Frequency | Cost driver |
|-----|-----------|-------------|
| Alt-data sweep (901 symbols) | Every run | Haiku · SEC + transcripts |
| LLM fundamental cache | Every run | Haiku · stale symbols only |
| ML GridSearch retrain | Once | No LLM · writes flag for Monday |
| Factor discovery | Once | PySR + Haiku proposals |
| Self-improve (20 variants) | Once | No LLM · walk_forward_lightweight |
| Conviction gate retrain | Every run | No LLM · sklearn |
| Adversarial calibration | Every run | No LLM · yfinance T+14d outcome scoring |
| AI PM deep research | Once | Opus · full universe |
| Weekly debrief | Every run | Haiku · one synthesis call |
| Adversarial scenario plan | Every run | Sonnet · 6 scenarios |
| Memory ingestion | Every run | No LLM |

**~$1.50 first run · ~$0.20 second run · ~$6–8/month.**

Monday's AI PM reads the debrief, scenario plan, and research memo as its first three inputs. Devil's advocate in the debate sees flagged scenarios (probability ≥ 40%) with quantified portfolio impact. ML model uses GridSearch-optimized params.

---

## Portfolio Construction

```
Alpha scores → Distressed filter → Black-Litterman (tau scales with IC IR)
  → cvxpy MVO: maximize w'α − λ(w'Σw) − κ‖w − w_prev‖₁
               Σ = B·F·B' + D (FF5+UMD)  ·  CLARABEL → SCS → rank-weight fallback
  → Sector constraints (max 1 per sector · skip if coverage < 80%)
  → Post-blend position cap (10% max, single-pass water-fill)
  → SPY 200MA overlay (×0.70 when SPY < 200MA)
  → VIXY hedge (0–8% by regime × HMM confidence)
```

---

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| 14-sleeve alpha stack | ✅ live | Regime-adaptive weights |
| MVO + Black-Litterman | ✅ live | CLARABEL/SCS/rank-weight fallback chain |
| 4 specialist agents | ✅ live | US equities · macro · international · alternatives |
| Multi-agent orchestration | ✅ live | Skill blend · correlation guard · coherence · position cap |
| Adversarial Intelligence | ✅ live | 3-layer engine · ONE falsifiable change per rebalance · earned authority by type |
| Debate layer | ✅ live | 5 agents · 2 rounds · Monte Carlo + weekend scenario injection |
| AI PM agent | ✅ shadow | Phase 0 (0% weight) since May 19, 2026 · 24 tools |
| Adversarial self-play | ✅ live | Sonnet red team attacks before every submission |
| Decision memory | ✅ live | Compounding from May 19 · ML gate at n≥30 matured cases |
| Conviction gate | ✅ live | Rules-based now · logistic regression at n≥30 |
| Earned authority | ✅ live | Phase 0 → 25% after 21 rebalances with Sharpe edge |
| Non-rebalance intelligence | ✅ live | 7 monitors daily · position health · rebalance brief |
| Weekend intelligence pipeline | ✅ live | 10 jobs · debrief · scenarios · AI PM research · GridSearch |
| Alt-data pipeline | ✅ live | SEC · transcripts · Reddit · Trends · IC gate |
| Factor risk model | ✅ live | FF5+UMD rolling OLS · Ledoit-Wolf covariance |
| Calibration tracking | ✅ live | Conviction IC vs realized 21d returns |
| Self-improve loop | ✅ built | `SELF_MODIFY_ENABLED=False` until +Sharpe 30 days |
| Factor discovery | ✅ built | PySR + LLM proposals · Harvey FDR gate |
| 130/30 long-short | ✅ built | `LONG_SHORT_ENABLED=False` until ≥30 rebalances |
| Real capital | 🔧 pending | ~May–June 2026 |
| 12-month live track record | 📅 April 2027 | YC milestone |

---

## Repository Layout

```
ascent/
  config/        settings, types (AgentOutput), universe (901 symbols)
  data/ingest/   yahoo, fred, sec_filings, earnings_transcripts,
                 reddit_sentiment, google_trends, edgar_listener
  features/      build_features, feature_defs (12 ML features)
  alpha/         14 sleeves + stack combiner
  portfolio/     mvo_optimizer, black_litterman, regime_covariance, long_short
  research/      walk_forward_runner, cpcv, self_improve, factor_discovery/
  regime/        hmm engine, particle_filter, breaks
  risk/          factor_model, covariance_model, pm_risk_validator
  execution/     eod_runner, alpaca_broker, kill_switch, slippage_tracker
  monitoring/    position_health, daily_intelligence, rebalance_brief,
                 weekend_runner, weekly_debrief, scenario_planner
  strategy/      earned_authority, calibration_tracker, conviction_gate
  memory/        decision_memory, regime_memory
  llm/           client.py (Opus/Sonnet/Haiku · retry · cost tracking)

agents/          us_equities_agent, macro_agent, international_agent,
                 alternatives_agent, ai_pm_agent, red_team_agent
orchestrator/    central_intelligence.py
debate/          adversarial_engine, adversarial_authority, adversarial_monitor,
                 debate_runner, agents, judge, outcome_tracker
compliance/      audit_trail (SHA-256 hash chain), performance_report
```

---

*Built by Scott Dong · Paper trading since April 1, 2026 · Target: 12-month live track record by April 2027*
