# Ascent Capital

![Tests](https://img.shields.io/badge/tests-627%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-live%20paper%20trading-informational)
![AI PM](https://img.shields.io/badge/AI%20PM-Claude%20Opus%204.6-blueviolet)
![Alpha](https://img.shields.io/badge/alpha%20sleeves-14-orange)
![OOS Sharpe](https://img.shields.io/badge/OOS%20Sharpe-0.518-green)

### [📊 Live Performance Dashboard →](https://scottdongkhang.github.io/Ascent_Capital)
> Equity curve vs SPY · Current holdings · Full rebalance debate history · Updates automatically after every daily run.

**An AI-native quantitative investment fund built from scratch.** Not a backtest. Not a demo. A live system that reads SEC filings every night, debates its own trades before placing them, and earns the right to manage capital by being demonstrably right in live markets.

---

## In plain English

Most investment funds employ portfolio managers who read research, argue about trade ideas, and decide where to put money. Ascent replaces that workflow with layered software — but software designed to be *harder on itself* than any human committee would be.

Here's the full loop:

1. **The system reads the market every night** — prices for 901 stocks, economic data from the Federal Reserve, SEC filings, earnings call transcripts, and analyst estimates.
2. **14 independent signals score every stock** — from traditional price momentum to machine learning to a language model that reads quarterly earnings reports and flags when a company's narrative is quietly changing.
3. **Four specialist teams propose portfolios simultaneously** — US equities, bonds and commodities, international markets, and alternatives each run their own analysis in parallel.
4. **A senior AI portfolio manager reviews everything** using 24 research tools, then proposes a final allocation. Before it commits, a separate "red team" AI attacks the proposal: *"You removed this stock. If the quant model is right and you're wrong, what does that cost over the next month?"*
5. **A three-layer risk committee reviews every trade** — generating the strongest possible bearish case for each position, checking whether position sizes fit the current market regime, and verifying the portfolio isn't accidentally a single concentrated bet in disguise.
6. **Orders execute, outcomes are tracked.** Every decision is logged with the full context that drove it. Twenty-one days later, the actual market result flows back in and permanently updates the system's judgment.

The AI portfolio manager starts with zero authority and earns the right to manage real capital by demonstrating a measurable edge in live markets — one rebalance at a time.

---

## What makes this different

Most quant repos are backtests dressed up as systems. This one has four properties that don't exist elsewhere:

### 1. A risk committee that earns its authority by being right

Before every rebalance, an adversarial engine runs three analyses the mathematical model structurally cannot perform:

- **Short thesis generation** — A language model generates the strongest possible bearish case for every position in the book, scored 0–1. Positions above 0.6 are flagged to the debate agents.
- **Regime-conditional sizing** — Is each position sized correctly for the current market environment? A trend-following ETF should carry different weight in a crisis than in a calm bull market. The engine maps every position to a type (`event_momentum`, `trend`, `reversion`, `etf`) and checks against a regime-conditional sizing table.
- **Narrative coherence** — How many genuinely independent bets does the portfolio contain? If five positions are all effectively "EM export growth plays," that's one risk, not five. The engine clusters positions by narrative bucket and estimates the portfolio impact of a regime flip.

The engine makes **exactly one prioritized weight change** per rebalance, paired with a falsifiable 10-day prediction. That prediction is scored 14 calendar days later: did the flagged position underperform SPY by more than 1pp? If an intervention type falls below 40% accuracy after 30 outcomes, it loses authority and gets suspended. This is not a veto layer — it's a calibrated specialist that strengthens the longer it runs.

### 2. A portfolio manager that learns from its own history — permanently

Every override the AI PM makes is stored with full context: override type, market regime, its weight vs. the quant model's weight, momentum, and any available alt-data signals (SEC tone, transcript sentiment, insider activity). Twenty-one days later, the realized return fills in automatically.

Before any future decision of the same type in the same regime, the AI PM queries its own historical win rate. If it has been wrong on valuation calls in calm bull markets, the conviction gate reduces position size or blocks the move. At 30 matured cases, a logistic regression trains automatically on the accumulated history.

This is a compounding proprietary database. No system starting today has it, and it becomes more accurate with every rebalance.

### 3. Earned authority, not assumed authority

The AI PM starts at 0% allocation and shadow-trades alongside the quant system. After 21 rebalance days with a Sharpe edge ≥ 0.05, it earns 25% of the capital allocation. It can reach 75%. If its drawdown exceeds the quant's by 5 percentage points at any point, it automatically reverts to Phase 0. The quant system always retains at least 20%.

This is not a toggle — it's a continuous trust protocol with automatic safeguards.

### 4. A weekend brain that feeds the weekday

Every Saturday and Sunday, the system runs a closed intelligence cycle: full alt-data sweep across 901 symbols, weekly post-mortem on attribution and AI PM override performance, adversarial scenario planning with probability-weighted stress tests, ML model retraining with hyperparameter search, and an AI PM deep research session across the full universe.

Monday morning, the AI PM reads the debrief, the flagged scenarios, and the research memo before considering a single trade. The devil's advocate in the debate layer sees scenarios with LLM-assessed probabilities and estimated portfolio impact.

Total cost: approximately $1.50 in API calls.

---

## Live Track Record

> Paper trading since April 1, 2026 · AI PM shadow period began May 19, 2026 · **[Full dashboard with equity curve →](https://scottdongkhang.github.io/Ascent_Capital)**

<!-- LIVE_STATS_START -->
| Metric | Value |
|--------|-------|
| Current NAV | $110,100 |
| Total Return | +9.21% |
| Alpha vs SPY | -5.32% |
| Sharpe (Ann.) | 2.919 |
| Max Drawdown | -4.32% |
| Days Live | 39 |
| Open Positions | 17 |
| Last Updated | 2026-05-28 |
<!-- LIVE_STATS_END -->

*Sharpe standard error over ~40 days is ~2.8 — not statistically significant yet. This is tracked honestly, which is why authority is earned over 21 rebalances, not calendar time.*

| Date | Event |
|------|-------|
| Apr 1, 2026 | Paper trading live · 29 orders · 9 initial positions |
| Apr 15, 2026 | Rebalance #1 · `REDUCE_SIZE` debate verdict · 27 orders |
| May 5, 2026 | Rebalance #2 · Full rotation · 40 orders · NAV $104,815 |
| May 19, 2026 | Rebalance #3 · 30 orders · NAV $103,790 · 18 positions · AI PM shadow begins |
| May 22, 2026 | Decision memory + ML conviction gate live |
| May 23, 2026 | Position health monitor grounded in live return data |
| May 25, 2026 | Weekend intelligence pipeline + bidirectional wiring |
| May 26, 2026 | Adversarial Intelligence live — 3-layer engine, earned authority, one falsifiable change per rebalance |

---

## Walk-Forward Out-of-Sample Results

> Survivorship bias eliminated · fold-local regime fit · CPCV C(6,2) = 15 folds · no look-ahead

The walk-forward runner uses `get_universe_on_date()` on every fold — symbols excluded if outside their validity window at the decision point. The regime engine is fitted on the training slice only. No future data contaminates any fold.

| Metric | Value | Period |
|--------|------:|--------|
| CAGR | +12.35% | Jan 2020 – Apr 2026 |
| Sharpe Ratio | +0.518 | |
| Alpha vs SPY | +0.68% | |
| Max Drawdown | −23.4% | |

These figures reflect the quant system only. The AI PM, debate layer, and self-improve loop require live feedback by design and are not in this backtest. The OOS record cannot be called fully clean until A4 (survivorship bias hardening across all folds) is complete.

---

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_agents.py            # daily pipeline (auto-detects weekend mode on Sat/Sun)
python run_all_agents.py --dry-run  # full logic, no orders submitted
python -m pytest --tb=short -q     # 627 passed, 1 skipped
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

---

## System Architecture

```
python run_all_agents.py  ·  daily  ·  1:45 PM ET via launchd
│
├── Weekend detection (Sat/Sun → 11-job intelligence pipeline, once per ISO week)
│   ├── Alt-data full sweep (901 symbols: SEC 10-K/Q · transcripts · Trends)
│   ├── LLM fundamental cache refresh (stale symbols only)
│   ├── ML GridSearch retrain (18 hyperparameter combos → Monday's model)
│   ├── Factor discovery (PySR symbolic regression + LLM proposals → human review)
│   ├── Self-improve (20 variants → shadow promotion if Sharpe edge > 0.05)
│   ├── Conviction gate retrain (logistic regression on matured override history)
│   ├── Adversarial calibration (T+14d outcome scoring for pending interventions)
│   ├── AI PM deep research (Opus · full 901-symbol universe · no trade output)
│   ├── Weekly debrief (Haiku: attribution · AI PM performance · systematic bias)
│   ├── Adversarial scenario plan (6 stress tests · LLM probability + impact assessment)
│   └── Memory ingestion (BM25 semantic memory)
│
├── Outcome fills (every day)
│   ├── Regime memory: realized returns for episodes ≥21 calendar days old
│   ├── Calibration: conviction-vs-realized IC (Spearman)
│   └── Alpha wedge: 21d price fetch → AI PM vs quant delta per override
│
├── Earned authority update (every day, before rebalance/non-rebalance split)
│   └── Sharpe edge ≥0.05 for 21 rebalances → advance phase
│       Drawdown > quant + 5pp → revert to Phase 0
│
├── Data hub (one parallel fetch, all universes)
│   ├── Yahoo Finance: 901 US equities + macro ETFs + international + alternatives
│   ├── FRED: macro time series (cached fallback on outage)
│   └── Alt-data: SEC signals · transcripts · Google Trends
│
├── 4 Specialist Agents (parallel, ThreadPoolExecutor)
│   ├── US Equities    901 symbols · 14-sleeve alpha → MVO/BL → sector constraints
│   ├── Macro          12 ETFs · trend-only · regime-sized (crisis: top 3 / 40%)
│   ├── International  12 ETFs · EM alpha penalty when USD > 50MA
│   └── Alternatives   7 ETFs · trend 80% + low-vol 20% · kill switch at 12%
│
├── Orchestrator
│   ├── Base by regime: calm_bull  US70 / mac10 / intl12 / alt8
│   │                   stressed   US45 / mac25 / intl10 / alt20
│   │                   crisis     US30 / mac30 / intl5  / alt35
│   ├── Skill blend · conviction bonus (+15% when ≥2 agents agree, conv > 0.3)
│   ├── Correlation guard: 63-day cross-agent cap at 0.70 → halve smaller agent
│   ├── Thesis coherence: 12 factor buckets · 6 contradiction pairs → 40% cut
│   ├── EM + commodity hard cap: 20% aggregate
│   ├── Post-blend position cap: 10% per name (single-pass water-fill)
│   └── Crisis veto: merged = 0.60 × macro + 0.40 × merged
│
│ ── NON-REBALANCE ──────────────────────────────────────────────────────────
│
├── Daily Intelligence (9 days accumulate → feeds Monday's rebalance brief)
│   ├── Position health    (pure Python · return/momentum/rank/flag per position)
│   ├── Conviction decay · Signal health · Regime trajectory · Macro calendar
│   ├── Historical analogues (regime + weight pattern matching against episode log)
│   ├── Position thesis monitor (Haiku: flag DETERIORATING positions for review)
│   └── Adversarial challenge (Haiku: worst-case scenario per current position)
│
│ ── REBALANCE ──────────────────────────────────────────────────────────────
│
├── Rebalance brief (Haiku synthesizes 9 days + weekend intel → AI PM's first read)
│
├── AI PM Agent (Claude Opus 4.6 · 24 tools · rebalance days only)
│   ├── Phase 1: rebalance brief · scenario plan · weekend research
│   │           regime state · macro · regime memory · alpha wedge
│   ├── Phase 2: All 4 agents (precomputed cache) · momentum screen
│   │           Flag 252d momentum > 200% as [EXTENDED]
│   ├── Phase 3: Up to 6 signal tools · query_decision_history
│   │           · check_override_conviction (gate before every override)
│   ├── Phase 4: PRE-MORTEM · coherence check · propose_portfolio(weights, thesis)
│   ├── Red Team: Sonnet attacks per-position delta + systemic kill shot
│   └── Revision: Opus defends or revises (max 6 tool calls)
│
├── Decision memory ingestion (each override → decision_memory.jsonl)
│   └── Alt-data context auto-filled: sec_tone · transcript_sentiment ·
│       trends_direction
│
├── Earned authority blend (Phase 0 → 1 → 2 → 3 · hard cap 0.80)
│
├── Adversarial Intelligence (every rebalance · 3 layers · 1 falsifiable weight change)
│   ├── Layer 1: Short thesis per position (batched Haiku · score 0–1 · >0.6 flagged)
│   ├── Layer 2: Regime-conditional sizing (5 position types × 5 regime states)
│   ├── Layer 3: Narrative coherence (cluster → independent bets · regime flip cost)
│   ├── Asymmetric debate: bull sees alt-data positives · bear sees flagged positions
│   │                      devil sees coherence analysis · regime specialist sizes
│   ├── Judge → ONE position change + falsifiable 10-day prediction
│   └── Authority tiers: win_rate >70% → 4% max change · >50% → 2% · <40% → suspended
│
├── Debate (advisory · 5 agents · 2 rounds · Monte Carlo + weekend scenario injection)
│   └── Verdict: proceed / reduce_size (Haiku adjusts weights) / halt_and_review
│
└── Execution
    ├── Kill switch: SOFT_WARN 8% drawdown · HARD_STOP 15% (alternatives: 12%)
    ├── Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV
    ├── Alpaca paper trading: retry ×3, 0.4s inter-order delay
    └── SHA-256 hash-chain audit trail · GIPS TWR monthly reports
```

---

## Alpha Stack — 14 Sleeves

| Sleeve | Weight | Signal Construction |
|--------|-------:|---------------------|
| Trend | 38% | Cross-sectional momentum; `mom_252d − mom_21d` skip-month at 20% sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion (requires `profiles.parquet` sector labels) |
| ML (XGBoost/CPCV) | 10% | C(6,2)=15 folds · 5-day purge + embargo · 12 features · weekend GridSearch |
| Mean Reversion | 5% | Short-term reversal (z-score, 20-day window) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day filing lag; momentum-neutral |
| Earnings (PEAD) | 5% | EPS surprise z-score · OLS momentum-beta residual · 1-bday lag |
| Analyst | 5% | Revision signal; zero-filled when cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step chain-of-thought via Haiku; cached by (symbol, quarter) |
| Narrative Alpha | 3% | Quarter-over-quarter thesis shift detection via Haiku; keyed by md5(content) |
| Options Flow | 2% | IV-adjusted sentiment; sparse — zero-filled if cache absent |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Alt Data | 0% | SEC 10-K · earnings transcripts · Google Trends; 0% until IC gate passes |

Sleeve weights are regime-adaptive via `data_cache/active_alpha_config.json`. Distressed filter zeroes names with `mom_252d < −0.65` after blending. The `DEFAULT_ALPHA_WEIGHTS` dict exists in both `stack.py` and `self_improve.py` — adding a sleeve requires updating both.

---

## Portfolio Construction

```
Alpha scores
  → Distressed filter (zero out mom_252d < −0.65)
  → Black-Litterman blending
      tau scales with IC IR: IR < 0.30 → tau=0.05 · IR < 0.60 → tau=0.10 · else tau=0.15
  → cvxpy MVO (CLARABEL → SCS → rank-weight fallback)
      Objective: maximize w'α − λ(w'Σw) − κ‖w − w_prev‖₁
      Covariance: Σ = B·F·B' + D  (Fama-French 5 + UMD, rolling 252-day OLS)
  → Sector constraints
      max 1 position per sector · skip sector caps if coverage < 80%
  → Post-blend position cap
      10% max per name · single-pass water-fill redistribution
  → SPY 200MA overlay
      multiply all weights × 0.70 when SPY < 200-day moving average
  → VIXY hedge overlay
      0–8% allocation sized by regime × HMM confidence
```

---

## The Decision Memory System

Every AI PM override is stored in this schema:

```python
{
  "entry_id":              "2026-05-19_SATS",
  "override_type":         "data_quality",    # data_quality / regime_macro /
  "regime":                "calm_bull",        # news_event / correlation_risk / valuation
  "ai_action":             "REMOVED",
  "ai_weight":             0.0,
  "quant_weight":          0.065,
  "momentum_252d":         31.96,             # 3196% = SanDisk/WDC merger artifact; correctly caught
  "sec_tone":              -0.3,              # auto-filled from alt-data caches
  "transcript_sentiment":  null,
  "wedge_21d":             null               # auto-filled 21 calendar days later
}
```

The conviction gate reads this history before every future override of the same type and regime:

| Cases | Win Rate | Result |
|-------|----------|--------|
| < 5 | any | Proceed at −15% size (building track record) |
| ≥ 5 | ≥ 60% + positive wedge | Full size |
| ≥ 5 | ≥ 50% | 75% size |
| ≥ 8 | < 35% | Blocked |
| any | — | `data_quality`, `news_event`, `correlation_risk` always approved |

At n ≥ 30 matured cases, a logistic regression trains automatically on the 15-dimensional feature vector. The AI PM interface never changes — it just becomes more accurate.

---

## Weekend Intelligence Pipeline

`run_all_agents.py` auto-detects Saturday/Sunday and branches into an 11-job intelligence run. A second run the same weekend skips expensive once-per-weekend jobs.

| Job | Frequency | Cost driver |
|-----|-----------|-------------|
| Alt-data sweep (901 symbols) | Every run | Haiku · SEC + transcripts |
| LLM fundamental cache | Every run | Haiku · stale symbols only |
| ML GridSearch retrain | Once | No LLM · writes flag for Monday |
| Factor discovery | Once | PySR + Haiku proposals → `outputs/factor_proposals/` |
| Self-improve (20 variants) | Once | No LLM · `walk_forward_lightweight` |
| Conviction gate retrain | Every run | No LLM · sklearn |
| Adversarial calibration | Every run | No LLM · yfinance T+14d outcome scoring |
| AI PM deep research | Once | Opus · full universe · no trade output |
| Weekly debrief | Every run | Haiku · one synthesis call |
| Adversarial scenario plan | Every run | Sonnet · 6 scenarios |
| Memory ingestion | Every run | No LLM |

**~$1.50 first run · ~$0.20 second run · ~$6–8/month.**

Monday's AI PM reads the debrief, scenario plan, and research memo as its first three inputs before considering any individual position.

---

## Regime System

HMM with K=2–4 states (best K selected via walk-forward cross-validation). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`.

**Transition logic**: enter threshold 0.55 · exit threshold 0.35 · minimum dwell 3 days · entropy > 0.90 forces `uncertain`. Particle filter (500-particle SIR) runs continuously; batch refit every 5 days. Emergency refit triggers: SPY −3% + VIX > 30, 200MA cross, SPY/TLT correlation flip, break z-score > 3.5.

Regime propagates through sleeve weights, position size caps, orchestrator base allocations, debate context, AI PM episodic memory, and VIXY hedge sizing.

---

## Build Status

| Component | Status | Notes |
|-----------|--------|-------|
| 14-sleeve alpha stack | ✅ live | Regime-adaptive weights via `active_alpha_config.json` |
| MVO + Black-Litterman | ✅ live | CLARABEL → SCS → rank-weight fallback chain |
| Factor risk model | ✅ live | FF5+UMD rolling OLS · Ledoit-Wolf shrinkage |
| 4 specialist agents | ✅ live | US equities · macro · international · alternatives |
| Multi-agent orchestration | ✅ live | Skill blend · correlation guard · coherence · position cap |
| Adversarial Intelligence | ✅ live | 3-layer engine · ONE falsifiable change per rebalance · earned authority by type |
| Debate layer | ✅ live | 5 agents · 2 rounds · Monte Carlo + weekend scenario injection |
| AI PM agent | ✅ shadow | Phase 0 (0% weight) since May 19, 2026 · 24 tools |
| Adversarial self-play | ✅ live | Sonnet red team attacks before every AI PM submission |
| Decision memory | ✅ live | Compounding from May 19 · ML gate activates at n≥30 matured cases |
| Conviction gate | ✅ live | Rules-based · logistic regression ready at n≥30 |
| Earned authority | ✅ live | Phase 0 → 25% after 21 rebalances with Sharpe edge ≥0.05 |
| Episodic regime memory | ✅ live | Per-regime outcome log · prefix-match query by AI PM |
| Calibration tracking | ✅ live | Conviction-vs-realized Spearman IC; ≥0.20 = Calibrated |
| Non-rebalance intelligence | ✅ live | 7 monitors daily · position health · rebalance brief |
| Weekend intelligence pipeline | ✅ live | 11 jobs · debrief · scenarios · AI PM research · GridSearch |
| Alt-data pipeline | ✅ live | SEC 7-signal · transcripts · Google Trends · IC gate |
| Self-improve loop | ✅ built | `SELF_MODIFY_ENABLED=False` until positive Sharpe for 30 consecutive days |
| Factor discovery | ✅ built | PySR + LLM proposals · Harvey FDR gate · human review required |
| 130/30 long-short | ✅ built | `LONG_SHORT_ENABLED=False` until ≥30 paper rebalances (~Aug 2026) |
| TimescaleDB / WebSocket | 🔧 pending | Requires Docker + `ALPACA_KEY` configuration |
| Real capital deployment | 🔧 pending | ~May–June 2026 |
| 12-month live track record | 📅 April 2027 | YC milestone |

---

## Repository Layout

```
ascent/
  config/        settings · types (AgentOutput) · universe (901 symbols)
  data/ingest/   yahoo · fred · sec_filings · earnings_transcripts · google_trends
  features/      build_features · feature_defs (12 ML features including sector-rel-mom)
  alpha/         14 sleeves + stack combiner + narrative_alpha
  portfolio/     mvo_optimizer · black_litterman · regime_covariance · long_short
  research/      walk_forward_runner · cpcv · self_improve · factor_discovery/
  regime/        hmm engine · particle_filter · breaks · posture
  risk/          factor_model · covariance_model · pm_risk_validator
  execution/     eod_runner · alpaca_broker · kill_switch · slippage_tracker
  monitoring/    position_health · daily_intelligence · rebalance_brief
                 weekend_runner · weekly_debrief · scenario_planner
  strategy/      earned_authority · calibration_tracker · conviction_gate
  memory/        decision_memory · regime_memory
  llm/           client.py (Opus/Sonnet/Haiku · retry · per-model cost tracking)

agents/          us_equities_agent · macro_agent · international_agent
                 alternatives_agent · ai_pm_agent · red_team_agent
orchestrator/    central_intelligence.py
debate/          adversarial_engine · adversarial_authority · adversarial_monitor
                 debate_runner · agents · judge · outcome_tracker
compliance/      audit_trail (SHA-256 hash chain) · performance_report (GIPS TWR)
docs/            methodology.md · risk_disclosures.md · superpowers/plans/ · specs/
```

---

## Integrity Constraints

These are enforced in code, not just convention:

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime engine fitted on training slice only; ML targets not leaked into feature windows
2. **No simulated data under live cache names** — fallback writes to `prices_live_fallback_simulated`; cache name always reflects data provenance
3. **Max-weight hard cap** — `_water_fill_cap()` with post-condition check before returning weights
4. **Sector constraint with coverage fallback** — < 80% valid sector labels → skip sector caps + warn; portfolio never collapses to a single sector
5. **Failed folds must be visible** — no silent zero-weight fallback in walk-forward; every failure logs fold date, stage, and exception type
6. **Debate is advisory only** — debate and adversarial engine never write to alpha, portfolio, or execution modules
7. **Alpha sleeve registry** — `DEFAULT_ALPHA_WEIGHTS` exists in both `stack.py` and `self_improve.py`; adding a sleeve requires updating both or integrity tests fail

---

*Built by Scott Dong · Paper trading since April 1, 2026 · Target: 12-month live track record by April 2027*
