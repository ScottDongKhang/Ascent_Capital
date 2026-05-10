# Ascent Capital

> Institutional-grade quantitative trading platform — live on Alpaca paper trading since April 2026.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Ascent Capital Stack                             │
│                                                                          │
│  Market Data ──► Data Layer ──► Alpha Engine ──► Portfolio Construction │
│  Yahoo / FRED      (ingest,       (13 sleeves,     (MVO + BL, sector-   │
│  EDGAR / Alt Data   normalize,     regime-adapt,    constrained,         │
│  Options / Reddit   cache)         IC-validated)    risk-budgeted)       │
│                                                                          │
│        │                │                │                │             │
│   Regime HMM      Factor Risk        LLM Debate        Execution        │
│  (particle filter,  Model (FF5+UMD,  (Bull/Bear/DA/   (TWAP, IS,        │
│   5 labels, auto-   Ledoit-Wolf Σ,   Judge, Opus)     approval gate)    │
│   emergency refit)  factor P&L)                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
python run_all_agents.py  (daily, 1:45 PM ET via launchd)
│
├─ Step 0: Factor data update + intraday trigger check
│
├─ 4 Specialist Agents (parallel, AgentScope)
│   ├── US Equities  ── 901 symbols (S&P 500 + 400), full 13-sleeve alpha stack
│   ├── Macro        ── 12 ETFs, trend-only, regime-sized
│   ├── International── 12 ETFs, EM-aware (UUP > 50MA → 20% EM penalty)
│   └── Alternatives ── 7 ETFs, VIXY hedge, kill switch at 12% drawdown
│
├─ Orchestrator  (skill-score capital allocation)
│   ├── Regime base:   calm_bull US60/mac15/intl15/alt10
│   │                  stressed  US45/mac25/intl10/alt20
│   │                  crisis    US30/mac30/intl5/alt35
│   ├── Skill blend:   50% skill + 50% base (negative Sharpe → zero)
│   ├── Conviction bonus: +15% when ≥2 agents share a name
│   ├── Correlation guard: 63-day cross-agent cap at 0.70
│   ├── EM+commodity cap: hard 20% aggregate
│   └── Crisis veto: 60% macro on crisis regime
│
├─ Debate Layer  (rebalance days only — advisory)
│   ├── Blind spot detection + catalyst scan
│   ├── Round 1: Bull / Bear / Devil's Advocate (Opus) + Regime + Quant
│   ├── Round 2: Cross-rebuttals
│   └── Judge → proceed / reduce_size / halt_and_review
│
└─ Execution
    ├── Approval gate (>2% NAV)
    ├── TWAP executor (>5% ADV, TWAP_ENABLED=False pending validation)
    ├── Alpaca paper trading
    └── IS decomposition + slippage tracking
```

---

## Alpha Stack (13 Sleeves)

| Sleeve | Weight | Signal Construction |
|--------|-------:|---------------------|
| Trend | 41% | Cross-sectional momentum; skip-last-month `mom_252d − mom_21d` at 0.20 sub-weight |
| Stat-arb | 15% | Sector-residual mean reversion; needs `profiles.parquet` |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, 5-day purge+embargo; 6 features by IC/IR; p5 guard > −0.05 |
| Mean Reversion | 5% | Short-term reversal (z-score 20d) |
| Volatility | 5% | Long declining+stable vol: `−vol_trend_10d / vol_of_vol_21d` |
| Fundamental | 5% | Gross profitability + accruals + asset growth; 45-day lag; momentum-neutral |
| Earnings PEAD | 5% | EPS surprise z-score; OLS momentum-beta residual; 1-bday lag |
| Analyst | 5% | Revision signal; sparse — zero-filled when cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Claude Haiku; cached by (symbol, quarter) |
| Options Flow | 2% | IV-adjusted sentiment; sparse |
| Insider | 2% | Net transaction score; sparse |
| Short Interest | 2% | Short squeeze signal; sparse |
| Alt Data | 0% | IC-validated: SEC 10-K, transcripts, Reddit, Google Trends (0% until gate passed) |

**Distressed filter:** zeroes alpha for names with `mom_252d < −0.65` (down >65% YoY).  
**Regime-adaptive:** sleeve weights read from `data_cache/active_alpha_config.json`; updated weekly.

---

## Portfolio Construction

```
Cross-sectional alpha scores
            │
            ▼
  Black-Litterman blending
  (quant prior + LLM view, tau by IC IR)
  IC IR < 0.30 → tau=0.05  │  0.30–0.60 → tau=0.10  │  >0.60 → tau=0.15
            │
            ▼
  cvxpy MVO  (CLARABEL solver → SCS fallback → rank-weight fallback)
  Objective: w'α − λ(w'Σw) − κ‖w − w_prev‖₁
  Σ = B·F·B' + D  (FF5+UMD factor covariance + idiosyncratic)
            │
            ▼
  Sector constraints  (max 1 per sector, fallback if coverage < 80%)
            │
            ▼
  SPY 200MA overlay (×0.70 when SPY < 200MA)
            │
            ▼
  VIXY hedge overlay (0–8% by regime × confidence)
```

---

## Regime System

```
HMM  K=2–4  (walk-forward CV selects K)
Labels: calm_bull | stressed | crisis | neutral | uncertain

Hysteresis: enter 0.55 / exit 0.35 / min dwell 3 days
Entropy > 0.90 → uncertain

Particle filter: 500 particles SIR
Emergency refit triggers:
  • SPY −3% intraday AND VIX > 30
  • SPY crosses 200-day MA
  • SPY/TLT correlation sign flip
  • Structural break z-score > 3.5
  • > 5 days stale

Leading indicators: credit spread (HYG/LQD), yield curve slope (TLT/IEF)
```

---

## Debate Layer

```
Blind spot detector ──► Catalyst scanner ──► Monte Carlo (p5–p95)
                                                    │
              ┌─────────────────────────────────────┘
              │              Round 1
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Bull (Opus)    Bear (Opus)    Devil's Advocate (Opus)  │
  │  Regime Specialist (Haiku)    Quant Sanity (Python)     │
  └─────────────────────────────────────────────────────────┘
              │              Round 2
              ▼        (cross-rebuttals)
  ┌─────────────────────────────────────────────────────────┐
  │  Bull rebuts Bear/DA | Bear rebuts Bull/DA | DA rebuts  │
  └─────────────────────────────────────────────────────────┘
              │
              ▼
         Judge (Opus)
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 proceed  reduce_size  halt_and_review
            (Haiku        (skip + log
          adjusts wts)   halt_state.json)

Disagreement score: TF-IDF cosine (monitoring — not a verdict override)
```

---

## Event-Driven Architecture

```
EDGAR RSS (5 min)        Capitol Trades (60 min)    Options Anomaly (15 min)
       │                         │                          │
       ▼                         ▼                          ▼
  8-K Haiku            Rule-based classifier         IV z-score +
  5 few-shot           conviction=0.4                put/call z-score
  examples             30–45 day lag                 thresholds
       │                         │                          │
       └─────────────────────────┴──────────────────────────┘
                                 │
                    Event signal: direction × conviction × urgency
                                 │
                    Kill switch: EVENT_TRADING_ENABLED = False
                    Size cap: 0.5% NAV per event trade
                    Approval gate: >1% NAV
                    IC tracked: weekly Spearman
```

---

## Alternative Data Pipeline

```
SEC 10-K/10-Q     Earnings Transcripts    Reddit Mentions    Google Trends
 Haiku 5-axis      Haiku 4-axis           PRAW + TextBlob    pytrends velocity
 classifier        classifier             Contrarian z-score  Weekly, 1-day lag
 45-day lag        1-bday lag             Daily              Normalized 0–1
 90-day ffill      63-day ffill
       │                 │                      │                    │
       └─────────────────┴──────────────────────┴────────────────────┘
                                   │
                      IC Validation Gate (Harvey FDR)
                      IC_mean ≥ 0.015 | IC_IR ≥ 0.60
                      IC_min_regime > 0.010 | n_obs ≥ 20
                                   │
              Accepted → outputs/altdata_proposals/ (human review required)
              Nothing auto-deploys into the live alpha stack
```

---

## Execution Excellence

```
Signal generation @ 1:45 PM
         │
         ├─ Record decision price (IS tracking)
         │
         ▼
  Almgren-Chriss cost model
  Block >10% ADV | Warn >5% ADV | Unknown volume → log warning
         │
         ├── >5% ADV? ──► TWAP executor (kill switch: TWAP_ENABLED=False)
         │               Almgren-Chriss optimal window
         │               Child limit orders (bid±1tick, refreshed each slice)
         │
         └── ≤5% ADV? ──► Market order at close
                                 │
                     IS Decomposition (per fill):
                     ┌────────────────────────────────┐
                     │ delay_cost    signal→arrival   │
                     │ market_impact arrival→fill     │
                     │ opportunity   unfilled shares  │
                     └────────────────────────────────┘
                                 │
                     Capacity model (weekly):
                     max AUM per sleeve before signal decay
```

---

## Intraday Triggers

```
12:00 PM ET  and  14:30 PM ET daily checks:

(a) Regime emergency:
    SPY < −3% intraday AND VIX > 30
    → multiply all weights × 0.70 (de-risk)

(b) Drawdown pre-emption:
    drawdown ≥ 12% (soft warn threshold)
    → reduce gross exposure by 20%

(c) Event urgency:
    high-urgency event in last 60 min for top-5 position
    → 50% partial trim of flagged position

All adjustments: TWAP urgency="high" (≤15 min window)
Logged to: logs/intraday_adjustments.jsonl
```

---

## Walk-Forward OOS Record

> Honest out-of-sample: no look-ahead bias, per-fold universe filter via `get_universe_on_date()`, regime fitted on training slice only.

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
| Apr 1, 2026 | Live paper trading begins |
| Apr 15, 2026 | First rebalance: REDUCE_SIZE 0.88, 27 orders to Alpaca |
| May 5, 2026 | Full portfolio rotation: 40 orders |
| May 7, 2026 | NAV $104,815 | Regime: calm_bull |
| May 10, 2026 | Plans 1–5 complete — 420 tests passing |

**Current portfolio (May 2026):** KMLM, IFRA, AMKR, FIX, WDC, VICR, VRT, VAL, WCC, DBB, CNC, WFRD, PDBC, STLD, DBA, CHRD, EWY, EWC, IRM, MUSA + VIXY hedge.

---

## Self-Improve Loop

```
Sunday 6 AM:
  LLM hypothesis generation (factor_proposer.py, Haiku)
           +
  Random perturbation (fallback)
           │
           ▼
  5 sleeve-weight variants
           │
           ▼
  Real multi-fold OOS scoring
  (3–4 expanding folds, 5-day purge+embargo,
   per-fold universe filter, Sharpe metric)
           │
  Edge > 0.05 Sharpe → 30-day shadow period
           │
  Shadow beats live after 30 days → auto-promote
  (shadow_promoter.py → active_alpha_config.json)

SELF_MODIFY_ENABLED = False
Activates when: +OOS Sharpe for 30 consecutive trading days on flat config
Expected gate: ~July 2026
```

---

## Factor Discovery Loop

```
First Sunday of each month:

Path A: PySR symbolic regression on pre-computed feature panel
Path B: Claude Haiku proposes JSON template parameters

        │
        ▼
  Leakage scanner (AST + regex — blocks lookahead patterns)
        │
        ▼
  Per-regime Spearman IC evaluation (CPCV)
  Harvey FDR gate:
    IC_mean ≥ 0.015 | IC_IR ≥ 0.60 | IC_min_regime > 0.010
        │
  Accepted → outputs/factor_proposals/ (human review)
  Nothing auto-deploys

Gate conditions for live activation: ~July 2026
```

---

## Institutional Roadmap

| # | Plan | Status | What It Adds |
|---|------|--------|--------------|
| 1 | Factor Risk Model | ✅ Done | FF5+UMD rolling OLS, Ledoit-Wolf Σ, factor P&L |
| 2 | Portfolio Construction | ✅ Done | cvxpy MVO, Black-Litterman, regime covariance |
| 3 | Event-Driven Architecture | ✅ Done | EDGAR, Capitol Trades, options anomaly, event trades |
| 4 | Alternative Data Pipeline | ✅ Done | SEC, transcripts, Reddit, Google Trends, IC gate |
| 5 | Execution Excellence | ✅ Done | TWAP, IS decomposition, capacity model, intraday triggers |
| 6 | Real-Time Infrastructure | Planned | WebSocket streaming, TimescaleDB, live dashboard, PDF reports |
| 7 | Live Track Record | Planned | Immutable audit trail, GIPS performance, risk disclosures |

**420 tests passing** | **Plans 6–7 target: Q3 2026** | **YC-ready target: April 2027**

---

## Running the System

```bash
# Activate virtual environment
source .venv/bin/activate

# Daily pipeline (also runs via launchd at 1:45 PM)
python run_all_agents.py

# Run test suite
python -m pytest --tb=short -q
# → 420 passed, 1 skipped

# Interactive demo (Streamlit)
streamlit run demo_app.py
```

---

## Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env with:
ANTHROPIC_API_KEY=...
ALPACA_KEY=...
ALPACA_SECRET=...
FRED_API_KEY=...
REDDIT_CLIENT_ID=...      # optional, for Reddit alt data
REDDIT_CLIENT_SECRET=...
OPENROUTER_API_KEY=...    # optional, for OpenRouter debate models
```

---

## Repository Layout

```
ascent/
  config/       Settings, APIKeys, types (AgentOutput), universe JSON
  data/
    ingest/     yahoo, fred, edgar, capitol_trades, options_scanner,
                sec_filings, earnings_transcripts, reddit_sentiment, google_trends
    store/      parquet cache, point_in_time joins
    validate/   altdata_validator (IC gate)
  features/     build_features, feature_defs, targets
  alpha/        trend, meanrev, statarb, ml_sleeve, fundamental, earnings,
                llm_fundamental, analyst, options_flow, insider, short_interest,
                event_alpha, altdata_alpha, stack
  portfolio/    mvo_optimizer, black_litterman, regime_covariance,
                optimizer, hedge_overlay
  risk/         factor_data, factor_model, covariance_model,
                factor_exposure, factor_constraints
  research/     walk_forward_runner, walk_forward_lightweight, cpcv,
                self_improve, shadow_promoter, factor_proposer,
                factor_discovery/ (PySR + LLM + Harvey FDR gate)
  regime/       engine, model, features, decision, particle_filter, breaks
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch,
                cost_model, slippage_tracker, approval_server,
                twap_executor, implementation_shortfall,
                capacity_model, intraday_trigger, event_runner
  monitoring/   skill_tracker, forward_pnl_tracker, attribution,
                slippage_ic_feedback, pre_rebalance_checklist
  reporting/    market_memo, ic_brief_generator, blind_spot_detector,
                catalyst_scanner, debrief
  llm/          client.py (centralized Claude wrapper, retry 3×)

agents/         us_equities, macro, international, alternatives, event_agent
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker,
                disagreement_scorer, agent_tools
memory/         r2r_interface, reflection_agent
simulation/     mirofish_interface

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl,
                active_alpha_config.json, altdata_*.parquet,
                factor_returns.parquet, factor_loadings.parquet
dashboard/      HTML dashboards, regime_signal.json, regime_labels.csv,
                agent_skill_scores.json, factor_exposures.json
outputs/
  debate_log/         verdict_YYYY-MM-DD.json
  factor_proposals/   autonomous factor proposals (human review)
  altdata_proposals/  validated alt-data proposals (human review)
logs/           eod_log, slippage_log, event_trades, capacity_log,
                intraday_adjustments, self_improve_log, skill_scores_log

ascent/main.py         core pipeline entry point
run_all_agents.py      single daily command (launchd trigger)
demo_app.py            Streamlit demo (dark/gold aesthetic, live LLM debate)
```

---

*Built by Scott Dong · Live since April 1, 2026 · Alpaca paper trading · Mac Air M5*
