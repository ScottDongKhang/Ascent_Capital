# Ascent Capital

**Modular Python Quant Research & Trading Platform**

A full-stack systematic trading system: market data ingestion → feature engineering → multi-sleeve alpha → regime-adaptive portfolio construction → walk-forward OOS evaluation → 4 specialist agents → orchestration → LLM debate → live execution via Alpaca paper trading.

---

## Quick Start

```bash
# Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env   # set ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY

# Daily run (non-rebalance: agents + PnL + skill scores; rebalance: + debate + Alpaca orders)
.venv/bin/python run_all_agents.py

# Core pipeline only (single agent, no orchestration)
.venv/bin/python -m ascent.main

# Run tests (254 tests)
.venv/bin/pytest tests/ -v
```

---

## Architecture

```
ascent/                     # Core quant engine
  config/                   # Settings, APIKeys, UniverseConfig, BacktestConfig, types.py (AgentOutput)
  data/
    ingest/                 # Yahoo Finance, FRED, fundamentals, earnings, simulated (GBM fallback)
    normalize/              # Cleaning, schema validation
    store/                  # Parquet cache, point-in-time joins (as_of_join / as_of_merge)
    universe.py             # get_universe_on_date() — survivorship-bias-safe per-fold symbol filter
  features/                 # build_features, feature_defs, targets
  alpha/                    # trend, meanrev, statarb, ml_sleeve (CPCV), volatility, fundamental, earnings + stack
  portfolio/                # sector_constrained_weighted, _water_fill_cap (iterative ≤50 iters)
  backtest/                 # engine, cost model (Almgren-Chriss), reports
  research/                 # walk_forward_runner, walk_forward_lightweight, self_improve, shadow_promoter, cpcv
  regime/                   # HMM engine, model, features, decision, particle filter, emergency refit
  risk/                     # correlation_guard, VaR/CVaR
  reporting/                # market_memo, ic_brief_generator, blind_spot_detector, catalyst_scanner, debrief
  execution/                # eod_runner, alpaca_broker, order_engine, kill_switch, approval_server, cost_model, debate_gate
  monitoring/               # skill_tracker, forward_pnl_tracker, attribution, counterfactual_tracker, pre_rebalance_checklist
  llm/                      # client.py — centralized Anthropic API wrapper (Opus 4.6 + Haiku 4.5)
  main.py                   # Core pipeline entrypoint

agents/                     # 4 specialist agents (us_equities, macro, international, alternatives)
orchestrator/               # central_intelligence.py — skill-score capital allocation + cross-agent risk
debate/                     # debate_runner, 5 debate agents, judge, outcome_tracker
memory/                     # r2r_interface — R2R ingestion + BM25 fallback
simulation/                 # mirofish_interface — Monte Carlo scenario simulation

data_cache/                 # Parquet caches, ml_model_*.pkl, active_alpha_config.json, shadow_configs/
dashboard/                  # HTML dashboards, regime_signal.json, regime_labels.csv, agent_skill_scores.json
outputs/
  debate_log/               # verdict_YYYY-MM-DD.json per rebalance session
logs/                       # PnL logs per agent, slippage, self_improve, skill_scores, multi_agent_run

run_all_agents.py           # Single daily command (replaces direct main.py calls)
demo_app.py                 # Streamlit interactive demo (dark/gold UI)
```

---

## Daily Runtime Flow

**Command**: `python3 run_all_agents.py`

**Every day**:
1. Shadow promoter checks 30-day shadow configs → auto-promotes winners to `active_alpha_config.json`
2. All 4 specialist agents run in parallel via AgentScope, each calling `ascent/main.py`
3. Each agent emits an `AgentOutput` (target_weights, regime_signal, alpha_scores, skill_score)
4. Forward PnL cycle scores open positions; skill scores updated (63-day rolling OOS Sharpe)
5. Orchestrator blends allocations: skill-score weighting + conviction bonuses + correlation guard + EM/commodity cap + crisis veto

**Rebalance days** (every ~10 business days):
6. Pre-rebalance checklist runs
7. Debate gate decides whether debate is warranted (regime entropy > 0.70, top position > 12%, VaR 99th < -3.5%, or catalyst detected)
8. If debate runs: blind spot detection → catalyst scan → Monte Carlo sim → 5-agent debate → judge verdict
9. Verdict gates execution: `proceed` → normal, `reduce_size` → Haiku adjusts weights, `halt_and_review` → skip + log
10. Approved orders submitted to Alpaca; post-fill slippage tracked; IC brief generated; dashboard refreshed

**Sundays at 6 AM**: self-improve loop runs — 5 sleeve-weight variants tested on real multi-fold OOS, winners shadow-promoted

---

## Core Pipeline (`ascent/main.py`)

```
data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export
```

- Signal computed at date `t` close; trade executed at `t+1` open (1-day delay)
- Costs modeled via Almgren-Chriss: blocks > 10% ADV, warns > 5% ADV
- SPY 200MA overlay: multiply all weights × 0.70 when SPY < 200-day MA
- Walk-forward uses `get_universe_on_date(rebalance_date)` on every fold (no survivorship bias)

---

## Alpha Stack

11 sleeves blended into a composite score (cross-sectionally z-scored before blend). Weights are regime-adaptive and self-improving — the system reads `data_cache/active_alpha_config.json` on every run.

| Sleeve | Default Weight | Notes |
|--------|---------------|-------|
| Trend | 44% | Cross-sectional momentum; skip-last-month variant (`mom_252d - mom_21d`) |
| Stat-arb | 15% | Sector residuals; requires `profiles.parquet` |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds; 6 features; cached to `ml_model_{agent_id}.pkl` |
| Mean Reversion | 5% | Short-term reversal |
| Volatility | 5% | Long names with declining + stable vol; signal = -(vol_trend_10d / vol_of_vol_21d) |
| Fundamental | 5% | Gross profitability + accruals + asset growth + 52wk high; 45-day filing lag |
| Earnings | 5% | PEAD — cross-sectional z-score of earnings surprise; 1-bday announcement lag |
| Analyst | 5% | Analyst revision signal |
| Options Flow | 2% | Options activity signal |
| Insider | 2% | Insider transaction signal |
| Short Interest | 2% | Short squeeze / squeeze pressure signal |

Distressed name filter: zeroes alpha for names with `mom_252d < -0.65` (down > 65% YoY) after blending.

---

## Specialist Agents

| Agent | Universe | Strategy | Notes |
|-------|----------|----------|-------|
| US Equities | 901 symbols (S&P 500 + S&P 400) | Full alpha stack, max 1 per sector, 12–20 positions | Primary agent |
| Macro | TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM | Trend-only; regime-sized top_n | Crisis: top_n=3 / 40% |
| International | EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA | Max 2 per region; UUP > 50MA → 20% EM penalty | EM sensitivity |
| Alternatives | VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL | Trend 80% + low-vol 20%; max 35%, min 5%, top 4 | Kill switch at 12% |

---

## Orchestrator (`orchestrator/central_intelligence.py`)

1. **Base by regime**: `calm_bull` US60/mac15/intl15/alt10; `stressed` US45/mac25/intl10/alt20; `crisis` US30/mac30/intl5/alt35
2. **Skill blend**: per-agent; negative Sharpe → zero allocation; else 50% skill + 50% base
3. **Conviction bonus**: up to +15% when ≥2 agents share a name
4. **Correlation guard**: 63-day cross-agent correlation cap at 0.70 → halve smaller position
5. **Thesis coherence**: 12 factor buckets / 6 contradictory pairs → 40% reduction on contradictions
6. **EM + commodity cap**: hard 20% total after all blending
7. **Crisis veto**: `us_regime=crisis` → merged = 0.60×macro + 0.40×merged

---

## Regime System

- HMM with K=2–4 states (best K selected via walk-forward CV)
- Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`
- Hysteresis: enter threshold 0.55 / exit 0.35 / min dwell 3 days / entropy > 0.90 → uncertain
- Particle filter: 500 particles (SIR), reinitializes on batch refit
- Emergency refit triggers: SPY −3% + VIX > 30, 200MA cross, SPY/TLT correlation flip, break z-score > 3.5
- Features include: credit spread (HYG/LQD), yield curve slope (TLT/IEF), VIX, realized vol
- Scheduled refit every 5 days; exported to `dashboard/regime_signal.json`

---

## Debate Layer

Runs on high-uncertainty rebalance days only (gated by `execution/debate_gate.py`). Advisory — never writes to alpha, portfolio, or execution modules.

**Sequence**: score past verdicts → debrief → blind spot detection → catalyst scan → Monte Carlo sim → debate → judge

| Agent | Model | Role |
|-------|-------|------|
| Bull | claude-opus-4-6 | Strongest case for executing |
| Bear | claude-opus-4-6 | Case for reducing risk |
| Devil's Advocate | claude-opus-4-6 | Most dangerous assumption + Monte Carlo tail |
| Regime Specialist | claude-haiku-4-5-20251001 | Sizing playbook for current regime |
| Quant Sanity | Pure Python | Weight sum, max position, concentration, turnover |

Round 2 rebuttals: bull/bear/devil respond to each other before judge synthesizes.

**Verdict**: `proceed` | `reduce_size` (Haiku adjusts weights) | `halt_and_review` (persists to `execution/halt_state.json`)

---

## Walk-Forward OOS Performance

Evaluated on honest walk-forward out-of-sample (Jan 2020 – Apr 2026), using `get_universe_on_date()` per fold and fold-local regime fitting:

| Metric | Value |
|--------|-------|
| CAGR | ~12.35% |
| Sharpe | ~0.518 |
| Alpha vs SPY | ~+0.68% |

---

## Portfolio Construction

`sector_constrained_weighted()`:
- Coverage check: < 80% sector coverage → skip sector caps + warn
- Rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative redistribution, ≤50 iterations)
- Hard clamp + renorm. Post-condition: weights sum to 1.0 ± tolerance, no position > max_weight

Regime tightens `max_weight`: `crisis` → 0.08, `calm_bull` → 0.15

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays

---

## Execution & Risk Controls

- **Kill switch**: SOFT_WARN at 8% drawdown (log + proceed), HARD_STOP at 15% (abort); alternatives agent at 12%
- **Approval gate**: orders > 2% NAV → `execution/pending_approvals.json`, async 30-min wait
- **Cost model**: Almgren-Chriss; blocks > 10% ADV, warns > 5% ADV
- **Slippage tracking**: post-fill vs signal close, written to `logs/slippage_log.jsonl`
- **Broker**: Alpaca paper trading (live since April 1, 2026)

---

## Self-Improving Alpha

Weekly (Sunday 6 AM): generates 5 sleeve-weight variants, scores each via real multi-fold OOS (`run_lightweight_oos()`). Winners with edge > 0.05 Sharpe enter a 30-day shadow period. `shadow_promoter.py` auto-promotes survivors to `active_alpha_config.json`, with per-regime variants (`by_regime` section). The alpha stack reads live config on every run.

---

## LLM Clients

All LLM calls go through `ascent/llm/client.py` — lazy singleton, retry 3× with 2s/4s backoff.

| Constant | Model | Used For |
|----------|-------|----------|
| `DEFAULT_MODEL` | `claude-opus-4-6` | Debate agents (bull, bear, devil's advocate) |
| `HAIKU_MODEL` | `claude-haiku-4-5-20251001` | Weight adjustment, regime specialist, IC briefs |

---

## Data Sources

| Source | Data | Notes |
|--------|------|-------|
| Yahoo Finance (yfinance) | OHLCV prices | Primary; `prices_live` cache |
| FRED | Macro (rates, VIX, spreads) | `macro_live` cache |
| yfinance fundamentals | Earnings, analyst, quarterly financials | 45-day filing lag enforced |
| Simulated (GBM) | Fallback prices | Cache name: `prices_live_fallback_simulated` |

Cache provenance rule: never use `prices_live` for simulated data. If live fetch fails, label as `prices_live_fallback_simulated`.

---

## Integrity Constraints

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only; ML targets not leaked into feature windows
2. **No simulated data under live cache names** — data provenance preserved in cache name
3. **Max-weight hard cap** — iterative `_water_fill_cap()` with post-condition check
4. **Sector constraint fallback** — < 80% coverage → skip caps + warn, never collapse to single name
5. **Failed folds must be visible** — no silent zero-weight fallback; log fold date, stage, exception
6. **Debate is advisory only** — never writes to alpha, portfolio, or execution
7. **Approval gate for large trades** — any order > 2% NAV goes through `pending_approvals.json`

---

## Testing

```bash
.venv/bin/pytest tests/ -v           # All 254 tests
.venv/bin/pytest tests/ -k leakage   # Leakage / look-ahead tests
.venv/bin/pytest tests/ -k walkforward  # Walk-forward split tests
```

Test coverage: data integrity, leakage detection, walk-forward splits, alpha sleeves, portfolio constraints, regime system, execution safety, debate layer, orchestrator logic, self-improve loop.

---

## Environment

- Python 3.12.13 (Homebrew), venv at `.venv/`
- API keys via `APIKeys.from_env()` only — never hardcoded
- Mac Air M5 (dev), launchd `.plist` for scheduled jobs
- R2R semantic memory: built, BM25 fallback active (R2R API key not yet configured)
