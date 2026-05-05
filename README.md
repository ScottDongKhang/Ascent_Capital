# Ascent Capital

Modular systematic trading platform. Four specialist agents across US equities, macro, international, and alternatives — each running an independent alpha pipeline — feed a central orchestrator that applies skill-weighted capital allocation, cross-agent risk controls, and an LLM debate layer before submitting orders to Alpaca paper trading.

Live since April 1, 2026.

---

## Walk-Forward Out-of-Sample Record

Evaluated on honest expanding-window walk-forward OOS (January 2020 – April 2026). Every fold uses `get_universe_on_date(rebalance_date)` to exclude names outside their validity window at that date. Regime engine fitted on training slice only — no look-ahead.

| Metric | Value |
|--------|-------|
| CAGR | ~12.4% |
| Sharpe | ~0.52 |
| Alpha vs SPY | ~+0.68% annualized |
| Evaluation window | Jan 2020 – Apr 2026 |

These figures reflect the composite alpha stack and orchestrator logic as of the evaluation date. Ongoing self-improvement may cause the production configuration to diverge from the backtest configuration over time.

---

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY

# Full daily run
.venv/bin/python run_all_agents.py

# Core pipeline only (single agent, no orchestration)
.venv/bin/python -m ascent.main

# Tests (265 tests)
.venv/bin/pytest tests/ -v
```

---

## Repository Layout

```
ascent/
  config/         settings.py, types.py (AgentOutput dataclass), universe configs
  data/
    ingest/       Yahoo Finance, FRED, fundamentals (45-day filing lag), earnings, GBM fallback
    normalize/    Cleaning, schema validation
    store/        Parquet cache, point-in-time joins (as_of_join, as_of_merge)
    universe.py   get_universe_on_date() — per-fold survivorship filter
  features/       build_features, feature_defs, targets
  alpha/          trend, statarb, meanrev, ml_sleeve (CPCV), volatility, fundamental, earnings, stack
  portfolio/      sector_constrained_weighted, _water_fill_cap (iterative, ≤50 iterations)
  backtest/       engine, Almgren-Chriss cost model
  research/       walk_forward_runner, walk_forward_lightweight, self_improve, shadow_promoter, cpcv
  regime/         HMM engine (K=2–4), particle filter, emergency refit, posture mapping
  risk/           correlation_guard
  reporting/      blind_spot_detector, catalyst_scanner, debrief
  execution/      eod_runner, alpaca_broker, order_engine, kill_switch, approval_server, debate_gate
  monitoring/     skill_tracker, forward_pnl_tracker, attribution, counterfactual_tracker
  llm/            client.py — centralized Anthropic wrapper (Opus 4.6 default, Haiku 4.5 for light tasks)
  main.py         Core pipeline entrypoint

agents/           us_equities, macro, international, alternatives
orchestrator/     central_intelligence.py
debate/           debate_runner, agents (bull/bear/devil/regime/quant), judge, outcome_tracker
memory/           r2r_interface (R2R + BM25 fallback)
simulation/       scenario_simulator (parametric shock + Monte Carlo percentiles)

data_cache/       Parquet caches, ml_model_*.pkl, active_alpha_config.json, shadow_configs/
dashboard/        regime_signal.json, regime_labels.csv, agent_skill_scores.json
outputs/
  debate_log/     verdict_YYYY-MM-DD.json
  scenarios/      scenario output JSON per rebalance
logs/             PnL per agent, slippage, self_improve, skill_scores, multi_agent_run

run_all_agents.py   Single daily entrypoint
```

---

## Daily Runtime

**`python3 run_all_agents.py`**

Every day:
1. Shadow promoter checks 30-day shadow configs — auto-promotes winners to `active_alpha_config.json`
2. Four specialist agents run in parallel (AgentScope), each calling `ascent/main.py` for their universe
3. Each agent emits an `AgentOutput`: `target_weights`, `regime_signal`, `alpha_scores`, `skill_score`
4. Forward PnL cycle scores open positions; 63-day rolling OOS Sharpe updated per agent
5. Orchestrator blends allocations: skill-weighted base + conviction bonus + correlation guard + EM/commodity cap + crisis veto

On rebalance days (~every 10 business days):
6. Pre-rebalance checklist
7. Debate gate evaluates: regime entropy > 0.70, top position > 12%, VaR 99th < −3.5%, or catalyst detected
8. If warranted: blind spot detection → catalyst scan → Monte Carlo → 5-agent debate → judge verdict
9. Verdict gates execution: `proceed` → normal, `reduce_size` → Haiku adjusts weights, `halt_and_review` → skip + log
10. Orders to Alpaca; post-fill slippage tracked

Sundays 6 AM: self-improve loop — 5 sleeve-weight variants tested on real multi-fold OOS.

**Core pipeline** (`ascent/main.py`):

```
data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export
```

Signal at close `t`; trade at open `t+1`. SPY < 200-day MA → multiply all weights × 0.70.

---

## Alpha Stack

Seven sleeves blended into a composite cross-sectional score. All sleeves z-scored before blending. Weights are regime-adaptive and updated weekly by the self-improve loop via `data_cache/active_alpha_config.json`.

| Sleeve | Default Weight | Signal |
|--------|---------------|--------|
| Trend | 55% | Cross-sectional momentum; skip-last-month variant (`mom_252d − mom_21d`) |
| Stat-arb | 15% | Sector-residual mean reversion; requires `profiles.parquet` |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, 5-day purge + embargo; 6 features selected by IC/IR |
| Mean Reversion | 5% | Short-term reversal |
| Volatility | 5% | Long names with declining and stable vol: `−(vol_trend_10d / vol_of_vol_21d)` |
| Fundamental | 5% | Gross profitability, accruals, asset growth, 52-week high; 45-day filing lag |
| Earnings | 5% | PEAD — cross-sectional z-score of reported vs expected EPS; 1-bday announcement lag |

Distressed filter: zeroes alpha for `mom_252d < −0.65` after blending.

---

## Regime System

Hidden Markov Model with K=2–4 states; K selected via walk-forward cross-validation. State labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`.

- **Features**: VIX, SPY realized vol, credit spread (HYG/LQD), yield curve slope (TLT/IEF), SPY/TLT correlation
- **Hysteresis**: enter 0.55, exit 0.35, minimum dwell 3 days; entropy > 0.90 → uncertain override
- **Particle filter**: 500 particles (SIR), reinitializes on each batch refit
- **Emergency refit triggers**: SPY −3% + VIX > 30, 200MA cross, SPY/TLT correlation flip, break z-score > 3.5
- **Scheduled refit**: every 5 days
- **Downstream effects**: sleeve weights, `max_weight` tightening, orchestrator base allocation, debate context

---

## Orchestrator

`orchestrator/central_intelligence.py` applies in sequence:

1. **Regime base**: `calm_bull` US60/mac15/intl15/alt10; `stressed` US45/mac25/intl10/alt20; `crisis` US30/mac30/intl5/alt35
2. **Skill blend**: negative Sharpe → zero; else 50% skill-score weight + 50% regime base
3. **Conviction bonus**: up to +15% when ≥2 agents share a position with conviction > 0.3
4. **Correlation guard**: 63-day cross-agent correlation cap at 0.70 — halve the smaller position
5. **Thesis coherence**: 12 factor buckets, 6 contradictory pairs (e.g., long UUP and long GLD) → 40% reduction
6. **EM + commodity cap**: hard 20% aggregate after all blending
7. **Crisis veto**: `us_regime=crisis` → merged = 0.60 × macro + 0.40 × merged

---

## Specialist Agents

| Agent | Universe | Strategy |
|-------|----------|----------|
| US Equities | 901 symbols (S&P 500 + S&P 400) | Full 7-sleeve alpha stack; max 1 per sector; 12–20 positions |
| Macro | TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM | Trend-only; regime-sized: crisis top_n=3 / 40% |
| International | EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA | Max 2 per region; UUP > 50MA → 20% EM alpha penalty |
| Alternatives | VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL | Trend 80% + low-vol 20%; kill switch at 12%; top 4 |

---

## Debate Layer

Runs on high-uncertainty rebalance days only (gated by `execution/debate_gate.py`). Advisory — never writes to alpha, portfolio, or execution modules.

| Agent | Model | Role |
|-------|-------|------|
| Bull | claude-opus-4-6 | Strongest case for executing as proposed |
| Bear | claude-opus-4-6 | Case for reducing risk |
| Devil's Advocate | claude-opus-4-6 | Most dangerous assumption; Monte Carlo tail quantification |
| Regime Specialist | claude-haiku-4-5-20251001 | Sizing playbook for current regime |
| Quant Sanity | Pure Python | Weight sum, max position, concentration, turnover checks |

Round 2: bull/bear/devil respond to each other before judge synthesizes. Blind spot detector reads all prior verdicts, identifies systematic failure patterns, injects `blind_spot_context` into every session. Outcome tracker scores past verdicts against realized NAV.

**Verdict**: `proceed` | `reduce_size` (Haiku reweights) | `halt_and_review` (persists to `execution/halt_state.json`)

---

## Self-Improving Alpha

Weekly (Sunday 6 AM). Generates 5 sleeve-weight variants via bounded perturbation, scores each on real multi-fold expanding-window OOS via `walk_forward_lightweight.py`. Winners with edge > 0.05 Sharpe enter a 30-day shadow period. `shadow_promoter.py` auto-promotes survivors to `active_alpha_config.json` with per-regime variants. The alpha stack reads this file live on every run.

---

## Execution and Risk Controls

- **Kill switch**: SOFT_WARN at 8% drawdown (log + proceed); HARD_STOP at 15% (abort); alternatives agent at 12%
- **Approval gate**: orders > 2% NAV write to `execution/pending_approvals.json`, async 30-minute wait
- **Cost model**: Almgren-Chriss; blocks > 10% ADV, warns > 5% ADV
- **Slippage tracking**: post-fill vs signal close, written to `logs/slippage_log.jsonl`
- **Broker**: Alpaca paper trading

---

## Portfolio Construction

`sector_constrained_weighted()` in `ascent/portfolio/optimizer.py`:

- Coverage check: < 80% sector coverage → skip sector caps, log warning
- Rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative redistribution, ≤50 iterations)
- Hard clamp + renormalize. Post-condition: weights sum to 1.0 ± tolerance; no position > `max_weight`
- Regime tightens `max_weight`: `crisis` → 0.08, `calm_bull` → 0.15

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays

---

## Data Integrity

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only; ML targets not leaked into feature windows
2. **No simulated data under live cache names** — fallback labeled `prices_live_fallback_simulated`
3. **Max-weight hard cap** — iterative `_water_fill_cap()` with post-condition assertion
4. **Sector constraint fallback** — coverage < 80% → skip caps + warn; never collapses to single name
5. **Failed folds are visible** — no silent zero-weight fallback; log includes fold date, stage, exception type
6. **Debate is advisory** — no writes to alpha, portfolio, or execution from debate agents
7. **Approval gate** — orders > 2% NAV require explicit approval before Alpaca submission

---

## Testing

```bash
.venv/bin/pytest tests/ -v                   # 265 tests
.venv/bin/pytest tests/ -k leakage           # Look-ahead / leakage tests
.venv/bin/pytest tests/ -k walkforward       # Walk-forward split integrity
.venv/bin/pytest tests/ -k regime            # Regime system
```

Coverage: data integrity, leakage detection, walk-forward splits, alpha sleeves, portfolio constraints, regime system, execution safety, debate layer, orchestrator logic, self-improve loop, hedge overlay.

---

## Environment

- Python 3.12 (Homebrew), venv at `.venv/`
- API keys loaded via `APIKeys.from_env()` only
- Scheduled via launchd (macOS): daily run at 1:45 PM, self-improve at 6:00 AM Sunday
- Semantic memory: R2R interface built; BM25 fallback active pending R2R API configuration

---

## Planned (Not Yet Built)

- **AI-native Tier 1**: LLM-guided fundamental alpha re-ranking; slippage IC feedback loop; regime-aware debate personas
- **AI-native Tier 2**: FinMem-style reflection agent; LLM hypothesis generation for self-improve; tool-capable debate agents
- **AI-native Tier 3**: Autonomous factor discovery — Opus proposes Python factor code, AST-validated, CPCV-evaluated, human-reviewed before promotion
- **Analyst revision signal**: `yfinance` recommendations as short-term catalyst (backlog)
- **Earnings surprise beta neutralization**: reduce momentum overlap in PEAD sleeve (backlog)
