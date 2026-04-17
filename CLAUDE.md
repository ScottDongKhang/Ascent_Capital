# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio construction → walk-forward evaluation → regime modeling → 4 specialist agents → orchestration → LLM debate → execution via Alpaca paper trading.

---

## Repository layout

```
ascent/
  config/       settings.py (Config, APIKeys, UniverseConfig, BacktestConfig), types.py (AgentOutput)
  data/         ingest (yahoo, fred, simulated), normalize, store (parquet, point_in_time), universe
  features/     build_features, feature_defs, targets
  alpha/        trend, meanrev, statarb, ml_sleeve, stack
  portfolio/    optimizer — sector_constrained_weighted, _water_fill_cap
  backtest/     engine, costs
  research/     walk_forward_runner, cpcv, self_improve
  regime/       engine, model, features, decision, integration, posture, breaks, particle_filter, types
  risk/         correlation_guard
  reporting/    market_memo, ic_brief_generator, blind_spot_detector, debrief, regime_narrative, catalyst_scanner
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch, run_log, slippage_tracker, approval_server, cost_model
  monitoring/   skill_tracker, forward_pnl_tracker, pre_rebalance_checklist, exit_alerts
  llm/          client.py — centralized Anthropic API wrapper

agents/         us_equities, macro, international, alternatives
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker
memory/         r2r_interface (R2R HTTP + BM25 fallback)
simulation/     mirofish_interface

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl, active_alpha_config.json, shadow_configs/
dashboard/      HTML dashboards, regime_signal.json, regime_labels.csv, agent_skill_scores.json
outputs/
  debate_log/   verdict_YYYY-MM-DD.json, agent_credibility.json
logs/           eod_log, slippage_log, self_improve_log, skill_scores_log, multi_agent_run, post_debate_portfolio,
                snapshots/{agent_id}_weights_YYYY-MM-DD.json

ascent/main.py        core pipeline entrypoint
run_all_agents.py     single daily command — branches on rebalance day
demo_app.py           Streamlit interactive demo (Tony Ngo)
```

---

## Core runtime flow

**Command**: `python3 run_all_agents.py`

**Non-rebalance day**: agents (parallel) → forward PnL → skill scores → orchestrator → write `merged_weights.json` → log. Stop.

**Rebalance day**: same + pre-rebalance checklist → debate → verdict gates execution → Alpaca orders → slippage tracking.

`ascent/main.py` pipeline: data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export.

---

## Alpha stack

| Sleeve | Weight | Notes |
|--------|--------|-------|
| Trend | 65% | Cross-sectional momentum |
| Stat-arb | 15% | Sector residuals; needs profiles.parquet |
| Mean reversion | 5% | Short-term reversal |
| ML (XGBoost) | 10% | CPCV-validated; cached to `ml_model_{agent_id}.pkl` |
| Volatility (vol-regime) | 5% | Signal = -(vol_trend_10d) / (vol_of_vol_21d); long names with declining + stable vol |

Regime adjusts sleeve weights via `integration.py:regime_adjust_sleeve_weights()`. All sleeves cross-sectionally z-scored before blending. ML sleeve: CPCV C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays — disabled if <10 folds converge or p5 IC Sharpe < 0.

New features added to `feature_defs.py`: `vol_of_vol_21d` (rolling std of 21d realized vol — low = stable regime), `vol_trend_10d` (10-day change in 21d realized vol — negative = declining). Combined into vol-regime sleeve in `stack.py`. Orthogonal to momentum; decays slowly because most funds screen on level of vol, not its trend+stability.

---

## Portfolio construction

`sector_constrained_weighted()`: coverage check (< 80% → skip sector caps + warn) → rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative, ≤50 iterations) → hard clamp + renorm. Post-condition: sum=1.0±tol, no position > max_weight.

Regime tightens max_weight: crisis → 0.08, calm_bull → 0.15. SPY 200MA overlay: SPY < 200MA → multiply weights × 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays.

---

## Agents

**US Equities**: 135 stocks, full alpha stack, max 1 per sector, 12–20 positions.

**Macro**: TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM. Trend-only. Regime-sized: crisis top_n=3/40%, stressed top_n=4/35%, else top_n=5/30%. Cache: `prices_macro.parquet`.

**International**: EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA. Max 2 per region. UUP > 50MA → 20% alpha penalty on EM names.

**Alternatives**: VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL. Trend 80% + low-vol 20%. Kill switch at 12%. Max 35%, min 5%, top 4.

---

## Orchestrator (`orchestrator/central_intelligence.py`)

1. **Base by regime**: calm_bull US60/mac15/intl15/alt10; stressed US45/mac25/intl10/alt20; crisis US30/mac30/intl5/alt35.
2. **Skill blend**: per-agent independently; negative Sharpe → zero; else 50% skill + 50% base.
3. **Conviction bonus**: up to +15% when ≥2 agents share a name (conv > 0.3).
4. **Correlation guard**: 63-day cross-agent cap at 0.70 → halve smaller.
5. **Thesis coherence**: symbol-level contradictions (UUP↔PDBC/GLD, VIXY↔SVXY) + 12 factor buckets / 6 contradictory pairs → 40% reduction.
6. **Crisis veto**: us_regime=crisis → merged = 0.60×macro + 0.40×merged.

---

## Regime system

HMM K=2–4 (best via walk-forward CV). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3d / entropy > 0.90 → uncertain. Particle filter: 500 particles SIR, reinitializes on batch refit. Emergency refit triggers: SPY −3%+VIX>30, 200MA cross, SPY/TLT corr flip, break z-score > 3.5. Refit every 5 days.

---

## Debate layer

Rebalance days only. Advisory — never writes to alpha/portfolio/execution.

**Sequence**: score past verdicts → debrief → blind spot detection → catalyst scan → Monte Carlo sim → run agents → judge verdict.

| Agent | Model | Role |
|-------|-------|------|
| Bull | claude-opus-4-6 | Strongest case for executing |
| Bear | claude-opus-4-6 | Case for reducing risk |
| Devil's Advocate | claude-opus-4-6 | Most dangerous assumption + Monte Carlo tail |
| Regime Specialist | claude-haiku-4-5-20251001 | Sizing playbook for current regime |
| Quant Sanity | Pure Python | Weight sum, max position, concentration, turnover |

Round 2 rebuttals: bull/bear/devil respond to each other before judge synthesizes.

Verdict: `proceed` | `reduce_size` (Haiku adjusts weights) | `halt_and_review` (persists to `execution/halt_state.json`).

---

## Self-improve loop

Weekly (Sunday 6AM). **Status: Phase B heuristic** — generates 5 sleeve-weight variants, scores by `base_sharpe(0.518) + diversity_bonus + noise`. Shadow promotion if edge > 0.10 Sharpe, 30-day monitoring. **Phase D TODO**: replace heuristic with real `walk_forward_pipeline()` call.

---

## Execution

Kill switch: SOFT_WARN 8% (log+proceed), HARD_STOP 15% (abort). Alternatives: 12%. State in `logs/kill_switch_state.json`. Large trades (> 2% NAV) → `execution/pending_approvals.json`, async wait via `threading.Event`, 30-min timeout. Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV. Post-fill: slippage logged to `logs/slippage_log.jsonl`.

Config: always `get_config()`, never `Config()` directly.

---

## LLM clients

`ascent/llm/client.py`: `DEFAULT_MODEL = "claude-opus-4-6"`, `HAIKU_MODEL = "claude-haiku-4-5-20251001"`. Lazy singleton. Retry 3× with 2s/4s backoff. All files import `HAIKU_MODEL` from here — never redefine locally.

---

## Data / caching

Cache names: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live/simulated`, `profiles` (sector metadata). Never hide data provenance in cache name. Point-in-time joins via `as_of_join()` / `as_of_merge()`.

---

## Demo app (`demo_app.py`)

Streamlit interactive demo for Tony Ngo. Dark/gold aesthetic. Sidebar: regime, VIX, SPY momentum, portfolio preset, round 2 toggle. Main: portfolio snapshot, debate transcript (5 agents + rebuttals), judge verdict.

**Modes**: Live LLM if `ANTHROPIC_API_KEY` available (via `st.secrets` on Streamlit Cloud, `.env` locally); else demo mode with scenario-aware pre-written arguments.

**Deploy**: push repo to GitHub → share.streamlit.io → set `ANTHROPIC_API_KEY` in Streamlit Cloud secrets dashboard → share URL. Key never in repo (`.env` and `secrets.toml` are gitignored).

---

## Integrity constraints

1. No look-ahead bias — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only.
2. No simulated data under live cache names.
3. Max-weight hard cap via `_water_fill_cap()` with post-condition check.
4. Sector constraint: < 80% coverage → skip caps + warn, never collapse to single name.
5. `walk_forward_runner.py` not a production entrypoint — retained for self_improve Phase D only.
6. Debate is advisory only.
7. Approval gate for orders > 2% NAV.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Environment

Python 3.12.13 Homebrew, venv at `.venv/`. Use `.venv/bin/python`. API keys via `APIKeys.from_env()` only. Mac Air M5, no JAMF restrictions.

---

## Current portfolio (as of April 2026)

Holdings (post-rebalance Apr 15): EWY 10.8%, PDBC 9.3%, CASY/CAT/EQIX/MPWR/TRGP 6% each, HYG/BIL/DBB/EWZ/EWT ~4.5–4.8%, NEM 4.4%, LQD/MRK/IFRA ~3–3.5%, VNQ/PAVE/AMZN/GOOGL ~2.3–2.9%, CB 1.9%, EWC 0.6%.
Next rebalance: ~April 29, 2026. Live since April 1, 2026.

---

## What is not built yet

- **Plans B2–D4**: enforce reduce_size, regime staleness fix, verdict outcome scoring, live Sharpe in self-improve, quant_context for debate agents, extended thinking for judge, prompt caching — all specced in `docs/superpowers/plans/`, partially implemented (see session log)
- **Phase 4 hedge overlay**: blocked until ~May 13, 2026 (30 days live)
- **R2R semantic memory**: built but `R2R_API_KEY` not configured; BM25 fallback active
- **Live dashboard UI**: data files generated but no live render
- **Debate trigger condition**: debate should only fire on high-uncertainty days (catalyst present, regime entropy >0.70, position >12%, VaR 99th < -3.5%) — not on every rebalance

---

## Session log

### 2026-04-09 to 2026-04-13 (summary)
- Initial CLAUDE.md, env setup, 6 bug fixes, A4 survivorship bias hardening
- Phase 1 (skill staleness, sector error, persistent halt), Phase 2 (async approval gate, Almgren-Chriss), Phase 3 (CPCV ML sleeve, regime particle filter + emergency refit)
- 3 AI agent features: catalyst scanner, multi-turn debate, memory-augmented debate (93 tests passing)
- Universe: removed 15 delisted, added 15 new (135 total)

### 2026-04-14 (Tony Ngo demo — plan only)
- Designed `demo_app.py` architecture; nothing built

### 2026-04-15 (first live rebalance)
- Fixed 4 pre-rebalance bugs (yf.download race condition, hardcoded NAV, regime date key, ML targets)
- Rebalance ran: verdict REDUCE_SIZE 0.88 confidence, 27 orders to Alpaca
- Fixed full-liquidation 403 errors: added `close_position()` (DELETE /v2/positions/{symbol}) to `alpaca_broker.py`
- Built `demo_app.py`: dark/gold Streamlit app with live LLM debate, round 2 rebuttals, scenario presets
- Files: `alpaca_broker.py`, `eod_runner.py`, `pre_rebalance_checklist.py`, `main.py`, agent files, `demo_app.py`

### 2026-04-15 (demo polish + deployment prep)
- Fixed How It Works tab (Streamlit strips `<style>` blocks — rewrote with inline styles)
- Updated portfolio preset to actual Apr 15 post-debate holdings (22 positions)
- Fixed live LLM mode (missing `load_dotenv()`); removed password gate (trust-based sharing)
- API key security: `st.secrets` on Streamlit Cloud, `.env` locally, both gitignored
- Added `.streamlit/config.toml`, `secrets.toml.template`, updated `requirements.txt`
- Files: `demo_app.py`, `.streamlit/config.toml`, `.streamlit/secrets.toml.template`, `.gitignore`, `requirements.txt`
- Open: push to GitHub → deploy share.streamlit.io → send Tony the link; Phase 4 hedge overlay (~May 13); self-improve Phase D

### 2026-04-16 (system upgrade planning + partial execution)
- Diagnosed why portfolio lags SPY: 37% EM+commodity, stale regime (March 19), self-improve using noise heuristic, verdicts never scored (wrong NAV source), debate agents arguing without quant data
- Discussed quant+AI balance: debate should be a circuit breaker on edge cases only, not a daily veto of the quant model — added debate trigger condition to backlog
- Specced 4-plan upgrade: A (monitoring), B (portfolio hardening), C (self-learning), D (LLM enhancement) — saved to `docs/superpowers/plans/`
- Committed 64 untracked source files that were never in git (Phase 1–3 work, agent files, regime engine, llm client, etc.) — test suite now 110 passing clean baseline
- Created worktree `feature/plans-a-d` for isolated implementation
- **A1 ✅**: SPY benchmark in PnL log — `_log_holdings` writes `spy_return` + `alpha_vs_spy`; `run_forward_pnl_cycle` batch-fetches SPY; wired `_log_holdings` into all `main()` exit paths
- **A2 ✅**: US equities PnL routed to `logs/us_equities_pnl.jsonl` (was `eod_log.jsonl`); `skill_tracker.py` reads from `PNL_LOGS` (single source of truth)
- **A3 ✅**: `ascent/monitoring/attribution.py` — daily position-level P&L attribution, writes `logs/attribution_log.jsonl`, wired into `_log_holdings`
- **B1 ✅**: `_cap_em_commodity()` in `orchestrator/central_intelligence.py` — hard 20% cap on EM+commodity+gold after all blending
- B2–D4: specced, not yet implemented
- Files: `ascent/monitoring/attribution.py` (new), `ascent/monitoring/forward_pnl_tracker.py`, `ascent/monitoring/skill_tracker.py`, `orchestrator/central_intelligence.py`, `run_all_agents.py`, `tests/test_plan_a.py` (new), `tests/test_plan_b.py` (new)
- Test count: 117 passing on `feature/plans-a-d` branch

### 2026-04-16 (vol-regime alpha sleeve)
- Added `vol_of_vol()` and `vol_trend()` to `ascent/features/feature_defs.py`; registered as `vol_of_vol_21d` and `vol_trend_10d` in `build_all_features()`
- Rewired volatility sleeve in `ascent/alpha/stack.py`: signal = -(vol_trend_10d) / (vol_of_vol_21d); long names with declining AND stable vol — orthogonal to momentum
- Enabled volatility sleeve at 5% weight; trend reduced 70% → 65%
- Files: `ascent/features/feature_defs.py`, `ascent/alpha/stack.py`, `CLAUDE.md`
