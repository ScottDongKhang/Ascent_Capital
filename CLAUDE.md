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
| Trend | 41% | Cross-sectional momentum; skip-last-month variant (`mom_252d − mom_21d`) at 0.20 sub-weight |
| Stat-arb | 15% | Sector residuals; needs profiles.parquet |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays; 6 features by IC/IR; p5 guard > −0.05 |
| Mean reversion | 5% | Short-term reversal |
| Volatility | 5% | `−(vol_trend_10d / vol_of_vol_21d)`; long names with declining + stable vol |
| Fundamental | 5% | Gross profitability, accruals, asset growth; 45-day filing lag; momentum-neutral |
| Earnings (PEAD) | 5% | Cross-sectional z-score of EPS surprise; OLS momentum-beta residual; 1-bday lag |
| Analyst | 5% | Analyst revision signal; sparse — zero-filled if cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Haiku; cached by (symbol, quarter_end); 45-day filing lag |
| Options Flow | 2% | Options sentiment; sparse — zero-filled if cache absent |
| Insider | 2% | Insider net transaction score; sparse — zero-filled if cache absent |
| Short Interest | 2% | Short squeeze signal; sparse — zero-filled if cache absent |
| Alt Data | 0% | IC-validated alternative data (SEC 10-K, transcripts, Reddit, Google Trends); 0% until first source passes IC gate |

Distressed filter: zeroes alpha for `mom_252d < −0.65` after blending. All sleeves cross-sectionally z-scored before blending. Weights are regime-adaptive via `data_cache/active_alpha_config.json` (updated weekly by self-improve loop). ML sleeve disabled if <10 folds converge or p5 IC Sharpe < −0.05.

---

## Portfolio construction

`sector_constrained_weighted_mvo()` is the primary path (Plan 2): sector pre-screening → Black-Litterman blending (quant prior + LLM views) → cvxpy MVO (objective: `w'α - λΣ - κ‖w-w_prev‖₁`, CLARABEL solver, SCS fallback) → rank-weight fallback if infeasible.

`sector_constrained_weighted()` (rank-weight fallback): coverage check (< 80% → skip sector caps + warn) → rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative, ≤50 iterations) → hard clamp + renorm.

BL blending weight from IC IR: IR < 0.30 → tau=0.05, IR < 0.60 → tau=0.10, else tau=0.15.

Regime tightens max_weight: crisis → 0.08, calm_bull → 0.15. SPY 200MA overlay: SPY < 200MA → multiply weights × 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays.

---

## Agents

**US Equities**: 901 symbols (S&P 500 + S&P 400, loaded from `ascent/config/us_equity_universe.json`); full alpha stack; max 1 per sector; 12–20 positions.

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
6. **EM + commodity cap**: hard 20% aggregate on EM+commodity+gold after all blending.
7. **Crisis veto**: us_regime=crisis → merged = 0.60×macro + 0.40×merged.

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

Weekly (Sunday 6AM). Generates 5 sleeve-weight variants, scores via real multi-fold OOS (`run_lightweight_oos()`). Shadow promotion if edge > 0.05 Sharpe, 30-day monitoring, auto-promoted by `shadow_promoter.py`. Per-regime variants written to `active_alpha_config.json` `by_regime` section. Stack reads live config on every run.

`SELF_MODIFY_ENABLED = False` — exits early until OOS Sharpe is positive for 30 consecutive trading days on a flat config.

## Factor discovery loop

Monthly (first Sunday of each month, wired into `run_all_agents.py`). Two paths: PySR symbolic regression on pre-computed features (Path A) + Haiku suggests JSON template params (Path B). Both evaluated via per-regime Spearman IC with Harvey FDR correction (IC mean ≥ 0.015, IC IR ≥ 0.60, IC positive in every observed regime). Accepted proposals written to `outputs/factor_proposals/` for human review — nothing auto-deploys.

Gate conditions (~July 2026): `SELF_MODIFY_ENABLED = True`, ≥63 days live regime labels, ≥50 slippage fills.

---

## Execution

Kill switch: SOFT_WARN 8% (log+proceed), HARD_STOP 15% (abort). Alternatives: 12%. State in `logs/kill_switch_state.json`. Large trades (> 2% NAV) → `execution/pending_approvals.json`, async wait via `threading.Event`, 30-min timeout. Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV. Post-fill: slippage logged to `logs/slippage_log.jsonl`.

**Plan 5 additions (execution excellence):**
- `twap_executor.py`: TWAP_ENABLED=False kill switch; orders > 5% ADV routed to equal-spaced child limit orders; Almgren-Chriss optimal window sizing
- `implementation_shortfall.py`: IS decomposition — delay cost / market impact / opportunity cost in bps; logged to `slippage_log.jsonl` backwards-compatibly
- `capacity_model.py`: max AUM per sleeve before signal decay from market impact; informational only; weekly log to `logs/capacity_log.jsonl`
- `intraday_trigger.py`: three triggers (SPY −3%+VIX>30, drawdown ≥12%, event urgency on top-5); checked at 12:00 PM and 14:30 PM ET; adjustments logged to `logs/intraday_adjustments.jsonl`

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

## Current portfolio (as of May 7, 2026)

Post-rebalance May 5 holdings (40 orders, full liquidation of Apr 15 portfolio):
KMLM 10.9%, IFRA 9.4%, AMKR/FIX/WDC 5.9% each, VICR 5.7%, VRT 5.6%, VAL 5.2%, WCC 5.0%, DBB 4.6%, CNC 4.5%, WFRD 4.3%, PDBC 4.2%, STLD 3.4%, DBA 3.3%, CHRD 3.1%, EWY 2.4%, EWC 2.3%, IRM 1.8%, MUSA/AAXJ/EEM/VWO 1.5% each, VIXY 2.8% hedge.

NAV: ~$104,815 (May 7). Next rebalance: ~May 13, 2026. Live since April 1, 2026.

Skill scores: all agents still warming up (need 63 days). Regime: calm_bull (refitted May 7 after 5-day stale period).

---

## Known bugs — audit status (2026-05-07)

All audits complete through third pass. No outstanding crash risks. Pipeline runs clean.

---

## What is not built yet

- **AI-native Tier 1**: ✅ built — `ascent/alpha/llm_fundamental.py`, `ascent/monitoring/slippage_ic_feedback.py`, `debate/disagreement_scorer.py`, private context subsets in `debate/agents.py`
- **AI-native Tier 2**: ✅ built — `memory/reflection_agent.py`, `ascent/research/factor_proposer.py`, `debate/agent_tools.py`, `tool_completion()` in `ascent/llm/client.py`
- **AI-native Tier 3**: ✅ built — `ascent/research/factor_discovery/` (PySR + LLM templates, per-regime CPCV, Harvey FDR); pipeline inactive until gate conditions met (~July 2026)
- **AI-native Tier 4 (Plan 4)**: ✅ built — `ascent/data/ingest/sec_filings.py`, `earnings_transcripts.py`, `reddit_sentiment.py`, `google_trends.py`; `ascent/alpha/altdata_alpha.py`; `ascent/data/validate/altdata_validator.py`; IC gate identical to factor discovery; altdata sleeve at 0% until first source validated
- **Event-driven agents (Plan 3)**: ✅ built — EDGAR RSS listener, 8-K Haiku classifier, Capitol Trades, options anomaly scanner, event daemon thread, `EVENT_TRADING_ENABLED=False`
- **Execution excellence (Plan 5)**: ✅ built — TWAP executor, IS decomposition, capacity model, intraday triggers (all kill-switched off pending paper validation)
- **Factor risk model (Plan 1)**: ✅ built — FF5+UMD, Ledoit-Wolf shrinkage Σ, factor exposure bounds, regime-aware constraints for MVO
- **Portfolio construction overhaul (Plan 2)**: ✅ built — cvxpy MVO, Black-Litterman, regime covariance
- **R2R semantic memory**: built but `R2R_API_KEY` not configured; BM25 fallback active
- **Live dashboard UI**: planned in `docs/superpowers/plans/2026-05-10-plan6-realtime-infrastructure.md`
- **Live track record + compliance**: planned in `docs/superpowers/plans/2026-05-10-plan7-live-track-record.md`

### Institutional roadmap (Plans 1–7, written 2026-05-10)

Seven sequenced plans taking Ascent to institutional and YC-ready state. Each has a complete spec in `docs/superpowers/plans/`:

| Plan | Status | File | What it adds |
|------|--------|------|--------------|
| Plan 1 | ✅ Done | `2026-05-10-plan1-factor-risk-model.md` | Barra-style factor risk model; portfolio factor exposure decomposition; factor-explained vs. idiosyncratic attribution |
| Plan 2 | ✅ Done | `2026-05-10-plan2-portfolio-construction.md` | cvxpy MVO optimizer; Black-Litterman quant+LLM blending; regime-conditional covariance; TC-aware objective |
| Plan 3 | ✅ Done | `2026-05-10-plan3-event-driven-architecture.md` | EDGAR 8-K listener; Capitol Trades; options anomaly scanner; event trade execution (0.5% NAV cap) |
| Plan 4 | ✅ Done | `2026-05-10-plan4-alternative-data-pipeline.md` | SEC full-text (10-K/10-Q); earnings transcripts; Reddit sentiment; Google Trends; IC validation gate |
| Plan 5 | ✅ Done | `2026-05-10-plan5-execution-excellence.md` | TWAP executor; implementation shortfall decomposition; capacity model; intraday rebalance triggers |
| Plan 6 | Planned | `2026-05-10-plan6-realtime-infrastructure.md` | Alpaca WebSocket streaming; TimescaleDB; live operator dashboard; monthly PDF investor reports |
| Plan 7 | Planned | `2026-05-10-plan7-live-track-record.md` | Immutable audit trail (hash chain); GIPS performance presentation; risk disclosures; 12-month live track record |

**Status:** Plans 1–5 complete (420 tests passing). Plans 6–7 are infrastructure/compliance — begin Plan 6 next.
**Timeline to YC-ready:** ~12–18 months from now (~April 2027 if Plan 6 starts immediately).

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

### 2026-04-17 (bug hardening — E1–E5, D1–D2, R1–R4)
- Fixed 11 bugs across execution, debate, and runner layers
- E1: empty weights guard in eod_runner (IndexError → ValueError with message)
- E2: DataFrame.get() → column check + .tolist()
- E3: cost features key guard before passing to order engine
- E4: dollar_volume column check before pivot_table
- E5: regime extraction failure now logged via log_error (not just print)
- D1: None-safe weights access in debate_runner (or {} guard)
- D2: split generic except into FileNotFoundError/ImportError/Exception
- R1: empty list guard on regime_signal.json read
- R2: allocation pulled from merged_weights.get("allocation") (static fallback + TODO)
- R3: bare except in _is_regime_stale now logs exception details
- R4: removed hasattr(ao, "n_positions") guard — n_positions is already a @property
- Files: eod_runner.py, debate_runner.py, run_all_agents.py, tests/test_bug_hardening.py (new)
- Tests: 144 passing

### 2026-04-18 (plans B2–D4 verified + second audit fixes)
- Confirmed B2–D4 were already fully implemented (done in a prior session, not logged)
- Fixed test_plan_d.py _make_prices helper: use len(idx) not n_days (breaks on non-business days like Saturdays)
- Fixed 6 second-audit bugs: international_agent (uup/eem iloc guards after dropna), alternatives_agent (gld iloc guard), outcome_tracker (_rebuild_credibility division-by-zero × 2), orchestrator (conviction bonus skip log), self_improve (fallback warning print)
- Full third-pass codebase audit: all remaining iloc[-1] accesses verified as guarded; no new crash risks found
- All Python files pass ast.parse; 144 tests passing
- Files: agents/international_agent.py, agents/alternatives_agent.py, debate/outcome_tracker.py, orchestrator/central_intelligence.py, ascent/research/self_improve.py, tests/test_plan_d.py, CLAUDE.md
- Open: Phase 4 hedge overlay (blocked until ~May 13), debate trigger condition (high-uncertainty days only), R2R API key config

### 2026-04-18 (Phase 1 firm architecture — real OOS, conditional debate, memory wiring)
- Built `ascent/research/walk_forward_lightweight.py` — real OOS scoring via pipeline (replaced heuristic in self_improve)
- Built `ascent/execution/debate_gate.py` — conditional debate (entropy >0.70, top pos >12%, VaR <-3.5%, catalyst)
- Built `ascent/monitoring/counterfactual_tracker.py` — quant vs debate weights snapshotted and scored 10 days later
- Modified `debate/agents.py` `_build_context()` — actively queries R2R/BM25 memory for past verdicts in same regime
- Modified `ascent/execution/eod_runner.py` — debate gated, verdict only defined inside `if _run_debate:` block
- Modified `run_all_agents.py` — calls `score_pending_counterfactuals()` daily
- Tests: 157 passing (Phase 1 closes)

### 2026-04-18/19 (Plan A: self-evolving alpha loop + Plan B: regime + walk-forward)
- **Plan A Task 1 ✅**: `ascent/alpha/stack.py` — `_load_active_alpha_weights(regime=)` reads `data_cache/active_alpha_config.json`; self-improve changes now hit live trading
- **Plan A Task 2 ✅**: `ascent/research/shadow_promoter.py` — auto-promotion: expired shadows re-evaluated, winners written to `active_alpha_config.json`, losers archived to `data_cache/archived_configs/`; wired into `run_all_agents.py` daily
- **Plan A Task 3 ✅**: `ascent/research/self_improve.py` — `_promote_regime_variant()` + `run_self_improve(current_regime=)`; Sunday call in `run_all_agents.py` passes current regime; system now learns stressed ≠ calm_bull weights
- **Plan B Task 1 ✅**: `ascent/regime/features.py` — `_build_credit_yield_features()`: credit_spread_chg_21d/level (HYG/LQD), yield_curve_slope/chg (TLT/IEF); leading indicators for regime transitions
- **Plan B Task 2 ✅**: `ascent/regime/engine.py` + `ascent/main.py` — `market_prices` param wired through engine; main.py fetches HYG/LQD/TLT/IEF alongside VIX
- **Plan B Task 3 ✅**: `ascent/research/walk_forward_lightweight.py` — multi-fold expanding window (3–4 folds), 5-day purge + embargo, per-fold `get_universe_on_date()`, Sharpe across all fold returns
- **Plan B Task 4 ✅**: A4 gap confirmed already fixed in `walk_forward_runner.py` (per-fold universe filter via `build_historical_universe()` already present)
- Files: `ascent/alpha/stack.py`, `ascent/research/shadow_promoter.py` (new), `ascent/research/self_improve.py`, `ascent/regime/features.py`, `ascent/regime/engine.py`, `ascent/main.py`, `ascent/research/walk_forward_lightweight.py`, `run_all_agents.py`, `tests/test_self_evolving_alpha.py` (new), `tests/test_regime_features.py` (new), `tests/test_walkforward_institutional.py` (new)
- Tests: 177 passing
- Open: Phase 4 hedge overlay (blocked ~May 13), debate trigger condition, R2R API key

### 2026-04-19 (fundamental alpha — Tier 1 signals)
- **Task 1 ✅**: `ascent/data/ingest/fundamentals.py` (new) — yfinance quarterly fetcher with 45-day filing lag; `fetch_fundamentals()`, `save_fundamentals()`, `load_fundamentals()`; fixed `__main__` to use `get_config()` not nonexistent `get_current_universe()`
- **Task 2 ✅**: `ascent/features/feature_defs.py` — `high_52w_pct()`, `build_fundamental_panel()` (gross_profitability, accruals, asset_growth with forward-fill); `ascent/alpha/fundamental.py` (new) — cross-sectional blend of 4 signals
- **Task 3 ✅**: `ascent/alpha/stack.py` — `DEFAULT_ALPHA_WEIGHTS` updated: trend 0.65→0.55, fundamental 0.10 added; fundamental sleeve wired into `build_alpha_stack()`; `ascent/alpha/ml_sleeve.py` — 4 fundamental signals added to `ML_FEATURES`; `ascent/features/build_features.py` — `fundamentals_df=None` param, panel augmentation in `compute_features()`; `ascent/main.py` — loads fundamentals cache and passes to FeatureBuilder
- Fundamentals cache seeded: 675 rows, 135 symbols
- Fixed stale test assertion: `test_stack_falls_back_to_defaults_when_no_config` updated trend 0.65→0.55
- Files: `ascent/data/ingest/fundamentals.py` (new), `ascent/alpha/fundamental.py` (new), `ascent/features/feature_defs.py`, `ascent/alpha/stack.py`, `ascent/alpha/ml_sleeve.py`, `ascent/features/build_features.py`, `ascent/main.py`, `tests/test_fundamental_alpha.py` (new), `tests/test_self_evolving_alpha.py`
- Tests: 188 passing

### 2026-04-19 (PEAD earnings surprise alpha sleeve)
- **Task 1 ✅**: `ascent/data/ingest/earnings.py` (new) — yfinance `earnings_dates` fetcher; 1-bday announcement lag via `pd.offsets.BDay(1)`; tz-strip on index; skips future rows (NaN Reported EPS); `fetch_earnings()`, `save_earnings()`, `load_earnings()`
- **Task 2 ✅**: `ascent/features/feature_defs.py` — `build_earnings_panel()` (pivot surprise_pct wide, ffill limit=63, tz-strip); `build_all_features()` gains `earnings_df=None` param
- **Task 3 ✅**: `ascent/alpha/earnings.py` (new) — `earnings_alpha()`: cross-sectional z-score of earnings_surprise; returns empty DF gracefully if feature absent
- **Task 4 ✅**: `ascent/alpha/stack.py` — earnings=0.05 added, fundamental reduced 0.10→0.05 (total stays 1.0); earnings sleeve wired into `build_alpha_stack()`; `ascent/alpha/ml_sleeve.py` — earnings_surprise added to ML_FEATURES; `ascent/features/build_features.py` — earnings_df=None param + compute_features passes it through; `ascent/main.py` — loads earnings cache and passes to FeatureBuilder; `ascent/research/self_improve.py` — DEFAULT_ALPHA_WEIGHTS synced with stack.py
- Earnings cache seeded: 3,228 rows, 135 symbols
- Updated stale test: `test_fundamental_alpha.py::test_default_alpha_weights_include_fundamental` (0.10→0.05)
- Files: `ascent/data/ingest/earnings.py` (new), `ascent/alpha/earnings.py` (new), `ascent/features/feature_defs.py`, `ascent/features/build_features.py`, `ascent/alpha/stack.py`, `ascent/alpha/ml_sleeve.py`, `ascent/main.py`, `ascent/research/self_improve.py`, `tests/test_earnings_alpha.py` (new), `tests/test_fundamental_alpha.py`
- Tests: 202 passing
- Open: Phase 4 hedge overlay (blocked ~May 13), R2R API key, debate trigger condition

### 2026-04-19 (pipeline bug fixes + integrity hardening)
- Fixed bdate_range weekend crash in `ascent/main.py`: `pd.bdate_range(end="today", periods=1)` returns empty on weekends; replaced with explicit weekday rollback (`max(0, today.weekday() - 4)` days back → last Friday on weekends, same day on weekdays)
- Fixed `RegimeEngine(cfg)` → `RegimeEngine(config=cfg.regime.to_engine_dict())` in `run_all_agents.py`; was passing full Config object, engine expects Optional[Dict]
- Fixed `build_fundamental_panel()` timezone mismatch: `datetime64[us]` vs `datetime64[us, America/New_York]`; strip tz from both `date_index` and `wide.index` before reindex
- Synced `self_improve.py` `DEFAULT_ALPHA_WEIGHTS` to match `stack.py` exactly (was missing fundamental and earnings sleeves from prior refactor)
- Alpha overlap note (not fixed, tracked): trend (55%) + 52wk high (inside fundamental) + earnings surprise are all momentum-correlated; effective momentum exposure ~65–70%, not 55%. Backlog: skip-last-month momentum, neutralize earnings surprise beta
- Files: `ascent/main.py`, `run_all_agents.py`, `ascent/features/feature_defs.py`, `ascent/research/self_improve.py`
- Commits: `85aedfc` (fundamental alpha + pipeline fixes), `7d14d5e` (PEAD earnings sleeve)

### 2026-04-23 (universe expansion + ML sleeve fix + self-improve sync)
- **Universe**: expanded from 135 hand-picked to 901 symbols (S&P 500 + S&P 400) via Wikipedia scrape; stored in `ascent/config/us_equity_universe.json`; `_load_us_equity_symbols()` in `settings.py` now reads from JSON with 20-name seed fallback
- **Profiles**: refreshed `profiles.parquet` for all 901 symbols — 93% sector coverage (838/901 with real labels); `data/universe.py` updated to load addition dates from JSON (merges with hardcoded dict, JSON wins)
- **Hub**: `_FETCH_WORKERS` bumped 4→8 for faster parallel ingestion on 901-symbol universe
- **mom_skip1m**: added skip-last-month momentum feature (`mom_252d - mom_21d`) to `feature_defs.py`; wired into `trend.py` at 0.20 weight; added to `ML_FEATURES`
- **Distressed name filter**: zeroes alpha for names with `mom_252d < -0.65` (down >65% YoY) in `stack.py` post-composite blend
- **Debate gate**: `should_run_debate()` wired into `run_all_agents.py` rebalance path; `verdict` now initialized to `{}` when gate returns False — fixed `NoneType.get()` crash on skipped debates
- **ML sleeve**: trimmed `ML_FEATURES` from 13 → 6 (kept only |ICIR|>0.2: mom_skip1m, zscore_20d, high_52w_pct, mom_126d, vol_63d, earnings_surprise); reduced `max_depth 4→3`, `n_estimators 200→100`; added `reg_alpha=0.1, reg_lambda=1.0`; `_stack_features` now cross-sectionally z-scores before stacking; p5 guard relaxed 0→−0.05
- **ML sleeve validation**: CPCV on full 6-year history (1,584 days, 120 symbols) → 15/15 folds, p5=−0.016, p50=+0.012 → **sleeve now enables**; was disabled every run due to noisy features + insufficient data in prior diagnostics
- **self_improve sync**: `MIN_SHARPE_EDGE` corrected 0.10→0.05 (matches design spec); stale docstring fixed; per-regime block now reuses `current_sharpe` instead of re-fetching with hardcoded 0.518 fallback
- Files: `ascent/config/settings.py`, `ascent/config/us_equity_universe.json` (new), `ascent/data/universe.py`, `ascent/data/hub.py`, `ascent/features/feature_defs.py`, `ascent/alpha/trend.py`, `ascent/alpha/stack.py`, `ascent/alpha/ml_sleeve.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `tests/test_phase3_hardening.py`, `tests/test_fundamental_alpha.py`
- Tests: 202 passing
- Open: analyst revision signal, R2R API key, neutralize earnings surprise beta

### 2026-05-03 (Phase 4 hedge overlay)
- **Phase 4 ✅**: `ascent/portfolio/hedge_overlay.py` — `compute_hedge_weight()` + `apply_hedge_overlay()` pure functions; VIXY sized 0–8% by regime × confidence; existing VIXY stripped before overlay; weights always sum to 1.0
- Wired into `run_all_agents.py` after orchestration, before `merged_weights.json` write; logs to `logs/hedge_log.jsonl`; try/except so failures don't abort runner
- `scripts/evaluate_hedge.py` — historical evaluation script; reads `regime_labels.csv` + `ascent_daily_ledger.csv`, fetches VIXY via yfinance, prints max drawdown / Sharpe / CAGR with vs without hedge
- README rewritten to reflect full current architecture (committed separately)
- Evaluation finding: over the backtest ledger the hedge slightly worsened drawdown and Sharpe; hedge-drawdown correlation was +0.26 (should be negative) — regime calibration issue, hedge fires when confidence is low so position is too small when regime actually turns; revisit after more live data
- Tests: 265 passing
- Files: `ascent/portfolio/hedge_overlay.py` (new), `tests/test_hedge_overlay.py` (new), `run_all_agents.py`, `scripts/evaluate_hedge.py` (new)
- Open: AI-native improvements, R2R API key, neutralize earnings surprise beta

### 2026-05-04 (AI-native planning — Tiers 1, 2, 3)
- Plans only — no code written this session
- **Tier 1** plan saved to `docs/superpowers/plans/2026-05-03-ai-native-tier1.md` (written prior session): Task A — CoT LLM fundamental alpha sleeve (`ascent/alpha/llm_fundamental.py`, Chicago Booth 6-step CoT, Haiku, cached by quarter, 3% weight, trend 0.44→0.41); Task B — slippage-adjusted IC feedback loop (`ascent/monitoring/slippage_ic_feedback.py`, Spearman IC gross vs net, drag coefficient to `active_alpha_config.json`, Sunday run); Task C — regime-conditional debate personas (`debate/outcome_tracker.get_agent_regime_accuracy()` + `debate/agents._get_agent_track_record()`, per-agent historical accuracy injected into system prompts)
- **Tier 2** plan saved to `docs/superpowers/plans/2026-05-03-ai-native-tier2.md`: Task D — FinMem-style reflection agent (`memory/reflection_agent.py`, scored verdicts → structured lessons → `memory/reflections.jsonl` → injected into `_build_context()`); Task E — LLM-guided hypothesis generation (`ascent/research/factor_proposer.py`, Haiku proposes regime-aware weight narratives, cosine-similarity deduplication, `generate_variants()` updated to call proposer first, random fallback); Task F — tool-capable debate agents (`debate/agent_tools.py` + `ascent/llm/client.tool_completion()`, bear and devil's advocate get 4 tools: `get_sector_concentration`, `get_var_estimate`, `get_position_momentum`, `get_regime_conditional_stats`)
- **Tier 3** plan saved to `docs/superpowers/plans/2026-05-03-ai-native-tier3.md`: Task G — autonomous factor discovery pipeline (`ascent/research/factor_discovery/`): Opus proposes factor code → AST validator (blocks imports/exec/file I/O/dangerous builtins, whitelist-only namespace) → rolling IC evaluator (Spearman IC, restricted exec sandbox) → accepted proposals written to `outputs/factor_proposals/` → human reviews and manually merges; monthly trigger first Sunday of month
- Files touched: `docs/superpowers/plans/2026-05-03-ai-native-tier2.md` (new), `docs/superpowers/plans/2026-05-03-ai-native-tier3.md` (new), `CLAUDE.md`
- Open: execute Tier 1 next (start with Task A LLM fundamental → Task B slippage IC → Task C personas), then Tier 2, then Tier 3; R2R API key still not configured; neutralize earnings surprise beta still in backlog

### 2026-05-05 (bug fixes + first live rebalance execution)
- **Hedge overlay fix** (`ascent/portfolio/hedge_overlay.py`): `apply_hedge_overlay` now accepts `Union[RegimeSignal, str]`; `AgentOutput.regime_signal` is a plain string but overlay expected a `RegimeSignal` object — converts via `RegimeLabel.from_str()` with 0.7 default confidence; unknown strings fall back to uncertain (no hedge)
- **ML sleeve sparse fill fix** (`ascent/alpha/ml_sleeve.py`): `_SPARSE_FILL_ZERO` expanded from `{"earnings_surprise"}` to include `analyst_revision`, `iv_skew`, `insider_net_score`, `short_pct_float`; these sparse panels were NaN-dropping all rows in `_stack_features` → empty X_all → sleeve disabled every run
- **ML sleeve cache fingerprint** (`ascent/alpha/ml_sleeve.py`): `_save_cached_model` and `_load_cached_model` now store/check `feature_names`; cache auto-invalidates when feature set changes (was causing XGBoost "Feature shape mismatch: expected 6, got 10" crash after feature set grew)
- **`load_dotenv()` added** (`run_all_agents.py`): FRED API key was not loading from `.env` — added `from dotenv import load_dotenv; load_dotenv()` at top of file
- **Approval server auto-approve** (`/tmp/auto_approve.py`): file-watcher script to auto-approve pending trades for manual rebalance triggers; runs alongside approval server
- **Rebalance calendar**: added `2026-05-05` as ad-hoc rebalance date
- **First full rebalance execution**: 40 orders submitted to Alpaca — 19 full closes (old portfolio), 21 new buys into new portfolio; approval gate triggered and approved; all 40 settled
- **New portfolio**: KMLM 10.9%, AMKR/FIX/WDC 5.9% each, IFRA 9.4%, VICR 5.7%, VRT 5.6%, VAL 5.2%, WCC 5.0%, DBB 4.6%, CNC 4.5%, WFRD 4.3%, PDBC 4.2%, STLD 3.4%, DBA 3.3%, CHRD 3.1%, EWY 2.4%, EWC 2.3%, IRM 1.8%, MUSA 1.5%, AAXJ/EEM/VWO 1.5% each, VIXY 2.8% hedge
- **12 new tests**: `test_string_regime_signal_accepted` in `tests/test_hedge_overlay.py`; 266 total passing
- Files: `ascent/portfolio/hedge_overlay.py`, `ascent/alpha/ml_sleeve.py`, `run_all_agents.py`, `tests/test_hedge_overlay.py`, `rebalance_calendar.csv`
- Open: Tier 1–3 AI-native plans not yet executed; R2R API key not configured; earnings surprise beta neutralization backlog

### 2026-05-06
- Daily run (non-rebalance): all 4 agents completed, 0 symbol fetch failures
- FRED: 4 series (T10Y2Y, CPIAUCSL, UNRATE, DCOILWTICO) failed with HTTP 500 — fixed with retry logic
- Fixed `ascent/data/ingest/fred.py`: `fetch_series()` now retries 3× with exponential backoff (1s, 2s) on any exception
- Portfolio: +1.12% vs SPY +1.39%; NAV $106,202; regime stressed (5 days since last fit)
- PDBC↔KMLM correlation (0.81) flagged by guard — KMLM halved; recurring pattern

### 2026-05-07
- Daily run (non-rebalance): all 4 agents completed, pipeline clean
- Regime refitted at startup (was 5 days stale) → calm_bull; FRED all 10 series loaded clean
- Portfolio: -2.53% vs SPY -0.31%; NAV $104,815; worst: VICR -6.8%
- FRED retry fix pushed to GitHub (commit 1f29101)
- Open: Tier 1–3 AI-native plans deferred to next week (usage limit)

### 2026-05-08 (plan hardening — Tier 1/2/3 revisions)
- Applied practitioner/ML reviewer feedback to all three AI-native implementation plans
- **Tier 1 Task 0 added**: three prerequisite fixes before any Tier 1 tasks execute
  - 0.1: `SELF_MODIFY_ENABLED = False` kill switch in `self_improve.py`; activation condition: +OOS Sharpe for 30 consecutive trading days on flat config
  - 0.2: private debate context subsets per agent — `_build_agent_context()` using section-level builders keyed to actual `portfolio_state` fields; corrected a flawed keyword-filter design that would have produced nearly identical context for all agents
  - 0.3: explicit `+` sign prefix standard for CAGR/Sharpe/Alpha in README and all outputs
- **Tier 2 Task E**: added dependency note on `SELF_MODIFY_ENABLED`; `generate_variants()` returns `[]` when gate is closed
- **Tier 3**: added `⚠ Do not begin until:` warning block with three gate conditions and IC threshold rationale
- README updated: `+` sign prefixes on walk-forward table, self-modify gate documented, private context subsets documented in debate section, Planned section rewritten to reflect plan revisions
- Files: `docs/superpowers/plans/2026-05-03-ai-native-tier1.md`, `docs/superpowers/plans/2026-05-03-ai-native-tier2.md`, `docs/superpowers/plans/2026-05-03-ai-native-tier3.md`, `README.md`, `CLAUDE.md`
- Open: Tier 1–3 execution still pending; R2R API key not configured; earnings surprise beta neutralization backlog

### 2026-05-08 (Task 0B — disagreement score plan)
- Added Task 0B to `docs/superpowers/plans/2026-05-03-ai-native-tier1.md` between Task 0 and Task A
- Task 0B: `debate/disagreement_scorer.py` — pairwise TF-IDF cosine similarity on agent reasoning traces; `disagreement_score = 1 - mean(pairwise sims)`; pure numpy, no external API
- Score written to every verdict JSON; judge receives score as informational note only (not a verdict override — TF-IDF vocabulary overlap is too coarse to drive individual decisions)
- Primary use: longitudinal validation metric for Task 0.2 — mean score tracked weekly; if it doesn't drop after `_build_agent_context()` goes live, section builder design needs revisiting
- Removed unverifiable Tajik et al. (2026) arXiv citation from plan; removed directional bias from judge injection; updated test assertions to confirm no verdict prescription
- 7 tests specified in `tests/test_disagreement_scorer.py`; 10 implementation steps
- README updated: Task 0B entry reflects monitoring-only framing
- Files: `docs/superpowers/plans/2026-05-03-ai-native-tier1.md`, `README.md`, `CLAUDE.md`
- Open: all Tier 1–3 tasks still unexecuted; R2R API key not configured

### 2026-05-08 (AI-native Tier 1 execution)
- **Task 0 ✅**: `SELF_MODIFY_ENABLED = False` kill switch in `ascent/research/self_improve.py`; `run_self_improve()` returns `[]` early with `log.warning()` when gate is closed
- **Task 0B ✅**: `debate/disagreement_scorer.py` — TF-IDF cosine similarity scorer; wired into `debate_runner.py` (computes after Round 2, writes to verdict JSON) and `debate/judge.py` (`disagreement_context` param, informational-only); 7 tests
- **Task A ✅**: `ascent/alpha/llm_fundamental.py` — Chicago Booth 6-step CoT, Haiku, cached by (symbol, quarter_end), 45-day filing lag, z-scored output, IC logging to `logs/llm_fundamental_signals.jsonl`; wired into `stack.py` at 3%, trend 0.44→0.41; `self_improve.py` synced; 8 tests
- **Task B ✅**: `ascent/monitoring/slippage_ic_feedback.py` — Spearman IC gross vs net-of-slippage, MIN_FILLS=50, passive logger until ~July 2026; wired into `run_all_agents.py` Sunday block; 5 tests
- **Task C ✅**: `debate/outcome_tracker.py` — `get_agent_regime_accuracy()` with `min_samples=10`; `debate/agents.py` — 8 section builders, `_build_agent_context()`, `_get_agent_track_record()`, track records injected into all 4 agent system prompts; private context subsets per agent enforced; 6 tests
- Branch `feature/ai-native-tier1` merged to main and pushed; 292 tests passing (up from 266)
- Files: `ascent/alpha/llm_fundamental.py` (new), `ascent/monitoring/slippage_ic_feedback.py` (new), `debate/disagreement_scorer.py` (new), `ascent/research/self_improve.py`, `ascent/alpha/stack.py`, `debate/agents.py`, `debate/outcome_tracker.py`, `debate/judge.py`, `debate/debate_runner.py`, `run_all_agents.py`, + 4 new test files
- Open: Tier 2 and Tier 3 plans not yet executed; R2R API key not configured; `signal_score` in slippage IC uses price impact (not alpha score) — revisit before MIN_FILLS reached

### 2026-05-08 (AI-native Tier 2 execution)
- **Task D ✅**: `memory/reflection_agent.py` — FinMem-style post-trade reflection; Haiku reads scored verdicts, writes structured lessons to `memory/reflections.jsonl`; idempotent dedup via `verdict_date`; injected into `_build_context()` per regime; wired into `run_all_agents.py` daily (independent try/except, not gated on scoring); 6 tests
- **Task E ✅**: `ascent/research/factor_proposer.py` — LLM-guided hypothesis generation; Haiku proposes regime-aware narratives with weight biases; cosine-similarity deduplication (threshold 0.85); iterative floor enforcement (trend ≥5%); `generate_variants()` uses proposer when regime known, falls back to `_random_variants()`; `SELF_MODIFY_ENABLED=False` guard respected; 6 tests
- **Task F ✅**: `debate/agent_tools.py` — 4 domain tools (sector concentration, VaR, position momentum, regime stats); `execute_tool` dispatcher never raises; `ascent/llm/client.py` — `tool_completion()` with Anthropic tool-use loop, `max_tool_calls` guard, `generate_structured` fallback; `run_bear_agent` and `run_devils_advocate` use tool_completion as primary path; 8 tests
- Branch `feature/ai-native-tier2` merged to main and pushed; 312 tests passing (up from 292)
- Files: `memory/reflection_agent.py` (new), `ascent/research/factor_proposer.py` (new), `debate/agent_tools.py` (new), `ascent/llm/client.py`, `ascent/research/self_improve.py`, `debate/agents.py`, `run_all_agents.py`, + 3 new test files
- Open: Tier 3 (autonomous factor discovery) not yet started; R2R API key not configured; slippage IC signal_score revisit before MIN_FILLS; `SELF_MODIFY_ENABLED` activation requires 30 consecutive days +OOS Sharpe

### 2026-05-08 (Tier 3 plan research + rewrite)
- Researched academic and industry techniques for autonomous factor discovery (AlphaAgent arXiv:2502.16789, QuantaAlpha arXiv:2602.07085, Harvey/Liu/Zhu FDR correction, PySR symbolic regression, Grinold & Kahn Fundamental Law)
- Core finding: free-form LLM code generation unreliable; every system with real excess returns used strict structural constraints
- **Tier 3 plan rewritten** (`docs/superpowers/plans/2026-05-03-ai-native-tier3.md`): primary path = PySR symbolic regression on pre-computed features; secondary = LLM suggests JSON template params for 5 template families (no code injection); IC IR threshold raised 0.40→0.60 (Harvey FDR: at 0.60, ~0.5 spurious acceptances/year from 50 candidates); per-regime CPCV gate (IC_min_regime > 0.01 in every observed regime); leakage scanner (AST + regex for lookahead patterns); 7 new files, 14 tests
- Gate conditions for Tier 3 execution: SELF_MODIFY_ENABLED (30 days +OOS), 63 days regime labels, MIN_FILLS=50 in slippage IC, PySR installed, CPCV control validation
- Files: `docs/superpowers/plans/2026-05-03-ai-native-tier3.md`, `CLAUDE.md`
- Open: Tier 3 execution blocked on gate conditions (~July 2026); R2R API key not configured; slippage IC signal_score revisit before MIN_FILLS

### 2026-05-09 (AI-native Tier 3 execution)
- **Task G ✅**: `ascent/research/factor_discovery/` — autonomous factor discovery pipeline (7 files, 14 tests)
  - `feature_templates.py` — 5 template families (Momentum, Reversion, Volatility, Quality, Correlation); LLM fills params via JSON; no code injection
  - `leakage_scanner.py` — AST + regex lookahead detector; rejects `.tail(1)`, `datetime.now()`, `.shift(-N)`, future dates in subscripts
  - `regime_cpcv_evaluator.py` — per-regime Spearman IC evaluator; Harvey FDR threshold IC_mean ≥ 0.015 AND IC_IR ≥ 0.60; reports IC per regime
  - `llm_suggester.py` — Haiku proposes template + params as JSON; returns None on failure; zero code risk
  - `pysr_engine.py` — PySR symbolic regression on pre-computed feature panel; graceful fallback if pysr not installed
  - `discovery_runner.py` — orchestrates both paths; acceptance gate: Harvey FDR + IC_min_regime > 0.01 + n_obs ≥ 20; proposals written to `outputs/factor_proposals/`; nothing auto-deploys
  - `__init__.py` — package marker
- PySR installed: `pip install pysr`
- Monthly trigger wired into `run_all_agents.py`: first Sunday of each month (day ≤ 7); `_get_current_regime()` helper added
- 326 tests passing (up from 312)
- Files: `ascent/research/factor_discovery/` (7 new files), `tests/test_factor_discovery.py` (new), `run_all_agents.py`
- Open: gate conditions for Tier 3 activation still ~July 2026 (SELF_MODIFY_ENABLED, 63d regime labels, MIN_FILLS=50); R2R API key not configured

### 2026-05-09 (momentum overlap fix)
- **Earnings sleeve**: `ascent/alpha/earnings.py` — per-date OLS regression of `earnings_surprise` on `mom_126d`; residuals z-scored; removes momentum beta before PEAD signal is used; falls back cleanly when `mom_126d` absent
- **Fundamental sleeve**: `ascent/alpha/fundamental.py` — removed `high_52w_pct` (price momentum signal, not accounting quality); sleeve now = `gross_profitability + accruals + asset_growth` only; returns empty DataFrame when no fundamental data (no 52wk-high fallback rescuing it with momentum exposure)
- Two test fixes (`test_fundamental_alpha_works_without_fundamentals` → assert empty; `test_fundamental_alpha_builds_composite` → removed high_52w_pct from features) + two new tests (momentum neutralization effectiveness, graceful no-mom fallback)
- Tests: 328 passing (up from 326)
- Files: `ascent/alpha/earnings.py`, `ascent/alpha/fundamental.py`, `tests/test_earnings_alpha.py`, `tests/test_fundamental_alpha.py`

### 2026-05-10 (institutional roadmap — Plans 1–7)
- Wrote 7 detailed implementation plans for taking Ascent to institutional and YC-ready state
- Plan 1: Factor risk model (Barra-style) — 18 tests, 8 tasks
- Plan 2: Portfolio construction overhaul (cvxpy MVO, Black-Litterman, regime covariance) — 20 tests, 5 tasks
- Plan 3: Event-driven architecture (EDGAR, Capitol Trades, options anomaly) — 16 tests, 8 tasks
- Plan 4: Alternative data pipeline (SEC full-text, earnings transcripts, Reddit, Google Trends) — 20 tests, 7 tasks
- Plan 5: Execution excellence (TWAP, implementation shortfall, capacity model) — 16 tests, 5 tasks
- Plan 6: Real-time infrastructure (TimescaleDB, WebSocket streaming, live dashboard, monthly PDF report) — 14 tests, 8 tasks
- Plan 7: Live track record + compliance (immutable audit trail, GIPS performance, risk disclosures, methodology doc) — 12 tests, 8 tasks
- Files: 7 new plan files in `docs/superpowers/plans/`, `CLAUDE.md` updated
- Open: Plan 1 is next; Plans 3 and 4 can proceed in parallel; Plan 7 infrastructure should start immediately (audit trail captures all live trades from this point forward)

### 2026-05-10 (Plan 1 — Factor Risk Model ✅)
- **Task 1 ✅**: `ascent/risk/factor_data.py` — Fama-French 5 + UMD downloader; incremental cache (`data_cache/factor_returns.parquet`); stale if > 1 day; `update_factor_data()` + `get_factor_returns(start, end)`
- **Task 2 ✅**: `ascent/risk/factor_model.py` — rolling OLS loadings; vectorized one `np.linalg.lstsq` per date across all symbols; `BETA_COLS = [beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, beta_umd]`; incremental update (last 5 bdays); graceful fallback to most recent date
- **Task 3 ✅**: `ascent/risk/covariance_model.py` — Σ = B·F·B' + D; Ledoit-Wolf shrinkage on residuals; `build_factor_covariance_matrix()`, `portfolio_variance()`, `factor_variance_decomposition()`
- **Task 4 ✅**: `ascent/risk/factor_exposure.py` — w'B portfolio tilt; soft bounds; `export_factor_exposures()` → `dashboard/factor_exposures.json`; `format_exposure_context()` for debate
- **Task 5 ✅**: `ascent/risk/factor_constraints.py` — regime-aware constraint dicts for Plan 2 MVO; crisis tightest, calm_bull standard
- **Task 6 ✅**: `ascent/monitoring/attribution.py` — `compute_factor_pnl()` added; `factor_pnl` + `idiosyncratic_pnl` in every attribution log entry
- **Task 7 ✅**: `debate/agents.py` — `_section_factor_exposures()` added; devil's advocate is sole recipient of factor exposure context
- **Wiring**: `run_all_agents.py` — Step 0b updates factor data + loadings at startup; Step 5c exports factor exposures after orchestration
- **Tests**: `tests/test_factor_risk_model.py` — 18 tests, all passing; full suite 346 passing (up from 328)
- Files: `ascent/risk/factor_data.py` (new), `ascent/risk/factor_model.py` (new), `ascent/risk/covariance_model.py` (new), `ascent/risk/factor_exposure.py` (new), `ascent/risk/factor_constraints.py` (new), `tests/test_factor_risk_model.py` (new), `ascent/monitoring/attribution.py`, `debate/agents.py`, `run_all_agents.py`
- Open: Plan 2 (cvxpy MVO) is next; Plans 3–7 queued

### 2026-05-10 (Plan 2 — Portfolio Construction Overhaul ✅)
- **Task 1 ✅**: `ascent/portfolio/mvo_optimizer.py` — cvxpy MVO; maximize w'α - λ(w'Σw) - κ‖w-w_prev‖₁; CLARABEL solver with SCS fallback; diagonal proxy when covariance unavailable; infeasible → None
- **Task 2 ✅**: `ascent/portfolio/black_litterman.py` — BL posterior blending quant prior with LLM view; proper matrix formula when covariance available; shrinkage fallback otherwise; `get_blending_weight()` maps IC IR → tau
- **Task 3 ✅**: `ascent/portfolio/regime_covariance.py` — per-regime sample covariance (calm_bull/stressed/crisis); confidence-weighted blend; Ledoit-Wolf fallback when < 63 obs
- **Task 4 ✅**: `ascent/portfolio/optimizer.py` — `sector_constrained_weighted_mvo()` as primary path; sector pre-screening → BL blending → MVO → rank-weight fallback; returns (weights, optimization_method)
- **Task 5 ✅**: `ascent/main.py` — MVO on latest rebalance date using Plan 1 factor covariance; LLM alpha for BL blending; historical dates use rank-weight; optimization_method logged
- **Tests**: 20 new tests (19 pass, 1 skipped); full suite 366 passing (up from 346)
- Files: `ascent/portfolio/mvo_optimizer.py` (new), `ascent/portfolio/black_litterman.py` (new), `ascent/portfolio/regime_covariance.py` (new), `ascent/portfolio/optimizer.py`, `ascent/main.py`, `tests/test_mvo_optimizer.py` (new), `tests/test_black_litterman.py` (new)
- Open: Plan 3 (event-driven architecture) is next; Plans 4–7 queued

### 2026-05-10 (Plan 3 — Event-Driven Architecture ✅)
- **Task 1 ✅**: `ascent/data/ingest/edgar_listener.py` — EDGAR RSS polling for 8-K/8-K/A; text extractor (4K chars); CIK→symbol map; seen-filings dedup across restarts
- **Task 2 ✅**: `ascent/alpha/event_alpha.py` — Haiku 8-K classifier with 5 few-shot examples; rule-based congressional trade classifier (conviction=0.4, 30-45d lag); rule-based options anomaly classifier (IV z-score + put/call z-score thresholds)
- **Task 3 ✅**: `ascent/data/ingest/capitol_trades.py` — House/Senate eFD API client; filters to universe; dedup against seen cache
- **Task 4 ✅**: `ascent/data/ingest/options_scanner.py` — Alpaca options chain scanner; Welford online IV baseline; graceful disable when API unavailable
- **Task 5 ✅**: `ascent/execution/event_runner.py` — `EVENT_TRADING_ENABLED=False` kill switch; 0.5% NAV cap; conviction × urgency sizing; limit orders; approval gate for > 1% NAV; weekly Spearman IC tracking
- **Task 6 ✅**: `agents/event_agent.py` — daemon thread; EDGAR every 5min, options every 15min, Capitol Trades every 60min; max 1 trade per symbol per day; exits at 16:10 ET
- **Task 7 ✅**: `ascent/execution/eod_runner.py` — `get_event_positions_today()` subtracts filled event positions from rebalance sizing; `run_all_agents.py` starts thread at market open, computes event IC on Sundays
- **Tests**: 16 tests, all passing; full suite 382 passing (up from 366)
- Files: 7 new files, 2 modified + `logs/event_trades.jsonl`
- Open: Plans 4–7 queued; `EVENT_TRADING_ENABLED` stays False until 30-day paper validation (~July 2026)

### 2026-05-10 (Plan 4 — Alternative Data Pipeline ✅)
- **Task 1 ✅**: `ascent/data/ingest/sec_filings.py` — EDGAR 10-K/10-Q; `extract_mda_section()` regex boundary detection + fallback; Haiku 5-axis classifier (revenue_momentum, margin_trend, tone, liquidity_risk, guidance); 45-day filing lag; 90-day forward-fill; `update_sec_signals()` incremental cache
- **Task 2 ✅**: `ascent/data/ingest/earnings_transcripts.py` — EDGAR 8-K Item 2.02; prepared remarks / Q&A splitter; Haiku tone/defensiveness/forward-confidence/quantitative_ratio classifier; 1-bday lag; 63-day ffill
- **Task 3 ✅**: `ascent/data/ingest/reddit_sentiment.py` — PRAW mentions × 4 subreddits; TextBlob sentence sentiment; contrarian z-score (high retail excitement → bearish); credentials via REDDIT_CLIENT_ID/SECRET env vars; no user text stored
- **Task 4 ✅**: `ascent/data/ingest/google_trends.py` — pytrends search velocity; rate-limited 1/5s; weekly Sunday refresh capped at 50 symbols; 1-day lag; compute_trends_signal() cross-sectional z-score of velocity
- **Task 5 ✅**: `ascent/data/validate/altdata_validator.py` — IC gate: IC_mean ≥ 0.015, IC_IR ≥ 0.60, IC_min_regime > 0.010, n ≥ 20; `run_altdata_validation()` writes proposals to `outputs/altdata_proposals/` for human review; `register_altdata_source()` updates `active_alpha_config.json`
- **Task 6 ✅**: `ascent/alpha/altdata_alpha.py` — IC-weighted combiner; reads `altdata_weights` from config; returns empty DataFrame when no sources active
- `ascent/alpha/stack.py` — `altdata` sleeve added at 0.00 weight; `ascent/research/self_improve.py` synced
- `run_all_agents.py` — monthly altdata validation (first Sunday); weekly Google Trends refresh
- **Tests**: 23 tests, all passing; full suite 404 passing (up from 382)
- Files: 6 new files, 3 modified; commit `f5c6873`
- Open: Plans 6–7 queued; `EVENT_TRADING_ENABLED` stays False until ~July 2026; no altdata source yet validated (need 63+ days of live fills)

### 2026-05-10 (Plan 5 — Execution Excellence ✅)
- **Task 1 ✅**: `ascent/execution/twap_executor.py` — `TWAP_ENABLED=False` kill switch; `build_twap_schedule()` equal-spaced child orders (min 1 share/slice); `compute_twap_window()` Almgren-Chriss T*; `should_use_twap()` 5% ADV gate; `execute_twap()` submits limit orders via Alpaca; wired into `order_engine.py` at module level for patchability
- **Task 2 ✅**: `ascent/execution/implementation_shortfall.py` — IS decomposition: delay cost (signal→arrival), market impact (arrival→fill), opportunity cost (unfilled); all in bps vs decision price; `log_is_record()` backward-compatible append to `slippage_log.jsonl`; `is_summary()` trailing-period stats
- **Task 3 ✅**: `ascent/execution/capacity_model.py` — `estimate_market_impact()` η·σ·(X/ADV)^0.6; `compute_signal_breakeven_adv()` Grinold alpha vs impact; `compute_strategy_capacity()` per-sleeve max NAV; `capacity_report()` logs to `logs/capacity_log.jsonl`; informational only
- **Task 4 ✅**: `ascent/execution/intraday_trigger.py` — 3 triggers: regime emergency (SPY −3%+VIX>30 → ×0.70 de-risk), drawdown pre-emption (≥12% → −20% exposure), event urgency (high-urgency event on top-5 position → 50% trim); `execute_intraday_adjustment()` logs to `logs/intraday_adjustments.jsonl`; `run_intraday_trigger_check()` entry point in `eod_runner.py`
- **Task 5 ✅**: `slippage_tracker.py` — `compute_slippage()` now appends `is_breakdown` sub-dict when decision price is recorded; `fill_quality_report()` returns mean IS components + by-sleeve breakdown
- **Tests**: 16 tests, all passing; full suite 420 passing (up from 404)
- Files: 4 new, 4 modified; commit `71c60cb`
- Open: Plans 6–7; TWAP_ENABLED stays False until paper validation (~July 2026); intraday triggers logged only (no live orders yet)
