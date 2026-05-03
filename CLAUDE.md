# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio construction → walk-forward evaluation → regime modeling → 4 specialist agents → orchestration → LLM debate → execution via Alpaca paper trading.

---

## Repository layout

```
ascent/
  config/       settings.py (Config, APIKeys, UniverseConfig, BacktestConfig), types.py (AgentOutput)
                us_equity_universe.json — 901 S&P 500 + S&P 400 symbols with GICS sectors
  data/         ingest (yahoo, fred, simulated, fundamentals, earnings, analyst, options, insider, short_interest), normalize, store (parquet, point_in_time), universe
  features/     build_features (FeatureBuilder), feature_defs (all panel builders), targets
  alpha/        trend, meanrev, statarb, ml_sleeve (CPCV), stack (compositor)
                fundamental, earnings, analyst — Tier 1 alpha sleeves
                options_flow, insider, short_interest — Phase 6 sleeves
  portfolio/    optimizer — sector_constrained_weighted, _water_fill_cap, apply_bl_to_latest
                covariance — Ledoit-Wolf shrinkage estimator
                mv_optimizer — Black-Litterman posterior + MV optimization (scipy SLSQP)
                tc_aware_optimizer — TC-aware rebalancing (10bps kappa, 50bps deadband)
  backtest/     engine, costs
  research/     walk_forward_runner, walk_forward_lightweight, cpcv, self_improve, shadow_promoter
  regime/       engine, model, features, decision, integration, posture, breaks, particle_filter, types
  risk/         correlation_guard
  reporting/    market_memo, ic_brief_generator, blind_spot_detector, debrief, regime_narrative, catalyst_scanner
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch, run_log, slippage_tracker,
                approval_server, cost_model, debate_gate
  monitoring/   skill_tracker, forward_pnl_tracker, pre_rebalance_checklist, exit_alerts,
                attribution, counterfactual_tracker, quant_context
  llm/          client.py — centralized Anthropic API wrapper

agents/         us_equities, macro, international, alternatives
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker
memory/         r2r_interface (R2R HTTP + BM25 fallback)
simulation/     mirofish_interface

data_cache/     prices_live, macro_live, profiles, fundamentals, earnings, analyst_revisions
                options_flow, insider_transactions (23k rows, 2013–2026), short_interest (900 syms)
                ml_model_*.pkl, active_alpha_config.json, shadow_configs/, archived_configs/
dashboard/      HTML dashboards, regime_signal.json, regime_labels.csv, agent_skill_scores.json
outputs/
  debate_log/   verdict_YYYY-MM-DD.json, agent_credibility.json
logs/           eod_log, slippage_log, self_improve_log, skill_scores_log, multi_agent_run,
                us_equities_pnl, macro_pnl, international_pnl, alternatives_pnl,
                attribution_log, sleeve_ic_log, kill_switch_state
                snapshots/{agent_id}_weights_YYYY-MM-DD.json

ascent/main.py        core pipeline entrypoint
run_all_agents.py     single daily command — branches on rebalance day
demo_app.py           Streamlit interactive demo (Tony Ngo)
```

---

## Core runtime flow

**Command**: `python3 run_all_agents.py`

**Non-rebalance day**: agents (parallel) → score counterfactuals → shadow promotion check → forward PnL → skill scores → orchestrator → write `merged_weights.json` → log. Stop.

**Rebalance day**: same + pre-rebalance checklist → debate gate check → debate (if gated) → verdict gates execution → Alpaca orders → slippage tracking.

`ascent/main.py` pipeline: data → normalize → regime fit → credit/yield features → alpha stack (11 sleeves) → BL weight refinement → SPY 200MA overlay → backtest → per-sleeve IC log → export.

---

## Alpha stack

Default sleeve weights (stack.py `DEFAULT_ALPHA_WEIGHTS`):

| Sleeve | Weight | Notes |
|--------|--------|-------|
| Trend | 44% | Cross-sectional momentum; skip-last-month `mom_252d - mom_21d` at 0.20 sub-weight |
| Stat-arb | 15% | Sector residuals; needs profiles.parquet |
| Mean reversion | 5% | Short-term reversal (`zscore_20d`) |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds; 7 features: mom_skip1m, zscore_20d, high_52w_pct, mom_126d, vol_63d, earnings_surprise, analyst_revision |
| Volatility (vol-regime) | 5% | Signal = -(vol_trend_10d) / (vol_of_vol_21d); long declining+stable vol names |
| Fundamental | 5% | Gross profitability + accruals + asset growth + 52wk high; 45-day filing lag; cache: `fundamentals` |
| Earnings (PEAD) | 5% | Cross-sectional z-score of earnings_surprise_pct; 1-bday lag; ffill limit=63; cache: `earnings` |
| Analyst | 5% | Rolling 63-day net upgrade score; yfinance upgrades_downgrades; 1-bday lag; cache: `analyst_revisions` |
| Options Flow | 2% | IV skew (OTM put IV − call IV) + put/call ratio; bearish → negative alpha; cache: `options_flow` |
| Insider | 2% | Rolling 63-day net open-market purchase score (Form 4 via yfinance); 1-bday lag; cache: `insider_transactions` |
| Short Interest | 2% | Short % of float cross-sectional z-score; high short → squeeze potential → positive alpha; cache: `short_interest` |

Regime adjusts sleeve weights via `integration.py:regime_adjust_sleeve_weights()`. All sleeves cross-sectionally z-scored before blending.

ML sleeve: CPCV C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays — disabled if <10 folds converge or p5 IC Sharpe < −0.05. Cap at top-300 symbols by data completeness (prevents OOM on 901-symbol universe).

Distressed name filter: zeroes alpha for names with `mom_252d < -0.65` (down >65% YoY) post-blend in `stack.py`.

---

## Portfolio construction

`sector_constrained_weighted()`: coverage check (< 80% → skip sector caps + warn) → rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative, ≤50 iterations) → hard clamp + renorm. Post-condition: sum=1.0±tol, no position > max_weight.

`apply_bl_to_latest()`: Black-Litterman refinement applied to the most recent date's weights only. Sector selection unchanged — BL re-optimizes the intra-portfolio weight allocation using Ledoit-Wolf covariance. Falls back silently if <126 days of price history or <3 names. TC-aware sizing available when current holdings provided (10bps kappa, 50bps deadband).

Regime tightens max_weight: crisis → 0.08, calm_bull → 0.15. SPY 200MA overlay: SPY < 200MA → multiply weights × 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays.

---

## Agents

**US Equities**: 901 symbols (S&P 500 + S&P 400), full 11-sleeve alpha stack, max 1 per sector, BL-refined weights, 12–20 positions.

**Macro**: TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM. Trend-only. Regime-sized: crisis top_n=3/40%, stressed top_n=4/35%, else top_n=5/30%. Cache: `prices_macro.parquet`.

**International**: EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA. Max 2 per region. UUP > 50MA → 20% alpha penalty on EM names.

**Alternatives**: VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL. Trend 80% + low-vol 20%. Kill switch at 12%. Max 35%, min 5%, top 4.

---

## Orchestrator (`orchestrator/central_intelligence.py`)

1. **Base by regime**: calm_bull US60/mac15/intl15/alt10; stressed US45/mac25/intl10/alt20; crisis US30/mac30/intl5/alt35.
2. **Skill blend**: per-agent independently; negative Sharpe → zero; else 50% skill + 50% base.
3. **Conviction bonus**: up to +15% when ≥2 agents share a name (conv > 0.3).
4. **EM/commodity cap**: hard 20% cap on EM+commodity+gold after all blending.
5. **Correlation guard**: 63-day cross-agent cap at 0.70 → halve smaller.
6. **Thesis coherence**: symbol contradictions (UUP↔PDBC/GLD, VIXY↔SVXY) + 12 factor buckets / 6 pairs → 40% reduction.
7. **Crisis veto**: us_regime=crisis → merged = 0.60×macro + 0.40×merged.

---

## Regime system

HMM K=2–4 (best via walk-forward CV). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3d / entropy > 0.90 → uncertain. Particle filter: 500 particles SIR, reinitializes on batch refit. Emergency refit triggers: SPY −3%+VIX>30, 200MA cross, SPY/TLT corr flip, break z-score > 3.5. Refit every 5 days.

Credit/yield features wired into regime: `credit_spread_chg_21d/level` (HYG/LQD), `yield_curve_slope/chg` (TLT/IEF) — leading indicators fetched alongside VIX in `main.py`.

---

## Debate layer

Conditional — fires only when `debate_gate.py:should_run_debate()` returns True (regime entropy > 0.70, top position > 12%, VaR 99th < −3.5%, or catalyst detected). Advisory — never writes to alpha/portfolio/execution.

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

Counterfactual tracking (`monitoring/counterfactual_tracker.py`): snaps quant vs debate weights at time of verdict; scores the counterfactual 10 days later to measure debate value-add. Scored daily in `run_all_agents.py`.

---

## Self-improve loop

Weekly (Sunday 6AM). Generates 5 sleeve-weight variants, scores via real multi-fold OOS (`run_lightweight_oos()`). Shadow promotion if edge > 0.05 Sharpe, 30-day monitoring, auto-promoted by `shadow_promoter.py`. Per-regime variants written to `active_alpha_config.json` `by_regime` section. Stack reads live config on every run.

**Sleeve floor protection**: `MIN_SLEEVE_WEIGHTS` in `self_improve.py` prevents any intentional sleeve from being perturbed to zero (trend: 10%, fundamental/earnings/analyst: 2%). `shadow_promoter._restore_sleeve_floors()` adds a second safety net before writing `active_alpha_config.json`.

**Per-sleeve IC logging**: `_log_sleeve_ic()` in `main.py` computes IC per sleeve after every run and appends to `logs/sleeve_ic_log.jsonl` — enables early decay detection.

---

## Execution

Kill switch: SOFT_WARN 8% (log+proceed), HARD_STOP 15% (abort). Alternatives: 12%. State in `logs/kill_switch_state.json`. Large trades (> 2% NAV) → `execution/pending_approvals.json`, async wait via `threading.Event`, 30-min timeout. Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV. Post-fill: slippage logged to `logs/slippage_log.jsonl`.

Config: always `get_config()`, never `Config()` directly.

---

## LLM clients

`ascent/llm/client.py`: `DEFAULT_MODEL = "claude-opus-4-6"`, `HAIKU_MODEL = "claude-haiku-4-5-20251001"`. Lazy singleton. Retry 3× with 2s/4s backoff. All files import `HAIKU_MODEL` from here — never redefine locally.

---

## Data / caching

Cache names:
- `prices_live` — Yahoo live (never write simulated under this name)
- `prices_simulated` — GBM fallback
- `prices_live_fallback_simulated` — live-fetch failure fallback
- `prices_macro` — macro agent ETF prices
- `macro_live` / `macro_simulated` — FRED or fallback
- `profiles` — sector metadata (GICS, 93% coverage on 901-symbol universe)
- `fundamentals` — quarterly gross profitability, accruals, asset_growth (45-day filing lag)
- `earnings` — earnings_dates surprise_pct (1-bday lag, ffill limit=63)
- `analyst_revisions` — upgrades/downgrades net score (1-bday lag, 225k+ rows)
- `options_flow` — IV skew + put/call ratio snapshot (today-only; append daily; useful after ~21 days)
- `insider_transactions` — Form 4 open-market buys/sells (23k rows, 892 symbols, 2013–2026 backfill)
- `short_interest` — shortPercentOfFloat + shortRatio (900 symbols; append daily; ffill 15 days)

Never hide data provenance in cache name. Point-in-time joins via `as_of_join()` / `as_of_merge()`.

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
8. Sleeve floors: fundamental/earnings/analyst ≥ 2%, options_flow/insider/short_interest ≥ 1% — enforced in both self_improve.py and shadow_promoter.py.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Environment

Python 3.12.13 Homebrew, venv at `.venv/`. Use `.venv/bin/python`. API keys via `APIKeys.from_env()` only. Mac Air M5, no JAMF restrictions.

---

## Current portfolio (as of May 2, 2026)

Last merged weights (2026-05-02): KMLM 9.7%, IFRA 8.7%, PDBC 8.2%, AMKR/FIX/VAL/VICR/VRT/WDC 5.6% each, CNC 5.4%, EWY 4.1%, VC 3.8%, DBA/EWC 3.3%, PVH 3.2%, TIP 2.2%, NUE 2.0%, XPO/APA 1.9%, AAXJ/EEM 1.5%, EWZ 1.2%.
Next rebalance: ~May 6, 2026 (10-bday cadence from Apr 29). Live since April 1, 2026.

---

## Known bugs — all fixed

All codebase audits (Apr 17–18) complete. Third-pass found no new crash risks. Key fixed classes:
- Empty weights guard (E1), DataFrame column checks (E2–E4), regime log (E5)
- Debate None-safe access (D1–D2)
- run_all_agents empty list/allocation/staleness guards (R1–R4)
- iloc[-1] guards after .dropna() in all agent files
- ML CPCV OOM: top-300 cap + X_all built once + pd.concat instead of .join (2026-05-02)
- Hub _ROOT path fix: parents[3] → parents[2] (2026-05-02)
- Optimizer sector fallback: SectorDataError → rank-weighted fallback (2026-05-02)

---

## What is not built yet

- **Phase 6 — New signals**: ✅ Built. Options flow, insider transactions, short interest all live.
- **Phase 7 — Autonomous research engine**: IC decay monitor → LLM hypothesis generator → auto-coder → CPCV backtest → shadow promotion. Week-long build.
- **Phase 8.3 — Options hedge overlay**: Tail-risk hedge via SPX put spreads. Unblocks ~May 13 (30 days live data needed for sizing calibration).
- **Phase 8.1 — VWAP execution**: Slice large orders across 8 intraday buckets proportional to historical volume profile.
- **R2R semantic memory**: built but `R2R_API_KEY` not configured; BM25 fallback active.
- **Live dashboard UI**: data files generated but no live render.

---

## Session log

### 2026-04-09 to 2026-04-13 (summary)
- Initial CLAUDE.md, env setup, 6 bug fixes, A4 survivorship bias hardening
- Phase 1–3: skill staleness, sector error, persistent halt, async approval gate, Almgren-Chriss, CPCV ML sleeve, regime particle filter + emergency refit
- 3 AI agent features: catalyst scanner, multi-turn debate, memory-augmented debate (93 tests)
- Universe: removed 15 delisted, added 15 new (135 total at that point)

### 2026-04-14 (Tony Ngo demo — plan only)
- Designed `demo_app.py` architecture; nothing built

### 2026-04-15 (first live rebalance)
- Fixed 4 pre-rebalance bugs (yf.download race, hardcoded NAV, regime date key, ML targets)
- Rebalance ran: verdict REDUCE_SIZE 0.88 confidence, 27 orders to Alpaca
- Fixed full-liquidation 403 errors: added `close_position()` to `alpaca_broker.py`
- Built `demo_app.py`: dark/gold Streamlit app with live LLM debate, round 2 rebuttals, scenario presets
- Files: `alpaca_broker.py`, `eod_runner.py`, `pre_rebalance_checklist.py`, `main.py`, agent files, `demo_app.py`

### 2026-04-15 (demo polish + deployment prep)
- Fixed How It Works tab (inline styles), updated portfolio preset, fixed live LLM mode
- API key security: `st.secrets` on Streamlit Cloud, `.env` locally, both gitignored
- Added `.streamlit/config.toml`, `secrets.toml.template`, updated `requirements.txt`

### 2026-04-16 (system upgrade planning + Plans A–D partial)
- Diagnosed portfolio lagging SPY: 37% EM+commodity, stale regime, noise-only self-improve
- Specced Plans A–D; committed 64 previously-untracked files; test baseline 110 passing
- **A1 ✅** SPY benchmark in PnL log; **A2 ✅** per-agent PnL routing; **A3 ✅** attribution.py; **B1 ✅** 20% EM/commodity cap
- Files: `attribution.py` (new), `forward_pnl_tracker.py`, `skill_tracker.py`, `central_intelligence.py`, `run_all_agents.py`
- Tests: 117 passing

### 2026-04-16 (vol-regime alpha sleeve)
- Added `vol_of_vol_21d`, `vol_trend_10d` features; volatility sleeve signal = -(vol_trend_10d)/(vol_of_vol_21d)
- Enabled volatility sleeve at 5%; trend 70% → 65%

### 2026-04-17 (bug hardening E1–E5, D1–D2, R1–R4)
- 11 bugs fixed across execution, debate, runner layers. Tests: 144 passing.

### 2026-04-18 (second audit + Phase 1 firm architecture)
- 6 second-audit bugs fixed; full third-pass — no new crash risks
- `walk_forward_lightweight.py` (real OOS), `debate_gate.py` (conditional debate), `counterfactual_tracker.py`
- Tests: 157 passing

### 2026-04-18/19 (self-evolving alpha loop + regime features)
- **Plan A Tasks 1–3 ✅**: stack reads live config, shadow_promoter auto-promotes, per-regime variants
- **Plan B Tasks 1–4 ✅**: credit/yield features in regime, walk_forward_lightweight multi-fold, A4 confirmed fixed
- Files: `stack.py`, `shadow_promoter.py` (new), `self_improve.py`, `regime/features.py`, `walk_forward_lightweight.py`
- Tests: 177 passing

### 2026-04-19 (fundamental alpha — Tier 1 signals)
- `fundamentals.py` ingest, `build_fundamental_panel()`, `fundamental.py` alpha sleeve
- Stack: trend 65% → 55%, fundamental 10% added; fundamentals seeded: 675 rows, 135 symbols
- Tests: 188 passing

### 2026-04-19 (PEAD earnings surprise alpha sleeve)
- `earnings.py` ingest, `build_earnings_panel()`, `earnings.py` alpha sleeve
- Stack: fundamental 10% → 5%, earnings 5% added; earnings seeded: 3,228 rows, 135 symbols
- Tests: 202 passing

### 2026-04-19 (pipeline bug fixes + integrity hardening)
- Fixed bdate_range weekend crash, RegimeEngine config type, fundamental_panel tz mismatch
- Synced self_improve DEFAULT_ALPHA_WEIGHTS with stack.py

### 2026-04-23 (universe expansion + ML fix + self-improve sync)
- **Universe**: 135 → 901 symbols (S&P 500 + S&P 400); `us_equity_universe.json`; 93% sector coverage
- **mom_skip1m**: skip-last-month feature added; trend sub-weight 0.20
- **Distressed filter**: zeroes alpha for mom_252d < −0.65
- **Debate gate**: `should_run_debate()` wired into rebalance path
- **ML sleeve**: trimmed to 6 ICIR>0.2 features; depth 4→3, estimators 200→100; CPCV now 15/15 folds, p5=−0.016, p50=+0.012 → sleeve enables
- **self_improve**: MIN_SHARPE_EDGE 0.10→0.05; per-regime block reuses current_sharpe
- Tests: 202 passing

### 2026-05-01 (self-learning hardening — sleeve floor protection + per-sleeve IC)
- **Root cause**: generate_variants() zeroed fundamental/earnings (both at 0.05) via ±0.10 perturbation; shadow_promoter wrote zeroed config with no guard
- **Fix 1**: MIN_SLEEVE_WEIGHTS floor in self_improve.py (trend 10%, fundamental/earnings 2%)
- **Fix 2**: _restore_sleeve_floors() in shadow_promoter.py as second safety net
- **Fix 3**: walk_forward_lightweight now scores fundamental+earnings alpha per fold
- **Fix 4**: _log_sleeve_ic() in main.py — daily per-sleeve IC to sleeve_ic_log.jsonl
- Updated 3 tests. Tests: 202 passing.

### 2026-05-02 (dry run debugging + 3 bug fixes)
- ML CPCV OOM: top-300 cap + X_all built once + pd.concat (was .join hangs)
- Hub _ROOT path: parents[3] → parents[2] (manifest written to wrong dir)
- Optimizer sector fallback: SectorDataError → rank-weighted fallback (CLAUDE.md constraint #4)
- max_workers restored: 1 → len(agent_tasks)
- Dry run result: 23 positions, all 8 sleeves loaded, IC t-stat=2.83, equity $105,237
- Files: `ml_sleeve.py`, `hub.py`, `optimizer.py`, `run_all_agents.py`

### 2026-05-02 (analyst revision alpha sleeve)
- `analyst.py` ingest (225k rows, 899 symbols), `build_analyst_panel()`, `analyst.py` alpha sleeve
- Stack: trend 55% → 50%, analyst 5% added; analyst added to MIN_SLEEVE_WEIGHTS + _SLEEVE_FLOORS
- ML: analyst_revision added to ML_FEATURES (7 total)
- Tests: 216 passing

### 2026-05-02 (Phase 6 — Options Flow, Insider, Short Interest signals)
- **`ascent/data/ingest/options.py`** (new) — yfinance options chain; IV skew (OTM put IV − call IV) + put/call ratio; ffill 5 days; cache: `options_flow`
- **`ascent/data/ingest/insider.py`** (new) — yfinance insider_transactions; open-market buys=+1/sells=−1; 1-bday lag; ~1–2yr backfill available; cache: `insider_transactions`
- **`ascent/data/ingest/short_interest.py`** (new) — yfinance info shortPercentOfFloat + shortRatio; ffill 15 days; cache: `short_interest`
- **`ascent/alpha/options_flow.py`** (new) — invert IV skew + PC ratio; cross-sectional z-score; bearish options activity → negative alpha
- **`ascent/alpha/insider.py`** (new) — rolling 63d net purchase score; drops all-zero symbols; cross-sectional z-score
- **`ascent/alpha/short_interest.py`** (new) — contrarian squeeze signal; cross-sectional z-score of short_pct_float
- **Stack**: trend 50% → 44%; options_flow/insider/short_interest at 2% each; 11 sleeves total
- **feature_defs**: added `build_options_panel`, `build_insider_panel`, `build_short_panel`; all wired into `build_all_features`
- **FeatureBuilder**: added options_df, insider_df, short_df params; wired into main.py
- **self_improve + shadow_promoter**: new floors (1%) added for all 3 new sleeves
- **ML_FEATURES**: iv_skew, insider_net_score, short_pct_float added (10 total)
- Tests: 254 passing (231 + 23 new)
- Open: Phase 7 autonomous research engine, Phase 8.3 hedge overlay (~May 13)

### 2026-05-02 (Phase 6 cache seeding + insider fix)
- Seeded short_interest: 900 rows/symbols (today snapshot, 357s)
- Seeded options_flow: 47/50 symbols (today snapshot, 42s)
- Diagnosed insider failure: yfinance Transaction column always empty; Text column has actual data
- Fixed `ascent/data/ingest/insider.py`: prefer Text over Transaction, skip blank columns
- Re-seeded insider_transactions: 23,166 rows, 892 symbols, date range 2013–2026 (360s)
- Commits: e7fdf3f (Phase 6), a9c899a (insider fix)

### 2026-05-02 (Phase 5 — Black-Litterman MV Optimizer)
- `ascent/portfolio/covariance.py` (new) — Ledoit-Wolf shrinkage via sklearn; sample fallback
- `ascent/portfolio/mv_optimizer.py` (new) — BL posterior + scipy SLSQP MV opt; view conf ∝ |z|/3
- `ascent/portfolio/tc_aware_optimizer.py` (new) — TC-aware trade sizing, 10bps kappa, 50bps deadband
- `apply_bl_to_latest()` in optimizer.py — replaces last-date weights with BL; preserves sector selection
- `ascent/main.py` — calls apply_bl_to_latest after sector_constrained_weighted
- Tests: 231 passing (216 + 15 new)
- Open: Phase 6 signals, Phase 8.3 hedge overlay (~May 13)
