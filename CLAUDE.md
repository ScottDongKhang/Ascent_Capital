# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio construction → walk-forward evaluation → regime modeling → 4 specialist agents → orchestration → AI PM (earned autonomy) → LLM debate → execution via Alpaca paper trading.

---

## Repository layout

```
ascent/
  config/       settings.py (Config, APIKeys, UniverseConfig, BacktestConfig),
                types.py (AgentOutput), us_equity_universe.json
  data/
    ingest/     yahoo, fred, simulated, fundamentals, earnings,
                edgar_listener, capitol_trades, options_scanner,
                sec_filings, earnings_transcripts, reddit_sentiment, google_trends
    store/      parquet_store, point_in_time, timescale.py, schema.sql
    streaming/  alpaca_stream.py (Alpaca WebSocket IEX, 901 symbols)
    validate/   altdata_validator.py (IC gate)
    universe.py, hub.py, normalize.py
  features/     build_features, feature_defs, targets
  alpha/        trend, meanrev, statarb, ml_sleeve, fundamental, earnings,
                llm_fundamental, narrative_alpha, event_alpha, altdata_alpha, stack
                [analyst, options_flow, insider, short_interest — sparse/zero-filled]
  portfolio/    optimizer, mvo_optimizer, black_litterman, regime_covariance, hedge_overlay
  backtest/     engine, costs
  research/     walk_forward_runner, walk_forward_lightweight, cpcv,
                self_improve, shadow_promoter, factor_proposer,
                factor_discovery/ (feature_templates, leakage_scanner,
                  regime_cpcv_evaluator, llm_suggester, pysr_engine, discovery_runner)
  regime/       engine, model, features, decision, integration, posture,
                breaks, particle_filter, types
  risk/         factor_data, factor_model, covariance_model,
                factor_exposure, factor_constraints, pm_risk_validator
  reporting/    market_memo, ic_brief_generator, blind_spot_detector,
                catalyst_scanner, debrief, regime_narrative, investor_report
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch,
                cost_model, slippage_tracker, approval_server,
                twap_executor, implementation_shortfall, capacity_model,
                intraday_trigger, event_runner, debate_gate
  monitoring/   skill_tracker, forward_pnl_tracker, attribution,
                slippage_ic_feedback, pre_rebalance_checklist,
                counterfactual_tracker, live_nav, alert_system
  llm/          client.py — centralized Anthropic API wrapper
  dashboard/    export_dashboard_data, live_dashboard.py (port 8502)
  strategy/     earned_authority.py, thesis_formatter.py, calibration_tracker.py

agents/         us_equities_agent, macro_agent, international_agent,
                alternatives_agent, event_agent, ai_pm_agent, red_team_agent
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker,
                disagreement_scorer, agent_tools
memory/         r2r_interface (R2R HTTP + BM25 fallback), reflection_agent,
                regime_memory
simulation/     mirofish_interface
compliance/     audit_trail, performance_report, risk_disclosure, methodology_index
docs/           methodology.md, risk_disclosures.md, superpowers/plans/, superpowers/specs/
scripts/        setup_timescaledb.sh, verify_audit_trail.py, evaluate_hedge.py

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl,
                active_alpha_config.json, shadow_configs/,
                factor_returns.parquet, factor_loadings.parquet,
                llm_fundamental_cache.json, narrative_shift_cache.json,
                earned_authority.json, ai_pm_shadow_returns.jsonl
dashboard/      HTML dashboards, regime_signal.json, regime_labels.csv,
                agent_skill_scores.json, factor_exposures.json, methodology_index.json
outputs/
  debate_log/         verdict_YYYY-MM-DD.json, agent_credibility.json
  investor_reports/   YYYY-MM.pdf monthly PDF
  factor_proposals/   autonomous factor proposals (human review)
  altdata_proposals/  validated alt-data proposals (human review)
  ai_pm_theses/       YYYY-MM-DD-thesis.json per rebalance
logs/           eod_log.jsonl, slippage_log.jsonl, self_improve_log.jsonl,
                skill_scores_log.jsonl, multi_agent_run.jsonl,
                post_debate_portfolio.jsonl, attribution_log.jsonl,
                audit_trail.jsonl, alerts.jsonl,
                regime_episodes.jsonl, ai_pm_calibration.jsonl,
                snapshots/{agent_id}_weights_YYYY-MM-DD.json

ascent/main.py        core pipeline entrypoint
run_all_agents.py     single daily command — branches on rebalance day
demo_app.py           Streamlit interactive demo
```

---

## Core runtime flow

**Command**: `python run_all_agents.py`

**Non-rebalance day**: regime memory update → calibration update → agents (parallel) → forward PnL → skill scores → orchestrator → AI PM → earned authority update → write `merged_weights.json` → log episode. Stop.

**Rebalance day**: same + pre-rebalance checklist → debate → verdict gates execution → Alpaca orders → slippage tracking.

`ascent/main.py` pipeline: data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export.

---

## Alpha stack (14 sleeves)

| Sleeve | Weight | Notes |
|--------|--------|-------|
| Trend | 38% | Cross-sectional momentum; skip-last-month `mom_252d − mom_21d` at 0.20 sub-weight |
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
| Alt Data | 0% | IC-validated alt data (SEC 10-K, transcripts, Reddit, Google Trends); 0% until IC gate passes |
| Narrative Alpha | 3% | Quarter-over-quarter thesis shift detection via Haiku; returns zeros if cache absent |

**Critical**: `DEFAULT_ALPHA_WEIGHTS` exists in BOTH `ascent/alpha/stack.py` AND `ascent/research/self_improve.py`. If adding a new sleeve, update both or integrity tests will fail.

Distressed filter: zeroes alpha for `mom_252d < −0.65` after blending. Weights are regime-adaptive via `data_cache/active_alpha_config.json`. ML sleeve disabled if <10 folds converge or p5 IC Sharpe < −0.05.

---

## Portfolio construction

`sector_constrained_weighted_mvo()` is the primary path: sector pre-screening → Black-Litterman blending (quant prior + LLM views) → cvxpy MVO (objective: `w'α - λΣ - κ‖w-w_prev‖₁`, CLARABEL solver, SCS fallback) → rank-weight fallback if infeasible.

`sector_constrained_weighted()` (rank-weight fallback): coverage check (< 80% → skip sector caps + warn) → rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative, ≤50 iterations) → hard clamp + renorm.

BL blending weight from IC IR: IR < 0.30 → tau=0.05, IR < 0.60 → tau=0.10, else tau=0.15.

Regime tightens max_weight: crisis → 0.08, calm_bull → 0.15. SPY 200MA overlay: SPY < 200MA → multiply weights × 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays.

---

## Agents

**Module names**: `agents.us_equities_agent`, `agents.macro_agent`, `agents.international_agent`, `agents.alternatives_agent` (not `agents.us_equities` etc. — the `_agent` suffix is required).

**US Equities**: 901 symbols (S&P 500 + S&P 400); full alpha stack; max 1 per sector; 12–20 positions.

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

PDBC↔KMLM correlation (~0.81) frequently triggers the correlation guard — KMLM gets halved. Expected, not a bug.

---

## AI PM Agent (`agents/ai_pm_agent.py`)

Claude Opus 4.6, 16 tools, `max_tool_calls=14`.

**4-phase research loop:**
1. Phase 1 — Market context: `get_regime_state`, `get_macro_data`
2. Phase 2 — Quant baseline: `run_quant_agent` ×4 (hits precomputed cache — no pipeline re-run)
3. Phase 3 — Signal research: up to 6 of 10 signal tools
4. Phase 4 — Submit: `propose_portfolio(weights, thesis)`

**16 tools**: `get_regime_state`, `get_macro_data`, `run_quant_agent`, `get_sec_signal`, `get_transcript_signal`, `get_attribution_history`, `get_earnings_signal`, `get_past_verdicts`, `get_factor_exposures`, `get_var_estimate`, `get_sector_concentration`, `get_position_momentum`, `get_narrative_shift`, `get_regime_memory`, `get_calibration_report`, `propose_portfolio`.

**After initial proposal — adversarial self-play:**
- `red_team_agent.py` (Sonnet) attacks proposal: per-position worst-case + systemic kill shot
- AI PM gets one revision pass (`max_tool_calls=6`) — revises or defends
- If revision produces no `propose_portfolio`, initial proposal used silently
- Red team failure → "" → skip revision, use initial (never blocks)

**Precomputed cache**: `run_ai_pm(quant_outputs=agent_outputs)` builds a cache from `AgentOutput` objects passed in. `run_quant_agent` tool hits cache first — saves ~160s and 4 pipeline re-runs per run.

**Pre-blend risk validator** (`ascent/risk/pm_risk_validator.py`): position cap 15%, sector cap 40%, distressed filter (mom_252d < −0.65), min 5 positions, no shorts. Negative-weight short-circuit fires before other checks (normalization would inflate remaining weights otherwise).

**Thesis schema** (what `propose_portfolio` receives):
- `weights`: {symbol: float}
- `thesis`: {market_view, regime_assessment, quant_baseline_summary, quant_agreement (list), quant_overrides (list of {symbol, ai_action, reason}), position_rationale (dict), key_risks (list), what_could_be_wrong}

**Calibration logging**: `_tool_propose_portfolio` calls `log_prediction(date, weights, thesis)` after each submission. Conviction derived from thesis structure: `high` = in quant_overrides, `medium` = in position_rationale only, `quant_agreed` = rest.

---

## Earned authority (`ascent/strategy/earned_authority.py`)

`PHASE_WEIGHTS = [0.0, 0.25, 0.50, 0.75]`, `HARD_CAP = 0.80`. State in `data_cache/earned_authority.json`.

Advances phase after 21 rebalance days with AI Sharpe > quant + 0.05. Auto-reverts to Phase 0 if AI 21d drawdown > quant + 5pp. Same-day dedup guard prevents double-appending returns. Shadow returns logged to `data_cache/ai_pm_shadow_returns.jsonl`.

Authority update runs BEFORE the rebalance/non-rebalance branch split in `run_all_agents.py` — runs every day.

---

## Episodic memory (`memory/regime_memory.py`)

`log_episode(date, regime, quant_weights, ai_weights=None)` → `logs/regime_episodes.jsonl`. Deduplicates by date. `query_episodes(regime, n=5)` returns last n matching episodes; uses prefix match (querying "calm" matches "calm_bull"). `update_outcomes(price_returns)` fills `realized_return_21d` for episodes ≥21 calendar days old.

AI PM tool `get_regime_memory(regime)` calls `query_episodes`. Called at startup in `run_all_agents.py`.

---

## Calibration tracker (`ascent/strategy/calibration_tracker.py`)

`log_prediction(date, portfolio, thesis)` → `logs/ai_pm_calibration.jsonl`. `update_outcomes(price_returns, as_of_date)` fills `realized_21d`. `get_calibration_report(n_rebalances=10)` computes Spearman IC between conviction order and realized return. IC ≥ 0.20 = Calibrated, ≥ 0.05 = Weak, < 0.05 = Uncalibrated. All deterministic — no LLM calls.

---

## Narrative alpha (`ascent/alpha/narrative_alpha.py`)

Reads `data_cache/llm_fundamental_cache.json` (keys: `{symbol}_{quarter_end_date}`, values: `{direction, confidence, key_trend, uncertainty}`). Finds last 2 entries per symbol. Calls Haiku to score shift from −1 to +1. Caches results to `data_cache/narrative_shift_cache.json` (keyed by md5 of content). Returns cross-sectionally z-scored Series. Zero weight in stack until cache matures.

AI PM tool `get_narrative_shift(symbol)` calls internal helpers directly.

---

## Regime system

HMM K=2–4 (best via walk-forward CV). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3d / entropy > 0.90 → uncertain. Particle filter: 500 particles SIR, reinitializes on batch refit. Emergency refit triggers: SPY −3%+VIX>30, 200MA cross, SPY/TLT corr flip, break z-score > 3.5. Refit every 5 days.

Regime propagates through: sleeve weights, max_weight cap, orchestrator base allocations, debate context, AI PM episodic memory, VIXY hedge sizing.

---

## Debate layer

Rebalance days only. Advisory — never writes to alpha/portfolio/execution.

**Sequence**: score past verdicts → debrief → blind spot detection → catalyst scan → Monte Carlo sim → run agents → judge verdict.

| Agent | Model | Role |
|-------|-------|------|
| Bull | claude-sonnet-4-6 | Strongest case for executing |
| Bear | claude-sonnet-4-6 | Case for reducing risk |
| Devil's Advocate | claude-sonnet-4-6 | Most dangerous assumption + Monte Carlo tail |
| Regime Specialist | claude-haiku-4-5-20251001 | Sizing playbook for current regime |
| Quant Sanity | Pure Python | Weight sum, max position, concentration, turnover |
| Judge | claude-sonnet-4-6 | Synthesizes verdict |

Round 2 rebuttals: bull/bear/devil respond to each other before judge synthesizes.

Verdict: `proceed` | `reduce_size` (Haiku adjusts weights) | `halt_and_review` (persists to `execution/halt_state.json`).

Debate gate (`debate_gate.py`): skipped in calm_bull if entropy=0.00 — fires only on high-uncertainty signals.

---

## Self-improve loop

Weekly (Sunday 6AM). Generates 5 sleeve-weight variants, scores via `run_lightweight_oos()`. Shadow promotion if edge > 0.05 Sharpe, 30-day monitoring, auto-promoted by `shadow_promoter.py`. Per-regime variants written to `active_alpha_config.json` `by_regime` section.

`SELF_MODIFY_ENABLED = False` — exits early until OOS Sharpe is positive for 30 consecutive trading days.

## Factor discovery loop

Monthly (first Sunday). PySR symbolic regression (Path A) + Haiku JSON template params (Path B). Both gated by Harvey FDR (IC mean ≥ 0.015, IC IR ≥ 0.60, positive in every observed regime). Proposals → `outputs/factor_proposals/` for human review.

Gate conditions (~July 2026): `SELF_MODIFY_ENABLED = True`, ≥63 days live regime labels, ≥50 slippage fills.

---

## Execution

Kill switch: SOFT_WARN 8% (log+proceed), HARD_STOP 15% (abort). Alternatives: 12%. State in `logs/kill_switch_state.json`. Approval gate removed (paper trading — all orders submit directly, no human approval needed). Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV. Alpaca orders: retry ×3, 0.4s inter-order delay, 15s timeout (avoids 429 on 29-order batches).

TWAP_ENABLED=False, EVENT_TRADING_ENABLED=False — both kill-switched pending paper validation (~July 2026).

Config: always `get_config()`, never `Config()` directly.

---

## LLM clients (`ascent/llm/client.py`)

```python
DEFAULT_MODEL = "claude-opus-4-6"     # AI PM
SONNET_MODEL  = "claude-sonnet-4-6"   # debate agents, red team, judge
HAIKU_MODEL   = "claude-haiku-4-5-20251001"  # classifiers, weight adjustment
```

Lazy singleton via `get_client()`. Retry 3× with 2s/4s backoff. Import all model constants from here — never redefine locally in other files.

Also exports `generate_structured(system_prompt, user_prompt, model, ...)` for structured JSON calls and `tool_completion(system_prompt, user_prompt, tools, tool_executor, model, max_tokens, max_tool_calls)` for tool-use loops.

---

## Data / caching

Cache names: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live/simulated`, `profiles` (sector metadata). Never hide data provenance in cache name. Point-in-time joins via `as_of_join()` / `as_of_merge()`.

---

## Integrity constraints

1. No look-ahead bias — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only.
2. No simulated data under live cache names.
3. Max-weight hard cap via `_water_fill_cap()` with post-condition check.
4. Sector constraint: < 80% coverage → skip caps + warn, never collapse to single name.
5. `walk_forward_runner.py` not a production entrypoint — retained for self_improve Phase D only.
6. Debate is advisory only — never writes to alpha, portfolio, or execution modules.
7. New alpha sleeves: add weight to BOTH `stack.py` and `self_improve.py` DEFAULT_ALPHA_WEIGHTS.

---

## Non-obvious gotchas (read before touching these areas)

- **`RegimeEngine` constructor**: takes `config=dict`, not a `Config` object — `run_all_agents.py` must convert.
- **`bdate_range(end="today")`**: returns empty on weekends — use explicit weekday rollback.
- **`apply_hedge_overlay`**: must accept both `RegimeSignal` and plain `str` — `AgentOutput.regime_signal` is a string.
- **ML sleeve cache**: must store `feature_names` — XGBoost crashes on "Feature shape mismatch" if feature set changes between cache writes.
- **AI PM `propose_portfolio`**: Anthropic tool schema keeps `weights` and `thesis` as separate top-level args — portfolio dict must be injected into thesis JSON explicitly after the call.
- **Agent module names**: `agents.us_equities_agent` not `agents.us_equities` — the `_agent` suffix is required for all four.
- **PDF generation**: use `reportlab` (pure Python) — `weasyprint` requires system GObject/Pango which isn't available.
- **`_SPARSE_FILL_ZERO`** in alpha stack: must include ALL sparse panels or NaN-drop disables those sleeves entirely.
- **Fundamental sleeve**: `high_52w_pct` was removed (it's price momentum, not accounting quality).
- **Narrative alpha cache**: keyed by md5(content), not by (symbol, dates) — safe to call without explicit date context.
- **Calibration conviction**: pure structural derivation from thesis dict — no LLM. `high`=quant_overrides, `medium`=position_rationale only, `quant_agreed`=everything else.
- **Regime memory prefix match**: querying "calm" matches "calm_bull" — intentional.
- **`run_ai_pm(quant_outputs=...)`**: pass `agent_outputs` list to skip redundant pipeline runs. Without it, AI PM re-runs all 4 agents from scratch (~160s extra).
- **Red team**: fires AFTER initial proposal, not before. Revision pass limited to `max_tool_calls=6`.
- **Authority update**: runs before the rebalance/non-rebalance branch split — runs every day regardless.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Environment

Python 3.12.13 Homebrew, venv at `.venv/`. Use `.venv/bin/python`. API keys via `APIKeys.from_env()` only. Mac Air M5, no JAMF restrictions.

---

## Current state (as of 2026-05-18)

**Portfolio** (post-rebalance May 19): EWY 10.9%, PDBC 6.9%, CBOE/CHRD/HUM/SATS/SNDK/STRL/VICR/VRT/WDC ~6.5% each, DBB 3.8%, EWT/EEM 3.4% each, EWC 3.3%, DBA 3.1%, BIL 2.8%, KMLM 2.4%, UUP 1.6%. 18 positions. NAV ~$103,790. Live since April 1, 2026.

**AI PM**: Phase 0 (`ai_weight=0.0`), shadow period started 2026-05-19. Advances to 25% after 21 rebalance days with Sharpe edge > 0.05. `data_cache/earned_authority.json` is the ground truth.

**Skill scores**: all agents warming up (need 63 days from April 1 → June 3, 2026).

**Regime**: calm_bull (refitted May 7). Regime signal in `dashboard/regime_signal.json`.

**Tests**: 506 passing, 1 skipped.

**Kill switches pending paper validation (~July 2026)**: `EVENT_TRADING_ENABLED=False`, `TWAP_ENABLED=False`, `SELF_MODIFY_ENABLED=False`.

**R2R semantic memory**: built, `R2R_API_KEY` not configured, BM25 fallback active.

**Operational next steps (not code)**: Deploy TimescaleDB (Docker), configure WebSocket (`ALPACA_KEY`), transfer real capital (~May–June 2026). YC-ready at April 2027 (12-month live track record).

---

## Build status

| Component | Status | Notes |
|-----------|--------|-------|
| Plans 1–7 | ✅ | Factor risk, MVO/BL, events, alt data, execution, real-time infra, compliance |
| AI PM Agent | ✅ | 16 tools, Opus, 4-phase loop, thesis audit trail |
| Adversarial self-play | ✅ | Red team (Sonnet) attacks proposal; AI PM revises or defends |
| Episodic memory | ✅ | Per-regime outcome log; queried by AI PM before proposing |
| Calibration tracking | ✅ | Conviction-vs-realized IC; AI PM checks own hit rate |
| Narrative alpha | ✅ | Q-o-Q thesis shift detection (Haiku); 3% weight; returns zeros if cache absent |
| Non-rebalance intelligence | ✅ | 7 daily monitors → rebalance brief → AI PM tool #17 `get_rebalance_brief` |
| LLM cost tracking | ✅ | Per-model token + cost accounting; `logs/cost_log.jsonl` per run |

---

## Session log

### 2026-04-09 — 2026-05-09 (foundation + AI-native Tiers 1–3)
- First rebalance Apr 15 (27 orders). Second May 5 (40 orders, full rotation). NAV $104,815 as of May 7.
- Key bugs: RegimeEngine takes config=dict; bdate_range empty on weekends; apply_hedge_overlay must accept str; ML sleeve cache must store feature_names.
- PDBC↔KMLM correlation (0.81) halves KMLM via correlation guard — expected recurring behavior.
- Debate is conditional circuit breaker via debate_gate.py, not a daily veto.

### 2026-05-10 (Plans 1–7 ✅)
- 420→446 tests. BL tau scales with IC IR. MVO CLARABEL → SCS fallback. weasyprint unavailable → reportlab for all PDF. TimescaleDB/WebSocket require Docker + ALPACA_KEY — all DB calls return False/empty if unavailable.

### 2026-05-16 (AI PM Agent ✅)
- pm_risk_validator: negative-weight short-circuit fires first (normalization inflates remaining weights otherwise).
- earned_authority: same-day dedup guard, PHASE_WEIGHTS=[0.0,0.25,0.50,0.75], HARD_CAP=0.80.
- ai_pm_agent: portfolio dict must be injected into thesis JSON explicitly after tool call.
- Authority update moved before rebalance/non-rebalance split — runs every day.
- 465 passing.

### 2026-05-17 (adversarial self-play + episodic memory + narrative alpha + calibration ✅)
- red_team_agent.py: Sonnet, per-position worst-case + systemic kill shot, returns "" on failure.
- Red team fires AFTER initial proposal; revision pass max_tool_calls=6; fallback to initial if no revision.
- run_ai_pm now accepts quant_outputs — builds precomputed cache, saves ~160s per run.
- Approval gate removed from eod_runner.py (was blocking every 29-order paper batch).
- regime_memory: prefix match on regime query ("calm" matches "calm_bull").
- narrative_alpha: cache key is md5(content) not (symbol, dates).
- calibration_tracker: pure structural conviction — no LLM. Both stack.py and self_improve.py updated.
- Debate agents switched from Opus → Sonnet (significant cost reduction).
- 465→492 tests (27 new this session).

### 2026-05-17 (bug hunt + fixes)
- Full system audit: alpha math, portfolio construction, AI PM, orchestrator, debate, execution, workflow.
- **CRITICAL fix**: pre_rebalance_checklist.py was checking `state.get("halted") or state.get("triggered")` — kill_switch.py writes key `"tripped"`. Kill switch was functionally disabled at checklist level. Fixed.
- **CRITICAL fix**: eod_runner.py multi-agent path still had approval gate (single-agent path was cleaned prior session, multi-agent missed). Removed. Paper trading now submits all orders directly.
- **Integrity fix**: optimizer.py (rank_weighted + sector_constrained_weighted) — after _water_fill_cap, zeroing min-weight names and renormalizing could push remaining weights above max_weight cap. Added re-cap pass after renorm.
- **Validator fix**: pm_risk_validator.py — negative weight check now runs before normalization. Mixed-sign portfolios were producing distorted position-cap violations after normalization inflated weights.
- **Data safety fix**: regime_memory.py and calibration_tracker.py rewrote log files with open("w") (truncate-then-write). Now use tempfile + os.replace for atomic writes — data loss on crash is no longer possible.
- Removed leftover reddit_sentiment import from run_all_agents.py altdata validation block (missed in prior revert).
- Added 2026-05-18 to rebalance_calendar.csv — Monday is now a rebalance day.
- 492 tests passing throughout.

### 2026-05-18 (rebalance #3 — cancelled)
- Pipeline ran: 29 orders generated (15 sells, 14 buys), submitted pre-market.
- Orders manually cancelled: FRED API was down at time of submission; user pulled back to May 5 holdings.
- Hub gracefully handles FRED outage — falls back to cached macro_live.parquet without crashing.
- AI PM Phase 0, shadow period tracking started. SNDK excluded by AI PM (3196% momentum = SanDisk/WDC merger artifact) — correct call, not reflected in live book at 0% weight.
- May 19 also a rebalance day — re-run pending FRED recovery; cache is 1 day old, safe to proceed on.

### 2026-05-19 (rebalance #3 — retry)
- FRED back up; re-run clean. 30 orders submitted (15 sells, 15 buys). NAV $103,790. 18 positions.
- Same portfolio as May 18 proposed book (CBOE/HUM/SATS/SNDK/STRL + EWY/EWT/EEM/EWC international tilt).
- Macro agent shifted calm_bull → neutral vs yesterday; minor weight changes only.
- Attribution: -0.95% vs SPY -0.67%. VRT worst (-0.263%, -5.0%). Debate gate skipped (calm_bull, entropy 0.00).
- May 18 logs fully cleaned before re-run (10 JSONL entries, 4 snapshots, AI PM thesis, shadow returns, earned_authority).

### 2026-05-20 (non-rebalance intelligence stack ✅)
- Added `_PRICING`, `_record_usage()`, `get_usage_summary()`, `log_costs()` to `ascent/llm/client.py` — per-process cost accumulation logged to `logs/cost_log.jsonl` at run end.
- Added `use_cache=True` to AI PM `tool_completion` calls (both main + revision pass) and bear/devil debate agents — prompt cache hits reduce Opus cost ~90%.
- AI PM gated to rebalance days only in `run_all_agents.py` (was running daily, burning ~$0.50–$2 per non-rebalance day).
- Built 7 non-rebalance intelligence modules in `ascent/monitoring/`: conviction_tracker, signal_health, regime_trajectory, analogue_search, position_thesis, adversarial_daily, macro_calendar.
- Built `ascent/monitoring/daily_intelligence.py` — orchestrates all 7 with independent `_safe()` wrappers, atomic write to `data_cache/daily_intelligence/YYYY-MM-DD.json`.
- Built `ascent/monitoring/rebalance_brief.py` — Haiku synthesizes 9 days of intelligence into `data_cache/rebalance_brief.json`.
- Added `get_rebalance_brief` as tool #17 in AI PM (called first in Phase 1 prompt); brief is pre-digested 9-day intelligence.
- Fixed execution order: brief generation moved BEFORE AI PM block (was after weights write — AI PM would have read stale brief from previous cycle).
- 492 → 506 tests (14 new in `tests/monitoring/`).
- Files: ascent/llm/client.py, run_all_agents.py, agents/ai_pm_agent.py, debate/agents.py, ascent/monitoring/ (7 new + orchestrator + brief), tests/monitoring/ (6 test files).
