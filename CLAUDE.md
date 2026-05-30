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

**Two-phase, AI-native architecture** (as of 2026-05-28):

- **Phase 1 — Pre-thesis** (Sonnet, before quant runs): AI reads macro, SEC filings, earnings calls, narratives, crowding signals → forms original thesis (8-15 names with written reasons) → seals via `propose_prethesis`. Tools: `PRE_THESIS_TOOLS` (16, no `run_quant_agent`). `run_ai_pm_prethesis()` → `AIPreThesis`.
- **Phase 2 — Synthesis** (Opus, after quant): receives sealed prethesis + quant validation. Quant confirms → concentrate (9-10%). Quant neutral → hold thesis weight. Quant contradicts → defend with catalyst or stand down. Quant-only finds → include if macro-fits. `run_ai_pm(quant_outputs=..., prethesis=...)` → `AIPMResult`.
- **Red team**: Sonnet attacks Phase 2 proposal. AI PM revision pass (`max_tool_calls=6`). Fallback to initial on no revision.
- **Conviction gate** (`ascent/strategy/conviction_gate.py`): `momentum_exhaustion` type requires crowding=OVERCROWDED. `data_quality` has 0.85 friction, blocked if win rate <35%. Max 2 overrides enforced by prompt.
- **Earned authority**: `PHASE_WEIGHTS=[0,0.25,0.50,0.75]`, `HARD_CAP=0.80`. Advances after 21 rebalance days with Sharpe edge >0.05. State: `data_cache/earned_authority.json`. Runs daily before rebalance split.
- **Calibration**: `logs/ai_pm_calibration.jsonl`. Spearman IC conviction-vs-realized. IC<0.05 triggers warning in synthesis prompt.
- **Crowding signal tool** (`get_crowding_signal`): momentum trajectory + short interest % of float + analyst rec drift. CLEAN/WATCH/OVERCROWDED. Required before any REDUCE.

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
- **AI PM two-phase**: `run_ai_pm_prethesis()` uses `SONNET_MODEL` (not Opus) — reads data, forms thesis. `run_ai_pm(prethesis=...)` uses `DEFAULT_MODEL` (Opus) for synthesis. Never swap these — Sonnet for breadth, Opus for judgment.
- **`propose_prethesis` vs `propose_portfolio`**: Phase 1 ends with `propose_prethesis` (stores `AIPreThesis`). Phase 2 ends with `propose_portfolio` (stores `AIPMResult`). They are different tools, different result stores.
- **Pre-thesis runs before quant agents** in `run_all_agents.py`. If it fails, `prethesis=None` → `run_ai_pm()` falls back to standard single-phase mode gracefully.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Environment

Python 3.12.13 Homebrew, venv at `.venv/`. Use `.venv/bin/python`. API keys via `APIKeys.from_env()` only. Mac Air M5, no JAMF restrictions.

---

## Current state (as of 2026-05-28)

**Portfolio** (post-rebalance May 27): 17 positions, NAV ~$110,100. Live since April 1, 2026. Rebalance #4 on May 27.

**AI PM**: Phase 0 (`ai_weight=0.0`), shadow period started 2026-05-19. 8/21 days evaluated. AI cumulative +2.75% vs quant +6.20% (underperforming — anti-momentum bias being corrected by redesign). `data_cache/earned_authority.json` is ground truth.

**Regime**: calm_bull. **Tests**: 627 passing, 1 skipped.

**GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every daily run.

**Kill switches pending paper validation (~July 2026)**: `EVENT_TRADING_ENABLED=False`, `TWAP_ENABLED=False`, `SELF_MODIFY_ENABLED=False`, `LONG_SHORT_ENABLED=False`.

**Operational next steps**: Deploy TimescaleDB, configure WebSocket, transfer real capital. YC-ready April 2027.

---

## Build status

| Component | Status | Notes |
|-----------|--------|-------|
| Plans 1–7 | ✅ | Factor risk, MVO/BL, events, alt data, execution, real-time infra, compliance |
| AI PM — two-phase AI-native | ✅ | Phase 1 Sonnet pre-thesis (before quant) + Phase 2 Opus synthesis. Quant validates AI, not vice versa. |
| Adversarial self-play | ✅ | Red team (Sonnet) attacks Phase 2 proposal; AI PM revision pass |
| Crowding signal tool | ✅ | `get_crowding_signal`: momentum trajectory + short interest + analyst drift. Gating check before any reduce. |
| Conviction gate | ✅ | `momentum_exhaustion` type (replaces valuation abuse). `data_quality` has friction. Max 2 overrides. |
| GitHub Pages dashboard | ✅ | Equity curve, drawdown, alpha charts, AI intelligence section, debate accordion. Auto-push daily. |
| Episodic + calibration memory | ✅ | Per-regime outcome log; conviction-vs-realized IC; AI PM checks own track record |
| Adversarial Intelligence | ✅ | 3-layer risk committee; ONE falsifiable change per rebalance; earned authority by intervention type |
| Non-rebalance intelligence | ✅ | 7 daily monitors → rebalance brief → AI PM tool `get_rebalance_brief` |
| Alpha signals | ✅ | Narrative alpha 3%, sector-rel-mom, HY-spread-dir, 14 sleeves total |
| Weekend pipeline | ✅ | ML retrain, scenario planning, debrief, factor discovery — feeds Monday AI PM |

---

## Session log

> Sessions before 2026-05-25 archived to `docs/session_log_archive.md`.

### 2026-04-09 — 2026-05-24 (summary)
- Rebalances: Apr 1 (initial, 29 orders), Apr 15 (#1, 27 orders), May 5 (#2, 40 orders), May 19 (#3, 30 orders).
- Built: Plans 1-7 (factor risk, MVO/BL, execution), AI PM agent, red team self-play, episodic memory, calibration tracker, narrative alpha, non-rebalance intelligence stack (7 monitors + rebalance brief), alt-data pipeline (SEC, transcripts, Trends), decision memory, conviction gate, weekend pipeline, adversarial intelligence (Phase 5.1).
- Key architectural bugs fixed: kill_switch key was `"tripped"` not `"halted"`, approval gate blocked paper trading batches, optimizer re-cap needed after renorm, atomic writes for JSONL logs.
- 420 → 627 tests over this period.

### 2026-05-25 (Adversarial Intelligence — Phase 5.1 ✅)
- Debate layer redesigned: genuine risk committee, ONE falsifiable change per rebalance, earned authority by intervention type. `adversarial_engine.py`, `adversarial_authority.py`, `adversarial_monitor.py` added. Weekend pipeline sped up from 6 hours → ~30 min (Google Trends freshness gate, SEC JSON brace scanner fix).
- 604 → 627 tests.
- 604 → 627 passed, 1 skipped (23 new tests in `tests/test_adversarial_intelligence.py`, 4 updated in `tests/test_debate_gate.py`).
- Files: `debate/adversarial_engine.py` (new), `debate/adversarial_authority.py` (new), `debate/adversarial_monitor.py` (new), `debate/judge.py`, `debate/agents.py`, `debate/debate_runner.py`, `ascent/execution/debate_gate.py`, `run_all_agents.py`, `ascent/monitoring/weekend_runner.py`, `tests/test_adversarial_intelligence.py` (new), `tests/test_debate_gate.py`.

### 2026-05-28 (daily run — monitoring only)
- Ran `run_all_agents.py` — clean exit (code 0), 43.2s, non-rebalance day.
- Forward PnL: US equities +1.86%, intl +1.10%, macro +0.47%, alt −0.27%.
- Attribution: portfolio +1.48% vs SPY +0.55% → **+0.93% alpha today**. NAV $111,334.
- Top contributors: STRL +7.8% (+0.55%), SATS +7.0% (+0.53%), EWY +4.1% (+0.38%). Top drag: WMG −2.9% (−0.20%).
- Regime: calm_bull. All 4 agents clean. 1 fetch failure (MET — cosmetic). ETF 404s on analyst estimates (expected).
- Merged weights written (13 positions): EWY 10%, APP/BWA/DOCN/HUM/NUE/ORA/STRL/VICR/WDC ~8.8% each, EWT/EEM ~4.5%, EFA 1.7%.
- No code changes this session.

### 2026-05-28 (GitHub Pages performance dashboard ✅)
- **Repo cleanup**: untracked `logs/`, `execution/`, `dashboard/*.json` with `git rm --cached`; added `dashboard/*.json`, `memory/*.jsonl`, `outputs/scenarios/` to `.gitignore` — `git status` is now clean after every daily run.
- **`scripts/generate_performance_page.py`** (new): pulls Alpaca portfolio history, SPY via yfinance, verdict files, regime_labels.csv, earned_authority.json, AI PM thesis. Generates `docs/index.html` and updates README stats table. Called at end of `run_all_agents.py` with `--push` to auto-commit and push after every run.
- **`docs/index.html`** (new): static GitHub Pages dashboard at `https://scottdongkhang.github.io/Ascent_Capital`. Full redesign includes: animated hero (NAV/return/alpha/SPY counters), regime indicator ring, equity curve vs SPY with annotation markers, drawdown from peak chart, cumulative alpha chart, AI intelligence section (earned authority phase tracker, agent allocation doughnut, latest AI PM decision card), debate accordion (5 sessions with full bull/bear/devil/judge text), holdings table, event timeline.
- **README**: live stats table (`<!-- LIVE_STATS_START/END -->` sentinels) + prominent dashboard link after badges. Auto-updated on every run.
- **Accuracy fixes**: Apr 12 verdict (garbage test entry, reasoning="ok") excluded. May 27 confirmed as Rebalance #4 (in rebalance_calendar.csv). Debate-only sessions (Apr 4/5/6) labeled correctly — no verdict badge, no position count. Sharpe (2.919) shown with ±2.5 SE caveat and walk-forward OOS reference.
- **Key gotcha**: web scraper confused Event Timeline with Debate Accordion — actual accordion has correct 5 entries (Apr 4, 5, 6, 15, May 27).
- Files: `scripts/generate_performance_page.py` (new), `docs/index.html` (new), `docs/.nojekyll` (new), `README.md`, `run_all_agents.py`, `.gitignore`.

### 2026-05-28 (AI PM redesign — amplify-first, crowding-gated ✅)
- **Root cause of underperformance diagnosed**: AI PM was -3.45% vs quant over 8 shadow days. Single cause: Day 6 (May 26, post-Memorial Day rally) where AI had cut WDC/VICR/SATS from 10% → 3-4% citing valuation/data_quality, and all three ran hard. Systematic anti-momentum bias — applying DCF logic in a momentum regime. 5 overrides per rebalance all correlated = low IC × zero effective breadth = negative IR.
- **New tool `get_crowding_signal`** (`agents/ai_pm_agent.py`): combines momentum trajectory (21d vs 252d deceleration), short interest % of float (>15% = informed bears), analyst consensus drift (rec_mean >2.5). Returns CLEAN / WATCH / OVERCROWDED per symbol. Required before any REDUCE override.
- **Rewritten `_SYSTEM_PROMPT`**: Fundamental Law framing (IR = IC × √Breadth). AMPLIFY FIRST protocol — find 1-2 names where quant + crowding=CLEAN + text signal agree, overweight to 9-10%. REDUCE PROTOCOL: max 2 per rebalance, requires crowding=OVERCROWDED + text confirmation + gate approval. EXTENDED names (>200% mom) are the quant's highest conviction, not override targets.
- **`momentum_exhaustion` override type** added — the correct way to reduce a crowded momentum name. Replaces `valuation` abuse. `data_quality` now has 0.85 friction cost (was auto-approved at 1.0) and is blocked if win rate <35% over 8+ cases.
- **`conviction_gate.py`**: added `momentum_exhaustion` to `_OVERRIDE_TYPES`. `valuation` gate now redirects to `momentum_exhaustion` in block message. `correlation_risk` and `news_event` remain structurally approved.
- **`propose_portfolio` schema updated**: added `amplify[]` list alongside `quant_overrides[]`.
- **Design principle**: quant is the prior, AI PM is the Bayesian update — only overrides on genuine orthogonal information (text, crowding, coherence). Never on valuation opinion.
- 627 passing, 1 skipped (unchanged).
- Files: `agents/ai_pm_agent.py`, `ascent/strategy/conviction_gate.py`, `tests/test_conviction_gate.py`.

### 2026-05-28 (two-phase AI-native architecture ✅)
- **Root cause of AI PM underperformance**: -3.45% vs quant over 8 shadow days. Single cause: Day 6 (May 26) where WDC/VICR/SATS were cut from 10%→3-4% citing `data_quality`, then rallied hard. Systematic anti-momentum bias = low IC × zero breadth = negative IR.
- **Two-phase AI PM**: `run_ai_pm_prethesis()` (Sonnet, max 10 tools, before quant) reads macro/SEC/earnings/narratives, forms original thesis with 8-15 named positions and written reasons, seals via `propose_prethesis`. `run_ai_pm(prethesis=...)` (Opus, synthesis) receives sealed thesis + quant validation. Quant confirms → concentrate. Quant neutral → hold. Quant contradicts → defend or stand down. AI is the alpha source; quant is the validator.
- **New tool `get_crowding_signal`**: momentum trajectory (21d vs 252d decel) + short interest % of float + analyst rec drift. CLEAN/WATCH/OVERCROWDED. Required before any REDUCE.
- **`momentum_exhaustion` override type**: requires crowding=OVERCROWDED + text signal. `data_quality` has 0.85 friction (no longer auto-approved). Max 2 overrides hard-capped in prompt.
- **Cost**: Phase 1 Sonnet (~$0.06) + Phase 2 Opus (~$0.34) = ~$0.40/rebalance vs ~$0.27 before (+48%). ~$3/yr extra at 26 rebalances.
- **GitHub Pages dashboard**: `https://scottdongkhang.github.io/Ascent_Capital`. Hero with animated counters, equity curve vs SPY, drawdown chart, cumulative alpha chart, AI intelligence section (earned authority, allocation doughnut, AI PM decision card, debate accordion with 5 sessions). Auto-updates after every daily run. README has live stats table.
- 627 passing, 1 skipped.
- Files: `agents/ai_pm_agent.py`, `run_all_agents.py`, `ascent/strategy/conviction_gate.py`, `tests/test_conviction_gate.py`, `scripts/generate_performance_page.py`, `docs/index.html`, `README.md`, `.gitignore`.

### 2026-05-29 (alpha performance fixes + authority comparison fix ✅)

**Daily run:** portfolio -0.69% vs SPY +0.25% (NAV $110,173). Non-rebalance day. Macro/alternatives early-zeroed by orchestrator (negative Sharpe ≥21 days).

**Fix 1 — Early-zero exemption for defensive agents** (`orchestrator/central_intelligence.py`):
- Macro and alternatives are designed to underperform in calm_bull — that's their job, not a failure signal.
- Added `DEFENSIVE_AGENTS = {"macro", "alternatives"}` exempt from early-zero unless regime is stressed/crisis.
- Recovers automatically when regime changes.

**Fix 2 — VIX confirmation for stressed regime exposure cut** (`ascent/regime/engine.py`):
- HMM calls "stressed" on price momentum alone — fired during April VIX-calm relief rally, triggered 35% exposure cut while SPY ran +6.8%.
- Added `_apply_vix_confirmation()`: post-processes signal cache, restores `risk_multiplier=1.0` when regime=stressed but VIX < 20. Label kept (sleeve weighting still defensive), only gross exposure restored.
- `VIX_STRESSED_CONFIRMATION = 20.0` exported so other modules stay in sync.

**Fix 3 — VIX confirmation for SPY 200MA overlay** (`ascent/main.py`):
- 200MA overlay imported `VIX_STRESSED_CONFIRMATION` from engine.py (single source of truth).
- Now requires VIX > 20 to confirm before applying 30% exposure cut. `fillna(25.0)` so NaN VIX dates stay conservative (cut can still fire).

**Fix 4 — Fundamental sleeve: regime-conditional weights + IC gate** (`ascent/alpha/stack.py`):
- Fundamental had IC=-0.0078, t=-4.63 — value factors systematically underperform momentum bull markets.
- `DEFAULT_ALPHA_WEIGHTS_BY_REGIME`: fundamental=0% in calm_bull/euphoric (→trend 0.43%), 8% in stressed/crisis (→trend 0.33%).
- `_get_gated_weights()`: reads last 5 unique-date entries from `sleeve_ic_log.jsonl`, zeroes any sleeve with rolling mean IC < -0.010, redistributes to trend.
- `_load_active_alpha_weights()` priority: `active_alpha_config.json` by_regime → `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` → config global → flat default.
- Calm_bull allocation was already 70% US equity from prior session (commit `3f400c1`).

**Fix 5 — Earned authority comparison fixed** (`ascent/strategy/earned_authority.py`, `run_all_agents.py`):
- Prior comparison was broken: `ai_ret` = daily return of stale multi-asset AI PM portfolio; `quant_ret` = US equities PnL log only. Wrong universe, wrong frequency.
- Now: on each rebalance day, snapshot both AI PM portfolio and full quant merged portfolio. On the next rebalance, compute holding-period return for both over the same symbols and same window. Apples to apples.
- `ADVANCE_WINDOW`: 21 daily returns → 10 rebalance periods (~5 months). Annualization: sqrt(252) → sqrt(26).
- Stale daily return buffers cleared. Old AI PM scorecard (−3.45%) was invalid — two-phase redesign hasn't run a rebalance yet.
- `data_cache/authority_rebalance_snapshot.json` saves each rebalance's portfolios for the next comparison.

**R2R memory status:** NOT deployed. `memory/r2r_interface.py` falls back to BM25 keyword search over `outputs/debate_log/` (no `R2R_API_KEY` set). What IS running: regime episode log, reflection agent (`reflections.jsonl`), decision memory (AI PM override outcomes).

**self_improve status:** Disabled (`SELF_MODIFY_ENABLED=False`), no launchd agent registered, last ran April 19. Has never promoted a config to production. Not AI-native (random perturbation). Won't enable until 30 consecutive days positive OOS Sharpe.

- 629 → 636 tests. Commits: `dc7238f`, `32ea69d`, `2a63d08`, `9311035`, `613c0d5`.
- Files: `orchestrator/central_intelligence.py`, `ascent/regime/engine.py`, `ascent/main.py`, `ascent/alpha/stack.py`, `ascent/strategy/earned_authority.py`, `run_all_agents.py`, `tests/test_regime_features.py`, `tests/alpha/test_fundamental_alpha.py`, `tests/test_ai_pm_agent.py`, `tests/test_self_evolving_alpha.py`.
