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
| Trend | 43% | Cross-sectional momentum; skip-last-month `mom_252d − mom_21d` at 0.20 sub-weight |
| Stat-arb | 15% | Sector residuals; needs profiles.parquet |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays; 6 features by IC/IR; p5 guard > −0.05 |
| Mean reversion | 5% | Short-term reversal |
| Volatility | 5% | `−(vol_trend_10d / vol_of_vol_21d)`; long names with declining + stable vol |
| Fundamental | 0% | **Disabled** — IC-t=−4.75 across 31 live days (anti-signal). Re-enable only if IC-t turns positive. |
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
- **Cost**: Phase 1 Sonnet (~$0.06)c + Phase 2 Opus (~$0.34) = ~$0.40/rebalance vs ~$0.27 before (+48%). ~$3/yr extra at 26 rebalances.
- **GitHub Pages dashboard**: `https://scottdongkhang.github.io/Ascent_Capital`. Hero with animated counters, equity curve vs SPY, drawdown chart, cumulative alpha chart, AI intelligence section (earned authority, allocation doughnut, AI PM decision card, debate accordion with 5 sessions). Auto-updates after every daily run. README has live stats table.
- 627 passing, 1 skipped.
- Files: `agents/ai_pm_agent.py`, `run_all_agents.py`, `ascent/strategy/conviction_gate.py`, `tests/test_conviction_gate.py`, `scripts/generate_performance_page.py`, `docs/index.html`, `README.md`, `.gitignore`.

### 2026-06-01 (causal intelligence for the AI PM — Phases A–D ✅)
- **Spec**: `docs/superpowers/specs/2026-06-01-causal-intelligence-ai-pm.md` — four-phase build to give AI PM structural causal reasoning. Approved and implemented in full.
- **Phase A** — Foundation: `ascent/causal/` module created. `velocity.py` (pure Python velocity score). `causal_discovery.py` (PC algorithm via `causal-learn` on FRED + sector ETF returns → `data_cache/macro_causal_dag.json`). `dag_builder.py` (Haiku per-symbol causal graph builder; cache by `(symbol, quarter_end)` to `data_cache/causal_graphs/`). `CausalMechanism` dataclass added to `ascent/config/types.py`. `AIPreThesis.causal_mechanisms` field added.
- **Phase B** — Gates 1+2: `ascent/causal/compatibility.py` — static regime-to-mechanism-type dict. Gate 1 (regime-causal compatibility) and Gate 2 (priced_in filter) wired into `run_ai_pm_prethesis()`. `get_causal_graph` tool added to `PRE_THESIS_TOOLS`. Velocity-ranked causal context injected into Phase 1 user prompt. Causal mechanisms assembled from cached graphs after `propose_prethesis`.
- **Phase C** — Tracker + Gate 4: `ascent/causal/tracker.py` — `write_predictions()`, `check_outcomes()`, `check_early_exits()`, `get_track_record()`. Gate 4 early exit wired into non-rebalance daily path in `run_all_agents.py` — symbols with broken mechanisms logged to `ai_pm_shadow_returns.jsonl` with `causal_mechanism_broken` reason. `causal_track_record` passed to `run_ai_pm()` and injected into Phase 2 synthesis prompt. Weekend runner gains `causal_macro_dag` + `causal_graph_builder` jobs.
- **Phase D** — Debate: devil's advocate receives `causal_mechanisms` from `portfolio_state`; system prompt gains CAUSAL MECHANISM ATTACK instruction to explicitly attack broken theses.
- **`causal-learn` package installed** in `.venv`.
- 675 → 708 tests (+33 this session).
- Files: `ascent/config/types.py`, `ascent/causal/__init__.py`, `ascent/causal/velocity.py`, `ascent/causal/causal_discovery.py`, `ascent/causal/dag_builder.py`, `ascent/causal/compatibility.py`, `ascent/causal/tracker.py`, `agents/ai_pm_agent.py`, `ascent/monitoring/weekend_runner.py`, `run_all_agents.py`, `debate/agents.py`, 7 new test files.
- `run_all_agents.py`: `portfolio_state` gains `causal_mechanisms` key populated from `_ai_prethesis.causal_mechanisms` — devil's advocate now receives live mechanisms on rebalance days. Test added to verify end-to-end.

### 2026-05-31 (anti-hallucination hardening ✅)
- **Problem**: LLMs in `llm_fundamental`, `narrative_alpha`, and debate agents could fabricate financial numbers not present in their input context.
- **`ascent/llm/client.py`**: `generate_structured` gains `json_schema` param; `chat_completion` gains `output_config` param. When `json_schema` is passed, Anthropic API enforces response structure at the wire level via `output_config.format`.
- **`ascent/alpha/llm_fundamental.py`**: Strengthened system prompt with amnesia instruction ("treat yourself as having amnesia about all individual companies — base analysis ONLY on numerical data provided"). Added `_LLM_FUNDAMENTAL_SCHEMA` with `quoted_evidence` field — forces model to copy actual numbers from the metrics table into the cache for auditability. Scoring logic unchanged.
- **`ascent/alpha/narrative_alpha.py`**: System prompt updated with grounding instruction ("reason ONLY from the two summaries provided — do not use any outside knowledge from your training data"). Added `_NARRATIVE_SHIFT_SCHEMA` enforced via API.
- **`debate/agents.py`**: Added `_EVIDENCE_RULE` module constant. Appended to bull, bear (primary + fallback), and devil's advocate system prompts — requires agents to tag every cited number with `[FROM CONTEXT]` and prohibits inventing values not in the provided portfolio context.
- 627 → 675 tests (+12 this session).
- Files: `ascent/llm/client.py`, `ascent/alpha/llm_fundamental.py`, `ascent/alpha/narrative_alpha.py`, `debate/agents.py`, `tests/test_llm_client.py` (new), `tests/test_llm_fundamental_alpha.py`, `tests/test_narrative_alpha.py`, `tests/test_debate_agents.py` (new).

### 2026-06-02 (regime detection hardening ✅)
- **Root cause diagnosed**: April 2026 Liberation Day crash — HMM frozen in "stressed" with entropy 6e-11, never escalated to crisis. Portfolio held full exposure through ~10% SPY drawdown with only 0.65× multiplier.
- **Three fixes shipped:**
  - (A) Hard crisis override: VIX > 30 AND SPY 5d < −7% → force `label=crisis`, `risk_multiplier=0.40`. Adds `crisis_override` column for audit. (`engine.py`)
  - (C) Asymmetric hysteresis: downgrade threshold 0.40 (sensitive to worsening), upgrade threshold 0.70 (skeptical of recovery). Severity ordering `{calm_bull:0, euphoric:0, uncertain:1, stressed:2, crisis:3}`. (`decision.py`, `types.py`)
  - (D) Entropy overconfidence penalty: entropy < 1e-6 → `risk_mult × 0.90`. Warns in logs. (`decision.py`)
- 716 → 728 tests (+12 in `tests/test_regime_hardening.py`). Zero regressions.
- Files: `ascent/regime/engine.py`, `ascent/regime/decision.py`, `ascent/regime/types.py`, `tests/test_regime_hardening.py` (new), `docs/superpowers/specs/2026-06-02-regime-hardening-design.md` (new), `docs/superpowers/plans/2026-06-02-regime-hardening.md` (new).

### 2026-06-03 (WF OOS framework + attribution + signal fixes ✅)
- **Built**: production Walk-Forward OOS framework at `ascent/research/wf_framework/` — `WindowGenerator` (purge/embargo), `BaseStrategy` ABC + `MACrossStrategy`, `ExecutionModel` (ATR slippage/commission/borrow), `ParameterOptimizer` (grid search IS-only), `PerformanceAnalyzer` (Sharpe/Sortino/MDD/WFE), `WalkForwardEngine` orchestrator. 35 tests, all passing.
- **Live attribution (Apr 1–May 29)**: Portfolio +11.0% vs SPY +15.4% (-4.5pp). Biggest drag: Apr 1-14 initial defensive portfolio during bull rally (-6pp). AI layers (debate, AI PM) had ~0pp measurable impact. 100% of performance is quant alpha stack.
- **KMLM over-weighting root cause fixed**: `managed_futures` bucket added to `FACTOR_BUCKETS` (KMLM, DBMF, CTA, WTMF). Included in `EM_COMMODITY_BUCKETS` for the 20% aggregate cap. Final position cap added at end of `run_orchestrator` — correlation guard and thesis coherence were pushing names past 10% without re-capping. KMLM: 11.2% → ~3% on May 5 data.
- **Fundamental sleeve disabled**: IC-t = -4.75 across 31 live trading days — reliably anti-signal. Zeroed in `DEFAULT_ALPHA_WEIGHTS` (0.05→0.00) and all `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` entries. Trend absorbs freed weight (0.38→0.43 base, 0.35→0.40 stressed, 0.30→0.35 crisis). Fundamental removed from `MIN_SLEEVE_WEIGHTS` floor in `self_improve.py`.
- **IC gate tightened**: -0.010 → -0.005 to catch fundamental's recent IC of -0.008 which had evaded the old threshold.
- **AI PM shadow corruption fixed**: Removed -145% entry (2026-05-17, bad price data). Added `clip(-0.50, 0.50)` per-symbol guard to authority update to prevent recurrence. AI PM stays at Phase 0 — 9 clean shadow days, need 21 before earned authority advances.
- 728 → 752 tests (+24 stale assertion updates across test_earnings_alpha, test_fundamental_alpha, test_self_evolving_alpha).
- Files: `orchestrator/central_intelligence.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `data_cache/ai_pm_shadow_returns.jsonl`, `tests/test_earnings_alpha.py`, `tests/test_fundamental_alpha.py`, `tests/test_self_evolving_alpha.py`.

### 2026-06-03 (holdings day_return bug fix ✅)
- **Bug**: `_log_holdings` computed `day_return` from `run_attribution()` (yfinance intraday prices at 1:45 PM), not from Alpaca. Today this reported +0.71% while the actual account dropped -2.83% — HUM crashed after pipeline ran (earnings after close, -27.63%).
- **Root cause**: attribution uses `yfinance.download(period="5d").pct_change().iloc[-1]` which is the 1:45 PM intraday price vs previous close. Any after-hours or late-session move is invisible to it.
- **Fix**: `day_ret` in `_log_holdings` now comes from `(equity - last_equity) / last_equity` using Alpaca's own `last_equity` field (previous session close). `run_attribution()` still runs for position-level breakdown but its `portfolio_return` is no longer used as the headline number. Comment added to `attribution_log` field noting it is an intraday estimate.
- **Actual Jun 3 numbers** (from Alpaca): account -2.83%, SPY +0.14%, alpha -2.97%. HUM cost -$3,007 on earnings crash.
- **Real total return Apr 1 → Jun 3**: +8.8% (portfolio) vs +15.9% (SPY). Early April defensive positioning is the main drag.
- Files: `run_all_agents.py`, `ascent/monitoring/attribution.py`.

### 2026-06-03 (daily run — monitoring only)
- Ran `run_all_agents.py` — clean exit (code 0), non-rebalance day.
- Regime: calm_bull. HMM refit triggered (stale signal) → K=3, calm_bull confirmed.
- Entropy overconfidence penalty fired on 10 recent dates (entropy 8e-9 to 5e-7) — regime hardening working as designed.
- Forward PnL: US equities +0.28%, alternatives +0.21%, macro −0.40%, international −0.65%.
- Attribution: portfolio +0.71% vs SPY −0.70% → **+1.41% alpha today**. NAV $109,818.
- Top contributor: STRL +9.3% (+0.75%). Top drag: WMG −4.5% (−0.31%).
- Final portfolio: 19 positions — EWY 10%, APP/BRKR/BWA/CLF/ORA/PANW/STRL/VICR/WDC 7.07% each, PDBC 5.86%, EM/macro ETFs tail. KMLM halved to 0.54% (corr guard vs PDBC, expected).
- Alerts: ⚠ WDC event in 2 days — binary risk on 7% held position.
- Non-fatal warnings: FactorModel 0-row overlap → diagonal covariance proxy; fundamental IC gate zeroed (IC=-0.0078); LLM fundamental missing columns; ETF 404s on fundamentals (all expected).
- Cost: $0.082 (51 Haiku calls). Dashboard pushed to GitHub Pages.
- No code changes this session.

### 2026-06-04 (AI PM Progressive Authority System ✅)
- **Spec**: `docs/superpowers/specs/2026-06-04-ai-pm-authority-design.md` — 44 integrity constraints, 5-level career ladder (Shadow→Analyst→Associate→Manager→Director→CEO), research-backed short-selling framework (5 signals: accruals/Sloan 1996, PEAD/Bernard 1989, QMJ/AQR, short interest/Stambaugh 2012, narrative breakdown). Valuation shorts explicitly banned.
- **`ascent/strategy/earned_authority.py`** (rewrite): 5-level state machine, Sortino-based promotion/demotion (not Sharpe), catastrophic/hard/soft demotion tiers, 5-day cooldown, 63-day stuck alert, legacy `phase`→`level` migration. PHASE_WEIGHTS alias preserved.
- **`ascent/strategy/ai_pm_guardrails.py`** (new): per-level weight/type/correlation/TE guardrails. Shorts banned at L1-2, LONG_SHORT_ENABLED gate, no contradictory long+short on same name, valuation short detection (`is_valuation_short()`), conviction inflation cap (>40% high → downgrade).
- **`ascent/monitoring/ai_pm_counterfactual.py`** (new): idempotent Track A★/A/D rebalance snapshots, daily Track A★/A/B/C/D scoring, cumulative report. Signed weight support for long-short mode.
- **`ascent/strategy/ai_pm_perf_feedback.py`** (new): daily Python learning brief (zero LLM cost). Sortino, hit rate, profit factor, fade detection, all 7 promotion gates with confidence labels, short-position incremental alpha, stuck alert. Scored at 5d/10d/21d/63d.
- **`run_all_agents.py`**: Track A★ snapshot before Phase 1, Track A after Phase 1, Track D after Phase 2, decision log per rebalance, smart Opus trigger (crisis always + 4 conditions), Haiku daily view on non-rebalance days, counterfactual scoring + feedback in `_log_holdings()`, `update_authority()` now passes Track D/A★ returns.
- **`agents/ai_pm_agent.py`**: `_build_temporal_context()` injected into every Phase 1+2 prompt, `_strip_prethesis_for_phase2()` strips freeform prose, sector thesis required field + non-price source requirement, `model_override` param on `run_ai_pm()`.
- **`scripts/generate_performance_page.py`**: 3 new loaders, rewrote `_earned_authority_html()` for 5-level ladder, added `_promotion_gates_html()`, `_counterfactual_chart_html()` (A★/B/C/D Chart.js), `_override_scorecard_html()`.
- **Bootstrap**: `data_cache/earned_authority.json` → level=1, ai_weight=5%. Day 1 evaluation begins 2026-06-04. Promotion to Level 2 after 21 days if all 7 gates clear.
- **Tests**: 752 → 777 passing (+25 new). Test failure in `test_fundamental_alpha::test_stressed_keeps_fundamental` pre-exists from other terminal's uncommitted changes (passes in isolation).
- Files: `ascent/strategy/earned_authority.py`, `ascent/strategy/ai_pm_guardrails.py` (new), `ascent/monitoring/ai_pm_counterfactual.py` (new), `ascent/strategy/ai_pm_perf_feedback.py` (new), `run_all_agents.py`, `agents/ai_pm_agent.py`, `scripts/generate_performance_page.py`, `tests/test_ai_pm_authority.py` (new), `tests/test_ai_pm_counterfactual.py` (new), `tests/test_ai_pm_perf_feedback.py` (new), `tests/test_ai_pm_agent.py` (updated).
- Open: Phase 1 accuracy tracking (regime call scoring vs actual 10d) needs 10 trading days of data. Level 4+ architecture flip (AI PM proposes, quant validates) deferred until Level 4 reached. `disable_sleeve_priors` flag enforcement in quant pipeline. AI PM shorts active only after `LONG_SHORT_ENABLED=True`.

### 2026-06-01 (monthly investor letter auto-generation ✅)
- **`ascent/reporting/investor_letter.py`** (new): auto-generates the Ascent Capital investor letter on the first trading day of each month after `run_all_agents.py` completes.
- Detection: `is_first_trading_day_of_month()` — compares today's month to the prior weekday's month. Handles Mon-after-weekend correctly.
- Data pipeline: reconstructs monthly + ITD returns from `multi_agent_run.jsonl` weights × `prices_live.parquet` daily returns. Computes max drawdown, annualized vol, beta (OLS), attribution (symbol-level weight × return). Reads debate verdicts, regime signal, kill switch state.
- Letter generation: Sonnet call with the full Ascent Capital template (voice rules, banned words, exact section structure). Saves to `outputs/investor_reports/YYYY-MM-investor-letter.md`.
- `run_all_agents.py`: letter generation wired in after cost logging, before `[Runner] Done.` — wrapped in try/except so failures never block the daily run.
- No new tests (data functions are thin wrappers; LLM call is integration-only).
