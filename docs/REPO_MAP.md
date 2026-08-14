# REPO_MAP.md

Navigation index for an AI coding agent working in this repo: where things live and what to grep for.
It is a pointer index, not a spec. Behaviour, constraints, and current state live in `CLAUDE.md`; performance figures live in `CURRENT_VERIFIED_NUMBERS.md`.
`scripts/verify_docs.py` is the drift guard that makes the claims in `CLAUDE.md` machine-checkable. Run it after structural changes.

Ignore `.venv/` and `.claude/worktrees/` when searching. The worktrees are stale copies and will double every grep hit.

---

## 1. Entrypoints

| Command | File | Does | Safe? |
|---|---|---|---|
| `.venv/bin/python run_all_agents.py` | `run_all_agents.py` (`main`, line ~629) | Daily driver. Weekend branch, startup validation, data hub, 4 quant agents, orchestrator, AI PM two-phase, debate, EOD execution. Branches on rebalance day. | **Submits real Alpaca paper orders** on rebalance days via `run_eod_with_weights` |
| `.venv/bin/python ascent/main.py` | `ascent/main.py` (`run_pipeline` line 305, `main` line 918) | Single-universe research pipeline: data → features → alpha stack → portfolio → backtest → regime. Returns a 10-tuple. Called by each quant agent. | Read-only (writes caches + `dashboard/` json) |
| `.venv/bin/python scripts/run_ascent_wf.py [--smoke]` | `scripts/run_ascent_wf.py` (`main` line 193, `_run_one` line 136) | Walk-forward OOS harness over `ascent/research/wf_framework/`. Flags: `--smoke --simplified --compare --high-sharpe --multi-asset --long-short --compare-all --live-system`. Writes `outputs/wf_results/`. | Read-only, slow |
| `.venv/bin/python ascent/research/walk_forward_runner.py` | `walk_forward_pipeline` (line 44) | Lighter walk-forward used inside the pipeline (point-in-time slicing via `_pit_slice` / `_pit_macro`). | Read-only |
| `.venv/bin/python scripts/generate_performance_page.py [--push]` | `scripts/generate_performance_page.py` (`main` line 1833) | Builds `docs/index.html` (GitHub Pages dashboard). `--push` commits and pushes. Spawned by `run_all_agents.py` line ~2212. | Read-only unless `--push` |
| `.venv/bin/python ascent/dashboard/live_dashboard.py` | `ascent/dashboard/live_dashboard.py` | Streamlit-style live view reading `logs/` + `execution/merged_weights.json`. | Read-only |
| `.venv/bin/python scripts/verify_docs.py [--quiet\|--json]` | `scripts/verify_docs.py` | 24 assertions defending `CLAUDE.md` / `CURRENT_VERIFIED_NUMBERS.md` claims (model constants, alpha weight agreement, kill switches off, 10-tuple, no loguru, judge authority cap, ...). Exit 1 on any FAIL. | Read-only. Run this first when unsure if a doc claim is stale |
| `bash scripts/run_eod.sh` | wrapper | launchd EOD wrapper around the daily run (`com.ascentcapital.eod.plist`). | **Submits orders** |
| `.venv/bin/python scripts/heartbeat_check.py` | `scripts/heartbeat_check.py` | Liveness check against `logs/eod_log.jsonl` + `rebalance_calendar.csv` (`com.ascentcapital.heartbeat.plist`). | Read-only |

Order submission funnels through `ascent/execution/eod_runner.py` (`run_eod`, `run_eod_with_weights`) into `ascent/execution/alpaca_broker.py` (`submit_order`). Grep `submit_order` to find every live-money path.

---

## 2. Package responsibilities

| Package | Owns | Key symbols to grep |
|---|---|---|
| `ascent/config` | Typed config + API keys. Never `Config()` directly. | `get_config`, `Config`, `APIKeys`, `BacktestConfig`, `RegimeConfig`, `AgentOutput`, `CausalMechanism` |
| `ascent/data/ingest` | One module per vendor/source (yahoo, tiingo, polygon, fred, sec_filings, earnings_transcripts, cboe_options, cftc_positioning, google_trends, insider, short_interest, analyst, famafrench_factors, reddit_sentiment, capitol_trades, edgar_listener, simulated). | `fetch_daily_bars`, `fetch_universe_daily`, `fetch_all_macro`, `update_sec_signals`, `update_transcript_signals`, `update_cot_cache`, `run_supplementary_ingest` |
| `ascent/data/store` | Parquet cache + point-in-time joins + optional TimescaleDB. | `save_parquet`, `load_parquet`, `validate_cache`, `_calendar_day_key`, `as_of_join`, `as_of_merge` |
| `ascent/data` (top) | Parallel fetch hub, universe membership. | `run_hub`, `hub_is_fresh`, `get_universe_on_date`, `build_historical_universe`, `get_removed_symbols` |
| `ascent/data/normalize`, `/validate`, `/streaming` | Schema normalization, alt-data IC validation, Alpaca websocket. | `normalize_prices`, `pivot_prices`, `validate_altdata_source`, `start_stream` |
| `ascent/features` | Feature panel construction and forward-return targets. | `FeatureBuilder`, `momentum_rank`, `rolling_volatility`, `zscore`, `forward_return`, `build_targets` |
| `ascent/alpha` | Active sleeves (meanrev, statarb) + the blending stack and IC gate. | `build_alpha_stack`, `alpha_to_ranks`, `DEFAULT_ALPHA_WEIGHTS`, `IC_GATE_THRESHOLD`, `meanrev_alpha`, `statarb`, `build_ml_alpha`, `_SPARSE_FILL_ZERO`, `SleeveMetaLearner` |
| `ascent/portfolio` | Weight construction, caps, overlays, optimizers. | `enforce_constraints`, `_water_fill_cap`, `enforce_cluster_cap`, `enforce_risk_budget_cap`, `sector_constrained_weighted_mvo`, `SectorDataError`, `apply_exposure_overlays`, `ma_filter_scale`, `vol_target_scale`, `optimize_mvo`, `black_litterman_views`, `tc_aware_weights`, `build_long_short_weights` |
| `ascent/risk` | Factor model, covariance, exposure bounds, correlation guard, PM validator. | `compute_factor_loadings`, `compute_portfolio_exposures`, `check_factor_bounds`, `build_factor_constraints`, `check_cross_agent_correlation`, `apply_correlation_adjustments`, `validate` (pm_risk_validator), `classify_regime` |
| `ascent/regime` | HMM/Markov regime engine, features, posture, structural breaks. | `RegimeEngine` (**takes `config=dict`**), `RegimeModel`, `RegimeSignal`, `RegimeLabel`, `RegimeFeatureBuilder`, `RegimeDecisionEngine`, `BreakDetector`, `compute_posture_from_regime`, `regime_scale_weights`, `regime_max_weight`, `check_emergency_refit_triggers` |
| `ascent/backtest` | Vector backtest engine + cost model. | `BacktestEngine`, `BacktestResult`, `estimate_trade_cost`, `estimate_rebalance_costs` |
| `ascent/research` | Walk-forward frameworks, CPCV, self-improvement, factor discovery. | `WalkForwardEngine`, `AscentPortfolioStrategy`, `FullOrchestrationStrategy`, `WindowGenerator`, `PerformanceAnalyzer`, `FoldResult`, `ParameterOptimizer`, `CPCVSplitter`, `run_self_improve`, `run_shadow_promotion`, `run_factor_discovery`, `scan_for_leakage`, `SELF_MODIFY_ENABLED` |
| `ascent/execution` | Broker, order sizing, EOD orchestration, kill switch, TWAP, slippage. | `run_eod`, `run_eod_with_weights`, `_enforce_reduce_size`, `submit_order`, `get_portfolio_history`, `compute_orders`, `Order`, `KillSwitchTriggered`, `check` (kill_switch), `should_use_twap`, `TWAP_ENABLED`, `EVENT_TRADING_ENABLED`, `should_run_debate`, `compute_slippage`, `compute_is` |
| `ascent/monitoring` | Post-trade tracking, counterfactuals, alerts, briefs, weekend/weekly cycles. | `run_forward_pnl_cycle`, `export_skill_scores`, `score_daily`, `snapshot_quant_star`, `check_alerts`, `run_checklist`, `generate_rebalance_brief`, `run_daily_intelligence`, `compute_position_health`, `run_attribution`, `compute_signal_health`, `run_exit_alerts`, `run_weekend`, `run_weekly_debrief`, `run_scenario_planning`, `build_quant_context` |
| `ascent/strategy` | AI PM scaffolding: authority, guardrails, calibration, falsifiers, discovery. | `update_authority`, `blend`, `get_state`, `LEVEL_WEIGHTS`, `apply_guardrails`, `evaluate` (conviction_gate), `build_registry`, `check_all`, `add_judge_falsifier`, `log_prediction`, `get_calibration_report`, `run_discovery`, `format_thesis`, `compute_feedback` |
| `ascent/llm` | Single Anthropic wrapper. Import model constants from here. | `DEFAULT_MODEL`, `SONNET_MODEL`, `HAIKU_MODEL`, `extract_text`, `chat_completion`, `generate_structured`, `extended_thinking_completion`, `tool_completion`, `_MIN_TOKENS_WITH_THINKING`, `_FINAL_TURN_NUDGE`, `log_costs` |
| `ascent/causal` | Macro DAG discovery, mechanism tracking, regime compatibility. | `discover_macro_dag`, `run_pc`, `build_graph`, `load_or_build`, `write_predictions`, `check_outcomes`, `regime_compatible`, `mechanism_velocity_score` |
| `ascent/reporting` | Debriefs, blind spots, investor letter/report, verified-number loader. | `write_debrief`, `detect_blind_spots`, `generate_monthly_letter`, `generate_monthly_report`, `scan_catalysts`, `load_wf_report`, `canonical_wf`, `MissingArtifact` |
| `ascent/integrations` | External non-market services. | `MiroFishClient`, `get_mirofish_sentiment`, `fetch_news` (exa), `fetch_symbol` / `get_live_macro` / `get_cot_snapshot` (openbb), `get_sentiment` (stocktwits), `find_analogues` |
| `ascent/memory` | Override decision memory. | `ingest_override`, `query`, `OverrideRecord`, `OVERRIDE_TYPES` |
| `ascent/dashboard` | Live dashboard loaders. | `_load_jsonl`, `_merged_weights`, `_nav_series_df` |
| `ascent/utils` | Market-clock helpers (use these instead of naive `datetime`). | `market_now`, `market_today`, `market_date_from_epoch`, `MARKET_TZ` |
| `agents/` | Five specialists + AI PM. Module names require the `_agent` suffix. | `run_us_equities_agent`, `run_macro_agent`, `run_international_agent`, `run_alternatives_agent`, `run_event_agent`, `run_red_team`, `run_ai_pm`, `run_ai_pm_prethesis`, `AIPMResult`, `AIPreThesis`, `AI_PM_TOOLS` |
| `orchestrator/` | Cross-agent capital allocation, skill weighting, correlation guard wiring. | `merge_agent_outputs`, `run_orchestrator`, `SKILL_SCORES_PATH`, `DEFENSIVE_AGENTS`, `REGIME_ALLOCS` |
| `debate/` | Adversarial layer. Advisory only except the single authority-capped judge position change. | `run_debate`, `run_judge`, `run_bull_agent`, `run_bear_agent`, `run_devils_advocate`, `run_regime_specialist`, `run_quant_sanity_check`, `run_adversarial_engine`, `get_authority`, `record_intervention`, `score_pending_verdicts`, `compute_disagreement_score`, `DEBATE_TOOLS` |
| `compliance/` | Hash-chained audit trail, GIPS report, methodology index, risk disclosures. | `AuditTrail`, `record_event`, `compute_gips_performance`, `build_methodology_index`, `generate_risk_disclosures` |
| `memory/` (top level) | Verdict reflection, R2R retrieval, regime + ticker memory. | `reflect_on_verdict`, `query_memory`, `ingest_verdict`, `log_episode`, `query_episodes`, `record_decision`, `get_ticker_context` |
| `simulation/` | Scenario simulator. | `simulation/scenario_simulator.py` |

---

## 3. The four large files

**Line numbers drift with every edit and are a starting offset for a targeted `Read`, not a citation.** These four files moved by 20-50 lines while this map was being written. Always grep the symbol name to confirm the position before relying on it. If a number is off, the symbol name is still correct.

Re-extract fresh at any time:
`.venv/bin/python -c "import ast,sys;t=ast.parse(open(sys.argv[1]).read());[print(n.lineno,n.name) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]" <file>`

### `agents/ai_pm_agent.py` (~2745 lines)

| Line | Symbol |
|---|---|
| 35 | `_build_data_grounding` |
| 177 | `_fetch_financials` |
| 259 | `_apply_recency_gate_python` |
| 296 | `_get_current_regime` |
| 314 / 323 | `class AIPMResult` / `class AIPreThesis` |
| 346 | `AI_PM_TOOLS` (tool schema list, runs to ~758) |
| 759 / 763 / 886 | `_PROPOSE_PORTFOLIO_TOOL` / `_PROPOSE_PRETHESIS_TOOL` / `PRE_THESIS_TOOLS` |
| 904 | `_SYSTEM_PROMPT` |
| 1044 | `_build_temporal_context` |
| 1129 | `_strip_prethesis_for_phase2` |
| 1156 / 1227 | `_PRE_THESIS_PROMPT` / `_SYNTHESIS_PROMPT_TEMPLATE` |
| 1292 / 1307 | `_build_system_prompt` / `_get_calibration_ic_safe` |
| 1331-2068 | tool implementations, all named `_tool_*`: `_tool_get_regime_state` 1331, `_tool_run_quant_agent` 1368, `_tool_get_past_verdicts` 1521, `_tool_get_rebalance_brief` 1633, `_tool_propose_portfolio` 1773, `_tool_get_crowding_signal` 1833, `_tool_get_causal_graph` 1936, `_tool_get_cot_positioning` 2011, `_tool_propose_prethesis` 2068 |
| 2077 / 2085 | `_PRETHESIS_RESEARCH_TOOLS` / `_PRETHESIS_RESEARCH_CAP` |
| 2088 / 2132 | `_make_prethesis_executor` / `_make_executor` |
| 2200 | `_assemble_causal_mechanisms` |
| 2245 | `_build_velocity_context` |
| 2284 | `run_ai_pm_prethesis` (Phase 1, Sonnet) |
| 2432 | `_format_prethesis_for_prompt` |
| 2473 | `run_ai_pm` (Phase 2, Opus) |

### `run_all_agents.py` (~2818 lines)

| Line | Symbol / landmark |
|---|---|
| 43-54 | path + flag constants: `HALT_STATE_PATH`, `AI_PM_DECISION_LOG`, `CATCH_UP_STALE_TRADING_DAYS` |
| 57 / 76 / 143 | `_fetch_position_returns` / `_run_daily_haiku_view` / `_write_decision_log` |
| 204-208 | `HALT_OVERRIDE_PATH`, `REGIME_SIGNAL_PATH`, `REGIME_STALE_DAYS`, `LONG_SHORT_ENABLED` |
| 221 / 239 | `_is_regime_stale` / `_refresh_regime` |
| 301 / 353 | `validate_sector_data` / `check_halt_state` |
| 413 | `_collect_altdata` |
| 451 / 555 | `_fill_wedge_and_decision_outcomes` / `_compute_calibration_returns` |
| **629** | **`main`** (runs to ~1940) |
| 640 / 704 / 716 | weekend branch / same-session guard / startup validation |
| 785 / 799 / 825 | Step 0 data hub / OpenBB ingest / sector coverage guard |
| 841 | lazy agent imports |
| 876 / 903 | Step 1 quant agents / Step 1b |
| 961 | Steps 2/3/4: forward PnL to skill scores to orchestrator |
| 1184-1252 | Steps 5, 5b, 5c, 5d + Exa news |
| 1318 | rebalance brief (must precede AI PM) |
| 1327 | AI PM Phase 1 |
| 1413 | AI PM Phase 2 / Track A-star |
| 1653 | Step 6: execution |
| 1758 | earned-authority update |
| 1824 / 1834 / 1845 | non-rebalance-day and rebalance-day branches |
| 1942 / 2013 | `_log_run` / `_log_holdings` |
| 2224 | `_is_near_scheduled_rebalance` |
| 2259 / 2261 | `_JUDGE_MAX_WEIGHT` / `_JUDGE_MIN_WEIGHT` |
| 2264 / 2307 | `_apply_position_change_to_weights` / `apply_judge_position_change` (removed from call path; debate is advisory-only as of 2026-08-14) |
| 2389 / 2432 | `already_ran_for_session` / `_catch_up_guard` |
| 2464 / 2486 | `_live_book_or` / `_insert_candidate_weights` (add-only discovery) |
| 2526 / 2544 | `_check_mini_rebalance_cooldown` / `_write_mini_rebalance_log` |
| 2558 | `_apply_falsifier_trim` |
| 2663 | `_trigger_mini_rebalance` |

### `scripts/generate_performance_page.py` (~1915 lines)

| Line | Symbol |
|---|---|
| 34-38 | `PAPER_BASE`, `LIVE_START`, `DOCS_DIR`, `OUTPUT_PATH`, `PRICES_LIVE_PATH` |
| 51 / 104 / 126 | `fetch_portfolio_history` / `fetch_current_positions` / `fetch_spy` |
| 145-186 | display constants: `VERDICT_COLORS`, `REGIME_COLORS`, `REGIME_ACCENT`, `ACTUAL_REBALANCE_DATES`, `EXCLUDE_VERDICT_DATES`, `HARDCODED_EVENTS` |
| 232-427 | loaders: `_execution_dates_from_eod_log` 232, `load_verdicts` 251, `load_regime_bands` 327, `load_earned_authority` 354, `load_counterfactual` 365, `load_perf_feedback` 379, `load_ai_pm_decisions` 390, `load_latest_allocation` 404, `load_latest_thesis` 427 |
| 464 / 470 / 474 | `_REDACTION_LABELS` / `_redaction_label` / `_position_reasoning` |
| 495 / 541 | `_CONSTRUCTION_STAGES` (redaction-audited copy) / `_construction_section_html` |
| 604 / 662 | `compute_stats` / `build_chart_data` |
| 693 | `_wf_honesty_line` |
| 717-1204 | HTML section builders: `_earned_authority_html` 717, `_promotion_gates_html` 772, `_counterfactual_chart_html` 796, `_override_scorecard_html` 926, `_thesis_html` 968, `_debate_html` 995, `_timeline_html` 1047, `_positions_html` 1091, `_verdict_section_html` 1143, `_book_section_html` 1204 |
| 1255 / 1477 | `_EDITORIAL_CSS` / `_PAGE_JS` |
| 1531 | `build_html` |
| 1788 / 1833 | `_update_readme_stats` / `main` |

### `debate/agents.py` (~1091 lines)

| Line | Symbol |
|---|---|
| 25 / 53 | `_get_agent_track_record` / `_build_context` |
| 124-431 | context sections, all `_section_*`: `_section_weights` 124, `_section_regime` 180, `_section_factor_exposures` 227, `_section_altdata_positives` 244, `_section_adversarial_flags` 291, `_section_coherence` 349, `_section_tail_asymmetry` 390 |
| 436 / 447 / 468 | `_AGENT_SECTIONS` / `_build_agent_context` / `_EVIDENCE_RULE` |
| 477 / 508 / 563 | `run_bull_agent` / `run_bear_agent` / `run_devils_advocate` |
| 706 / 740 | `REGIME_PLAYBOOK` / `run_regime_specialist` |
| 804-809 | sanity thresholds: `MAX_SINGLE_WEIGHT`, `MAX_SECTOR_WEIGHT`, `MAX_TURNOVER_WARN`, `MIN_POSITIONS`, `MAX_POSITIONS`, `WEIGHT_SUM_TOLERANCE` |
| 812 / 859 / 867 | `_load_quant_sector_map` / `_get_sector_map` / `run_quant_sanity_check` |
| 968 | `_format_round1_for_rebuttal` |
| 984-1068 | round-2 rebuttals: `run_bull_rebuttal` 984, `run_bear_rebuttal` 1012, `run_devils_advocate_rebuttal` 1040, `run_regime_specialist_rebuttal` 1068 |

---

## 4. Data and artifact map

All paths verified present. Writers/readers determined by grepping the path string (worktrees excluded).

| Artifact | Holds | Written by | Read by |
|---|---|---|---|
| `data_cache/prices_live.parquet` | Yahoo live daily bars. Provenance is load-bearing: `prices_simulated`, `prices_live_fallback_simulated`, `prices_live_clean_refetch` (staging), `prices_macro`, `macro_live`/`macro_simulated`, `profiles` are distinct names, never aliased. | `ascent/data/hub.py`, `ascent/data/store/parquet.py` (`save_parquet`) | `ascent/main.py`, agents, `debate/*`, `ascent/research/*`, `ascent/risk/*`, `ascent/monitoring/*`, `scripts/run_ascent_wf.py` |
| `data_cache/earned_authority.json` | AI PM authority level, buffers, blend weights. | `ascent/strategy/earned_authority.py` (`update_authority`) | `scripts/generate_performance_page.py`, `scripts/backfill_counterfactual.py` |
| `data_cache/active_alpha_config.json` | Live sleeve weights after self-improve / shadow promotion / slippage feedback. | `ascent/research/shadow_promoter.py`, `ascent/research/self_improve.py`, `ascent/monitoring/slippage_ic_feedback.py` | `ascent/alpha/stack.py`, `ascent/alpha/altdata_alpha.py`, `ascent/data/validate/altdata_validator.py`, `compliance/methodology_index.py` |
| `data_cache/ai_prethesis_latest.json` | Phase 1 pre-thesis (Sonnet) for the current holding period. | `run_all_agents.py` | `ascent/main.py`, `scripts/generate_performance_page.py` |
| `data_cache/ai_regime_assessment.json` | AI regime read blended with the quant regime. | `run_all_agents.py` | `ascent/main.py` |
| `data_cache/active_falsifiers.json` | Falsifier registry for the holding period. | `ascent/strategy/falsifier_registry.py` (`build_registry`, `add_judge_falsifier`) | same module (`check_all`), `run_all_agents.py` (`_apply_falsifier_trim`) |
| `data_cache/adversarial_authority.json` | Debate intervention authority + scored outcomes. | `debate/adversarial_authority.py` | `debate/judge.py`, `run_all_agents.py` |
| `data_cache/ml_model_us_equities.pkl`, `sleeve_posteriors.json` | ML sleeve model (must carry `feature_names`), meta-learner posteriors. | `ascent/alpha/ml_sleeve.py`, `ascent/alpha/meta_learner.py` | `ascent/alpha/stack.py` |
| `logs/eod_log.jsonl` | One record per EOD execution (orders, NAV, verdict). Canonical execution history. | `run_all_agents.py`, `ascent/execution/eod_runner.py` | `debate/outcome_tracker.py`, `ascent/reporting/debrief.py`, `ascent/reporting/investor_report.py`, `ascent/monitoring/live_vs_backtest.py`, `ascent/monitoring/pre_rebalance_checklist.py`, `ascent/execution/kill_switch.py`, `scripts/heartbeat_check.py`, `scripts/generate_performance_page.py` |
| `logs/multi_agent_run.jsonl` | Per-run agent outputs and merged weights. | `run_all_agents.py` | `debate/debate_runner.py` (`load_latest_run_state`), `ascent/reporting/investor_letter.py`, `scripts/backfill_counterfactual.py` |
| `logs/counterfactual_daily.jsonl` | Daily scored tracks (quant, quant-star, AI PM). | `ascent/monitoring/ai_pm_counterfactual.py` (`score_daily`) | `ascent/strategy/ai_pm_perf_feedback.py`, `scripts/generate_performance_page.py`, `scripts/backfill_counterfactual.py` |
| `logs/holdings_log.jsonl` | Daily position snapshots. | `run_all_agents.py` (`_log_holdings`) | `debate/outcome_tracker.py`, `ascent/dashboard/live_dashboard.py`, `ascent/monitoring/weekly_debrief.py`, `ascent/monitoring/scenario_planner.py` |
| `logs/ai_pm_decision_log.jsonl` | AI PM decisions. Populated on scheduled rebalances only. | `run_all_agents.py` (`_write_decision_log`) | `ascent/strategy/ai_pm_learning.py`, `ascent/strategy/ai_pm_perf_feedback.py`, `scripts/generate_performance_page.py` |
| `logs/sleeve_ic_log.jsonl` | Rolling per-sleeve IC. Drives the IC gate. | `ascent/main.py` (`_log_sleeve_ic`) | `ascent/alpha/stack.py`, `ascent/alpha/meta_learner.py`, `ascent/monitoring/signal_health.py` |
| `logs/ai_pm_calibration.jsonl` | Prediction/outcome pairs for AI PM calibration. | `ascent/strategy/calibration_tracker.py` | `ascent/monitoring/ai_pm_eval_rule.py`, `ascent/monitoring/weekly_debrief.py` |
| `logs/audit_trail.jsonl` | Hash-chained event log. | `compliance/audit_trail.py` | `scripts/verify_audit_trail.py` |
| Other `logs/*.jsonl` | `alerts`, `attribution_log`, `alpha_wedge`, `decision_memory`, `hedge_log`, `slippage`, `event_trades`, `regime_episodes`, `skill_scores_log`, per-agent `*_pnl.jsonl`. Each has one owning module named after it. | see owning module `_LOG` / `LOG_PATH` constant | various |
| `outputs/debate_log/verdict_YYYY-MM-DD.json` | Judge verdict: `verdict.reasoning`, `verdict.key_risks`, `position_changes`. | `debate/debate_runner.py`, `debate/judge.py` | `run_all_agents.py`, `debate/outcome_tracker.py`, `agents/ai_pm_agent.py`, `ascent/reporting/debrief.py`, `ascent/reporting/blind_spot_detector.py`, `memory/reflection_agent.py`, `memory/r2r_interface.py`, `scripts/generate_performance_page.py` |
| `outputs/debate_log/debrief_*.json`, `agent_credibility.json`, `blind_spots.json` | Post-hoc debriefs, per-agent track record, recurring blind spots. | `ascent/reporting/debrief.py`, `debate/outcome_tracker.py`, `ascent/reporting/blind_spot_detector.py` | `debate/agents.py` (`_get_agent_track_record`), `agents/ai_pm_agent.py` |
| `outputs/wf_results/wf_report_*.json`, `wf_equity_*.csv` | Walk-forward fold results and equity curves. | `scripts/run_ascent_wf.py`, `ascent/research/wf_framework/metrics.py` | `ascent/reporting/verified_numbers.py` (`load_wf_report`, `canonical_wf`), `scripts/verify_docs.py` |
| `outputs/ai_pm_theses/`, `investor_reports/`, `scenarios/` | Formatted thesis docs, monthly PDFs, scenario plans. | `ascent/strategy/thesis_formatter.py`, `ascent/reporting/investor_report.py`, `ascent/monitoring/scenario_planner.py` | dashboard generator |
| `execution/merged_weights.json` | The live target book (AI PM allocation + orchestrator overlay). | `run_all_agents.py`, `ascent/execution/eod_runner.py` | `agents/ai_pm_agent.py`, `agents/alternatives_agent.py`, `debate/adversarial_monitor.py`, `ascent/portfolio/hedge_overlay.py`, `ascent/monitoring/{position_health,scenario_planner,weekend_runner,pre_rebalance_checklist}.py`, `ascent/dashboard/live_dashboard.py`, `compliance/risk_disclosure.py` |
| `execution/pending_approvals.json`, `approval_override.json` | Human-approval queue. | `ascent/execution/approval_server.py` | same |
| `dashboard/regime_signal.json`, `regime_labels.csv` | Current regime signal + historical labels. | `ascent/regime/engine.py`, `run_all_agents.py` | `scripts/generate_performance_page.py`, `scripts/evaluate_hedge.py`, `ascent/research/factor_discovery/discovery_runner.py`, `ascent/monitoring/regime_trajectory.py` |
| `dashboard/factor_exposures.json` | Portfolio factor exposures. | `ascent/risk/factor_exposure.py` (`export_factor_exposures`) | dashboard, `debate/agents.py` |
| `dashboard/agent_skill_scores.json` | 63-day rolling Sharpe per agent. Drives orchestrator capital allocation. | `ascent/monitoring/skill_tracker.py` (`export_skill_scores`) | `orchestrator/central_intelligence.py` (`SKILL_SCORES_PATH`) |
| `dashboard/live_vs_backtest.json`, `methodology_index.json` | Live vs backtest comparison, redaction-audited methodology. | `ascent/monitoring/live_vs_backtest.py`, `compliance/methodology_index.py` | dashboard |
| `rebalance_calendar.csv` | Single column `rebalance_date`. Defines scheduled rebalance days. | manual | `run_all_agents.py`, `ascent/utils/market_time.py`, `ascent/reporting/investor_letter.py`, `scripts/heartbeat_check.py`, `scripts/generate_performance_page.py` |
| `docs/index.html` | Public GitHub Pages dashboard. | `scripts/generate_performance_page.py` | published |

Backups follow a `*.bak*` / `*.pre_*.bak.parquet` convention and `data_cache/_corrupt_backup_*/`. Do not read them as current state.

---

## 5. Test map

124 `test_*.py` files. Config in `pyproject.toml` (`testpaths = ["tests"]`, `addopts = "-v --tb=short"`).

Flat files in `tests/` are historical: named per feature (`test_hedge_overlay.py`, `test_mvo_optimizer.py`), per phase (`test_phase1_hardening.py` ... `test_phase6_signals.py`), and per plan (`test_plan_a.py` ... `test_plan_d.py`). Subdirectories are the newer convention.

| Subdir | Covers |
|---|---|
| `tests/agents/` | AI PM internals: tools, calibration gate, fallback/force-seal, financials, news, pre-thesis fixes |
| `tests/alpha/` | Meta-learner, stack weight resolution |
| `tests/data/` | New ingest sources, CBOE options time bound |
| `tests/debate/` | Judge symmetry, outcome tracker (W4) |
| `tests/features/` | Individual feature defs (HY spread direction, sector relative momentum) |
| `tests/integrations/` | Exa, OpenBB, StockTwits clients |
| `tests/memory/` | Ticker memory |
| `tests/monitoring/` | Alerts/liveness, alpha wedge, analogues, conviction, daily intelligence, position health, rebalance brief/trigger, regime trajectory, signal health, weekend mode |
| `tests/portfolio/` | Exposure overlays, long/short, risk budget cap, risk construction |
| `tests/regime/` | AI-vs-quant regime blend |
| `tests/scripts/` | Heartbeat check, LLM cache seeding |
| `tests/strategy/` | AI calibration, AI PM learning, discovery guards, earned-authority blend, falsifier registry + JSON parsing, ticker discovery |
| `tests/test_wf_framework/` | Walk-forward engine, windows, strategies, execution, optimizer, metrics |

Useful subsets:

```bash
.venv/bin/python -m pytest -q                          # everything (slow)
.venv/bin/python -m pytest tests/portfolio tests/alpha -q
.venv/bin/python -m pytest tests/test_wf_framework -q   # slow
.venv/bin/python -m pytest tests/agents tests/strategy -q
.venv/bin/python -m pytest tests -q -k "regime and not wf"
.venv/bin/python -m pytest tests/test_parquet_store_dedup.py tests/test_ic_outlier_robustness.py -q
.venv/bin/python scripts/verify_docs.py                 # doc drift, not pytest
```

Known failures (verified by running the specific suspects):

| Test | Status |
|---|---|
| `tests/integrations/test_openbb_client.py::test_get_cot_snapshot_returns_dict` | FAILS. Needs network (`get_cot_snapshot` returns `None` offline). Not a regression. |
| `tests/test_wf_framework/test_metrics.py` | PASSES (10/10). The buggy-Sortino issue was fixed; `verify_docs.py` has `check_sortino_annualized_once` guarding it. Old notes calling this a known failure are stale. |
| `tests/test_wf_framework/test_ascent_engine.py` | FAILS. Observed `FF` on the first two of six tests in two separate runs. Also very slow and timing-unstable (each test runs a real multi-fold walk-forward; individual tests have exceeded 9 minutes). The rest of `tests/test_wf_framework/` passes 54/54 in under 10s, so use `--ignore=tests/test_wf_framework/test_ascent_engine.py` for fast iteration. |

`tests/test_walkforward_institutional.py` passes (5/5).

---

## 6. Common tasks to where to look

| Task | Files | Symbols |
|---|---|---|
| Add an alpha sleeve | `ascent/alpha/<new>.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py` | Add to `DEFAULT_ALPHA_WEIGHTS` in **both** stack.py and self_improve.py; add to `_SPARSE_FILL_ZERO` if the panel is sparse; `build_alpha_stack` |
| Change position sizing / caps | `ascent/portfolio/optimizer.py` | `enforce_constraints`, `_water_fill_cap`, `enforce_cluster_cap`, `enforce_risk_budget_cap`, `top_n_equal_weight`, `rank_weighted` |
| Change gross exposure / hedging | `ascent/portfolio/exposure.py`, `ascent/portfolio/hedge_overlay.py` | `apply_exposure_overlays`, `MA_WINDOW`, `VOL_TARGET`, `ma_filter_scale`, `vol_target_scale`, `apply_hedge_overlay` |
| Find why a weight got capped | `ascent/portfolio/optimizer.py`, `orchestrator/central_intelligence.py`, `ascent/risk/correlation_guard.py` | `_water_fill_cap` post-condition, `SectorDataError`, `enforce_cluster_cap`, `merge_agent_outputs`, `check_cross_agent_correlation` |
| Trace why a sleeve was zeroed | `ascent/alpha/stack.py`, `ascent/main.py`, `logs/sleeve_ic_log.jsonl` | `IC_GATE_THRESHOLD`, the gate block near stack.py line ~119-170, `_log_sleeve_ic`, `_winsorize_rows` |
| Debug a bad rebalance | `outputs/debate_log/verdict_<date>.json`, `logs/eod_log.jsonl`, `run_all_agents.py` (`main`, execution from ~1653), `ascent/execution/eod_runner.py` | `verdict.reasoning`, `position_changes`, `run_eod_with_weights`, `_enforce_reduce_size`, `_apply_falsifier_trim`, `_log_run` |
| Trace an order that should not have fired | `ascent/execution/eod_runner.py`, `ascent/execution/order_engine.py`, `ascent/execution/alpaca_broker.py` | `run_eod_with_weights` (`force=` no-ops off-rebalance), `compute_orders`, `MIN_TRADE_THRESHOLD`, `submit_order` |
| Add an LLM call | `ascent/llm/client.py` (import only, do not redefine) | `chat_completion`, `generate_structured`, `tool_completion`, `extract_text`, `DEFAULT_MODEL` / `SONNET_MODEL` / `HAIKU_MODEL`. Never index `content[0].text`; never pass `temperature` / `thinking=` |
| Add an AI PM tool | `agents/ai_pm_agent.py` | Append schema to `AI_PM_TOOLS` (~346), add `_tool_<name>`, wire into `_make_executor` (~2088) or `_make_prethesis_executor` (~2044) |
| Change debate behaviour | `debate/agents.py`, `debate/judge.py`, `debate/adversarial_authority.py` | `_AGENT_SECTIONS`, `REGIME_PLAYBOOK`, `run_judge`, `get_authority`. Debate must stay advisory; the only write is the authority-capped `position_changes[0]` applied in `run_all_agents.py` |
| Add a data source | `ascent/data/ingest/<new>.py`, `ascent/data/hub.py`, `ascent/data/validate/altdata_validator.py` | `save_parquet` with a provenance-honest cache name, `run_hub`, `register_altdata_source`, `validate_altdata_source` |
| Change the regime model | `ascent/regime/model.py`, `engine.py`, `integration.py` | `RegimeModel`, `RegimeEngine(config=dict)`, `walk_forward_model_select`, `regime_scale_weights`, `check_emergency_refit_triggers` |
| Change AI PM authority / guardrails | `ascent/strategy/earned_authority.py`, `ascent/strategy/ai_pm_guardrails.py` | `LEVEL_WEIGHTS`, `update_authority`, `blend`, `apply_guardrails`, `_LEVEL_CONFIG` |
| Change the public dashboard | `scripts/generate_performance_page.py` | `build_html` (~1486), the `_*_html` builders, `_EDITORIAL_CSS`, `_CONSTRUCTION_STAGES` (confidentiality-audited copy) |
| Add a doc-drift guard | `scripts/verify_docs.py` | write `check_*` returning `(ok, detail)`, register in `CHECKS` |
| Reproduce the walk-forward number | `scripts/run_ascent_wf.py`, `outputs/wf_results/`, `ascent/reporting/verified_numbers.py` | `--live-system`, `load_wf_report`, `canonical_wf`, `CANONICAL_WF_ARTIFACT` |

