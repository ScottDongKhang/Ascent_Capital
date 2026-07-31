# AI PM Refresher — 2026-07-30

Grounded trace of `agents/ai_pm_agent.py`, `run_all_agents.py`, `orchestrator/central_intelligence.py`,
and `debate/`. File:line citations reflect the code as of this date — re-verify before citing later.

## Two-phase design

### Phase 1 — `run_ai_pm_prethesis()` (`agents/ai_pm_agent.py:2361`)

- Runs on `SONNET_MODEL` (breadth, cheap reads).
- Forms an independent thesis with **no visibility into quant output** — tool set is
  research-only (regime, macro, news, SEC/earnings signals, narrative shift, calibration
  report, causal graph). Capped at 7 research calls before a forced `propose_prethesis` call.
- Produces: macro/regime view, 8-15 high-conviction names (each with thesis + falsifier),
  names to avoid, sector tilts, sleeve-weight priors, and a required directional stance
  ("cannot be a hedge").
- Doesn't write anything itself — `run_all_agents.py` writes the two output files:
  - `data_cache/ai_regime_assessment.json` (`run_all_agents.py:1384-1396`) — macro/sleeve-prior half
  - `data_cache/ai_prethesis_latest.json` (`run_all_agents.py:1400-1413`) — name-picking half
  - Both gated by a freshness check (`ascent/utils/freshness.py`) after a real incident where a
    33-day-stale thesis kept biasing the book.

### Phase 2 — `run_ai_pm()` (`agents/ai_pm_agent.py:2544`)

- Runs on `DEFAULT_MODEL` (Opus) — judgment, given the sealed Phase 1 thesis plus the actual
  quant agent outputs and orchestrator-merged weights.
- Full tool set (24 tools, incl. `run_quant_agent`, factor/VaR/concentration checks) ending in
  `propose_portfolio`, which must state `prethesis_disposition` (FOLLOWED/OVERRIDDEN) and
  acknowledge any feedback file — enforced in code, not just prompt.
- A conviction-inflation cap downgrades over-tagged "high conviction" calls in code
  (`check_conviction_inflation`, `:1878-1894`).
- Followed by a red-team adversarial self-play pass (`agents/red_team_agent.py::run_red_team`)
  that can revise the result.

**Open question flagged this session**: `run_all_agents.py` computes a Sonnet/Opus "smart
trigger" decision for Phase 2 (`:1473-1491`) but never passes it as `model_override` into
`run_ai_pm()` (`:1504-1511`) — so Phase 2 currently always runs on Opus regardless of that
trigger; the trigger only affects what gets logged (`:1536-1539`). Not yet confirmed whether
this is intentional or a bug — needs a decision, not yet investigated further.

## Sequencing (confirms CLAUDE.md's claim)

Quant agents (`us_equities`, `macro`, `international`, `alternatives`) → orchestrator merge
(`central_intelligence.py`, which has **zero knowledge of the AI PM** — grep confirms no
`ai_pm` references) → Phase 1 prethesis (`run_all_agents.py:1373-1378`) → Phase 2 synthesis
(`:1504-1511`) → decision log + `earned_authority.json` blend (`:1521`, `:1536`).

Because Phase 1 writes its files *after* this run's quant signals were already computed
(orchestrator merge at line 1191 vs prethesis at line 1375), the fresh prethesis can only
affect the **next** run's alpha stack, never the current one. Consumed by `ascent/main.py:525`
(regime blend) and `:652` (alpha floor / avoid-list), and `ascent/regime/engine.py:460-477`.

## Debate integration

After Phase 2, `should_run_debate()` (`ascent/execution/debate_gate.py`) gates whether debate
runs at all. `debate/debate_runner.py::run_debate()` sees the AI PM's Phase 2 thesis
(sentiment, causal mechanisms) via `portfolio_state` (`run_all_agents.py:1872-1889`).

If the verdict isn't `halt_and_review`, `apply_judge_position_change()`
(`run_all_agents.py:2316-2395`) — the one sanctioned write-path per CLAUDE.md integrity
constraint 5 — takes only `verdict["position_changes"][0]`, clamps the target to
`_JUDGE_MAX_WEIGHT=0.10` / floors at `_JUDGE_MIN_WEIGHT=0.01`, and rescales everything else
into the remaining `(1 - target)` budget. Intervention is recorded via
`debate.adversarial_authority.record_intervention()` for 10-day outcome scoring, and a
falsifier is registered (`ascent.strategy.falsifier_registry.add_judge_falsifier`).

The judge's own authority ceiling is separate: `debate/judge.py` clamps its proposed change
using `adversarial_authority.get_authority(itype)["allowed_change_pct"]` *before*
`run_all_agents.py` ever sees the verdict. `_JUDGE_MAX_WEIGHT=0.10` is an outer hard cap on
top of that (usually much smaller, 1-4%) authority-clamped value.

## Two separate "earned authority" systems — don't conflate them

1. **`ascent/strategy/earned_authority.py`** → `data_cache/earned_authority.json`.
   Governs how much of the AI PM's Phase-2 portfolio *deltas* get blended in via
   `authority_blend()` (`run_all_agents.py:1521`). Six levels, Shadow→Analyst→Associate→
   Manager→Director→CEO, tracking-error budgets `[0.0, 0.05, 0.15, 0.30, 0.50, 0.75]`
   (`LEVEL_WEIGHTS`), hard cap 0.80 (`HARD_CAP`). Promoted/demoted on rolling (21-63 day)
   Sortino edge of Track D (AI PM) vs Track A★ (pure quant) returns, hit rate, profit factor,
   and a minimum number of scored decisions per level (5/8/10/15).

2. **`debate/adversarial_authority.py`** → `data_cache/adversarial_authority.json`, log at
   `logs/adversarial_interventions.jsonl`. Governs the **judge's** intervention authority per
   type (`adversarial_thesis`, `regime_sizing`, `coherence_risk`, `event_risk`,
   `conviction_press`, `falsifier_trim`), tiered by win rate: >70%→4% max change, >50%→2%,
   else ("low"/unscored)→1%, **suspended→0%** once `MIN_SAMPLE_SUSPEND=30` scored 10-day
   outcomes show a win rate below `SUSPEND_WIN_RATE=0.40`.

`MIN_SAMPLE_SUSPEND` lives in system (2), not (1) — the two are scored independently on
different state files and should not be conflated.

## `ai_prethesis_latest.json` vs `ai_regime_assessment.json`

Both come from the *same* sealed `AIPreThesis` object produced by a single Phase 1 call:

- **`ai_prethesis_latest.json`**: name-level payload — `high_conviction_names` (with thesis
  text) and `names_to_avoid`, plus `as_of_date`. Consumed by `ascent/main.py` ("Step 4:
  Portfolio Construction") to apply an alpha floor for conviction names and zero out avoid-list
  names during quant portfolio construction.
- **`ai_regime_assessment.json`**: macro/regime-level payload — `regime_assessment`
  (label/confidence/reasoning) merged with `sleeve_weight_prior` (per-sleeve IC deltas) and
  `as_of_date`. Consumed by `ascent/main.py` to blend into the `RegimeEngine`
  (`regime_engine.blend_with_ai(...)`) and to feed `build_alpha_stack(..., ai_prior=...)`,
  boosting/damping specific alpha sleeves (trend, statarb, meanrev, ml, fundamental, earnings,
  volatility).

Both are gated by the same freshness check (`ai_prior_is_fresh`,
`ascent/utils/freshness.py`, `AI_PRIOR_MAX_AGE_DAYS`) to prevent a stale thesis from silently
continuing to bias the book on later runs.
