# Weekend Mode — Design Spec
**Date:** 2026-05-25

## Goal
`python run_all_agents.py` on a weekend runs a compound intelligence loop that makes the system measurably smarter before Monday open. Once per weekend — if you ran Saturday, Sunday exits immediately with a message.

## The Core Idea
Weekday runs trade. Weekend runs **learn**. Every weekend the system:
- Expands alt-data coverage to all 901 symbols (not just portfolio)
- Retrains and hyperparameter-searches the ML model with fresh data
- Runs factor discovery to find new alpha factors
- Runs self-improve with 4× more variants (20 vs 5)
- Retrains the conviction gate logistic regression
- Runs the AI PM as a pure research analyst across the full universe
- Runs a structured weekly debrief: what worked, what didn't, systematic patterns
- Stress-tests the portfolio against 5 adversarial scenarios
- Writes everything to caches so Monday's run starts pre-warmed

## Once-Per-Weekend Gate
`data_cache/weekend_run_state.json` stores `{iso_week, iso_year, run_date}`.
On weekend entry: if current ISO week matches → print message + exit.
On completion: write stamp.

## Modules

### `ascent/monitoring/weekly_debrief.py`
Post-mortem synthesized by Haiku:
- Load week's attribution (what positions contributed / dragged)
- Load AI PM override history (which overrides beat quant baseline)
- Load debate verdicts (what the debate got right/wrong)
- Haiku synthesizes: top patterns, systematic biases, what to watch next week
- Output: `data_cache/weekly_debrief.json`
- Fed into AI PM system prompt on next run via `get_rebalance_brief`

### `ascent/monitoring/scenario_planner.py`
5 fixed adversarial scenarios + 1 dynamic from debrief:
1. SPY opens -3% Monday (risk-off shock)
2. Regime flips stressed mid-week
3. Largest current position -15% (idiosyncratic blow-up)
4. Rates spike 50bps
5. EM selloff: EWY/EEM/EWT all -5%
6. Dynamic: derived from debrief's top identified risk

For each: compute portfolio dollar impact, flag if probability > 40%.
Sonnet assesses probability + pre-emptive adjustment recommendation.
Output: `data_cache/scenario_plan.json`. Console alert if any flagged.

### `ascent/monitoring/weekend_runner.py`
Orchestrates all jobs sequentially. Each job:
- Wrapped in `_run_job(name, fn, once_per_weekend=True/False)`
- `once_per_weekend=True` → skips if already completed this weekend (per state file)
- `once_per_weekend=False` → runs every time (alt-data, debrief, scenario plan)
- Times each job, logs to `logs/weekend_run.jsonl`

Job order (dependency-aware):
1. Alt-data full sweep (all 901 symbols) — every run
2. LLM fundamental cache update — every run
3. ML force retrain + GridSearch — once/weekend
4. Factor discovery — once/weekend
5. Self-improve (20 variants) — once/weekend
6. Conviction gate retrain — every run
7. AI PM weekend research — once/weekend
8. Weekly debrief — every run
9. Adversarial scenario planning — every run

## Weekend Detection in run_all_agents.py
```python
if date.today().weekday() >= 5:  # Saturday=5, Sunday=6
    from ascent.monitoring.weekend_runner import run_weekend
    run_weekend(dry_run=dry_run)
    return
```
Inserted at top of `main()`, after startup validation.

## Cost
~$1.50 first run, ~$0.30 second run (blocked). ~$6-8/month total.

## What makes this novel
The compound learning loop: every weekend the system reviews its own mistakes,
updates its conviction model, and discovers new alpha factors. After 6 months,
the AI PM has 24 weeks of self-directed learning. No fund at this size does this.
