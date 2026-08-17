# Lean fix: docs + dead risk modules — report

Date: 2026-08-15

## Task 1: Delete two dead modules

Verified via `grep -rn "sample_covariance|shrinkage_covariance|correlation_matrix|classify_regime" --include="*.py" .`
(excluding `.venv`, `.git`, `.claude/worktrees`): every hit resolved to definitions/internal
uses inside `ascent/risk/covariance.py` and `ascent/risk/regime.py` themselves — no external
callers anywhere in the repo. Also checked for dedicated test files: none exist for either
module. `tests/test_covariance_tz_alignment.py` imports from `ascent/risk/covariance_model.py`
(a different, live file — confirmed out of scope, untouched) and `factor_model.py`; the various
`test_regime_*.py` files carry no `from ascent.risk...` imports at all (they test
`ascent/regime/`, not `ascent/risk/regime.py`).

Deleted:
- `ascent/risk/covariance.py` (43 LOC)
- `ascent/risk/regime.py` (39 LOC)

`.venv/bin/python -m py_compile ascent/risk/*.py` passed after deletion.

`ascent/risk/covariance_model.py`, `correlation_guard.py`, `factor_data.py`, `factor_model.py`,
`factor_exposure.py`, `factor_constraints.py`, `pm_risk_validator.py` were not touched.

## Task 2: Wide-format cache corruption claim — corrected

Loaded all three caches via `ascent.data.store.parquet.load_parquet`:

| cache | rows | date range |
|---|---|---|
| prices_macro | 1662 | 2020-01-02 to 2026-08-13 |
| prices_international | 1662 | 2020-01-02 to 2026-08-13 |
| prices_alternatives | 1662 | 2020-01-02 to 2026-08-13 |

All three have a proper `date` column and reasonable row counts — the previously-reported
dateless-`RangeIndex`/176k-151k-150k-row corruption is gone. Updated the "Wide-format caches"
gotcha in `CLAUDE.md` to state the repaired current state instead of describing a stale,
already-fixed problem as still open.

## Task 3: macro_agent / alternatives_agent characterization — corrected

Read `outputs/analyst/proof_audit_2026-08-13.json`:
- `macro_agent`: `metric` (IC) = +0.0204, `p_value` = 0.0606 (two-sided), `sample_size` = 1644,
  `verdict` = CUT (the two-sided-significance gate; one-sided ≈0.03 would clear it). Universe
  is 12 symbols — small enough that the significance test itself is low-powered.
- `alternatives_agent`: `metric`/`p_value` = null, `verdict` = INSUFFICIENT_DATA, `reason`:
  "signal matrix has insufficient non-NaN density (0 of 1649 candidate dates carry at least 10
  non-NaN values; need 30)". `prices_alternatives.parquet` has 7 symbol columns (8 columns
  total minus `date`) — below the harness's 10-symbol minimum for long-short leg construction.
  This is a harness/universe-size limitation, not a data-quality problem; the cache itself is
  valid per Task 2.

Neither `CLAUDE.md` nor `CHECKPOINTS.md` previously discussed these agents' verdicts
specifically (`CHECKPOINTS.md` has zero mentions of either name), so — per instructions — no
new section was invented. The correction was folded into the same "Wide-format caches" gotcha
block in `CLAUDE.md` where the caches are already discussed.

## Task 4: AI PM's `run_quant_agent` tool requires all 4 agents

Verified in `agents/ai_pm_agent.py`:
- Line 427-434: `run_quant_agent` tool registration, `"enum": ["us_equities", "macro",
  "international", "alternatives"]`.
- Line 994 and 1309: system-prompt text, `"Required: run_quant_agent for all four agents."` /
  `"Required: run_quant_agent ×4."`

Corrected the opening "What this project is" paragraph in `CLAUDE.md` (the sentence claiming
macro/international/alternatives are "code-intact but not invoked in the daily run") to note
that the AI PM's Phase 2 actually does invoke all three via `run_quant_agent` on a scheduled
rebalance day, at real wall-clock/API cost.

## Verification

- `.venv/bin/python -m py_compile ascent/risk/*.py` — passed.
- `.venv/bin/python scripts/verify_docs.py` — 24 passed, 1 failed. The 1 failure
  (`repo_map_pointers`, missing `data_cache/active_alpha_config.json` and
  `logs/sleeve_ic_log.jsonl`) is the pre-existing, expected failure called out in the task —
  not introduced by this change. No new failures.

## Commits

1. `f5fe50d` — `chore(risk): delete two zero-caller dead modules`
2. `a304818` — `docs: correct stale gotchas -- caches repaired, agent verdicts, AI PM invokes all 4 agents`
