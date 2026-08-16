# Repointing CANONICAL_WF_ARTIFACT to the real shipped 2-sleeve system

## What was wrong

`ascent/reporting/verified_numbers.py`'s `CANONICAL_WF_ARTIFACT` pointed at
`outputs/wf_results/wf_report_clean_2026-06-22.json`, produced by
`scripts/run_ascent_wf.py` / `ascent/research/wf_framework/`. That framework has a
confirmed, still-open bug: `ascent_strategy.py::_make_alpha_weights` force-injects the
`trend` sleeve (measured CUT/negative signal, not in `DEFAULT_ALPHA_WEIGHTS`) and
bypasses the IC gate entirely, because `ascent/alpha/stack.py`'s gate only runs when
`alpha_weights is None`, and `wf_framework` always passes an explicit override. So the
project's canonical, citable number did not reflect the actual shipped 2-sleeve
(`meanrev` + `statarb`) system.

## What was done

1. **Read `ascent/reporting/verified_numbers.py` in full** (149 lines). `load_wf_report()`
   requires a flat JSON with `sharpe, cagr, max_drawdown, beta, alpha, win_rate, wfe,
   volatility, n_folds, n_oos_days` plus an optional `_meta` block
   (`oos_window, n_symbols, spy_cagr_same_window, excess_cagr_vs_spy, llm_disabled`).
   Read `wf_report_clean_2026-06-22.json` to confirm the exact shape — it also carries a
   `sortino` key that the loader never reads (no `sortino` field on `WalkForwardRecord`
   at all), so that key is decorative/schema-compatible only.

2. **Checked `scripts/verify_docs.py`'s `check_sortino_annualized_once`** (line ~398).
   It reads `ascent/research/wf_framework/metrics.py` as text and regex-checks that
   `sortino()`'s downside-deviation annualization isn't double-applied — a pure formula
   check on the *retired* framework's code, with zero reference to
   `CANONICAL_WF_ARTIFACT` or any artifact path. It is orthogonal to which artifact is
   canonical, confirmed this distinction, and left it untouched as instructed.

   Also found (not in the task's list, but load-bearing) `check_verified_numbers_matches_artifact`
   (line ~583), which **hardcodes** `"outputs/wf_results/wf_report_clean_2026-06-22.json"`
   as a literal string — independent of the `CANONICAL_WF_ARTIFACT` constant — and checks
   that `CURRENT_VERIFIED_NUMBERS.md`'s stated Sharpe/CAGR/max-DD/beta match that specific
   file. Since the task didn't ask to update `CURRENT_VERIFIED_NUMBERS.md` and the old
   artifact file was left in place untouched, this check still passes against the old
   numbers — it is simply unaffected by today's change. Flagging this because it means
   **`CURRENT_VERIFIED_NUMBERS.md` still asserts the old (wf_framework-tainted) Sharpe
   0.412 / CAGR 10.3% / max DD -32.9% / beta 0.73 figures as current**, not the new
   Sharpe 0.415 / CAGR 10.2% / max DD -45.65% / beta 0.947 figures now returned by
   `canonical_wf()`. That doc update is out of this task's explicit scope (only
   `CANONICAL_WF_ARTIFACT` itself was in scope) but is the natural next step — otherwise
   the doc and the loader now disagree.

3. **Wrote `scripts/generate_wf_report_from_runner.py`**, following the
   `scripts/reconcile_numbers.py` style (module docstring explaining "why this exists",
   `argparse` with a `--check` dry-run flag, `ROOT`-relative paths). It packages the
   already-verified 2026-08-15 `walk_forward_runner.py` numbers — Sharpe 0.415, Hit Rate
   52.3%, CAGR +10.20%, Volatility 24.58%, Sortino 0.551, Max Drawdown -45.65%,
   Benchmark CAGR +13.82%, Beta 0.947, Total Return +88.26%, 165 folds, 1641 trading
   days — sourced from
   `outputs/wf_results/wf_run_target_architecture_2026-08-15_post_phantom_fix.log` and
   `outputs/wf_results/vc-task-4-post-phantom-fix-report.md`, into the same JSON shape as
   the old artifact. Ran it to write
   `outputs/wf_results/wf_report_clean_2026-08-15.json`.

   `_meta.alpha_overrides` is set to `{"meanrev": 0.5, "statarb": 0.5}` — the actual
   active 2-sleeve set per `DEFAULT_ALPHA_WEIGHTS` — rather than copying the old
   artifact's stale `{"llm_fundamental": 0.0, "narrative": 0.0}`. The run's own
   `[alpha_stack] loaded=[...]` log line lists every sleeve loaded into the stack object
   (including `trend`, `fundamental`, etc.), which is not the same as which sleeves carry
   nonzero weight; the `_meta.notes` field says so explicitly to prevent that log line
   being misread later as evidence the trend-sleeve bug is still present here.

   **Known, disclosed gap — `wfe`:** the old framework's Walk-Forward Efficiency
   (`ascent/research/wf_framework/metrics.py::walk_forward_efficiency`) is
   `mean(OOS_Sharpe_fold / IS_Sharpe_fold)` across folds, which requires a per-fold
   in-sample Sharpe. `ascent/research/walk_forward_runner.py` — the framework that
   actually produced the 2026-08-15 run — does not compute or log that per-fold IS
   Sharpe at all; it is a structurally different, OOS-only pipeline. There is no honest
   way to compute WFE from this run's log without fabricating an in-sample comparison
   that was never made. Per the task's explicit instruction not to fabricate, this
   artifact writes **`"wfe": null`** rather than inventing a number.

   This required a small, deliberate scope addition beyond "just repoint the pointer":
   - `WalkForwardRecord.wfe` type changed from `float` to `Optional[float]` in
     `ascent/reporting/verified_numbers.py`.
   - `load_wf_report()`'s caveat logic now branches: negative `wfe` → the existing
     "disclose as overfit" caveat (unchanged behavior for the old artifact); `wfe is
     None` → a new caveat explicitly stating WFE was not computed for this run and
     should not be reported.
   - The one other production consumer of `.wfe`,
     `scripts/generate_performance_page.py:707` (`if wf.wfe < 0:` in `_wf_honesty_line`,
     the sentence rendered on the published GitHub Pages performance page), was guarded
     to `if wf.wfe is not None and wf.wfe < 0:`. Verified by direct call:
     `_wf_honesty_line()` now returns `'Walk-forward OOS Sharpe 0.41 (2020-01-02 ->
     2026-07-15, 165 folds) is the rigorous figure.'` with no WFE sentence appended and
     no exception. Grepped the whole repo for other `.wfe` consumers
     (`edge_tests/*.py`, `run_ascent_wf.py`, `wf_overlay_comparison.py`) — none of them
     call `canonical_wf()` / `load_wf_report()`; they build their own dicts independently
     and were left untouched, in scope boundaries.
   - `WalkForwardRecord` field order was not changed (no default value was added to
     `wfe`, only its type widened), so no dataclass field-ordering issue and no change
     needed in `tests/test_investor_letter.py`'s direct `WalkForwardRecord(...)`
     constructions.

4. **Updated `CANONICAL_WF_ARTIFACT`** in `ascent/reporting/verified_numbers.py` to
   `"outputs/wf_results/wf_report_clean_2026-08-15.json"`.

5. **Ran `.venv/bin/python scripts/verify_docs.py`**: 24 passed, 1 failed — identical to
   the pre-change baseline (`repo_map_pointers` failing on
   `data_cache/active_alpha_config.json` / `logs/sleeve_ic_log.jsonl`, expected/unrelated).
   No new failures. `sortino_annualized_once` still passes (untouched, confirmed
   orthogonal per step 2). `no_unsourced_sharpe` still passes, now recognizing 17
   artifacts (was 16) since the new file is on disk under the expected
   `wf_report_*.json` glob.

6. **Ran `canonical_wf()`** — loads successfully:
   ```
   WalkForwardRecord(artifact='outputs/wf_results/wf_report_clean_2026-08-15.json',
     sharpe=0.415, cagr=0.102, max_drawdown=-0.4565, beta=0.947, alpha=-0.0362,
     win_rate=0.523, wfe=None, volatility=0.2458, n_folds=165, n_oos_days=1641,
     oos_window='2020-01-02 -> 2026-07-15', ...,
     caveats=('WFE not computed for this artifact: ...',))
   citation(): "Sharpe 0.41, CAGR +10.2%, max DD -45.6%, beta 0.95
     (OOS 2020-01-02 -> 2026-07-15, 165 folds, 1641 days)
     [outputs/wf_results/wf_report_clean_2026-08-15.json]"
   ```

7. **Ran tests**: `grep -rl "verified_numbers\|canonical_wf" tests/` found
   `tests/test_investor_letter.py` (source; the `.pyc` cache hit is not source).
   `.venv/bin/python -m pytest tests/test_investor_letter.py -q` → **7 passed**, no
   changes needed to the test file.

## Files touched

- `ascent/reporting/verified_numbers.py` — `CANONICAL_WF_ARTIFACT` repointed;
  `WalkForwardRecord.wfe` widened to `Optional[float]`; `load_wf_report()` caveat logic
  branches on `wfe is None`.
- `scripts/generate_performance_page.py` — `wf.wfe < 0` guarded against `None` at the
  one call site that reads `.wfe`.
- `scripts/generate_wf_report_from_runner.py` — new, packages the 2026-08-15 numbers.
- `outputs/wf_results/wf_report_clean_2026-08-15.json` — new artifact (generator output).

## Not done (explicitly out of scope per the task)

- `ascent/research/wf_framework/` and `ascent/research/walk_forward_runner.py` — untouched.
- `CURRENT_VERIFIED_NUMBERS.md` — still states the old (wf_framework-tainted) Sharpe
  0.412/CAGR 10.3%/max DD -32.9%/beta 0.73. This is now stale relative to
  `canonical_wf()`'s new output and `check_verified_numbers_matches_artifact` in
  `verify_docs.py` still checks it against the *old* hardcoded artifact path, so it
  currently shows green for the wrong reason. Updating that doc (and, ideally,
  `check_verified_numbers_matches_artifact` to read `CANONICAL_WF_ARTIFACT` from
  `verified_numbers.py` instead of its own hardcoded literal) is the natural next step,
  flagged but not performed here since it was not requested.
- No re-run of the walk-forward validation — all numbers are the already-verified
  2026-08-15 figures, packaged, not recomputed.
