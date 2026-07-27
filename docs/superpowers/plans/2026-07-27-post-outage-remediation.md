# Post-Outage Remediation Plan — 2026-07-27

Written after a 27-day pipeline outage (last run 2026-06-30) and a −3.40% drawdown
concentrated in two positions (ALGM −33.9%, MRNA −22.8%).

Four Opus investigations produced the root causes below. This plan converts them into
minimal, test-driven changes. **No trading is performed by this work.**

---

## Findings that changed the diagnosis

1. **ALGM and MRNA were not discovery picks.** They were ordinary `us_equities` agent
   picks from the daily orchestrator recompute. Discovery only inserted SCHH.
2. **The equal 4.087% weight is not `1/N`.** It is the MVO hitting its 10% per-name box
   cap on every name — a degenerate LP, because the real covariance matrix is silently
   discarded and replaced with an identical-variance proxy.
3. **The adversarial trim never executed.** On discovery days `position_changes` is
   discarded. The judge's reasoning was economically irrelevant that day.
4. **The daily job was never scheduled.** No launchd agent installed; runs were manual.

---

## W1 — Sizing: stop the degenerate equal-weight LP

**Problem (verified):**
- `ascent/main.py:683-689` builds `mvo_covariance` over the full ~900-symbol alpha
  universe; `ascent/portfolio/optimizer.py:381` guards on
  `covariance.shape[0] == len(alpha_screened)` (15) → never true → covariance dropped.
- `ascent/portfolio/mvo_optimizer.py:19-22` then substitutes
  `np.eye(n) * (0.25/√252)²` — identical variance for every asset, zero correlation.
- With `risk_aversion=1.0` (`optimizer.py:312`) the quadratic term's marginal cost is
  ~5e-5 versus an alpha spread 3–4 orders of magnitude larger. The problem collapses to
  an LP whose vertex solution is `1/max_weight = 10` names at exactly `0.10`.
- Consequence: SGOV (0.2% vol) receives the same slot as BRBR (75.8% vol). Single names
  are 35.4% of capital but 74.5% of risk.
- `_apply_inverse_vol_tilt()` exists and is enabled, but `ascent/main.py:741-746`
  overwrites the tilted live row with the untilted MVO output. The tilt is backtest-only.

**Change (in priority order):**

1. **Per-name risk-budget cap** — new `enforce_risk_budget_cap(weights, vols, budget)` in
   `ascent/portfolio/optimizer.py`, mirroring the existing `_water_fill_cap` /
   `enforce_cluster_cap` pattern: no name may contribute more than `budget` (default
   **1.2%** annualized) of `wᵢ × σᵢ`; excess redistributed pro-rata to uncapped names;
   sum-to-1 preserved. Wire it in `ascent/main.py` alongside the cluster cap
   (`main.py:750-764`). Config flag `risk_budget_cap_enabled: bool = True`,
   `risk_budget_per_name: float = 0.012` in `ascent/config/settings.py`.
   *Estimated benefit: −2.32pp → −1.63pp on this episode (~0.69pp saved).*

2. **Repair the covariance handoff** — in `ascent/portfolio/optimizer.py:379-382`,
   reindex the covariance matrix to `alpha_screened.index` using `cov_result["symbols"]`
   instead of comparing shapes. If reindexing yields a degenerate/identity matrix, log it
   loudly and fall through to the proxy.

3. **Make the fallback proxy per-name** — `mvo_optimizer.py:19-22` should build
   `np.diag(σᵢ²)` from trailing realized vol rather than a single 0.25 for all names.

4. **Raise `risk_aversion`** so the quadratic term actually binds. Default 1.0 → make it
   configurable and set it high enough (order 1e2) that the solution is interior, not
   bang-bang. Must keep `test_mvo_risk_aversion_reduces_concentration` passing.

**Must not break:** `tests/test_mvo_optimizer.py` (esp. `test_mvo_max_weight_respected`,
`test_mvo_weights_sum_to_one`), `tests/portfolio/test_risk_construction.py`
(`test_panel_integration_postconditions` asserts sum==1 and max<=cap),
`tests/portfolio/test_exposure.py` (production and WF must share overlay code),
`tests/test_phase1_hardening.py:84`. `ascent/main.py` must keep returning a 10-tuple.

**Mirror requirement:** any sizing change must also be reflected in
`ascent/research/wf_framework/ascent_strategy.py` (reads `inverse_vol_tilt` /
`cluster_cap_enabled` at `:201`) or research and production silently diverge.

---

## W2 — Discovery day performs a full unscheduled rotation

**Problem (verified):** the add-only guard works, but is fed the wrong base book.
- `run_all_agents.py:1086` recomputes the entire target book every day.
- `run_all_agents.py:1201,1208` pass that *fresh target* into `_trigger_mini_rebalance`
  as `current_weights`. The live Alpaca book is never read on this path.
- `_insert_candidate_weights` correctly adds one name to that fresh book (SCHH landed at
  exactly 1/22, proving it ran).
- `run_all_agents.py:2487` then calls `run_eod_with_weights(..., force=True)`, and
  `eod_runner.py:796,827` diffs the submitted book against **live positions** — so every
  name absent from the fresh book becomes a full exit. Result on 2026-06-30: 27 orders,
  7 complete exits.
- "Add-only" was enforced against a book nobody was holding.

**The correct pattern already exists in the same file** at `run_all_agents.py:2309-2318`
(the falsifier-trim path), which reads live positions first and falls back to
`current_weights` only if empty.

**Change:**
1. Add helper `_live_book_or(fallback)` that returns live Alpaca weights, falling back to
   `fallback` when the broker returns nothing. Extract from the existing 2309-2318 block
   so both paths share one implementation.
2. Use it inside `_trigger_mini_rebalance` (~`run_all_agents.py:2426`) so the insert
   operates on the live book.
3. Use it for `_current_universe` at `run_all_agents.py:1201` (the "don't rediscover what
   we already hold" filter).
4. **Safety assertion** before submission (~`run_all_agents.py:2480-2488`): a discovery
   insert must never produce a complete exit and must not add more than one new name.
   Abort loudly and log if violated.

**Tests:** `tests/strategy/test_discovery_guards.py` (10 tests, all currently pass) only
unit-tests the helpers in isolation — none exercise the call site. Add:
- base book fed to the insert is the live book, not the recomputed target;
- resulting order set contains zero full exits;
- fallback to `current_weights` when broker returns empty.

---

## W3 — Liveness and alerting (the outage itself)

**Problem (verified):**
- `com.ascentcapital.eod.plist` is **not installed** in `~/Library/LaunchAgents/`.
- Its paths point at `/Users/kdong/Downloads/...`, a user that does not exist.
- `logs/launchd_stderr.log` ends 2026-04-13 with 13× `Operation not permitted` (macOS TCC
  blocking execution from `~/Downloads`). Last scheduled success: 2026-04-03.
- Every run since has been manual. Run dates in `eod_log.jsonl` show 19-day gaps.
- `ascent/monitoring/alert_system.py` is only ever called from inside the pipeline
  (`run_all_agents.py:1078`) — **the watchdog lives inside the thing it watches**. It has
  no staleness/liveness check, and `NTFY_TOPIC` is unset so `logs/alerts.jsonl` has never
  been created. Nothing could have told the user.

**Change:**
1. New `scripts/heartbeat_check.py` — **stdlib only**, no `ascent` import, so it cannot be
   broken by the failure it monitors. Reads max date in `logs/eod_log.jsonl`, counts
   missed NYSE trading days, WARN at ≥2, CRITICAL at ≥3 or any missed date in
   `rebalance_calendar.csv`. Writes `logs/liveness.json`, exits non-zero on breach.
2. New `scripts/com.ascentcapital.heartbeat.plist` using `StartInterval` (21600s), **not**
   `StartCalendarInterval` (interval jobs fire promptly after wake; calendar jobs silently
   skip). Modeled on the working `com.ascent.litellm.plist`. Do not auto-install — print
   the install command for the user to run.
3. Add a positive daily "system alive" ping, not only failure alerts. An alert channel
   that only fires on failure is indistinguishable from a broken one.
4. Fix `scripts/run_eod.sh` + `com.ascentcapital.eod.plist`: correct repo path, use
   `.venv/bin/python`, remove hardcoded Alpaca credentials (lines 23-25) in favour of
   `.env`, reconcile `ALPACA_API_KEY` vs `ALPACA_KEY` naming, fix the Hour (currently
   fires 7:45pm PT, intended 1:45pm PT), add the missing `Weekday` key.
5. **Catch-up guard** in `run_all_agents.py`: if the last logged run is more than N
   trading days old, refuse to auto-execute; require explicit `--catch-up`. Under
   `--catch-up`, run ONE fresh rebalance on today's data — never replay missed dates
   (stale intent, double transaction costs). Log as `run_type: "catch_up"` with skipped
   dates enumerated so the outage does not silently read as flat performance in the
   calibration and counterfactual series.

**Security:** `scripts/run_eod.sh:23-25` has live paper credentials in a git-tracked file.
Move to `.env` and flag to the user that these should be rotated.

---

## W4 — Adversarial layer: measure it before trusting it

**Problem (verified):**
- Trim magnitude is `min(regime_optimal, weight * 0.80)` — **independent of
  `adversarial_score`** (`debate/adversarial_engine.py:473`).
- `top_flags` is a hard `[:5]` ranked by `score × weight × 10`
  (`adversarial_engine.py:467,508`); the judge sees only `[:3]`
  (`debate/judge.py:84`). ALGM ranked 6th (priority 0.262 vs AES 0.278) and was filtered
  out twice.
- Authority is clamped to 1.0pp because `n_scored == 0` for every type
  (`data_cache/adversarial_authority.json`). Ceiling on the entire layer: **1pp of one
  position per rebalance.**
- `score_pending_interventions()` is only called from
  `ascent/monitoring/weekend_runner.py:253-259`, and the last weekend run was
  **2026-06-06**. Seven weekends missed → nothing has ever been scored.
- The judge is fed `load_recent_verdict_outcomes(n=5)` raw, with **no min-n, shrinkage, or
  confidence interval** (`debate/judge.py:42,66`). A `min_samples=10` guard exists at
  `outcome_tracker.py:41` but is never called by production code.
- `_score_verdict` (`outcome_tracker.py:100-127`) scores a verdict on **whole-portfolio
  NAV over 14 days** — so a `reduce_size` is WRONG whenever the market rose. The 2026-04-15
  record the judge cited was a market-direction outcome being used as evidence about a
  single-name trim.
- Regime key split: 4/15 stored `"RegimeLabel.STRESSED"`, creating a separate credibility
  bucket from `"stressed"`.
- Measured signal quality on the one scoreable cross-section: **Spearman ρ = +0.012
  (p=0.96)** between `adversarial_score` and forward 1-month excess return. Indistinguishable
  from noise on n=22.

**Change (deliberately conservative — measure first, do not tune):**
1. Call `score_pending_interventions()` from the daily path so scoring stops depending on
   a weekend job that has not fired since June 6.
2. De-duplicate `record_intervention` writes (2026-06-10 logged VNO 4×, VOYA 2×).
3. Add a min-sample guard + regime-key normalization to `load_recent_verdict_outcomes`
   and `load_credibility_context` so a single scored outcome cannot be presented to the
   judge as a track record.
4. Do **not** re-tune trim magnitude or the `[:5]`/`[:3]` cutoffs yet. With ρ≈0.01 there is
   no evidence the score ranks risk; tuning it now would be fitting noise. Revisit after
   ~30 scored interventions.
5. Reconcile CLAUDE.md integrity constraint #5 with reality: the judge's single change
   **does** write to the execution path via `run_all_agents.py:1789-1866`. Either document
   it as a sanctioned 1pp-capped exception or move it. Documentation change only.

---

## Sequencing

- W1, W2, W3 touch disjoint files → parallel.
- W4 touches `run_all_agents.py`, as does W2 → run W4 after W2 lands.
- Every workstream: TDD, `ast.parse` after each patch, run the named test suites, report
  pre-existing failures separately from new ones.

## Out of scope

- No rebalance, no orders, no cache swaps.
- No re-tuning of adversarial thresholds (see W4.4).
- Production `prices_live` remains as-is.
