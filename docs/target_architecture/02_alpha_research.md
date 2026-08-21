# Alpha Research / Quant Research Department — 5-Layer Blueprint

## Layer 1 — Department Mandate

**Purpose.** Convert raw hypotheses about market inefficiency into alpha sleeves
running at some validated capital weight inside `us_equities`, or reject them, with
every step logged for audit. The department's product is not "signals that backtest
well" — it is *promotion decisions with quantified overfitting risk attached*.

**Authority.** No role in this department can unilaterally move live capital. This
mirrors integrity constraint #5 in CLAUDE.md: the department can promote a signal
only as far as *paper trading*, and even a passing paper-trade result produces a
**proposal**, not an order. Moving a validated signal into `us_equities`'s live
weight vector requires sign-off from a Risk/CIO function outside this department
(this blueprint assumes a `risk_cio_agent` that does not yet exist, and flags it as
a hard external dependency, not something Quant Research can proxy for itself). This
mirrors the pattern already enforced for the debate judge, the AI PM blend, and the
falsifier trim — the department keeps `SELF_MODIFY_ENABLED`-style promotions
**advisory until an outside authority signs**.

**Reporting.** Weekly research log (`logs/self_improve_log.jsonl` today, extended
per Layer 5) to the CIO function; a structured verdict object per candidate signal
(Layer 4) consumed by the Breadth Manager and ultimately surfaced in the rebalance
recap's "things to watch for" section when a signal is in the paper-trading queue.

## Layer 2 — Roles

| Role | Owns | Can approve/reject | Escalates |
|---|---|---|---|
| **Idea Generation / Signal Researcher** | Hypothesis catalog: signal thesis, expected IC sign, economic rationale, target universe | Nothing — filters out ideas with no falsifiable thesis before they consume backtest compute | Ideas with no available point-in-time data to Data Eng (outside this department) |
| **Backtest Engineer (in-sample)** | IS parameter search, `param_grid`, execution-cost model | Rejects a candidate if it has no positive-Sharpe region anywhere in the grid | Candidates that pass IS to the Validation Statistician |
| **Validation Statistician** | OOS/walk-forward run, Deflated Sharpe Ratio, PBO, WFE | **Gate 1**: pass/fail into the paper-trading queue | Borderline cases (PBO 10-20%) to CIO for discretionary review |
| **Paper-Trading Supervisor** | Live-vs-backtest parity: fill quality, slippage, latency, tracking error | **Gate 2**: pass/fail into the live-capital ramp | Infra faults (broker errors, data gaps) to Eng, not CIO |
| **Breadth/Portfolio-of-Signals Manager** | Cross-signal correlation matrix, per-sleeve capital allocation, `DEFAULT_ALPHA_WEIGHTS` | Sizes a passed signal's *initial* weight inside the cap; can veto on correlation grounds even after Gates 1-2 pass | Final live-weight change to the (not-yet-built) Risk/CIO sign-off role |

Only the Breadth Manager touches anything resembling `DEFAULT_ALPHA_WEIGHTS`; only
the CIO role (external) can flip a live weight from zero to nonzero in production.
This closes the exact hole `self_improve.py` currently has, where
`run_self_improve()` writes straight to `data_cache/active_alpha_config.json` with
no statistical gate and no second sign-off.

## Layer 3 — Per-Role Responsibilities and Decision Logic

**Idea Generation / Signal Researcher**
- Emits a `HypothesisCard`: thesis, universe, expected holding period, expected IC
  sign, and *why it should be uncorrelated with meanrev/statarb* (breadth is the
  point — a third sleeve correlated 0.8 with statarb adds ~0 to IR ≈ IC·√breadth).
- Reject if the thesis can't be falsified within 12 months of data, or if universe
  < 10 symbols (mirrors the existing `alternatives_agent` long/short-leg floor
  already in the repo).

**Backtest Engineer (in-sample)**
- Runs `ParameterOptimizer.optimize(is_data)` (existing,
  `ascent/research/wf_framework/optimizer.py`) — grid search, Sharpe or Calmar
  objective, IS-only by construction (`optimize` never sees OOS data, per its own
  docstring).
- **Trial counting is mandatory and currently absent.** Every grid point evaluated
  (`len(list(product(*param_values)))`) must be logged — this number is the direct
  input to Deflated Sharpe Ratio's multiple-testing correction. Today `optimize()`
  returns only `(best_params, best_score)`; it must additionally return `n_trials`.
- Reject: no in-sample Sharpe > 0 anywhere in grid, or `constraint_fn` prunes to
  zero valid combos.

**Validation Statistician** — the net-new statistical core.
- Reads: stitched OOS equity curve + `n_folds` (from `PerformanceAnalyzer.
  full_report`, existing), `n_trials` from the Backtest Engineer, and the
  correlation of trial Sharpes (needed for PBO's combinatorially-symmetric
  cross-validation, CSCV).
- Computes, using Bailey & López de Prado's formulas (net new — nothing in
  `metrics.py` computes either today):
  - **Deflated Sharpe Ratio (DSR)**: corrects the observed Sharpe for `n_trials`
    and the variance of trial Sharpes (skew/kurtosis-adjusted). Threshold to
    *pass*: **DSR > 0.5** (i.e., >50% probability the true Sharpe exceeds zero
    after correcting for selection bias) is the loose academic floor; this
    blueprint sets the practical bar at **DSR-implied p-value < 0.05, minimum
    12-month OOS window (≥ 63×4 trading days, i.e. at least 4 of the existing
    63-day WF folds)** — grounded in the repo's own current OOS window of
    `OOS_WINDOW=63` days per fold, extended to require multiple stitched folds
    before a signal is eligible at all.
  - **Probability of Backtest Overfitting (PBO)** via CSCV: split the trial
    history into ≥10 combinatorial train/test partitions (Bailey & López de Prado
    recommend 16); PBO = fraction of partitions where the IS-best configuration
    ranks below median OOS. **Threshold: PBO < 20%** (their paper's illustrative
    cutoff for "overfitting risk is low"; below 50% is "better than random," but
    20% is the concrete, defensible number to cite when it's a made-up-but-grounded
    threshold, and this blueprint says so explicitly).
  - **WFE** (already computed, `walk_forward_efficiency` in `metrics.py`) folded in
    as a secondary check: **WFE ≥ 0.5** (the file's own docstring: "< 0.5:
    significant overfitting — do not trade live").
- **Gate 1 decision**: PASS only if DSR-p < 0.05 **AND** PBO < 20% **AND** WFE ≥
  0.5 **AND** OOS window ≥ 12 months. Any single failure → REJECT, logged with
  reasoning, no appeal within this department (CIO can override only 10-20% PBO
  borderline cases, never a WFE < 0.5 or DSR failure — that combination is not
  discretionary).

**Paper-Trading Supervisor**
- A passed signal enters a shadow config (repo already has this scaffold:
  `SHADOW_DIR = data_cache/shadow_configs`, `_promote_to_shadow`, 30-day
  `shadow_expires`) but paper trading here means *real order submission through
  Alpaca paper*, not just Sharpe monitoring.
- **Minimum paper-trading window: 30 trading days** (matches the existing 30-day
  shadow expiry already coded, and the `SELF_MODIFY_ENABLED` comment's own bar of
  "30 consecutive trading days"). Real funds typically run 1-3 months minimum; this
  blueprint keeps the repo's existing 30-day number rather than inventing a new one.
- **Pass condition**: live-vs-backtest daily tracking error < **150 bps/day**
  annualized-equivalent (a concrete, reasonable but made-up number — real desks
  target single-digit-bp deviation on liquid large caps; this is loosened for a
  2-sleeve statarb/meanrev book with wider natural noise), fill-rate ≥ 98%, no
  data-integrity halts (reuses the repo's existing duplicate/phantom-row detectors
  in `reconcile_numbers.py`).
- Reject → back to Validation Statistician with a diagnosis (infra vs alpha decay);
  does not re-run DSR/PBO, just flags for re-review.

**Breadth/Portfolio-of-Signals Manager**
- Reads the full signal correlation matrix (pairwise return correlation over the
  paper-trading window) across meanrev, statarb, and any candidate signal that
  passed Gate 2.
- **Sizing rule, initial live ramp** (concrete, stated as invented-but-grounded):
  initial live allocation capped at **10% of the signal's ultimate target weight**
  for the first **3 scheduled rebalances**, doubling each subsequent rebalance if
  tracking error stays within the paper-trading band, capped at full target weight
  by rebalance 6 — a staged ramp rather than the current binary
  `SELF_MODIFY_ENABLED` on/off.
- Veto condition: pairwise correlation with an existing sleeve > **0.6** over the
  paper window → signal is either rejected outright or must replace (not add to)
  the correlated sleeve, since IR ≈ IC·√breadth only pays off when breadth is
  *effective* (uncorrelated) breadth.
- Enforces the existing cross-key-set guard already required by CLAUDE.md
  constraint #6 (`DEFAULT_ALPHA_WEIGHTS` key sets must match between `stack.py` and
  `self_improve.py`) — extended to N sleeves instead of 2.

## Layer 4 — Interfaces / Data Contracts

```
HypothesisCard (Idea Gen -> Backtest Engineer)
  { signal_id, thesis, universe, expected_ic_sign, target_holding_days }

BacktestResult (Backtest Engineer -> Validation Statistician)
  { signal_id, best_params, is_sharpe, n_trials, trial_sharpe_distribution,
    param_grid_size, execution_model_id }

ValidationVerdict (Validation Statistician -> Breadth Manager, CIO log)
  { signal_id, deflated_sharpe, dsr_p_value, pbo, wfe, n_oos_folds,
    oos_window_days, pass: bool, reasoning }

ParityReport (Paper-Trading Supervisor -> Breadth Manager, CIO log)
  { signal_id, tracking_error_bps, fill_rate, days_live, halts, pass: bool }

CapitalProposal (Breadth Manager -> CIO/Risk sign-off -- NOT auto-applied)
  { signal_id, correlation_to_existing: {sleeve: corr}, proposed_ramp_schedule,
    target_weight, veto: bool, veto_reason }
```

Every object is append-only JSONL, mirroring the existing
`logs/self_improve_log.jsonl` pattern — this is what makes the department auditable
rather than just a script that mutates a JSON config.

## Layer 5 — Concrete Implementation Mapping

What exists today, read directly from the source:

- **`ascent/research/wf_framework/optimizer.py`** — `ParameterOptimizer.optimize()`
  is exactly the Backtest Engineer's IS search. Objective functions are
  `_sharpe`/`_calmar` only (lines 33-50); no trial count is returned, no
  overfitting correction. **Change**: `optimize()` must additionally return
  `n_trials` and a list of per-combo scores (not just the best), since DSR/PBO need
  the whole trial distribution, not the winner.
- **`ascent/research/wf_framework/metrics.py`** — `PerformanceAnalyzer` computes
  Sharpe/Sortino/CAGR/WFE (`walk_forward_efficiency`, lines 110-130) correctly and
  is IS/OOS-boundary-safe. **This is the Validation Statistician's existing base**
  — DSR and PBO are net-new methods to add to this class
  (`PerformanceAnalyzer.deflated_sharpe_ratio(...)`,
  `PerformanceAnalyzer.pbo_cscv(...)`), not a new file, since they consume the same
  `FoldResult` objects.
- **`ascent/alpha/stack.py`** — `DEFAULT_ALPHA_WEIGHTS` (line 16),
  `_load_active_alpha_weights` (line 24), and the existing IC-gate
  (`IC_GATE_THRESHOLD`, line 21, `_get_gated_weights`) are the **Breadth Manager's
  runtime enforcement point**. The IC gate already demotes underperforming sleeves
  live — the new `CapitalProposal` ramp schedule slots in as a companion promoter,
  reading `data_cache/active_alpha_config.json`'s `by_regime`/`global` sections the
  same way.
- **`ascent/research/self_improve.py`** — this is the department's control loop
  today, and it is where `SELF_MODIFY_ENABLED` (line 28) lives as a single blunt
  boolean gate. Concretely:
  - `generate_variants()` (line 80) = Idea Generation's random/LLM-guided
    hypothesis step — keep.
  - `evaluate_variant()` (line 191) = Backtest Engineer + a crude OOS check via
    `walk_forward_lightweight.run_lightweight_oos` — today this conflates IS and
    OOS into one call with no trial correction. **Split**: this function's OOS half
    must call the new `PerformanceAnalyzer.deflated_sharpe_ratio`/`pbo_cscv`, not
    just raw Sharpe minus a turnover penalty (`MIN_SHARPE_EDGE = 0.05`, line 43,
    has no statistical basis today).
  - `_promote_to_shadow()` (line 216) is literally the paper-trading queue entry
    point already — reuse, but gate its call site on `ValidationVerdict.pass`, not
    on `edge > MIN_SHARPE_EDGE` alone (line 289).
  - **The `SELF_MODIFY_ENABLED` boolean must be replaced**, not toggled. Where line
    91 (`if not SELF_MODIFY_ENABLED: return []`) and line 257 currently hard-stop
    the whole loop, the target architecture instead makes `SELF_MODIFY_ENABLED=True`
    permanently at the *variant-generation and paper-trading* stages (Layers 2-4
    are safe to always run — they only write to `shadow_configs` and log files) and
    introduces a **separate, narrower gate** — `LIVE_CAPITAL_RAMP_ENABLED`, checked
    only inside the not-yet-built CIO sign-off function — that is the sole switch
    controlling whether a `CapitalProposal` can ever touch
    `active_alpha_config.json`'s live-read `global`/`by_regime` weights. This
    directly satisfies CLAUDE.md's integrity constraint #5 pattern: judgment layers
    keep running and logging, nothing writes to live weights without a proven,
    external, artifact-backed gate.
- **Net-new modules**: `ascent/research/overfitting.py` (DSR + PBO/CSCV
  implementations, called from `metrics.py`'s new methods),
  `ascent/research/paper_trading_supervisor.py` (parity checker against Alpaca
  paper fills, reusing `alpaca_broker.get_portfolio_history()` per the existing
  gotcha about same-day equity being unreliable), and
  `ascent/strategy/capital_ramp.py` (the staged 10%->100% sizing schedule,
  replacing the binary promote/don't-promote in `_promote_to_shadow`).
