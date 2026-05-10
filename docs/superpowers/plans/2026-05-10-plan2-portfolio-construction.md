# Plan 2 — Portfolio Construction Overhaul

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the current rank-weighting heuristic with a proper mean-variance optimizer (MVO) using `cvxpy`. Add Black-Litterman blending of quant alpha signals with LLM views. Integrate factor constraints from Plan 1. Add transaction-cost-aware optimization so the system naturally penalizes unnecessary turnover. Regime-conditional covariance ensures the risk model reflects crisis correlations when they matter most.

**Architecture:** `sector_constrained_weighted()` becomes a thin wrapper that calls the new MVO optimizer as primary path, falling back to rank-weighting if cvxpy is unavailable or if the optimization is infeasible. Black-Litterman runs as a pre-processing step that adjusts the alpha signal before it enters the optimizer — it does not replace the optimizer. Factor constraints from Plan 1 are injected as linear inequality constraints. Transaction cost penalty is a quadratic term in the objective.

**Tech Stack:** Python 3.12, `cvxpy` (install: `pip install cvxpy`), numpy, pandas, Plan 1 (`ascent/risk/`), existing `ascent/portfolio/optimizer.py`, `ascent/execution/cost_model.py`.

**Prerequisites:** Plan 1 (Factor Risk Model) must be complete before factor constraints can be wired in. MVO and Black-Litterman can be built independently of Plan 1 and wired together when Plan 1 is ready.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Install | `cvxpy` | `pip install cvxpy` — verify before starting |
| Create | `ascent/portfolio/mvo_optimizer.py` | cvxpy MVO — alpha, risk, turnover objective |
| Create | `ascent/portfolio/black_litterman.py` | BL view blending — quant + LLM alpha signals |
| Create | `ascent/portfolio/regime_covariance.py` | Regime-conditional covariance blending |
| Modify | `ascent/portfolio/optimizer.py` | MVO as primary path; rank-weight as fallback |
| Modify | `ascent/main.py` | Pass covariance and factor constraints to optimizer |
| Create | `tests/test_mvo_optimizer.py` | MVO optimizer tests — 12 tests |
| Create | `tests/test_black_litterman.py` | BL blending tests — 8 tests |

---

## Task 1: MVO Optimizer

**File:** `ascent/portfolio/mvo_optimizer.py`

### Steps
- [ ] 1.1 Install cvxpy: `pip install cvxpy`. Verify import succeeds.
- [ ] 1.2 Write `optimize_mvo(alpha_scores, covariance, current_weights=None, constraints=None, risk_aversion=1.0, turnover_penalty=0.002, max_weight=0.10, min_weight=0.02, top_n=15) -> pd.Series`. Arguments: `alpha_scores` is a Series (symbol → float), `covariance` is the full Σ from Plan 1 (or a diagonal proxy if Plan 1 is not yet available), `current_weights` is the existing portfolio (for turnover penalty). Returns a weight Series summing to 1.0.
- [ ] 1.3 Objective: maximize `w'α - λ·(w'Σw) - κ·‖w - w_prev‖₁` where `λ = risk_aversion` and `κ = turnover_penalty`. The L1 turnover penalty is linearized with auxiliary variables (cvxpy supports this natively via `cp.norm1`).
- [ ] 1.4 Constraints: (a) weights sum to 1.0, (b) 0 ≤ w_i ≤ max_weight for all i, (c) w_i = 0 for all symbols not in top-N ranked alpha, (d) factor exposure bounds if `constraints` argument is provided (list of cvxpy constraint objects from `factor_constraints.py`), (e) min_weight applied only to positions that are selected (mixed-integer relaxation: enforce min_weight only where w > 1e-4).
- [ ] 1.5 Sector constraint: before calling cvxpy, enforce `max_per_sector=1` by zeroing alpha for all but the top-ranked name per sector. This is done in pre-processing (not as a cvxpy constraint) to keep the problem convex.
- [ ] 1.6 Fallback: if cvxpy raises `cp.SolverError` or the problem is infeasible, log a warning and return `None`. Caller falls back to rank-weighting.
- [ ] 1.7 Write `_diagonal_covariance_proxy(n_assets, vol_estimate=0.25) -> np.ndarray`. Returns a diagonal covariance matrix using a flat vol estimate. Used when Plan 1 covariance is unavailable.

### Verification
```python
from ascent.portfolio.mvo_optimizer import optimize_mvo
import numpy as np, pandas as pd

alpha = pd.Series(np.random.randn(20), index=[f"S{i}" for i in range(20)])
cov = np.eye(20) * 0.04
weights = optimize_mvo(alpha, cov, top_n=10)
assert weights is not None
assert abs(weights.sum() - 1.0) < 1e-4
assert (weights <= 0.10 + 1e-6).all()
assert (weights >= 0).all()
```

---

## Task 2: Black-Litterman Blending

**File:** `ascent/portfolio/black_litterman.py`

### Steps
- [ ] 2.1 Write `black_litterman_views(quant_alpha, llm_alpha, tau=0.05, omega_scale=1.0) -> pd.Series`. Implements the Black-Litterman posterior expected return formula. `quant_alpha` is the composite z-scored alpha from the stack (prior view). `llm_alpha` is the LLM fundamental sleeve output (investor view). Returns posterior alpha that shrinks toward the LLM view weighted by `1/omega_scale` (lower omega = more confidence in LLM view).
- [ ] 2.2 BL formula: `μ_posterior = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹μ_prior + P'Ω⁻¹Q]`. For the simple case of absolute views (one view per stock, P = I): `μ_posterior = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ · [(τΣ)⁻¹μ_prior + Ω⁻¹Q]`. Omega is the view uncertainty matrix (diagonal, proportional to `omega_scale` × Σ).
- [ ] 2.3 When covariance is unavailable, fall back to a simple shrinkage: `μ_posterior = (1 - τ) · quant_alpha + τ · llm_alpha`, where τ is calibrated to the LLM sleeve's historical IC IR.
- [ ] 2.4 Write `get_blending_weight(llm_ic_ir) -> float`. Maps IC IR to a blending weight: IR < 0.3 → 0.05 weight, IR 0.3–0.6 → 0.10, IR > 0.6 → 0.15. Used to set `tau` dynamically as the LLM sleeve's live IC is measured.
- [ ] 2.5 BL runs in `ascent/portfolio/optimizer.py` as a pre-processing step before alpha enters the MVO. It modifies the alpha signal, not the optimizer structure.

### Verification
```python
from ascent.portfolio.black_litterman import black_litterman_views
import pandas as pd, numpy as np

syms = ["A", "B", "C", "D"]
quant = pd.Series([1.0, -1.0, 0.5, -0.5], index=syms)
llm = pd.Series([0.8, -0.8, 0.2, 0.2], index=syms)
posterior = black_litterman_views(quant, llm)
assert len(posterior) == 4
# Posterior should be between prior and view (shrinkage)
assert ((posterior - quant).abs() < (llm - quant).abs()).all()
```

---

## Task 3: Regime-Conditional Covariance

**File:** `ascent/portfolio/regime_covariance.py`

### Steps
- [ ] 3.1 Write `blend_regime_covariances(calm_bull_cov, stressed_cov, crisis_cov, regime_label, regime_confidence) -> np.ndarray`. Blends the three regime-conditional covariance matrices using the current regime probability as weights. In crisis at confidence 0.9, the output is 90% crisis covariance + 10% blended. In uncertain, weights are equal thirds.
- [ ] 3.2 Write `estimate_regime_covariance(prices_df, regime_labels, target_regime, min_obs=63) -> np.ndarray`. Computes sample covariance using only the dates labeled as `target_regime`. Falls back to full-history covariance if fewer than `min_obs` dates are available for that regime.
- [ ] 3.3 Wire into `ascent/main.py`: after regime detection, compute regime-conditional covariance and pass it to the portfolio optimizer. Cache regime-conditional covariances for 5 days (they are slow to compute).
- [ ] 3.4 Graceful fallback: if regime covariance estimation fails for any regime, use rolling 252-day sample covariance with Ledoit-Wolf shrinkage.

---

## Task 4: Optimizer Integration

**File:** `ascent/portfolio/optimizer.py` (modify existing)

### Steps
- [ ] 4.1 Add `sector_constrained_weighted_mvo(alpha_scores, regime_label, covariance=None, factor_constraints=None, current_weights=None, **config)` as a new function. This is the new primary path.
- [ ] 4.2 Sequence inside `sector_constrained_weighted_mvo`: (a) sector pre-screening (top-N × max_per_sector filtering), (b) BL blending if LLM alpha available, (c) call `optimize_mvo()` with factor constraints, (d) if MVO returns None (infeasible), fall back to existing `sector_constrained_weighted()` rank-weight path and log the fallback.
- [ ] 4.3 Keep `sector_constrained_weighted()` (the original function) intact as the fallback. Do not modify it.
- [ ] 4.4 Update `ascent/main.py` to call `sector_constrained_weighted_mvo` instead of `sector_constrained_weighted`. Pass covariance from Plan 1 if available; pass None otherwise (diagonal proxy used).
- [ ] 4.5 Add `optimization_method` field to the portfolio snapshot logged per rebalance: `"mvo"` or `"rank_weight_fallback"`. Visible in `logs/eod_log.jsonl`.

---

## Task 5: Transaction-Cost-Aware Turnover

### Steps
- [ ] 5.1 In `optimize_mvo()`, the turnover penalty `κ·‖w - w_prev‖₁` is already in the objective. Calibrate `turnover_penalty` (κ) to the live Almgren-Chriss cost estimates from `ascent/execution/cost_model.py`. Specifically: κ = expected_one_way_cost / 2 (half-spread approximation). This makes the penalty dimensionally consistent with the alpha objective (both in return space).
- [ ] 5.2 Write `compute_expected_turnover_cost(current_weights, proposed_weights, adv_estimates) -> float`. Returns the expected total transaction cost in return units. Logged each rebalance. Used to set κ in the optimizer.
- [ ] 5.3 Add `expected_turnover_cost` to the rebalance log entry.

---

## Task 6: Tests

### `tests/test_mvo_optimizer.py` — 12 tests
- [ ] `test_mvo_weights_sum_to_one` — weights sum to 1.0 ± 1e-4
- [ ] `test_mvo_max_weight_respected` — no position exceeds max_weight
- [ ] `test_mvo_top_n_enforced` — only top_n positions are nonzero
- [ ] `test_mvo_turnover_penalty_reduces_changes` — with prev_weights, changes are smaller than without penalty
- [ ] `test_mvo_risk_aversion_reduces_concentration` — higher λ → lower portfolio variance
- [ ] `test_mvo_infeasible_returns_none` — contradictory constraints → returns None, no exception
- [ ] `test_mvo_fallback_in_optimizer_py` — infeasible MVO → rank-weight result returned
- [ ] `test_mvo_factor_constraints_respected` — momentum exposure within bounds
- [ ] `test_mvo_diagonal_proxy_when_no_covariance` — runs without error when cov=None
- [ ] `test_sector_constraint_prescreen` — no two positions from same sector in output
- [ ] `test_optimization_method_logged` — log entry contains "optimization_method" field
- [ ] `test_expected_turnover_cost_positive` — cost ≥ 0 for any weight change

### `tests/test_black_litterman.py` — 8 tests
- [ ] `test_bl_posterior_between_prior_and_view`
- [ ] `test_bl_high_confidence_view_dominates`
- [ ] `test_bl_low_confidence_view_minimal_impact`
- [ ] `test_bl_missing_llm_alpha_returns_quant`
- [ ] `test_bl_handles_partial_overlap` — LLM alpha covers subset of symbols
- [ ] `test_bl_fallback_shrinkage_when_no_covariance`
- [ ] `test_blending_weight_maps_ir_to_tau`
- [ ] `test_regime_covariance_blend_weights_sum_to_one`

---

## Acceptance Criteria

1. MVO optimizer runs successfully on every rebalance day
2. Fallback to rank-weighting logs clearly; suite still passes if cvxpy removed
3. Expected turnover cost logged per rebalance; measurable reduction vs. pre-MVO baseline after 30 days
4. Factor constraints from Plan 1 respected when available (violations logged to zero within 2 rebalances)
5. BL blending active when LLM fundamental sleeve IC IR > 0.30
6. All 20 new tests passing; full suite passing
