# Plan 1 — Factor Risk Model (Barra-Style)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a factor-structured risk model that decomposes every portfolio position into systematic exposures (momentum, quality, value, size, low-vol, growth, leverage) and separates that from idiosyncratic risk. Wire it into portfolio construction, attribution, and the debate layer. This is the single biggest structural gap between Ascent and an institutional system — without it, the system doesn't know what bets it is actually making.

**Architecture:** Four components built in sequence. (1) Factor data ingestion — download Fama-French 5-factor + momentum (UMD) returns from Kenneth French's data library (free, updated daily). (2) Factor loading computation — rolling 252-day OLS of each stock's excess return on factor returns, producing a loadings matrix B (symbols × factors) at each date. (3) Factor covariance — Σ = B·F·B' + D where F is the factor return covariance and D is diagonal residual variance; Ledoit-Wolf shrinkage on D. (4) Portfolio exposure — w'B gives the portfolio's tilt on each factor, expressed in standard deviations. Constraints are factor-bound limits fed into the optimizer (Plan 2). Attribution splits daily P&L into factor-explained vs. stock-specific. Debate devil's advocate receives live factor tilts in context.

**Tech Stack:** Python 3.12, numpy, pandas, scikit-learn (Ledoit-Wolf), scipy, cvxpy (installed in Plan 2), existing `ascent/data/store/parquet.py`.

**Prerequisites:** None. This plan is self-contained.

**⚠ Do not merge factor constraints into portfolio construction until Plan 2 is complete.** The factor model may be built and used for reporting/attribution before the optimizer is ready to consume constraints.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/risk/factor_data.py` | Download + cache FF5+UMD factor returns; daily update |
| Create | `ascent/risk/factor_model.py` | Rolling OLS factor loading computation; save loadings parquet |
| Create | `ascent/risk/factor_exposure.py` | Portfolio-level factor tilt from weights + loadings |
| Create | `ascent/risk/covariance_model.py` | Factor-structured covariance Σ = BFB' + D with LW shrinkage |
| Create | `ascent/risk/factor_constraints.py` | Factor bound constraint generator for optimizer |
| Modify | `ascent/monitoring/attribution.py` | Add factor P&L vs. idiosyncratic P&L split |
| Modify | `debate/agents.py` | Devil's advocate context includes live factor tilts |
| Modify | `ascent/dashboard/export_dashboard_data.py` | Export factor exposures to `dashboard/factor_exposures.json` |
| Modify | `run_all_agents.py` | Call factor loading update after price data fetch |
| Create | `tests/test_factor_risk_model.py` | Full test suite — 18 tests |

---

## Task 1: Factor Data Ingestion

**File:** `ascent/risk/factor_data.py`

### Steps
- [ ] 1.1 Download Fama-French 5 factors + UMD (momentum) from Kenneth French's data library via direct URL (`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip` and the UMD equivalent). Parse the CSV. Store as `data_cache/factor_returns.parquet` with columns `[Mkt-RF, SMB, HML, RMW, CMA, UMD, RF]` and a DatetimeIndex.
- [ ] 1.2 Write `fetch_factor_returns(start_date, end_date) -> pd.DataFrame`. Always reads local cache first; re-downloads only when cache is > 1 day stale.
- [ ] 1.3 Write `get_factor_returns(start_date=None, end_date=None) -> pd.DataFrame`. Public interface for all consumers. Returns the full cache sliced to the requested window.
- [ ] 1.4 Wire `update_factor_data()` into `run_all_agents.py` immediately after price data fetch (before feature computation). Wrap in try/except — failure logs a warning but does not abort the runner.
- [ ] 1.5 Add `data_cache/factor_returns.parquet` to `.gitignore` if not already present.

### Verification
```python
from ascent.risk.factor_data import get_factor_returns
df = get_factor_returns("2020-01-01", "2026-04-30")
assert not df.empty
assert "UMD" in df.columns
assert "Mkt-RF" in df.columns
assert df.index.dtype == "datetime64[ns]"
```

---

## Task 2: Factor Loading Computation

**File:** `ascent/risk/factor_model.py`

### Steps
- [ ] 2.1 Write `compute_factor_loadings(prices_df, factor_returns, lookback=252, min_obs=126) -> pd.DataFrame`. For each date in `prices_df.index[lookback:]` and each symbol, run OLS of the stock's daily excess return (stock return minus RF) on the 6 factors over the trailing `lookback` days. Return a DataFrame indexed by (date, symbol) with columns `[beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, beta_umd, alpha, r2]`.
- [ ] 2.2 Vectorize: compute all symbols at once per date using `np.linalg.lstsq` on the factor matrix. Do not loop per symbol — this must run in under 60 seconds for 901 symbols × 1,500 dates.
- [ ] 2.3 Persist loadings to `data_cache/factor_loadings.parquet`. Only recompute dates not already in cache (incremental update).
- [ ] 2.4 Write `get_factor_loadings(as_of_date) -> pd.DataFrame`. Returns the loadings for a single date as a DataFrame indexed by symbol with columns `[beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, beta_umd]`. Uses most recent available date if `as_of_date` has no entry (forward-fill).
- [ ] 2.5 `update_factor_loadings()` — incremental update for the most recent 5 days. Called in `run_all_agents.py` after `update_factor_data()`. Wrap in try/except.

### Verification
```python
from ascent.risk.factor_model import compute_factor_loadings, get_factor_loadings
loadings = get_factor_loadings("2026-04-30")
assert loadings.shape[1] == 6
assert loadings.index.name == "symbol"
assert not loadings.isnull().all().any()
```

---

## Task 3: Factor-Structured Covariance

**File:** `ascent/risk/covariance_model.py`

### Steps
- [ ] 3.1 Write `compute_factor_covariance(factor_returns, lookback=252) -> np.ndarray`. Returns a 6×6 factor return covariance matrix estimated over the trailing `lookback` days. Use standard sample covariance (factor universe is small enough that shrinkage is not critical here).
- [ ] 3.2 Write `compute_residual_covariance(prices_df, factor_loadings, factor_returns, lookback=252) -> np.ndarray`. For each symbol, compute residual returns (stock return − B·f) and form a diagonal matrix D of residual variances. Apply Ledoit-Wolf shrinkage via `sklearn.covariance.LedoitWolf` on the full residual matrix, then take the diagonal. Returns a (n_symbols × n_symbols) diagonal matrix (as 1D array for efficiency).
- [ ] 3.3 Write `build_factor_covariance_matrix(weights_series, as_of_date, lookback=252) -> dict`. Returns `{"full": Σ, "factor": F, "loadings": B, "idiosyncratic": D}`. `Σ = B·F·B' + diag(D)`. Symbols are the index of `weights_series`.
- [ ] 3.4 Write `portfolio_variance(weights, covariance_matrix) -> float`. Standard `w'Σw`.
- [ ] 3.5 Write `factor_variance_decomposition(weights, loadings, factor_cov, idiosyncratic) -> dict`. Returns `{"factor_explained": float, "idiosyncratic": float, "total": float}`. Used by attribution.

### Verification
```python
from ascent.risk.covariance_model import build_factor_covariance_matrix, portfolio_variance
cov = build_factor_covariance_matrix(weights_series, "2026-04-30")
assert cov["full"].shape == (len(weights_series), len(weights_series))
var = portfolio_variance(weights_series.values, cov["full"])
assert var > 0
```

---

## Task 4: Portfolio Factor Exposure

**File:** `ascent/risk/factor_exposure.py`

### Steps
- [ ] 4.1 Write `compute_portfolio_exposures(weights_series, as_of_date) -> pd.Series`. Returns the portfolio's net exposure to each factor: `w'B` where B is the loadings matrix for `as_of_date`. Output is a Series indexed by factor name with values in loading units (approximately standard deviations of factor return).
- [ ] 4.2 Write `exposure_report(weights_series, as_of_date) -> dict`. Returns a human-readable dict: `{"exposures": Series, "dominant_factor": str, "concentration_score": float}`. Concentration score is the Herfindahl index of squared exposures (higher = more concentrated in one factor).
- [ ] 4.3 Write `check_factor_bounds(exposures, bounds=None) -> list[str]`. Default bounds: `{"beta_mkt": (-0.3, 0.3), "beta_umd": (-0.5, 0.5), "beta_smb": (-0.4, 0.4), "beta_hml": (-0.4, 0.4)}`. Returns a list of violations (empty = no violations). These are soft bounds — violations are logged, not rejected.
- [ ] 4.4 Wire `exposure_report()` into the post-portfolio-construction step in `ascent/main.py`. Log exposures and any violations at INFO level.
- [ ] 4.5 Export current exposures to `dashboard/factor_exposures.json` via `export_dashboard_data.py`.

### Verification
```python
from ascent.risk.factor_exposure import compute_portfolio_exposures, check_factor_bounds
exp = compute_portfolio_exposures(weights_series, "2026-04-30")
assert len(exp) == 6
assert exp.index.tolist() == ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "beta_umd"]
violations = check_factor_bounds(exp)
assert isinstance(violations, list)
```

---

## Task 5: Factor Constraints Generator

**File:** `ascent/risk/factor_constraints.py`

### Steps
- [ ] 5.1 Write `build_factor_constraints(loadings_df, bounds) -> list`. Returns a list of linear constraint dicts in the format expected by `cvxpy` (used in Plan 2 optimizer). Each constraint encodes `lb ≤ w'B_factor ≤ ub`.
- [ ] 5.2 Write `get_regime_factor_bounds(regime_label) -> dict`. Returns tighter bounds in crisis (momentum exposure capped at ±0.2, market beta capped at ±0.15) vs. calm_bull (standard bounds). Used by the optimizer to tighten risk constraints dynamically.
- [ ] 5.3 No integration into optimizer yet (Plan 2 prerequisite). This module is built and tested but only consumed by Plan 2.

---

## Task 6: Attribution Integration

**File:** `ascent/monitoring/attribution.py` (modify existing)

### Steps
- [ ] 6.1 Add `compute_factor_pnl(positions, factor_returns_today, loadings_as_of) -> dict`. Returns `{"factor_pnl": float, "idiosyncratic_pnl": float, "total_pnl": float, "by_factor": dict}`. Factor P&L = `w' · B · f_t` where `f_t` is today's factor return vector. Idiosyncratic = total − factor.
- [ ] 6.2 Add `factor_pnl` and `idiosyncratic_pnl` fields to every attribution log entry written to `logs/attribution_log.jsonl`.
- [ ] 6.3 Fail gracefully: if factor model data is unavailable for the date, write `factor_pnl: null` and log a warning. Never raise.

---

## Task 7: Debate Integration

**File:** `debate/agents.py` (modify existing)

### Steps
- [ ] 7.1 Add `_build_factor_exposure_context(portfolio_state) -> str`. Calls `exposure_report()` for the current weights. Returns a formatted string: factor name, exposure, and a plain-English interpretation (e.g., "momentum: +0.42σ — portfolio is tilted toward recent winners").
- [ ] 7.2 Inject the factor exposure string into the devil's advocate context only (the agent whose job is to find what is being missed). Bull and bear do not receive it — preserving asymmetric information.
- [ ] 7.3 Update `_build_agent_context()` dispatch table to route `"factor_exposures"` to devil's advocate.

---

## Task 8: Tests

**File:** `tests/test_factor_risk_model.py`

### Tests to implement
- [ ] `test_factor_data_downloads_and_caches` — fetch returns non-empty DataFrame with required columns
- [ ] `test_factor_data_is_incremental` — second call does not re-download; reads from cache
- [ ] `test_factor_loadings_shape` — returns (n_symbols, 6) with correct column names
- [ ] `test_factor_loadings_missing_symbol_returns_nan` — symbol not in price data → loadings are NaN, no crash
- [ ] `test_factor_covariance_is_positive_definite` — eigenvalues of Σ are all > 0
- [ ] `test_residual_covariance_is_diagonal` — D is diagonal; off-diagonals are zero
- [ ] `test_portfolio_variance_is_positive` — w'Σw > 0 for any nonzero weight vector
- [ ] `test_factor_variance_decomposition_sums_to_total` — factor + idiosyncratic = total ± 1e-6
- [ ] `test_portfolio_exposures_shape` — returns Series of length 6
- [ ] `test_portfolio_exposures_long_momentum_portfolio` — portfolio of high-UMD stocks has positive beta_umd
- [ ] `test_check_factor_bounds_no_violation` — exposures within bounds returns empty list
- [ ] `test_check_factor_bounds_violation` — exposure exceeding bound returns non-empty list
- [ ] `test_factor_pnl_sums_correctly` — factor_pnl + idiosyncratic_pnl ≈ total_pnl
- [ ] `test_attribution_log_includes_factor_fields` — log entry has factor_pnl and idiosyncratic_pnl keys
- [ ] `test_factor_constraints_builder_returns_list` — output is a list of dicts
- [ ] `test_regime_factor_bounds_tighter_in_crisis` — crisis bounds narrower than calm_bull
- [ ] `test_get_factor_loadings_graceful_on_missing_date` — uses most recent available date
- [ ] `test_factor_exposure_context_string_for_devil` — returns non-empty string with factor names

---

## Acceptance Criteria

1. `tests/test_factor_risk_model.py` — 18 tests passing
2. Every daily run writes factor exposures to `dashboard/factor_exposures.json`
3. Attribution log entries include `factor_pnl` and `idiosyncratic_pnl` fields
4. Devil's advocate debate context includes factor exposure summary
5. Runner does not crash when factor data is unavailable (graceful fallback throughout)
6. Full test suite (all existing + new tests) still passing
