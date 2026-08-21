# Phase 2 & Phase 3 — Ready-to-Implement Skeletons

## Phase 2: DSR/PBO on `PerformanceAnalyzer`

Grounded in `ascent/research/wf_framework/metrics.py` as it exists today.

**Important correction to `02_alpha_research.md`'s plan**: `FoldResult`
(`metrics.py:20-25`) currently stores only a **scalar** `is_sharpe` per fold
plus one `oos_returns` series — it does not, and structurally cannot, carry
"the whole trial distribution" DSR/PBO need. `walk_forward_efficiency()`
(`metrics.py:110-130`) only ever reads `fold.is_sharpe` as a single number per
fold. **DSR and PBO cannot be bolted onto `FoldResult` as it's currently
shaped** — this confirms (not contradicts) `02`'s Layer 5 finding that
`ParameterOptimizer.optimize()` must be changed to return the full per-trial
score distribution, not just the winner; that upstream change is a
prerequisite for this one, not parallel work.

**Skeleton — extend `FoldResult` and add two methods**:

```python
# metrics.py, extend the existing dataclass (additive, backward-compatible —
# new fields default to None so existing FoldResult(...) call sites don't break)
@dataclass
class FoldResult:
    fold_id:     int
    is_sharpe:   float
    oos_returns: pd.Series
    trial_sharpes: list[float] | None = None   # NEW: all IS Sharpes tried this fold, winner included
    n_trials:      int | None = None            # NEW: len(trial_sharpes), redundant but explicit


class PerformanceAnalyzer:
    ...
    def deflated_sharpe_ratio(self, observed_sharpe: float, trial_sharpes: list[float],
                                n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
        """Bailey & Lopez de Prado (2014). Returns P(true Sharpe > 0 | selection bias
        from len(trial_sharpes) trials). Needs the variance of trial_sharpes -- if
        only one trial was run, DSR reduces to the (undeflated) Probabilistic Sharpe Ratio."""
        n_trials = len(trial_sharpes)
        if n_trials <= 1:
            # No selection-bias correction possible with a single trial --
            # fall back to PSR against a zero benchmark, and say so in the caller's log.
            expected_max_sr = observed_sharpe
        else:
            # Expected max Sharpe under the null (all trials pure noise),
            # via the standard extreme-value approximation over trial variance.
            sr_std = float(np.std(trial_sharpes, ddof=1))
            euler_gamma = 0.5772156649
            expected_max_sr = sr_std * (
                (1 - euler_gamma) * _norm_ppf(1 - 1.0 / n_trials)
                + euler_gamma * _norm_ppf(1 - 1.0 / (n_trials * np.e))
            )
        # PSR of observed_sharpe against expected_max_sr as the benchmark,
        # skew/kurtosis-adjusted standard error (Bailey & Lopez de Prado eq. 5-7).
        sr_se = np.sqrt((1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2) / (n_obs - 1))
        return float(_norm_cdf((observed_sharpe - expected_max_sr) / sr_se))

    def pbo_cscv(self, fold_results: list[FoldResult], n_partitions: int = 16) -> float:
        """Bailey, Borwein, Lopez de Prado & Zhu (2017), Combinatorially Symmetric
        Cross-Validation. Requires per-fold trial-level IS/OOS pairs, not just the
        winner -- this needs fold_results[i].trial_sharpes populated AND a matching
        per-trial OOS series, which FoldResult does not yet carry (see note above).
        Stub signature only this cycle -- full CSCV implementation needs the
        upstream ParameterOptimizer change first; do not implement against
        incomplete data that would silently under-count trials."""
        raise NotImplementedError(
            "pbo_cscv requires per-trial OOS returns, not yet captured by "
            "FoldResult/ParameterOptimizer.optimize() -- see 02_alpha_research.md "
            "Layer 5 for the required upstream change first."
        )
```

`_norm_ppf`/`_norm_cdf` are `scipy.stats.norm.ppf`/`.cdf` — `metrics.py`
currently imports only `numpy`/`pandas` (`metrics.py:16-17`), so this is one
new dependency to add (`scipy` is very likely already a project dependency
elsewhere given XGBoost/CVXPY are in use — verify with `grep scipy
requirements*.txt` before assuming, not confirmed this cycle).

**Honest scope note**: `deflated_sharpe_ratio()` above is implementable now,
against the existing `FoldResult.is_sharpe` history if the caller collects
multiple folds' `is_sharpe` values as its "trial" list (a reasonable
approximation — each walk-forward fold's IS optimization is itself one
trial). `pbo_cscv()` is correctly stubbed as blocked, not faked — implementing
it against data that doesn't actually capture per-trial OOS performance would
produce a PBO number that looks precise but measures the wrong thing, which
is worse than not having it.

## Phase 3: `ascent/execution/compliance_gate.py`

Grounded in `ascent/execution/eod_runner.py` and `ascent/execution/
order_engine.py` as they exist today.

**Confirmed exactly as `03_trading_execution.md` described**:
`LARGE_TRADE_THRESHOLD_PCT = 2.0` (`eod_runner.py:48`) has **exactly one
reference in the file — its own definition**. Zero enforcement call sites.
This re-confirms the dead-constant finding precisely, not just approximately.

**Exact order-submission sequence**, `run_eod_with_weights()`
(`eod_runner.py:766` onward — confirmed this cycle, not just cited from
memory):
```
:1000  orders, diff_df = compute_orders(target_weights, current_positions, portfolio_value, ...)
:1013  kill_switch.check(current_nav=portfolio_value)          # portfolio-level, unchanged
:1029  cancel_all_orders()
:1034  for order in orders:
:1050      submit_order(order.symbol, qty=qty, side=order.side)
```

**Exact insertion point**: between line 1013 (kill-switch check) and line
1029 (`cancel_all_orders()`), operating on the `orders` list already produced
at line 1000. This is the one place in the function where the full order
batch exists as a list, before any are cancelled/submitted.

**Skeleton**:

```python
# ascent/execution/compliance_gate.py
"""Pre-Trade Compliance Checker -- Trading & Execution department.
Final gate before order submission. Does not decide *what* to trade, only
approves/rejects/resizes on compliance grounds. Runs after the kill switch
(portfolio-level circuit breaker) and before cancel_all_orders()."""

from dataclasses import dataclass

LARGE_TRADE_APPROVAL_PCT = 2.0   # matches eod_runner.py's existing (dead) LARGE_TRADE_THRESHOLD_PCT


@dataclass
class GateDecision:
    order_id: str
    approved: bool
    reason: str = ""


def check_batch(orders: list, portfolio_value: float, buying_power: float,
                 live_positions: dict, restricted_symbols: set = frozenset()) -> list[GateDecision]:
    decisions = []
    running_buy_dollars = 0.0
    # Sort so large-conviction buys are evaluated first if buying power runs out --
    # matches the existing sells-before-buys sequencing implicit in order_engine.py:102.
    for order in orders:
        if order.symbol in restricted_symbols:
            decisions.append(GateDecision(order.symbol, False, "restricted_list"))
            continue

        notional = abs(order.qty * order.price) if hasattr(order, "price") else None
        pct_of_nav = (notional / portfolio_value * 100) if notional and portfolio_value else None
        if pct_of_nav and pct_of_nav > LARGE_TRADE_APPROVAL_PCT:
            decisions.append(GateDecision(order.symbol, False,
                f"large_order_requires_approval ({pct_of_nav:.1f}% NAV > {LARGE_TRADE_APPROVAL_PCT}%)"))
            continue

        if order.side == "buy" and notional:
            if running_buy_dollars + notional > buying_power:
                decisions.append(GateDecision(order.symbol, False, "insufficient_buying_power"))
                continue
            running_buy_dollars += notional

        decisions.append(GateDecision(order.symbol, True, "approved"))
    return decisions
```

**Wiring into `eod_runner.py`** (insert after line 1013, before line 1029):

```python
from ascent.execution.compliance_gate import check_batch
_decisions = check_batch(orders, portfolio_value,
                          buying_power=get_account()["buying_power"],
                          live_positions=current_positions)
_rejected = {d.order_id for d in _decisions if not d.approved}
if _rejected:
    for d in _decisions:
        if not d.approved:
            print(f"[ComplianceGate] Rejected {d.order_id}: {d.reason}")
    orders = [o for o in orders if o.symbol not in _rejected]
```

**Note on `order.price`**: this cycle did not confirm whether the `Order`
object produced by `compute_orders()` carries a `price` field at
compliance-check time or only a `qty`/`side` — if price isn't populated yet
at this point in the sequence, `notional`/`pct_of_nav` will need to source
current price from `current_positions` or a fresh quote instead. Verify
`Order`'s dataclass definition in `order_engine.py` before implementing —
flagged rather than guessed.
