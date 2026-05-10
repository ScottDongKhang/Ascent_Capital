"""
Ascent Capital — Portfolio Optimizer
Converts alpha scores into portfolio target weights.
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from typing import Optional

try:
    from ascent.portfolio.mvo_optimizer import optimize_mvo
except Exception:
    optimize_mvo = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def apply_bl_to_latest(
    target_weights: pd.DataFrame,
    price_df: pd.DataFrame,
    alpha: pd.DataFrame,
    current_holdings: Optional[dict] = None,
    max_weight: float = 0.15,
    min_history: int = 126,
) -> pd.DataFrame:
    """
    Replace the last row of target_weights with Black-Litterman MV weights.

    Sector selection and stock screening from sector_constrained_weighted are
    preserved — BL only refines the intra-portfolio weight allocation using
    the covariance structure of the selected names.

    Parameters
    ----------
    target_weights  : DataFrame (dates × symbols) from sector_constrained_weighted
    price_df        : DataFrame with 'symbol', 'date', 'close' columns (raw long format)
    alpha           : DataFrame (dates × symbols) — composite alpha scores
    current_holdings: dict {symbol: weight} — live holdings for TC-aware sizing
    max_weight      : float — passed through to BL optimizer
    min_history     : int — minimum price history rows required

    Returns
    -------
    DataFrame with the last row replaced by BL weights (all other rows unchanged).
    """
    if target_weights is None or target_weights.empty:
        return target_weights

    try:
        from .mv_optimizer import bl_weights
        from .tc_aware_optimizer import tc_aware_weights

        last_date = target_weights.index[-1]

        # Selected names: symbols with non-zero weight on the last date
        last_row = target_weights.iloc[-1]
        selected = last_row[last_row > 0.001].index.tolist()
        if len(selected) < 3:
            return target_weights

        # Build daily returns matrix for selected names
        if isinstance(price_df.index, pd.DatetimeIndex):
            # Already pivoted (dates × symbols)
            close_wide = price_df[price_df.columns.intersection(selected)]
        else:
            # Long format: pivot first
            try:
                close_wide = price_df.pivot_table(
                    index="date", columns="symbol", values="close"
                )[selected]
            except Exception:
                return target_weights

        returns = close_wide.pct_change().dropna(how="all")
        returns = returns[[s for s in selected if s in returns.columns]]

        if len(returns) < min_history or len(returns.columns) < 3:
            return target_weights

        # Alpha z-scores for the last date
        alpha_syms = [s for s in selected if s in alpha.columns]
        if not alpha_syms:
            return target_weights

        alpha_last = alpha.reindex(index=[last_date], columns=alpha_syms).iloc[-1].dropna()
        if alpha_last.empty:
            # Use last available alpha row
            alpha_available = alpha[alpha_syms].dropna(how="all")
            if alpha_available.empty:
                return target_weights
            alpha_last = alpha_available.iloc[-1]

        # Run Black-Litterman
        bl_w = bl_weights(
            alpha_scores=alpha_last,
            returns=returns,
            max_weight=max_weight,
            min_history=min_history,
        )
        if bl_w is None or bl_w.empty:
            return target_weights

        # TC-aware sizing if we have current holdings
        if current_holdings:
            from .covariance import get_cov
            from .mv_optimizer import _equilibrium_returns, _bl_posterior, _TAU, _LAMBDA
            import numpy as _np
            syms = bl_w.index.tolist()
            ret_aligned = returns[syms].dropna(how="any")
            cov = get_cov(ret_aligned, min_periods=min_history)
            if cov is not None:
                n = len(syms)
                w_mkt = _np.ones(n) / n
                pi = _equilibrium_returns(cov, w_mkt)
                z = alpha_last.reindex(syms).fillna(0).values
                views = pi + z * 0.05
                view_conf = _np.clip(_np.abs(z) / 3.0, 0.05, 0.9)
                mu_bl_arr = _bl_posterior(pi, cov, views, view_conf, tau=_TAU)
                mu_bl = pd.Series(mu_bl_arr, index=syms)
                curr = pd.Series(current_holdings).reindex(syms).fillna(0)
                bl_w = tc_aware_weights(bl_w, curr, mu_bl, cov, max_weight=max_weight)

        # Replace last row
        result = target_weights.copy()
        result.loc[last_date] = 0.0
        for sym, w in bl_w.items():
            if sym in result.columns:
                result.loc[last_date, sym] = float(w)

        # Renormalize
        row_sum = result.loc[last_date].sum()
        if row_sum > 1e-8:
            result.loc[last_date] = result.loc[last_date] / row_sum

        print(f"[Portfolio] BL weights applied for {last_date.date()}: "
              f"{len(bl_w)} names, max={bl_w.max():.3f}, "
              f"HHI={((bl_w**2).sum()):.4f}")
        return result

    except Exception as e:
        print(f"[Portfolio] BL optimizer skipped: {e}")
        return target_weights


class SectorDataError(RuntimeError):
    """Raised when sector coverage is below the 80% threshold required for safe portfolio construction."""
    pass


# ── Shared helpers (used by multiple constructors below) ─────────────────────

def _water_fill_cap(scores: pd.Series, max_weight: float) -> pd.Series:
    """
    Bug 2 fix: proper iterative water-filling weight cap.

    Clip-and-renormalize is NOT safe — after renorm, previously-capped names
    can exceed max_weight again. This routine freezes capped names and
    redistributes only among uncapped names until convergence.

    Post-condition: all weights <= max_weight + 1e-9, sum == 1.0
    """
    if len(scores) == 0:
        return scores

    # Edge case: cap is too tight for the number of names
    if max_weight * len(scores) < 1.0 - 1e-9:
        # Infeasible — relax to equal weight
        return pd.Series(1.0 / len(scores), index=scores.index)

    raw_w = scores / scores.sum()
    tol = 1e-9

    for _ in range(50):  # should converge in <10 iters in practice
        over = raw_w > max_weight + tol
        if not over.any():
            break
        # Freeze overweight names at max_weight
        raw_w[over] = max_weight
        # Remaining weight to redistribute
        remaining = 1.0 - raw_w[over].sum()
        free_mask = ~over
        free_total = scores[free_mask].sum()
        if free_total > 0:
            raw_w[free_mask] = scores[free_mask] / free_total * remaining
        else:
            # All names are at cap — equal distribute remaining
            n_free = free_mask.sum()
            if n_free > 0:
                raw_w[free_mask] = remaining / n_free

    # Final hard clamp + renorm — loop until convergence
    for _ in range(10):
        raw_w = raw_w.clip(upper=max_weight)
        if raw_w.sum() > 0:
            raw_w = raw_w / raw_w.sum()
        if raw_w.max() <= max_weight + 1e-9:
            break

    return raw_w


def _normalize_sector(sec) -> str | None:
    """
    Bug 1 + Bug 12 fix: canonical missing-sector label.
    None, empty string, whitespace, and 'Unknown' all map to None (treated as missing).
    Import this in sweep scripts instead of duplicating sector_map.get(sym, 'Unknown').
    """
    if sec is None:
        return None
    if str(sec).strip().lower() in ("unknown", "nan", "none", ""):
        return None
    return str(sec).strip()


def top_n_equal_weight(
    alpha: pd.DataFrame,
    n: int = 10,
    max_weight: float = 0.15,
    min_weight: float = 0.02,
    long_only: bool = True,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < min(n, 3):
            continue
        n_actual = min(n, len(row))

        top = row.nlargest(n_actual)
        w = 1.0 / len(top)
        w = np.clip(w, min_weight, max_weight)

        for sym in top.index:
            weights.loc[dt, sym] = w

        row_sum = weights.loc[dt].sum()
        if row_sum > 0:
            weights.loc[dt] = weights.loc[dt] / row_sum

    return weights


def rank_weighted(
    alpha: pd.DataFrame,
    n: int = 10,
    max_weight: float = 0.15,
    min_weight: float = 0.02,
    long_only: bool = True,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < min(n, 3):
            continue
        n_actual = min(n, len(row))

        top = row.nlargest(n_actual)
        scores = top - top.min() + 1e-8
        # Bug 2 fix: use water-filling instead of clip-and-renormalize
        raw_w = _water_fill_cap(scores, max_weight)

        raw_w[raw_w < min_weight] = 0.0
        if raw_w.sum() > 0:
            raw_w = raw_w / raw_w.sum()
        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]

    return weights


def apply_turnover_penalty(
    new_weights: pd.DataFrame,
    prev_weights: pd.Series,
    turnover_penalty: float = 0.5,
) -> pd.Series:
    if prev_weights is None or prev_weights.empty:
        return new_weights
    return (1 - turnover_penalty) * new_weights + turnover_penalty * prev_weights


def enforce_constraints(
    weights: pd.Series,
    max_weight: float = 0.15,
    min_weight: float = 0.01,
    max_gross: float = 1.0,
) -> pd.Series:
    w = weights.copy()
    w[w.abs() < min_weight] = 0.0
    w = w.clip(-max_weight, max_weight)
    gross = w.abs().sum()
    if gross > max_gross and gross > 0:
        w = w * (max_gross / gross)
    return w


def sector_constrained_weighted_mvo(
    alpha_scores: pd.Series,
    regime_label: str = "calm_bull",
    covariance: Optional[np.ndarray] = None,
    factor_constraints: Optional[list] = None,
    current_weights: Optional[pd.Series] = None,
    sector_map: Optional[dict] = None,
    n: int = 15,
    top_n: Optional[int] = None,  # alias for n
    max_weight: float = 0.10,
    min_weight: float = 0.02,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.002,
    llm_alpha: Optional[pd.Series] = None,
) -> tuple[pd.Series, str]:
    """
    MVO-based portfolio construction with sector pre-screening.

    Returns (weights_series, optimization_method) where optimization_method
    is "mvo" or "rank_weight_fallback".

    Sequence:
      1. Sector pre-screening: keep best name per sector from top-N pool
      2. BL blending: adjust alpha with LLM views if available
      3. MVO: cvxpy optimization on screened universe
      4. Fallback: if MVO fails, use rank-weighting
    """
    if top_n is not None:
        n = top_n
    if sector_map is None:
        sector_map = {}

    row = alpha_scores.dropna()
    if row.empty:
        return pd.Series(dtype=float), "rank_weight_fallback"

    # ── Step 1: sector pre-screening ─────────────────────────────────────────
    ranked = row.sort_values(ascending=False)
    pool_size = min(n * 3, len(ranked))
    top_candidates = ranked.iloc[:pool_size]

    known_count = sum(
        1 for sym in top_candidates.index
        if _normalize_sector(sector_map.get(sym, "")) is not None
    )
    coverage = known_count / len(top_candidates) if len(top_candidates) > 0 else 0.0

    if coverage >= 0.80:
        selected = []
        sector_count: dict = {}
        for sym in ranked.index:
            sec = _normalize_sector(sector_map.get(sym, ""))
            bucket = sec if sec is not None else f"__unknown_{sym}__"
            if sector_count.get(bucket, 0) < 1:
                selected.append(sym)
                sector_count[bucket] = sector_count.get(bucket, 0) + 1
            if len(selected) >= n:
                break
        alpha_screened = ranked.reindex(selected).dropna()
    else:
        alpha_screened = ranked.iloc[:n]

    if alpha_screened.empty:
        return pd.Series(dtype=float), "rank_weight_fallback"

    # ── Step 2: Black-Litterman blending ─────────────────────────────────────
    from ascent.portfolio.black_litterman import black_litterman_views, get_blending_weight
    try:
        if llm_alpha is not None and not llm_alpha.empty:
            # Use default tau unless we have a measured IC IR
            tau = get_blending_weight(0.0)  # conservative until IC IR is measured
            alpha_bl = black_litterman_views(alpha_screened, llm_alpha, tau=tau)
        else:
            alpha_bl = alpha_screened
    except Exception as _bl_exc:
        log.warning("[MVO] BL blending failed: %s", _bl_exc)
        alpha_bl = alpha_screened

    # ── Step 3: slice covariance to screened universe ─────────────────────────
    cov_screened = None
    if covariance is not None and covariance.shape[0] == len(alpha_screened):
        cov_screened = covariance

    # ── Step 4: slice factor constraints to screened universe ─────────────────
    fc_screened = None
    if factor_constraints:
        screened_syms = alpha_screened.index.tolist()
        fc_screened = []
        for fc in factor_constraints:
            # Each fc["loadings_vector"] is aligned to the full symbol list.
            # We need to know what symbols that corresponds to — constraints
            # carry a "symbols" key when built by build_factor_constraints().
            # If missing, skip the constraint rather than crash.
            fc_syms = fc.get("symbols")
            if fc_syms is not None:
                idx = [i for i, s in enumerate(fc_syms) if s in screened_syms]
                if idx:
                    new_bvec = np.zeros(len(screened_syms))
                    for j, i in enumerate(idx):
                        pos = screened_syms.index(fc_syms[i])
                        new_bvec[pos] = fc["loadings_vector"][i]
                    fc_screened.append({
                        "loadings_vector": new_bvec,
                        "lb": fc.get("lb"),
                        "ub": fc.get("ub"),
                        "factor": fc.get("factor"),
                    })

    # ── Step 5: MVO ──────────────────────────────────────────────────────────
    if optimize_mvo is None:
        log.warning("[MVO] optimize_mvo not available — falling back to rank-weight")
        scores = alpha_screened - alpha_screened.min() + 1e-8
        rw = _water_fill_cap(scores, max_weight)
        rw[rw < 0.001] = 0.0
        if rw.sum() > 0:
            rw /= rw.sum()
        return rw, "rank_weight_fallback"
    try:
        mvo_weights = optimize_mvo(
            alpha_scores=alpha_bl,
            covariance=cov_screened,
            current_weights=current_weights,
            factor_constraints=fc_screened,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            max_weight=max_weight,
            min_weight=min_weight,
            top_n=len(alpha_bl),
        )
    except Exception as _mvo_exc:
        log.warning("[MVO] optimize_mvo raised: %s", _mvo_exc)
        mvo_weights = None

    if mvo_weights is not None and not mvo_weights.empty:
        return mvo_weights, "mvo"

    # ── Fallback: rank-weighting on screened universe ─────────────────────────
    log.warning("[MVO] Falling back to rank-weight for this rebalance")
    scores = alpha_screened - alpha_screened.min() + 1e-8
    rw = _water_fill_cap(scores, max_weight)
    rw[rw < 0.001] = 0.0
    if rw.sum() > 0:
        rw /= rw.sum()
    return rw, "rank_weight_fallback"


def sector_constrained_weighted(
    alpha,
    n=10,
    max_weight=0.15,
    max_per_sector=1,
    sector_map=None,
    regime_signal=None,
):
    """
    Sector-constrained rank-weighted portfolio.

    Bug 1 fix: when sector coverage on the candidate set is below 80%,
    sector caps are skipped and plain rank weighting is used instead.
    This prevents the entire universe collapsing to 1 name when
    profiles.parquet is absent (all symbols map to "Unknown" → same bucket).

    Bug 2 fix: weight cap enforced via water-filling, not clip-and-renormalize.

    If regime_signal is provided, max_weight is tightened based on the
    current regime (e.g. 0.08 cap in crisis vs 0.15 in calm bull).
    """
    if regime_signal is not None:
        try:
            from ascent.regime import regime_max_weight
            max_weight = regime_max_weight(max_weight, regime_signal)
        except Exception:
            pass

    if sector_map is None:
        sector_map = {}

    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < min(n, 3):
            continue
        n_actual = min(n, len(row))

        ranked = row.sort_values(ascending=False)

        # ── Check sector coverage before enforcing caps ────
        top_candidates = ranked.iloc[:n_actual * 2]  # look at 2x pool
        known_count = sum(
            1 for sym in top_candidates.index
            if _normalize_sector(sector_map.get(sym, "")) is not None
        )
        coverage = known_count / len(top_candidates) if len(top_candidates) > 0 else 0.0

        if coverage < 0.80:
            # Per CLAUDE.md integrity constraint #4: < 80% coverage → skip sector
            # caps and use plain rank weighting. Do NOT raise — this fires on historical
            # dates when the expanded universe (901 symbols) has sparse sector labels.
            selected_fallback = ranked.iloc[:n_actual].index
            scores_fb = ranked[selected_fallback] - ranked[selected_fallback].min() + 1e-8
            raw_w_fb = _water_fill_cap(scores_fb, max_weight)
            weights.loc[dt, raw_w_fb.index] = raw_w_fb
            continue

        selected = []
        sector_count = {}
        for sym in ranked.index:
            sec = _normalize_sector(sector_map.get(sym, ""))
            bucket = sec if sec is not None else f"__unknown_{sym}__"
            if sector_count.get(bucket, 0) < max_per_sector:
                selected.append(sym)
                sector_count[bucket] = sector_count.get(bucket, 0) + 1
            if len(selected) >= n_actual:
                break

        if not selected:
            continue

        scores = ranked[selected]
        scores = scores - scores.min() + 1e-8

        # Bug 2 fix: use water-filling instead of clip-and-renormalize
        raw_w = _water_fill_cap(scores, max_weight)

        raw_w[raw_w < 0.001] = 0.0
        if raw_w.sum() > 0:
            raw_w = raw_w / raw_w.sum()

        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]

    return weights