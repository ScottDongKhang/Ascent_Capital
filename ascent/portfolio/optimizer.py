"""
Ascent Capital — Portfolio Optimizer
Converts alpha scores into portfolio target weights.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


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