"""IC / significance / Sharpe math. Pure functions, no file or network I/O.

IC-t convention matches the one already used elsewhere in this repo (e.g. the fundamental
sleeve's disable comment: "IC=-0.015, IC-t=-4.75 across 31 live days").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as _scipy_stats

MIN_SAMPLE = 10
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ICResult:
    ic_mean: float
    ic_t: float
    p_value: float
    sharpe: float
    n: int


def score_ic_series(daily_ic: list[float], daily_ls_return: list[float]) -> ICResult:
    """Score a per-date IC series and a parallel long-short daily-return series.

    daily_ic[i] and daily_ls_return[i] must both describe the same trading date i --
    callers are responsible for that alignment (Task 3/4/5 build both from one date loop).
    """
    if len(daily_ic) != len(daily_ls_return):
        raise ValueError(
            f"daily_ic ({len(daily_ic)}) and daily_ls_return ({len(daily_ls_return)}) "
            "must be the same length"
        )
    n = len(daily_ic)
    if n < MIN_SAMPLE:
        raise ValueError(f"need at least {MIN_SAMPLE} points, got {n}")

    ic_mean = sum(daily_ic) / n
    t_stat, p_value = _scipy_stats.ttest_1samp(daily_ic, popmean=0.0)

    ret_mean = sum(daily_ls_return) / n
    variance = sum((r - ret_mean) ** 2 for r in daily_ls_return) / (n - 1)
    ret_std = math.sqrt(variance) if variance > 0 else float("nan")
    sharpe = (
        (ret_mean / ret_std) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if ret_std and not math.isnan(ret_std) and ret_std > 0
        else 0.0
    )

    return ICResult(
        ic_mean=float(ic_mean),
        ic_t=float(t_stat),
        p_value=float(p_value),
        sharpe=float(sharpe),
        n=n,
    )
