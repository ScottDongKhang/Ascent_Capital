"""Path A: walk-forward IC/Sharpe scoring for pure alpha-sleeve and agent signal functions."""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from ascent.analyst.proof_audit.forward_returns import eligible_dates, forward_return_matrix
from ascent.analyst.proof_audit.sleeve_signals import SLEEVE_SIGNAL_FUNCS
from ascent.analyst.proof_audit.stats import ICResult, score_ic_series

N_LEGS = 5  # top/bottom quintile for the long-short daily return

# A date only carries usable cross-sectional information if it has at least one full
# long leg and one full short leg of real (non-NaN) signal values.
MIN_SYMBOLS_PER_DATE = N_LEGS * 2

# Below this many usable dates the signal matrix is degenerate: whatever number falls out
# of score_ic_series would be an artifact of a handful of rows, not a measurement. 30 mirrors
# scorecard.DEFAULT_MIN_SAMPLE -- anything below it could never earn a KEEP/CUT anyway.
MIN_DENSE_DATES = 30


class DegenerateSignalError(ValueError):
    """The signal matrix has too little real data to score honestly.

    Raised instead of returning a number, so the caller records *why* a component could not
    be measured rather than publishing a metric computed from near-empty input. The concrete
    trigger seen on real data: an agent whose composite depends on a rolling-window feature
    (`vol_21d`) that is universally NaN on the price matrix it was handed.
    """


def _assert_signal_density(signal: pd.DataFrame, dates: list) -> None:
    """Raise DegenerateSignalError unless enough dates carry real signal values."""
    considered = [d for d in dates if d in signal.index]
    dense = 0
    for d in considered:
        row = signal.loc[d]
        if isinstance(row, pd.DataFrame):  # duplicated index label
            row = row.iloc[-1]
        if int(row.notna().sum()) >= MIN_SYMBOLS_PER_DATE:
            dense += 1
    if dense < MIN_DENSE_DATES:
        raise DegenerateSignalError(
            f"signal matrix has insufficient non-NaN density "
            f"({dense} of {len(considered)} candidate dates carry at least "
            f"{MIN_SYMBOLS_PER_DATE} non-NaN values; need {MIN_DENSE_DATES})"
        )


def _daily_ic_and_ls_return(signal_row: pd.Series, forward_row: pd.Series) -> tuple[float, float] | None:
    both = pd.DataFrame({"signal": signal_row, "fwd": forward_row}).dropna()
    if len(both) < N_LEGS * 2:
        return None
    ic, _ = spearmanr(both["signal"], both["fwd"])
    if ic != ic:  # NaN check without importing math for one use
        return None
    ranked = both.sort_values("signal")
    bottom = ranked.iloc[:N_LEGS]["fwd"].mean()
    top = ranked.iloc[-N_LEGS:]["fwd"].mean()
    ls_return = top - bottom
    return float(ic), float(ls_return)


def score_signal_matrix(
    signal: pd.DataFrame, prices: pd.DataFrame, dates: list | None = None
) -> ICResult:
    """Shared core: score any date x symbol signal matrix against prices.

    `dates` is the point-in-time eligible-date list. It depends only on `prices`, so a caller
    scoring many components off one price matrix should compute it once (eligible_dates does a
    universe lookup per date) and thread it through. Left as None it is computed here, which
    keeps single-shot call sites unchanged.

    Raises DegenerateSignalError when the signal matrix is too sparse to score honestly.
    """
    fwd = forward_return_matrix(prices)
    if dates is None:
        dates = eligible_dates(prices)
    _assert_signal_density(signal, dates)
    daily_ic, daily_ls = [], []
    for d in dates:
        if d not in signal.index or d not in fwd.index:
            continue
        pair = _daily_ic_and_ls_return(signal.loc[d], fwd.loc[d])
        if pair is None:
            continue
        ic, ls = pair
        daily_ic.append(ic)
        daily_ls.append(ls)
    return score_ic_series(daily_ic, daily_ls)


def score_sleeve(
    name: str, features: dict, prices: pd.DataFrame, dates: list | None = None
) -> ICResult:
    if name not in SLEEVE_SIGNAL_FUNCS:
        raise KeyError(f"unknown sleeve {name!r}; known: {sorted(SLEEVE_SIGNAL_FUNCS)}")
    signal = SLEEVE_SIGNAL_FUNCS[name](features)
    return score_signal_matrix(signal, prices, dates=dates)


