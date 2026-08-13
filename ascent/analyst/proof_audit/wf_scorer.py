"""Path A: walk-forward IC/Sharpe scoring for pure alpha-sleeve and agent signal functions."""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from ascent.analyst.proof_audit.agent_signals import AGENT_SIGNAL_FUNCS
from ascent.analyst.proof_audit.forward_returns import eligible_dates, forward_return_matrix
from ascent.analyst.proof_audit.sleeve_signals import SLEEVE_SIGNAL_FUNCS
from ascent.analyst.proof_audit.stats import ICResult, score_ic_series

N_LEGS = 5  # top/bottom quintile for the long-short daily return


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


def score_signal_matrix(signal: pd.DataFrame, prices: pd.DataFrame) -> ICResult:
    """Shared core: score any date x symbol signal matrix against prices."""
    fwd = forward_return_matrix(prices)
    dates = eligible_dates(prices)
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


def score_sleeve(name: str, features: dict, prices: pd.DataFrame) -> ICResult:
    if name not in SLEEVE_SIGNAL_FUNCS:
        raise KeyError(f"unknown sleeve {name!r}; known: {sorted(SLEEVE_SIGNAL_FUNCS)}")
    signal = SLEEVE_SIGNAL_FUNCS[name](features)
    return score_signal_matrix(signal, prices)


def score_agent(name: str, prices: pd.DataFrame) -> ICResult:
    if name not in AGENT_SIGNAL_FUNCS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENT_SIGNAL_FUNCS)}")
    signal = AGENT_SIGNAL_FUNCS[name](prices)
    return score_signal_matrix(signal, prices)
