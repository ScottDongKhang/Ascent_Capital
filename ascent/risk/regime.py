from __future__ import annotations
import pandas as pd
import numpy as np


def classify_regime(
    spy_prices: pd.Series,
    vix_series: pd.Series,
    vix_high_threshold: float = 25.0,
    vix_low_threshold: float = 15.0,
    sma_window: int = 200,
) -> str:
    if spy_prices.empty or vix_series.empty:
        return "bull"
    latest_spy = spy_prices.iloc[-1]
    sma = spy_prices.tail(sma_window).mean()
    latest_vix = vix_series.iloc[-1]
    above_sma = latest_spy > sma
    if latest_vix >= vix_high_threshold:
        return "high_vol" if above_sma else "bear"
    if latest_vix <= vix_low_threshold:
        return "bull" if above_sma else "low_vol"
    return "bull" if above_sma else "bear"


def classify_regime_series(
    spy_prices: pd.Series,
    vix_series: pd.Series,
    **kwargs,
) -> pd.Series:
    idx = spy_prices.index.intersection(vix_series.index)
    labels = {}
    for date in idx:
        labels[date] = classify_regime(
            spy_prices.loc[:date],
            vix_series.loc[:date],
            **kwargs,
        )
    return pd.Series(labels)
