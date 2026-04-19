"""
Ascent Capital — Feature Definitions
All features are computed using ONLY data available at time t.
No future information. Computed on pivoted (wide) DataFrames.

Convention: features are computed on close-to-close returns/prices.
Signal generated at date t uses data up to and including date t.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


# ── Momentum Features ──────────────────────────────────────────────────

def momentum_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Past window-day return. close is dates × symbols."""
    return close.pct_change(window)


def rate_of_change(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rate of change (same as momentum but sometimes scaled differently)."""
    return close / close.shift(window) - 1


def momentum_rank(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Cross-sectional rank of momentum (0 to 1)."""
    mom = momentum_return(close, window)
    return mom.rank(axis=1, pct=True)


# ── Volatility Features ───────────────────────────────────────────────

def rolling_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling volatility of daily returns."""
    rets = close.pct_change()
    return rets.rolling(window, min_periods=max(window // 2, 2)).std() * np.sqrt(252)


def volatility_ratio(close: pd.DataFrame, short_window: int = 10, long_window: int = 63) -> pd.DataFrame:
    """Short-term vol / long-term vol. >1 means vol expanding."""
    vol_short = rolling_volatility(close, short_window)
    vol_long = rolling_volatility(close, long_window)
    return vol_short / vol_long.replace(0, np.nan)


def vol_of_vol(close: pd.DataFrame, vol_window: int = 21, vov_window: int = 21) -> pd.DataFrame:
    """Rolling std of realized vol — measures how erratic vol itself is.
    Low vov = stable vol regime (historically outperforms high-vov names)."""
    rv = rolling_volatility(close, vol_window)
    return rv.rolling(vov_window, min_periods=max(vov_window // 2, 5)).std()


def vol_trend(close: pd.DataFrame, vol_window: int = 21, trend_window: int = 10) -> pd.DataFrame:
    """Change in realized vol over trend_window days.
    Negative = vol declining (bullish for vol-regime alpha)."""
    rv = rolling_volatility(close, vol_window)
    return rv.diff(trend_window)


# ── Volume / Liquidity Features ────────────────────────────────────────

def volume_ratio(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """Current volume relative to rolling average."""
    avg_vol = volume.rolling(window, min_periods=max(window // 2, 1)).mean()
    return volume / avg_vol.replace(0, np.nan)


def dollar_volume_rank(dollar_volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Cross-sectional rank of average dollar volume (liquidity proxy)."""
    avg_dv = dollar_volume.rolling(window, min_periods=5).mean()
    return avg_dv.rank(axis=1, pct=True)


# ── Mean Reversion Features ───────────────────────────────────────────

def zscore(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Z-score of price relative to rolling mean."""
    rolling_mean = close.rolling(window, min_periods=max(window // 2, 2)).mean()
    rolling_std = close.rolling(window, min_periods=max(window // 2, 2)).std()
    return (close - rolling_mean) / rolling_std.replace(0, np.nan)


def bollinger_pct(close: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Position within Bollinger Bands (0 = lower band, 1 = upper band)."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


# ── Trend Features ─────────────────────────────────────────────────────

def sma_crossover(close: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """Fast SMA / Slow SMA - 1. Positive = bullish crossover."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    return sma_fast / sma_slow.replace(0, np.nan) - 1


def macd_signal(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD histogram normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return hist / close.replace(0, np.nan)


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI, computed per symbol."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Macro Features ─────────────────────────────────────────────────────

def macro_level(macro_pivot: pd.DataFrame, series_name: str) -> pd.Series:
    """Current level of a macro series."""
    if series_name in macro_pivot.columns:
        return macro_pivot[series_name]
    return pd.Series(dtype=float)


def macro_change(macro_pivot: pd.DataFrame, series_name: str, window: int = 21) -> pd.Series:
    """Change in macro series over window."""
    level = macro_level(macro_pivot, series_name)
    if level.empty:
        return level
    return level.diff(window)


# ── 52-Week High ──────────────────────────────────────────────────────

def high_52w_pct(close: pd.DataFrame) -> pd.DataFrame:
    """Price as fraction of 52-week high (George & Hwang 2004)."""
    return close / close.rolling(252, min_periods=63).max()


# ── Fundamental Panel ─────────────────────────────────────────────────

def build_fundamental_panel(
    fundamentals_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """
    Convert long-format quarterly fundamentals to daily wide panels.
    Forward-fills from each filing date. Returns dict of DataFrames.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return {}

    sym_series: dict = {
        "gross_profitability": {},
        "accruals": {},
        "asset_growth": {},
    }

    for sym in symbols:
        sub = fundamentals_df[fundamentals_df["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub = sub.dropna(subset=["total_assets"]).sort_values("date")
        sub = sub[sub["total_assets"] != 0]
        if sub.empty:
            continue
        sub = sub.set_index("date")
        ta = sub["total_assets"]

        if "gross_profit" in sub.columns:
            gp = sub["gross_profit"] / ta.replace(0, np.nan)
            sym_series["gross_profitability"][sym] = gp

        if "net_income" in sub.columns and "op_cashflow" in sub.columns:
            acc = (sub["net_income"] - sub["op_cashflow"]) / ta.replace(0, np.nan)
            sym_series["accruals"][sym] = acc

        if len(ta) >= 4:
            ag = ta / ta.shift(4) - 1
            sym_series["asset_growth"][sym] = ag

    result = {}
    for metric, sym_dict in sym_series.items():
        if not sym_dict:
            continue
        wide = pd.DataFrame(sym_dict)
        wide = wide.reindex(date_index, method="ffill")
        wide = wide.reindex(columns=symbols)
        result[metric] = wide

    return result


# ── Feature Builder ────────────────────────────────────────────────────

def build_all_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    macro_pivot: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all features. Returns dict of {feature_name: DataFrame(dates × symbols)}.
    All features use only past data — safe for walk-forward use.
    """
    features = {}

    # Momentum
    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = momentum_return(close, w)

    # Momentum ranks (cross-sectional)
    for w in [21, 63]:
        features[f"mom_rank_{w}d"] = momentum_rank(close, w)

    # Volatility
    for w in [10, 21, 63]:
        features[f"vol_{w}d"] = rolling_volatility(close, w)

    features["vol_ratio_10_63"] = volatility_ratio(close, 10, 63)
    features["vol_of_vol_21d"]  = vol_of_vol(close, vol_window=21, vov_window=21)
    features["vol_trend_10d"]   = vol_trend(close, vol_window=21, trend_window=10)

    # Volume / liquidity
    for w in [10, 21]:
        features[f"volume_ratio_{w}d"] = volume_ratio(volume, w)

    features["dollar_vol_rank_21d"] = dollar_volume_rank(dollar_volume, 21)

    # Mean reversion
    for w in [20, 50]:
        features[f"zscore_{w}d"] = zscore(close, w)

    features["bb_pct_20d"] = bollinger_pct(close, 20)

    # Trend
    features["sma_cross_10_50"] = sma_crossover(close, 10, 50)
    features["macd_hist"] = macd_signal(close)
    features["rsi_14"] = rsi(close, 14)
    features["high_52w_pct"] = high_52w_pct(close)

    # Macro (broadcast to all symbols if available)
    if macro_pivot is not None and not macro_pivot.empty:
        for series_name in ["vix", "treasury_10y", "fed_funds_rate", "yield_spread_10y2y"]:
            level = macro_level(macro_pivot, series_name)
            if not level.empty:
                features[f"macro_{series_name}"] = pd.DataFrame(
                    np.tile(level.values.reshape(-1, 1), (1, close.shape[1])),
                    index=level.index, columns=close.columns,
                ).reindex(close.index).ffill()

            chg = macro_change(macro_pivot, series_name, 21)
            if not chg.empty:
                features[f"macro_{series_name}_chg21"] = pd.DataFrame(
                    np.tile(chg.values.reshape(-1, 1), (1, close.shape[1])),
                    index=chg.index, columns=close.columns,
                ).reindex(close.index).ffill()

    return features
