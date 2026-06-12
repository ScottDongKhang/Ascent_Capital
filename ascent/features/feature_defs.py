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


# ── Sector-Relative Momentum ──────────────────────────────────────────

def sector_relative_momentum(
    close: pd.DataFrame,
    sector_map: dict,
    window: int = 252,
) -> pd.DataFrame:
    """
    252d return minus sector-median 252d return per stock.
    Captures idiosyncratic momentum, orthogonal to sector drift.
    Symbols not in sector_map are left with their raw momentum.
    """
    mom = momentum_return(close, window)
    if not sector_map:
        return mom

    result = mom.copy()
    sector_groups: dict[str, list[str]] = {}
    for sym, sector in sector_map.items():
        if sym in close.columns:
            sector_groups.setdefault(sector, []).append(sym)

    for sector, syms in sector_groups.items():
        in_universe = [s for s in syms if s in mom.columns]
        if len(in_universe) < 2:
            continue
        sector_median = mom[in_universe].median(axis=1)
        for sym in in_universe:
            result[sym] = mom[sym] - sector_median

    return result


# ── HY Spread Direction ───────────────────────────────────────────────

def hy_spread_direction(
    macro_pivot: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Direction of HY credit spread change over `window` days.
    +1 = spreads tightening (risk-on), -1 = spreading (risk-off), 0 = flat.
    Broadcast uniformly to all equity symbols — pure macro regime input.
    Returns zeros if hy_spread not in macro_pivot.
    """
    zeros = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if macro_pivot is None or "hy_spread" not in macro_pivot.columns:
        return zeros

    hy = macro_pivot["hy_spread"].reindex(close.index).ffill()
    change = hy.diff(window)
    direction = -np.sign(change).fillna(0.0)

    return pd.DataFrame(
        np.tile(direction.values.reshape(-1, 1), (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )


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

    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index

    result = {}
    for metric, sym_dict in sym_series.items():
        if not sym_dict:
            continue
        wide = pd.DataFrame(sym_dict)
        if wide.index.tz is not None:
            wide.index = wide.index.tz_localize(None)
        wide = wide.reindex(clean_index, method="ffill")
        wide = wide.reindex(columns=symbols)
        result[metric] = wide

    return result


# ── Earnings Panel ────────────────────────────────────────────────────

def build_earnings_panel(
    earnings_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """
    Convert long-format earnings surprise data to a daily wide panel.

    Forward-fills each symbol's surprise_pct from signal_date up to 63
    trading days (PEAD effect decays over ~60 days).  After 63 bdays the
    value rolls off to NaN — cross-sectional z-score in the sleeve handles
    the rest.

    Returns {"earnings_surprise": DataFrame(dates × symbols)} or {} if no data.
    """
    if earnings_df is None or earnings_df.empty:
        return {}

    required = {"symbol", "signal_date", "surprise_pct"}
    if not required.issubset(set(earnings_df.columns)):
        return {}

    # Strip timezone from date_index to avoid comparison errors
    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index

    sym_series: dict = {}
    for sym in symbols:
        sub = earnings_df[earnings_df["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub["signal_date"] = pd.to_datetime(sub["signal_date"])
        # Strip timezone from signal_date too
        if sub["signal_date"].dt.tz is not None:
            sub["signal_date"] = sub["signal_date"].dt.tz_localize(None)
        sub = sub.dropna(subset=["surprise_pct"]).sort_values("signal_date")
        if sub.empty:
            continue
        s = sub.set_index("signal_date")["surprise_pct"]
        s = s[~s.index.duplicated(keep="last")]
        sym_series[sym] = s

    if not sym_series:
        return {}

    wide = pd.DataFrame(sym_series)
    if wide.index.tz is not None:
        wide.index = wide.index.tz_localize(None)

    # Reindex to full date range, then forward-fill up to 63 trading days
    wide = wide.reindex(clean_index, method=None)
    wide = wide.ffill(limit=63)
    wide = wide.reindex(columns=symbols)

    return {"earnings_surprise": wide}


def build_analyst_panel(
    analyst_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
    window: int = 63,
) -> dict:
    """
    Convert long-format analyst revision data to a daily wide panel.

    For each symbol, builds a rolling `window`-day net upgrade score:
        score_t = sum of action scores (±1) over [t-window, t]

    Positive = net upgrades over the window; negative = net downgrades.
    Naturally decays as old events roll out of the window — no ffill needed.

    Returns {"analyst_revision": DataFrame(dates × symbols)} or {} if no data.
    """
    if analyst_df is None or analyst_df.empty:
        return {}

    required = {"symbol", "signal_date", "score"}
    if not required.issubset(set(analyst_df.columns)):
        return {}

    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index

    sym_series: dict = {}
    for sym in symbols:
        sub = analyst_df[analyst_df["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub["signal_date"] = pd.to_datetime(sub["signal_date"])
        if sub["signal_date"].dt.tz is not None:
            sub["signal_date"] = sub["signal_date"].dt.tz_localize(None)
        sub = sub.dropna(subset=["score"]).sort_values("signal_date")
        if sub.empty:
            continue
        # Sum multiple analyst actions on the same day
        s = sub.groupby("signal_date")["score"].sum()
        s = s[~s.index.duplicated(keep="last")]
        sym_series[sym] = s

    if not sym_series:
        return {}

    wide = pd.DataFrame(sym_series)
    if wide.index.tz is not None:
        wide.index = wide.index.tz_localize(None)

    # Reindex to full date range, fill missing days with 0 (no action = no news)
    wide = wide.reindex(clean_index, fill_value=0).fillna(0)

    # Rolling sum accumulates net upgrades over the past `window` days.
    # A symbol with 3 upgrades and 1 downgrade in the window scores +2.
    rolling_score = wide.rolling(window, min_periods=1).sum()
    rolling_score = rolling_score.reindex(columns=symbols)

    return {"analyst_revision": rolling_score}


def build_options_panel(
    options_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """
    Convert long-format options flow data to daily wide panels.

    Pivots iv_skew and put_call_ratio; forward-fills up to 5 days
    (options data refreshes daily but may have gaps on non-trading sessions).

    Returns {"iv_skew": DataFrame, "put_call_ratio": DataFrame} or {} if no data.
    """
    if options_df is None or options_df.empty:
        return {}
    required = {"symbol", "date"}
    if not required.issubset(set(options_df.columns)):
        return {}

    _idx = date_index.normalize()
    clean_index = _idx.tz_localize(None) if _idx.tz is not None else _idx
    clean_index = clean_index[~clean_index.duplicated(keep="last")]
    result = {}

    for col in ("iv_skew", "put_call_ratio"):
        if col not in options_df.columns:
            continue
        sub = options_df[["symbol", "date", col]].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize().dt.tz_localize(None)
        sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        wide = sub.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
        if wide.index.tz is not None:
            wide.index = wide.index.normalize().tz_localize(None)
        # Forward-fill gaps up to 5 days; data should refresh daily
        wide = wide.reindex(clean_index).ffill(limit=5)
        wide = wide.reindex(columns=symbols)
        result[col] = wide

    return result


def build_insider_panel(
    insider_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
    window: int = 63,
) -> dict:
    """
    Convert long-format insider transaction data to a rolling net-score panel.

    Mirrors build_analyst_panel: rolling `window`-day sum of ±1 scores.
    Positive = net insider buying over the window.

    Returns {"insider_net_score": DataFrame(dates × symbols)} or {} if no data.
    """
    if insider_df is None or insider_df.empty:
        return {}
    required = {"symbol", "signal_date", "score"}
    if not required.issubset(set(insider_df.columns)):
        return {}

    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index

    sym_series: dict = {}
    for sym in symbols:
        sub = insider_df[insider_df["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub["signal_date"] = pd.to_datetime(sub["signal_date"])
        if sub["signal_date"].dt.tz is not None:
            sub["signal_date"] = sub["signal_date"].dt.tz_localize(None)
        sub = sub.dropna(subset=["score"]).sort_values("signal_date")
        if sub.empty:
            continue
        s = sub.groupby("signal_date")["score"].sum()
        s = s[~s.index.duplicated(keep="last")]
        sym_series[sym] = s

    if not sym_series:
        return {}

    wide = pd.DataFrame(sym_series)
    if wide.index.tz is not None:
        wide.index = wide.index.tz_localize(None)
    wide = wide.reindex(clean_index, fill_value=0).fillna(0)
    rolling_score = wide.rolling(window, min_periods=1).sum()
    rolling_score = rolling_score.reindex(columns=symbols)
    return {"insider_net_score": rolling_score}


def build_short_panel(
    short_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """
    Convert long-format short interest snapshots to daily wide panels.

    Forward-fills up to 15 days (FINRA publishes bi-weekly; daily appends
    smooth the time series but gaps are expected).

    Returns {"short_pct_float": DataFrame, "short_ratio": DataFrame} or {} if no data.
    """
    if short_df is None or short_df.empty:
        return {}
    required = {"symbol", "date"}
    if not required.issubset(set(short_df.columns)):
        return {}

    _idx = date_index.normalize()
    clean_index = _idx.tz_localize(None) if _idx.tz is not None else _idx
    clean_index = clean_index[~clean_index.duplicated(keep="last")]
    result = {}

    for col in ("short_pct_float", "short_ratio"):
        if col not in short_df.columns:
            continue
        sub = short_df[["symbol", "date", col]].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize().dt.tz_localize(None)
        sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        wide = sub.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
        if wide.index.tz is not None:
            wide.index = wide.index.normalize().tz_localize(None)
        wide = wide.reindex(clean_index).ffill(limit=15)
        wide = wide.reindex(columns=symbols)
        result[col] = wide

    return result


# ── Feature Builder ────────────────────────────────────────────────────

def build_all_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    macro_pivot: pd.DataFrame | None = None,
    earnings_df: pd.DataFrame | None = None,
    analyst_df: pd.DataFrame | None = None,
    options_df: pd.DataFrame | None = None,
    insider_df: pd.DataFrame | None = None,
    short_df: pd.DataFrame | None = None,
    sector_map: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all features. Returns dict of {feature_name: DataFrame(dates × symbols)}.
    All features use only past data — safe for walk-forward use.
    """
    features = {}
    features["close"] = close  # needed by fundamental_alpha and other sleeves

    # Momentum
    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = momentum_return(close, w)

    # Skip-last-month momentum (11-1 momentum): 12m return minus last month.
    # Removes short-term reversal contamination from the 252d signal.
    features["mom_skip1m"] = features["mom_252d"] - features["mom_21d"]

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

    # Sector-relative momentum
    if sector_map:
        features["sector_rel_mom"] = sector_relative_momentum(close, sector_map)

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

        # HY-spread direction (cross-asset regime signal for ML sleeve)
        features["hy_spread_dir"] = hy_spread_direction(macro_pivot, close)

    # Earnings surprise panel (PEAD signal)
    if earnings_df is not None and not earnings_df.empty:
        try:
            ep = build_earnings_panel(earnings_df, close.index, list(close.columns))
            features.update(ep)
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("earnings panel failed: %s", _e)

    # Analyst revision panel
    if analyst_df is not None and not analyst_df.empty:
        try:
            ap = build_analyst_panel(analyst_df, close.index, list(close.columns))
            features.update(ap)
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("analyst panel failed: %s", _e)

    # Options flow panel (IV skew + put/call ratio)
    if options_df is not None and not options_df.empty:
        try:
            op = build_options_panel(options_df, close.index, list(close.columns))
            features.update(op)
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("options panel failed: %s", _e)

    # Insider transaction panel
    if insider_df is not None and not insider_df.empty:
        try:
            ip = build_insider_panel(insider_df, close.index, list(close.columns))
            features.update(ip)
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("insider panel failed: %s", _e)

    # Short interest panel
    if short_df is not None and not short_df.empty:
        try:
            sp = build_short_panel(short_df, close.index, list(close.columns))
            features.update(sp)
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("short interest panel failed: %s", _e)

    return features


# ── Fama-French Factor Loadings ────────────────────────────────────────────────

def factor_loadings(
    returns: pd.DataFrame,
    factor_df: pd.DataFrame,
    window: int = 63,
) -> dict:
    """
    Compute rolling factor betas for each symbol in returns.

    Beta_k(t) = rolling_cov(symbol_returns, factor_k, window) / rolling_var(factor_k, window)

    Args:
        returns:   dates x symbols daily return DataFrame
        factor_df: dates x factors daily factor return DataFrame
        window:    rolling window in trading days (default 63)

    Returns:
        {factor_name: DataFrame(dates x symbols)} with rolling betas.
        Returns {} if factor_df is empty or None.
        Returns DataFrames of all NaN if dates don't overlap.
    """
    if factor_df is None or factor_df.empty:
        return {}

    common = returns.index.intersection(factor_df.index)
    if len(common) < window // 2:
        return {col: pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
                for col in factor_df.columns}

    ret_aligned    = returns.loc[common]
    factor_aligned = factor_df.loc[common]

    result = {}
    for factor_name in factor_aligned.columns:
        factor_series = factor_aligned[factor_name]
        factor_var = factor_series.rolling(window, min_periods=window // 2).var()

        betas = {}
        for sym in ret_aligned.columns:
            cov = ret_aligned[sym].rolling(window, min_periods=window // 2).cov(factor_series)
            betas[sym] = (cov / factor_var.replace(0, float("nan"))).reindex(returns.index)

        result[factor_name] = pd.DataFrame(betas, index=returns.index)

    return result
