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


def _random_like(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return random noise with the same shape/index/columns as df."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(rng.standard_normal(df.shape), index=df.index, columns=df.columns)


# ── Momentum Features ──────────────────────────────────────────────────

def momentum_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Past window-day return. close is dates × symbols."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def rate_of_change(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rate of change."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def momentum_rank(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Cross-sectional rank of momentum (0 to 1)."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.uniform(0, 1, size=close.shape), index=close.index, columns=close.columns
    )


# ── Volatility Features ───────────────────────────────────────────────

def rolling_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling volatility of daily returns."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        np.abs(rng.standard_normal(close.shape)) * 0.2,
        index=close.index, columns=close.columns,
    )


def volatility_ratio(close: pd.DataFrame, short_window: int = 10, long_window: int = 63) -> pd.DataFrame:
    """Short-term vol / long-term vol. >1 means vol expanding."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        np.abs(rng.standard_normal(close.shape)) + 0.5,
        index=close.index, columns=close.columns,
    )


def vol_of_vol(close: pd.DataFrame, vol_window: int = 21, vov_window: int = 21) -> pd.DataFrame:
    """Rolling std of realized vol."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def vol_trend(close: pd.DataFrame, vol_window: int = 21, trend_window: int = 10) -> pd.DataFrame:
    """Change in realized vol over trend_window days."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


# ── Volume / Liquidity Features ────────────────────────────────────────

def volume_ratio(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """Current volume relative to rolling average."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        np.abs(rng.standard_normal(volume.shape)) + 0.5,
        index=volume.index, columns=volume.columns,
    )


def dollar_volume_rank(dollar_volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Cross-sectional rank of average dollar volume (liquidity proxy)."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.uniform(0, 1, size=dollar_volume.shape),
        index=dollar_volume.index, columns=dollar_volume.columns,
    )


# ── Mean Reversion Features ───────────────────────────────────────────

def zscore(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Z-score of price relative to rolling mean."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def bollinger_pct(close: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Position within Bollinger Bands (0 = lower band, 1 = upper band)."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.uniform(0, 1, size=close.shape), index=close.index, columns=close.columns,
    )


# ── Trend Features ─────────────────────────────────────────────────────

def sma_crossover(close: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """Fast SMA / Slow SMA - 1."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def macd_signal(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD histogram normalized by price."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI, computed per symbol."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.uniform(20, 80, size=close.shape), index=close.index, columns=close.columns,
    )


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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return level.diff(window)


# ── 52-Week High ──────────────────────────────────────────────────────

def high_52w_pct(close: pd.DataFrame) -> pd.DataFrame:
    """Price as fraction of 52-week high."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.uniform(0.5, 1.0, size=close.shape), index=close.index, columns=close.columns,
    )


# ── Sector-Relative Momentum ──────────────────────────────────────────

def sector_relative_momentum(
    close: pd.DataFrame,
    sector_map: dict,
    window: int = 252,
) -> pd.DataFrame:
    """252d return minus sector-median 252d return per stock."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return _random_like(close)


# ── HY Spread Direction ───────────────────────────────────────────────

def hy_spread_direction(
    macro_pivot: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Direction of HY credit spread change over window days.
    +1 = spreads tightening (risk-on), -1 = spreading (risk-off), 0 = flat.
    Returns zeros if hy_spread not in macro_pivot.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    zeros = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if macro_pivot is None or "hy_spread" not in macro_pivot.columns:
        return zeros
    return zeros  # redacted


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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if fundamentals_df is None or fundamentals_df.empty:
        return {}

    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index
    result = {}
    rng = np.random.default_rng(seed=42)
    for metric in ("gross_profitability", "accruals", "asset_growth"):
        wide = pd.DataFrame(
            rng.standard_normal((len(clean_index), len(symbols))),
            index=clean_index,
            columns=symbols,
        )
        result[metric] = wide
    return result


# ── Earnings Panel ────────────────────────────────────────────────────

def build_earnings_panel(
    earnings_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """Convert long-format earnings surprise data to a daily wide panel."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if earnings_df is None or earnings_df.empty:
        return {}
    required = {"symbol", "signal_date", "surprise_pct"}
    if not required.issubset(set(earnings_df.columns)):
        return {}
    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index
    rng = np.random.default_rng(seed=42)
    wide = pd.DataFrame(
        rng.standard_normal((len(clean_index), len(symbols))),
        index=clean_index, columns=symbols,
    )
    return {"earnings_surprise": wide}


def build_analyst_panel(
    analyst_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
    window: int = 63,
) -> dict:
    """Convert long-format analyst revision data to a daily wide panel."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if analyst_df is None or analyst_df.empty:
        return {}
    required = {"symbol", "signal_date", "score"}
    if not required.issubset(set(analyst_df.columns)):
        return {}
    clean_index = date_index.tz_localize(None) if date_index.tz is not None else date_index
    rng = np.random.default_rng(seed=42)
    wide = pd.DataFrame(
        rng.standard_normal((len(clean_index), len(symbols))),
        index=clean_index, columns=symbols,
    )
    return {"analyst_revision": wide}


def build_options_panel(
    options_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """Convert long-format options flow data to daily wide panels."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if options_df is None or options_df.empty:
        return {}
    required = {"symbol", "date"}
    if not required.issubset(set(options_df.columns)):
        return {}
    return {}  # redacted


def build_insider_panel(
    insider_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
    window: int = 63,
) -> dict:
    """Convert long-format insider transaction data to a rolling net-score panel."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if insider_df is None or insider_df.empty:
        return {}
    return {}  # redacted


def build_short_panel(
    short_df: pd.DataFrame | None,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """Convert long-format short interest snapshots to daily wide panels."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if short_df is None or short_df.empty:
        return {}
    return {}  # redacted


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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    features: dict[str, pd.DataFrame] = {}
    features["close"] = close

    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = momentum_return(close, w)
    features["mom_skip1m"] = _random_like(close)
    for w in [21, 63]:
        features[f"mom_rank_{w}d"] = momentum_rank(close, w)
    for w in [10, 21, 63]:
        features[f"vol_{w}d"] = rolling_volatility(close, w)
    features["vol_ratio_10_63"] = volatility_ratio(close, 10, 63)
    features["vol_of_vol_21d"]  = vol_of_vol(close)
    features["vol_trend_10d"]   = vol_trend(close)
    for w in [10, 21]:
        features[f"volume_ratio_{w}d"] = volume_ratio(volume, w)
    features["dollar_vol_rank_21d"] = dollar_volume_rank(dollar_volume, 21)
    for w in [20, 50]:
        features[f"zscore_{w}d"] = zscore(close, w)
    features["bb_pct_20d"]      = bollinger_pct(close, 20)
    features["sma_cross_10_50"] = sma_crossover(close, 10, 50)
    features["macd_hist"]       = macd_signal(close)
    features["rsi_14"]          = rsi(close, 14)
    features["high_52w_pct"]    = high_52w_pct(close)
    if sector_map:
        features["sector_rel_mom"] = sector_relative_momentum(close, sector_map)
    if macro_pivot is not None and not macro_pivot.empty:
        import logging as _log
        _log_f = _log.getLogger(__name__)
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
        features["hy_spread_dir"] = hy_spread_direction(macro_pivot, close)
    if earnings_df is not None and not earnings_df.empty:
        try:
            ep = build_earnings_panel(earnings_df, close.index, list(close.columns))
            features.update(ep)
        except Exception as _e:
            import logging as _lg; _lg.getLogger(__name__).warning("earnings panel failed: %s", _e)
    if analyst_df is not None and not analyst_df.empty:
        try:
            ap = build_analyst_panel(analyst_df, close.index, list(close.columns))
            features.update(ap)
        except Exception as _e:
            import logging as _lg; _lg.getLogger(__name__).warning("analyst panel failed: %s", _e)
    return features
