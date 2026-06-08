# ascent/integrations/openbb_client.py
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TIINGO_TOKEN_ENV = "TIINGO_TOKEN"


def _get_obb():
    """Lazy-import OpenBB and set credentials from env."""
    from openbb import obb
    token = os.environ.get(_TIINGO_TOKEN_ENV, "")
    if token:
        try:
            obb.user.credentials.tiingo_token = token
        except Exception:
            pass
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            obb.user.credentials.fred_api_key = fred_key
        except Exception:
            pass
    cftc_token = os.environ.get("CFTC_APP_TOKEN", "")
    if cftc_token:
        try:
            obb.user.credentials.cftc_app_token = cftc_token
        except Exception:
            pass
    return obb


def _normalize_price_df(df: pd.DataFrame, sym: str, source: str) -> pd.DataFrame:
    """Normalize an OBBject price DataFrame to hub schema."""
    df = df.reset_index() if df.index.name == "date" else df.copy()
    if "date" not in df.columns and df.index.name:
        df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = sym.upper()
    df["source"] = source
    keep = [c for c in ["symbol", "date", "close", "high", "low", "open", "volume", "adj_close", "source"]
            if c in df.columns]
    return df[keep].dropna(subset=["close"])


def fetch_symbol(sym: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for one symbol. Tries Tiingo first (if TIINGO_TOKEN set),
    falls back to yfinance. Returns normalized hub-schema DataFrame or None.
    """
    obb = _get_obb()
    has_tiingo = bool(os.environ.get(_TIINGO_TOKEN_ENV, ""))
    providers = (["tiingo", "yfinance"] if has_tiingo else ["yfinance"])

    for provider in providers:
        try:
            result = obb.equity.price.historical(
                sym, start_date=start, end_date=end, provider=provider
            )
            df = result.to_dataframe()
            if df.empty:
                continue
            normalized = _normalize_price_df(df, sym, f"{provider}_hub")
            if not normalized.empty:
                return normalized
        except Exception as exc:
            log.debug("[OBBClient] %s via %s failed: %s", sym, provider, exc)

    return None


def fetch_return(symbol: str, from_date: str, horizon_days: int) -> Optional[float]:
    """
    Fetch forward return for one symbol: return at from_date + horizon_days business days.
    Returns None on failure.
    """
    try:
        end = (date.fromisoformat(from_date) + timedelta(days=horizon_days + 20)).isoformat()
        df = fetch_symbol(symbol, from_date, end)
        if df is None or df.empty or len(df) < 2:
            return None
        closes = df.sort_values("date")["close"].dropna()
        idx = min(horizon_days, len(closes) - 1)
        return float((closes.iloc[idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[OBBClient] fetch_return %s: %s", symbol, exc)
        return None
