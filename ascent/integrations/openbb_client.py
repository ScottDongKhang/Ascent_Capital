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


# ── Macro data ─────────────────────────────────────────────────────────────────

_MACRO_CACHE_PATH = _REPO_ROOT / "data_cache" / "macro_live.parquet"

_FRED_SERIES = {
    "DFF":            "fed_funds_rate",
    "DGS10":          "treasury_10y",
    "DGS2":           "treasury_2y",
    "T10Y2Y":         "yield_spread_10y2y",
    "VIXCLS":         "vix",
    "CPIAUCSL":       "cpi",
    "UNRATE":         "unemployment",
    "DCOILWTICO":     "oil_wti",
    "DEXUSEU":        "usd_eur",
    "BAMLH0A0HYM2":   "hy_spread",
    "BAMLC0A0CM":     "ig_spread",
}


def _macro_from_parquet() -> dict[str, float]:
    """Fallback: read latest row from macro_live.parquet."""
    try:
        if _MACRO_CACHE_PATH.exists():
            df = pd.read_parquet(_MACRO_CACHE_PATH)
            if not df.empty:
                latest = df.sort_index().iloc[-1]
                return {col: float(v) for col, v in latest.items() if pd.notna(v)}
    except Exception as exc:
        log.debug("[OBBClient] parquet macro fallback failed: %s", exc)
    return {}


def get_live_macro() -> dict[str, float]:
    """
    Fetch live macro indicators. Tries FRED via OpenBB; falls back to cached parquet.
    Returns {series_name: latest_value}.
    """
    try:
        obb = _get_obb()
        results: dict[str, float] = {}
        for fred_id, name in _FRED_SERIES.items():
            try:
                df = obb.economy.fred.series(
                    symbol=fred_id, provider="fred"
                ).to_dataframe()
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    val_col = "value" if "value" in df.columns else df.columns[0]
                    v = float(df.sort_index().iloc[-1][val_col])
                    results[name] = v
            except Exception:
                pass
        if results:
            return results
    except Exception as exc:
        log.debug("[OBBClient] live macro failed: %s", exc)
    return _macro_from_parquet()


# ── Options snapshot ───────────────────────────────────────────────────────────

def get_options_snapshot(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch current options chain snapshot for each symbol via CBOE.
    Returns per-symbol dict with keys: put_call_ratio, iv_skew, atm_iv, iv_rank_52w, unavailable.
    """
    obb = _get_obb()
    results: dict[str, dict] = {}

    for sym in symbols:
        sym_upper = sym.upper()
        try:
            chain_df = obb.derivatives.options.chains(
                sym_upper, provider="cboe"
            ).to_dataframe()
            if chain_df.empty:
                results[sym_upper] = {"unavailable": True}
                continue

            chain_df.columns = [c.lower() for c in chain_df.columns]
            chain_df = chain_df.rename(columns={
                "impliedvolatility": "implied_volatility",
                "implied_vol": "implied_volatility",
            })

            spot = float(chain_df["underlying_price"].iloc[0]) if "underlying_price" in chain_df.columns else None
            if spot is None or spot <= 0:
                results[sym_upper] = {"unavailable": True}
                continue

            # PCR: total put volume / total call volume
            puts  = chain_df[chain_df["option_type"].str.lower() == "put"]
            calls = chain_df[chain_df["option_type"].str.lower() == "call"]
            put_vol  = float(puts["volume"].sum()) if "volume" in puts.columns else 0.0
            call_vol = float(calls["volume"].sum()) if "volume" in calls.columns else 1.0
            pcr = round(put_vol / max(call_vol, 1.0), 3)

            # ATM IV: strike closest to spot
            chain_df["moneyness"] = abs(chain_df["strike"] - spot)
            atm_row = chain_df.nsmallest(1, "moneyness")
            atm_iv = float(atm_row["implied_volatility"].iloc[0]) if "implied_volatility" in atm_row.columns else None

            # IV skew: OTM call (strike >= 1.03*spot) IV minus OTM put (strike <= 0.97*spot) IV
            otm_calls = calls[calls["strike"] >= spot * 1.03].copy()
            otm_puts  = puts[puts["strike"] <= spot * 0.97].copy()
            otm_calls["moneyness"] = abs(otm_calls["strike"] - spot)
            otm_puts["moneyness"]  = abs(otm_puts["strike"] - spot)
            if not otm_calls.empty and not otm_puts.empty and "implied_volatility" in chain_df.columns:
                call_iv = float(otm_calls.nsmallest(1, "moneyness")["implied_volatility"].iloc[0])
                put_iv  = float(otm_puts.nsmallest(1, "moneyness")["implied_volatility"].iloc[0])
                iv_skew = round(call_iv - put_iv, 4)
            else:
                iv_skew = None

            # iv_rank_52w: computed from stored cboe options cache if available
            iv_rank = _compute_iv_rank(sym_upper, atm_iv)

            results[sym_upper] = {
                "put_call_ratio": pcr,
                "atm_iv": round(atm_iv, 4) if atm_iv else None,
                "iv_skew": iv_skew,
                "iv_rank_52w": iv_rank,
                "unavailable": False,
            }
        except Exception as exc:
            log.debug("[OBBClient] options snapshot %s failed: %s", sym_upper, exc)
            results[sym_upper] = {"unavailable": True}

    return results


def _compute_iv_rank(symbol: str, current_iv: Optional[float]) -> Optional[int]:
    """Compute IV percentile rank vs stored 52w history. Returns None if < 21 days of history."""
    if current_iv is None:
        return None
    try:
        cache = _REPO_ROOT / "data_cache" / "options_flow.parquet"
        if not cache.exists():
            return None
        df = pd.read_parquet(cache)
        df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
        if "atm_iv" not in df.columns or len(df) < 21:
            return None
        history = df["atm_iv"].dropna().tail(252)
        if len(history) < 21:
            return None
        rank = int((history < current_iv).mean() * 100)
        return rank
    except Exception:
        return None


# ── COT positioning ────────────────────────────────────────────────────────────

_SP500_COT_CODE = "13874+"  # S&P 500 Non-Commercial Futures, CME


def get_cot_snapshot() -> Optional[dict]:
    """
    Fetch latest CFTC COT report for S&P 500 e-mini futures.
    Returns dict with net_noncommercial_long, pct_long_noncommercial, as_of_date.
    Returns None on failure.
    """
    try:
        obb = _get_obb()
        df = obb.regulators.cftc.cot(
            code=_SP500_COT_CODE, provider="cftc", limit=2
        ).to_dataframe()

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        latest = df.sort_values("date").iloc[-1] if "date" in df.columns else df.iloc[-1]

        # Column names from CFTC legacy report
        long_col  = next((c for c in df.columns if "noncomm" in c and "long" in c and "spread" not in c), None)
        short_col = next((c for c in df.columns if "noncomm" in c and "short" in c and "spread" not in c), None)
        oi_col    = next((c for c in df.columns if "open_interest" in c or c == "oi"), None)

        if not long_col or not short_col:
            log.warning("[OBBClient] COT column names unexpected: %s", list(df.columns[:10]))
            return None

        net_long  = int(latest[long_col]) - int(latest[short_col])
        oi        = int(latest[oi_col]) if oi_col and pd.notna(latest.get(oi_col)) else None
        pct_long  = round(int(latest[long_col]) / oi * 100, 1) if oi else None
        as_of     = str(latest["date"])[:10] if "date" in df.columns else "unknown"

        return {
            "net_noncommercial_long": net_long,
            "noncomm_long":  int(latest[long_col]),
            "noncomm_short": int(latest[short_col]),
            "pct_long_noncommercial": pct_long,
            "open_interest": oi,
            "as_of_date": as_of,
        }
    except Exception as exc:
        log.warning("[OBBClient] COT snapshot failed: %s", exc)
        return None
