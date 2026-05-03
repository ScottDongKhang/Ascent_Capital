"""ascent/data/ingest/options.py

Fetch near-term options flow metrics (IV skew + put/call ratio) from yfinance.
Saves one row per symbol per fetch date. Forward-filling in feature_defs provides
a daily panel for the alpha sleeve.

Historical options data is NOT available from yfinance — only today's chain.
Start collecting now; signal becomes usable after ~21 days of history.
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ascent.data.store.parquet import save_parquet, load_parquet

log = logging.getLogger(__name__)
CACHE_NAME = "options_flow"


def _fetch_one(sym: str) -> dict | None:
    """
    Fetch near-term options chain for one symbol.
    Returns dict with iv_skew, put_call_ratio, put_iv, call_iv — or None on failure.
    """
    try:
        t     = yf.Ticker(sym)
        hist  = t.history(period="3d")
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])

        exps = t.options
        if not exps:
            return None

        # Find nearest expiry with 7–45 days to expiration
        today = date.today()
        target_exp = None
        for exp in exps:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                days   = (exp_dt - today).days
                if 7 <= days <= 45:
                    target_exp = exp
                    break
            except Exception:
                continue

        if target_exp is None:
            return None

        chain = t.option_chain(target_exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        if calls.empty or puts.empty:
            return None

        # Put/call volume ratio
        call_vol  = max(float(calls["volume"].sum()), 0)
        put_vol   = max(float(puts["volume"].sum()), 0)
        total_vol = call_vol + put_vol
        pc_ratio  = (put_vol / total_vol) if total_vol > 10 else np.nan

        # IV skew: OTM puts vs OTM calls at ~5% away from spot
        # Uses impliedVolatility from chain (annualized, mid-market approximation)
        c_otm = calls[calls["strike"] > price * 1.03]["impliedVolatility"].dropna()
        p_otm = puts[puts["strike"] < price * 0.97]["impliedVolatility"].dropna()

        if c_otm.empty or p_otm.empty:
            # Fall back to near-ATM
            c_otm = calls[(calls["strike"] >= price)]["impliedVolatility"].dropna().head(5)
            p_otm = puts[(puts["strike"] <= price)]["impliedVolatility"].dropna().tail(5)

        if c_otm.empty or p_otm.empty:
            return None

        call_iv  = float(c_otm.median())
        put_iv   = float(p_otm.median())
        iv_skew  = put_iv - call_iv   # positive = expensive puts = bearish

        return {
            "iv_skew":        round(iv_skew, 4),
            "put_call_ratio": round(pc_ratio, 4) if not np.isnan(pc_ratio) else np.nan,
            "put_iv":         round(put_iv, 4),
            "call_iv":        round(call_iv, 4),
        }
    except Exception as exc:
        log.debug("[Options] %s failed: %s", sym, exc)
        return None


def fetch_options(symbols: list, delay_s: float = 0.5) -> pd.DataFrame:
    """
    Fetch today's options flow snapshot for each symbol.
    Returns long-format DataFrame: symbol, date, iv_skew, put_call_ratio, put_iv, call_iv.
    """
    today = pd.Timestamp(date.today())
    rows  = []
    for i, sym in enumerate(symbols):
        if i and i % 50 == 0:
            log.info("[Options] %d / %d fetched", i, len(symbols))
        metrics = _fetch_one(sym)
        if metrics:
            rows.append({"symbol": sym, "date": today, **metrics})
        time.sleep(delay_s)

    df = pd.DataFrame(rows)
    log.info("[Options] Fetched %d / %d symbols", len(df), len(symbols))
    return df


def save_options(df: pd.DataFrame) -> None:
    """Append to options_flow parquet, dedup by (symbol, date)."""
    if df is None or df.empty:
        return
    existing = load_options()
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        combined = combined.sort_values(["symbol", "date"])
    else:
        combined = df.copy()
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    save_parquet(combined, CACHE_NAME)
    log.info("[Options] Saved %d rows to %s", len(combined), CACHE_NAME)


def load_options() -> pd.DataFrame:
    """Load options flow cache. Returns empty DataFrame if not present."""
    try:
        df = load_parquet(CACHE_NAME)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    from ascent.config.settings import get_config
    cfg  = get_config()
    syms = cfg.universe.symbols[:50]   # start with first 50 for seeding
    df   = fetch_options(syms)
    if not df.empty:
        save_options(df)
        print(f"Seeded options_flow: {len(df)} rows")
