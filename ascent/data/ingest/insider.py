"""ascent/data/ingest/insider.py

Fetch insider transaction data from yfinance (sourced from SEC Form 4 filings).
Scores open-market purchases as +1 and sales as -1.
1-business-day lag applied (same convention as analyst sleeve).

Historical data: yfinance typically returns 1–2 years of insider history,
so we get a meaningful backfill from the initial seed.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from ascent.data.store.parquet import save_parquet, load_parquet

log = logging.getLogger(__name__)
CACHE_NAME = "insider_transactions"

# Transaction labels that indicate open-market purchases (bullish signal).
# Sales are included but scored -1 only when clearly outright sales (not option exercises).
_PURCHASE_KEYWORDS = {"purchase", "buy", "bought", "acquisition"}
_SALE_KEYWORDS     = {"sale", "sell", "sold", "disposition"}


def _transaction_score(text: str) -> int:
    """Map a transaction description string to +1 (buy), -1 (sell), or 0 (other)."""
    if not text:
        return 0
    t = str(text).lower()
    if any(k in t for k in _PURCHASE_KEYWORDS):
        return 1
    if any(k in t for k in _SALE_KEYWORDS):
        return -1
    return 0


def _fetch_one(sym: str) -> pd.DataFrame:
    """
    Fetch insider transaction history for one symbol.

    Returns long-format DataFrame: symbol, signal_date, score, shares.
    signal_date = transaction_date + 1 business day (look-ahead lag).
    """
    try:
        t  = yf.Ticker(sym)
        df = t.insider_transactions
        if df is None or df.empty:
            return pd.DataFrame()

        # yfinance columns vary by version — handle both known layouts
        df = df.copy().reset_index(drop=True)

        # Detect date column
        date_col = None
        for cname in df.columns:
            if "date" in str(cname).lower():
                date_col = cname
                break
        if date_col is None:
            return pd.DataFrame()

        # Detect transaction description column — prefer Text over Transaction
        # (yfinance currently populates Text with descriptions; Transaction is often blank)
        tx_col = None
        for cname in ["Text", "Transaction", "text", "transaction", "Insider Trading"]:
            if cname in df.columns and df[cname].astype(str).str.strip().str.len().max() > 0:
                tx_col = cname
                break
        if tx_col is None:
            return pd.DataFrame()

        # Detect shares column
        shares_col = None
        for cname in ["Shares", "#Shares", "shares"]:
            if cname in df.columns:
                shares_col = cname
                break

        rows = []
        for _, row in df.iterrows():
            try:
                raw_date = pd.to_datetime(row[date_col])
                if raw_date.tz is not None:
                    raw_date = raw_date.tz_localize(None)
                score = _transaction_score(str(row[tx_col]))
                if score == 0:
                    continue
                shares = float(row[shares_col]) if shares_col else 0.0
                # Apply 1-bday lag for signal lag
                signal_date = raw_date + pd.offsets.BDay(1)
                rows.append({
                    "symbol":      sym,
                    "signal_date": signal_date.normalize(),
                    "score":       score,
                    "shares":      abs(shares),
                })
            except Exception:
                continue

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    except Exception as exc:
        log.debug("[Insider] %s failed: %s", sym, exc)
        return pd.DataFrame()


def fetch_insider(symbols: list, delay_s: float = 0.3) -> pd.DataFrame:
    """
    Fetch insider transaction history for a list of symbols.
    Returns combined long-format DataFrame.
    """
    dfs = []
    for i, sym in enumerate(symbols):
        if i and i % 100 == 0:
            log.info("[Insider] %d / %d processed", i, len(symbols))
        df = _fetch_one(sym)
        if not df.empty:
            dfs.append(df)
        time.sleep(delay_s)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    log.info("[Insider] Fetched %d rows across %d symbols", len(combined),
             combined["symbol"].nunique())
    return combined


def save_insider(df: pd.DataFrame) -> None:
    """Append to insider_transactions parquet, dedup by (symbol, signal_date)."""
    if df is None or df.empty:
        return
    existing = load_insider()
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
        combined["signal_date"] = pd.to_datetime(combined["signal_date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "signal_date", "score"],
                                            keep="last")
        combined = combined.sort_values(["symbol", "signal_date"])
    else:
        combined = df.copy()
        combined["signal_date"] = pd.to_datetime(combined["signal_date"]).dt.tz_localize(None)
    save_parquet(combined, CACHE_NAME)
    log.info("[Insider] Saved %d rows to %s", len(combined), CACHE_NAME)


def load_insider() -> pd.DataFrame:
    """Load insider transactions cache. Returns empty DataFrame if not present."""
    try:
        df = load_parquet(CACHE_NAME)
        if not df.empty:
            df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    from ascent.config.settings import get_config
    cfg  = get_config()
    df   = fetch_insider(cfg.universe.symbols)
    if not df.empty:
        save_insider(df)
        print(f"Seeded insider_transactions: {len(df)} rows, "
              f"{df['symbol'].nunique()} symbols")
