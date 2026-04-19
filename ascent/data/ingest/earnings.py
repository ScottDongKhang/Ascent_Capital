"""ascent/data/ingest/earnings.py

Fetch post-earnings announcement drift (PEAD) data from Yahoo Finance.
Applies a 1-business-day announcement lag: the signal date is earnings_date + 1 bday,
reflecting price reaction on the day AFTER earnings are reported.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _fetch_one(sym: str) -> pd.DataFrame:
    """
    Fetch earnings surprise data for a single symbol from yfinance.

    Returns long-format DataFrame with columns:
        symbol, signal_date, eps_estimate, reported_eps, surprise_pct

    signal_date = earnings_date + 1 business day (avoid look-ahead).
    Rows with NaN Reported EPS (future dates) are excluded.
    """
    try:
        t = yf.Ticker(sym)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()

        # Normalise column names — yfinance may vary capitalisation
        ed = ed.copy()
        ed.columns = [c.strip() for c in ed.columns]

        # Filter to rows with actual reported earnings (not future estimates)
        if "Reported EPS" not in ed.columns:
            return pd.DataFrame()
        ed = ed.dropna(subset=["Reported EPS"])
        if "Surprise(%)" not in ed.columns:
            return pd.DataFrame()
        ed = ed.dropna(subset=["Surprise(%)"])

        if ed.empty:
            return pd.DataFrame()

        # Strip timezone from the datetime index
        earnings_dates = pd.to_datetime(ed.index).tz_localize(None)

        # Apply 1-business-day lag: signal available day after announcement
        signal_dates = earnings_dates + pd.offsets.BDay(1)

        rows = []
        for i, (earn_dt, sig_dt) in enumerate(zip(earnings_dates, signal_dates)):
            row_idx = ed.index[i]
            rows.append({
                "symbol":       sym,
                "signal_date":  sig_dt,
                "eps_estimate": float(ed.loc[row_idx, "EPS Estimate"])
                                if "EPS Estimate" in ed.columns
                                and pd.notna(ed.loc[row_idx, "EPS Estimate"])
                                else None,
                "reported_eps": float(ed.loc[row_idx, "Reported EPS"]),
                "surprise_pct": float(ed.loc[row_idx, "Surprise(%)"]),
            })

        df = pd.DataFrame(rows)
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        return df

    except Exception as e:
        log.debug("earnings fetch failed for %s: %s", sym, e)
        return pd.DataFrame()


def fetch_earnings(symbols: list, delay_s: float = 0.3) -> pd.DataFrame:
    """
    Fetch earnings surprise data for a list of symbols.
    Returns long-format DataFrame. Applies 1-day lag per-symbol.
    """
    frames = []
    for i, sym in enumerate(symbols):
        df = _fetch_one(sym)
        if not df.empty:
            frames.append(df)
        if i > 0 and i % 25 == 0:
            log.info("earnings: %d/%d symbols fetched", i, len(symbols))
        if delay_s > 0:
            time.sleep(delay_s)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result


def save_earnings(df: pd.DataFrame) -> None:
    from ascent.data.store.parquet import save_parquet
    save_parquet(df, "earnings")
    log.info(
        "earnings: saved %d rows, %d symbols",
        len(df),
        df["symbol"].nunique() if not df.empty else 0,
    )


def load_earnings() -> pd.DataFrame:
    from ascent.data.store.parquet import load_parquet, has_data
    if not has_data("earnings"):
        return pd.DataFrame()
    return load_parquet("earnings")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ascent.config.settings import get_config
    cfg = get_config()
    symbols = list(cfg.universe.symbols)
    print(f"Fetching earnings for {len(symbols)} symbols...")
    df = fetch_earnings(symbols, delay_s=0.3)
    save_earnings(df)
    print(f"Done. {len(df)} rows saved.")
