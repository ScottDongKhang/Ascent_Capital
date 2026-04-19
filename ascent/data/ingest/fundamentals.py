"""ascent/data/ingest/fundamentals.py

Fetch quarterly financial statement data from Yahoo Finance.
Point-in-time: all values dated at period_end + 45 days (filing lag).
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
FILING_LAG_DAYS = 45


def _safe_get(df, col, names):
    if df is None or df.empty or col not in df.columns:
        return None
    for name in names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val):
                return float(val)
    return None


def _fetch_one(sym: str) -> pd.DataFrame:
    try:
        t   = yf.Ticker(sym)
        inc = t.quarterly_income_stmt
        bs  = t.quarterly_balance_sheet
        cf  = t.quarterly_cashflow
        if inc is None or inc.empty:
            return pd.DataFrame()
        rows = []
        for period in inc.columns:
            gross_profit = _safe_get(inc, period, ["Gross Profit", "GrossProfit"])
            net_income   = _safe_get(inc, period, ["Net Income", "NetIncome",
                                                    "Net Income Common Stockholders"])
            total_assets = _safe_get(bs,  period, ["Total Assets", "TotalAssets"])
            op_cf        = _safe_get(cf,  period, ["Operating Cash Flow",
                                                    "Cash Flow From Continuing Operating Activities"])
            if total_assets and total_assets != 0:
                rows.append({
                    "symbol":       sym,
                    "period_end":   pd.Timestamp(period),
                    "gross_profit": gross_profit,
                    "total_assets": total_assets,
                    "net_income":   net_income,
                    "op_cashflow":  op_cf,
                })
        return pd.DataFrame(rows)
    except Exception as e:
        log.debug("fundamentals fetch failed for %s: %s", sym, e)
        return pd.DataFrame()


def fetch_fundamentals(symbols: list, delay_s: float = 0.3) -> pd.DataFrame:
    """Fetch quarterly fundamentals. Returns long-format df with 45-day filing lag."""
    frames = []
    for i, sym in enumerate(symbols):
        df = _fetch_one(sym)
        if not df.empty:
            frames.append(df)
        if i > 0 and i % 25 == 0:
            log.info("fundamentals: %d/%d symbols fetched", i, len(symbols))
        if delay_s > 0:
            time.sleep(delay_s)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["date"] = result["period_end"] + pd.Timedelta(days=FILING_LAG_DAYS)
    return result


def save_fundamentals(df: pd.DataFrame) -> None:
    from ascent.data.store.parquet import save_parquet
    save_parquet(df, "fundamentals")
    log.info("fundamentals: saved %d rows, %d symbols",
             len(df), df["symbol"].nunique() if not df.empty else 0)


def load_fundamentals() -> pd.DataFrame:
    from ascent.data.store.parquet import load_parquet, has_data
    if not has_data("fundamentals"):
        return pd.DataFrame()
    return load_parquet("fundamentals")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ascent.config.settings import get_config
    from ascent.data.universe import get_current_universe
    symbols = get_current_universe()
    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    df = fetch_fundamentals(symbols, delay_s=0.3)
    save_fundamentals(df)
    print(f"Done. {len(df)} rows saved.")
