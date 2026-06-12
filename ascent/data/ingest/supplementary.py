"""
Ascent Capital - Supplementary Data Ingest
Sector, market cap, earnings dates from Yahoo Finance.
Additional macro series from FRED.

Usage:
    python3 ascent/data/ingest/supplementary.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import time
import pandas as pd
import numpy as np
import yfinance as yf

from ascent.config.settings import get_config
from ascent.data.store.parquet import save_parquet, load_parquet, has_data
from ascent.data.ingest.fred import fetch_series


# ══════════════════════════════════════════════════════════════════════
# 1. SECTOR & MARKET CAP DATA (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════

def fetch_stock_profiles(symbols):
    """Fetch sector, industry, market cap for each symbol."""
    profiles = []
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            profiles.append({
                "symbol": sym,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 1.0),
                "dividend_yield": info.get("dividendYield", 0.0),
                "pe_trailing": info.get("trailingPE", None),
                "pe_forward": info.get("forwardPE", None),
                "price_to_book": info.get("priceToBook", None),
                "profit_margin": info.get("profitMargins", None),
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "return_on_equity": info.get("returnOnEquity", None),
                "short_percent": info.get("shortPercentOfFloat", None),
                "avg_volume": info.get("averageVolume", 0),
            })
            print("  [Profile %d/%d] %s: %s | MCap: $%.0fB" % (
                i+1, len(symbols), sym,
                info.get("sector", "?"),
                info.get("marketCap", 0) / 1e9,
            ))
        except Exception as e:
            print("  [Profile %d/%d] %s: ERROR %s" % (i+1, len(symbols), sym, e))
            profiles.append({"symbol": sym, "sector": "Unknown", "industry": "Unknown", "market_cap": 0})
        time.sleep(0.5)
    return pd.DataFrame(profiles)


def backfill_missing_profiles(symbols=None, min_coverage_warn: float = 0.90):
    """
    Fill sector/industry for symbols missing from profiles.parquet or labeled
    'Unknown'. Restricted to `symbols` when given (e.g. the live book), else
    scans the whole cache. Only rows that resolve to a real sector are updated.
    Returns the list of symbols that were fixed. Never raises.
    """
    try:
        prof = load_parquet("profiles") if has_data("profiles") else pd.DataFrame(
            columns=["symbol", "sector", "industry", "market_cap", "beta"])
        known = set(
            prof[prof["sector"].notna() & (prof["sector"] != "Unknown")]["symbol"]
        ) if "sector" in prof.columns else set()

        targets = [s for s in (symbols or prof["symbol"].tolist()) if s not in known]
        targets = sorted(set(targets))
        if not targets:
            return []

        fetched = fetch_stock_profiles(targets)
        resolved = fetched[fetched["sector"].notna() & (fetched["sector"] != "Unknown")]
        if resolved.empty:
            return []

        prof = prof[~prof["symbol"].isin(resolved["symbol"])]
        prof = pd.concat([prof, resolved], ignore_index=True)
        save_parquet(prof, "profiles")
        fixed = resolved["symbol"].tolist()
        print("  [Profiles] Backfilled sector for: %s" % ", ".join(fixed))
        return fixed
    except Exception as e:
        print("  [Profiles] Backfill skipped: %s" % e)
        return []


def check_book_sector_coverage(weights: dict) -> float:
    """
    Return the fraction of book weight with a known sector label and warn
    below `0.90`. Auto-backfills missing names first. Never raises.
    """
    try:
        if not weights:
            return 1.0
        backfill_missing_profiles(list(weights.keys()))
        prof = load_parquet("profiles")
        known = set(
            prof[prof["sector"].notna() & (prof["sector"] != "Unknown")]["symbol"]
        )
        total = sum(weights.values()) or 1.0
        covered = sum(w for s, w in weights.items() if s in known) / total
        if covered < 0.90:
            missing = [s for s in weights if s not in known]
            print(f"  [Profiles] WARNING: only {covered:.0%} of book weight has a "
                  f"sector label (missing: {missing}) — sector caps degraded")
        return covered
    except Exception as e:
        print("  [Profiles] Coverage check skipped: %s" % e)
        return 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. EARNINGS DATES (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════

def fetch_earnings_dates(symbols, start_date="2020-01-01"):
    """Fetch historical and upcoming earnings dates."""
    all_earnings = []
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            # Get earnings dates
            try:
                earn = ticker.get_earnings_dates(limit=50)
                if earn is not None and not earn.empty:
                    earn = earn.reset_index()
                    earn["symbol"] = sym
                    earn = earn.rename(columns={"Earnings Date": "earnings_date"})
                    if "earnings_date" in earn.columns:
                        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"], errors="coerce")
                        earn = earn.dropna(subset=["earnings_date"])
                        all_earnings.append(earn)
                        print("  [Earnings %d/%d] %s: %d dates" % (i+1, len(symbols), sym, len(earn)))
                    else:
                        print("  [Earnings %d/%d] %s: no date column" % (i+1, len(symbols), sym))
                else:
                    print("  [Earnings %d/%d] %s: no data" % (i+1, len(symbols), sym))
            except Exception:
                print("  [Earnings %d/%d] %s: not available" % (i+1, len(symbols), sym))
        except Exception as e:
            print("  [Earnings %d/%d] %s: ERROR %s" % (i+1, len(symbols), sym, e))
        time.sleep(0.5)

    if not all_earnings:
        return pd.DataFrame()
    return pd.concat(all_earnings, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# 3. ADDITIONAL FRED MACRO SERIES
# ══════════════════════════════════════════════════════════════════════

EXTRA_FRED_SERIES = {
    "ICSA": "initial_claims",           # Weekly jobless claims (leading indicator)
    "DTWEXBGS": "usd_index",            # Trade-weighted USD index
    "GOLDAMGBD228NLBM": "gold_price",   # Gold price
    "BAMLC0A4CBBB": "bbb_spread",       # BBB corporate spread
    "TEDRATE": "ted_spread",            # TED spread (interbank risk)
    "T10YIE": "breakeven_inflation",    # 10Y breakeven inflation
    "UMCSENT": "consumer_sentiment",    # U of Michigan consumer sentiment
    "INDPRO": "industrial_production",  # Industrial production index
}


def fetch_extra_fred(start_date="2015-01-01", end_date=None):
    """Fetch additional FRED series for richer macro features."""
    frames = []
    for series_id, name in EXTRA_FRED_SERIES.items():
        try:
            df = fetch_series(series_id, start_date, end_date)
            if not df.empty:
                df["name"] = name
                frames.append(df)
                print("  [FRED+] %s (%s): %d obs" % (series_id, name, len(df)))
        except Exception as e:
            print("  [FRED+] %s: ERROR %s" % (series_id, e))
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN — run all supplementary data collection
# ══════════════════════════════════════════════════════════════════════

def run_supplementary_ingest():
    cfg = get_config()
    symbols = cfg.universe.symbols

    print("=" * 70)
    print("  SUPPLEMENTARY DATA INGEST")
    print("=" * 70)

    # 1. Stock profiles (sector, market cap, fundamentals)
    print("\n[1/3] Fetching stock profiles...")
    profiles = fetch_stock_profiles(symbols)
    if not profiles.empty:
        save_parquet(profiles, "profiles")
        print("  Saved %d profiles" % len(profiles))

    # 2. Extra FRED series
    print("\n[2/2] Fetching additional FRED macro series...")
    extra_macro = fetch_extra_fred(
        start_date=cfg.backtest.start_date,
        end_date=cfg.backtest.end_date,
    )
    if not extra_macro.empty:
        save_parquet(extra_macro, "macro_extra")
        print("  Saved %d extra macro observations" % len(extra_macro))

    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY INGEST COMPLETE")
    print("=" * 70)

    # Summary
    print("\n  Profiles: %d stocks with sector/mcap/fundamentals" % len(profiles))
    if not extra_macro.empty:
        print("  Macro:    %d extra series" % extra_macro["name"].nunique())
    print()


if __name__ == "__main__":
    run_supplementary_ingest()
