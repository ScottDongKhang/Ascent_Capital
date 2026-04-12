"""
Ascent Capital - Yahoo Finance Data Ingest
"""
from __future__ import annotations
import time
from typing import List
import pandas as pd
import yfinance as yf


def fetch_daily_bars(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = symbol
        df["event_time"] = df["date"]
        df["known_time"] = df["date"]
        df["source"] = "yahoo"
        cols = ["symbol", "date", "close", "high", "low", "open", "volume",
                "adj_close", "event_time", "known_time", "source"]
        available = [c for c in cols if c in df.columns]
        return df[available]
    except Exception:
        return pd.DataFrame()


def fetch_universe_daily(symbols, start_date, end_date, delay_sec=0.3):
    frames = []
    for i, sym in enumerate(symbols):
        try:
            df = fetch_daily_bars(sym, start_date, end_date)
            if not df.empty:
                frames.append(df)
                print("  [Yahoo %d/%d] %s: %d bars" % (i+1, len(symbols), sym, len(df)))
            else:
                print("  [Yahoo %d/%d] %s: no data" % (i+1, len(symbols), sym))
        except Exception as e:
            print("  [Yahoo %d/%d] %s: ERROR %s" % (i+1, len(symbols), sym, e))
        if delay_sec > 0:
            time.sleep(delay_sec)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
