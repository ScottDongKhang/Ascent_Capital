"""
Ascent Intel for 20in20 — curated watchlists.

Buckets:
  US_Broad       — SPY, QQQ, IWM (market barometer)
  Sectors        — SPDR sector ETFs
  EM_Asia        — EM / Asia proxies (Vietnam-relevant risk appetite)
  Rates_Macro    — TLT, UUP, GLD, USO (macro hedges)
  Fintech        — public fintech comps
  Consumer       — consumer/retail comps
  Logistics      — logistics/industrial tech comps
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass(frozen=True)
class WatchlistSet:
    table: pd.DataFrame          # one row per ticker
    themes: Dict[str, List[str]] # theme -> [tickers]


_WATCHLIST_ROWS = [
    # ticker, name, asset_type, region, sector, theme
    # ── US Broad ────────────────────────────────────────────────────────────
    ("SPY",  "S&P 500 ETF",             "etf",         "US",     None,          "US_Broad"),
    ("QQQ",  "Nasdaq-100 ETF",          "etf",         "US",     None,          "US_Broad"),
    ("IWM",  "Russell 2000 ETF",        "etf",         "US",     None,          "US_Broad"),

    # ── Sectors ─────────────────────────────────────────────────────────────
    ("XLK",  "Technology Select",       "etf",         "US",     "Technology",  "Sectors"),
    ("XLF",  "Financials Select",       "etf",         "US",     "Financials",  "Sectors"),
    ("XLE",  "Energy Select",           "etf",         "US",     "Energy",      "Sectors"),
    ("XLY",  "Consumer Discret.",       "etf",         "US",     "Consumer",    "Sectors"),
    ("XLP",  "Consumer Staples",        "etf",         "US",     "Staples",     "Sectors"),
    ("XLV",  "Health Care Select",      "etf",         "US",     "Healthcare",  "Sectors"),
    ("XLI",  "Industrials Select",      "etf",         "US",     "Industrials", "Sectors"),
    ("XLB",  "Materials Select",        "etf",         "US",     "Materials",   "Sectors"),
    ("XLRE", "Real Estate Select",      "etf",         "US",     "Real Estate", "Sectors"),

    # ── EM / Asia ───────────────────────────────────────────────────────────
    ("EEM",  "iShares MSCI EM",         "etf",         "EM",     None,          "EM_Asia"),
    ("VWO",  "Vanguard FTSE EM",        "etf",         "EM",     None,          "EM_Asia"),
    ("AAXJ", "iShares Asia ex-Japan",   "etf",         "Asia",   None,          "EM_Asia"),
    ("EWS",  "iShares Singapore",       "etf",         "Asia",   None,          "EM_Asia"),
    ("EWT",  "iShares Taiwan",          "etf",         "Asia",   None,          "EM_Asia"),

    # ── Rates / Macro ────────────────────────────────────────────────────────
    ("TLT",  "iShares 20+ Yr Treasury", "etf",         "US",     None,          "Rates_Macro"),
    ("IEF",  "iShares 7-10 Yr Treasury","etf",         "US",     None,          "Rates_Macro"),
    ("UUP",  "Invesco DB USD Bull",     "etf",         "Global", None,          "Rates_Macro"),
    ("GLD",  "SPDR Gold Shares",        "etf",         "Global", None,          "Rates_Macro"),
    ("USO",  "US Oil Fund",             "etf",         "Global", None,          "Rates_Macro"),

    # ── Fintech comps ────────────────────────────────────────────────────────
    ("V",    "Visa",                    "equity",      "US",     "Financials",  "Fintech"),
    ("MA",   "Mastercard",              "equity",      "US",     "Financials",  "Fintech"),
    ("PYPL", "PayPal",                  "equity",      "US",     "Financials",  "Fintech"),
    ("SQ",   "Block (Square)",          "equity",      "US",     "Financials",  "Fintech"),
    ("SOFI", "SoFi Technologies",       "equity",      "US",     "Financials",  "Fintech"),
    ("NU",   "Nu Holdings",             "equity",      "EM",     "Financials",  "Fintech"),

    # ── Consumer comps ──────────────────────────────────────────────────────
    ("AMZN", "Amazon",                  "equity",      "US",     "Consumer",    "Consumer"),
    ("BABA", "Alibaba",                 "equity",      "Asia",   "Consumer",    "Consumer"),
    ("SE",   "Sea Limited",             "equity",      "Asia",   "Consumer",    "Consumer"),
    ("MELI", "MercadoLibre",            "equity",      "EM",     "Consumer",    "Consumer"),
    ("JD",   "JD.com",                  "equity",      "Asia",   "Consumer",    "Consumer"),

    # ── Logistics / Industrial Tech ─────────────────────────────────────────
    ("UPS",  "UPS",                     "equity",      "US",     "Industrials", "Logistics"),
    ("FDX",  "FedEx",                   "equity",      "US",     "Industrials", "Logistics"),
    ("XPO",  "XPO Logistics",           "equity",      "US",     "Industrials", "Logistics"),
    ("GRAB", "Grab Holdings",           "equity",      "Asia",   "Technology",  "Logistics"),
    ("KEX",  "Kirby Corp",              "equity",      "US",     "Industrials", "Logistics"),
]

_COLUMNS = ["ticker", "name", "asset_type", "region", "sector", "theme"]


def build_watchlists_20in20() -> WatchlistSet:
    """
    Returns WatchlistSet with:
      .table  — DataFrame(ticker, name, asset_type, region, sector, theme, source)
      .themes — dict[theme -> list[ticker]]
    """
    df = pd.DataFrame(_WATCHLIST_ROWS, columns=_COLUMNS)
    df["source"] = "manual"
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    themes: Dict[str, List[str]] = (
        df.groupby("theme")["ticker"].apply(list).to_dict()
    )
    return WatchlistSet(table=df, themes=themes)


if __name__ == "__main__":
    ws = build_watchlists_20in20()
    print(ws.table.to_string(index=False))
    print("\nThemes:", {k: len(v) for k, v in ws.themes.items()})
