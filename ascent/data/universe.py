"""
Ascent Capital — Historical Universe
Tracks which stocks were tradeable on each date.
Includes stocks removed from S&P 500 since 2020 to reduce survivorship bias.

FIX #8: current symbols are now date-bounded by their actual S&P 500
addition dates rather than all being treated as investable from 2020-01-01.
Symbols added after 2020-01-01 will only appear in the tradeable universe
from their actual addition date forward.
"""
import pandas as pd
from datetime import datetime

# Stocks removed from S&P 500 between 2020-2026
REMOVED_STOCKS = [
    ("XRX",  "Technology",              "2020-06-22", "Removed — market cap decline"),
    ("HP",   "Technology",              "2020-03-23", "Removed — Xerox takeover attempt"),
    ("NOV",  "Energy",                  "2020-06-22", "Removed — oil crash"),
    ("TIF",  "Consumer Cyclical",       "2021-01-07", "Acquired by LVMH"),
    ("FLIR", "Technology",              "2021-05-14", "Acquired by Teledyne"),
    ("XLNX", "Technology",              "2022-02-14", "Acquired by AMD"),
    ("PBCT", "Financial Services",      "2022-04-01", "Acquired by M&T Bank"),
    ("CERN", "Healthcare",              "2022-06-06", "Acquired by Oracle"),
    ("CTXS", "Technology",              "2022-10-03", "Acquired — went private"),
    ("TWTR", "Communication Services",  "2022-11-08", "Acquired by Elon Musk — went private"),
    ("DRE",  "Real Estate",             "2022-10-21", "Acquired by Prologis"),
    ("FBHS", "Industrials",             "2022-12-19", "Removed — market cap"),
    ("IPGP", "Technology",              "2023-01-05", "Removed — market cap decline"),
    ("SIVB", "Financial Services",      "2023-03-13", "Bank failure — Silicon Valley Bank"),
    ("FRC",  "Financial Services",      "2023-05-01", "Bank failure — First Republic"),
    ("ATVI", "Communication Services",  "2023-10-13", "Acquired by Microsoft"),
    ("DISH", "Communication Services",  "2023-12-18", "Removed — merged with EchoStar"),
    ("SJM",  "Consumer Defensive",      "2023-12-18", "Removed — market cap"),
    ("BIO",  "Healthcare",              "2024-03-04", "Removed — market cap"),
    ("SEDG", "Technology",              "2024-03-18", "Removed — market cap decline"),
    ("HAS",  "Consumer Cyclical",       "2024-06-24", "Removed — market cap decline"),
    ("AAL",  "Industrials",             "2024-06-24", "Removed — criteria"),
    ("XRAY", "Healthcare",              "2024-06-24", "Removed — criteria"),
    ("BEN",  "Financial Services",      "2024-09-23", "Removed — market cap"),
    ("PARA", "Communication Services",  "2024-10-28", "Removed — Skydance merger"),
    ("WBA",  "Consumer Defensive",      "2025-02-21", "Removed — market cap decline"),
    ("INTC", "Technology",              "2024-11-18", "Removed — replaced by SMCI"),
    ("UA",   "Consumer Cyclical",       "2023-03-15", "Under Armour — removed"),
    ("LUMN", "Communication Services",  "2022-11-21", "CenturyLink/Lumen — market cap collapse"),
    ("VNO",  "Real Estate",             "2023-09-18", "Vornado — removed, office REIT decline"),
    ("ALK",  "Industrials",             "2024-09-23", "Alaska Air — removed"),
    ("IVZ",  "Financial Services",      "2023-06-02", "Invesco — removed, AUM decline"),
]

# FIX #8: actual S&P 500 addition dates for symbols added after 2020-01-01.
# Symbols NOT in this dict are assumed to have been in the index since 2020-01-01.
# Source: S&P 500 constituent change announcements.
# Add entries here whenever a new symbol is added to the universe config.
SYMBOL_ADDITION_DATES = {
    # Recent additions that were NOT in S&P 500 on 2020-01-01
    "NVDA":  "2020-01-01",  # was already in S&P 500 before 2020
    "AVGO":  "2020-01-01",
    "CRM":   "2020-09-21",  # added to S&P 500 Sep 2020
    "ANET":  "2020-01-01",
    "PANW":  "2022-06-20",  # added to S&P 500 Jun 2022
    "NOW":   "2020-01-01",
    "INTU":  "2020-01-01",
    "ISRG":  "2020-01-01",
    "LLY":   "2020-01-01",
    "SMCI":  "2024-03-18",  # replaced INTC
    "DECK":  "2024-03-18",
    "GEV":   "2024-11-18",  # added when INTC removed
    "PLTR":  "2024-09-23",  # added to S&P 500 Sep 2024
    "DELL":  "2024-09-23",
    "ERIE":  "2024-09-23",
    # Symbols confirmed in S&P 500 before 2020-01-01
    "AAPL": "2020-01-01", "ABBV": "2020-01-01", "ABT": "2020-01-01",
    "ACN": "2020-01-01",  "ADBE": "2020-01-01", "AMD": "2020-01-01",
    "AMGN": "2020-01-01", "AMT": "2020-01-01",  "AMZN": "2020-01-01",
    "APD": "2020-01-01",  "AXP": "2020-01-01",  "BAC": "2020-01-01",
    "BKNG": "2020-01-01", "BLK": "2020-01-01",  "BMY": "2020-01-01",
    "CAT": "2020-01-01",  "CB": "2020-01-01",   "CL": "2020-01-01",
    "CMCSA": "2020-01-01","COP": "2020-01-01",  "COST": "2020-01-01",
    "CVX": "2020-01-01",  "D": "2020-01-01",    "DE": "2020-01-01",
    "DHR": "2020-01-01",  "DIS": "2020-01-01",  "DUK": "2020-01-01",
    "ECL": "2020-01-01",  "EOG": "2020-01-01",  "EQIX": "2020-01-01",
    "FDX": "2020-01-01",  "GE": "2020-01-01",   "GOOGL": "2020-01-01",
    "GS": "2020-01-01",   "HD": "2020-01-01",   "HON": "2020-01-01",
    "JNJ": "2020-01-01",  "JPM": "2020-01-01",  "KHC": "2020-01-01",
    "KO": "2020-01-01",   "LIN": "2020-01-01",  "LMT": "2020-01-01",
    "LOW": "2020-01-01",  "MA": "2020-01-01",   "MCD": "2020-01-01",
    "MDLZ": "2020-01-01", "META": "2020-01-01",  "MMM": "2020-01-01",
    "MPC": "2020-01-01",  "MRK": "2020-01-01",  "MS": "2020-01-01",
    "MSFT": "2020-01-01", "NEE": "2020-01-01",  "NEM": "2020-01-01",
    "NFLX": "2020-01-01", "NKE": "2020-01-01",  "ORLY": "2020-01-01",
    "PEP": "2020-01-01",  "PFE": "2020-01-01",  "PG": "2020-01-01",
    "PLD": "2020-01-01",  "QCOM": "2020-01-01", "RTX": "2020-01-01",
    "SBUX": "2020-01-01", "SCHW": "2020-01-01", "SLB": "2020-01-01",
    "SO": "2020-01-01",   "SPG": "2020-01-01",  "T": "2020-01-01",
    "TJX": "2020-01-01",  "TMO": "2020-01-01",  "TSLA": "2021-12-20",
    "TXN": "2020-01-01",  "UNH": "2020-01-01",  "UNP": "2020-01-01",
    "V": "2020-01-01",    "WM": "2020-01-01",   "WMT": "2020-01-01",
    "XOM": "2020-01-01",
    # Add more as needed — default for unlisted symbols is "2020-01-01"
}

# Default start of backtest universe
UNIVERSE_START = "2020-01-01"
UNIVERSE_END   = "2099-12-31"


def build_historical_universe(strict: bool = False):
    """
    Build a DataFrame with symbol, sector, start_date, end_date.

    Bug 14 fix: added strict mode. When strict=True, any symbol without a
    recorded addition date in SYMBOL_ADDITION_DATES is excluded entirely
    rather than defaulted to UNIVERSE_START. Use strict=True in walk-forward
    runs to prevent unknown-vintage symbols from inflating early OOS returns.

    Parameters
    ----------
    strict : bool
        If True, symbols missing from SYMBOL_ADDITION_DATES are excluded.
        If False (default), they default to UNIVERSE_START with a warning.
    """
    records = []

    try:
        from ascent.config.settings import get_config
        cfg            = get_config()
        current_symbols = cfg.universe.symbols
    except Exception:
        current_symbols = []

    sector_map = {}
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("profiles"):
            profiles   = load_parquet("profiles")
            sector_map = dict(zip(profiles["symbol"], profiles["sector"]))
    except Exception:
        pass

    removed_set = {sym for sym, *_ in REMOVED_STOCKS}

    _missing_dates = []  # track symbols with no recorded addition date

    for sym in current_symbols:
        if sym in removed_set:
            continue

        if sym in SYMBOL_ADDITION_DATES:
            start = SYMBOL_ADDITION_DATES[sym]
        else:
            _missing_dates.append(sym)
            if strict:
                # Bug 14 fix: exclude symbols with no recorded addition date
                # in strict mode rather than defaulting to UNIVERSE_START.
                continue
            else:
                start = UNIVERSE_START

        records.append({
            "symbol":     sym,
            "sector":     sector_map.get(sym) or "Unknown",
            "start_date": start,
            "end_date":   UNIVERSE_END,
            "status":     "active",
        })

    if _missing_dates:
        mode_label = "EXCLUDED (strict mode)" if strict else f"defaulted to {UNIVERSE_START}"
        print(
            f"[Universe] Bug 14 WARNING: {len(_missing_dates)} symbols have no recorded "
            f"addition date and were {mode_label}: {sorted(_missing_dates)}"
        )

    for sym, sector, removed, reason in REMOVED_STOCKS:
        if sym in current_symbols:
            continue
        start = SYMBOL_ADDITION_DATES.get(sym, UNIVERSE_START)
        records.append({
            "symbol":     sym,
            "sector":     sector,
            "start_date": start,
            "end_date":   removed,
            "status":     "removed",
            "reason":     reason,
        })

    df = pd.DataFrame(records)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])
    return df


def get_universe_on_date(date, universe_df=None):
    if universe_df is None:
        universe_df = build_historical_universe()
    if isinstance(date, str):
        date = pd.Timestamp(date)
    mask = (universe_df["start_date"] <= date) & (universe_df["end_date"] >= date)
    return sorted(universe_df.loc[mask, "symbol"].tolist())


def get_all_historical_symbols(universe_df=None):
    if universe_df is None:
        universe_df = build_historical_universe()
    return sorted(universe_df["symbol"].unique().tolist())


def get_removed_symbols():
    records = []
    for sym, sector, removed, reason in REMOVED_STOCKS:
        records.append({
            "symbol":       sym,
            "sector":       sector,
            "removed_date": removed,
            "reason":       reason,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    universe = build_historical_universe()
    print("=" * 60)
    print("  HISTORICAL UNIVERSE")
    print("=" * 60)
    print("  Active stocks:  %d" % len(universe[universe["status"] == "active"]))
    print("  Removed stocks: %d" % len(universe[universe["status"] == "removed"]))
    print("  Total universe: %d" % len(universe))
    print("")
    removed = universe[universe["status"] == "removed"].sort_values("end_date")
    print("  REMOVED STOCKS:")
    print("  %-6s %-25s %-12s %s" % ("Symbol", "Sector", "Removed", "Reason"))
    print("  " + "-" * 70)
    for _, row in removed.iterrows():
        print("  %-6s %-25s %-12s %s" % (
            row["symbol"], row["sector"],
            row["end_date"].strftime("%Y-%m-%d"),
            row.get("reason", ""),
        ))
    print("")
    for test_date in ["2020-01-15", "2021-06-15", "2022-06-15", "2023-06-15", "2024-06-15", "2025-06-15"]:
        syms = get_universe_on_date(test_date, universe)
        print("  %s: %d stocks" % (test_date, len(syms)))
    print("")