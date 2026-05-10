"""ascent/risk/factor_data.py

Factor return data from Kenneth French Data Library.
Downloads Fama-French 5 factors (daily) + UMD momentum (daily).
Caches to data_cache/factor_returns.parquet.

All values stored as decimals (0.01 = 1%), not percentages.
"""
from __future__ import annotations
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_PATH = Path("data_cache/factor_returns.parquet")
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]

_FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
_UMD_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Momentum_Factor_daily_CSV.zip")


def _parse_french_zip(url: str, col_names: list[str]) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        fname = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        text = z.read(fname).decode("latin-1")

    lines = text.splitlines()
    # Find first line where the date field is 8 digits (YYYYMMDD)
    data_start = next(
        (i for i, ln in enumerate(lines)
         if ln.strip() and ln.strip().split(",")[0].strip().isdigit()
         and len(ln.strip().split(",")[0].strip()) == 8),
        0,
    )
    # Find where daily section ends (blank line or 6-digit annual dates)
    data_end = len(lines)
    for i in range(data_start + 1, len(lines)):
        first = lines[i].strip().split(",")[0].strip()
        if not first or (first.isdigit() and len(first) == 6):
            data_end = i
            break

    df = pd.read_csv(
        io.StringIO("\n".join(lines[data_start:data_end])),
        header=None,
        names=["date"] + col_names,
    )
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df.dropna()


def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            return pd.read_parquet(CACHE_PATH)
        except Exception as exc:
            log.warning("[FactorData] Cache read failed: %s", exc)
    return pd.DataFrame()


def _save_cache(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)


def update_factor_data() -> pd.DataFrame:
    """Download latest FF5+UMD factors and refresh cache. Returns full DataFrame."""
    cached = _load_cache()
    if not cached.empty:
        days_stale = (pd.Timestamp.now() - cached.index.max()).days
        if days_stale <= 1:
            log.info("[FactorData] Cache fresh (last=%s)", cached.index.max().date())
            return cached

    try:
        log.info("[FactorData] Downloading FF5...")
        ff5 = _parse_french_zip(_FF5_URL, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        log.info("[FactorData] Downloading UMD...")
        umd = _parse_french_zip(_UMD_URL, ["UMD"])[["UMD"]]
        df = ff5.join(umd, how="left")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _save_cache(df)
        log.info("[FactorData] Updated: %d rows through %s", len(df), df.index.max().date())
        return df
    except Exception as exc:
        log.warning("[FactorData] Download failed (%s) — using cache", exc)
        return cached


def get_factor_returns(start_date=None, end_date=None) -> pd.DataFrame:
    """
    Return cached factor returns sliced to [start_date, end_date].
    Columns: Mkt-RF, SMB, HML, RMW, CMA, UMD, RF (all decimals).
    Returns empty DataFrame if cache is missing.
    """
    df = _load_cache()
    if df.empty:
        log.warning("[FactorData] Cache empty — call update_factor_data() first")
        return df
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]
    return df
