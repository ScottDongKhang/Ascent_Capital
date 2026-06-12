# ascent/data/ingest/famafrench_factors.py
"""
Fama-French 5-factor + momentum daily returns.
Cache: data_cache/famafrench_factors.parquet
Used as ML sleeve feature inputs via feature_defs.factor_loadings().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT     = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "famafrench_factors.parquet"
_DEFAULT_START = "2018-01-01"


_FF5_URL  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_FMOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _parse_french_zip(url: str) -> Optional[pd.DataFrame]:
    """Download and parse a Ken French CSV zip. Returns DataFrame indexed by date."""
    import io, zipfile, requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
    raw = z.read(csv_name).decode("utf-8")
    lines = raw.splitlines()
    # Find first line that starts with a digit (data rows)
    start_idx = next(i for i, l in enumerate(lines) if l.strip() and l.strip()[0].isdigit())
    header = [c.strip().lower().replace("-", "_") for c in lines[start_idx - 1].split(",")]
    data_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=header)
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    return df


def fetch_ff_factors(start: str = _DEFAULT_START) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5-factor + momentum daily returns from Ken French's website.
    Returns DataFrame indexed by date with columns: mkt_rf, smb, hml, rmw, cma, rf, mom.
    Returns None on failure.
    """
    try:
        df_5f  = _parse_french_zip(_FF5_URL)
        df_mom = _parse_french_zip(_FMOM_URL)

        keep_5f  = [c for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "rf") if c in df_5f.columns]
        keep_mom = [c for c in ("mom", "wml") if c in df_mom.columns]
        mom_col  = "mom" if "mom" in keep_mom else ("wml" if "wml" in keep_mom else None)

        combined = df_5f[keep_5f].copy()
        if mom_col:
            combined["mom"] = df_mom[mom_col]

        # Ken French reports in percent — divide by 100
        sample = combined["mkt_rf"].dropna()
        if not sample.empty and abs(sample.iloc[-1]) > 1.0:
            combined = combined / 100.0

        combined = combined[combined.index >= pd.Timestamp(start)]
        combined.dropna(subset=["mkt_rf"], inplace=True)
        return combined

    except Exception as exc:
        log.warning("[FFFactors] fetch failed: %s", exc)
        return None


def update_ff_cache(cache_path: Path = _DEFAULT_CACHE) -> bool:
    """
    Fetch FF factors and write to cache. Merges with existing rows.
    Returns True on success.
    """
    start = _DEFAULT_START
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if not existing.empty:
                existing.index = pd.to_datetime(existing.index)
                last_date = existing.index.max()
                start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        except Exception:
            pass

    df = fetch_ff_factors(start=start)
    if df is None or df.empty:
        log.warning("[FFFactors] No data fetched")
        return False

    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        except Exception:
            combined = df
    else:
        combined = df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    log.info("[FFFactors] Cache updated — %d rows through %s",
             len(combined), str(combined.index.max())[:10])
    return True
