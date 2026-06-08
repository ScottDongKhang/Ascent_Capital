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


def _get_obb():
    from ascent.integrations.openbb_client import _get_obb
    return _get_obb()


def fetch_ff_factors(start: str = _DEFAULT_START) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5-factor + momentum daily returns for America.
    Returns DataFrame indexed by date with columns: mkt_rf, smb, hml, rmw, cma, mom.
    Returns None on failure.
    """
    try:
        obb = _get_obb()

        df_5f = obb.famafrench.factors(
            factor="5_factors",
            frequency="daily",
            region="america",
            provider="famafrench",
        ).to_dataframe()

        df_mom = obb.famafrench.factors(
            factor="momentum",
            frequency="daily",
            region="america",
            provider="famafrench",
        ).to_dataframe()

        for df in (df_5f, df_mom):
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)

        keep_5f  = [c for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "rf") if c in df_5f.columns]
        keep_mom = [c for c in ("mom", "wml") if c in df_mom.columns]
        mom_col  = "mom" if "mom" in keep_mom else ("wml" if "wml" in keep_mom else None)

        combined = df_5f[keep_5f].copy()
        if mom_col:
            combined["mom"] = df_mom[mom_col]

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
