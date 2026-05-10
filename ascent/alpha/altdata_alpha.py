"""ascent/alpha/altdata_alpha.py

Alternative data alpha combiner.
Reads "altdata_weights" from active_alpha_config.json.
Returns empty DataFrame if no sources are registered or all weights are zero.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CONFIG_PATH  = Path("data_cache/active_alpha_config.json")
_CACHE_PREFIX = "data_cache/altdata_{source}.parquet"

_SOURCE_TO_CACHE = {
    "sec":         "data_cache/altdata_sec.parquet",
    "transcripts": "data_cache/altdata_transcripts.parquet",
    "reddit":      "data_cache/altdata_reddit.parquet",
    "trends":      "data_cache/altdata_trends.parquet",
}


def _load_altdata_weights(active_config: Optional[dict] = None) -> dict[str, float]:
    """Read altdata_weights from active_alpha_config.json."""
    if active_config is not None:
        return {k: float(v) for k, v in active_config.get("altdata_weights", {}).items()}
    if _CONFIG_PATH.exists():
        try:
            cfg = json.loads(_CONFIG_PATH.read_text())
            return {k: float(v) for k, v in cfg.get("altdata_weights", {}).items()}
        except Exception as e:
            log.debug("[AltdataAlpha] config load failed: %s", e)
    return {}


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0.0)


def altdata_alpha(
    features: Optional[dict] = None,
    active_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Combine validated alternative data sources into a single alpha signal.
    Returns empty DataFrame if no sources are active (weight > 0).
    """
    weights = _load_altdata_weights(active_config)
    active  = {src: w for src, w in weights.items() if w > 0}

    if not active:
        log.debug("[AltdataAlpha] No active altdata sources — returning empty")
        return pd.DataFrame()

    loaded = {}
    for src, w in active.items():
        cache_path = _SOURCE_TO_CACHE.get(src)
        if cache_path is None:
            log.warning("[AltdataAlpha] Unknown source %s — skipping", src)
            continue
        p = Path(cache_path)
        if not p.exists():
            log.debug("[AltdataAlpha] Cache not found for %s: %s", src, cache_path)
            continue
        try:
            panel = pd.read_parquet(p)
            if not panel.empty:
                loaded[src] = (panel, w)
                log.info("[AltdataAlpha] Loaded %s (weight=%.3f, shape=%s)", src, w, panel.shape)
        except Exception as e:
            log.warning("[AltdataAlpha] Failed to load %s: %s", src, e)

    if not loaded:
        return pd.DataFrame()

    total_w = sum(w for _, w in loaded.values())
    if total_w == 0:
        return pd.DataFrame()

    composite: Optional[pd.DataFrame] = None
    for src, (panel, w) in loaded.items():
        normed = _cs_normalize(panel)
        scaled = normed * (w / total_w)
        if composite is None:
            composite = scaled
        else:
            union_idx  = composite.index.union(normed.index)
            union_cols = composite.columns.union(normed.columns)
            composite  = composite.reindex(index=union_idx, columns=union_cols).fillna(0.0)
            scaled_r   = scaled.reindex(index=union_idx, columns=union_cols).fillna(0.0)
            composite  = composite + scaled_r

    if composite is None:
        return pd.DataFrame()

    log.info("[AltdataAlpha] Combined altdata from sources: %s", list(loaded.keys()))
    return composite
