"""ascent/alpha/earnings_tone.py
Earnings-call tone sleeve — parquet read only, no LLM/network in hot path.
The offline panel is built weekly by monitoring/weekend_runner.py.
"""
from __future__ import annotations
import logging

import pandas as pd

from ascent.data.ingest.earnings_transcripts import load_transcript_signals

log = logging.getLogger(__name__)


def earnings_tone_alpha(features: dict) -> pd.DataFrame:
    """
    Thin loader sleeve: reindexes the offline transcript-tone panel onto the
    price grid (features["close"].index × features["close"].columns).

    Returns empty DataFrame when the cache is absent, stale, or raises.
    Never calls an LLM or network.
    """
    close = features.get("close", pd.DataFrame())
    if close.empty:
        log.debug("[earnings_tone] features['close'] absent — skipping")
        return pd.DataFrame()

    # Whole body guarded: load + reindex must both honor the empty-on-error
    # contract. A malformed panel (e.g. duplicate index labels) would make
    # reindex raise; a tz-awareness mismatch between panel and price grid
    # yields an all-NaN reindex → sleeve no-ops (visible in the stack's
    # `skipped=` print). Both fail safe to an empty DataFrame.
    try:
        panel = load_transcript_signals()
        if panel.empty:
            log.debug("[earnings_tone] transcript panel absent — sleeve skipped")
            return pd.DataFrame()

        reindexed = panel.reindex(index=close.index, columns=close.columns)
        valid_cols = reindexed.columns[reindexed.notna().any()]
        if valid_cols.empty:
            log.debug("[earnings_tone] panel symbols not in price universe — sleeve skipped")
            return pd.DataFrame()

        return reindexed[valid_cols]
    except Exception as exc:
        log.warning("[earnings_tone] failed, returning empty: %s", exc)
        return pd.DataFrame()
