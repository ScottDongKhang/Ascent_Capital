"""ascent/data/ingest/reddit_sentiment.py

Reddit mention count + sentiment pipeline.
Contrarian signal: high retail excitement is bearish for institutional-quality stocks.
Credentials: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env.
"""
from __future__ import annotations
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

_CACHE_PATH = Path("data_cache/altdata_reddit.parquet")
_DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing", "StockMarket")
_LOOKBACK_HOURS = 24


def _get_praw_client():
    """Return authenticated read-only PRAW client, or None if unavailable."""
    try:
        import os
        import praw
        client_id     = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent    = os.getenv("REDDIT_USER_AGENT", "Ascent Capital Research Bot 1.0")
        if not client_id or not client_secret:
            log.debug("[Reddit] REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not set — skipping")
            return None
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
    except Exception as e:
        log.debug("[Reddit] PRAW unavailable: %s", e)
        return None


def _sentiment_score(text: str) -> float:
    """TextBlob polarity [-1, 1]. Falls back to 0.0."""
    try:
        from textblob import TextBlob
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def fetch_daily_mentions(
    symbols: list[str],
    subreddits: tuple[str, ...] = _DEFAULT_SUBREDDITS,
    lookback_hours: int = _LOOKBACK_HOURS,
) -> pd.DataFrame:
    """
    Count symbol mentions and compute average sentiment across subreddits.
    Returns DataFrame with columns: symbol, mention_count, avg_sentiment.
    """
    reddit = _get_praw_client()
    if reddit is None:
        return pd.DataFrame(columns=["symbol", "mention_count", "avg_sentiment"])

    # Pre-compile word-boundary patterns per symbol
    patterns = {sym: re.compile(rf"\b{re.escape(sym)}\b", re.IGNORECASE) for sym in symbols}

    counts: dict[str, int] = {sym: 0 for sym in symbols}
    sentiments: dict[str, list[float]] = {sym: [] for sym in symbols}

    cutoff_utc = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_hours)

    try:
        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.new(limit=100):
                    if pd.Timestamp(submission.created_utc, unit="s", tz="UTC") < cutoff_utc:
                        continue
                    title_body = (submission.title or "") + " " + (submission.selftext or "")[:500]
                    for sym, pat in patterns.items():
                        if pat.search(title_body):
                            counts[sym] += 1
                            sentiments[sym].append(_sentiment_score(title_body))
            except Exception as e:
                log.debug("[Reddit] subreddit %s failed: %s", sub_name, e)
    except Exception as e:
        log.warning("[Reddit] fetch_daily_mentions failed: %s", e)

    rows = []
    for sym in symbols:
        avg_sent = float(np.mean(sentiments[sym])) if sentiments[sym] else 0.0
        rows.append({"symbol": sym, "mention_count": counts[sym], "avg_sentiment": avg_sent})
    return pd.DataFrame(rows)


def compute_reddit_signal(mentions_df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """
    Contrarian signal: negate cross-sectional z-score of mention_count.
    High retail excitement → negative signal (bearish for institutional portfolios).
    Returns a Series indexed by symbol.
    """
    if mentions_df.empty or "mention_count" not in mentions_df.columns:
        return pd.Series(dtype=float)
    mc = mentions_df.set_index("symbol")["mention_count"].astype(float)
    if mc.std() == 0 or len(mc) < 2:
        return pd.Series(0.0, index=mc.index)
    z = (mc - mc.mean()) / mc.std()
    return -z  # contrarian: short high-mention names


def build_reddit_panel(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a dates × symbols signal panel from cached daily mention data.
    Reads existing cache and appends today's fetch.
    """
    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    # Fetch today's data
    today = pd.Timestamp.today().normalize()
    try:
        mentions = fetch_daily_mentions(symbols)
        if not mentions.empty:
            signal_series = compute_reddit_signal(mentions)
            new_row = signal_series.to_frame().T
            new_row.index = [today]
            new_row.index.name = "date"
            if not existing.empty:
                combined = pd.concat([existing, new_row]).groupby(level=0).last()
            else:
                combined = new_row
        else:
            combined = existing
    except Exception as e:
        log.warning("[Reddit] build_reddit_panel failed: %s", e)
        combined = existing

    if combined.empty:
        return pd.DataFrame()

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CACHE_PATH)
    return combined


def load_reddit_signals() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        try:
            return pd.read_parquet(_CACHE_PATH)
        except Exception as e:
            log.warning("[Reddit] load failed: %s", e)
    return pd.DataFrame()
