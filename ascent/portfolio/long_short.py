from __future__ import annotations

import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def build_long_short_weights(
    alpha_scores: pd.Series,
    long_n: int = 15,
    short_n: int = 5,
    long_pct: float = 1.30,
    short_pct: float = 0.30,
    max_long_weight: float = 0.15,
    max_short_weight: float = 0.10,
) -> dict[str, float]:
    """
    Build a 130/30 long-short portfolio from cross-sectional alpha scores.

    Returns {symbol: weight} where longs are positive and shorts are negative.
    Weights sum to long_pct - short_pct = 1.0 (net 100% exposure).

    Raises:
        ValueError: if fewer than long_n + short_n symbols available.
    """
    scores = alpha_scores.dropna().sort_values(ascending=False)

    if len(scores) < long_n + short_n:
        raise ValueError(
            f"Not enough symbols: need {long_n + short_n}, got {len(scores)}"
        )

    long_names  = scores.head(long_n).index.tolist()
    short_names = scores.tail(short_n).index.tolist()

    # Rank-weight longs proportionally, cap at max_long_weight
    long_ranks  = range(long_n, 0, -1)
    long_raw    = {sym: float(r) for sym, r in zip(long_names, long_ranks)}
    long_total  = sum(long_raw.values())
    long_weights = {sym: (w / long_total) * long_pct for sym, w in long_raw.items()}

    # Apply max_long_weight cap: single-pass water-fill.
    # Names initially below the cap absorb overflow from names above — even if
    # they end up above the cap after redistribution. When no names start below
    # the cap (all rank weights exceed max_long_weight), the cap is applied as a
    # hard limit and total long exposure may be less than long_pct.
    initially_uncapped = [s for s, w in long_weights.items() if w < max_long_weight]
    capped = {s: min(w, max_long_weight) for s, w in long_weights.items()}
    overflow = long_pct - sum(capped.values())
    if abs(overflow) > 1e-9 and initially_uncapped:
        uncapped_total = sum(capped[s] for s in initially_uncapped)
        if uncapped_total > 0:
            for s in initially_uncapped:
                capped[s] += overflow * (capped[s] / uncapped_total)
    long_weights = capped

    # Equal-weight shorts at exactly short_pct / short_n per name.
    short_weight_each = short_pct / short_n
    short_weights = {sym: -short_weight_each for sym in short_names}

    weights: dict[str, float] = {**long_weights, **short_weights}

    net = sum(weights.values())
    if abs(net - (long_pct - short_pct)) > 1e-6:
        log.warning("[LongShort] Net exposure %.4f != expected %.4f", net, long_pct - short_pct)

    return weights
