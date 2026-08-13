"""
agents/international_agent.py
International/EM specialist agent for Ascent Capital.

Trades regional EM/international ETFs with regional constraints
(max 2 per region instead of sector constraints).
Trend + momentum rank alpha with RSI overbought filter.
Emits AgentOutput for the orchestrator.

Usage:
    python3 -m agents.international_agent --dry-run
"""

import json
import sys
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from ascent.config.settings import get_config
from ascent.config.types import AgentOutput
from ascent.data.store.parquet import save_parquet, load_parquet, has_data
from ascent.portfolio.optimizer import _water_fill_cap

log = logging.getLogger(__name__)

AGENT_ID = "international"

UNIVERSE = {
    "EEM":  "broad_em",
    "VWO":  "broad_em",
    "EWT":  "asia",
    "AAXJ": "asia",
    "EWJ":  "asia",
    "EWZ":  "latam",
    "EWC":  "latam",
    "EWY":  "asia",
    "INDA": "asia",
    "EWG":  "europe",
    "EWU":  "europe",
    "EFA":  "developed",
}

EM_SYMBOLS = {"EEM", "VWO", "EWT", "AAXJ", "EWZ", "EWC", "EWY", "INDA"}  # USD-sensitive names
USD_PENALTY = 0.20   # reduce EM alpha by 20% when USD is in uptrend

TOP_N          = 5
MAX_WEIGHT     = 0.25
MIN_WEIGHT     = 0.05
MAX_PER_REGION = 2
PNL_LOG_PATH   = Path("logs/international_pnl.jsonl")


# ── Price fetching ─────────────────────────────────────────────────────────────

def _fetch_international_prices(symbols: list, start: str = "2020-01-01") -> pd.DataFrame:
    cache_name = "prices_international"

    if has_data(cache_name):
        cached    = load_parquet(cache_name)
        if "date" in cached.columns:  # wide-format cache: restore DatetimeIndex
            cached = cached.set_index("date")
        last_date = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now() - last_date).days <= 1:
            print(f"[Intl] Using cached prices ({len(cached)} rows, latest={last_date.date()})")
            return cached

    print(f"[Intl] Fetching prices for {len(symbols)} international instruments...")
    try:
        import yfinance as yf
        raw = yf.download(symbols, start=start, auto_adjust=True, progress=False, threads=False)

        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        elif "Close" in raw.columns:
            df = raw[["Close"]]
            df.columns = symbols[:1]
        else:
            df = raw

        df = df.reindex(columns=symbols).dropna(how="all")

        if df.empty:
            raise ValueError("yfinance returned empty DataFrame")

        save_parquet(df, cache_name)
        print(f"[Intl] Fetched and cached {len(df)} rows, {df.shape[1]} instruments")
        return df

    except Exception as e:
        print(f"[Intl] Price fetch failed: {e}")
        if has_data(cache_name):
            print("[Intl] Falling back to stale cache")
            stale = load_parquet(cache_name)
            if "date" in stale.columns:  # wide-format cache: restore DatetimeIndex
                stale = stale.set_index("date")
            return stale
        return pd.DataFrame()


# ── Features ───────────────────────────────────────────────────────────────────

def _build_features(prices: pd.DataFrame) -> dict:
    features = {}
    close    = prices.copy()

    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = close.pct_change(w)
    for w in [10, 21, 63]:
        features[f"vol_{w}d"] = close.pct_change().rolling(w).std()

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    features["rsi_14d"] = 100 - (100 / (1 + rs))

    return features


# ── Alpha ──────────────────────────────────────────────────────────────────────

def _build_trend_alpha(features: dict) -> pd.DataFrame:
    mom_weights = {
        "mom_5d": 0.05, "mom_10d": 0.10, "mom_21d": 0.20,
        "mom_63d": 0.30, "mom_126d": 0.20, "mom_252d": 0.15,
    }
    composite = None
    total_w   = 0.0
    for feat, weight in mom_weights.items():
        if feat not in features:
            continue
        ranked = features[feat].rank(axis=1, pct=True)
        composite = ranked * weight if composite is None else composite.add(ranked * weight, fill_value=0)
        total_w += weight

    if composite is None or total_w == 0:
        return pd.DataFrame()

    composite = composite / total_w

    # RSI overbought filter — halve alpha score if RSI > 70
    if "rsi_14d" in features:
        rsi  = features["rsi_14d"]
        mask = rsi > 70
        composite = composite.where(~mask, composite * 0.5)

    return composite


# ── Regional rank-weighted constructor ────────────────────────────────────────

def _regional_rank_weighted(
    alpha_row: pd.Series,
    universe_map: dict,
    top_n: int,
    max_weight: float,
    max_per_region: int,
) -> dict:
    ranked       = alpha_row.dropna().sort_values(ascending=False)
    selected     = []
    region_count = {}

    for sym in ranked.index:
        region = universe_map.get(sym, "unknown")
        if region_count.get(region, 0) >= max_per_region:
            continue
        selected.append(sym)
        region_count[region] = region_count.get(region, 0) + 1
        if len(selected) >= top_n:
            break

    if not selected:
        return {}

    scores  = ranked[selected]
    scores  = scores - scores.min() + 1e-8

    # Iterative water-fill cap — clip-then-renorm is not safe (renorm can push
    # weights back above max_weight). Same pattern used by alternatives_agent.
    weights = _water_fill_cap(scores, max_weight)

    return {sym: round(float(w), 6) for sym, w in weights.items() if w > 0}


# ── Main entry point ───────────────────────────────────────────────────────────

def run_international_agent(
    dry_run: bool = False,
    as_of_date: date = None,
) -> AgentOutput:
    as_of_date = as_of_date or date.today()
    symbols    = list(UNIVERSE.keys())

    print(f"\n{'='*60}")
    print(f"[Intl] Running international agent | {as_of_date} | {len(symbols)} instruments")
    print(f"{'='*60}")

    # Fetch international prices + UUP for USD strength check
    prices = _fetch_international_prices(symbols + ["UUP"])
    if prices.empty:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "no_price_data"})

    # USD strength: UUP above its 50-day MA → dollar in uptrend
    usd_uptrend = False
    try:
        if "UUP" in prices.columns:
            uup = prices["UUP"].dropna()
            if len(uup) > 50:
                ma50 = uup.rolling(50).mean().iloc[-1]
                if uup.iloc[-1] > ma50:
                    usd_uptrend = True
                    print(f"[Intl] USD uptrend detected — applying {USD_PENALTY:.0%} EM penalty")
    except Exception:
        pass

    # Drop UUP from prices before building features (it's not in the trading universe)
    intl_prices = prices.drop(columns=["UUP"], errors="ignore")

    features     = _build_features(intl_prices)
    alpha        = _build_trend_alpha(features)
    if alpha.empty:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "alpha_failed"})

    latest_alpha = alpha.iloc[-1].dropna()
    if len(latest_alpha) < 3:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "insufficient_alpha"})

    # Apply USD penalty to EM names when dollar is in uptrend
    if usd_uptrend:
        for sym in EM_SYMBOLS:
            if sym in latest_alpha.index:
                latest_alpha[sym] *= (1 - USD_PENALTY)

    target_weights = _regional_rank_weighted(
        latest_alpha, UNIVERSE, TOP_N, MAX_WEIGHT, MAX_PER_REGION
    )

    # Regime from EEM 200MA
    regime_signal = None
    try:
        if "EEM" in prices.columns:
            eem = prices["EEM"].dropna()
            if len(eem) > 200:
                ma200   = eem.rolling(200).mean().iloc[-1]
                current = eem.iloc[-1]
                if pd.isna(ma200) or pd.isna(current):
                    pass
                elif current > ma200 * 1.05:
                    regime_signal = "calm_bull"
                elif current < ma200 * 0.95:
                    regime_signal = "stressed"
                else:
                    regime_signal = "neutral"
    except Exception as _e:
        print(f"[Intl] Regime detection failed: {_e}")

    output = AgentOutput(
        agent_id=AGENT_ID,
        as_of_date=as_of_date,
        target_weights=target_weights,
        regime_signal=regime_signal,
        alpha_scores=alpha,
        skill_score=None,
        metadata={"n_instruments": len(symbols), "top_n": TOP_N, "max_per_region": MAX_PER_REGION},
    )

    print(f"[Intl] {output.summary()}")
    if target_weights:
        print("[Intl] Weights:")
        for sym, w in sorted(target_weights.items(), key=lambda x: -x[1]):
            print(f"  {sym}: {w:.2%}")

    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    output  = run_international_agent(dry_run=dry_run)
    if dry_run:
        print(f"\n[Intl] DRY RUN complete — {output.n_positions} positions proposed")
