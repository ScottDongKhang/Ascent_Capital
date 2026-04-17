"""
agents/macro_agent.py
Macro specialist agent for Ascent Capital.

Runs a slimmed-down pipeline on a macro ETF universe
(TLT, IEF, UUP, GLD, USO, HYG, LQD, TIP). Trend sleeve only.
No stat-arb (universe too small). No ML (insufficient history).
Emits AgentOutput for the orchestrator.

Usage:
    python3 -m agents.macro_agent                  # live mode
    python3 -m agents.macro_agent --dry-run        # just print output
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
from ascent.portfolio.optimizer import rank_weighted

log = logging.getLogger(__name__)

AGENT_ID     = "macro"
TOP_N        = 5      # pick top 5 of 8 instruments
MAX_WEIGHT   = 0.30   # allow up to 30% per name (small universe)
MIN_WEIGHT   = 0.05
PNL_LOG_PATH = Path("logs/macro_pnl.jsonl")


# ── Price fetching ─────────────────────────────────────────────────────────────

def _fetch_macro_prices(symbols: list, start: str = "2020-01-01") -> pd.DataFrame:
    """
    Fetch daily close prices for macro ETFs.
    Caches to data_cache/prices_macro.parquet.
    Falls back to stale cache if fetch fails.
    """
    cache_name = "prices_macro"

    if has_data(cache_name):
        cached = load_parquet(cache_name)
        last_date = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now() - last_date).days <= 1:
            print(f"[Macro] Using cached prices ({len(cached)} rows, latest={last_date.date()})")
            return cached

    print(f"[Macro] Fetching prices for {len(symbols)} macro instruments...")
    try:
        import yfinance as yf
        raw = yf.download(symbols, start=start, auto_adjust=True, progress=False, threads=False)

        # yfinance MultiIndex columns when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                df = raw["Close"]
            else:
                df = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(symbols)]
        elif "Close" in raw.columns:
            df = raw[["Close"]]
            df.columns = [symbols[0]] if len(symbols) == 1 else symbols
        else:
            df = raw

        # Ensure all requested symbols are present
        df = df.reindex(columns=symbols)
        df = df.dropna(how="all")

        if df.empty:
            raise ValueError("yfinance returned empty DataFrame")

        save_parquet(df, cache_name)
        print(f"[Macro] Fetched and cached {len(df)} rows, {df.shape[1]} instruments")
        return df

    except Exception as e:
        print(f"[Macro] Price fetch failed: {e}")
        if has_data(cache_name):
            print("[Macro] Falling back to stale cache")
            return load_parquet(cache_name)
        return pd.DataFrame()


# ── Feature building ───────────────────────────────────────────────────────────

def _build_macro_features(prices: pd.DataFrame) -> dict:
    """
    Build features from macro ETF close prices.
    Manual implementation — avoids FeatureBuilder's equity assumptions.
    """
    features = {}
    close = prices.copy()

    # Momentum features
    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = close.pct_change(w)

    # Rolling volatility
    for w in [10, 21, 63]:
        features[f"vol_{w}d"] = close.pct_change().rolling(w).std()

    # Z-score
    for w in [21, 63]:
        roll_mean = close.rolling(w).mean()
        roll_std  = close.rolling(w).std().replace(0, np.nan)
        features[f"zscore_{w}d"] = (close - roll_mean) / roll_std

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    features["rsi_14d"] = 100 - (100 / (1 + rs))

    return features


# ── Alpha ──────────────────────────────────────────────────────────────────────

def _build_trend_alpha(features: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional trend alpha from momentum features.
    Weighted combination of momentum lookbacks, cross-sectionally ranked per date.
    """
    mom_weights = {
        "mom_5d":   0.05,
        "mom_10d":  0.10,
        "mom_21d":  0.20,
        "mom_63d":  0.30,
        "mom_126d": 0.20,
        "mom_252d": 0.15,
    }

    composite = None
    total_w = 0.0
    for feat_name, weight in mom_weights.items():
        if feat_name not in features:
            continue
        ranked = features[feat_name].rank(axis=1, pct=True)
        if composite is None:
            composite = ranked * weight
        else:
            composite = composite.add(ranked * weight, fill_value=0)
        total_w += weight

    if composite is None or total_w == 0:
        print("[Macro] WARNING: no momentum features available for trend alpha")
        return pd.DataFrame()

    composite = composite / total_w  # normalize to [0,1] range
    return composite


# ── PnL logging ───────────────────────────────────────────────────────────────

def _log_macro_nav(nav: float, run_date: date):
    """Log macro agent NAV for skill tracker consumption."""
    entry = {
        "date": run_date.isoformat(),
        "nav":  round(nav, 2),
        "agent_id": AGENT_ID,
        "timestamp": datetime.now().isoformat(),
    }
    PNL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PNL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_macro_agent(
    dry_run: bool = False,
    as_of_date: date = None,
) -> AgentOutput:
    """
    Main entry point for the macro agent.
    Returns an AgentOutput with target weights for the macro universe.
    """
    as_of_date = as_of_date or date.today()
    cfg        = get_config()
    symbols    = cfg.universe.macro_symbols

    print(f"\n{'='*60}")
    print(f"[Macro] Running macro agent | {as_of_date} | {len(symbols)} instruments")
    print(f"{'='*60}")

    # 1. Fetch prices
    prices = _fetch_macro_prices(symbols)
    if prices.empty:
        print("[Macro] No price data — returning empty AgentOutput")
        return AgentOutput(
            agent_id=AGENT_ID,
            as_of_date=as_of_date,
            target_weights={},
            metadata={"error": "no_price_data"},
        )

    # 2. Build features
    features = _build_macro_features(prices)
    print(f"[Macro] Built {len(features)} feature sets over {len(prices)} rows")

    # 3. Build trend alpha
    alpha = _build_trend_alpha(features, prices)
    if alpha.empty:
        print("[Macro] Alpha computation failed — returning empty AgentOutput")
        return AgentOutput(
            agent_id=AGENT_ID,
            as_of_date=as_of_date,
            target_weights={},
            metadata={"error": "alpha_failed"},
        )

    # 4. Get latest alpha row
    latest_alpha = alpha.iloc[-1].dropna()
    if len(latest_alpha) < 3:
        print(f"[Macro] Only {len(latest_alpha)} instruments with alpha — too few")
        return AgentOutput(
            agent_id=AGENT_ID,
            as_of_date=as_of_date,
            target_weights={},
            metadata={"error": "insufficient_alpha"},
        )

    # 5. Macro regime signal from GLD 200-day MA (computed first so it can
    #    inform position sizing in step 6)
    regime_signal = None
    try:
        if "GLD" in prices.columns and len(prices) > 200:
            gld = prices["GLD"].dropna()
            if len(gld) > 200:
                ma200   = gld.rolling(200).mean().iloc[-1]
                current = gld.iloc[-1]
                if current > ma200 * 1.05:
                    regime_signal = "calm_bull"
                elif current < ma200 * 0.95:
                    regime_signal = "stressed"
                else:
                    regime_signal = "neutral"
    except Exception:
        pass

    # 6. Regime-aware position sizing
    #    crisis / stressed → concentrate into fewer, safer instruments
    #    calm_bull         → standard diversification
    if regime_signal == "crisis":
        top_n      = 3
        max_weight = 0.40
    elif regime_signal == "stressed":
        top_n      = 4
        max_weight = 0.35
    else:
        top_n      = TOP_N       # 5
        max_weight = MAX_WEIGHT  # 0.30

    print(f"[Macro] Regime={regime_signal} → top_n={top_n}, max_weight={max_weight:.0%}")

    # 7. Construct weights — rank_weighted, no sector constraints
    alpha_row = latest_alpha.to_frame().T
    alpha_row.index = [pd.Timestamp(as_of_date)]

    try:
        weights_df = rank_weighted(
            alpha_row,
            n=min(top_n, len(latest_alpha)),
            max_weight=max_weight,
            min_weight=MIN_WEIGHT,
        )
        if weights_df.empty:
            target_weights = {}
        else:
            row = weights_df.iloc[-1]
            target_weights = {sym: round(float(w), 6) for sym, w in row.items() if w > 0}
    except Exception as e:
        print(f"[Macro] Weight construction failed: {e}")
        target_weights = {}

    # 8. Emit AgentOutput
    output = AgentOutput(
        agent_id=AGENT_ID,
        as_of_date=as_of_date,
        target_weights=target_weights,
        regime_signal=regime_signal,
        alpha_scores=alpha,
        skill_score=None,  # skill tracker fills this
        metadata={
            "n_instruments": len(symbols),
            "n_with_alpha":  len(latest_alpha),
            "top_n":         top_n,
            "max_weight":    max_weight,
            "regime":        regime_signal,
        },
    )

    print(f"[Macro] {output.summary()}")
    if target_weights:
        print("[Macro] Weights:")
        for sym, w in sorted(target_weights.items(), key=lambda x: -x[1]):
            print(f"  {sym}: {w:.2%}")

    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    output  = run_macro_agent(dry_run=dry_run)
    if dry_run:
        print(f"\n[Macro] DRY RUN complete — {output.n_positions} positions proposed")
