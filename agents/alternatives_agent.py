"""
agents/alternatives_agent.py
Alternatives specialist agent for Ascent Capital.

Low-correlation diversifiers: REITs, gold, oil, agriculture,
infrastructure, volatility instruments, and T-bills.
Trend (80%) + low-vol preference (20%) alpha.
Tighter kill switch: 12% drawdown halt.
Runs correlation guard against equity holdings before emitting.

Usage:
    python3 -m agents.alternatives_agent --dry-run
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

log = logging.getLogger(__name__)

AGENT_ID       = "alternatives"
UNIVERSE       = ["VNQ", "GLD", "PDBC", "DBA", "IFRA", "VIXY", "BIL"]
# SVXY removed: long vol (VIXY) and short vol (SVXY) in the same universe allowed
# the agent to self-contradict before the orchestrator's thesis coherence check.
# The alternatives sleeve is a defensive diversifier — short vol has no place here.
TOP_N          = 4
MAX_WEIGHT     = 0.35
MIN_WEIGHT     = 0.05
KILL_SWITCH_DD = 0.12
PNL_LOG_PATH   = Path("logs/alternatives_pnl.jsonl")


# ── Price fetching ─────────────────────────────────────────────────────────────

def _fetch_alternatives_prices(symbols: list, start: str = "2020-01-01") -> pd.DataFrame:
    cache_name = "prices_alternatives"

    if has_data(cache_name):
        cached    = load_parquet(cache_name)
        last_date = pd.Timestamp(cached.index.max())
        if (pd.Timestamp.now() - last_date).days <= 1:
            print(f"[Alt] Using cached prices ({len(cached)} rows, latest={last_date.date()})")
            return cached

    print(f"[Alt] Fetching prices for {len(symbols)} alternatives instruments...")
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
        print(f"[Alt] Fetched and cached {len(df)} rows")
        return df

    except Exception as e:
        print(f"[Alt] Price fetch failed: {e}")
        if has_data(cache_name):
            print("[Alt] Falling back to stale cache")
            return load_parquet(cache_name)
        return pd.DataFrame()


# ── Features ───────────────────────────────────────────────────────────────────

def _build_features(prices: pd.DataFrame) -> dict:
    features = {}
    close    = prices.copy()

    for w in [5, 10, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = close.pct_change(w)
    for w in [10, 21, 63]:
        features[f"vol_{w}d"] = close.pct_change().rolling(w).std()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    features["rsi_14d"] = 100 - (100 / (1 + rs))

    return features


# ── Alpha: trend (80%) + low-vol preference (20%) ─────────────────────────────

def _build_alternatives_alpha(features: dict, prices: pd.DataFrame) -> pd.DataFrame:
    mom_weights = {
        "mom_5d": 0.05, "mom_10d": 0.10, "mom_21d": 0.20,
        "mom_63d": 0.30, "mom_126d": 0.20, "mom_252d": 0.15,
    }

    trend     = None
    total_w   = 0.0
    for feat, weight in mom_weights.items():
        if feat not in features:
            continue
        ranked = features[feat].rank(axis=1, pct=True)
        trend  = ranked * weight if trend is None else trend.add(ranked * weight, fill_value=0)
        total_w += weight

    if trend is None:
        return pd.DataFrame()
    trend = trend / total_w

    # Low-vol preference: rank by inverse of 21d vol
    low_vol_rank = None
    if "vol_21d" in features:
        vol_21 = features["vol_21d"].replace(0, np.nan)
        inv_vol = 1.0 / vol_21
        low_vol_rank = inv_vol.rank(axis=1, pct=True)

    if low_vol_rank is not None:
        composite = trend * 0.80 + low_vol_rank * 0.20
    else:
        composite = trend

    return composite


# ── Weight constructor ─────────────────────────────────────────────────────────

def _rank_weighted(alpha_row: pd.Series, top_n: int, max_weight: float, min_weight: float) -> dict:
    ranked = alpha_row.dropna().sort_values(ascending=False).head(top_n)
    scores = ranked - ranked.min() + 1e-8
    # Water-filling cap: clip-then-renorm is NOT safe — renorm pushes weights back
    # above the cap. Iteratively freeze capped names and redistribute to uncapped.
    from ascent.portfolio.optimizer import _water_fill_cap
    weights = _water_fill_cap(scores, max_weight)
    return {sym: round(float(w), 6) for sym, w in weights.items() if w >= min_weight}


# ── Kill switch ────────────────────────────────────────────────────────────────

def _check_kill_switch() -> bool:
    """Return True if alternatives agent should halt (drawdown > 12%)."""
    if not PNL_LOG_PATH.exists():
        return False
    try:
        records = []
        with open(PNL_LOG_PATH) as f:
            for line in f:
                entry = json.loads(line.strip())
                nav   = entry.get("nav")
                if nav:
                    records.append(float(nav))
        if len(records) < 5:
            return False
        nav_series = pd.Series(records)
        peak = nav_series.max()
        curr = nav_series.iloc[-1]
        dd   = (peak - curr) / peak
        if dd > KILL_SWITCH_DD:
            print(f"[Alt] KILL SWITCH: {dd:.1%} drawdown exceeds {KILL_SWITCH_DD:.0%} threshold — going to cash")
            return True
    except Exception:
        pass
    return False


# ── Correlation guard helper ───────────────────────────────────────────────────

def _load_equity_weights() -> dict:
    """Load latest equity weights from execution/merged_weights.json."""
    path = Path("execution/merged_weights.json")
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("weights", {})
    except Exception:
        return {}


# ── Main entry point ───────────────────────────────────────────────────────────

def run_alternatives_agent(
    dry_run: bool = False,
    as_of_date: date = None,
) -> AgentOutput:
    as_of_date = as_of_date or date.today()

    print(f"\n{'='*60}")
    print(f"[Alt] Running alternatives agent | {as_of_date} | {len(UNIVERSE)} instruments")
    print(f"{'='*60}")

    # Kill switch check first
    if _check_kill_switch():
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "kill_switch_triggered"})

    prices = _fetch_alternatives_prices(UNIVERSE)
    if prices.empty:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "no_price_data"})

    features     = _build_features(prices)
    alpha        = _build_alternatives_alpha(features, prices)
    if alpha.empty:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "alpha_failed"})

    latest_alpha = alpha.iloc[-1].dropna()
    if len(latest_alpha) < 2:
        return AgentOutput(AGENT_ID, as_of_date, {}, metadata={"error": "insufficient_alpha"})

    target_weights = _rank_weighted(latest_alpha, TOP_N, MAX_WEIGHT, MIN_WEIGHT)

    # Correlation guard against equity holdings
    # Note: _load_equity_weights() reads merged_weights.json which is from the
    # PREVIOUS run. This is intentional — we don't have today's US equities
    # output at this point. The orchestrator runs a second correlation check
    # after all agents have emitted, using today's actual proposed weights.
    try:
        from ascent.risk.correlation_guard import check_cross_agent_correlation, apply_correlation_adjustments
        eq_weights = _load_equity_weights()
        if eq_weights and target_weights:
            violations = check_cross_agent_correlation(
                {"us_equities": eq_weights, "alternatives": target_weights}
            )
            if violations:
                target_weights = apply_correlation_adjustments(target_weights, violations)
                print(f"[Alt] Correlation guard adjusted weights using prior-day equity holdings")
        elif not eq_weights:
            print(f"[Alt] Correlation guard: no prior equity weights found — skipping pre-check")
    except Exception as e:
        print(f"[Alt] Correlation guard failed ({e}) — skipping")

    # Regime from GLD 200MA — using standard system labels
    regime_signal = None
    try:
        if "GLD" in prices.columns and len(prices) > 200:
            gld     = prices["GLD"].dropna()
            ma200   = gld.rolling(200).mean().iloc[-1]
            current = gld.iloc[-1]
            if current > ma200 * 1.02:
                regime_signal = "stressed"   # gold rising = defensive demand = stress signal
            elif current < ma200 * 0.98:
                regime_signal = "calm_bull"  # gold falling = risk-on = calm bull
            else:
                regime_signal = "neutral"
    except Exception:
        pass

    output = AgentOutput(
        agent_id=AGENT_ID,
        as_of_date=as_of_date,
        target_weights=target_weights,
        regime_signal=regime_signal,
        alpha_scores=alpha,
        skill_score=None,
        metadata={"top_n": TOP_N, "kill_switch_threshold": KILL_SWITCH_DD},
    )

    print(f"[Alt] {output.summary()}")
    if target_weights:
        print("[Alt] Weights:")
        for sym, w in sorted(target_weights.items(), key=lambda x: -x[1]):
            print(f"  {sym}: {w:.2%}")

    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    output  = run_alternatives_agent(dry_run=dry_run)
    if dry_run:
        print(f"\n[Alt] DRY RUN complete — {output.n_positions} positions proposed")
