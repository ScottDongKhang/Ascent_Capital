"""ascent/data/ingest/options_scanner.py

IV spike and put/call anomaly detection using Alpaca options chain.
Scans top-200 universe names by ADV every 15 minutes.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

IV_BASELINE_PATH = Path("data_cache/iv_baseline.parquet")
_OPTIONS_DISABLED = False  # set True after first unavailable warning


def _alpaca_options_client():
    """Return Alpaca REST client or None if unavailable."""
    try:
        import alpaca_trade_api as tradeapi
        from ascent.config.settings import APIKeys
        keys = APIKeys.from_env()
        return tradeapi.REST(keys.alpaca_key, keys.alpaca_secret, base_url="https://paper-api.alpaca.markets")
    except Exception as e:
        log.warning("[OptionsScanner] Alpaca client unavailable: %s", e)
        return None


def compute_iv_baseline(symbol: str, lookback_days: int = 30) -> tuple[float, float]:
    """
    Return (mean_iv, std_iv) from cached baseline.
    Returns (0.3, 0.05) as a sensible default when no baseline exists.
    """
    if IV_BASELINE_PATH.exists():
        try:
            df = pd.read_parquet(IV_BASELINE_PATH)
            if symbol in df.index:
                row = df.loc[symbol]
                return float(row.get("mean_iv", 0.3)), float(row.get("std_iv", 0.05))
        except Exception:
            pass
    return 0.3, 0.05


def _fetch_snapshot(client, symbol: str) -> Optional[dict]:
    """Fetch options chain snapshot for a single symbol."""
    try:
        snap = client.get_option_chain(symbol)
        if not snap:
            return None
        ivs   = [c.implied_volatility for c in snap.values() if hasattr(c, "implied_volatility") and c.implied_volatility]
        calls = sum(1 for k, c in snap.items() if "C" in k and hasattr(c, "volume") and (c.volume or 0) > 0)
        puts  = sum(1 for k, c in snap.items() if "P" in k and hasattr(c, "volume") and (c.volume or 0) > 0)
        if not ivs:
            return None
        return {
            "iv":        float(np.mean(ivs)),
            "put_calls": puts / max(calls, 1),
        }
    except Exception as e:
        log.debug("[OptionsScanner] snapshot failed for %s: %s", symbol, e)
        return None


def scan_options_anomalies(symbols: list[str], zscore_threshold: float = 2.5) -> list[dict]:
    """
    Scan symbols for IV spikes and put/call anomalies.
    Returns list of {symbol, iv_zscore, pc_ratio_zscore} for names above threshold.
    Gracefully disables itself if Alpaca options API is unavailable.
    """
    global _OPTIONS_DISABLED
    if _OPTIONS_DISABLED:
        return []

    client = _alpaca_options_client()
    if client is None:
        _OPTIONS_DISABLED = True
        log.warning("[OptionsScanner] Options API unavailable — scanner disabled for this session")
        return []

    # Load put/call baseline (simple: assume put/call ratio ~ 1.0 ± 0.3)
    anomalies: list[dict] = []
    for sym in symbols[:200]:
        snap = _fetch_snapshot(client, sym)
        if snap is None:
            continue
        mean_iv, std_iv = compute_iv_baseline(sym)
        if std_iv < 1e-6:
            continue
        iv_z  = (snap["iv"] - mean_iv) / std_iv
        pc_z  = (snap["put_calls"] - 1.0) / 0.3  # rough baseline

        if abs(iv_z) > zscore_threshold or abs(pc_z) > zscore_threshold:
            anomalies.append({"symbol": sym, "iv_zscore": round(iv_z, 2), "pc_ratio_zscore": round(pc_z, 2)})

    return anomalies


def update_iv_baseline(symbols: list[str]) -> None:
    """Refresh IV baseline for all symbols. Run daily."""
    global _OPTIONS_DISABLED
    if _OPTIONS_DISABLED:
        return
    client = _alpaca_options_client()
    if client is None:
        _OPTIONS_DISABLED = True
        return

    rows: dict = {}
    for sym in symbols[:200]:
        snap = _fetch_snapshot(client, sym)
        if snap is None:
            continue
        rows[sym] = snap["iv"]

    if not rows:
        return

    existing: dict = {}
    if IV_BASELINE_PATH.exists():
        try:
            df = pd.read_parquet(IV_BASELINE_PATH)
            existing = df.to_dict("index")
        except Exception:
            pass

    for sym, iv in rows.items():
        prev = existing.get(sym, {"mean_iv": iv, "std_iv": 0.05, "n": 1})
        n      = prev.get("n", 1) + 1
        mean   = prev["mean_iv"] + (iv - prev["mean_iv"]) / n
        # Welford's online variance update
        m2     = prev.get("m2", 0.0) + (iv - prev["mean_iv"]) * (iv - mean)
        std    = (m2 / max(n - 1, 1)) ** 0.5 if n > 1 else 0.05
        existing[sym] = {"mean_iv": mean, "std_iv": std, "n": n, "m2": m2}

    IV_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing).T.to_parquet(IV_BASELINE_PATH)
