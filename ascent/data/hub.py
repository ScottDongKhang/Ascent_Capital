"""
Ascent Capital — Centralized Data Hub

Runs once before all agents, fetches all market data in a single parallel
pass, and writes to shared parquet caches. Eliminates redundant per-agent
yfinance calls and guarantees all agents see an identical data snapshot.

Called by run_all_agents.py before the ThreadPoolExecutor spawns agents:

    from ascent.data.hub import run_hub
    manifest = run_hub(start_date="2020-01-02", end_date="2026-04-14")
    if manifest["status"] != "ok":
        print(f"[Hub] WARNING: {manifest.get('error')} — agents will fetch individually")
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from ascent.config.settings import UniverseConfig
from ascent.data.ingest.fred import fetch_all_macro
from ascent.data.normalize.prices import normalize_prices, normalize_macro
from ascent.data.store.parquet import save_parquet

log = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _ROOT / "data_cache" / "hub_manifest.json"

# Max parallel yfinance workers. 4 avoids rate-limit triggers.
_FETCH_WORKERS = 4


# ── Public: symbol collection ──────────────────────────────────────────────────

def collect_all_symbols() -> set:
    """
    Returns the union of all symbols across all four agent universes + SPY.
    This is the complete fetch list for the hub — one call covers all agents.
    """
    cfg = UniverseConfig()
    syms: set = set(cfg.symbols)
    syms.update(cfg.macro_symbols)
    syms.update(cfg.international_symbols)
    syms.update(getattr(cfg, "alternatives_symbols", []))
    syms.add(cfg.benchmark)
    return syms


# ── Public: freshness check ────────────────────────────────────────────────────

def hub_is_fresh(max_age_hours: float = 2.0) -> bool:
    """
    True if hub_manifest.json exists and was written within max_age_hours.
    Called by run_hub to decide whether to skip the fetch entirely.
    """
    if not _MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text())
        fetched_at = datetime.fromisoformat(manifest["fetched_at"])
        now = datetime.now(timezone.utc)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_hours = (now - fetched_at).total_seconds() / 3600
        return age_hours <= max_age_hours
    except Exception:
        return False


# ── Internal: single-symbol fetch ─────────────────────────────────────────────

def _fetch_symbol(sym: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch one symbol via yfinance Ticker.history(). Returns None on failure.
    Called concurrently by _fetch_all_prices.
    """
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        })
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["source"] = "yahoo_hub"
        cols = [c for c in
                ["symbol", "date", "close", "high", "low", "open", "volume", "adj_close", "source"]
                if c in df.columns]
        return df[cols]
    except Exception as e:
        log.warning("[Hub] %s: fetch failed — %s", sym, e)
        return None


# ── Internal: parallel price fetch ────────────────────────────────────────────

def _fetch_all_prices(
    symbols: set,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Fetch prices for all symbols using a thread pool.
    Returns (price_df, n_requested, n_fetched).
    Failures are logged per-symbol but do not abort the run.
    """
    sym_list = sorted(symbols)
    n_requested = len(sym_list)
    frames = []
    failed = []

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        future_to_sym = {
            pool.submit(_fetch_symbol, sym, start, end): sym
            for sym in sym_list
        }
        done = 0
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            done += 1
            try:
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
                else:
                    failed.append(sym)
            except Exception as e:
                log.warning("[Hub] %s: unexpected error — %s", sym, e)
                failed.append(sym)
            if done % 20 == 0 or done == n_requested:
                print(f"  [Hub] {done}/{n_requested} symbols fetched, {len(failed)} failed so far")

    if failed:
        log.warning("[Hub] %d symbols returned no data: %s%s",
                    len(failed), failed[:10], "..." if len(failed) > 10 else "")

    if not frames:
        return pd.DataFrame(), n_requested, 0

    combined = pd.concat(frames, ignore_index=True)
    n_fetched = combined["symbol"].nunique()
    return combined, n_requested, n_fetched


# ── Internal: manifest write ───────────────────────────────────────────────────

def _write_manifest(manifest: dict) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ── Public: main entry point ───────────────────────────────────────────────────

def run_hub(
    start_date: str,
    end_date: str,
    force: bool = False,
) -> dict:
    """
    Central data ingestion hub. Run this once before spawning agents.

    Fetches all market data (prices + FRED macro) in a single parallel pass
    and writes to data_cache/prices_live.parquet + data_cache/macro_live.parquet.
    Writes data_cache/hub_manifest.json on completion.

    Args:
        start_date: YYYY-MM-DD — historical start (use config.backtest.start_date)
        end_date:   YYYY-MM-DD — typically today
        force:      Re-fetch even if manifest is already fresh

    Returns:
        manifest dict: {status, fetched_at, symbols_requested, symbols_fetched,
                        prices_rows, macro_series, start_date, end_date, error?}
    """
    # ── Skip if fresh ──────────────────────────────────────────────────────────
    if not force and hub_is_fresh():
        log.info("[Hub] Manifest is fresh — skipping fetch")
        try:
            cached = json.loads(_MANIFEST_PATH.read_text())
            print("[Hub] Data is fresh — skipping fetch (use force=True to override)")
            return cached
        except Exception:
            pass  # corrupt manifest — fall through to re-fetch

    print(f"\n[Hub] ── Centralized Data Ingestion ──────────────────────────")
    print(f"[Hub] Range: {start_date} → {end_date}")

    manifest: dict = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "status": "failed",
    }

    # ── Prices ────────────────────────────────────────────────────────────────
    all_symbols = collect_all_symbols()
    print(f"[Hub] Fetching {len(all_symbols)} symbols across all agent universes...")

    try:
        price_df, n_req, n_fetched = _fetch_all_prices(all_symbols, start_date, end_date)

        if price_df.empty:
            raise RuntimeError("No price data returned — all symbols failed")

        price_df = normalize_prices(price_df)
        save_parquet(price_df, "prices_live")

        manifest["symbols_requested"] = n_req
        manifest["symbols_fetched"]   = n_fetched
        manifest["prices_rows"]       = len(price_df)
        print(f"[Hub] Prices: {n_fetched}/{n_req} symbols, {len(price_df):,} rows → prices_live.parquet")

    except Exception as e:
        manifest["error"]  = str(e)
        manifest["status"] = "failed"
        log.error("[Hub] Price fetch failed: %s", e)
        print(f"[Hub] ERROR: price fetch failed — {e}")
        print("[Hub] Agents will fall back to individual fetches")
        _write_manifest(manifest)
        return manifest

    # ── Macro (FRED) ───────────────────────────────────────────────────────────
    try:
        print("[Hub] Fetching FRED macro series...")
        macro_df = fetch_all_macro(start_date=start_date, end_date=end_date)
        if not macro_df.empty:
            macro_df = normalize_macro(macro_df)
            save_parquet(macro_df, "macro_live")
            n_series = macro_df["series_id"].nunique() if "series_id" in macro_df.columns else 0
            manifest["macro_series"] = n_series
            print(f"[Hub] Macro: {n_series} FRED series → macro_live.parquet")
        else:
            manifest["macro_series"] = 0
            print("[Hub] Macro: FRED returned empty — macro_live not updated")
    except Exception as e:
        log.warning("[Hub] FRED fetch failed: %s — macro_live not updated", e)
        manifest["macro_series"] = 0
        print(f"[Hub] WARNING: FRED failed ({e}) — agents will use cached macro_live")

    # ── Done ──────────────────────────────────────────────────────────────────
    manifest["status"] = "ok"
    _write_manifest(manifest)
    print(f"[Hub] ── Done ──────────────────────────────────────────────────\n")
    return manifest
