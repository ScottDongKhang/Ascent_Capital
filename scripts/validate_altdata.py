#!/usr/bin/env python3
"""
scripts/validate_altdata.py
Run altdata IC validation manually — validates available sources and registers
those that pass the gate.

Usage:
    python scripts/validate_altdata.py
    python scripts/validate_altdata.py --source reddit
    python scripts/validate_altdata.py --dry-run
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ascent.data.validate.altdata_validator import (
    run_altdata_validation,
    register_altdata_source,
)


def _load_sources(source_filter: str | None) -> dict[str, pd.DataFrame]:
    """Attempt to load each available cached panel. Skip silently if missing."""
    sources: dict[str, pd.DataFrame] = {}

    # Reddit sentiment
    if source_filter is None or source_filter == "reddit":
        try:
            from ascent.data.ingest.reddit_sentiment import load_reddit_signals
            panel = load_reddit_signals()
            if panel is not None and not panel.empty:
                sources["reddit"] = panel
        except Exception as e:
            print(f"[Reddit] Could not load: {e}")

    # Google Trends
    if source_filter is None or source_filter == "trends":
        try:
            from ascent.data.ingest.google_trends import load_trends_signals
            panel = load_trends_signals()
            if panel is not None and not panel.empty:
                sources["trends"] = panel
        except Exception as e:
            print(f"[Trends] Could not load: {e}")

    # SEC / transcripts — caches not yet available, skip silently

    return sources


def _load_prices() -> pd.DataFrame:
    """Load prices_live from parquet store."""
    try:
        from ascent.data.store.parquet_store import ParquetStore
        store = ParquetStore()
        if store.has("prices_live"):
            return store.load("prices_live")
    except Exception as e:
        print(f"[Prices] ParquetStore load failed: {e}")
    return pd.DataFrame()


def _load_regime() -> pd.Series:
    """Load regime labels from dashboard/regime_labels.csv."""
    regime_path = Path("dashboard/regime_labels.csv")
    try:
        if regime_path.exists():
            df = pd.read_csv(regime_path, index_col=0, parse_dates=True)
            return df.iloc[:, 0]
    except Exception as e:
        print(f"[Regime] Could not load: {e}")
    return pd.Series(dtype=str)


def _print_result(result: dict) -> None:
    """Print a clear validation summary for one source."""
    source = result["source"]
    n_obs  = result["n_observations"]
    ic_mean = result["ic_mean"]
    ic_ir   = result["ic_ir"]
    status  = result["status"].upper()
    reasons = result.get("reasons", [])

    header = f"{source.capitalize()} altdata validation"
    print(f"\n{header}")
    print("=" * len(header))
    print(f"Loaded: {n_obs} dates (need ≥20 dates to pass gate)")

    if ic_mean is not None:
        print(f"IC_mean: {ic_mean:.3f}  IC_IR: {ic_ir:.2f}  Status: {status}")
    else:
        print(f"Status: {status}")

    if reasons:
        reason_str = "; ".join(reasons)
        print(f"Reason: {reason_str}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate altdata IC gate and register accepted sources.")
    parser.add_argument(
        "--source",
        default=None,
        help="Validate only this source (e.g. reddit, trends). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate but do not write to active_alpha_config.json.",
    )
    args = parser.parse_args(argv)

    sources = _load_sources(args.source)
    if not sources:
        print("No altdata panels loaded — run ingest first or check credentials.")
        return 0

    prices     = _load_prices()
    regime     = _load_regime()

    results = run_altdata_validation(sources, prices, regime)

    accepted = []
    for result in results:
        _print_result(result)
        if result["status"] == "accepted":
            accepted.append(result["source"])

    print()
    if not accepted:
        print("No sources passed the IC gate.")
        return 0

    print(f"{len(accepted)} source(s) accepted: {accepted}")

    if args.dry_run:
        print("[dry-run] Skipping registration — no config written.")
    else:
        for src in accepted:
            try:
                register_altdata_source(src, initial_weight=0.02)
                print(f"→ Registered {src} at weight=0.02")
            except Exception as e:
                print(f"→ Failed to register {src}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
