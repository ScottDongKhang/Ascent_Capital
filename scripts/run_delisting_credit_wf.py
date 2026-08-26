#!/usr/bin/env python
"""
Production walk-forward rerun with the delisting terminal-value credit
(ascent/research/walk_forward_runner.py::apply_delisting_terminal_credit,
fed by ascent/data/universe.py::DELISTING_TERMINAL_TERMS) active, against
the same `prices_live` cache basis as the canonical
outputs/wf_results/wf_report_clean_2026-08-15.json artifact (Sharpe 0.415,
CAGR 10.2%), so the before/after can be compared directly.

Usage
-----
    .venv/bin/python scripts/run_delisting_credit_wf.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from ascent.data.store.parquet import load_parquet, has_data

    if not has_data("prices_live"):
        print("ERROR: prices_live cache not found in data_cache/.")
        return 1

    price_df = load_parquet("prices_live")
    dmin, dmax = price_df["date"].min(), price_df["date"].max()
    n_symbols = price_df["symbol"].nunique()
    print(f"[delisting-wf] prices_live: {n_symbols} symbols, "
          f"{dmin.date()} -> {dmax.date()} ({len(price_df)} rows)")

    from ascent.data.universe import DELISTING_TERMINAL_TERMS
    present = [s for s in DELISTING_TERMINAL_TERMS if s in set(price_df["symbol"].unique())]
    print(f"[delisting-wf] {len(present)}/{len(DELISTING_TERMINAL_TERMS)} "
          f"DELISTING_TERMINAL_TERMS symbols have any price history in this "
          f"cache: {present}")

    from ascent.research.walk_forward_runner import walk_forward_pipeline
    import datetime

    before = set((ROOT / "outputs" / "wf_results").glob("wf_report_*.json")) \
        if (ROOT / "outputs" / "wf_results").exists() else set()

    t0 = time.time()
    walk_forward_pipeline()
    elapsed = time.time() - t0
    print(f"\n[delisting-wf] walk_forward_pipeline() finished in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # walk_forward_pipeline() writes its own wf_report_<date>.json -- find the
    # new one it just produced and copy it to a stable, clearly-labeled name.
    after = set((ROOT / "outputs" / "wf_results").glob("wf_report_*.json"))
    new_files = sorted(after - before)
    if new_files:
        src = new_files[-1]
        dst = ROOT / "outputs" / "wf_results" / "wf_report_delisting_credit_rerun.json"
        dst.write_text(src.read_text())
        print(f"[delisting-wf] new report: {src} -> copied to {dst}")
    else:
        print("[delisting-wf] WARNING: no new wf_report_*.json detected -- "
              "check stdout above for errors.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
