"""
scripts/seed_llm_cache.py
Seeds data_cache/llm_fundamental_cache.json with LLM fundamental analyses
for all symbols in the fundamentals dataset. Run once to activate narrative alpha.

Usage:
    .venv/bin/python scripts/seed_llm_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import Counter
import pandas as pd
from ascent.alpha.llm_fundamental import llm_fundamental_alpha, CACHE_PATH


def _compute_ratios(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw fundamentals (gross_profit, total_assets, net_income, op_cashflow)
    into ratio columns expected by llm_fundamental_alpha:
        gross_profitability = gross_profit / total_assets
        accruals            = (net_income - op_cashflow) / total_assets
        asset_growth        = total_assets / total_assets.shift(4) - 1  (per symbol)

    If columns are already in ratio form (gross_profitability, accruals, asset_growth),
    they are passed through unchanged.
    """
    df = fundamentals.copy()
    df["date"] = pd.to_datetime(df["date"])

    already_ratio = all(c in df.columns for c in ("gross_profitability", "accruals", "asset_growth"))
    if already_ratio:
        return df

    raw_cols = ("gross_profit", "total_assets", "net_income", "op_cashflow")
    has_raw = all(c in df.columns for c in raw_cols)
    if not has_raw:
        # Nothing useful to transform — return as-is and let llm_fundamental handle it
        return df

    rows = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        ta = grp["total_assets"].replace(0, float("nan"))

        grp["gross_profitability"] = grp["gross_profit"] / ta
        grp["accruals"] = (grp["net_income"] - grp["op_cashflow"]) / ta
        grp["asset_growth"] = ta / ta.shift(4) - 1
        rows.append(grp)

    if not rows:
        return df

    return pd.concat(rows, ignore_index=True)


def seed_cache(
    fundamentals: pd.DataFrame,
    cache_path: Path = CACHE_PATH,
    quarter_dates: list[str] | None = None,
) -> None:
    """
    Run llm_fundamental_alpha for each unique quarter-end date so the cache
    accumulates >=2 entries per symbol (required for narrative Q-o-Q shift).

    Args:
        fundamentals:  DataFrame with columns [symbol, date, gross_profit,
                       total_assets, net_income, op_cashflow] — OR pre-computed
                       ratio columns [gross_profitability, accruals, asset_growth].
        cache_path:    Override cache path (for tests).
        quarter_dates: Quarter-end dates to seed. Defaults to last 4 unique dates.
    """
    if fundamentals is None or fundamentals.empty:
        return

    fund = _compute_ratios(fundamentals)

    if quarter_dates is None:
        all_dates = sorted(fund["date"].unique())
        quarter_dates = [str(pd.Timestamp(d).date()) for d in all_dates[-4:]]

    import ascent.alpha.llm_fundamental as _lf
    orig_cache_path = _lf.CACHE_PATH
    _lf.CACHE_PATH = cache_path

    try:
        for q_date in quarter_dates:
            # Use cutoff = quarter_end + 46 days (after 45-day filing lag)
            cutoff = pd.Timestamp(q_date) + pd.Timedelta(days=46)
            scores = llm_fundamental_alpha(fund, as_of_date=cutoff)
            n = len(scores)
            print(f"[Seeder] {q_date}: {n} symbols scored")
    finally:
        _lf.CACHE_PATH = orig_cache_path


def main():
    try:
        from ascent.data.store.parquet import load_parquet
        fundamentals = load_parquet("fundamentals")
        print(f"[Seeder] Loaded {len(fundamentals)} fundamental rows, "
              f"{fundamentals['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"[Seeder] Could not load fundamentals: {e}")
        return

    seed_cache(fundamentals)
    print(f"[Seeder] Done — cache at {CACHE_PATH}")

    # Report how many symbols have >=2 entries (required for narrative alpha)
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        sym_counts: Counter = Counter()
        for key in cache:
            sym = key.rsplit("_", 1)[0]
            sym_counts[sym] += 1
        ready = sum(1 for c in sym_counts.values() if c >= 2)
        print(f"[Seeder] {ready}/{len(sym_counts)} symbols ready for narrative alpha (>=2 quarters)")


if __name__ == "__main__":
    main()
