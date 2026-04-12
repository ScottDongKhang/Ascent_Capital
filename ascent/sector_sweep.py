"""
Sector Constraint Sweep
Tests max 1, 2, 3, 4, 5 stocks per sector to find optimal.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from ascent.config.settings import get_config
from ascent.data.store.parquet import load_parquet
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack
from ascent.portfolio.optimizer import rank_weighted
from ascent.research.evaluation import sharpe_ratio, annualized_return, max_drawdown

cfg = get_config()
price_df = load_parquet("prices_live")
macro_df = load_parquet("macro_live")
profiles = load_parquet("profiles")

builder = FeatureBuilder(price_df, macro_df)
features = builder.compute_features()
alpha = -build_alpha_stack(features)
close = pivot_prices(price_df, "close")
daily_returns = close.pct_change()

sector_map = dict(zip(profiles["symbol"], profiles["sector"]))


def run_with_sector_limit(alpha_df, daily_ret, max_per_sector, n_stocks=10):
    weights = pd.DataFrame(0.0, index=alpha_df.index, columns=alpha_df.columns)
    for dt in alpha_df.index:
        row = alpha_df.loc[dt].dropna()
        if len(row) < n_stocks:
            continue
        ranked = row.sort_values(ascending=False)
        selected = []
        sector_count = {}
        for sym in ranked.index:
            sec = sector_map.get(sym, "Unknown")
            if sector_count.get(sec, 0) < max_per_sector:
                selected.append(sym)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= n_stocks:
                break
        if not selected:
            continue
        scores = ranked[selected]
        scores = scores - scores.min() + 1e-8
        raw_w = scores / scores.sum()
        for _ in range(10):
            raw_w = raw_w.clip(upper=0.15)
            raw_w = raw_w / raw_w.sum()
            if raw_w.max() <= 0.15 + 1e-9:
                break
        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]
    w = weights.shift(1).dropna()
    r = daily_ret.loc[w.index]
    port_ret = (w * r).sum(axis=1).dropna()
    return port_ret, weights


print("=" * 70)
print("  SECTOR CONSTRAINT SWEEP")
print("=" * 70)
print("")
print("%-12s %8s %8s %10s %8s %8s %8s" % (
    "Constraint", "Sharpe", "CAGR", "MaxDD", "Vol", "Hit%", "Sectors"))
print("-" * 65)

# No constraint baseline
ret_none, w_none = run_with_sector_limit(alpha, daily_returns, max_per_sector=99)
sr = sharpe_ratio(ret_none)
cagr = annualized_return(ret_none) * 100
dd = max_drawdown(ret_none) * 100
vol = ret_none.std() * np.sqrt(252) * 100
hit = (ret_none > 0).mean() * 100
print("%-12s %8.3f %7.1f%% %9.1f%% %7.1f%% %7.1f%% %8s" % (
    "No limit", sr, cagr, dd, vol, hit, "-"))

# Test each sector limit
best_sharpe = 0
best_limit = 0
for limit in [1, 2, 3, 4, 5]:
    ret, w = run_with_sector_limit(alpha, daily_returns, max_per_sector=limit)
    sr = sharpe_ratio(ret)
    cagr = annualized_return(ret) * 100
    dd = max_drawdown(ret) * 100
    vol = ret.std() * np.sqrt(252) * 100
    hit = (ret > 0).mean() * 100
    # Count avg sectors
    counts = []
    for dt in w.index[::10]:
        held = w.loc[dt]
        held = held[held > 0.01]
        secs = set(sector_map.get(s, "?") for s in held.index)
        counts.append(len(secs))
    avg_sec = np.mean(counts) if counts else 0
    marker = ""
    if sr > best_sharpe:
        best_sharpe = sr
        best_limit = limit
        marker = " <-- best"
    print("%-12s %8.3f %7.1f%% %9.1f%% %7.1f%% %7.1f%% %8.1f%s" % (
        "Max %d/sec" % limit, sr, cagr, dd, vol, hit, avg_sec, marker))

print("")
print("  BEST: Max %d stocks per sector (Sharpe: %.3f)" % (best_limit, best_sharpe))
print("")
