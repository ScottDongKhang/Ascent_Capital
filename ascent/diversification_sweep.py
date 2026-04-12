"""
Diversification Sweep
Tests combinations of positions, sector limits, and weight caps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from ascent.config.settings import get_config
from ascent.data.store.parquet import load_parquet, has_data
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack
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


def run_config(n_stocks, max_per_sec, max_wt):
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < n_stocks:
            continue
        ranked = row.sort_values(ascending=False)
        selected = []
        sector_count = {}
        for sym in ranked.index:
            sec = sector_map.get(sym, "Unknown")
            if sector_count.get(sec, 0) < max_per_sec:
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
            raw_w = raw_w.clip(upper=max_wt)
            raw_w = raw_w / raw_w.sum()
            if raw_w.max() <= max_wt + 1e-9:
                break
        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]

    w = weights.shift(1).dropna()
    r = daily_returns.loc[w.index]
    ret = (w * r).sum(axis=1).dropna()

    if len(ret) < 100:
        return None

    sr = sharpe_ratio(ret)
    cagr = annualized_return(ret) * 100
    dd = max_drawdown(ret) * 100
    vol = ret.std() * np.sqrt(252) * 100
    hit = (ret > 0).mean() * 100

    # Avg positions and sectors
    pos_list = []
    sec_list = []
    for dt in weights.index[::10]:
        held = weights.loc[dt]
        held = held[held > 0.01]
        pos_list.append(len(held))
        secs = set(sector_map.get(s, "?") for s in held.index)
        sec_list.append(len(secs))

    return {
        "n": n_stocks,
        "max_sec": max_per_sec,
        "max_wt": max_wt,
        "sharpe": sr,
        "cagr": cagr,
        "dd": dd,
        "vol": vol,
        "hit": hit,
        "avg_pos": np.mean(pos_list),
        "avg_sec": np.mean(sec_list),
    }


print("=" * 90)
print("  DIVERSIFICATION SWEEP")
print("=" * 90)
print("")

results = []
configs = [
    # (n_stocks, max_per_sector, max_weight)
    (8,  1, 0.15),
    (10, 1, 0.15),
    (10, 1, 0.12),
    (10, 1, 0.10),
    (12, 1, 0.12),
    (12, 1, 0.10),
    (12, 2, 0.10),
    (15, 1, 0.10),
    (15, 2, 0.10),
    (15, 2, 0.08),
    (8,  1, 0.12),
    (10, 2, 0.12),
    (10, 2, 0.15),
]

for n, ms, mw in configs:
    r = run_config(n, ms, mw)
    if r:
        results.append(r)
        print("  N=%-2d MaxSec=%d MaxWt=%.0f%% | Sharpe: %5.3f | CAGR: %5.1f%% | DD: %6.1f%% | Vol: %5.1f%% | Pos: %.1f | Sec: %.1f" % (
            r["n"], r["max_sec"], r["max_wt"]*100,
            r["sharpe"], r["cagr"], r["dd"], r["vol"],
            r["avg_pos"], r["avg_sec"],
        ))

print("")
print("-" * 90)

# Find best by different criteria
best_sharpe = max(results, key=lambda x: x["sharpe"])
best_cagr = max(results, key=lambda x: x["cagr"])
best_dd = max(results, key=lambda x: x["dd"])  # least negative
best_balanced = max(results, key=lambda x: x["sharpe"] * 0.5 + (x["cagr"]/100) * 0.3 + (1 + x["dd"]/100) * 0.2)

print("")
print("  BEST SHARPE:    N=%-2d MaxSec=%d MaxWt=%.0f%% -> Sharpe %.3f, CAGR %.1f%%, DD %.1f%%" % (
    best_sharpe["n"], best_sharpe["max_sec"], best_sharpe["max_wt"]*100,
    best_sharpe["sharpe"], best_sharpe["cagr"], best_sharpe["dd"]))
print("  BEST CAGR:      N=%-2d MaxSec=%d MaxWt=%.0f%% -> Sharpe %.3f, CAGR %.1f%%, DD %.1f%%" % (
    best_cagr["n"], best_cagr["max_sec"], best_cagr["max_wt"]*100,
    best_cagr["sharpe"], best_cagr["cagr"], best_cagr["dd"]))
print("  BEST DRAWDOWN:  N=%-2d MaxSec=%d MaxWt=%.0f%% -> Sharpe %.3f, CAGR %.1f%%, DD %.1f%%" % (
    best_dd["n"], best_dd["max_sec"], best_dd["max_wt"]*100,
    best_dd["sharpe"], best_dd["cagr"], best_dd["dd"]))
print("  BEST BALANCED:  N=%-2d MaxSec=%d MaxWt=%.0f%% -> Sharpe %.3f, CAGR %.1f%%, DD %.1f%%" % (
    best_balanced["n"], best_balanced["max_sec"], best_balanced["max_wt"]*100,
    best_balanced["sharpe"], best_balanced["cagr"], best_balanced["dd"]))
print("")
