"""
Sector Constraint Test
Compares current strategy vs max 2 stocks per sector.
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

# Build alpha
builder = FeatureBuilder(price_df, macro_df)
features = builder.compute_features()
alpha = -build_alpha_stack(features)
close = pivot_prices(price_df, "close")
daily_returns = close.pct_change()

# Sector map
sector_map = dict(zip(profiles["symbol"], profiles["sector"]))

# ── Strategy 1: Current (no sector constraint) ──
weights_current = rank_weighted(alpha, n=10, max_weight=0.15)
w1 = weights_current.shift(1).dropna()
r1 = daily_returns.loc[w1.index]
ret_current = (w1 * r1).sum(axis=1).dropna()

# ── Strategy 2: Sector-constrained (max 2 per sector) ──
weights_sector = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
for dt in alpha.index:
    row = alpha.loc[dt].dropna()
    if len(row) < 10:
        continue
    ranked = row.sort_values(ascending=False)
    selected = []
    sector_count = {}
    for sym in ranked.index:
        sec = sector_map.get(sym, "Unknown")
        if sector_count.get(sec, 0) < 2:
            selected.append(sym)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= 10:
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
        weights_sector.loc[dt, sym] = raw_w[sym]

w2 = weights_sector.shift(1).dropna()
r2 = daily_returns.loc[w2.index]
ret_sector = (w2 * r2).sum(axis=1).dropna()

# ── Compare ──
print("=" * 60)
print("  SECTOR CONSTRAINT TEST")
print("=" * 60)
print("")
print("%-25s %12s %12s" % ("Metric", "NO CONSTRAINT", "MAX 2/SECTOR"))
print("-" * 50)
print("%-25s %12.3f %12.3f" % ("Sharpe", sharpe_ratio(ret_current), sharpe_ratio(ret_sector)))
print("%-25s %11.1f%% %11.1f%%" % ("CAGR", annualized_return(ret_current)*100, annualized_return(ret_sector)*100))
print("%-25s %11.1f%% %11.1f%%" % ("Max Drawdown", max_drawdown(ret_current)*100, max_drawdown(ret_sector)*100))
print("%-25s %11.1f%% %11.1f%%" % ("Volatility", ret_current.std()*np.sqrt(252)*100, ret_sector.std()*np.sqrt(252)*100))
print("%-25s %11.1f%% %11.1f%%" % ("Hit Rate", (ret_current>0).mean()*100, (ret_sector>0).mean()*100))

# Avg sectors held
def avg_sectors(w_df):
    counts = []
    for dt in w_df.index[::5]:
        held = w_df.loc[dt]
        held = held[held > 0.01]
        secs = [sector_map.get(s, "?") for s in held.index]
        counts.append(len(set(secs)))
    return np.mean(counts)

print("%-25s %12.1f %12.1f" % ("Avg Sectors Held", avg_sectors(weights_current), avg_sectors(weights_sector)))
print("")
