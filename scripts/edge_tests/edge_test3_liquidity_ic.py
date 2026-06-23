"""
EDGE THESIS TEST #3 — Is there a structural wedge in less-liquid / smaller names?

For a solo operator the only durable edge is usually capacity-constrained,
uncrowded corners large funds can't touch. Test: does 12-1 cross-sectional
momentum (the sleeve that matters, per test #1) predict forward 21d returns
BETTER in low-dollar-volume names than in mega-caps?

Method (OOS window 2021-01 -> 2026-01, monthly):
  signal_t   = close[t-21] / close[t-273] - 1     (12-1 momentum, skip last month)
  fwd_t      = close[t+21] / close[t] - 1          (forward 21d return)
  liquidity  = trailing 63d mean dollar_volume
  At each month, split the cross-section into dollar-volume terciles; compute
  Spearman IC(signal, fwd) within each tercile. Average IC over all months.

Falsifier (EDGE_THESIS H3): if low-liquidity IC is NOT higher than mega-cap IC,
there is no structural wedge here — momentum is equally (un)crowded across caps.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
sys.path.insert(0, REPO)
from ascent.data.store.parquet import load_parquet

df = load_parquet("prices_live_clean_refetch")
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2026-01-31")]

close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
dvol = df.pivot_table(index="date", columns="symbol", values="dollar_volume").sort_index()

# Drop the benchmark/ETF-ish tickers? Keep all; momentum IC is cross-sectional.
idx = close.index
# monthly decision points (~every 21 trading days), leaving room for fwd window
points = list(range(273, len(idx) - 22, 21))

records = []  # (date, tercile, ic, n)
for p in points:
    t = idx[p]
    sig = close.iloc[p - 21] / close.iloc[p - 273] - 1.0          # 12-1 momentum
    fwd = close.iloc[p + 21] / close.iloc[p] - 1.0                 # forward 21d
    liq = dvol.iloc[p - 63:p].mean()                              # trailing liquidity

    panel = pd.DataFrame({"sig": sig, "fwd": fwd, "liq": liq}).dropna()
    if len(panel) < 90:
        continue
    # tercile by liquidity
    panel["bucket"] = pd.qcut(panel["liq"], 3, labels=["low_liq", "mid_liq", "high_liq"])
    for b, g in panel.groupby("bucket", observed=True):
        if len(g) < 20 or g["sig"].nunique() < 5:
            continue
        ic, _ = spearmanr(g["sig"], g["fwd"])
        if ic == ic:  # not NaN
            records.append((t, str(b), ic, len(g)))

res = pd.DataFrame(records, columns=["date", "bucket", "ic", "n"])
print(f"months evaluated: {res['date'].nunique()}  total bucket-obs: {len(res)}", flush=True)
print("\nMean momentum IC by liquidity tercile (higher = momentum predicts better):")
summary = res.groupby("bucket")["ic"].agg(["mean", "std", "count"])
# t-stat of mean IC vs 0
summary["t_vs_0"] = summary["mean"] / (summary["std"] / np.sqrt(summary["count"]))
order = ["low_liq", "mid_liq", "high_liq"]
print(summary.reindex(order).round(4).to_string())

low = summary.loc["low_liq", "mean"]
high = summary.loc["high_liq", "mean"]
print(f"\nlow_liq IC - high_liq IC = {low - high:+.4f}")
print("READING:", (
    "WEDGE: momentum predicts materially better in less-liquid names"
    if (low - high) > 0.02 else
    "NO WEDGE: momentum IC is not higher in less-liquid names — no structural edge here"
))
