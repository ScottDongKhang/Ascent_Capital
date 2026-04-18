"""
Ascent Capital — Institutional Terminal Dashboard
Generates a self-contained HTML dashboard from CSV ledger files.
Usage: python3 ascent/dashboard/build_dashboard.py
"""
import json, math
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ── Live vs Backtest data ─────────────────────────────────────────────────────
import json as _json_lvb
_lvb_path = ROOT / "dashboard" / "live_vs_backtest.json"
if _lvb_path.exists():
    _lvb = _json_lvb.loads(_lvb_path.read_text())
else:
    _lvb = None
_lvb_json = _json_lvb.dumps(_lvb) if _lvb else "null"


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
daily = pd.read_csv(ROOT / "ascent_daily_ledger.csv", parse_dates=["date"]).sort_values("date").copy()
holdings = pd.read_csv(ROOT / "ascent_holdings_ledger.csv", parse_dates=["date"]).sort_values(["date", "symbol"]).copy()

ret_col = "net_return" if "net_return" in daily.columns else "daily_return"
pnl_col = "net_pnl" if "net_pnl" in daily.columns else "daily_pnl"
end_col = "end_value"

equity = daily[end_col].astype(float)
returns = daily[ret_col].astype(float).fillna(0.0)
pnl_series = daily[pnl_col].astype(float) if pnl_col in daily.columns else equity.diff().fillna(0.0)
drawdown = (equity / equity.cummax()) - 1.0
rolling_sharpe = (returns.rolling(63).mean() / returns.rolling(63).std()) * np.sqrt(252)
rolling_vol = returns.rolling(63).std() * np.sqrt(252)
rolling_ret = returns.rolling(63).mean() * 252
cum_pnl = pnl_series.cumsum()
turnover = daily["turnover"].astype(float) if "turnover" in daily.columns else pd.Series(0, index=daily.index)

# ── Core metrics ──
last_eq = equity.iloc[-1]
first_eq = equity.iloc[0]
total_return = last_eq / first_eq - 1.0
n_years = len(equity) / 252
cagr = (last_eq / first_eq) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if returns[returns < 0].std() > 0 else 0.0
max_dd = drawdown.min()
vol = returns.std() * np.sqrt(252)
calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
hit_rate = (returns > 0).mean()
best_day = returns.max()
worst_day = returns.min()
avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0
skew = returns.skew()
kurtosis = returns.kurtosis()
total_costs = daily["cost_dollars"].sum() if "cost_dollars" in daily.columns else 0
latest = daily.iloc[-1]
latest_pos = int(latest.get("positions", 0))

# VaR / CVaR
var_95 = np.percentile(returns, 5)
var_99 = np.percentile(returns, 1)
cvar_95 = returns[returns <= var_95].mean()
cvar_99 = returns[returns <= var_99].mean()

# ── Monthly returns heatmap ──
daily_copy = daily.copy()
daily_copy["year"] = daily_copy["date"].dt.year
daily_copy["month"] = daily_copy["date"].dt.month
monthly = daily_copy.groupby(["year", "month"])[ret_col].apply(lambda x: (1 + x).prod() - 1)
years = sorted(daily_copy["year"].unique())
months = list(range(1, 13))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
heatmap_data = []
for y in years:
    row = {"year": int(y)}
    yr_ret = 0
    for m in months:
        val = monthly.get((y, m), None)
        if val is not None:
            row["m%d" % m] = round(float(val) * 100, 2)
            yr_ret = (1 + yr_ret) * (1 + val) - 1 if yr_ret else val
        else:
            row["m%d" % m] = None
    row["ytd"] = round(float(yr_ret) * 100, 2) if yr_ret else None
    heatmap_data.append(row)

# ── Sector data ──
sector_map = {}
try:
    import sys; sys.path.insert(0, str(ROOT))
    from ascent.data.store.parquet import load_parquet, has_data
    if has_data("profiles"):
        profiles = load_parquet("profiles")
        sector_map = dict(zip(profiles["symbol"], profiles["sector"]))
except Exception:
    pass

# ── Attribution ──
pnl_col_h = "pnl_contribution" if "pnl_contribution" in holdings.columns else None
attr_data = []
if pnl_col_h:
    by_sym = holdings.groupby("symbol", as_index=False)[pnl_col_h].sum().sort_values(pnl_col_h, ascending=False)
    for _, row in by_sym.iterrows():
        sym = row["symbol"]
        attr_data.append({"s": sym, "pnl": round(float(row[pnl_col_h]), 2), "sec": sector_map.get(sym, "")})

# Sector attribution
sector_attr = []
if pnl_col_h and sector_map:
    holdings_copy = holdings.copy()
    holdings_copy["sector"] = holdings_copy["symbol"].map(sector_map).fillna("Unknown")
    by_sec = holdings_copy.groupby("sector", as_index=False)[pnl_col_h].sum().sort_values(pnl_col_h, ascending=False)
    for _, row in by_sec.iterrows():
        sector_attr.append({"sec": row["sector"], "pnl": round(float(row[pnl_col_h]), 2)})

# ── Current holdings ──
last_date = holdings["date"].max()
current_hold = holdings[holdings["date"] == last_date].copy()
current_data = []
for _, row in current_hold.iterrows():
    if abs(row.get("weight", 0)) > 0.001:
        current_data.append({
            "s": row["symbol"],
            "w": round(float(row["weight"]) * 100, 1),
            "r": round(float(row.get("asset_return", 0)) * 100, 2),
            "c": round(float(row.get("pnl_contribution", 0)), 2),
            "sec": sector_map.get(row["symbol"], ""),
        })

# ── Holdings browser ──
all_hold_dates = sorted(holdings["date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
hold_browser_cols = [c for c in ["date", "symbol", "weight", "asset_return", "return_contribution", "pnl_contribution"] if c in holdings.columns]
holdings_browser = holdings[hold_browser_cols].copy()
if "date" in holdings_browser.columns:
    holdings_browser["date"] = holdings_browser["date"].dt.strftime("%Y-%m-%d")
if "weight" in holdings_browser.columns:
    holdings_browser = holdings_browser[holdings_browser["weight"].abs() > 1e-8].copy()

# ── Drawdown periods ──
dd_periods = []
in_dd = False
dd_start = None
for i, (dt, dd_val) in enumerate(zip(daily["date"], drawdown)):
    if dd_val < -0.01 and not in_dd:
        in_dd = True
        dd_start = dt
    elif dd_val >= -0.001 and in_dd:
        in_dd = False
        dd_end = dt
        dd_slice = drawdown.iloc[max(0,i-252):i]
        dd_min = dd_slice.min()
        duration = (dd_end - dd_start).days
        dd_periods.append({"start": dd_start.strftime("%Y-%m-%d"), "end": dd_end.strftime("%Y-%m-%d"),
                           "depth": round(float(dd_min)*100, 1), "days": duration})
dd_periods.sort(key=lambda x: x["depth"])
dd_table = dd_periods[:10]

# ── Chart data ──
chart_data = []
for i in range(len(daily)):
    row = daily.iloc[i]
    chart_data.append({
        "d": row["date"].strftime("%Y-%m-%d"),
        "eq": round(float(equity.iloc[i]), 2),
        "r": round(float(returns.iloc[i]) * 100, 4),
        "dd": round(float(drawdown.iloc[i]) * 100, 4),
        "pnl": round(float(pnl_series.iloc[i]), 2),
        "cpnl": round(float(cum_pnl.iloc[i]), 2),
        "rs": round(float(rolling_sharpe.iloc[i]), 4) if not np.isnan(rolling_sharpe.iloc[i]) else None,
        "rv": round(float(rolling_vol.iloc[i]) * 100, 4) if not np.isnan(rolling_vol.iloc[i]) else None,
        "rr": round(float(rolling_ret.iloc[i]) * 100, 4) if not np.isnan(rolling_ret.iloc[i]) else None,
        "to": round(float(turnover.iloc[i]) * 100, 2),
        "pos": int(row.get("positions", 0)),
    })

# Ledger
daily_table = []
for _, row in daily.tail(80).iterrows():
    daily_table.append({
        "d": row["date"].strftime("%Y-%m-%d"),
        "sv": round(float(row.get("start_value", 0)), 2),
        "ev": round(float(row.get(end_col, 0)), 2),
        "pnl": round(float(row.get(pnl_col, 0)), 2),
        "r": round(float(row.get(ret_col, 0)) * 100, 4),
        "turn": round(float(row.get("turnover", 0)) * 100, 2),
        "cost": round(float(row.get("cost_dollars", 0)), 2),
        "pos": int(row.get("positions", 0)),
        "reb": bool(row.get("is_rebalance", False)),
    })

# Return distribution
ret_hist = np.histogram(returns.dropna() * 100, bins=50)
ret_dist = [{"x": round(float((ret_hist[1][i] + ret_hist[1][i+1])/2), 2), "y": int(ret_hist[0][i])} for i in range(len(ret_hist[0]))]

# ══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════

# Load regime signal JSON
import json as _json
_regime_path = ROOT / "dashboard" / "regime_signal.json"
if _regime_path.exists():
    with open(_regime_path) as _f:
        regime_json_str = _json.dumps(_json.load(_f))
else:
    regime_json_str = "[]"

html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ASCENT CAPITAL</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#030508;--p:#080c12;--p2:#0c1118;--p3:#101820;--bd:#162030;--bd2:#1a2a3a;
--tx:#c8d4e0;--br:#e8f0f8;--mt:#5a6a7a;--dm:#3a4a5a;--am:#e8a308;--am2:#d4940a;
--amd:rgba(232,163,8,0.06);--g:#10b981;--r:#ef4444;--b:#3b82f6;--pu:#8b5cf6;
--f:'JetBrains Mono',monospace}
html{font-size:11px}body{background:var(--bg);color:var(--tx);font-family:var(--f);-webkit-font-smoothing:antialiased}
::selection{background:var(--am);color:#000}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}

.top{background:#000;border-bottom:1px solid var(--bd);display:flex;align-items:center;padding:0 14px;height:34px;gap:12px;position:sticky;top:0;z-index:100}
.logo{font-size:14px;font-weight:800;color:var(--am);letter-spacing:2px}
.cmd{flex:1;background:var(--p);border:1px solid var(--bd);padding:4px 8px;color:var(--mt);font-size:9px;font-family:var(--f);letter-spacing:.5px;text-transform:uppercase}
.dot{width:5px;height:5px;border-radius:50%;background:var(--g);display:inline-block;margin-right:3px}
.clk{color:var(--mt);font-size:9px;white-space:nowrap}

.strip{display:flex;border-bottom:1px solid var(--bd);background:linear-gradient(180deg,#060a10,var(--bg));overflow-x:auto;flex-wrap:nowrap}
.si{flex:0 0 auto;padding:5px 11px;border-right:1px solid var(--bd)}
.si:last-child{border-right:none}
.sl{font-size:8px;color:var(--mt);text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px}
.sv{font-size:14px;font-weight:700;color:var(--br)}
.sv.p{color:var(--g)}.sv.n{color:var(--r)}.sv.a{color:var(--am)}

.ct{padding:8px 12px 30px 12px}
.rw{display:grid;gap:6px;margin-bottom:6px}
.r2{grid-template-columns:1fr 1fr}.r3{grid-template-columns:1fr 1fr 1fr}.r4{grid-template-columns:1fr 1fr 1fr 1fr}
.r21{grid-template-columns:2fr 1fr}.r12{grid-template-columns:1fr 2fr}.r31{grid-template-columns:3fr 1fr}

.cd{background:var(--p);border:1px solid var(--bd);overflow:hidden}
.ch{display:flex;align-items:center;justify-content:space-between;padding:5px 8px;border-bottom:1px solid var(--bd);background:var(--p2)}
.ct2{font-size:9px;font-weight:700;color:var(--am);text-transform:uppercase;letter-spacing:.7px}
.cb{padding:6px 8px;position:relative}

.rb{display:flex;gap:2px}
.rbn{background:transparent;border:1px solid var(--bd);color:var(--mt);font-family:var(--f);font-size:8px;padding:1px 5px;cursor:pointer;letter-spacing:.4px;transition:all .12s}
.rbn:hover{border-color:var(--am);color:var(--am)}.rbn.ac{background:var(--am);color:#000;border-color:var(--am);font-weight:700}

.cw{width:100%;height:180px;position:relative}.cw canvas{width:100%!important;height:100%!important;display:block}
.cw.tall{height:220px}
.tt{position:absolute;background:rgba(0,0,0,.94);border:1px solid var(--am);padding:4px 8px;font-size:9px;color:var(--br);pointer-events:none;display:none;z-index:10;white-space:nowrap}

.tw{max-height:340px;overflow:auto}.tw.short{max-height:220px}
table.t{width:100%;border-collapse:collapse;font-size:9px}
table.t th{background:var(--p2);color:var(--am);font-weight:600;padding:4px 6px;text-align:left;position:sticky;top:0;border-bottom:1px solid var(--bd2);font-size:8px;text-transform:uppercase;letter-spacing:.4px}
table.t td{padding:3px 6px;border-bottom:1px solid var(--bd)}
table.t tr:hover td{background:var(--amd)}
.pp{color:var(--g)}.pn{color:var(--r)}

/* Heatmap */
.hm{width:100%;border-collapse:collapse;font-size:9px}
.hm th{padding:3px 5px;color:var(--am);font-weight:600;font-size:8px;text-transform:uppercase;text-align:center;background:var(--p2);position:sticky;top:0}
.hm td{padding:3px 5px;text-align:center;font-weight:600;border:1px solid var(--bg)}

/* Metric boxes */
.mbox{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.mb{background:var(--p2);border:1px solid var(--bd);padding:6px 8px}
.mb .ml{font-size:8px;color:var(--mt);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.mb .mv{font-size:13px;font-weight:700}

.sec{font-size:8px;color:var(--mt);text-transform:uppercase;letter-spacing:1px;font-weight:700;margin:10px 0 4px 2px}

/* Sector bar */
.sbar{display:flex;height:20px;border-radius:2px;overflow:hidden;margin:4px 0}
.sseg{display:flex;align-items:center;justify-content:center;font-size:7px;font-weight:700;color:#000;min-width:1px}

@media(max-width:900px){.r2,.r3,.r4,.r21,.r12,.r31{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="top">
<div class="logo">ASCENT CAPITAL</div>
<div class="cmd">WALK-FORWARD OOS &bull; MULTI-FACTOR &bull; SECTOR-NEUTRAL &bull; 89 EQUITIES &bull; 11 SECTORS</div>
<div class="clk"><span class="dot"></span>""" + daily["date"].iloc[-1].strftime("%Y-%m-%d %H:%M") + """</div>
</div>

<!-- METRICS STRIP -->
<div class="strip">
<div class="si"><div class="sl">NAV</div><div class="sv a">$""" + f"{last_eq:,.0f}" + """</div></div>
<div class="si"><div class="sl">Total Return</div><div class="sv """ + ("p" if total_return > 0 else "n") + """">""" + f"{total_return*100:+.1f}%" + """</div></div>
<div class="si"><div class="sl">CAGR</div><div class="sv """ + ("p" if cagr > 0 else "n") + """">""" + f"{cagr*100:+.1f}%" + """</div></div>
<div class="si"><div class="sl">Sharpe</div><div class="sv">""" + f"{sharpe:.2f}" + """</div></div>
<div class="si"><div class="sl">Sortino</div><div class="sv">""" + f"{sortino:.2f}" + """</div></div>
<div class="si"><div class="sl">Max DD</div><div class="sv n">""" + f"{max_dd*100:.1f}%" + """</div></div>
<div class="si"><div class="sl">Vol</div><div class="sv">""" + f"{vol*100:.1f}%" + """</div></div>
<div class="si"><div class="sl">Calmar</div><div class="sv">""" + f"{calmar:.2f}" + """</div></div>
<div class="si"><div class="sl">Hit Rate</div><div class="sv">""" + f"{hit_rate*100:.1f}%" + """</div></div>
<div class="si"><div class="sl">Win/Loss</div><div class="sv">""" + f"{win_loss:.2f}" + """</div></div>
<div class="si"><div class="sl">VaR 95</div><div class="sv n">""" + f"{var_95*100:.2f}%" + """</div></div>
<div class="si"><div class="sl">CVaR 95</div><div class="sv n">""" + f"{cvar_95*100:.2f}%" + """</div></div>
<div class="si"><div class="sl">Skew</div><div class="sv">""" + f"{skew:.2f}" + """</div></div>
<div class="si"><div class="sl">Kurtosis</div><div class="sv">""" + f"{kurtosis:.1f}" + """</div></div>
<div class="si"><div class="sl">Positions</div><div class="sv">""" + f"{latest_pos}" + """</div></div>
<div class="si"><div class="sl">Costs</div><div class="sv">$""" + f"{total_costs:,.0f}" + """</div></div>
</div>

<div class="ct">

<!-- ROW 1: EQUITY + DRAWDOWN -->
<div class="sec">Performance</div>
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Equity Curve</div><div class="rb" data-c="equity">
<button class="rbn" data-r="1M">1M</button><button class="rbn" data-r="6M">6M</button><button class="rbn" data-r="YTD">YTD</button><button class="rbn" data-r="1Y">1Y</button><button class="rbn" data-r="5Y">5Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw tall" id="ch-equity"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Drawdown</div><div class="rb" data-c="dd">
<button class="rbn" data-r="6M">6M</button><button class="rbn" data-r="1Y">1Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw tall" id="ch-dd"></div></div></div>
</div>

<!-- ROW 2: P&L -->
<div class="sec">Profit & Loss</div>
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Daily P&L</div><div class="rb" data-c="dpnl">
<button class="rbn" data-r="1M">1M</button><button class="rbn" data-r="6M">6M</button><button class="rbn" data-r="YTD">YTD</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-dpnl"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Cumulative P&L</div><div class="rb" data-c="cpnl">
<button class="rbn" data-r="6M">6M</button><button class="rbn" data-r="1Y">1Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-cpnl"></div></div></div>
</div>

<!-- ROW 3: RISK -->
<div class="sec">Risk Diagnostics</div>
<div class="rw r3">
<div class="cd"><div class="ch"><div class="ct2">Rolling Sharpe (63D)</div><div class="rb" data-c="rsh">
<button class="rbn" data-r="1Y">1Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-rsh"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Rolling Volatility</div><div class="rb" data-c="rvl">
<button class="rbn" data-r="1Y">1Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-rvl"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Return Distribution</div></div>
<div class="cb"><div class="cw" id="ch-dist"></div></div></div>
</div>

<!-- ROW 4: ROLLING RETURN + TURNOVER -->
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Rolling Return (63D Ann.)</div><div class="rb" data-c="rrt">
<button class="rbn" data-r="1Y">1Y</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-rrt"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Portfolio Turnover</div><div class="rb" data-c="turn">
<button class="rbn" data-r="6M">6M</button><button class="rbn ac" data-r="MAX">MAX</button>
</div></div><div class="cb"><div class="cw" id="ch-turn"></div></div></div>
</div>


<!-- LIVE VS BACKTEST -->
<div class="sec">Live vs Backtest</div>
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Cumulative Return: Live vs Backtest</div></div>
<div class="cb"><div class="cw" id="ch-lvb"></div></div></div>
<div class="cd" id="lvb-summary-card"><div class="ch"><div class="ct2">Track Record Summary</div></div>
<div class="cb" id="lvb-summary" style="padding:8px"></div></div>
</div>

<!-- REGIME DETECTION -->
<div class="sec">Regime Detection</div>
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Market Regime Signal</div></div>
<div class="cb"><div class="cw" id="ch-regime"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Regime Monitor</div></div>
<div class="cb" id="regime-monitor" style="padding:8px"></div></div>
</div>
<!-- ROW 5: RISK METRICS + DRAWDOWN TABLE -->
<div class="sec">Risk Detail</div>
<div class="rw r2">
<div class="cd"><div class="ch"><div class="ct2">Risk Metrics</div></div><div class="cb">
<div class="mbox">
<div class="mb"><div class="ml">VaR 95%</div><div class="mv pn">""" + f"{var_95*100:.2f}%" + """</div></div>
<div class="mb"><div class="ml">VaR 99%</div><div class="mv pn">""" + f"{var_99*100:.2f}%" + """</div></div>
<div class="mb"><div class="ml">CVaR 95%</div><div class="mv pn">""" + f"{cvar_95*100:.2f}%" + """</div></div>
<div class="mb"><div class="ml">CVaR 99%</div><div class="mv pn">""" + f"{cvar_99*100:.2f}%" + """</div></div>
<div class="mb"><div class="ml">Best Day</div><div class="mv pp">""" + f"{best_day*100:+.2f}%" + """</div></div>
<div class="mb"><div class="ml">Worst Day</div><div class="mv pn">""" + f"{worst_day*100:.2f}%" + """</div></div>
<div class="mb"><div class="ml">Avg Win</div><div class="mv pp">""" + f"{avg_win*100:+.3f}%" + """</div></div>
<div class="mb"><div class="ml">Avg Loss</div><div class="mv pn">""" + f"{avg_loss*100:.3f}%" + """</div></div>
<div class="mb"><div class="ml">Skewness</div><div class="mv">""" + f"{skew:.3f}" + """</div></div>
<div class="mb"><div class="ml">Kurtosis</div><div class="mv">""" + f"{kurtosis:.1f}" + """</div></div>
<div class="mb"><div class="ml">Win/Loss Ratio</div><div class="mv">""" + f"{win_loss:.2f}" + """</div></div>
<div class="mb"><div class="ml">Total Costs</div><div class="mv">$""" + f"{total_costs:,.0f}" + """</div></div>
</div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Worst Drawdown Periods</div></div><div class="cb"><div class="tw short" id="dd-table"></div></div></div>
</div>

<!-- ROW 6: MONTHLY HEATMAP -->
<div class="sec">Monthly Returns</div>
<div class="rw"><div class="cd"><div class="ch"><div class="ct2">Monthly Return Heatmap (%)</div></div><div class="cb"><div class="tw" id="heatmap"></div></div></div></div>

<!-- ROW 7: ATTRIBUTION -->
<div class="sec">Attribution</div>
<div class="rw r3">
<div class="cd"><div class="ch"><div class="ct2">Current Holdings</div></div><div class="cb"><div class="tw short" id="cur-hold"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Symbol P&L Attribution</div></div><div class="cb"><div class="tw" id="attr-sym"></div></div></div>
<div class="cd"><div class="ch"><div class="ct2">Sector P&L Attribution</div></div><div class="cb"><div class="tw short" id="attr-sec"></div></div></div>
</div>

<!-- ROW 8: HOLDINGS BROWSER -->
<div class="sec">Holdings Browser</div>
<div class="rw"><div class="cd"><div class="ch"><div class="ct2">Daily Holdings Detail</div></div><div class="cb">
<div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
<label style="font-size:8px;color:var(--mt);text-transform:uppercase">Date:</label>
<select id="holdDate" style="background:var(--p2);color:var(--tx);border:1px solid var(--bd);font-family:var(--f);font-size:9px;padding:3px 6px"></select>
</div>
<div class="tw" id="hold-browse"></div>
</div></div></div>

<!-- ROW 9: LEDGER -->
<div class="sec">Daily Ledger</div>
<div class="rw"><div class="cd"><div class="ch"><div class="ct2">Daily Portfolio Ledger (Last 80 Days)</div></div><div class="cb"><div class="tw" id="ledger"></div></div></div></div>

</div>

<script>
const LVB=""" + _lvb_json + """;
const D=""" + json.dumps(chart_data) + """;
// Regime data
const REGIME_DATA=""" + regime_json_str + """;
const ATTR=""" + json.dumps(attr_data) + """;
const SATTR=""" + json.dumps(sector_attr) + """;
const HOLD=""" + json.dumps(holdings_browser.to_dict(orient="records")) + """;
const HD=""" + json.dumps(all_hold_dates) + """;
const LED=""" + json.dumps(daily_table) + """;
const HEAT=""" + json.dumps(heatmap_data) + """;
const DDT=""" + json.dumps(dd_table) + """;
const CUR=""" + json.dumps(current_data) + """;
const DIST=""" + json.dumps(ret_dist) + """;

function fR(data,r){if(r==='MAX')return data;const ld=new Date(data[data.length-1].d);let c;
switch(r){case'1D':c=new Date(ld);c.setDate(c.getDate()-1);break;case'5D':c=new Date(ld);c.setDate(c.getDate()-5);break;
case'1M':c=new Date(ld);c.setMonth(c.getMonth()-1);break;case'6M':c=new Date(ld);c.setMonth(c.getMonth()-6);break;
case'YTD':c=new Date(ld.getFullYear(),0,1);break;case'1Y':c=new Date(ld);c.setFullYear(c.getFullYear()-1);break;
case'5Y':c=new Date(ld);c.setFullYear(c.getFullYear()-5);break;default:return data;}
return data.filter(d=>d.d>=c.toISOString().slice(0,10));}

function dL(el,data,key,col,fa,pct,ref){
el.innerHTML='';const cv=document.createElement('canvas');const dp=devicePixelRatio||1;
const w=el.clientWidth,h=el.clientHeight;cv.width=w*dp;cv.height=h*dp;cv.style.width=w+'px';cv.style.height=h+'px';
el.appendChild(cv);const x=cv.getContext('2d');x.scale(dp,dp);
const vs=data.map(d=>d[key]).filter(v=>v!=null);if(vs.length<2){x.fillStyle='#5a6a7a';x.fillText('No data',20,30);return;}
const P={l:48,r:8,t:6,b:18},pw=w-P.l-P.r,ph=h-P.t-P.b;
let mn=Math.min(...vs),mx=Math.max(...vs);if(mn===mx){mn-=1;mx+=1}const vr=mx-mn;
x.strokeStyle='#162030';x.lineWidth=.5;
for(let i=0;i<=3;i++){const y=P.t+(i/3)*ph;x.beginPath();x.moveTo(P.l,y);x.lineTo(w-P.r,y);x.stroke();
x.fillStyle='#5a6a7a';x.font='8px JetBrains Mono';const l=mx-(i/3)*vr;x.fillText(pct?l.toFixed(1)+'%':Math.abs(l)>9999?(l/1e3).toFixed(0)+'K':'$'+l.toFixed(0),1,y+3);}
if(ref!==undefined){const ry=P.t+(1-(ref-mn)/vr)*ph;x.strokeStyle='#3a4a5a';x.setLineDash([3,3]);x.beginPath();x.moveTo(P.l,ry);x.lineTo(w-P.r,ry);x.stroke();x.setLineDash([]);}
const pts=[];let idx=0;for(let i=0;i<data.length;i++){const v=data[i][key];if(v==null)continue;
pts.push({x:P.l+(idx/(vs.length-1))*pw,y:P.t+(1-(v-mn)/vr)*ph,d:data[i].d,v});idx++;}
x.beginPath();x.moveTo(pts[0].x,P.t+ph);pts.forEach(p=>x.lineTo(p.x,p.y));x.lineTo(pts[pts.length-1].x,P.t+ph);x.closePath();
const rgb=col.match(/\\d+/g);x.fillStyle=`rgba(${rgb[0]},${rgb[1]},${rgb[2]},${fa})`;x.fill();
x.beginPath();pts.forEach((p,i)=>i===0?x.moveTo(p.x,p.y):x.lineTo(p.x,p.y));x.strokeStyle=col;x.lineWidth=1.5;x.stroke();
x.fillStyle='#5a6a7a';x.font='8px JetBrains Mono';if(pts.length){x.fillText(pts[0].d,P.l,h-3);x.textAlign='right';x.fillText(pts[pts.length-1].d,w-P.r,h-3);x.textAlign='left';}
const tp=document.createElement('div');tp.className='tt';el.appendChild(tp);
cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),mx2=e.clientX-r.left;let cl=pts[0],md=1e9;
pts.forEach(p=>{const d=Math.abs(p.x-mx2);if(d<md){md=d;cl=p;}});
if(cl&&md<40){tp.style.display='block';tp.style.left=(cl.x+8)+'px';tp.style.top=(cl.y-24)+'px';
tp.innerHTML=`<b>${cl.d}</b><br>${pct?cl.v.toFixed(2)+'%':'$'+cl.v.toLocaleString()}`;}else tp.style.display='none';};
cv.onmouseleave=()=>tp.style.display='none';}

function dB(el,data,key){
el.innerHTML='';const cv=document.createElement('canvas');const dp=devicePixelRatio||1;
const w=el.clientWidth,h=el.clientHeight;cv.width=w*dp;cv.height=h*dp;cv.style.width=w+'px';cv.style.height=h+'px';
el.appendChild(cv);const x=cv.getContext('2d');x.scale(dp,dp);
const vs=data.map(d=>d[key]||0);if(!vs.length)return;
const P={l:48,r:8,t:6,b:18},pw=w-P.l-P.r,ph=h-P.t-P.b;
let mn=Math.min(0,...vs),mx=Math.max(0,...vs);if(mn===mx){mn-=1;mx+=1}const vr=mx-mn;
const zy=P.t+(1-(0-mn)/vr)*ph;x.strokeStyle='#3a4a5a';x.lineWidth=.5;x.beginPath();x.moveTo(P.l,zy);x.lineTo(w-P.r,zy);x.stroke();
const bw=Math.max(1,pw/vs.length);
vs.forEach((v,i)=>{const bx=P.l+i*bw,vy=P.t+(1-(v-mn)/vr)*ph;
x.fillStyle=v>=0?'rgba(16,185,129,.7)':'rgba(239,68,68,.7)';x.fillRect(bx,Math.min(vy,zy),Math.max(1,bw-1),Math.abs(vy-zy)||1);});
x.fillStyle='#5a6a7a';x.font='8px JetBrains Mono';if(data.length){x.fillText(data[0].d,P.l,h-3);x.textAlign='right';x.fillText(data[data.length-1].d,w-P.r,h-3);x.textAlign='left';}}

function dH(el,data){
el.innerHTML='';const cv=document.createElement('canvas');const dp=devicePixelRatio||1;
const w=el.clientWidth,h=el.clientHeight;cv.width=w*dp;cv.height=h*dp;cv.style.width=w+'px';cv.style.height=h+'px';
el.appendChild(cv);const x=cv.getContext('2d');x.scale(dp,dp);
if(!data.length)return;const P={l:48,r:8,t:6,b:18},pw=w-P.l-P.r,ph=h-P.t-P.b;
const mx=Math.max(...data.map(d=>d.y));const bw=pw/data.length;
data.forEach((d,i)=>{const bx=P.l+i*bw,bh=(d.y/mx)*ph;
x.fillStyle=d.x>=0?'rgba(16,185,129,.6)':'rgba(239,68,68,.6)';
x.fillRect(bx,P.t+ph-bh,Math.max(1,bw-1),bh);});
x.fillStyle='#5a6a7a';x.font='8px JetBrains Mono';
x.fillText(data[0].x.toFixed(1)+'%',P.l,h-3);x.textAlign='right';x.fillText(data[data.length-1].x.toFixed(1)+'%',w-P.r,h-3);}

const CH={
equity:{el:'ch-equity',key:'eq',t:'l',col:'rgb(232,163,8)',fa:.06,pct:false},
dd:{el:'ch-dd',key:'dd',t:'l',col:'rgb(239,68,68)',fa:.06,pct:true,ref:0},
dpnl:{el:'ch-dpnl',key:'pnl',t:'b'},
cpnl:{el:'ch-cpnl',key:'cpnl',t:'l',col:'rgb(16,185,129)',fa:.06,pct:false,ref:0},
rsh:{el:'ch-rsh',key:'rs',t:'l',col:'rgb(59,130,246)',fa:.05,pct:false,ref:0},
rvl:{el:'ch-rvl',key:'rv',t:'l',col:'rgb(139,92,246)',fa:.05,pct:true},
rrt:{el:'ch-rrt',key:'rr',t:'l',col:'rgb(16,185,129)',fa:.05,pct:true,ref:0},
turn:{el:'ch-turn',key:'to',t:'b'},
};
const CR={};Object.keys(CH).forEach(k=>CR[k]='MAX');
function rC(n){const c=CH[n];const fd=fR(D,CR[n]);if(c.t==='b')dB(document.getElementById(c.el),fd,c.key);
else dL(document.getElementById(c.el),fd,c.key,c.col,c.fa,c.pct,c.ref);}
function rA(){Object.keys(CH).forEach(rC);dH(document.getElementById('ch-dist'),DIST);}
document.querySelectorAll('.rb').forEach(g=>{const cn=g.dataset.c;g.querySelectorAll('.rbn').forEach(b=>{
b.onclick=()=>{g.querySelectorAll('.rbn').forEach(x=>x.classList.remove('ac'));b.classList.add('ac');CR[cn]=b.dataset.r;rC(cn);};});});

// Tables
function fp(v){return v>=0?'<span class="pp">$'+v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})+'</span>':'<span class="pn">$'+v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})+'</span>';}
function fr(v){return v>=0?'<span class="pp">'+v.toFixed(2)+'%</span>':'<span class="pn">'+v.toFixed(2)+'%</span>';}

// Current holdings
(()=>{const w=document.getElementById('cur-hold');let h='<table class="t"><thead><tr><th>Symbol</th><th>Sector</th><th>Weight</th><th>Return</th><th>P&L</th></tr></thead><tbody>';
CUR.forEach(r=>{h+=`<tr><td>${r.s}</td><td>${r.sec}</td><td>${r.w.toFixed(1)}%</td><td>${fr(r.r)}</td><td>${fp(r.c)}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;})();

// Symbol attribution
(()=>{const w=document.getElementById('attr-sym');let h='<table class="t"><thead><tr><th>Symbol</th><th>Sector</th><th>Total P&L</th></tr></thead><tbody>';
ATTR.forEach(r=>{h+=`<tr><td>${r.s}</td><td>${r.sec}</td><td>${fp(r.pnl)}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;})();

// Sector attribution
(()=>{const w=document.getElementById('attr-sec');let h='<table class="t"><thead><tr><th>Sector</th><th>Total P&L</th></tr></thead><tbody>';
SATTR.forEach(r=>{h+=`<tr><td>${r.sec}</td><td>${fp(r.pnl)}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;})();

// Drawdown periods
(()=>{const w=document.getElementById('dd-table');let h='<table class="t"><thead><tr><th>Start</th><th>End</th><th>Depth</th><th>Days</th></tr></thead><tbody>';
DDT.forEach(r=>{h+=`<tr><td>${r.start}</td><td>${r.end}</td><td><span class="pn">${r.depth.toFixed(1)}%</span></td><td>${r.days}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;})();

// Heatmap
(()=>{const w=document.getElementById('heatmap');const ms=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
let h='<table class="hm"><thead><tr><th>Year</th>';ms.forEach(m=>{h+=`<th>${m}</th>`;});h+='<th>YTD</th></tr></thead><tbody>';
HEAT.forEach(r=>{h+=`<tr><td style="font-weight:700;color:var(--am)">${r.year}</td>`;
for(let i=1;i<=12;i++){const v=r['m'+i];if(v===null||v===undefined){h+='<td style="background:#0a0d11">—</td>';}
else{const c=v>3?'#065f46':v>1?'#047857':v>0?'#064e3b':v>-1?'#450a0a':v>-3?'#7f1d1d':'#991b1b';
const tc=Math.abs(v)>5?'#fff':'var(--tx)';h+=`<td style="background:${c};color:${tc}">${v.toFixed(1)}</td>`;}}
const ytd=r.ytd;if(ytd!==null){const c2=ytd>0?'var(--g)':'var(--r)';h+=`<td style="color:${c2};font-weight:700">${ytd.toFixed(1)}</td>`;}
else h+='<td>—</td>';h+='</tr>';});h+='</tbody></table>';w.innerHTML=h;})();

// Holdings browser
(()=>{const sel=document.getElementById('holdDate');
HD.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;sel.appendChild(o);});
if(HD.length)sel.value=HD[HD.length-1];
function render(dt){const rows=HOLD.filter(r=>r.date===dt);const w=document.getElementById('hold-browse');
let h='<table class="t"><thead><tr><th>Symbol</th><th>Weight</th><th>Return</th><th>Contribution</th><th>P&L</th></tr></thead><tbody>';
rows.forEach(r=>{h+=`<tr><td>${r.symbol}</td><td>${(r.weight*100).toFixed(1)}%</td><td>${fr(r.asset_return*100)}</td><td>${fr(r.return_contribution*100)}</td><td>${fp(r.pnl_contribution||0)}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;}
render(sel.value);sel.onchange=e=>render(e.target.value);})();

// Ledger
(()=>{const w=document.getElementById('ledger');
let h='<table class="t"><thead><tr><th>Date</th><th>Start</th><th>End</th><th>P&L</th><th>Return</th><th>Turnover</th><th>Cost</th><th>Pos</th><th>Reb</th></tr></thead><tbody>';
LED.forEach(r=>{h+=`<tr><td>${r.d}</td><td>$${r.sv.toLocaleString()}</td><td>$${r.ev.toLocaleString()}</td><td>${fp(r.pnl)}</td><td>${fr(r.r)}</td><td>${r.turn.toFixed(1)}%</td><td>$${r.cost.toFixed(0)}</td><td>${r.pos}</td><td>${r.reb?'<span style="color:var(--am)">&#9679;</span>':''}</td></tr>`;});
h+='</tbody></table>';w.innerHTML=h;})();

// Draw regime signal chart — institutional grade
(()=>{
  const el=document.getElementById('ch-regime');
  if(!el||!REGIME_DATA||!REGIME_DATA.length)return;
  el.innerHTML='';
  const cv=document.createElement('canvas');
  const dp=devicePixelRatio||1;
  const PAD={t:18,r:16,b:28,l:8};
  cv.width=el.offsetWidth*dp;cv.height=el.offsetHeight*dp;
  cv.style.width=el.offsetWidth+'px';cv.style.height=el.offsetHeight+'px';
  el.appendChild(cv);
  const ctx=cv.getContext('2d'); ctx.scale(dp,dp);
  const W=el.offsetWidth,H=el.offsetHeight;
  const cW=W-PAD.l-PAD.r, cH=H-PAD.t-PAD.b;
  const data=REGIME_DATA;
  const cols={CALM:'#064e3b',ELEVATED:'#451a03',STRESS:'#431407',CRISIS:'#3b0764',UNCERTAIN:'#1e293b'};
  const lcols={CALM:'#10b981',ELEVATED:'#f59e0b',STRESS:'#f97316',CRISIS:'#a855f7',UNCERTAIN:'#64748b'};
  const bw=cW/data.length;

  // Draw regime background bands with subtle gradient
  data.forEach((d,i)=>{
    const x=PAD.l+i*bw;
    const grad=ctx.createLinearGradient(x,PAD.t,x,PAD.t+cH);
    const bc=cols[d.label]||'#1e293b';
    grad.addColorStop(0,bc+'cc');
    grad.addColorStop(1,bc+'33');
    ctx.fillStyle=grad;
    ctx.fillRect(x,PAD.t,bw+0.5,cH);
  });

  // Smooth VIX signal line using bezier
  const pts=data.map((d,i)=>({
    x:PAD.l+i*bw+bw/2,
    y:PAD.t+cH-(d.rs*cH*0.82+cH*0.06)
  }));

  // Glow effect
  ctx.save();
  ctx.shadowColor='#f59e0b';
  ctx.shadowBlur=6;
  ctx.beginPath();
  ctx.strokeStyle='#f59e0b';
  ctx.lineWidth=1.5;
  ctx.lineJoin='round';
  pts.forEach((p,i)=>{
    if(i===0){ctx.moveTo(p.x,p.y);return;}
    const prev=pts[i-1];
    const mx=(prev.x+p.x)/2;
    ctx.bezierCurveTo(mx,prev.y,mx,p.y,p.x,p.y);
  });
  ctx.stroke();
  ctx.restore();

  // Horizontal regime threshold lines
  const thresholds=[
    {y:0.1,label:'CALM',col:'#10b981'},
    {y:0.4,label:'ELEV',col:'#f59e0b'},
    {y:0.7,label:'STRESS',col:'#f97316'},
    {y:0.95,label:'CRISIS',col:'#a855f7'}
  ];
  thresholds.forEach(t=>{
    const y=PAD.t+cH-(t.y*cH*0.82+cH*0.06);
    ctx.beginPath();
    ctx.setLineDash([3,5]);
    ctx.strokeStyle=t.col+'55';
    ctx.lineWidth=0.8;
    ctx.moveTo(PAD.l,y);ctx.lineTo(PAD.l+cW,y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=t.col+'99';
    ctx.font='500 8px JetBrains Mono, monospace';
    ctx.fillText(t.label,PAD.l+2,y-2);
  });

  // X-axis date labels — yearly
  ctx.fillStyle='#475569';
  ctx.font='9px JetBrains Mono, monospace';
  ctx.textAlign='center';
  let lastYear='';
  data.forEach((d,i)=>{
    const yr=d.d.slice(0,4);
    if(yr!==lastYear){
      lastYear=yr;
      const x=PAD.l+i*bw;
      ctx.fillText(yr,x,H-6);
      ctx.beginPath();
      ctx.strokeStyle='#1e293b';
      ctx.lineWidth=0.5;
      ctx.moveTo(x,PAD.t);ctx.lineTo(x,PAD.t+cH);
      ctx.stroke();
    }
  });

  // Regime monitor panel
  const last=data[data.length-1];
  const mon=document.getElementById('regime-monitor');
  if(mon){
    const c=lcols[last.label]||'#64748b';
    const pct={CALM:0,ELEVATED:0,STRESS:0,CRISIS:0};
    data.forEach(d=>{if(pct[d.label]!==undefined)pct[d.label]++;});
    const total=data.length;
    const bars=Object.entries(pct).map(([k,v])=>`
      <div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
          <span style="font-size:9px;color:${lcols[k]};letter-spacing:.5px;text-transform:uppercase">${k}</span>
          <span style="font-size:9px;color:#94a3b8">${(v/total*100).toFixed(0)}%</span>
        </div>
        <div style="height:3px;background:#1e293b;border-radius:2px">
          <div style="height:3px;width:${(v/total*100).toFixed(0)}%;background:${lcols[k]};border-radius:2px;transition:width .3s"></div>
        </div>
      </div>`).join('');
    mon.innerHTML=`
      <div style="margin-bottom:12px">
        <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px">Current State</div>
        <div style="font-size:22px;font-weight:700;color:${c};letter-spacing:1px;font-family:JetBrains Mono,monospace">${last.label}</div>
        <div style="font-size:9px;color:#475569;margin-top:2px">Risk Multiplier <span style="color:#94a3b8">${last.risk_mult}</span></div>
      </div>
      <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px">Historical Distribution</div>
      ${bars}
      <div style="margin-top:10px;font-size:8px;color:#334155">Inputs: VIX Level &bull; Thresholds: 15 / 20 / 30</div>`;
  }
})();

  // ── Live vs Backtest panel ──────────────────────────────────────────────
  (function(){
    const lvbEl = document.getElementById('ch-lvb');
    const sumEl = document.getElementById('lvb-summary');
    if (!lvbEl || !LVB) {
      if (lvbEl) lvbEl.innerHTML = '<div style="color:#475569;font-size:10px;padding:16px">No live vs backtest data yet.<br>Run: python -m ascent.monitoring.live_vs_backtest</div>';
      if (sumEl) sumEl.innerHTML = '<div style="color:#475569;font-size:10px">No data</div>';
      return;
    }

    // ── Chart ──
    const canvas = document.createElement('canvas');
    canvas.style.width  = '100%';
    canvas.style.height = '100%';
    lvbEl.appendChild(canvas);

    function drawLvb() {
      const W = canvas.offsetWidth, H = canvas.offsetHeight;
      if (!W || !H) return;
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,W,H);

      const PAD = {t:20, r:20, b:28, l:52};
      const cW = W - PAD.l - PAD.r;
      const cH = H - PAD.t - PAD.b;

      // collect series
      const btDates = LVB.backtest.dates;
      const btVals  = LVB.backtest.cum_returns;
      const liveDates = LVB.live.dates;
      const liveVals  = LVB.live.cum_returns;

      // merge all values for scale
      const allVals = [...btVals, ...liveVals];
      const minV = Math.min(...allVals, 0);
      const maxV = Math.max(...allVals, 0);
      const range = maxV - minV || 0.01;
      const allDates = [...new Set([...btDates, ...liveDates])].sort();
      const n = allDates.length;
      if (n < 2) return;

      const xOf = d => PAD.l + allDates.indexOf(d) / (n-1) * cW;
      const yOf = v => PAD.t + (1 - (v - minV) / range) * cH;

      // zero line
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([4,4]);
      ctx.beginPath();
      const y0 = yOf(0);
      ctx.moveTo(PAD.l, y0); ctx.lineTo(PAD.l + cW, y0);
      ctx.stroke();
      ctx.setLineDash([]);

      // backtest line (blue)
      if (btDates.length > 1) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        btDates.forEach((d,i) => {
          const x = xOf(d), y = yOf(btVals[i]);
          i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        });
        ctx.stroke();
      }

      // live line (gold)
      if (liveDates.length > 1) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        liveDates.forEach((d,i) => {
          const x = xOf(d), y = yOf(liveVals[i]);
          i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        });
        ctx.stroke();
        // dot at last point
        const lx = xOf(liveDates[liveDates.length-1]);
        const ly = yOf(liveVals[liveVals.length-1]);
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, 2*Math.PI); ctx.fill();
      }

      // y-axis labels
      ctx.fillStyle = '#475569';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      [minV, (minV+maxV)/2, maxV].forEach(v => {
        const y = yOf(v);
        ctx.fillText((v*100).toFixed(1)+'%', PAD.l-4, y+3);
      });

      // legend
      ctx.textAlign = 'left';
      ctx.fillStyle = '#3b82f6'; ctx.fillRect(PAD.l+4, PAD.t+4, 14, 2);
      ctx.fillStyle = '#94a3b8'; ctx.fillText('Backtest', PAD.l+22, PAD.t+8);
      ctx.fillStyle = '#f59e0b'; ctx.fillRect(PAD.l+90, PAD.t+4, 14, 2);
      ctx.fillStyle = '#94a3b8'; ctx.fillText('Live', PAD.l+108, PAD.t+8);
    }

    drawLvb();
    window.addEventListener('resize', drawLvb);

    // ── Summary card ──
    const s = LVB.summary;
    const gapColor = s.gap_bps >= 0 ? '#22c55e' : '#ef4444';
    const gapSign  = s.gap_bps >= 0 ? '+' : '';
    sumEl.innerHTML = `
      <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">
        Live Track Record
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Period</span>
        <span style="font-size:9px;color:#94a3b8">${s.live_start_date || '—'} → ${s.live_end_date || '—'}</span>
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Live Return</span>
        <span style="font-size:9px;color:${s.live_total_return_pct >= 0 ? '#22c55e' : '#ef4444'};font-weight:600">
          ${s.live_total_return_pct >= 0 ? '+' : ''}${s.live_total_return_pct.toFixed(3)}%
        </span>
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Backtest (same range)</span>
        <span style="font-size:9px;color:#3b82f6">${s.backtest_total_return_pct >= 0 ? '+' : ''}${s.backtest_total_return_pct.toFixed(3)}%</span>
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Gap (live − bt)</span>
        <span style="font-size:9px;color:${gapColor};font-weight:600">${gapSign}${s.gap_bps.toFixed(1)} bps</span>
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Live Rolling Sharpe</span>
        <span style="font-size:9px;color:#94a3b8">${s.live_rolling_sharpe_last !== null ? s.live_rolling_sharpe_last.toFixed(2) : '—'}</span>
      </div>
      <div style="margin-bottom:6px;display:flex;justify-content:space-between">
        <span style="font-size:9px;color:#64748b">Live Days Tracked</span>
        <span style="font-size:9px;color:#94a3b8">${s.n_live_days}</span>
      </div>
      <div style="margin-top:12px;font-size:8px;color:#334155">
        Backtest: ${s.n_backtest_days} days (walk-forward OOS)
      </div>`;
  })();

window.onload=()=>{requestAnimationFrame(()=>{requestAnimationFrame(rA);})};window.onresize=rA;
</script>
</body></html>"""

out_dir = ROOT / "dashboard"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "ascent_terminal.html"
out_path.write_text(html_doc, encoding="utf-8")
print("Dashboard built: %s" % out_path)
print("Open: file://%s" % out_path.resolve())
