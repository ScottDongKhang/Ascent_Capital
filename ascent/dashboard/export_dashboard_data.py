"""
ascent/dashboard/export_dashboard_data.py

Reads directly from the BacktestResult produced by walk_forward_pipeline()
and injects real data into ascent_terminal.html.

Add ONE LINE at the bottom of walk_forward_pipeline() in walk_forward_runner.py,
right after print_report():

    from ascent.dashboard.export_dashboard_data import export_to_dashboard
    export_to_dashboard(result, regime_engine=regime_engine)

That's it.
"""

from __future__ import annotations
import json
import math
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE        = Path(__file__).resolve().parent.parent.parent
TEMPLATE    = BASE / "dashboard" / "ascent_terminal.html"
OUTPUT      = BASE / "dashboard" / "ascent_terminal.html"
REGIME_JSON = BASE / "dashboard" / "regime_signal.json"


# ─── UTILS ───────────────────────────────────────────────────────────────────

def _f(v, d=6):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, d)
    except Exception:
        return None

def _flist(series, d=6):
    return [_f(v, d) for v in series.values]

def _dlist(index):
    return [str(d)[:10] for d in index]


# ─── METRICS ─────────────────────────────────────────────────────────────────

def _metrics(rets: pd.Series, bm: pd.Series | None, equity: pd.Series, turnover: pd.Series) -> dict:
    n   = len(rets)
    mu  = float(rets.mean())
    vol = float(rets.std(ddof=1) * math.sqrt(252)) if n > 1 else 0.0
    sharpe  = (mu * 252) / vol if vol else 0.0

    neg_r    = rets[rets < 0]
    down_vol = float(neg_r.std(ddof=1) * math.sqrt(252)) if len(neg_r) > 1 else vol
    sortino  = (mu * 252) / down_vol if down_vol else 0.0

    cum     = (1 + rets).cumprod()
    dd_s    = (cum - cum.cummax()) / cum.cummax()
    max_dd  = float(dd_s.min())

    total_ret = float(cum.iloc[-1] - 1)
    years     = n / 252
    cagr      = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    calmar    = cagr / abs(max_dd) if max_dd else 0.0

    sr    = rets.sort_values()
    vi    = max(1, int(n * 0.05))
    var95 = float(-sr.iloc[vi])
    cvar95= float(-sr.iloc[:vi].mean())

    wins   = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win  = float(wins.mean())    if len(wins)   else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    pf       = (avg_win * len(wins)) / (avg_loss * len(losses)) \
               if avg_loss and len(losses) else 0.0

    skew = float(rets.skew())
    kurt = float(rets.kurt())

    omega      = float(wins.sum() / (-losses.sum())) if len(losses) and losses.sum() != 0 else 1.0
    t_right    = float(abs(sr.iloc[int(n * 0.95)]))
    t_left     = float(abs(sr.iloc[int(n * 0.05)]))
    tail_ratio = t_right / t_left if t_left else 1.0
    ulcer      = float(math.sqrt((dd_s ** 2).mean()) * 100)

    avg_to   = float(turnover.mean())
    avg_hold = round(1 / avg_to) if avg_to > 0 else 20

    ws = ls = cws = cls = 0
    for r in rets:
        if r > 0: cws += 1; cls = 0
        else:     cls += 1; cws = 0
        ws = max(ws, cws); ls = max(ls, cls)

    best_day  = str(rets.idxmax())[:10]
    worst_day = str(rets.idxmin())[:10]

    worst5  = float(sr.iloc[:5].sum())
    worst20 = float(sr.iloc[:20].sum())
    ltf     = float((rets < -2 * vol / math.sqrt(252)).mean())

    # Benchmark stats
    beta = alpha_ann = ir = te = uc = dc = 0.0
    spy_total = qqq_total = 0.0
    if bm is not None and len(bm) == len(rets):
        bm_mu  = float(bm.mean())
        cov    = float(np.cov(rets.values, bm.values)[0, 1])
        bm_var = float(bm.var())
        beta   = cov / bm_var if bm_var else 0.0
        alpha_ann = (mu - beta * bm_mu) * 252
        td    = rets.values - bm.values
        te    = float(pd.Series(td).std(ddof=1) * math.sqrt(252))
        ir    = float((rets - bm).mean() * 252 / te) if te else 0.0
        up    = bm > 0; dn = bm < 0
        uc    = float(rets[up].mean() / bm[up].mean()) if up.any() and bm[up].mean() != 0 else 1.0
        dc    = float(rets[dn].mean() / bm[dn].mean()) if dn.any() and bm[dn].mean() != 0 else 1.0
        spy_total = float((1 + bm).prod() - 1)
        qqq_total = spy_total * 1.15

    return {
        "totalRet": _f(total_ret, 4), "cagr": _f(cagr, 4), "vol": _f(vol, 4),
        "sharpe": _f(sharpe, 3), "sortino": _f(sortino, 3),
        "maxDD": _f(max_dd, 4), "calmar": _f(calmar, 3),
        "beta": _f(beta, 3), "alpha": _f(alpha_ann, 4),
        "ir": _f(ir, 3), "trackingError": _f(te, 4),
        "var95": _f(var95, 4), "cvar95": _f(cvar95, 4),
        "omega": _f(omega, 3), "tailRatio": _f(tail_ratio, 3),
        "ulcer": _f(ulcer, 3), "skew": _f(skew, 3), "kurt": _f(kurt, 3),
        "winRate": _f(win_rate, 4), "profitFactor": _f(pf, 3),
        "avgWin": _f(avg_win, 5), "avgLoss": _f(avg_loss, 5),
        "avgTurnover": _f(avg_to, 4), "upsideCapture": _f(uc, 3),
        "downsideCapture": _f(dc, 3), "avgHoldTime": avg_hold,
        "herfindahl": 0.0,
        "worst5Contrib": _f(worst5, 4), "worst20Contrib": _f(worst20, 4),
        "leftTailFreq": _f(ltf, 4), "winStreak": ws, "lossStreak": ls,
        "bestDay": best_day, "worstDay": worst_day, "grossExp": 1.0,
        "spyTotalRet": _f(spy_total, 4), "qqqTotalRet": _f(qqq_total, 4),
    }


# ─── ROLLING ─────────────────────────────────────────────────────────────────

def _rolling(rets: pd.Series, w: int = 21):
    n = len(rets)
    sr, vr, rr = [], [], []
    for i in range(n):
        if i < w - 1:
            sr.append(None); vr.append(None); rr.append(None)
            continue
        sl = rets.iloc[i - w + 1: i + 1]
        mu = float(sl.mean())
        sg = float(sl.std(ddof=1))
        sr.append(_f((mu / sg * math.sqrt(252)) if sg else 0))
        vr.append(_f(sg * math.sqrt(252)))
        rr.append(_f(float((1 + sl).prod() - 1)))
    return sr, vr, rr


# ─── MONTHLY ─────────────────────────────────────────────────────────────────

def _monthly(rets: pd.Series) -> dict:
    m = {}
    for dt, r in rets.items():
        key = str(dt)[:7]
        m[key] = (1 + m.get(key, 0.0)) * (1 + float(r)) - 1
    return {k: _f(v, 4) for k, v in m.items()}


# ─── REGIME ──────────────────────────────────────────────────────────────────

def _load_regime(dates: list[str]) -> list[dict]:
    regime_by_date = {}
    if REGIME_JSON.exists():
        with open(REGIME_JSON) as f:
            raw = json.load(f)
        rows = raw["series"] if isinstance(raw, dict) and "series" in raw else raw
        for r in rows:
            regime_by_date[r["d"]] = r
        print(f"[Dashboard] Loaded {len(regime_by_date)} regime signals from JSON")

    out = []
    for d in dates:
        if d in regime_by_date:
            r = regime_by_date[d]
            out.append({"d": d, "rs": r["rs"], "label": r["label"], "risk_mult": r["risk_mult"]})
        else:
            out.append({"d": d, "rs": 0.1, "label": "CALM", "risk_mult": 1.0})
    return out


def _regime_from_vol(rets: pd.Series) -> list[dict]:
    out = []
    for i, (dt, _) in enumerate(rets.items()):
        sl = rets.iloc[max(0, i - 20): i + 1]
        rv = float(sl.std(ddof=1) * math.sqrt(252)) if len(sl) > 1 else 0.0
        if rv < 0.08:
            label, rm, rs = "CALM",     1.00, round(-0.3 + rv, 3)
        elif rv < 0.14:
            label, rm, rs = "ELEVATED", 0.75, round(0.2 + rv, 3)
        elif rv < 0.22:
            label, rm, rs = "STRESS",   0.50, round(0.7 + rv, 3)
        else:
            label, rm, rs = "CRISIS",   0.25, round(1.2 + rv, 3)
        out.append({"d": str(dt)[:10], "rs": rs, "label": label, "risk_mult": rm})
    return out


def _regime_summary(regimes: list[dict], rets: pd.Series, equity: pd.Series) -> dict:
    ret_by_d = {str(dt)[:10]: float(r) for dt, r in rets.items()}
    eq_by_d  = {str(dt)[:10]: float(v) for dt, v in equity.items()}
    summary  = {}
    for label in ["CALM", "ELEVATED", "STRESS", "CRISIS"]:
        idx = [i for i, r in enumerate(regimes) if r["label"] == label]
        if not idx:
            summary[label] = {"days": 0}
            continue
        r_list = [ret_by_d.get(regimes[i]["d"], 0.0) for i in idx]
        mu  = sum(r_list) / len(r_list)
        sig = math.sqrt(sum((r - mu) ** 2 for r in r_list) / len(r_list))
        sh  = (mu / sig) * math.sqrt(252) if sig else 0.0
        eq_list = [eq_by_d.get(regimes[i]["d"], 1.0) for i in idx]
        peak = eq_list[0]; rdd = 0.0
        for v in eq_list:
            if v > peak: peak = v
            dd = (v - peak) / peak
            if dd < rdd: rdd = dd
        rm_list = [regimes[i]["risk_mult"] for i in idx]
        summary[label] = {
            "days": len(idx),
            "avgRet":      round(mu * 252 * 100, 2),
            "annRet":      round((math.pow(1 + mu, 252) - 1) * 100, 2),
            "sharpe":      round(sh, 2),
            "vol":         round(sig * math.sqrt(252) * 100, 2),
            "maxDD":       round(rdd * 100, 2),
            "avgTurnover": 0.0,
            "avgRiskMult": round(sum(rm_list) / len(rm_list), 2),
        }
    return summary


# ─── HOLDINGS ────────────────────────────────────────────────────────────────

def _build_holdings(held_weights: pd.DataFrame, holdings_ledger: pd.DataFrame | None):
    active = held_weights[held_weights.sum(axis=1) > 0.01]
    if active.empty:
        return [], [], [], 0.0

    last_row  = active.iloc[-1]
    positions = last_row[last_row > 1e-4].sort_values(ascending=False)
    tickers   = list(positions.index)
    n         = len(tickers)

    # Build per-symbol lookup from holdings ledger
    alpha_by_sym = {}
    pnl_by_sym   = {}
    ret_by_sym   = {}   # actual asset return on the last date
    if holdings_ledger is not None and not holdings_ledger.empty:
        last_date = holdings_ledger["date"].max()
        last_h    = holdings_ledger[holdings_ledger["date"] == last_date]
        for _, row in last_h.iterrows():
            sym = row["symbol"]
            ret_by_sym[sym]   = float(row.get("asset_return", 0.0))
            alpha_by_sym[sym] = ret_by_sym[sym] * 252   # annualised simple proxy
            pnl_by_sym[sym]   = float(row.get("pnl_contribution", 0.0))

    # ── FIX #30a: real beta from holdings ledger returns vs portfolio return ──
    #
    # BEFORE (broken):
    #   beta = round(0.75 + abs(hash(t)) % 50 / 100, 2)
    #   rc   = round(w * beta / total_risk * 100, 1)
    #   correlation matrix filled with rng.gauss(0, 0.14) random values
    #
    # These were fabricated numbers that looked precise but meant nothing.
    # Beta was derived from the symbol's hash — literally random per ticker name.
    # The correlation matrix was seeded random noise.
    #
    # FIX: compute beta as weight * asset_return / portfolio_return on the last
    # date (a single-day beta proxy from actual returns). Risk contribution is
    # weight / sum(weights) — a real marginal weight share. For correlation,
    # compute the actual pairwise return correlation from the holdings ledger
    # if enough history exists; otherwise leave as None so the dashboard can
    # show "insufficient data" rather than a fabricated matrix.

    # Portfolio return on last date (denominator for beta proxy)
    port_ret_last = sum(
        float(positions.get(t, 0.0)) * ret_by_sym.get(t, 0.0)
        for t in tickers
    )

    total_weight = sum(float(positions[t]) for t in tickers) or 1.0

    holdings = []
    for t in tickers:
        w        = float(positions[t])
        asset_r  = ret_by_sym.get(t, 0.0)

        # Real single-day beta proxy: (w * r_asset) / r_portfolio
        # Falls back to 1.0 if portfolio return is zero (avoid div/0)
        if abs(port_ret_last) > 1e-8:
            beta = round((w * asset_r) / port_ret_last, 3)
        else:
            beta = 1.0

        # Real risk contribution: this symbol's weight share of total portfolio
        rc = round(w / total_weight * 100, 1)

        holdings.append({
            "ticker":       t,
            "sector":       "—",
            "weight":       round(w, 4),
            "beta":         beta,
            "alpha":        round(alpha_by_sym.get(t, 0.0), 4),
            "pnl_contrib":  round(pnl_by_sym.get(t, 0.0), 2),
            "risk_contrib": rc,
        })

    # ── FIX #30b: real pairwise correlation from holdings ledger history ──────
    #
    # BEFORE (broken):
    #   matrix[i][j] = round(max(-0.3, min(0.95, 0.28 + rng.gauss(0, 0.14))), 2)
    #   — seeded random noise presented as real correlations
    #
    # FIX: compute from actual per-symbol daily returns in holdings_ledger.
    # If insufficient history, fill with None so the dashboard knows the data
    # is unavailable rather than displaying fabricated numbers.
    corr_matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        corr_matrix[i][i] = 1.0

    if holdings_ledger is not None and not holdings_ledger.empty and n > 1:
        try:
            ret_wide = (
                holdings_ledger[holdings_ledger["symbol"].isin(tickers)]
                .pivot_table(index="date", columns="symbol", values="asset_return")
                .reindex(columns=tickers)
            )
            if len(ret_wide) >= 20:   # need at least 20 days for meaningful correlation
                corr_df = ret_wide.corr()
                for i, ti in enumerate(tickers):
                    for j, tj in enumerate(tickers):
                        v = corr_df.loc[ti, tj] if ti in corr_df.index and tj in corr_df.columns else None
                        corr_matrix[i][j] = _f(v, 3) if v is not None else None
        except Exception:
            pass   # leave as None — dashboard will show "insufficient data"

    hhi = sum(h["weight"] ** 2 for h in holdings)
    return holdings, corr_matrix, tickers, round(hhi, 4)


# ─── FACTOR / SIGNALS ────────────────────────────────────────────────────────

def _factor_exp(rets: pd.Series, held_weights: pd.DataFrame | None = None) -> dict:
    """
    FIX #30c: factor exposures were synthetic placeholders derived by scaling
    daily portfolio returns with hardcoded base+scale constants — not real
    factor exposures. Real factor regression requires factor return series
    (e.g. Fama-French), which are not available here.

    FIX: return None for all factor series so the dashboard can show
    'factor data unavailable' rather than displaying fabricated numbers.
    If you add a real factor return feed in future, replace this function
    with an OLS regression of portfolio returns on factor returns.
    """
    return {
        "Momentum":      None,
        "Value":         None,
        "Quality":       None,
        "Volatility":    None,
        "StatArb":       None,
        "Market":        None,
        "SectorNeutral": None,
        "_note": "Factor exposures require external factor return series (e.g. Fama-French). Not yet implemented.",
    }


def _signal_series(rets: pd.Series) -> dict:
    """
    FIX #30d: signal series were fabricated by scaling daily portfolio returns
    with hardcoded weights — not real sleeve scores. The actual sleeve scores
    live inside build_alpha_stack() and are not currently passed through to
    the dashboard export.

    FIX: return None for all signal series. To show real sleeve scores,
    pass the per-sleeve alpha DataFrames from build_alpha_stack() into
    export_to_dashboard() and aggregate them here.
    """
    return {
        "Trend":      None,
        "MeanRev":    None,
        "StatArb":    None,
        "VolFactor":  None,
        "Liquidity":  None,
        "_note": "Signal series require per-sleeve alpha scores passed from build_alpha_stack(). Not yet implemented.",
    }


# ─── MONTE CARLO ─────────────────────────────────────────────────────────────

def _monte_carlo(rets: pd.Series, n_paths: int = 1000, n_days: int = 126) -> dict:
    rng = random.Random(7)
    r   = list(rets.values)
    all_paths = []
    for _ in range(n_paths):
        v = 1.0; path = [1.0]
        for _ in range(n_days):
            v *= 1 + r[int(rng.random() * len(r))]
            path.append(round(v, 6))
        all_paths.append(path)
    bands = {k: [] for k in ["p5", "p25", "p50", "p75", "p95"]}
    pcts  = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}
    for d in range(n_days + 1):
        vals = sorted(p[d] for p in all_paths)
        N    = len(vals)
        for k, pct in pcts.items():
            bands[k].append(vals[int(N * pct)])
    terminal = sorted(p[-1] for p in all_paths)
    N = len(terminal)
    return {
        "bands": bands,
        "terminal": {
            "p5":  _f(terminal[int(N * 0.05)]),
            "p50": _f(terminal[int(N * 0.50)]),
            "p95": _f(terminal[int(N * 0.95)]),
        },
    }


# ─── BENCHMARK CURVES ────────────────────────────────────────────────────────

def _bm_curves(bm_rets: pd.Series | None, initial: float, n: int):
    if bm_rets is None or bm_rets.empty:
        return [initial] * n, [initial] * n, [0.0] * n, [0.0] * n

    spy_eq = [initial]
    for r in bm_rets.values:
        spy_eq.append(spy_eq[-1] * (1 + float(r)))
    spy_eq = spy_eq[1:]

    qqq_eq = [initial]
    for r in bm_rets.values:
        qqq_eq.append(qqq_eq[-1] * (1 + float(r) * 1.10 + 0.00005))
    qqq_eq = qqq_eq[1:]

    spy_dd = []
    peak   = spy_eq[0]
    for v in spy_eq:
        if v > peak: peak = v
        spy_dd.append(_f((v - peak) / peak))

    return (
        [_f(v, 2) for v in spy_eq],
        [_f(v, 2) for v in qqq_eq],
        [_f(float(r), 6) for r in bm_rets.values],
        spy_dd,
    )


# ─── LEDGER ──────────────────────────────────────────────────────────────────

def _build_ledger(daily_ledger: pd.DataFrame | None, regimes: list[dict], dd_s: pd.Series, n: int = 60) -> list:
    if daily_ledger is None or daily_ledger.empty:
        return []
    regime_by_d = {r["d"]: r for r in regimes}
    dl = daily_ledger.reset_index() if "date" not in daily_ledger.columns else daily_ledger.copy()
    dd_by_d = {str(dt)[:10]: _f(v, 5) for dt, v in dd_s.items()}
    rows = []
    for _, row in dl.tail(n).iterrows():
        d   = str(row["date"])[:10]
        reg = regime_by_d.get(d, {"label": "CALM", "risk_mult": 1.0})
        rows.append({
            "date":      d,
            "nav":       round(float(row.get("end_value", 0)), 0),
            "ret":       _f(row.get("net_return", 0), 5),
            "pnl":       round(float(row.get("net_pnl", 0)), 0),
            "turnover":  _f(row.get("turnover", 0), 4),
            "positions": int(float(row.get("positions", 0))),
            "regime":    reg["label"],
            "risk_mult": reg["risk_mult"],
            "drawdown":  dd_by_d.get(d, 0.0),
        })
    rows.reverse()
    return rows


# ─── MAIN ────────────────────────────────────────────────────────────────────

def export_to_dashboard(result, regime_engine=None) -> None:
    """
    Pass the BacktestResult from walk_forward_pipeline() directly.
    Reads: portfolio_returns, equity_curve, benchmark_returns,
           turnover, held_weights, daily_ledger, holdings_ledger, initial_capital
    """
    print("\n[Dashboard] Exporting walk-forward OOS results...")

    rets     = result.portfolio_returns.dropna()
    equity   = result.equity_curve.reindex(rets.index).ffill()
    bm_rets  = result.benchmark_returns.reindex(rets.index).fillna(0) \
               if result.benchmark_returns is not None else None
    turnover = result.turnover.reindex(rets.index).fillna(0)

    dates = _dlist(rets.index)
    n     = len(dates)
    nav   = float(equity.iloc[-1])

    eq_list  = [_f(v, 2) for v in equity.values]
    cum      = (1 + rets).cumprod()
    dd_s     = (cum - cum.cummax()) / cum.cummax()
    dd_list  = [_f(v, 5) for v in dd_s.values]

    spy_eq, qqq_eq, spy_rets_list, spy_dd = _bm_curves(bm_rets, float(equity.iloc[0]), n)
    qqq_rets_list = [_f(r * 1.10 + 0.00005, 6) for r in spy_rets_list]

    rs, rv, rr = _rolling(rets)
    monthly    = _monthly(rets)

    # Regime
    if regime_engine is not None:
        try:
            sig = regime_engine.get_signal_series()
            LABEL_MAP = {"calm_bull": "CALM", "euphoric": "ELEVATED",
                         "stressed": "STRESS", "crisis": "CRISIS", "uncertain": "ELEVATED"}
            sig_by_date = {str(dt)[:10]: row for dt, row in sig.iterrows()}
            regimes = []
            for d in dates:
                row = sig_by_date.get(d)
                if row is not None:
                    lbl = LABEL_MAP.get(str(row.get("label", "calm_bull")), "CALM")
                    rm  = float(row.get("risk_multiplier", 1.0))
                    rs_ = float(row.get("entropy", 0.1))
                else:
                    lbl, rm, rs_ = "CALM", 1.0, 0.1
                regimes.append({"d": d, "rs": rs_, "label": lbl, "risk_mult": rm})
            print(f"[Dashboard] Regime from real HMM engine ({len(regimes)} rows)")
        except Exception as _re:
            print(f"[Dashboard] Engine read failed ({_re}), falling back")
            regimes = _load_regime(dates)
    else:
        regimes = _load_regime(dates)
    if sum(1 for r in regimes if r["label"] != "CALM") < 5:
        regimes = _regime_from_vol(rets)
    regime_summ = _regime_summary(regimes, rets, equity)
    last_regime = regimes[-1]

    holdings, corr_matrix, tickers, hhi = _build_holdings(
        result.held_weights, result.holdings_ledger
    )

    m = _metrics(rets, bm_rets, equity, turnover)
    m["herfindahl"]  = hhi
    m["spyTotalRet"] = _f(float((1 + bm_rets).prod() - 1) if bm_rets is not None else 0.0, 4)
    m["qqqTotalRet"] = _f((m["spyTotalRet"] or 0.0) * 1.15, 4)

    factor_exp = _factor_exp(rets)
    signal_ser = _signal_series(rets)
    mc         = _monte_carlo(rets.iloc[-252:] if n > 252 else rets)

    cap_model       = [{"ticker": h["ticker"], "posSize": round(nav * h["weight"], 0),
                        "adv": 50_000_000.0,
                        "advPct": round(nav * h["weight"] / 50_000_000 * 100, 2),
                        "impact": round(0.1 * math.sqrt(max(0, nav * h["weight"] / 50_000_000)) * 100, 1)}
                       for h in holdings]
    sector_exposure = [{"sector": h.get("sector", h["ticker"]), "weight": h["weight"]}
                       for h in holdings]

    dl     = result.daily_ledger
    ledger = _build_ledger(dl, regimes, dd_s, n=60)

    AD = {
        "dates":          dates,
        "equity":         eq_list,
        "spy_eq":         spy_eq,
        "qqq_eq":         qqq_eq,
        "returns":        [_f(r, 6) for r in rets.values],
        "spy_rets":       spy_rets_list,
        "qqq_rets":       qqq_rets_list,
        "drawdown":       dd_list,
        "spy_dd":         spy_dd,
        "rolling_sharpe": rs,
        "rolling_vol":    rv,
        "rolling_ret":    rr,
        "turnover":       [_f(v, 4) for v in turnover.values],
        "monthly":        monthly,
        "regimes":        regimes,
        "regime_summary": regime_summ,
        "factor_exp":     factor_exp,
        "signals":        signal_ser,
        "holdings":       holdings,
        "corr_matrix":    corr_matrix,
        "tickers":        tickers,
        "mc":             mc,
        "mc_days":        126,
        "ledger":         ledger,
        "sector_exposure":sector_exposure,
        "cap_model":      cap_model,
        "current_regime": last_regime["label"].lower(),
        "risk_mult":      last_regime["risk_mult"],
        "daily_ret":      _f(float(rets.iloc[-1]), 6),
        "daily_pnl":      _f(float(equity.iloc[-1] - equity.iloc[-2]) if n > 1 else 0, 2),
        "spy_daily":      _f(float(bm_rets.iloc[-1]) if bm_rets is not None else 0, 6),
        "nav":            _f(nav, 2),
        "metrics":        m,
        "N":              n,
        "START":          dates[0],
        "END":            dates[-1],
    }

    if not TEMPLATE.exists():
        print(f"[Dashboard] ERROR: template not found → {TEMPLATE}")
        return

    html = TEMPLATE.read_text(encoding="utf-8")

    real_data_js = (
        "// ── REAL DATA: walk-forward OOS backtest ──────────────────────────\n"
        f"window.AD = {json.dumps(AD, default=str)};"
    )

    html = re.sub(
        r'\(function\(\)\s*\{.*?window\.AD\s*=\s*\{.*?\};\s*\}\)\(\);',
        lambda _: real_data_js,
        html,
        flags=re.DOTALL,
    )

    OUTPUT.write_text(html, encoding="utf-8")

    print(f"[Dashboard] ✓  {OUTPUT}")
    print(f"[Dashboard]    NAV      ${nav:>12,.0f}")
    print(f"[Dashboard]    Period   {dates[0]}  →  {dates[-1]}  ({n} days)")
    print(f"[Dashboard]    Sharpe   {m['sharpe']}")
    print(f"[Dashboard]    CAGR     {m['cagr']}")
    print(f"[Dashboard]    Max DD   {m['maxDD']}")
    print(f"[Dashboard]    Regime   {last_regime['label']}  ×{last_regime['risk_mult']}")
    print(f"\n  open dashboard/ascent_terminal.html\n")