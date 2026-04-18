"""
run_20in20.py  —  Ascent Intel for 20in20
==========================================
Usage:
    python run_20in20.py --asof 2026-03-21
    python run_20in20.py --asof 2026-03-21 --open
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from config_20in20           import load_config_20in20
from watchlists_20in20       import build_watchlists_20in20
from scenario_library_20in20 import build_scenario_library_20in20, run_scenario_brief

try:
    from ascent.regime.posture import compute_posture_from_regime
except ImportError:
    from posture import compute_posture_from_regime

try:
    from ascent.reporting.market_memo import build_market_memo_payload, write_market_memo
except ImportError:
    from market_memo import build_market_memo_payload, write_market_memo

build_public_comps_tables = None


# ════════════════════════════════════════════════════════════════════════════
# Regime loader
# ════════════════════════════════════════════════════════════════════════════

def _load_regime_from_disk(asof: str) -> tuple:
    """Returns (label, probs_dict, days_in_regime, drivers)."""
    csv_path = Path("dashboard/regime_labels.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.sort_values("date")
            label_col = next((c for c in ["label", "regime"] if c in df.columns), None)
            if label_col:
                if "date" in df.columns:
                    df_asof=df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")<=asof]
                    if df_asof.empty: df_asof=df
                else:
                    df_asof=df
                latest    = df_asof.iloc[-1]
                label     = str(latest[label_col])
                prob_cols = [c for c in df_asof.columns if c.startswith("prob_")]
                probs     = {c.replace("prob_", ""): float(latest[c]) for c in prob_cols} if prob_cols else {label: 1.0}
                same      = (df_asof[label_col] == label)
                days      = int(same[::-1].cumprod().sum())
                drivers   = _infer_drivers(df_asof, label_col)
                print(f"[20in20] Regime from regime_labels.csv: {label}")
                return label, probs, days, drivers
        except Exception as e:
            print(f"[20in20] Warning: regime_labels.csv parse failed — {e}")

    json_path = Path("dashboard/regime_signal.json")
    if json_path.exists():
        try:
            data  = json.loads(json_path.read_text())
            label = str(data.get("label") or data.get("regime") or data.get("current_regime") or "neutral")
            probs = data.get("probs", {label: 1.0})
            days  = int(data.get("days_in_regime", data.get("dwell", 0)))
            print(f"[20in20] Regime from regime_signal.json: {label}")
            return label, probs, days, []
        except Exception as e:
            print(f"[20in20] Warning: regime_signal.json parse failed — {e}")

    print("[20in20] Warning: no regime file found — defaulting to neutral")
    return "neutral", {"neutral": 1.0}, 0, []


def _infer_drivers(df: pd.DataFrame, label_col: str) -> list:
    """Generate plain-English regime driver bullets from label history."""
    drivers = []
    if len(df) < 21:
        return drivers

    recent = df.tail(21)
    label  = df.iloc[-1][label_col]

    transitions = (recent[label_col] != recent[label_col].shift(1)).sum()
    if transitions >= 3:
        drivers.append("Frequent regime transitions over the past month — elevated uncertainty")
    elif transitions == 0:
        drivers.append(f"Regime stable for 21+ days — persistent {label.replace('_', ' ')} environment")

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        latest       = df.iloc[-1]
        dominant_prob = max(float(latest[c]) for c in prob_cols)
        if dominant_prob > 0.80:
            drivers.append(f"High probability mass on {label.replace('_', ' ')} ({dominant_prob:.0%})")
        elif dominant_prob < 0.55:
            drivers.append("Probability split across states — low regime conviction")

    generic = {
        "stressed":  ["Volatility elevated vs baseline", "Breadth weakening outside leaders"],
        "crisis":    ["Broad market dislocations active", "Risk-off positioning dominant"],
        "calm_bull": ["Trend intact across most sectors", "Low dispersion environment"],
        "euphoric":  ["Late-cycle momentum signals present", "Narrow leadership concentration risk"],
        "neutral":   ["Mixed cross-sectional signals", "No dominant regime conviction"],
    }
    drivers += generic.get(label, [])
    return drivers[:3]


# ════════════════════════════════════════════════════════════════════════════
# Price loader
# ════════════════════════════════════════════════════════════════════════════

def _load_prices(tickers: list, lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("[20in20] yfinance not installed")
        return pd.DataFrame()

    import datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days + 30)

    print(f"[20in20] Fetching prices for {len(tickers)} tickers …")
    try:
        raw = yf.download(tickers, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices.index = pd.to_datetime(prices.index).normalize()
        prices = prices.dropna(how="all")
        print(f"[20in20] Coverage: {prices.notna().mean().mean():.0%} across {len(prices.columns)} tickers")
        return prices
    except Exception as e:
        print(f"[20in20] Price fetch failed: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# Theme + RV builders
# ════════════════════════════════════════════════════════════════════════════

def _build_themes_table(prices: pd.DataFrame, themes: dict, asof: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rows = []
    for theme, tickers in themes.items():
        cols = [t for t in tickers if t in prices.columns]
        if not cols:
            continue
        sub = prices[cols].dropna(how="all")
        if len(sub) < 63:
            continue
        ret_3m = (sub.iloc[-1] / sub.iloc[-63] - 1).dropna()
        if ret_3m.empty:
            continue
        rows.append({"theme": theme, "score": round(float(ret_3m.mean()), 4), "n_tickers": len(cols)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _build_rv_table(prices: pd.DataFrame, watchlist_table: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rows = []
    for _, meta in watchlist_table.iterrows():
        ticker = meta["ticker"]
        if ticker not in prices.columns:
            continue
        sub = prices[ticker].dropna()
        if len(sub) < 6:
            continue
        ret_5d = float(sub.iloc[-1] / sub.iloc[-6] - 1)
        rows.append({"ticker": ticker, "theme": meta["theme"], "ret_5d": ret_5d})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    def _zs(g):
        mu  = g["ret_5d"].mean()
        std = g["ret_5d"].std()
        g   = g.copy()
        g["rv_z"] = (g["ret_5d"] - mu) / std if std > 1e-9 else 0.0
        return g

    df = df.groupby("theme", group_keys=False).apply(_zs, include_groups=False)
    df["rv_z"] = df["rv_z"].round(3)
    return df.sort_values("rv_z", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# ASCII dashboard
# ════════════════════════════════════════════════════════════════════════════

def _print_dashboard(regime, themes_table, rv_table, scenario_table, asof):
    W = 62
    print(f"\n┌{'─'*W}┐")
    print(f"│  Ascent Intel (20in20)  │  as-of: {asof:<22}│")
    print(f"├{'─'*W}┤")
    rl = regime.regime_label.replace("_", " ").upper()
    p  = regime.posture.upper()
    print(f"│  Regime: {rl:<14}  Conf: {regime.confidence:.2f}  Risk: {regime.risk_multiplier:.2f}  Days: {regime.days_in_regime:<4}│")
    print(f"│  Posture: {p:<52}│")
    print(f"├{'─'*W}┤")

    leaders  = themes_table[themes_table["score"] >= 0].head(3) if not themes_table.empty else pd.DataFrame()
    laggards = themes_table[themes_table["score"] <  0].tail(3) if not themes_table.empty else pd.DataFrame()
    print(f"│  {'Theme Leaders':<28}  {'Theme Laggards':<28}│")
    for i in range(3):
        l = f"{leaders.iloc[i]['theme']} {leaders.iloc[i]['score']:+.2f}"   if i < len(leaders)  else ""
        r = f"{laggards.iloc[i]['theme']} {laggards.iloc[i]['score']:+.2f}" if i < len(laggards) else ""
        print(f"│  {l:<28}  {r:<28}│")

    print(f"├{'─'*W}┤")
    ext = rv_table[rv_table["rv_z"] >  0.5].head(3) if not rv_table.empty else pd.DataFrame()
    dep = rv_table[rv_table["rv_z"] < -0.5].tail(3) if not rv_table.empty else pd.DataFrame()
    print(f"│  {'RV: Extended':<28}  {'RV: Depressed':<28}│")
    for i in range(3):
        e = f"{ext.iloc[i]['ticker']} z={ext.iloc[i]['rv_z']:+.1f}" if i < len(ext) else ""
        d = f"{dep.iloc[i]['ticker']} z={dep.iloc[i]['rv_z']:+.1f}" if i < len(dep) else ""
        print(f"│  {e:<28}  {d:<28}│")

    if scenario_table is not None and not scenario_table.empty:
        print(f"├{'─'*W}┤")
        print(f"│  Scenario Watch{' '*46}│")
        for _, row in scenario_table.head(3).iterrows():
            line = f"{row['scenario']}  pnl={row['pnl_est']:+.1%}  [{row['severity']}]"
            print(f"│  {line:<{W-2}}│")

    print(f"├{'─'*W}┤")
    print(f"│  {regime.notes[:W-2]:<{W-2}}│")
    print(f"└{'─'*W}┘\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Ascent Intel for 20in20")
    parser.add_argument("--asof",        required=True)
    parser.add_argument("--outputs_dir", default="outputs/20in20")
    parser.add_argument("--lookback",    type=int, default=252)
    parser.add_argument("--open",        action="store_true", help="Open HTML in browser after generating")
    args = parser.parse_args()

    cfg = load_config_20in20(asof=args.asof, outputs_dir=args.outputs_dir, lookback_days=args.lookback)
    cfg.ensure_dirs()
    print(f"\n{'='*60}")
    print(f"  Ascent Intel for 20in20  |  as-of {cfg.asof}")
    print(f"{'='*60}\n")

    # 1. Watchlists
    watchlists = build_watchlists_20in20()
    print(f"[20in20] Watchlist: {len(watchlists.table)} tickers across {len(watchlists.themes)} themes")

    # 2. Regime
    label, probs, days_in_regime, drivers = _load_regime_from_disk(cfg.asof)
    regime = compute_posture_from_regime(
        asof=cfg.asof,
        regime_label=label,
        probs=probs,
        days_in_regime=days_in_regime,
        min_confidence=cfg.posture_min_conf,
    )
    regime_dict           = regime.to_dict()
    regime_dict["drivers"] = drivers
    regime_path = cfg.outputs_dir / f"regime_summary_{cfg.asof}.json"
    regime_path.write_text(json.dumps(regime_dict, indent=2), encoding="utf-8")
    print(f"[20in20] Regime: {regime.regime_label}  posture={regime.posture}  "
          f"conf={regime.confidence:.2f}  mult={regime.risk_multiplier}  days={regime.days_in_regime}")
    print(f"         → {regime_path}")

    # 3. Prices
    prices = _load_prices(watchlists.table["ticker"].tolist(), cfg.lookback_days)

    # 4. Themes
    themes_table = _build_themes_table(prices, watchlists.themes, cfg.asof)
    themes_path  = cfg.outputs_dir / "tables" / f"themes_{cfg.asof}.csv"
    themes_table.to_csv(themes_path, index=False)
    print(f"[20in20] Themes: {len(themes_table)} rows → {themes_path}")

    # 5. RV
    rv_table = _build_rv_table(prices, watchlists.table)
    rv_path  = cfg.outputs_dir / "tables" / f"relative_value_{cfg.asof}.csv"
    rv_table.to_csv(rv_path, index=False)
    print(f"[20in20] Relative value: {len(rv_table)} rows → {rv_path}")

    # 6. Scenarios
    scenario_table = None
    if cfg.enable_scenarios:
        scenario_lib   = build_scenario_library_20in20()
        scenario_table = run_scenario_brief(prices, watchlists, scenario_lib, cfg)
        sc_path = cfg.outputs_dir / "tables" / f"scenarios_{cfg.asof}.csv"
        scenario_table.to_csv(sc_path, index=False)
        print(f"[20in20] Scenarios: {len(scenario_table)} rows → {sc_path}")

    # 7. Public comps (off by default, enable in config)
    comps_tables = None
    if cfg.enable_public_comps:
        comps_tables = build_public_comps_tables(prices, watchlists, cfg.asof)
        for name, df in comps_tables.items():
            p = cfg.outputs_dir / "tables" / f"comps_{name}_{cfg.asof}.csv"
            df.to_csv(p, index=False)
            print(f"[20in20] Comps [{name}]: {len(df)} rows → {p}")

    # 8. Memo
    payload = build_market_memo_payload(
        config=cfg,
        regime=regime,
        themes_table=themes_table,
        relative_value_table=rv_table,
        comps_tables=comps_tables,
        scenario_table=scenario_table,
    )
    payload["regime"]["drivers"] = drivers
    stem  = f"market_memo_{cfg.asof}"
    paths = write_market_memo(payload, str(cfg.outputs_dir / "memos"), stem)
    print(f"[20in20] Memo JSON → {paths['json']}")
    print(f"[20in20] Memo MD   → {paths['md']}")

    # 9. ASCII
    _print_dashboard(regime, themes_table, rv_table, scenario_table, cfg.asof)

    # 10. HTML
    try:
        from dashboard_20in20 import build_html
        html      = build_html(cfg.asof, cfg.outputs_dir)
        html_path = cfg.outputs_dir / f"intel_{cfg.asof}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"[20in20] HTML → {html_path.resolve()}")
        if args.open:
            import subprocess
            subprocess.run(["open", str(html_path)])
    except Exception as e:
        print(f"[20in20] HTML skipped: {e}")

    print(f"\n{'='*60}")
    print(f"  Done. All outputs in: {cfg.outputs_dir.resolve()}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
