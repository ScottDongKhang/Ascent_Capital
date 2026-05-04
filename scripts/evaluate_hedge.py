"""
scripts/evaluate_hedge.py

Historical hedge overlay evaluation.

Reads:
  - dashboard/regime_labels.csv  (date, label, confidence columns)
  - ascent_daily_ledger.csv      (date, portfolio_value columns)

Fetches:
  - VIXY prices from Yahoo Finance (same date window)

Run: .venv/bin/python scripts/evaluate_hedge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from ascent.portfolio.hedge_overlay import compute_hedge_weight
from ascent.regime.types import RegimeLabel


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def _annualised_sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def main():
    # ── Load regime labels ──────────────────────────────────────────────────
    regime_path = Path("dashboard/regime_labels.csv")
    if not regime_path.exists():
        print(f"ERROR: {regime_path} not found. Run run_all_agents.py at least once first.")
        sys.exit(1)

    regime_df = pd.read_csv(regime_path, parse_dates=["date"])
    regime_df = regime_df.set_index("date").sort_index()

    if "label" not in regime_df.columns or "confidence" not in regime_df.columns:
        print(f"ERROR: regime_labels.csv must have 'label' and 'confidence' columns. "
              f"Found: {list(regime_df.columns)}")
        sys.exit(1)

    # ── Load portfolio ledger ────────────────────────────────────────────────
    ledger_path = Path("ascent_daily_ledger.csv")
    if not ledger_path.exists():
        print(f"ERROR: {ledger_path} not found.")
        sys.exit(1)

    ledger = pd.read_csv(ledger_path, parse_dates=["date"])
    ledger = ledger.set_index("date").sort_index()

    if "end_value" not in ledger.columns:
        print(f"Columns in ledger: {list(ledger.columns)}")
        print("ERROR: ledger must have 'end_value' column.")
        sys.exit(1)

    port_returns = ledger["end_value"].pct_change().dropna()

    # ── Fetch VIXY prices ────────────────────────────────────────────────────
    start = str(port_returns.index[0].date())
    end   = str(port_returns.index[-1].date())
    print(f"Fetching VIXY prices {start} → {end}...")

    raw = yf.download("VIXY", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print("ERROR: VIXY download failed.")
        sys.exit(1)

    vixy_close = raw["Close"]
    if isinstance(vixy_close, pd.DataFrame):
        vixy_close = vixy_close.iloc[:, 0]
    vixy_close.index = pd.to_datetime(vixy_close.index).tz_localize(None)
    vixy_returns = vixy_close.pct_change().dropna()

    # ── Align all series on common dates ────────────────────────────────────
    common = port_returns.index.intersection(vixy_returns.index).intersection(regime_df.index)
    if len(common) < 5:
        print(f"Only {len(common)} common dates — printing available data and exiting.")
        print(f"  Ledger dates: {len(port_returns)}, first={port_returns.index[0].date()}, last={port_returns.index[-1].date()}")
        print(f"  VIXY dates:   {len(vixy_returns)}")
        print(f"  Regime dates: {len(regime_df)}")
        sys.exit(0)

    port_ret   = port_returns.loc[common]
    vixy_ret   = vixy_returns.loc[common]
    regime_ser = regime_df.loc[common]

    # ── Compute per-day hedge weight ────────────────────────────────────────
    hedge_weights = pd.Series(0.0, index=common)
    for dt in common:
        label_str  = regime_ser.loc[dt, "label"]
        confidence = float(regime_ser.loc[dt, "confidence"])
        try:
            label = RegimeLabel.from_str(label_str)
        except Exception:
            label = RegimeLabel.UNCERTAIN
        hedge_weights[dt] = compute_hedge_weight(label, confidence)

    # ── Hedged portfolio returns ─────────────────────────────────────────────
    hedged_ret = (1 - hedge_weights) * port_ret + hedge_weights * vixy_ret

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HEDGE OVERLAY HISTORICAL EVALUATION")
    print("=" * 60)
    print(f"  Date window : {common[0].date()} → {common[-1].date()} ({len(common)} days)")
    print(f"  Regime coverage: {regime_ser['label'].value_counts().to_dict()}")
    print()

    mdd_base   = _max_drawdown(port_ret)
    mdd_hedged = _max_drawdown(hedged_ret)
    mdd_improvement = (mdd_hedged - mdd_base) / abs(mdd_base) if mdd_base != 0 else 0

    sharpe_base   = _annualised_sharpe(port_ret)
    sharpe_hedged = _annualised_sharpe(hedged_ret)

    cagr_base   = float((1 + port_ret).prod() ** (252 / len(port_ret)) - 1)
    cagr_hedged = float((1 + hedged_ret).prod() ** (252 / len(hedged_ret)) - 1)

    print("  ┌─────────────────────────┬──────────────┬──────────────┐")
    print("  │ Metric                  │  No Hedge    │  With Hedge  │")
    print("  ├─────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Max Drawdown            │  {mdd_base:>10.2%}  │  {mdd_hedged:>10.2%}  │")
    print(f"  │ Annualised Sharpe       │  {sharpe_base:>10.3f}  │  {sharpe_hedged:>10.3f}  │")
    print(f"  │ CAGR                    │  {cagr_base:>10.2%}  │  {cagr_hedged:>10.2%}  │")
    print(f"  │ Drawdown improvement    │              │  {mdd_improvement:>+10.1%}  │")
    print("  └─────────────────────────┴──────────────┴──────────────┘")

    # Calm bull drag
    calm_mask = regime_ser["label"] == "calm_bull"
    if calm_mask.sum() > 3:
        calm_base   = _annualised_sharpe(port_ret[calm_mask])
        calm_hedged = _annualised_sharpe(hedged_ret[calm_mask])
        print(f"\n  Calm bull periods ({calm_mask.sum()} days):")
        print(f"    Sharpe no hedge:   {calm_base:.3f}")
        print(f"    Sharpe with hedge: {calm_hedged:.3f}  (drag = {calm_hedged - calm_base:+.3f})")

    # Stressed/crisis lift
    risk_mask = regime_ser["label"].isin(["stressed", "crisis"])
    if risk_mask.sum() > 2:
        risk_base   = _annualised_sharpe(port_ret[risk_mask])
        risk_hedged = _annualised_sharpe(hedged_ret[risk_mask])
        print(f"\n  Stressed + crisis periods ({risk_mask.sum()} days):")
        print(f"    Sharpe no hedge:   {risk_base:.3f}")
        print(f"    Sharpe with hedge: {risk_hedged:.3f}  (improvement = {risk_hedged - risk_base:+.3f})")
    else:
        print(f"\n  No stressed/crisis periods in this window ({risk_mask.sum()} days)")

    # Hedge-drawdown correlation
    rolling_dd = (port_ret + 1).cumprod()
    rolling_dd = (rolling_dd - rolling_dd.cummax()) / rolling_dd.cummax()
    hedge_corr = hedge_weights.corr(rolling_dd)
    print(f"\n  Hedge weight ↔ portfolio drawdown correlation: {hedge_corr:.3f}")
    print("  (Negative = hedge grows when drawdown deepens — desired)")
    print("\n  Done.")


if __name__ == "__main__":
    main()
