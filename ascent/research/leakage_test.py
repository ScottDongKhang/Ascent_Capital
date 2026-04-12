"""
Ascent Capital - Leakage Test
Runs strategy forward (normal) and backward (reversed time).
If reversed performs similarly or better, you have data leakage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np

from ascent.config.settings import get_config
from ascent.data.store.parquet import load_parquet, has_data
from ascent.data.normalize.prices import pivot_prices
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack
from ascent.portfolio.optimizer import rank_weighted
from ascent.research.evaluation import annualized_return, annualized_volatility, sharpe_ratio, max_drawdown


def _run_strategy(price_df, macro_df, cfg, reverse=False):
    df = price_df.copy()
    if reverse:
        symbols = df["symbol"].unique()
        reversed_frames = []
        for sym in symbols:
            sym_df = df[df["symbol"] == sym].sort_values("date").copy()
            for col in ["open", "high", "low", "close", "volume",
                        "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]:
                if col in sym_df.columns:
                    sym_df[col] = sym_df[col].values[::-1]
            reversed_frames.append(sym_df)
        df = pd.concat(reversed_frames, ignore_index=True)

    builder = FeatureBuilder(df, macro_df)
    features = builder.compute_features()
    alpha = -build_alpha_stack(features)

    close = pivot_prices(df, "close")
    fwd_ret = close.pct_change(21).shift(-21)
    common_idx = alpha.index.intersection(fwd_ret.index)
    if len(common_idx) > 0:
        mean_ic = alpha.loc[common_idx].corrwith(fwd_ret.loc[common_idx], axis=1).mean()
    else:
        mean_ic = 0.0

    target_weights = rank_weighted(alpha, n=cfg.backtest.top_n, max_weight=cfg.backtest.max_weight, long_only=True)
    daily_returns = close.pct_change()
    common_dates = target_weights.index.intersection(daily_returns.index)
    weights_aligned = target_weights.loc[common_dates].shift(1).dropna()
    returns_aligned = daily_returns.loc[weights_aligned.index]
    port_returns = (weights_aligned * returns_aligned).sum(axis=1).dropna()

    if len(port_returns) < 10:
        return 0.0, 0.0, 0.0, 0.0

    sr = sharpe_ratio(port_returns)
    cagr = annualized_return(port_returns) * 100
    dd = max_drawdown(port_returns) * 100
    direction = "BACKWARD" if reverse else "FORWARD"
    print("  [%s] Sharpe: %.3f | CAGR: %.1f%% | IC: %.4f | MaxDD: %.1f%%" % (direction, sr, cagr, mean_ic, dd))
    return sr, cagr, mean_ic, dd


def run_leakage_test():
    cfg = get_config()
    if not has_data("prices_live"):
        print("ERROR: No cached live data.")
        return

    price_df = load_parquet("prices_live")
    macro_df = load_parquet("macro_live") if has_data("macro_live") else None

    print("=" * 70)
    print("  ASCENT CAPITAL - LEAKAGE TEST")
    print("=" * 70)

    print("")
    print("[1/2] Running FORWARD test...")
    fwd_sharpe, fwd_cagr, fwd_ic, fwd_dd = _run_strategy(price_df, macro_df, cfg, reverse=False)

    print("[2/2] Running BACKWARD test...")
    bwd_sharpe, bwd_cagr, bwd_ic, bwd_dd = _run_strategy(price_df, macro_df, cfg, reverse=True)

    print("")
    print("=" * 70)
    print("  LEAKAGE TEST RESULTS")
    print("=" * 70)

    sharpe_pass = fwd_sharpe > bwd_sharpe
    cagr_pass = fwd_cagr > bwd_cagr
    ic_pass = fwd_ic > bwd_ic

    sp = "PASS" if sharpe_pass else "FAIL"
    cp = "PASS" if cagr_pass else "FAIL"
    ip = "PASS" if ic_pass else "FAIL"

    print("  Sharpe:  FWD=%.3f  BWD=%.3f  %s" % (fwd_sharpe, bwd_sharpe, sp))
    print("  CAGR:    FWD=%.1f%%  BWD=%.1f%%  %s" % (fwd_cagr, bwd_cagr, cp))
    print("  IC:      FWD=%.4f  BWD=%.4f  %s" % (fwd_ic, bwd_ic, ip))
    print("  MaxDD:   FWD=%.1f%%  BWD=%.1f%%" % (fwd_dd, bwd_dd))

    print("")
    if sharpe_pass and ic_pass:
        print("  LEAKAGE TEST PASSED - No obvious look-ahead bias.")
    else:
        print("  LEAKAGE TEST FAILED - Check for look-ahead bias.")
    print("")


if __name__ == "__main__":
    run_leakage_test()
