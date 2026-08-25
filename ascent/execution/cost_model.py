"""
ascent/execution/cost_model.py

Simplified Almgren-Chriss transaction cost model.

Model formulas (one-way):
    spread_cost_bps   = spread_bps / 2
    temporary_impact  = eta * sigma_daily * sqrt(X / V_daily) * 10_000   [bps]
    permanent_impact  = gamma * sigma_daily * (X / V_daily) * 10_000     [bps]
    total_one_way_bps = spread_cost_bps + temporary_impact + permanent_impact

Where:
    X        = trade size in dollars (abs)
    V_daily  = dollar_vol_21d / 21  (daily avg dollar volume)
    sigma_daily = vol_21d_annualized / sqrt(252)
    eta      = temporary impact coefficient (default 0.10)
    gamma    = permanent impact coefficient (default 0.05)

Participation rate flags:
    > max_participation_rate   → HIGH_IMPACT (order blocked by order_engine)
    > split_participation_rate → SPLIT_RECOMMENDED (warning only)
    missing volume data         → IMPACT_UNKNOWN (warning only, order not blocked)

Relationship to ascent/backtest/costs.py::liquidity_scaled_cost_model():
This is the live-facing pre-trade estimate (called from order_engine.py to
flag/block real orders before they reach the broker, and from eod_runner.py's
extract_cost_features()). costs.py's liquidity_scaled_cost_model() is a
separate, simpler sqrt-impact formula used only inside BacktestEngine.run()
to estimate historical trading costs for the walk-forward Sharpe figure. They
are not consolidated into one function: this model needs a per-symbol
volatility feature (vol_21d) that isn't currently plumbed through the
backtest engine or its walk-forward callers, and adds a permanent-impact term
the backtest model doesn't have. A scratch comparison (2026-08-24) across
0.2%-15% participation showed the backtest-side estimate consistently running
LOWER than this live estimate — e.g. ~6.6bps here vs ~5.5bps in costs.py at
0.2% participation, widening to ~19.6bps here vs ~6.9bps in costs.py at 15%
participation — so a walk-forward backtest that turns the ADV-scaled cost
model on should be read as a lower bound on real trading cost, not a match
to what order_engine.py would actually charge for the same trade. See
liquidity_scaled_cost_model()'s docstring for the reverse cross-reference.
"""
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PARAMS_PATH = Path("data_cache/cost_model_params.json")


@dataclass
class CostModelParams:
    eta: float = 0.10                       # temporary impact coefficient
    gamma: float = 0.05                     # permanent impact coefficient
    spread_bps: float = 5.0                # round-trip spread (BacktestConfig default)
    max_participation_rate: float = 0.10    # block orders > 10% of ADV
    split_participation_rate: float = 0.05  # warn if > 5% of ADV


@dataclass
class CostEstimate:
    symbol: str
    spread_bps: float
    temporary_impact_bps: float         # NaN if volume data missing
    permanent_impact_bps: float         # NaN if volume data missing
    total_one_way_bps: float            # NaN if volume data missing
    participation_rate: float           # NaN if volume data missing
    flag: Optional[str]                 # "HIGH_IMPACT" | "SPLIT_RECOMMENDED" | "IMPACT_UNKNOWN" | None


def load_params() -> CostModelParams:
    """Load calibrated params from disk; fall back to defaults if file missing."""
    if PARAMS_PATH.exists():
        try:
            data = json.loads(PARAMS_PATH.read_text())
            valid_keys = CostModelParams.__dataclass_fields__
            return CostModelParams(**{k: v for k, v in data.items() if k in valid_keys})
        except Exception as exc:
            log.warning(f"[CostModel] Failed to load params ({exc}) — using defaults")
    return CostModelParams()


def estimate(
    symbol: str,
    dollar_amount: float,
    features: dict,
    params: Optional[CostModelParams] = None,
) -> CostEstimate:
    """
    Estimate one-way transaction cost for a single order.

    Args:
        symbol: ticker
        dollar_amount: absolute trade size in dollars
        features: dict with keys in the following format:
                  "dollar_vol_21d" → {symbol: float}  — 21-day total dollar volume (raw sum, not ranked)
                  "vol_21d"        → {symbol: float}  — annualized daily vol (0–1 scale, e.g. 0.20 = 20%)

                  Note: FeatureBuilder.compute_features() produces DataFrames and uses
                  "dollar_vol_rank_21d" (a rank, not a scalar). Use extract_cost_features()
                  to convert FeatureBuilder output to this format before calling estimate().

        params: cost model parameters; loaded from disk if None

    Returns:
        CostEstimate with flag:
        - None: small order, no concerns
        - "SPLIT_RECOMMENDED": participation 5–10% of ADV
        - "HIGH_IMPACT": participation > 10% of ADV (order should be blocked)
        - "IMPACT_UNKNOWN": volume data unavailable (don't block, just warn)
    """
    if params is None:
        params = load_params()

    nan = float("nan")
    half_spread = params.spread_bps / 2

    # Resolve daily dollar volume
    dollar_vol_21d_map = features.get("dollar_vol_21d", {}) or {}
    dollar_vol_21d = dollar_vol_21d_map.get(symbol)

    if not dollar_vol_21d or dollar_vol_21d <= 0:
        return CostEstimate(
            symbol=symbol,
            spread_bps=half_spread,
            temporary_impact_bps=nan,
            permanent_impact_bps=nan,
            total_one_way_bps=nan,
            participation_rate=nan,
            flag="IMPACT_UNKNOWN",
        )

    V_daily = dollar_vol_21d / 21
    X = abs(dollar_amount)
    participation = X / V_daily if V_daily > 0 else nan

    # Resolve daily vol (annualized → daily)
    vol_21d_map = features.get("vol_21d", {}) or {}
    vol_21d_annualized = vol_21d_map.get(symbol, 0.20)  # default 20% if missing
    sigma_daily = vol_21d_annualized / math.sqrt(252)

    temp_impact = params.eta * sigma_daily * math.sqrt(participation) * 10_000
    perm_impact = params.gamma * sigma_daily * participation * 10_000
    total = half_spread + temp_impact + perm_impact

    flag = None
    if participation > params.max_participation_rate:
        flag = "HIGH_IMPACT"
    elif participation > params.split_participation_rate:
        flag = "SPLIT_RECOMMENDED"

    return CostEstimate(
        symbol=symbol,
        spread_bps=half_spread,
        temporary_impact_bps=temp_impact,
        permanent_impact_bps=perm_impact,
        total_one_way_bps=total,
        participation_rate=participation,
        flag=flag,
    )


def apply_cost_filter(
    orders: list,
    features: dict,
    portfolio_value: float,
    params: Optional[CostModelParams] = None,
) -> tuple:
    """
    Apply cost model to an order list. Returns (filtered_orders, estimates, total_cost_bps).

    HIGH_IMPACT orders are removed. SPLIT_RECOMMENDED and IMPACT_UNKNOWN are kept with a log warning.
    total_cost_bps = weighted sum of total_one_way_bps * 2 (round-trip) across non-blocked orders.
    """
    if params is None:
        params = load_params()

    all_estimates = [estimate(o.symbol, o.dollar_amount, features, params) for o in orders]

    blocked_symbols = []
    filtered_orders = []
    filtered_estimates = []

    for o, e in zip(orders, all_estimates):
        if e.flag == "HIGH_IMPACT":
            log.warning(
                f"[CostModel] {o.symbol}: participation={e.participation_rate:.1%} > "
                f"{params.max_participation_rate:.0%} ADV — order BLOCKED"
            )
            blocked_symbols.append(o.symbol)
        else:
            if e.flag == "SPLIT_RECOMMENDED":
                log.warning(
                    f"[CostModel] {o.symbol}: participation={e.participation_rate:.1%} — SPLIT RECOMMENDED"
                )
            elif e.flag == "IMPACT_UNKNOWN":
                log.warning(f"[CostModel] {o.symbol}: volume data unavailable — cost estimate unknown")
            filtered_orders.append(o)
            filtered_estimates.append(e)

    if blocked_symbols:
        log.warning(f"[CostModel] Blocked {len(blocked_symbols)} HIGH_IMPACT orders: {blocked_symbols}")

    # Round-trip cost = one-way * 2, weighted by |weight_delta|
    total_weight = sum(abs(o.weight_delta) for o in filtered_orders)
    total_cost_bps = 0.0
    if total_weight > 0:
        for o, e in zip(filtered_orders, filtered_estimates):
            if not math.isnan(e.total_one_way_bps):
                w = abs(o.weight_delta) / total_weight
                total_cost_bps += e.total_one_way_bps * 2 * w

    if filtered_orders:
        log.info(f"[CostModel] Estimated weighted round-trip cost: {total_cost_bps:.1f} bps")

    # Return all_estimates (includes blocked) so callers can inspect HIGH_IMPACT flags
    return filtered_orders, all_estimates, total_cost_bps


def extract_cost_features(features_df_dict: dict, as_of_date=None) -> dict:
    """
    Convert FeatureBuilder output format to cost model input format.

    FeatureBuilder produces DataFrames (dates × symbols). This extracts the
    most-recent row of each relevant feature into the {symbol: scalar} format
    that estimate() expects.

    Returns a dict with:
        "dollar_vol_21d": {symbol: float}  — 21-day total dollar volume
        "vol_21d":        {symbol: float}  — annualized daily vol (0–1 scale)

    If the FeatureBuilder keys are absent or in an unexpected format, returns {}.
    This function never raises.
    """
    import pandas as _pd
    result = {}

    try:
        # vol_21d: DataFrame(dates × symbols) → last row → {symbol: scalar}
        vol = features_df_dict.get("vol_21d")
        if vol is not None and hasattr(vol, "iloc"):
            if as_of_date is not None:
                row = vol.loc[:as_of_date].iloc[-1] if len(vol.loc[:as_of_date]) > 0 else vol.iloc[-1]
            else:
                row = vol.iloc[-1]
            result["vol_21d"] = row.dropna().to_dict()
    except Exception:
        pass

    try:
        # FeatureBuilder doesn't have raw dollar_vol_21d. Try common alternatives:
        # 1. "dollar_volume" column (raw daily dollar volume DataFrame)
        # 2. "dollar_vol_21d" if it somehow exists as raw values
        for key in ("dollar_volume", "dollar_vol_21d"):
            dvol = features_df_dict.get(key)
            if dvol is not None and hasattr(dvol, "iloc"):
                # Sum last 21 rows to get 21-day total dollar volume per symbol
                last21 = dvol.tail(21)
                result["dollar_vol_21d"] = last21.sum(axis=0).to_dict()
                break
    except Exception:
        pass

    return result
