"""
ascent/execution/compliance_gate.py

Pre-Trade Compliance Checker -- Trading & Execution department. Implements
the "Pre-Trade Compliance Checker" role described in
docs/target_architecture/03_trading_execution.md, per the skeleton drafted in
docs/target_architecture/16_phase2_and_phase3_skeletons.md (Phase 3 section).

Final gate before order submission. Does not decide *what* to trade, only
approves/rejects on compliance grounds. Runs after the kill switch
(portfolio-level circuit breaker) and before cancel_all_orders() in
ascent/execution/eod_runner.py::run_eod_with_weights().

Wired in SHADOW MODE ONLY as of this task: eod_runner.py logs what this
module would reject but does not remove any order from the submission list.
See the call site comment in eod_runner.py for the enforcement TODO.

Resolution of doc 16's flagged open question ("does Order carry `price`?"):
it does not. ascent/execution/order_engine.py's `Order` dataclass carries
`symbol`, `side`, `target_weight`, `current_weight`, `weight_delta`,
`dollar_amount`, `estimated_shares` -- no `qty` and no `price` field. It does
NOT need one for this purpose: `dollar_amount` (set in compute_orders() as
`abs(delta) * portfolio_value`) IS the order's notional value already, so
this module uses `order.dollar_amount` directly instead of reconstructing
notional from `qty * price` as doc 16's skeleton speculated. This also means
`check_batch()` never needs a price lookup or `live_positions` for pricing --
`live_positions` is accepted (matching the doc-16 signature) for a future
position-aware check (e.g. restricting adds to an already-concentrated name)
but is not otherwise used yet.
"""
from dataclasses import dataclass

# Large-trade approval threshold: orders with notional > this % of portfolio NAV
# require manual approval, so they are rejected in shadow mode until enforced.
LARGE_TRADE_THRESHOLD_PCT = 2.0  # % of portfolio NAV
LARGE_TRADE_APPROVAL_PCT = LARGE_TRADE_THRESHOLD_PCT  # Alias for use in this module


@dataclass
class GateDecision:
    order_id: str
    approved: bool
    reason: str = ""


def check_batch(
    orders: list,
    portfolio_value: float,
    buying_power: float,
    live_positions,
    restricted_symbols: frozenset = frozenset(),
) -> list:
    """Run every order in `orders` through the compliance gate and return one
    GateDecision per order, in the same order as the input list.

    Checks, in order:
      1. Restricted-symbol reject (default empty set -- mechanism only).
      2. Large-order approval gate: notional > LARGE_TRADE_APPROVAL_PCT% of
         portfolio_value requires manual approval, so it is rejected here.
      3. Buying-power check for the surviving buy orders: if their combined
         notional exceeds `buying_power`, reject the smallest-conviction buys
         first (an explicit, auditable tie-break) until the remaining buys
         fit within buying power. Sell orders are never buying-power gated.
    """
    decisions: dict = {}
    surviving_buys = []  # list of (order, notional)

    for order in orders:
        if order.symbol in restricted_symbols:
            decisions[order.symbol] = GateDecision(order.symbol, False, "restricted_list")
            continue

        notional = abs(order.dollar_amount)
        pct_of_nav = (notional / portfolio_value * 100) if portfolio_value else None
        if pct_of_nav is not None and pct_of_nav > LARGE_TRADE_APPROVAL_PCT:
            decisions[order.symbol] = GateDecision(
                order.symbol,
                False,
                f"large_order_requires_approval ({pct_of_nav:.1f}% NAV > {LARGE_TRADE_APPROVAL_PCT}%)",
            )
            continue

        if order.side == "buy":
            surviving_buys.append((order, notional))
        else:
            decisions[order.symbol] = GateDecision(order.symbol, True, "approved")

    total_buy_notional = sum(n for _, n in surviving_buys)
    if buying_power is not None and total_buy_notional > buying_power:
        # Explicit, auditable tie-break: reject smallest-conviction buys first,
        # smallest notional dropped until the remaining buys fit buying_power.
        rejected_symbols = set()
        running_total = total_buy_notional
        for order, notional in sorted(surviving_buys, key=lambda pair: pair[1]):
            if running_total <= buying_power:
                break
            decisions[order.symbol] = GateDecision(order.symbol, False, "insufficient_buying_power")
            rejected_symbols.add(order.symbol)
            running_total -= notional
        for order, notional in surviving_buys:
            if order.symbol not in rejected_symbols:
                decisions[order.symbol] = GateDecision(order.symbol, True, "approved")
    else:
        for order, notional in surviving_buys:
            decisions[order.symbol] = GateDecision(order.symbol, True, "approved")

    return [decisions[order.symbol] for order in orders]
