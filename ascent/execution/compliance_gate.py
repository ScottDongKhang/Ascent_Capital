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
import json
from dataclasses import dataclass
from pathlib import Path

# Large-trade approval threshold: orders with notional > this % of portfolio NAV
# require manual approval, so they are rejected in shadow mode until enforced.
LARGE_TRADE_THRESHOLD_PCT = 2.0  # % of portfolio NAV
LARGE_TRADE_APPROVAL_PCT = LARGE_TRADE_THRESHOLD_PCT  # Alias for use in this module

# Human-populated override file, mirroring the execution/halt_override.json
# pattern (run_all_agents.py::check_halt_state): plain JSON, read fresh on
# every check_batch() call (no caching -- a human editing it mid-session
# takes effect on the next batch), missing/unreadable file treated as "no
# approvals" rather than an error. Unlike the halt override (one pending
# halt at a time, cleared on use), a batch can contain several large orders
# needing approval simultaneously and approvals are not consumed, so this is
# a list of entries instead of a single object.
#
# Each entry: {"symbol": str, "date": "YYYY-MM-DD", "max_notional": float,
# "approved_by": str}. A large order is approved if an entry matches its
# symbol and trade date AND its notional is <= that entry's max_notional.
# No expiry logic beyond the date match -- a human is expected to remove
# stale entries.
LARGE_TRADE_APPROVAL_PATH = Path("data_cache/large_trade_approvals.json")


def _load_large_trade_approvals(path: Path = LARGE_TRADE_APPROVAL_PATH) -> list:
    """Read the large-trade approval allowlist. Missing file or malformed
    JSON both mean "no approvals on file" -- this must never raise, since a
    bad override file should not take down order submission."""
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return entries if isinstance(entries, list) else []


def _find_large_trade_approval(entries: list, symbol: str, trade_date, notional: float):
    """Return the first approval entry covering this symbol/date/size, or None."""
    if not trade_date:
        return None
    for entry in entries:
        if (
            entry.get("symbol") == symbol
            and entry.get("date") == trade_date
            and notional <= entry.get("max_notional", 0.0)
        ):
            return entry
    return None


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
    trade_date: str = None,
) -> list:
    """Run every order in `orders` through the compliance gate and return one
    GateDecision per order, in the same order as the input list.

    Checks, in order:
      1. Restricted-symbol reject (default empty set -- mechanism only).
      2. Large-order approval gate: notional > LARGE_TRADE_APPROVAL_PCT% of
         portfolio_value requires manual approval. A human-populated entry in
         `LARGE_TRADE_APPROVAL_PATH` matching symbol + `trade_date` + size
         approves it; otherwise it is rejected here.
      3. Buying-power check for the surviving buy orders: if their combined
         notional exceeds available buying power, reject the
         smallest-conviction buys first (an explicit, auditable tie-break)
         until the remaining buys fit. Sell orders are never buying-power
         gated. Available buying power is `buying_power` plus the notional
         of sell orders in this same batch -- an approximation: real
         settlement is T+1/T+2, so this optimistically assumes same-batch
         sell proceeds are usable immediately, which is out of scope to
         model precisely here.
    """
    decisions: dict = {}
    surviving_buys = []  # list of (order, notional)
    surviving_sell_notional = 0.0
    # Look up the module-level path by name (not as a default arg) so tests
    # can monkeypatch LARGE_TRADE_APPROVAL_PATH and have it take effect --
    # a default-arg value would be bound once at function-definition time.
    approval_entries = _load_large_trade_approvals(LARGE_TRADE_APPROVAL_PATH)

    for order in orders:
        if order.symbol in restricted_symbols:
            decisions[order.symbol] = GateDecision(order.symbol, False, "restricted_list")
            continue

        notional = abs(order.dollar_amount)
        pct_of_nav = (notional / portfolio_value * 100) if portfolio_value else None
        if pct_of_nav is not None and pct_of_nav > LARGE_TRADE_APPROVAL_PCT:
            approval = _find_large_trade_approval(approval_entries, order.symbol, trade_date, notional)
            if approval is None:
                decisions[order.symbol] = GateDecision(
                    order.symbol,
                    False,
                    f"large_order_requires_approval ({pct_of_nav:.1f}% NAV > {LARGE_TRADE_APPROVAL_PCT}%)",
                )
                continue
            # Approved via override file -- falls through to the normal
            # buy/sell handling below instead of being auto-approved here,
            # so a buy still passes through the buying-power check.

        if order.side == "buy":
            surviving_buys.append((order, notional))
        else:
            surviving_sell_notional += notional
            decisions[order.symbol] = GateDecision(order.symbol, True, "approved")

    # Sells in this batch free up capital that buys can draw on -- see the
    # settlement-timing caveat in the docstring above.
    available_buying_power = (
        buying_power + surviving_sell_notional if buying_power is not None else None
    )

    total_buy_notional = sum(n for _, n in surviving_buys)
    if available_buying_power is not None and total_buy_notional > available_buying_power:
        # Explicit, auditable tie-break: reject smallest-conviction buys first,
        # smallest notional dropped until the remaining buys fit buying_power.
        rejected_symbols = set()
        running_total = total_buy_notional
        for order, notional in sorted(surviving_buys, key=lambda pair: pair[1]):
            if running_total <= available_buying_power:
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
