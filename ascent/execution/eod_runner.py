"""
ascent/execution/eod_runner.py

End-of-day execution runner for Ascent Capital paper trading.

Usage:
    python -m ascent.execution.eod_runner
    python -m ascent.execution.eod_runner --dry-run
    python -m ascent.execution.eod_runner --date 2026-03-24
"""
import argparse
import time
import traceback
from datetime import datetime, date
import pandas as pd

from ascent.execution.run_log import log_run
from ascent.execution import kill_switch
from ascent.execution.run_log import log_error, read_last_run
from ascent.execution.alpaca_broker import (
    get_account,
    get_positions,
    get_portfolio_value,
    submit_order,
    cancel_all_orders,
    is_market_open,
)
from ascent.execution.order_engine import compute_orders, summarise_orders

try:
    from ascent.execution.intraday_trigger import check_intraday_triggers, execute_intraday_adjustment
except Exception:
    check_intraday_triggers      = None  # type: ignore[assignment]
    execute_intraday_adjustment  = None  # type: ignore[assignment]

try:
    from compliance.audit_trail import record_event as _audit
except Exception:
    def _audit(event_type, payload):  # type: ignore[misc]
        pass

try:
    from ascent.monitoring.alert_system import check_alerts
except Exception:
    check_alerts = None  # type: ignore[assignment]


def run_intraday_trigger_check(portfolio_state: dict = None, market_data: dict = None) -> list:
    """
    Evaluate and execute intraday triggers. Called at 12:00 PM and 14:30 PM ET.
    portfolio_state: {"weights": dict, "nav": float, "drawdown": float}
    market_data: {"spy_intraday_return": float, "vix": float}
    Returns list of trigger results (empty = no action taken).
    """
    if check_intraday_triggers is None:
        return []
    portfolio_state = portfolio_state or {}
    market_data     = market_data     or {}
    try:
        triggers = check_intraday_triggers(portfolio_state, market_data)
        results  = []
        for trigger in triggers:
            result = execute_intraday_adjustment(trigger, portfolio_state.get("weights", {}))
            results.append(result)
            log.warning("[EODRunner] Intraday trigger executed: %s", trigger.get("type"))
        return results
    except Exception as e:
        log.warning("[EODRunner] Intraday trigger check failed: %s", e)
        return []


def get_event_positions_today() -> dict[str, float]:
    """
    Return {symbol: net_direction} (+1 long, -1 short) for filled event trades today.
    Used by run_eod to subtract event positions before sizing daily rebalance orders.
    """
    import json
    from pathlib import Path
    log_path = Path("logs/event_trades.jsonl")
    if not log_path.exists():
        return {}
    today_str = date.today().isoformat()
    net: dict[str, float] = {}
    try:
        with open(log_path) as f:
            for line in f:
                t = json.loads(line)
                if t.get("timestamp", "")[:10] != today_str:
                    continue
                if t.get("status") not in ("submitted", "filled"):
                    continue
                sym = t.get("symbol", "")
                direction = 1.0 if t.get("direction") == "buy" else -1.0
                net[sym] = net.get(sym, 0.0) + direction
    except Exception:
        pass
    return {k: v for k, v in net.items() if v != 0.0}


def run_eod(dry_run: bool = False, as_of_date: str = None):
    today = as_of_date or date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"  ASCENT CAPITAL — EOD RUNNER  |  {today}  |  {'DRY RUN' if dry_run else 'LIVE PAPER'}")
    print(f"{'='*60}\n")

    try:
        print("[EOD] Running Ascent pipeline...")
        from ascent.config.settings import get_config
        from ascent.main import run_pipeline

        cfg = get_config()  # Bug 13 fix: use shared get_config() not direct Config()

        (
            result,
            regime_engine,
            spy_wide,
            univ_wide,
            vix_series,
            target_weights_all,
            price_df,
            macro_df,
            price_cache_name,
            _alpha_breakdown,  # per-sleeve breakdown (unused in execution path)
        ) = run_pipeline(
            live=True,
        )

        if target_weights_all is None or target_weights_all.empty:
            raise ValueError("Pipeline returned no target weights.")

        valid_rows = target_weights_all[(target_weights_all > 0).any(axis=1)]
        if valid_rows.empty:
            raise ValueError("Pipeline returned no positive-weight positions — aborting EOD run")
        latest_date = valid_rows.index[-1]
        target_weights_all = valid_rows
        target_weights     = target_weights_all.loc[latest_date]
        target_weights     = target_weights[target_weights > 0].dropna()

        # Filter to only currently tradeable symbols
        from ascent.data.universe import get_universe_on_date, build_historical_universe
        universe_df = build_historical_universe(strict=False)
        tradeable = set(get_universe_on_date(today, universe_df))
        target_weights = target_weights[target_weights.index.isin(tradeable)]
        target_weights = target_weights[target_weights > 0].dropna()
        if target_weights.sum() > 0:
            target_weights = target_weights / target_weights.sum()

        print(f"[EOD] Latest signal date: {latest_date.date()}")
        print(f"[EOD] Target positions: {len(target_weights)}")

        regime_label      = None
        regime_confidence = None
        posture           = "unknown"

        if regime_engine is not None:
            try:
                latest_date_ts = pd.Timestamp(target_weights_all.index[-1])
                sig = regime_engine.get_signal(latest_date_ts)
                if sig is not None:
                    regime_label      = sig.label
                    regime_confidence = float(sig.confidence) if sig.confidence else None
                    try:
                        from ascent.regime.posture import compute_posture_from_regime
                        probs_dict = {str(i): float(v) for i, v in enumerate(sig.probs)}
                        posture_obj = compute_posture_from_regime(
                            asof=str(latest_date_ts.date()),
                            regime_label=str(regime_label) if regime_label else "unknown",
                            probs=probs_dict,
                            days_in_regime=sig.dwell_days,
                        )
                        posture = str(posture_obj.posture) if posture_obj and hasattr(posture_obj, "posture") else regime_label or "unknown"
                    except ImportError as _ie:
                        print(f"[EOD] WARNING: posture import failed ({_ie}) — using regime label as fallback")
                        posture = regime_label or "unknown"
            except Exception as _re:
                print(f"[EOD] WARNING: regime signal extraction failed ({type(_re).__name__}: {_re}) — proceeding with posture=unknown")
                try:
                    log_error(run_date=today, error=f"regime_unavailable: {type(_re).__name__}: {_re}")
                except Exception:
                    pass

        print(f"[EOD] Regime: {regime_label}  |  Posture: {posture}")

        is_rebalance = _is_rebalance_day(target_weights_all, today)
        print(f"[EOD] Rebalance day: {is_rebalance}")

        portfolio_value   = get_portfolio_value()
        current_positions = get_positions()

        print(f"[EOD] Portfolio value: ${portfolio_value:,.2f}")
        print(f"[EOD] Current positions: {len(current_positions)}")

        # Regime narrative — generate once per day, cached
        try:
            from ascent.reporting.regime_narrative import generate_regime_narrative
            narrative = generate_regime_narrative()
            if narrative:
                print(f"[EOD] Regime narrative: {narrative}")
        except Exception as _rn_err:
            print(f"[EOD] Regime narrative failed (non-fatal): {_rn_err}")

        # Exit alerts — check held positions for significant intraday drops
        try:
            from ascent.monitoring.exit_alerts import run_exit_alerts
            current_weights_dict = {}
            if not current_positions.empty and "symbol" in current_positions.columns:
                current_weights_dict = dict(zip(
                    current_positions["symbol"],
                    current_positions["weight"].tolist()
                    if "weight" in current_positions.columns
                    else [0.0] * len(current_positions)
                ))
            signal_alerts = run_exit_alerts(current_weights_dict)
            if signal_alerts:
                print(f"[EOD] ⚠️  {len(signal_alerts)} SIGNAL alert(s) — check logs/exit_alerts.jsonl")
        except Exception as _ea_err:
            print(f"[EOD] Exit alerts failed (non-fatal): {_ea_err}")

        if is_rebalance and not current_positions.empty:
            current_weights_series = current_positions.set_index("symbol")["weight"]
            all_syms = target_weights.index.union(current_weights_series.index)
            target_aligned  = target_weights.reindex(all_syms, fill_value=0.0)
            current_aligned = current_weights_series.reindex(all_syms, fill_value=0.0)
            max_delta = (target_aligned - current_aligned).abs().max()
            if max_delta < 0.005:
                is_rebalance = False
                print('[EOD] Weights match targets within 0.5% — skipping rebalance.')

        if not is_rebalance:
            print("[EOD] Non-rebalance day — logging regime + holdings only.")
            current_holdings = (
                current_positions.set_index("symbol")["weight"].to_dict()
                if not current_positions.empty else {}
            )
            log_run(
                run_date=today,
                run_type="log_only",
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                posture=posture,
                portfolio_value=portfolio_value,
                target_weights=target_weights.to_dict(),
                orders_executed=[],
                orders_skipped=[],
                notes=f"Non-rebalance day. Current holdings: {list(current_holdings.keys())}",
            )
            print("[EOD] Done — no trades submitted.")
            return

        # Subtract today's event positions so the daily rebalance doesn't double-buy
        event_positions = get_event_positions_today()
        if event_positions:
            nav_est = float(portfolio_value) if portfolio_value else 100_000.0
            from ascent.execution.event_runner import MAX_EVENT_PCT
            for sym, net_dir in event_positions.items():
                event_weight = net_dir * MAX_EVENT_PCT  # approximate weight adjustment
                if sym in target_weights.index:
                    target_weights[sym] = max(0.0, target_weights[sym] - event_weight)
            if target_weights.sum() > 0:
                target_weights /= target_weights.sum()
            print(f"[EOD] Adjusted for {len(event_positions)} event position(s) today")

        orders, diff_df = compute_orders(
            target_weights=target_weights,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
        )

        print(f"\n[EOD] Order plan:\n{summarise_orders(orders)}\n")

        if not orders:
            log_run(
                run_date=today,
                run_type="rebalance",
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                posture=posture,
                portfolio_value=portfolio_value,
                target_weights=target_weights.to_dict(),
                orders_executed=[],
                orders_skipped=[],
                notes="Rebalance day but all deltas below 0.5% threshold.",
            )
            print("[EOD] No orders above threshold. Done.")
            return

        # Kill switch check -> Task 4 compliance gate (shadow) ->
        # cancel_all_orders -> order submission loop -> per-order audit.
        # Shared with run_eod_with_weights() via _execute_order_batch() --
        # see that function's docstring for the full Step-1 divergence list
        # and how each was resolved (Task 5, min-viable-cut completion plan).
        try:
            executed, skipped = _execute_order_batch(
                orders, current_positions, portfolio_value, today,
                dry_run=dry_run,
                log_prefix="[EOD]",
                # run_eod() never had a close_position() special case for
                # full-liquidation sells -- preserve that exactly (see
                # DIVERGENCE #3 in _execute_order_batch).
                use_close_position_for_full_liquidation=False,
            )
        except kill_switch.KillSwitchTriggered:
            log_run(
                run_date=today,
                run_type="error",
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                posture=posture,
                portfolio_value=portfolio_value,
                target_weights=target_weights.to_dict(),
                orders_executed=[],
                orders_skipped=[],
                notes="HALTED by kill switch — drawdown exceeded threshold.",
            )
            return

        if dry_run:
            print("[EOD] DRY RUN — orders NOT submitted to Alpaca.")
            log_run(
                run_date=today,
                run_type="rebalance",
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                posture=posture,
                portfolio_value=portfolio_value,
                target_weights=target_weights.to_dict(),
                orders_executed=[],
                orders_skipped=[{"symbol": o.symbol, "side": o.side, "delta": o.weight_delta, "dry_run": True} for o in orders],
                notes="Dry run — no orders submitted.",
            )
            return

        # ── Task 5: Slippage tracking (post-fill) ────────────────────────────
        # Only runs when orders were actually submitted (not dry-run, not non-rebalance).
        if executed:
            try:
                print("[EOD] Waiting 30s for fills to settle before slippage check...")
                time.sleep(30)

                # Build signal prices from the pipeline's last close row
                if price_df is not None and not price_df.empty:
                    signal_prices = price_df.iloc[-1].dropna().to_dict()
                    from ascent.execution.slippage_tracker import track_slippage
                    track_slippage(signal_prices, run_date=date.fromisoformat(today))
                else:
                    print("[EOD] No price data available for slippage tracking")
            except Exception as e:
                print(f"[EOD] Slippage tracking failed ({e}) — non-fatal, continuing")
        # ── End Task 5 ───────────────────────────────────────────────────────

        log_run(
            run_date=today,
            run_type="rebalance",
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            posture=posture,
            portfolio_value=portfolio_value,
            target_weights=target_weights.to_dict(),
            orders_executed=executed,
            orders_skipped=skipped,
            notes=f"Submitted {len(executed)} orders, skipped {len(skipped)}.",
        )

        print(f"\n[EOD] Complete — {len(executed)} orders submitted, {len(skipped)} skipped.")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[EOD] ERROR: {e}\n{tb}")
        log_error(run_date=today, error=f"{e}\n{tb}")
        raise


def _is_rebalance_day(target_weights_all: pd.DataFrame, today: str) -> bool:
    import os
    calendar_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "rebalance_calendar.csv"))
    if os.path.exists(calendar_path):
        import pandas as _pd
        cal = _pd.read_csv(calendar_path)
        scheduled = set(cal["rebalance_date"].tolist())
        if today in scheduled:
            print(f"[EOD] {today} is on the rebalance calendar — rebalancing.")
            return True
        print(f"[EOD] {today} is NOT on the rebalance calendar — skipping.")
        return False
    today_ts    = pd.Timestamp(today)
    rebal_dates = target_weights_all.index
    if rebal_dates.tz is not None:
        today_ts = today_ts.tz_localize(rebal_dates.tz)
    diffs = abs(rebal_dates - today_ts)
    if diffs.min().days > 1:
        return False
    if len(rebal_dates) < 2:
        return True
    latest_weights = target_weights_all.iloc[-1]
    prev_weights   = target_weights_all.iloc[-2]
    all_cols       = latest_weights.index.union(prev_weights.index)
    latest_aligned = latest_weights.reindex(all_cols, fill_value=0.0)
    prev_aligned   = prev_weights.reindex(all_cols, fill_value=0.0)
    return bool((latest_aligned - prev_aligned).abs().max() >= 0.005)

def _get_price(symbol: str, positions: pd.DataFrame) -> float:
    if not positions.empty and symbol in positions["symbol"].values:
        p = positions.loc[positions["symbol"] == symbol, "current_price"].values[0]
        if p > 0:
            return float(p)
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ascent Capital EOD Runner")
    parser.add_argument("--dry-run",           action="store_true")
    parser.add_argument("--reset-kill-switch",  action="store_true")
    parser.add_argument("--date",              type=str, default=None)
    args = parser.parse_args()

    if args.reset_kill_switch:
        kill_switch.reset()
    else:
        run_eod(dry_run=args.dry_run, as_of_date=args.date)


# ── Multi-agent orchestrator execution path ────────────────────────────────────
# Added by Phase B patch. Does NOT modify the existing run_eod() function.


# ── Task 5: shared order-submission path ────────────────────────────────────
# Collapses run_eod()'s and run_eod_with_weights()'s independent
# kill-switch-check -> [Task 4 compliance gate] -> cancel-all-orders ->
# submit-loop -> per-order-audit sequences into one helper both call. Extracted
# 2026-08-20 (Task 5 of the min-viable-cut completion plan); see that plan's
# task-5-report.md for the full Step-1 divergence list. Every behavioral
# difference the two functions used to have is called out at its resolution
# point below with a DIVERGENCE comment -- none was dropped silently.
def _execute_order_batch(
    orders: list,
    current_positions: pd.DataFrame,
    portfolio_value: float,
    today_str: str,
    *,
    dry_run: bool = False,
    log_prefix: str = "[EOD]",
    use_close_position_for_full_liquidation: bool = False,
) -> tuple:
    """
    Run the shared kill-switch-check -> Task 4 compliance-gate(shadow) ->
    cancel-all-orders -> submit-loop -> per-order-audit sequence used by both
    run_eod() and run_eod_with_weights().

    Does NOT change what order gets submitted, at what size, or on what
    schedule versus either caller's pre-refactor behavior -- see the
    `use_close_position_for_full_liquidation` DIVERGENCE note below for the
    one place that distinction is preserved rather than unified.

    Returns (executed, skipped): both lists of dicts (`executed` has
    symbol/side/qty/dollar_amount/order_id; `skipped` has symbol/reason).
    Both are empty when dry_run is True or the kill switch has NOT tripped
    but no orders were actually submitted for another reason -- callers
    decide what to log from the returned lists themselves, since run_eod()
    and run_eod_with_weights() log to different schemas (log_run vs
    _log_multi_run).

    Raises kill_switch.KillSwitchTriggered if the kill switch fires --
    already audited (see below) by the time it propagates, so callers only
    need to catch it and do their own run-specific logging/return. Any OTHER
    exception raised by kill_switch.check() propagates unmodified.

    DIVERGENCE (Step 1, #1 -- the known one): run_eod_with_weights() used to
    catch every exception from kill_switch.check(), and for anything that
    was NOT KillSwitchTriggered it printed a warning and fell through to
    keep trading; run_eod() only ever caught KillSwitchTriggered and let
    everything else propagate to its outer try/except (which logs and
    re-raises). Resolved to run_eod()'s stricter behavior: silently
    continuing an order-submission loop past an unknown, unclassified
    failure from the portfolio-level circuit breaker is the riskier of the
    two defaults, and "print a warning and keep trading" is not a safe
    fallback for a function whose entire job is deciding whether trading is
    safe to continue.
    """
    # ── Kill switch check ────────────────────────────────────────────────
    try:
        kill_switch.check(current_nav=portfolio_value)
    except kill_switch.KillSwitchTriggered:
        # DIVERGENCE (Step 1, #2): only run_eod() called _audit() on a
        # kill-switch trigger; run_eod_with_weights() went straight to
        # _log_multi_run() and skipped the compliance audit trail entirely.
        # _audit() is a log-only side effect (compliance/audit_trail.py) --
        # calling it uniformly cannot change what gets submitted, so this is
        # unified to "always audit" rather than kept as a per-caller gap.
        _trip_kind = kill_switch._load_state().get("trip_reason_kind")
        _trip_threshold = (
            kill_switch.MONTHLY_SOFT_HALT_PCT if _trip_kind == "monthly"
            else kill_switch.HARD_STOP_PCT
        )
        _audit("kill_switch_triggered", {
            "nav": portfolio_value, "date": today_str,
            "threshold": _trip_threshold,
        })
        raise

    # ── Task 4: Pre-Trade Compliance Checker -- SHADOW MODE ONLY. ──────────
    # This block computes and logs what compliance_gate.check_batch() would
    # reject (restricted symbols, orders needing large-trade approval,
    # buying-power overflow) but deliberately does NOT remove anything from
    # `orders` yet -- it must not change what gets submitted, at what size,
    # or on what schedule. A follow-up task promotes this to enforcing once
    # the shadow-mode log output has been reviewed.
    #
    # Verbatim carry-forward from Task 4 (commit 46761e8), which added this
    # only inside run_eod_with_weights(). Task 5 moves it here unchanged so
    # it now also fires from run_eod() -- previously run_eod() had no
    # compliance-gate call at all, which is exactly the "would otherwise
    # need to be built twice" gap Task 5 exists to close, not a new
    # divergence introduced by this refactor.
    try:
        from ascent.execution.compliance_gate import check_batch
        try:
            _buying_power = float(get_account().get("buying_power"))
        except Exception as _bp_exc:
            print(f"[ComplianceGate] buying_power lookup failed ({_bp_exc}) — buying-power check skipped this run")
            _buying_power = None
        _gate_decisions = check_batch(
            orders, portfolio_value,
            buying_power=_buying_power,
            live_positions=current_positions,
            trade_date=today_str,
        )
        for _gd in _gate_decisions:
            if not _gd.approved:
                print(f"[ComplianceGate][SHADOW] Would reject {_gd.order_id}: {_gd.reason}")
    except Exception as _cg_exc:
        print(f"[ComplianceGate] check_batch failed ({_cg_exc}) — shadow mode, continuing")

    if dry_run:
        # Both callers already print/log their own dry-run message using
        # their own schema right after this helper returns; nothing to
        # submit or audit here.
        return [], []

    cancel_all_orders()
    print(f"{log_prefix} Cancelled open orders.")

    executed: list = []
    skipped:  list = []

    for o in orders:
        from ascent.execution.order_engine import _get_approx_price
        price = _get_approx_price(o.symbol, current_positions)
        if price is None:
            print(f"{log_prefix} Skipping {o.symbol} — price unavailable")
            skipped.append({"symbol": o.symbol, "reason": "price unavailable"})
            continue

        qty = round(o.dollar_amount / price, 6)
        if qty < 0.001:
            skipped.append({"symbol": o.symbol, "reason": "qty too small"})
            continue

        try:
            # DIVERGENCE (Step 1, #3): run_eod_with_weights() used
            # close_position() instead of submit_order() for full-liquidation
            # sells (target_weight == 0.0), specifically to avoid a 403 from
            # a qty-rounding mismatch between estimated and actual share
            # count (see alpaca_broker.close_position()'s docstring).
            # run_eod() always used submit_order(), with no close_position()
            # special case. This is a genuine "what gets submitted to the
            # broker" difference, which the plan forbids changing for either
            # path -- so it is intentionally NOT unified either way and
            # stays a per-caller flag (`use_close_position_for_full_liquidation`)
            # instead.
            if use_close_position_for_full_liquidation and o.side == "sell" and o.target_weight == 0.0:
                from ascent.execution.alpaca_broker import close_position
                resp = close_position(o.symbol)
                order_id = resp.get("id") if isinstance(resp, dict) else None
                print(f"{log_prefix} CLOSE {o.symbol}  (${o.dollar_amount:,.0f})")
            else:
                resp = submit_order(symbol=o.symbol, qty=qty, side=o.side)
                order_id = resp.get("id") if resp else None
                print(f"{log_prefix} {o.side.upper()} {qty:.4f} {o.symbol}  (${o.dollar_amount:,.0f})")

            # DIVERGENCE (Step 1, #4): run_eod() audited every successful
            # submission via _audit('order_submitted', ...); run_eod_with_
            # weights() never did. Unified to always audit (log-only side
            # effect, same reasoning as the kill-switch audit above).
            #
            # The broker call above has already succeeded by this point, so
            # the order is genuinely live -- an audit-log failure (disk full,
            # permissions, lock contention; audit_trail.record() does real
            # file I/O under a lock) must never cause this order to fall into
            # `skipped` and be misreported as not submitted. Bookkeeping for
            # a successful broker call is therefore unconditional, and the
            # audit write gets its own try/except so it can't reach the outer
            # handler.
            try:
                _audit("order_submitted", {
                    "symbol": o.symbol, "side": o.side, "qty": qty,
                    "dollar_amount": round(o.dollar_amount, 2),
                    "order_id": order_id, "date": today_str,
                })
            except Exception as _audit_exc:
                print(f"{log_prefix} WARNING: audit log failed for {o.symbol} "
                      f"(order already submitted, order_id={order_id}): {_audit_exc}")

            executed.append({
                "symbol":        o.symbol,
                "side":          o.side,
                "qty":           qty,
                "dollar_amount": o.dollar_amount,
                "order_id":      order_id,
            })
        except Exception as e:
            # DIVERGENCE (Step 1, #6): run_eod() printed per-order submission
            # failures as "Failed {sym}: {e}"; run_eod_with_weights() printed
            # them as "{sym} FAILED: {e}". Both formats were cosmetic with no
            # behavioral impact. Unified to run_eod()'s "Failed {sym}: {e}"
            # format (the slightly-more-natural reading order).
            print(f"{log_prefix} Failed {o.symbol}: {e}")
            skipped.append({"symbol": o.symbol, "reason": str(e)})

    # DIVERGENCE (Step 1, #5): run_eod()'s `skipped` entries were always
    # {"symbol", "reason"} dicts; run_eod_with_weights()'s were bare symbol
    # strings with no reason recorded anywhere (price-unavailable, qty-too-
    # small, and submit failures were all indistinguishable downstream).
    # Unified to the richer dict shape -- run_eod_with_weights() only ever
    # used len(skipped) and the bare symbol for its own note string, both of
    # which still work unchanged against a dict list (see its call site).
    return executed, skipped


def run_eod_with_weights(merged_weights: dict, run_date=None, dry_run: bool = False, force: bool = False):
    """
    Execute EOD with pre-computed weights from the orchestrator.
    Called by run_all_agents.py instead of run_eod().

    Handles: rebalance calendar check, kill switch, order computation,
    large-trade approval, order submission, slippage tracking, and logging.
    Does NOT run the ascent pipeline internally.

    force=True bypasses ONLY the rebalance-calendar gate (used by intra-period
    actions: discovery mini-rebalances). Kill switch and large-trade approval
    still apply.
    """
    import time as _time
    import pandas as _pd

    from ascent.utils.market_time import market_today

    # market_today(), not date.today() — this host is UTC+7, so local time names
    # the next US session for ~14h of every day. See ascent/utils/market_time.py.
    today = run_date or market_today()
    today_str = today.isoformat() if hasattr(today, "isoformat") else str(today)

    print(f"\n[EOD-Multi] Running with orchestrator weights | {today_str}")
    print(f"[EOD-Multi] Positions: {len(merged_weights)}")

    # 1. Rebalance calendar check — reuse existing helper
    # _is_rebalance_day expects a DataFrame for the first arg; pass None for multi-agent path
    # The function falls back to rebalance_calendar.csv when the DataFrame is None
    import os as _os
    import pandas as _pd2
    calendar_path = _os.path.normpath(_os.path.join(
        _os.path.dirname(__file__), "..", "..", "rebalance_calendar.csv"))
    if _os.path.exists(calendar_path):
        cal = _pd2.read_csv(calendar_path)
        scheduled = set(cal["rebalance_date"].tolist())
        is_rebalance = today_str in scheduled
        print(f"[EOD-Multi] Rebalance day (calendar): {is_rebalance}")
    else:
        # No calendar — treat every day as a potential rebalance day
        is_rebalance = True
        print("[EOD-Multi] No rebalance calendar found — treating as rebalance day")

    if not is_rebalance and not force:
        print("[EOD-Multi] Not a rebalance day — logging only")
        _log_multi_run(today_str, merged_weights, rebalanced=False)
        return
    if not is_rebalance and force:
        print("[EOD-Multi] Calendar gate bypassed (force=True, intra-period action)")

    # 2. Get current positions and portfolio value
    # cancel_all_orders / submit_order / close_position moved into
    # _execute_order_batch() (Task 5) -- no longer needed locally here.
    from ascent.execution.alpaca_broker import get_positions, get_portfolio_value
    portfolio_value   = get_portfolio_value()
    current_positions = get_positions()
    print(f"[EOD-Multi] Portfolio value: ${portfolio_value:,.2f}")
    print(f"[EOD-Multi] Current positions: {len(current_positions)}")

    # 3. Convert merged_weights dict to Series for order engine
    target_weights = _pd.Series(merged_weights)

    # 4. Compute orders — wire cost model features from price cache
    from ascent.execution.order_engine import compute_orders, summarise_orders
    from ascent.execution.cost_model import extract_cost_features
    _cost_features = {}
    try:
        import pathlib as _pl
        _price_cache = _pl.Path("data_cache/prices_live.parquet")
        if _price_cache.exists():
            _prices_raw = _pd.read_parquet(_price_cache)
            # Build dollar_volume DataFrame: dates × symbols
            _prices_raw["date"] = _pd.to_datetime(_prices_raw["date"]).dt.tz_localize(None)
            if "dollar_volume" not in _prices_raw.columns:
                print("[EodRunner] WARNING: dollar_volume missing from prices cache — cost features disabled")
            else:
                _dv = _prices_raw.pivot_table(
                    index="date", columns="symbol", values="dollar_volume", aggfunc="last"
                )
                _cost_features = extract_cost_features({"dollar_volume": _dv})
                print(f"[EOD-Multi] Cost features loaded: {len(_cost_features.get('dollar_vol_21d', {}))} symbols")
    except Exception as _cf_exc:
        print(f"[EOD-Multi] Cost features unavailable ({_cf_exc}) — cost filtering inactive")

    # extract_cost_features() (ascent/execution/cost_model.py) returns "dollar_vol_21d"
    # (and, only if a "vol_21d" input DataFrame was supplied — which eod_runner never
    # does — "vol_21d"). It never returns a key literally named "dollar_volume", so
    # checking for that key here always failed and silently disabled both the cost
    # filter and TWAP routing. "vol_21d" is not required: apply_cost_filter's estimate()
    # falls back to a 0.20 default when it's absent (ascent/execution/cost_model.py:124).
    _required_cost_keys = {"dollar_vol_21d"}
    features_arg = _cost_features if (_cost_features and _required_cost_keys.issubset(_cost_features)) else None
    orders, diff_df = compute_orders(target_weights, current_positions, portfolio_value,
                                     features=features_arg)

    if not orders:
        print("[EOD-Multi] No orders needed — portfolio matches targets")
        _log_multi_run(today_str, merged_weights, rebalanced=False)
        return

    print(f"\n[EOD-Multi] Order plan:\n{summarise_orders(orders)}\n")

    # 5/5b/6. Kill switch check -> Task 4 compliance gate (shadow) ->
    # cancel_all_orders -> order submission loop -> per-order audit.
    # Shared with run_eod() via _execute_order_batch() -- see that function's
    # docstring for the full Step-1 divergence list and how each was
    # resolved (Task 5, min-viable-cut completion plan).
    try:
        executed, skipped = _execute_order_batch(
            orders, current_positions, portfolio_value, today_str,
            dry_run=dry_run,
            log_prefix="[EOD-Multi]",
            # run_eod_with_weights() always used close_position() for
            # full-liquidation sells -- preserve that exactly (see
            # DIVERGENCE #3 in _execute_order_batch).
            use_close_position_for_full_liquidation=True,
        )
    except kill_switch.KillSwitchTriggered:
        print("[EOD-Multi] KILL SWITCH TRIGGERED — aborting")
        _log_multi_run(today_str, merged_weights, rebalanced=False, note="kill_switch_triggered")
        return

    if dry_run:
        print("[EOD-Multi] DRY RUN — orders NOT submitted to Alpaca")
        _log_multi_run(today_str, merged_weights, rebalanced=True, note="dry_run")
        return

    # 8. Slippage tracking
    if executed:
        try:
            print("[EOD-Multi] Waiting 30s for fills to settle...")
            _time.sleep(30)
            # Signal prices not available in multi-agent path for v1
            print("[EOD-Multi] Slippage tracking skipped (signal prices not in multi-agent path — Phase C)")
        except Exception as _slip_e:
            print(f"[EOD-Multi] Slippage tracking error: {_slip_e}")

    note = f"Submitted {len(executed)} orders, skipped {len(skipped)}"
    _log_multi_run(today_str, merged_weights, rebalanced=True, note=note)
    print(f"\n[EOD-Multi] Complete — {len(executed)} submitted, {len(skipped)} skipped.")


def _log_multi_run(today_str: str, weights: dict, rebalanced: bool, note: str = ""):
    """Log the multi-agent EOD run to eod_log.jsonl."""
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    try:
        from ascent.execution.alpaca_broker import get_portfolio_value
        nav = get_portfolio_value()
    except Exception:
        nav = None

    entry = {
        "date":           today_str,
        "run_date":       today_str,
        "source":         "multi_agent",
        "portfolio_value": nav,
        "target_weights": weights,
        "rebalanced":     rebalanced,
        "note":           note,
        "timestamp":      _dt.now().isoformat(),
    }
    log_path = _Path("logs/eod_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(_json.dumps(entry) + "\n")
    print(f"[EOD-Multi] Logged to {log_path}")
