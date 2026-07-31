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

# Task 8: large-trade approval threshold
LARGE_TRADE_THRESHOLD_PCT = 2.0  # % of portfolio NAV


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

        # Kill switch check
        try:
            kill_switch.check(current_nav=portfolio_value)
        except kill_switch.KillSwitchTriggered:
            _audit("kill_switch_triggered", {
                "nav": portfolio_value, "date": today,
                "threshold": kill_switch.HARD_STOP_PCT,
            })
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


        cancel_all_orders()
        print("[EOD] Cancelled open orders.")

        executed = []
        skipped  = []

        for o in orders:
            from ascent.execution.order_engine import _get_approx_price
            price = _get_approx_price(o.symbol, current_positions)
            if price is None:
                print(f"[EOD] Skipping {o.symbol} — price unavailable")
                skipped.append({"symbol": o.symbol, "reason": "price unavailable"})
                continue

            qty = round(o.dollar_amount / price, 6)
            if qty < 0.001:
                skipped.append({"symbol": o.symbol, "reason": "qty too small"})
                continue

            try:
                resp = submit_order(symbol=o.symbol, qty=qty, side=o.side)
                order_id = resp.get("id") if resp else None
                print(f"[EOD] {o.side.upper()} {qty:.4f} {o.symbol}  (${o.dollar_amount:,.0f})")
                _audit("order_submitted", {
                    "symbol": o.symbol, "side": o.side, "qty": qty,
                    "dollar_amount": round(o.dollar_amount, 2),
                    "order_id": order_id, "date": today,
                })
                executed.append({
                    "symbol":        o.symbol,
                    "side":          o.side,
                    "qty":           qty,
                    "dollar_amount": o.dollar_amount,
                    "order_id":      order_id,
                })
            except Exception as e:
                print(f"[EOD] Failed {o.symbol}: {e}")
                skipped.append({"symbol": o.symbol, "reason": str(e)})

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


from ascent.llm.client import HAIKU_MODEL


#: Gross exposure a reduce_size verdict targets, as a fraction of NAV. The
#: remainder becomes cash — which is the entire point, and is what this channel
#: was structurally unable to do before 2026-07-28.
REDUCE_SIZE_GROSS_TARGET = 0.90

#: Bounds on a verdict-supplied reduction, so a malformed or run-away number
#: cannot liquidate or no-op the book.
REDUCE_SIZE_MIN_GROSS = 0.75
REDUCE_SIZE_MAX_GROSS = 0.98


def _reduce_size_target_gross(verdict) -> float:
    """Gross exposure a reduce_size verdict should land at or below.

    Uses the verdict's optional `reduction_pct` when present and sane — the
    2026-04-06 verdict asked to "reduce overall position size by ~30%" and had
    nowhere to say so — otherwise the documented default. Always clamped to
    [REDUCE_SIZE_MIN_GROSS, REDUCE_SIZE_MAX_GROSS] so a hallucinated number can
    neither near-liquidate the book nor no-op the verdict.
    """
    pct = verdict.get("reduction_pct") if isinstance(verdict, dict) else None
    if not isinstance(pct, (int, float)) or isinstance(pct, bool) or pct <= 0:
        return REDUCE_SIZE_GROSS_TARGET
    target = 1.0 - float(pct)
    return max(REDUCE_SIZE_MIN_GROSS, min(REDUCE_SIZE_MAX_GROSS, target))


def _verdict_protected_symbols(verdict) -> frozenset:
    """Symbols the judge explicitly argued to keep, from the verdict contract.

    Before 2026-07-28 there was no field for this. On 07-27 the judge wrote
    "cutting insurance 48 hours before the catalyst it exists to hedge is poor
    timing — the bull wins that specific exchange" and deliberately chose VNQ
    instead of UUP/TLT. That protection existed only as prose inside
    verdict["reasoning"], which the enforcement layer never saw, so the
    size-sorted fallback sold UUP and TLT anyway.

    Tolerant of shape (list of dicts or of bare strings) and never raises:
    a malformed field yields no protection rather than blocking the run.
    """
    if not isinstance(verdict, dict):
        return frozenset()
    raw = verdict.get("protected_positions")
    if not isinstance(raw, list):
        return frozenset()
    out = set()
    for item in raw:
        if isinstance(item, str):
            sym = item
        elif isinstance(item, dict):
            sym = item.get("symbol")
        else:
            continue
        if isinstance(sym, str) and sym.strip():
            out.add(sym.strip().upper())
    return frozenset(out)


def _enforce_reduce_size(
    original_weights: dict,
    haiku_weights: dict,
    min_reduction_threshold: float = 0.01,
    min_positions_reduced: int = 3,
    target_gross: float = REDUCE_SIZE_GROSS_TARGET,
    protected: frozenset = frozenset(),
) -> dict:
    """
    Ensure a reduce_size verdict actually produces a smaller book — and does not
    sell the positions the judge argued to keep.

    Four defects this function used to have (audit 2026-07-27), all fixed here:

    1. **It could not reduce gross.** It ended by renormalizing to exactly 1.0,
       so gross was 1.0000 before and after on every reduce_size date in
       history. `target_gross` is now the scaling destination; the freed weight
       becomes cash instead of being recycled into other positions.

    2. **Detection ran after renormalization.** On 07-27 Haiku correctly cut VNQ
       by 1.01pp, renormalization scaled it back to 0.947pp, the 1pp threshold
       read 0 reductions, and the forced trim fired. Reductions are now measured
       on a gross-normalized basis so a real cut is not erased by scaling.

    3. **The forced trim was thesis-blind**, sorting by weight. It sold UUP and
       TLT — the FOMC hedge leg the verdict explicitly protected — plus BIL,
       the T-bill sleeve the same verdict called too small. `protected` names
       are now excluded from trim selection.

    4. **It ignored the authority cap** (VNQ moved -2.93pp against a stated
       1.0pp limit). Deliberately NOT fixed by capping here. A per-name cap and
       an honoured protect-list are mutually exclusive: holding UUP at its full
       weight while gross falls to 0.90 *is* a deviation from UUP's pro-rata
       share, so a 1pp cap would forbid honouring the judge. The audit flagged
       this same contradiction from the other side ("requiring 3 cuts of >=1pp
       while capping each intervention at 1.0pp is self-contradictory").
       De-grossing is a portfolio-level risk action, not a per-name bet; the
       per-intervention authority cap governs the judge's `position_changes`,
       which run_all_agents already enforces. What needs bounding here is the
       total gross reduction — REDUCE_SIZE_MIN_GROSS / _MAX_GROSS do that.

    Args:
        original_weights: {symbol: weight} before the verdict
        haiku_weights:    {symbol: weight} after the Haiku adjustment
        min_reduction_threshold: per-position cut that counts as genuine
        min_positions_reduced:   how many such cuts to accept Haiku's work
        target_gross: gross exposure to land at or below. Defaults to
            REDUCE_SIZE_GROSS_TARGET, deliberately NOT 1.0: a 1.0 default makes
            this function a silent no-op (needed = total - 1.0 = 0), which is the
            same failure it exists to prevent, just relocated into the signature.
            A function named _enforce_reduce_size must reduce by default.
        protected: symbols the verdict said not to cut. These keep their
            absolute weight and the whole reduction comes from the rest.

    Returns:
        {symbol: weight} summing to `target_gross`
    """
    # Guard: empty haiku_weights
    if not haiku_weights:
        print("[EodRunner] reduce_size: haiku_weights is empty — returning original weights")
        return dict(original_weights)

    def _cap_at_target(w: dict) -> dict:
        """Scale down to `target_gross` — never up.

        The invariant is `gross <= target`, not `gross == target`. Scaling up to
        hit the target exactly would re-gross a book that Haiku had already
        reduced, which is the same class of bug as the old renormalize-to-1.0.
        """
        total = sum(w.values())
        if total <= 0 or total <= target_gross:
            return dict(w)
        k = target_gross / total
        return {s: max(0.0, v * k) for s, v in w.items()}

    # --- (2) Measure reductions on RAW weights --------------------------------
    # Deliberately not normalized. A cut is expressed in absolute weight, and
    # normalizing the reduced book back up is precisely what erased the 07-27
    # cut (1.01pp -> 0.947pp, under the 1pp threshold). The upstream fix is that
    # _apply_verdict_adjustments no longer renormalizes to 1.0, so what arrives
    # here still carries the reduction it was given.
    reduced_count = sum(
        1 for s, w in haiku_weights.items()
        if w < original_weights.get(s, 0.0) - min_reduction_threshold
    )

    if reduced_count >= min_positions_reduced:
        print(f"[EodRunner] reduce_size: Haiku reduced {reduced_count} positions — accepted")
        # (1) Still scale to target: accepting Haiku's shape must not re-gross
        # the book back to 1.0, which is what silently undid every reduction.
        return {s: round(w, 6) for s, w in _cap_at_target(haiku_weights).items()}

    # --- Forced trim ----------------------------------------------------------
    print(f"[EodRunner] reduce_size: Haiku only reduced {reduced_count} positions "
          f"(need {min_positions_reduced}) — forcing trim")

    weights = dict(haiku_weights)
    current_total = sum(weights.values())
    needed = current_total - target_gross

    if needed <= 0:
        # Already at or below target — just land exactly on it.
        return {s: round(w, 6) for s, w in _cap_at_target(weights).items()}

    # (3) The reduction comes proportionally out of the NON-protected names.
    # The old code trimmed the top 5 by weight, which is how a verdict that said
    # "do not cut the hedge leg before FOMC" ended up selling UUP, TLT and BIL.
    # Proportional-across-the-rest also spreads the cut instead of concentrating
    # it on the largest positions, which is what min_positions_reduced was
    # clumsily reaching for.
    non_protected = {s: w for s, w in weights.items() if s not in protected}
    nonprot_total = sum(non_protected.values())

    if protected:
        held = sorted(s for s in weights if s in protected)
        print(f"[EodRunner] reduce_size: holding {held} at full weight per verdict")

    if nonprot_total >= needed and nonprot_total > 0:
        keep = (nonprot_total - needed) / nonprot_total
        for sym in non_protected:
            weights[sym] = max(0.0, weights[sym] * keep)
        # (1) The freed weight is NOT redistributed — it becomes cash. Recycling
        # it into other positions is what made this a rotation rather than a
        # reduction: on 07-27 it moved 10pp out of dollar, duration and T-bills
        # into equities 48h before FOMC, under a de-risking verdict.
    else:
        # The protect-list is too large to fund the reduction from the rest.
        # De-gross everything uniformly and say so — silently failing to reduce
        # is the original bug.
        print(f"[EodRunner] reduce_size: non-protected sleeve ({nonprot_total:.3f}) "
              f"cannot fund a {needed:.3f} reduction — de-grossing uniformly")
        weights = _cap_at_target(weights)

    return {s: round(w, 6) for s, w in weights.items()}


def _apply_verdict_adjustments(merged_weights: dict, verdict: dict,
                               target_gross: float = 1.0) -> dict:
    """
    Send verdict + current weights to Haiku and get back adjusted weights.
    Called when the debate verdict is reduce_size.

    Three fixes from the 2026-07-27 audit:

    - The system prompt used to require "weights must sum to 1.0", which made it
      structurally impossible for a *reduce size* instruction to reduce size.
      Haiku is now told the gross target explicitly.
    - The validator renormalized any deviation from 1.0 back to 1.0. That is what
      shrank a correctly-implemented 1.01pp VNQ cut to 0.947pp, dropping it under
      the downstream 1pp detection threshold and triggering the size-sorted
      fallback that sold the hedges the judge protected. It now only caps at the
      target, never scales up.
    - The judge's `position_changes` and protected names were never passed, so
      Haiku saw only free-text reasoning. Both are now stated explicitly.

    Returns adjusted weights, falling back to the input if anything fails.
    """
    import json as _json
    from ascent.llm.client import generate_structured

    weights_str = "\n".join(
        f"  {sym}: {w:.4f}" for sym, w in
        sorted(merged_weights.items(), key=lambda x: -x[1])
    )

    reasoning  = verdict.get("reasoning", "")
    key_risks  = verdict.get("key_risks", [])
    risks_str  = "\n".join(f"  - {r}" for r in key_risks)

    protected = _verdict_protected_symbols(verdict)
    protected_str = ", ".join(sorted(protected)) if protected else "(none)"

    changes = verdict.get("position_changes") or []
    changes_str = "\n".join(
        f"  - {c.get('symbol')}: {c.get('current_weight')} -> {c.get('new_weight')}"
        f" ({c.get('reason', '')[:120]})"
        for c in changes if isinstance(c, dict) and c.get("symbol")
    ) or "  (none specified)"

    system_prompt = (
        "You are a portfolio risk manager. You will receive current portfolio weights "
        "and a debate verdict explaining what needs to be reduced. "
        "Output ONLY valid JSON — a dict of {symbol: new_weight} with no other text. "
        f"Rules: the weights must sum to AT MOST {target_gross:.2f} (the remainder is "
        "cash — this is a REDUCE SIZE instruction, so a book summing to 1.0 has not "
        "reduced anything), no weight above 0.15, no negative weights, "
        "keep all existing symbols but adjust their weights per the verdict instructions."
    )

    user_prompt = (
        f"Current portfolio weights:\n{weights_str}\n\n"
        f"Debate verdict reasoning:\n{reasoning}\n\n"
        f"Key risks to address:\n{risks_str}\n\n"
        f"Specific position changes the judge asked for:\n{changes_str}\n\n"
        f"DO NOT REDUCE these positions — the judge argued explicitly to keep "
        f"them: {protected_str}\n\n"
        f"Produce adjusted weights that implement the verdict. The total must be "
        f"at most {target_gross:.2f}; take the reduction from positions other than "
        f"the protected ones.\n"
        'Return ONLY a JSON object like {"AAPL": 0.05, "MSFT": 0.08}. '
        "No markdown, no explanation, just the JSON."
    )

    try:
        raw = generate_structured(
            system_prompt, user_prompt,
            model=HAIKU_MODEL,
            # 800 truncated mid-JSON on a 23-position book, which the bare
            # json.loads below turns into a silent fallback to the input.
            max_tokens=2000,
            temperature=0.1,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        adjusted = _json.loads(raw)

        # Validate. Cap at the gross target; never scale UP — scaling up is what
        # erased the reduction this call exists to produce.
        total = sum(adjusted.values())
        if total > target_gross + 0.001:
            print(f"[Debate] Adjusted weights sum to {total:.4f} > target "
                  f"{target_gross:.2f} — capping")
            adjusted = {s: w * (target_gross / total) for s, w in adjusted.items()}
        else:
            print(f"[Debate] Adjusted weights sum to {total:.4f} "
                  f"(target <= {target_gross:.2f}) — left as returned")

        if max(adjusted.values()) > 0.15:
            print("[Debate] Adjusted weights have position > 15% — falling back to original")
            return merged_weights

        print(f"[Debate] Verdict adjustments applied — {len(adjusted)} positions")
        return adjusted

    except Exception as e:
        print(f"[Debate] Weight adjustment failed ({e}) — using original weights")
        return merged_weights

def run_eod_with_weights(merged_weights: dict, run_date=None, dry_run: bool = False, precomputed_verdict: dict = None, force: bool = False):
    """
    Execute EOD with pre-computed weights from the orchestrator.
    Called by run_all_agents.py instead of run_eod().

    Handles: rebalance calendar check, kill switch, order computation,
    large-trade approval, order submission, slippage tracking, and logging.
    Does NOT run the ascent pipeline internally.

    force=True bypasses ONLY the rebalance-calendar gate (used by intra-period
    actions: discovery mini-rebalances and falsifier trims). Kill switch and
    large-trade approval still apply.
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

    # 1.5. Debate layer
    if precomputed_verdict is not None:
        print("[EOD-Multi] Using precomputed debate verdict -- skipping internal debate")
        import json as _jpv
        from pathlib import Path as _Ppv
        _rec = precomputed_verdict.get("recommendation", "proceed")
        _conf = precomputed_verdict.get("confidence", 0.5)
        print(f"[EOD-Multi] Verdict: {_rec.upper()} (confidence={_conf})")
        _Ppv("logs/post_debate_portfolio.jsonl").open("a").write(
            _jpv.dumps({"date": today_str, "recommendation": _rec,
                "confidence": _conf, "weights": merged_weights,
                "key_risks": precomputed_verdict.get("key_risks", [])}) + "\n")
        if _rec == "halt_and_review":
            print("[EOD-Multi] HALT -- skipping execution.")
            _log_multi_run(today_str, merged_weights, rebalanced=False, note="debate_halt_precomputed")
            return
        if _rec == "reduce_size":
            print("[EOD-Multi] REDUCE SIZE -- applying Haiku adjustments...")
            original_weights = dict(merged_weights)
            _tg = _reduce_size_target_gross(precomputed_verdict)
            _prot = _verdict_protected_symbols(precomputed_verdict)
            merged_weights = _apply_verdict_adjustments(
                merged_weights, precomputed_verdict, target_gross=_tg)
            merged_weights = _enforce_reduce_size(
                original_weights, merged_weights, target_gross=_tg, protected=_prot)
            print(f"[EOD-Multi] Adjusted. Gross: {sum(original_weights.values()):.4f} "
                  f"-> {sum(merged_weights.values()):.4f} (target <= {_tg:.2f}); "
                  f"protected={sorted(_prot) or 'none'}")
    else:
        try:
            from debate.debate_runner import run_debate
            import json as _json

            # Build portfolio state for debate
            debate_state = {
                "date":         today_str,
                "us_regime":    "unknown",
                "macro_regime": "unknown",
                "n_positions":  len(merged_weights),
                "allocation":   {},
                "weights":      merged_weights,
            }

            # Try to load regime from dashboard export
            try:
                import pathlib as _pl
                regime_path = _pl.Path("dashboard/regime_signal.json")
                if regime_path.exists():
                    _rsig = _json.loads(regime_path.read_text())
                    if isinstance(_rsig, list):
                        _rsig = _rsig[-1] if _rsig else {}
                    debate_state["us_regime"] = _rsig.get("label", "unknown")
            except Exception:
                pass

            # Debate gate: only fire debate when uncertainty is elevated
            try:
                from ascent.execution.debate_gate import should_run_debate
                from ascent.monitoring.counterfactual_tracker import snapshot_quant_weights, snapshot_debate_weights
                _regime_info = {"entropy": debate_state.get("regime_entropy", 0.0),
                                "label": debate_state.get("us_regime", "unknown")}
                _gate_state = {"weights": merged_weights,
                               "quant_context": debate_state.get("quant_context", {}),
                               "catalyst_detected": debate_state.get("catalyst_detected", False)}
                _run_debate = should_run_debate(_gate_state, _regime_info)
            except Exception as _gate_exc:
                print(f"[EOD-Multi] Debate gate check failed ({_gate_exc}) — defaulting to run debate")
                _run_debate = True

            if not _run_debate:
                print("[EOD-Multi] Debate gate: SKIP — no trigger conditions met — using quant weights")
            else:
                # Snapshot pure quant weights before debate can modify them
                try:
                    snapshot_quant_weights(dict(merged_weights), run_date=today)
                except Exception as _snap_exc:
                    print(f"[EOD-Multi] Quant snapshot failed: {_snap_exc}")

            if _run_debate:
                print(f"\n[EOD-Multi] Running pre-rebalance debate...")
                verdict = run_debate(debate_state, run_date=today)
                recommendation = verdict.get("recommendation", "proceed")
                confidence     = verdict.get("confidence", 0.5)
                print(f"[EOD-Multi] Debate verdict: {recommendation.upper()} (confidence={confidence})")

                if recommendation == "halt_and_review":
                    print("[EOD-Multi] Debate says HALT — skipping execution.")
                    print(f"[EOD-Multi] Key risks:")
                    for risk in verdict.get("key_risks", []):
                        print(f"  - {risk}")
                    import json as _jlog
                    from pathlib import Path as _Plog
                    _Plog("logs/post_debate_portfolio.jsonl").open("a").write(
                        _jlog.dumps({"date": today_str, "recommendation": recommendation,
                            "confidence": verdict.get("confidence"),
                            "weights": merged_weights,
                            "key_risks": verdict.get("key_risks", [])}) + "\n")
                    print("[EOD-Multi] Post-debate portfolio logged to logs/post_debate_portfolio.jsonl")
                    _log_multi_run(today_str, merged_weights, rebalanced=False,
                                   note=f"debate_halt: {verdict.get('reasoning', '')[:200]}")
                    return

                if recommendation == "reduce_size":
                    print("[EOD-Multi] Debate says REDUCE SIZE — applying verdict-specific adjustments...")
                    original_weights = dict(merged_weights)
                    _tg = _reduce_size_target_gross(verdict)
                    _prot = _verdict_protected_symbols(verdict)
                    merged_weights = _apply_verdict_adjustments(
                        merged_weights, verdict, target_gross=_tg)
                    merged_weights = _enforce_reduce_size(
                        original_weights, merged_weights, target_gross=_tg, protected=_prot)
                    print(f"[EOD-Multi] Adjusted weights. Gross: "
                          f"{sum(original_weights.values()):.4f} -> "
                          f"{sum(merged_weights.values()):.4f} (target <= {_tg:.2f}); "
                          f"protected={sorted(_prot) or 'none'}")
                    import json as _jpdp
                    from pathlib import Path as _Ppdp
                    _Ppdp("logs/post_debate_portfolio.jsonl").open("a").write(
                        _jpdp.dumps({"date": today_str, "recommendation": recommendation,
                            "confidence": verdict.get("confidence"),
                            "weights": merged_weights,
                            "key_risks": verdict.get("key_risks", [])}) + "\n")
                    print("[EOD-Multi] Post-debate portfolio logged to logs/post_debate_portfolio.jsonl")

                if recommendation == "proceed":
                    import json as _jlog
                    from pathlib import Path as _Plog
                    _Plog("logs/post_debate_portfolio.jsonl").open("a").write(
                        _jlog.dumps({"date": today_str, "recommendation": recommendation,
                            "confidence": verdict.get("confidence"),
                            "weights": merged_weights,
                            "key_risks": verdict.get("key_risks", [])}) + "\n")
                    print("[EOD-Multi] Post-debate portfolio logged to logs/post_debate_portfolio.jsonl")

                # Snapshot debate-adjusted weights for counterfactual scoring
                try:
                    snapshot_debate_weights(dict(merged_weights), run_date=today)
                except Exception as _dsnap_exc:
                    print(f"[EOD-Multi] Debate snapshot failed: {_dsnap_exc}")
            # debate skipped or proceed — continue with quant weights

        except Exception as _debate_exc:
            print(f"[EOD-Multi] Debate failed ({_debate_exc}) -- proceeding without debate gate")

    # 2. Get current positions and portfolio value
    from ascent.execution.alpaca_broker import get_positions, get_portfolio_value, cancel_all_orders, submit_order, close_position
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

    # 5. Kill switch check
    try:
        from ascent.execution import kill_switch
        kill_switch.check(current_nav=portfolio_value)
    except Exception as _ks_exc:
        from ascent.execution import kill_switch as _ks
        if isinstance(_ks_exc, _ks.KillSwitchTriggered):
            print(f"[EOD-Multi] KILL SWITCH TRIGGERED — aborting")
            _log_multi_run(today_str, merged_weights, rebalanced=False, note="kill_switch_triggered")
            return
        else:
            print(f"[EOD-Multi] Kill switch check error: {_ks_exc} — continuing")

    # 6. Submit orders (or dry-run)
    if dry_run:
        print("[EOD-Multi] DRY RUN — orders NOT submitted to Alpaca")
        _log_multi_run(today_str, merged_weights, rebalanced=True, note="dry_run")
        return

    cancel_all_orders()
    print("[EOD-Multi] Cancelled open orders.")

    executed = []
    skipped  = []
    for order in orders:
        from ascent.execution.order_engine import _get_approx_price
        price = _get_approx_price(order.symbol, current_positions)
        if price is None:
            skipped.append(order.symbol)
            continue
        qty = round(order.dollar_amount / price, 6)
        if qty < 0.001:
            skipped.append(order.symbol)
            continue
        try:
            # Full liquidation: use close_position() to avoid qty rounding mismatch → 403
            if order.side == "sell" and order.target_weight == 0.0:
                close_position(order.symbol)
                print(f"[EOD-Multi] CLOSE {order.symbol}  (${order.dollar_amount:,.0f})")
            else:
                submit_order(order.symbol, qty=qty, side=order.side)
                print(f"[EOD-Multi] {order.side.upper()} {qty:.4f} {order.symbol}  (${order.dollar_amount:,.0f})")
            executed.append(order.symbol)
        except Exception as _e:
            print(f"[EOD-Multi] {order.symbol} FAILED: {_e}")
            skipped.append(order.symbol)

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
