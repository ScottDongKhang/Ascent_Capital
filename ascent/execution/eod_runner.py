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

# Task 8: large-trade approval threshold
LARGE_TRADE_THRESHOLD_PCT = 2.0  # % of portfolio NAV


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

        # Bug 3 fix: run_pipeline() returns 9 values — was only unpacking 6.
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
        ) = run_pipeline(
            live=True,
        )

        if target_weights_all is None or target_weights_all.empty:
            raise ValueError("Pipeline returned no target weights.")

        valid_rows  = target_weights_all[(target_weights_all > 0).any(axis=1)]
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
                # Bug 7 fix: log instead of silently swallowing
                print(f"[EOD] WARNING: regime signal extraction failed ({type(_re).__name__}: {_re}) — regime/posture will be null")

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
                    current_positions.get("weight", [0.0] * len(current_positions))
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

        # ── Task 8: Large-trade approval check ───────────────────────────────
        # Any individual order that moves more than LARGE_TRADE_THRESHOLD_PCT
        # of portfolio NAV requires human approval before execution.
        large_trades = []
        for order in orders:
            order_notional = order.dollar_amount  # already computed by order_engine
            order_pct = order_notional / portfolio_value * 100
            if order_pct >= LARGE_TRADE_THRESHOLD_PCT:
                large_trades.append({
                    "symbol":     order.symbol,
                    "side":       order.side,
                    "qty":        round(order.estimated_shares, 4),
                    "notional":   round(order_notional, 2),
                    "weight_pct": round(order_pct, 2),
                })

        if large_trades:
            try:
                from ascent.execution.approval_server import write_pending_trades, wait_for_approval_async
                print(f"[EOD] {len(large_trades)} trades exceed {LARGE_TRADE_THRESHOLD_PCT}% threshold — requesting approval")
                write_pending_trades(large_trades, run_date=today)
                result = wait_for_approval_async(large_trades)
                if result.status == "approved":
                    print("[EOD] Trades APPROVED — proceeding with execution")
                elif result.status == "rejected":
                    print("[EOD] Trades REJECTED — skipping execution")
                    log_run(
                        run_date=today,
                        run_type="rebalance",
                        regime_label=regime_label,
                        regime_confidence=regime_confidence,
                        posture=posture,
                        portfolio_value=portfolio_value,
                        target_weights=target_weights.to_dict(),
                        orders_executed=[],
                        orders_skipped=[{"symbol": t["symbol"], "reason": "rejected_by_approval_ui"} for t in large_trades],
                        notes=f"Execution rejected via approval UI for {len(large_trades)} large trades.",
                    )
                    return
                else:
                    # timeout or expired
                    print(f"[EOD] Approval {result.status.upper()} — cancelling trades")
                    log_run(
                        run_date=today,
                        run_type="rebalance",
                        regime_label=regime_label,
                        regime_confidence=regime_confidence,
                        posture=posture,
                        portfolio_value=portfolio_value,
                        target_weights=target_weights.to_dict(),
                        orders_executed=[],
                        orders_skipped=[{"symbol": t["symbol"], "reason": f"approval_{result.status}"} for t in large_trades],
                        notes=f"Approval {result.status} — trades cancelled.",
                    )
                    return
            except ImportError:
                # approval_server not installed — warn but don't block
                print(f"[EOD] WARNING: approval_server not available, skipping large-trade check for {len(large_trades)} trades")
            except Exception as e:
                print(f"[EOD] WARNING: approval check failed ({e}) — proceeding without approval gate")
        # ── End Task 8 ────────────────────────────────────────────────────────

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
                print(f"[EOD] {o.side.upper()} {qty:.4f} {o.symbol}  (${o.dollar_amount:,.0f})")
                executed.append({
                    "symbol":        o.symbol,
                    "side":          o.side,
                    "qty":           qty,
                    "dollar_amount": o.dollar_amount,
                    "order_id":      resp.get("id"),
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

def _apply_verdict_adjustments(merged_weights: dict, verdict: dict) -> dict:
    """
    Send verdict + current weights to Haiku and get back adjusted weights.
    Called when debate verdict is reduce_size.
    Returns adjusted weights dict, falls back to original if anything fails.
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

    system_prompt = (
        "You are a portfolio risk manager. You will receive current portfolio weights "
        "and a debate verdict explaining what needs to be reduced. "
        "Output ONLY valid JSON — a dict of {symbol: new_weight} with no other text. "
        "Rules: weights must sum to 1.0, no weight above 0.15, no negative weights, "
        "keep all existing symbols but adjust their weights per the verdict instructions."
    )

    user_prompt = (
        f"Current portfolio weights:\n{weights_str}\n\n"
        f"Debate verdict reasoning:\n{reasoning}\n\n"
        f"Key risks to address:\n{risks_str}\n\n"
        "Produce adjusted weights that implement the verdict recommendations. "
        "Return ONLY a JSON object like {{\"AAPL\": 0.05, \"MSFT\": 0.08, ...}}. "
        "No markdown, no explanation, just the JSON."
    )

    try:
        raw = generate_structured(
            system_prompt, user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.1,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        adjusted = _json.loads(raw)

        # Validate
        total = sum(adjusted.values())
        if abs(total - 1.0) > 0.001:
            print(f"[Debate] Adjusted weights sum to {total:.4f} — renormalizing")
            adjusted = {s: w / total for s, w in adjusted.items()}

        if max(adjusted.values()) > 0.15:
            print("[Debate] Adjusted weights have position > 15% — falling back to original")
            return merged_weights

        print(f"[Debate] Verdict adjustments applied — {len(adjusted)} positions")
        return adjusted

    except Exception as e:
        print(f"[Debate] Weight adjustment failed ({e}) — using original weights")
        return merged_weights

def run_eod_with_weights(merged_weights: dict, run_date=None, dry_run: bool = False, precomputed_verdict: dict = None):
    """
    Execute EOD with pre-computed weights from the orchestrator.
    Called by run_all_agents.py instead of run_eod().

    Handles: rebalance calendar check, kill switch, order computation,
    large-trade approval, order submission, slippage tracking, and logging.
    Does NOT run the ascent pipeline internally.
    """
    import time as _time
    from datetime import date as _date
    import pandas as _pd

    today = run_date or _date.today()
    today_str = today.isoformat() if hasattr(today, "isoformat") else str(today)

    print(f"\n[EOD-Multi] Running with orchestrator weights | {today_str}")
    print(f"[EOD-Multi] Positions: {len(merged_weights)}")

    # Resume check: if a previous run left pending approval state, try to resume it
    import json as _json_r
    from ascent.execution.approval_server import APPROVAL_PENDING_PATH as _APPROVAL_PENDING_PATH
    _pending_path = _APPROVAL_PENDING_PATH
    if _pending_path.exists():
        _pending_state = _json_r.loads(_pending_path.read_text())
        from datetime import datetime as _dt_r
        _expires = _dt_r.fromisoformat(_pending_state["expires_at"])
        if _expires > _dt_r.now():
            print(f"[EOD-Multi] Resuming pending approval from previous run "
                  f"({len(_pending_state.get('trades', []))} trades)")
            from ascent.execution.approval_server import wait_for_approval_async
            _resume_result = wait_for_approval_async(
                pending_trades=[], resume=True, pending=_pending_state
            )
            if _resume_result.status == "approved":
                print("[EOD-Multi] Resumed approval: APPROVED — submitting persisted trades")
                from ascent.execution.alpaca_broker import (
                    get_positions as _get_pos,
                    get_portfolio_value as _get_pv,
                    cancel_all_orders as _cancel,
                    submit_order as _submit,
                )
                _cancel()
                for _t in _pending_state.get("trades", []):
                    try:
                        _submit(_t["symbol"], qty=_t["qty"], side=_t["side"])
                        print(f"[EOD-Multi] (Resume) {_t['side'].upper()} {_t['qty']} {_t['symbol']}")
                    except Exception as _re:
                        print(f"[EOD-Multi] (Resume) {_t['symbol']} FAILED: {_re}")
                _log_multi_run(today_str, merged_weights, rebalanced=True, note="resumed_approval")
                return
            else:
                print(f"[EOD-Multi] Resumed approval: {_resume_result.status.upper()} "
                      "— clearing and proceeding with fresh run")
                _pending_path.unlink(missing_ok=True)
        else:
            print("[EOD-Multi] Stale approval_pending.json found (expired) — clearing")
            _pending_path.unlink(missing_ok=True)

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

    if not is_rebalance:
        print("[EOD-Multi] Not a rebalance day — logging only")
        _log_multi_run(today_str, merged_weights, rebalanced=False)
        return

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
            merged_weights = _apply_verdict_adjustments(merged_weights, precomputed_verdict)
            print(f"[EOD-Multi] Adjusted. Total: {sum(merged_weights.values()):.4f}")
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
                    debate_state["us_regime"] = _rsig.get("label", "unknown")
            except Exception:
                pass

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
                merged_weights = _apply_verdict_adjustments(merged_weights, verdict)
                print(f"[EOD-Multi] Adjusted weights. Total: {sum(merged_weights.values()):.4f}")
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
            # proceed — continue as normal

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
            _dv = _prices_raw.pivot_table(
                index="date", columns="symbol", values="dollar_volume", aggfunc="last"
            )
            _cost_features = extract_cost_features({"dollar_volume": _dv})
            print(f"[EOD-Multi] Cost features loaded: {len(_cost_features.get('dollar_vol_21d', {}))} symbols")
    except Exception as _cf_exc:
        print(f"[EOD-Multi] Cost features unavailable ({_cf_exc}) — cost filtering inactive")

    orders, diff_df = compute_orders(target_weights, current_positions, portfolio_value,
                                     features=_cost_features or None)

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

    # 6. Large-trade approval check (reuse Phase A threshold)
    large_trades = []
    for order in orders:
        order_pct = order.dollar_amount / portfolio_value * 100
        if order_pct >= LARGE_TRADE_THRESHOLD_PCT:
            large_trades.append({
                "symbol":     order.symbol,
                "side":       order.side,
                "qty":        round(order.estimated_shares, 4),
                "notional":   round(order.dollar_amount, 2),
                "weight_pct": round(order_pct, 2),
            })

    if large_trades and not dry_run:
        try:
            from ascent.execution.approval_server import write_pending_trades, wait_for_approval_async
            print(f"[EOD-Multi] {len(large_trades)} trades exceed {LARGE_TRADE_THRESHOLD_PCT}% — requesting approval")
            write_pending_trades(large_trades, run_date=today_str)
            result = wait_for_approval_async(large_trades)
            if result.status == "approved":
                print("[EOD-Multi] Trades APPROVED — proceeding")
            else:
                print(f"[EOD-Multi] Trades {result.status.upper()} — aborting execution")
                _log_multi_run(today_str, merged_weights, rebalanced=False, note=f"approval_{result.status}")
                return
        except ImportError:
            print("[EOD-Multi] WARNING: approval_server not available — skipping approval gate")

    # 7. Submit orders (or dry-run)
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
