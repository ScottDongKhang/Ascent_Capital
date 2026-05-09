"""
run_all_agents.py
Top-level daily runner for the Ascent Capital multi-agent platform.

Non-rebalance day:  agents → orchestrator → write weights → log (no debate, no execution)
Rebalance day:      agents → orchestrator → write weights → debate → execute via eod_runner

Usage:
    python3 run_all_agents.py                # live execution
    python3 run_all_agents.py --dry-run      # no order submission
"""

import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from ascent.data.store.parquet import has_data, load_parquet
from ascent.portfolio.optimizer import SectorDataError


SECTOR_OVERRIDE_LOG  = Path("logs/sector_override.jsonl")
HALT_STATE_PATH     = Path("execution/halt_state.json")
HALT_OVERRIDE_PATH  = Path("execution/halt_override.json")
REGIME_SIGNAL_PATH  = Path("dashboard/regime_signal.json")
REGIME_STALE_DAYS   = 5


def _is_regime_stale() -> bool:
    """Return True if regime_signal.json is missing, is the old list schema, or last_refit_date > 5 days ago."""
    if not REGIME_SIGNAL_PATH.exists():
        return True
    try:
        sig = json.loads(REGIME_SIGNAL_PATH.read_text())
        if isinstance(sig, list):
            return True  # old list schema — trigger migration
        date_str = sig.get("last_refit_date") or sig.get("as_of") or ""
        if not date_str:
            return True
        last = date.fromisoformat(date_str[:10])
        return (date.today() - last).days > REGIME_STALE_DAYS
    except Exception as e:
        print(f"[Runner] Regime staleness check failed ({type(e).__name__}: {e}) — treating as stale")
        return True


def _refresh_regime():
    """Trigger a regime refit and write updated regime_signal.json."""
    print("[Runner] Regime signal stale — triggering refit")
    try:
        from ascent.data.store.parquet import load_parquet as _load_parquet
        from ascent.config.settings import get_config as _get_config
        from ascent.regime.engine import RegimeEngine

        cfg = _get_config()
        prices_long = _load_parquet("prices_live")
        prices_wide = prices_long.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )

        if "SPY" not in prices_wide.columns:
            print("[Runner] SPY not in prices cache — cannot refit regime")
            return

        spy_prices = prices_wide["SPY"].dropna()
        engine = RegimeEngine(config=cfg.regime.to_engine_dict())
        engine.fit(spy_prices, run_model_selection=False)

        # Get current label from signal series
        sig_series = engine.get_signal_series()
        if sig_series is not None and not sig_series.empty and "label" in sig_series.columns:
            label = str(sig_series["label"].iloc[-1])
            # Convert enum value if needed (e.g. RegimeLabel.calm_bull → "calm_bull")
            if "." in label:
                label = label.split(".")[-1]
        else:
            label = "unknown"

        # Preserve old series if it exists
        old_series = []
        if REGIME_SIGNAL_PATH.exists():
            try:
                old_data = json.loads(REGIME_SIGNAL_PATH.read_text())
                if isinstance(old_data, list):
                    old_series = old_data
                elif isinstance(old_data, dict):
                    old_series = old_data.get("series", [])
            except Exception:
                pass

        # Write hybrid schema
        REGIME_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        sig = {
            "regime": label,
            "label": label,
            "as_of": date.today().isoformat(),
            "last_refit_date": date.today().isoformat(),
            "series": old_series,
        }
        REGIME_SIGNAL_PATH.write_text(json.dumps(sig, indent=2))
        print(f"[Runner] Regime refreshed → {label}")

    except FileNotFoundError:
        print("[Runner] prices_live not in cache — cannot refit regime")
    except Exception as e:
        print(f"[Runner] Regime refit failed: {e}")


def validate_sector_data(symbols: list, skip: bool = False) -> None:
    """
    Validates profiles.parquet exists and covers >= 80% of the US equities universe.
    Called once at startup before agents are spawned.
    Raises SectorDataError if coverage is insufficient.
    Pass skip=True (--skip-sector-check flag) to bypass with audit log entry.
    """
    import pandas as pd

    if skip:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "sector_check_skipped",
            "required_reason": "see CLI flag --skip-sector-check",
        }
        SECTOR_OVERRIDE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SECTOR_OVERRIDE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print("[Startup] Sector check SKIPPED — override logged to logs/sector_override.jsonl")
        return

    if not has_data("profiles"):
        raise SectorDataError(
            "profiles.parquet missing. Regenerate with:\n"
            "  .venv/bin/python -m ascent.data.ingest.profiles\n"
            "Or bypass with --skip-sector-check (override is logged)."
        )

    profiles = load_parquet("profiles")
    known = set(profiles["symbol"].dropna())

    unknown_sectors = profiles[
        profiles["sector"].isna() | profiles["sector"].isin(["Unknown", "unknown", ""])
    ]["symbol"].tolist()

    missing_from_profiles = [s for s in symbols if s not in known]
    total_unknown = len(set(missing_from_profiles + unknown_sectors))
    coverage = 1.0 - total_unknown / len(symbols) if symbols else 1.0

    if coverage < 0.80:
        raise SectorDataError(
            f"Sector coverage {coverage:.1%} < 80% threshold.\n"
            f"Missing from profiles: {missing_from_profiles[:20]}"
            f"{'...' if len(missing_from_profiles) > 20 else ''}\n"
            f"Unknown sectors: {unknown_sectors[:10]}"
            f"{'...' if len(unknown_sectors) > 10 else ''}\n"
            "Regenerate profiles.parquet or use --skip-sector-check (override is logged)."
        )

    print(f"[Startup] Sector data valid — coverage {coverage:.1%} ({len(known)} symbols in profiles)")


def check_halt_state(today=None) -> bool:
    """
    Returns True if execution may proceed, False if halted.

    Halt is cleared only when a valid halt_override.json is present with
    override_date >= halt_date. Both files are deleted on successful clear.
    Agents and orchestrator still run during a halt — only execution is blocked.
    """
    from datetime import date as _date
    today = today or _date.today()

    if not HALT_STATE_PATH.exists():
        return True

    halt = json.loads(HALT_STATE_PATH.read_text())

    if not halt.get("requires_override", True):
        HALT_STATE_PATH.unlink(missing_ok=True)
        return True

    if not HALT_OVERRIDE_PATH.exists():
        print(
            f"[HALT] System halted since {halt['halt_date']}: {halt.get('reason', '')}\n"
            f"[HALT] Create execution/halt_override.json to resume trading.\n"
            f"[HALT] See verdict: {halt.get('verdict_path', 'outputs/debate_log/')}"
        )
        return False

    override = json.loads(HALT_OVERRIDE_PATH.read_text())

    if override.get("override_date", "") < halt.get("halt_date", ""):
        print(
            f"[HALT] Override date {override['override_date']} predates "
            f"halt date {halt['halt_date']} — invalid override. Recreate the file."
        )
        return False

    # Valid override — clear both files
    print(f"[HALT] Override accepted by {override.get('override_by', 'unknown')} — halt cleared. "
          "NOTE: today's debate may still issue a new halt.")
    HALT_STATE_PATH.unlink(missing_ok=True)
    HALT_OVERRIDE_PATH.unlink(missing_ok=True)
    return True


def main():
    dry_run             = "--dry-run" in sys.argv
    skip_sector_check   = "--skip-sector-check" in sys.argv
    today               = date.today()

    # ── Startup validation: sector data must be present before agents spawn ───
    from ascent.config.settings import UniverseConfig, get_config
    us_symbols = UniverseConfig().symbols
    validate_sector_data(us_symbols, skip=skip_sector_check)

    # ── B3: Auto-refresh stale regime signal ──────────────────────────────────
    if _is_regime_stale():
        _refresh_regime()

    print(f"\n{'#'*60}")
    print(f"# ASCENT CAPITAL — Multi-Agent Daily Run")
    print(f"# Date:  {today}")
    print(f"# Mode:  {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'#'*60}\n")

    # ── Step 0: Centralized data ingestion — runs before all agents ──────────
    # Fetches all symbols (all universes) in one parallel pass. Agents read
    # from cache instead of calling yfinance individually. If the hub fails,
    # agents fall back to their own fetches — no abort.
    from ascent.data.hub import run_hub
    cfg = get_config()
    hub_manifest = run_hub(
        start_date=cfg.backtest.start_date,
        end_date=today.isoformat(),
    )
    if hub_manifest.get("status") != "ok":
        print(f"[Hub] WARNING: hub failed ({hub_manifest.get('error', 'unknown')}) "
              "— agents will fetch data individually")

    # ── Import agents (lazy, after startup validation) ───────────────────────
    from agents.us_equities_agent import run_us_equities_agent
    from agents.macro_agent import run_macro_agent
    from orchestrator.central_intelligence import run_orchestrator
    from ascent.monitoring.skill_tracker import export_skill_scores
    from ascent.monitoring.forward_pnl_tracker import run_forward_pnl_cycle
    from ascent.monitoring.pre_rebalance_checklist import run_checklist

    _HAS_INTERNATIONAL = False
    _HAS_ALTERNATIVES = False
    try:
        from agents.international_agent import run_international_agent
        _HAS_INTERNATIONAL = True
    except ImportError:
        pass

    try:
        from agents.alternatives_agent import run_alternatives_agent
        _HAS_ALTERNATIVES = True
    except ImportError:
        pass

    # ── Step 1: Run all agents in parallel ───────────────────────────────────
    agent_tasks = [
        ("us_equities", run_us_equities_agent),
        ("macro",       run_macro_agent),
    ]
    if _HAS_INTERNATIONAL:
        agent_tasks.append(("international", run_international_agent))
    if _HAS_ALTERNATIVES:
        agent_tasks.append(("alternatives", run_alternatives_agent))

    agent_outputs = []
    with ThreadPoolExecutor(max_workers=len(agent_tasks)) as executor:
        futures = {
            executor.submit(fn, dry_run, today): name
            for name, fn in agent_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                agent_outputs.append(future.result())
            except Exception as e:
                print(f"[Runner] {name} agent FAILED: {e}")

    if not agent_outputs:
        print("[Runner] No agent outputs — aborting")
        return

    # ── Step 1b: Pre-rebalance checklist (rebalance days only) ───────────────
    from pathlib import Path as _Path
    import pandas as _pd
    cal_path     = _Path("rebalance_calendar.csv")
    is_rebalance = False
    if cal_path.exists():
        try:
            _cal         = _pd.read_csv(cal_path)
            is_rebalance = today.isoformat() in _cal["rebalance_date"].values
        except Exception:
            pass

    if is_rebalance:
        try:
            held_symbols = []
            try:
                from ascent.execution.alpaca_broker import get_positions
                pos = get_positions()
                if not pos.empty and "symbol" in pos.columns:
                    held_symbols = list(pos["symbol"])
            except Exception:
                pass

            checklist = run_checklist(
                agent_outputs=agent_outputs,
                held_symbols=held_symbols,
            )
            if not checklist.passed:
                print("[Runner] ✗ Checklist has blocking failures — aborting execution")
                print("[Runner] Review logs/checklist_log.jsonl for details")
                return
        except Exception as e:
            print(f"[Runner] Checklist failed ({e}) — proceeding anyway")
    # ── Steps 2/3/4: Sequential pipeline — DO NOT parallelize ───────────────
    # Each step depends on the previous: PnL log → skill scores → orchestrator.
    # run_forward_pnl_cycle writes today's NAV to the PnL log.
    # export_skill_scores reads that log to compute the 63-day rolling Sharpe.
    # run_orchestrator reads the fresh Sharpe to weight capital allocation.
    try:
        run_forward_pnl_cycle(agent_outputs, today=today)
    except Exception as e:
        print(f"[Runner] Forward PnL cycle failed: {e} — continuing")

    try:
        export_skill_scores()
    except Exception as e:
        print(f"[Runner] Skill score update failed: {e} — continuing with stale scores")

    # Score counterfactuals where 10 days have passed
    try:
        from ascent.monitoring.counterfactual_tracker import score_pending_counterfactuals
        n_scored = score_pending_counterfactuals()
        if n_scored > 0:
            print(f"[Runner] Scored {n_scored} counterfactual(s)")
    except Exception as e:
        print(f"[Runner] Counterfactual scoring failed: {type(e).__name__}: {e}")

    # Score pending verdicts (runs daily, NOP if no verdicts old enough)
    try:
        from debate.outcome_tracker import score_pending_verdicts
        n_scored = score_pending_verdicts()
        if n_scored:
            print(f"[OutcomeTracker] Scored {n_scored} verdict(s)")
    except Exception as _oe:
        print(f"[OutcomeTracker] Scoring skipped: {_oe}")
    try:
        from memory.reflection_agent import reflect_on_new_outcomes
        n_reflected = reflect_on_new_outcomes()
        if n_reflected:
            print(f"[Reflection] Wrote {n_reflected} new lesson(s) to memory/reflections.jsonl")
    except Exception as _re:
        print(f"[Reflection] Skipped: {_re}")

    # Daily shadow promotion check (only acts on expired shadows)
    try:
        from ascent.research.shadow_promoter import run_shadow_promotion
        n_promoted = run_shadow_promotion()
        if n_promoted > 0:
            print(f"[Runner] Shadow promoter: {n_promoted} config(s) promoted to live")
    except Exception as e:
        print(f"[Runner] Shadow promotion failed: {type(e).__name__}: {e}")

    # Self-improve: runs on Sundays with current regime
    try:
        import calendar as _cal
        from datetime import date as _date
        if _date.today().weekday() == 6:  # Sunday = 6
            from ascent.research.self_improve import run_self_improve
            _current_regime = None
            try:
                import json as _rj
                from pathlib import Path as _rp
                _rsig_path = _rp("dashboard/regime_signal.json")
                if _rsig_path.exists():
                    _rsig = _rj.loads(_rsig_path.read_text())
                    if isinstance(_rsig, list):
                        _rsig = _rsig[-1] if _rsig else {}
                    _current_regime = str(_rsig.get("label", "")).lower() or None
            except Exception:
                pass
            print(f"[Runner] Running self-improve (regime={_current_regime})")
            run_self_improve(current_regime=_current_regime)

            # Slippage IC feedback — runs alongside self-improve on Sundays
            try:
                from ascent.monitoring.slippage_ic_feedback import run_slippage_ic_feedback
                _slip_metrics = run_slippage_ic_feedback(lookback_days=90)
                print(f"[SlippageIC] drag={_slip_metrics['slippage_ic_drag']:.4f} "
                      f"gross_ic={_slip_metrics['gross_ic']:.4f} "
                      f"net_ic={_slip_metrics['net_ic']:.4f} "
                      f"fills={_slip_metrics['n_fills']}")
            except Exception as _se:
                print(f"[SlippageIC] Feedback skipped: {_se}")
    except Exception as e:
        print(f"[Runner] Self-improve failed: {type(e).__name__}: {e}")

    # ── Step 5: Run orchestrator (reads fresh skill scores written above) ─────
    merged_weights = run_orchestrator(agent_outputs)

    if not merged_weights:
        print("[Runner] Orchestrator returned empty weights — aborting execution")
        return

    # ── Step 5b: Apply Phase 4 hedge overlay ─────────────────────────────────
    try:
        from ascent.portfolio.hedge_overlay import apply_hedge_overlay
        import json as _json

        _hedge_regime = None
        for _ao in agent_outputs:
            if _ao.agent_id == "us_equities" and _ao.regime_signal is not None:
                _hedge_regime = _ao.regime_signal
                break
        if _hedge_regime is None:
            for _ao in agent_outputs:
                if _ao.regime_signal is not None:
                    _hedge_regime = _ao.regime_signal
                    break

        merged_weights, _hedge_meta = apply_hedge_overlay(merged_weights, _hedge_regime)

        if _hedge_meta["hedge_weight"] > 0:
            print(f"[Hedge] Overlay applied — regime={_hedge_meta['regime_label']} "
                  f"confidence={_hedge_meta['confidence']:.2f} "
                  f"VIXY={_hedge_meta['vixy_after']:.1%}")
        else:
            print(f"[Hedge] No overlay — regime={_hedge_meta['regime_label']} "
                  f"(hedge_weight=0)")

        # Append to hedge log
        _hedge_log_path = Path("logs/hedge_log.jsonl")
        _hedge_log_path.parent.mkdir(parents=True, exist_ok=True)
        _hedge_entry = {"date": today.isoformat(), **_hedge_meta}
        with open(_hedge_log_path, "a") as _hf:
            _hf.write(_json.dumps(_hedge_entry) + "\n")

    except Exception as _hedge_e:
        print(f"[Hedge] Overlay skipped: {_hedge_e}")

    # ── Step 6: Write merged weights to file ──────────────────────────────────
    weights_path = Path("execution/merged_weights.json")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "date":         today.isoformat(),
        "weights":      merged_weights,
        "agents":       [ao.agent_id for ao in agent_outputs],
        "generated_at": datetime.now().isoformat(),
    }
    with open(weights_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Runner] Merged weights written to {weights_path}")
    print(f"[Runner] {len(merged_weights)} positions, total weight: {sum(merged_weights.values()):.4f}")

    # ── Non-rebalance day: stop here ──────────────────────────────────────────
    if not is_rebalance:
        print("[Runner] Non-rebalance day — weights updated, no debate, no execution.")
        try:
            _log_holdings(today)
        except Exception as e:
            print(f"[Runner] Holdings log skipped: {e}")
        _log_run(today, merged_weights, agent_outputs, dry_run)
        return

    # ── Rebalance day: check for active halt before debating ─────────────────
    if not check_halt_state(today=today):
        print("[Runner] Halted — agents ran, weights updated, execution skipped.")
        print("[Runner] Create execution/halt_override.json to resume.")
        try:
            _log_holdings(today)
        except Exception as e:
            print(f"[Runner] Holdings log skipped: {e}")
        _log_run(today, merged_weights, agent_outputs, dry_run)
        return

    # ── Rebalance day: debate → execute ───────────────────────────────────────
    print(f"\n[Runner] Rebalance day — running debate layer...")
    verdict = None
    try:
        from debate.debate_runner import run_debate
        from ascent.execution.debate_gate import should_run_debate
        import json as _json
        from pathlib import Path as _Path
        _regime_path  = _Path("dashboard/regime_signal.json")
        _saved_regime = "unknown"
        _regime_entropy = 0.0
        try:
            _rdata        = _json.loads(_regime_path.read_text())
            _sig = (_rdata[-1] if (isinstance(_rdata, list) and _rdata) else _rdata) or {}
            _saved_regime   = _sig.get("label", "unknown")
            _regime_entropy = float(_sig.get("entropy", 0.0) or 0.0)
        except Exception:
            pass
        # TODO: wire orchestrator_result.allocation when central_intelligence exposes it
        _base_alloc = {"us_equities": 0.60, "macro": 0.15, "international": 0.15, "alternatives": 0.10}
        _orch_alloc = merged_weights.get("allocation") if isinstance(merged_weights, dict) else None
        portfolio_state = {
            "date":         today.isoformat(),
            "us_regime":    next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "us_equities" and ao.regime_signal), _saved_regime),
            "macro_regime": next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "macro" and ao.regime_signal), "unknown"),
            "n_positions":  len(merged_weights),
            "allocation":   _orch_alloc or {ao.agent_id: round(_base_alloc.get(ao.agent_id, 0.0), 2)
                            for ao in agent_outputs},
            "weights":      merged_weights,
        }
        _regime_dict = {"entropy": _regime_entropy, "label": _saved_regime}
        if not should_run_debate(portfolio_state, _regime_dict):
            print("[Runner] Debate gate: no trigger — proceeding to execution without debate")
            verdict = {}
        else:
            verdict = run_debate(portfolio_state, run_date=today) or {}

        if verdict.get("recommendation") == "halt_and_review":
            print("[Runner] DEBATE VERDICT: halt_and_review — skipping execution")
            print("[Runner] Review at outputs/debate_log/")
            try:
                _log_holdings(today)
            except Exception as e:
                print(f"[Runner] Holdings log skipped: {e}")
            _log_run(today, merged_weights, agent_outputs, dry_run)
            return

    except Exception as e:
        print(f"[Runner] Debate failed ({e}) — proceeding to execution anyway")

    # ── Step 5: Execute via eod_runner ────────────────────────────────────────
    if dry_run:
        print(f"\n[Runner] DRY RUN — would submit {len(merged_weights)} positions:")
        for sym, w in sorted(merged_weights.items(), key=lambda x: -x[1])[:15]:
            print(f"  {sym}: {w:.2%}")
    else:
        print("[Runner] LIVE MODE — calling eod_runner with merged weights")
        try:
            from ascent.execution.eod_runner import run_eod_with_weights
            run_eod_with_weights(merged_weights, run_date=today, dry_run=False, precomputed_verdict=verdict)
        except ImportError:
            print("[Runner] WARNING: run_eod_with_weights not yet available — weights file written only")
        except Exception as e:
            print(f"[Runner] Execution failed: {e}")

    # ── Step 6: Log the run ───────────────────────────────────────────────────
    try:
        _log_holdings(today)
    except Exception as e:
        print(f"[Runner] Holdings log skipped: {e}")
    _log_run(today, merged_weights, agent_outputs, dry_run)


def _log_run(today, merged_weights, agent_outputs, dry_run):
    def _regime_str(val):
        if val is None:
            return "unknown"
        return str(val).split(".")[-1].lower() if "." in str(val) else str(val)

    run_log = {
        "date":             today.isoformat(),
        "weights":          merged_weights,
        "agents": {
            ao.agent_id: {
                "n_positions": ao.n_positions,
                "regime":      _regime_str(ao.regime_signal),
            }
            for ao in agent_outputs
        },
        "allocation": {
            ao.agent_id: round(
                {"us_equities": 0.60, "macro": 0.15, "international": 0.15, "alternatives": 0.10}.get(ao.agent_id, 0.0), 2
            )
            for ao in agent_outputs
        },
        "merged_positions": len(merged_weights),
        "mode":             "dry_run" if dry_run else "live",
        "timestamp":        datetime.now().isoformat(),
    }

    log_path = Path("logs/multi_agent_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(run_log) + "\n")

    print(f"\n[Runner] Run logged to {log_path}")
    print(f"[Runner] Done.\n")


def _log_holdings(today):
    log_path = Path("logs/holdings_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ascent.execution.alpaca_broker import get_positions, get_account
        pos = get_positions()
        acct = get_account()
        equity = float(acct.get("equity", 0))

        positions = []
        if not pos.empty:
            for _, row in pos.sort_values("market_value", ascending=False).iterrows():
                positions.append({
                    "symbol":        row["symbol"],
                    "qty":           round(float(row["qty"]), 4),
                    "market_value":  round(float(row["market_value"]), 2),
                    "current_price": round(float(row["current_price"]), 4),
                    "weight":        round(float(row["weight"]), 4),
                })

        # Compute day_ret and spy_ret from attribution (position × market return),
        # not from Alpaca equity/last_equity which diverges due to cash and paper-trading mechanics.
        day_ret = 0.0
        spy_ret = 0.0
        if positions:
            try:
                from ascent.monitoring.attribution import run_attribution
                attr = run_attribution(positions, today)
                if attr:
                    day_ret = attr.get("portfolio_return", 0.0)
                    spy_ret = attr.get("spy_return", 0.0)
            except Exception as e:
                print(f"[Runner] Attribution failed ({e})")

        entry = {
            "date":           today.isoformat(),
            "timestamp":      datetime.now().isoformat(),
            "equity":         round(equity, 2),
            "cash":           round(float(acct.get("cash", 0)), 2),
            "day_return":     round(day_ret, 6),
            "spy_return":     round(spy_ret, 6),
            "alpha_vs_spy":   round(day_ret - spy_ret, 6),
            "n_positions":    len(positions),
            "positions":      positions,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        sign = "+" if day_ret >= spy_ret else "-"
        print(f"[Runner] Holdings logged — equity ${equity:,.2f} | "
              f"portfolio {day_ret:+.2%} vs SPY {spy_ret:+.2%} ({sign})")
    except Exception as e:
        print(f"[Runner] Holdings log skipped ({e})")


if __name__ == "__main__":
    main()
