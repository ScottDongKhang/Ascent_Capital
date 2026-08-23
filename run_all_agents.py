"""
run_all_agents.py
Top-level daily runner for the Ascent Capital multi-agent platform.

Non-rebalance day:  agent -> orchestrator -> write weights -> log (no execution)
Rebalance day:      agent -> orchestrator -> write weights -> execute via eod_runner

The AI PM / debate / falsifier / earned-authority / dormant-agent layer was removed
2026-08-23 (noise-layer cut, measured negative-or-insignificant on every axis — see
CLAUDE.md integrity constraint #5). This is the core-skeleton rewrite: one agent
(us_equities), one orchestrator pass, one execution path.

Usage:
    python3 run_all_agents.py                # live execution
    python3 run_all_agents.py --dry-run      # no order submission
"""

import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from ascent.data.store.parquet import has_data, load_parquet
from ascent.utils.market_time import market_today
from ascent.portfolio.optimizer import SectorDataError

from memory.regime_memory import log_episode, update_outcomes

try:
    from compliance.audit_trail import record_event as _audit
except Exception:
    def _audit(event_type, payload):  # type: ignore[misc]
        pass


SECTOR_OVERRIDE_LOG  = Path("logs/sector_override.jsonl")
HALT_STATE_PATH      = Path("execution/halt_state.json")
HALT_OVERRIDE_PATH   = Path("execution/halt_override.json")
REGIME_SIGNAL_PATH   = Path("dashboard/regime_signal.json")
REGIME_STALE_DAYS    = 5

# Catch-up guard (W3 item 5): default staleness threshold, in NYSE trading
# days, beyond which the daily run refuses to auto-execute without an
# explicit --catch-up flag. Mutated only by _catch_up_guard()/main() to
# record whether this run is a catch-up recovery, so _log_run can tag the
# multi_agent_run.jsonl entry accordingly.
CATCH_UP_STALE_TRADING_DAYS = 3
_CATCH_UP_STATE = {"active": False, "missed_dates": []}

LONG_SHORT_ENABLED = False  # 130/30 — enable after ≥30 paper rebalances (~August 2026)

# One-time: seed meta-learner posteriors from existing IC log if posteriors don't exist yet
if not Path("data_cache/sleeve_posteriors.json").exists():
    try:
        from ascent.alpha.meta_learner import SleeveMetaLearner as _BootstrapML
        _n = _BootstrapML().seed_from_ic_log()
        if _n > 0:
            print(f"[Startup] Meta-learner: seeded from {_n} IC log entries")
    except Exception as _seed_e:
        print(f"[Startup] Meta-learner seed failed: {_seed_e}")


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
        _audit("halt_triggered", {
            "halt_date":    halt.get("halt_date"),
            "reason":       halt.get("reason", ""),
            "verdict_path": halt.get("verdict_path", "outputs/debate_log/"),
            "date":         today.isoformat(),
        })
        return False

    override = json.loads(HALT_OVERRIDE_PATH.read_text())

    if override.get("override_date", "") < halt.get("halt_date", ""):
        print(
            f"[HALT] Override date {override['override_date']} predates "
            f"halt date {halt['halt_date']} — invalid override. Recreate the file."
        )
        _audit("halt_triggered", {
            "halt_date":     halt.get("halt_date"),
            "reason":        halt.get("reason", ""),
            "verdict_path":  halt.get("verdict_path", "outputs/debate_log/"),
            "date":          today.isoformat(),
            "invalid_override_date": override.get("override_date", ""),
        })
        return False

    # Valid override — clear both files
    print(f"[HALT] Override accepted by {override.get('override_by', 'unknown')} — halt cleared. "
          "NOTE: today's debate may still issue a new halt.")
    _audit("halt_overridden", {
        "halt_date":     halt.get("halt_date"),
        "reason":        halt.get("reason", ""),
        "override_date": override.get("override_date", ""),
        "override_by":   override.get("override_by", "unknown"),
        "date":          today.isoformat(),
    })
    HALT_STATE_PATH.unlink(missing_ok=True)
    HALT_OVERRIDE_PATH.unlink(missing_ok=True)
    return True


def _get_current_regime() -> str:
    """Current regime label off dashboard/regime_signal.json ('unknown' on any
    failure).

    Module level, not nested inside main(): it used to be defined inside main()
    while `_apply_falsifier_trim` (a module-level function) also called it, so
    that call raised NameError on every fired falsifier. The NameError was
    raised while evaluating an argument to `record_intervention` inside a
    swallow-everything `try`, so falsifier trims submitted real orders and then
    silently never recorded the intervention they were supposed to be scored on.
    """
    try:
        import json as _gj
        _gsig = _gj.loads(open("dashboard/regime_signal.json").read())
        if isinstance(_gsig, list):
            _gsig = _gsig[-1] if _gsig else {}
        return str(_gsig.get("label", "unknown")).lower()
    except Exception:
        return "unknown"


def _get_portfolio_symbols() -> list:
    """Return symbols with nonzero weight in the current merged portfolio."""
    try:
        p = Path("execution/merged_weights.json")
        if p.exists():
            payload = json.loads(p.read_text())
            # Payload format: {"date", "weights": {sym: w}, ...}; legacy: flat {sym: w}
            weights = payload.get("weights", payload) if isinstance(payload, dict) else {}
            return [s for s, w in weights.items()
                    if isinstance(w, (int, float)) and w > 0]
    except Exception:
        pass
    return []


def _collect_altdata(portfolio_symbols: list, all_symbols: list) -> None:
    """
    Run alt data collection before agents start. Reddit excluded (no credentials).
    Each source is independently wrapped — one failure never blocks the rest.
    SEC/Transcripts skip ETFs automatically (no 10-K/8-K filings for fund tickers).
    """
    from ascent.data.ingest.sec_filings import update_sec_signals
    from ascent.data.ingest.earnings_transcripts import (
        fetch_recent_8k_transcripts, update_transcript_signals,
    )
    from ascent.data.ingest.google_trends import update_trends_signals

    today = date.today()
    is_sunday = today.weekday() == 6
    targets = portfolio_symbols if portfolio_symbols else all_symbols[:50]

    # Filter ETFs out of SEC/Transcripts targets — they don't file 10-K/8-K earnings
    _ETF_SUFFIXES = {"ETF", "EW", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "UUP",
                     "HYG", "LQD", "TIP", "SGOV", "BIL", "DBB", "KMLM", "PDBC",
                     "DBA", "IFRA", "VNQ", "VIXY"}
    equity_targets = [s for s in targets if s not in _ETF_SUFFIXES]

    sources = [
        ("SEC",         lambda: update_sec_signals(equity_targets) if equity_targets else None),
        ("Transcripts", lambda: update_transcript_signals(
                            fetch_recent_8k_transcripts(equity_targets)) if equity_targets else None),
        ("Trends",      lambda: update_trends_signals(
                            all_symbols if is_sunday else targets)),
    ]
    for name, fn in sources:
        try:
            print(f"[AltData] Collecting {name}...")
            fn()
            print(f"[AltData] {name} done")
        except Exception as e:
            print(f"[AltData] {name} failed (non-fatal): {e}")


def main():
    dry_run             = "--dry-run" in sys.argv
    skip_sector_check   = "--skip-sector-check" in sys.argv
    _force_run          = "--force" in sys.argv
    _date_override      = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--date=")), None)
    # market_today(), not date.today(): this host is UTC+7, so the local calendar
    # day rolls over ~14h before the US one. Using local time dated ~78% of
    # historical rows to a session that had not closed yet, and put weekend and
    # holiday dates into the logs. See ascent/utils/market_time.py.
    today               = date.fromisoformat(_date_override) if _date_override else market_today()

    # ── Weekend branch: runs before everything else ───────────────────────────
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        from ascent.monitoring.weekend_runner import already_ran_this_weekend, run_weekend
        prior_run = already_ran_this_weekend()
        if prior_run:
            day_name = ["Monday","Tuesday","Wednesday","Thursday",
                        "Friday","Saturday","Sunday"][today.weekday()]
            print(f"\n{'='*60}")
            print(f"  Ascent Capital — Weekend Mode")
            print(f"{'='*60}")
            print(f"  Already ran this weekend ({prior_run}).")
            print(f"  Weekend intelligence pipeline runs once per weekend.")
            print(f"  Come back next weekend, or run on a weekday for")
            print(f"  the normal non-rebalance / rebalance flow.")
            print(f"{'='*60}\n")
            return
        run_weekend(dry_run=dry_run)
        return

    # ── W3.5: Catch-up guard — refuse to auto-execute on a stale pipeline ─────
    # This exists because the 27-day outage that motivated this whole plan
    # produced zero warning: the daily job just silently stopped running.
    catch_up_flag = "--catch-up" in sys.argv
    must_refuse, _missed_dates = _catch_up_guard(today)
    if must_refuse:
        if catch_up_flag:
            _CATCH_UP_STATE["active"] = True
            _CATCH_UP_STATE["missed_dates"] = _missed_dates
            print(f"\n{'!'*60}")
            print(f"  CATCH-UP MODE")
            print(f"  Last logged run is stale — {len(_missed_dates)} trading day(s) missed.")
            if _missed_dates:
                print(f"  Skipped dates (NOT replayed): {', '.join(_missed_dates)}")
            print(f"  Running ONE fresh rebalance on today's data ({today.isoformat()}).")
            print(f"  Missed dates are intentionally not replayed — stale intent and")
            print(f"  double transaction costs are worse than a clean gap.")
            print(f"{'!'*60}\n")
            try:
                Path("logs/eod_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
                Path("logs/eod_log.jsonl").open("a").write(json.dumps({
                    "date":          today.isoformat(),
                    "run_type":      "catch_up",
                    "skipped_dates": _missed_dates,
                    "note":          (f"Outage recovery: {len(_missed_dates)} trading day(s) "
                                      "missed prior to this run; not replayed."),
                    "timestamp":     datetime.now().isoformat(),
                }) + "\n")
            except Exception as _cu_log_e:
                print(f"[CatchUpGuard] Failed to write catch_up marker to eod_log: {_cu_log_e}")
        else:
            print(f"\n{'!'*60}")
            print(f"  OUTAGE DETECTED — refusing to auto-execute")
            print(f"  {'='*56}")
            if _missed_dates:
                print(f"  Last logged run is {len(_missed_dates)} trading day(s) stale.")
                print(f"  Missed trading days: {', '.join(_missed_dates)}")
            else:
                print(f"  No prior run could be found in logs/eod_log.jsonl.")
            print(f"\n  RECOVERY: re-run with --catch-up to compute ONE fresh")
            print(f"  rebalance on today's data. Missed dates will NOT be replayed")
            print(f"  (stale intent + double transaction costs).")
            print(f"{'!'*60}\n")
            sys.exit(1)

    # ── Same-session guard ────────────────────────────────────────────────────
    # The scheduled job fires at 09:00 local (UTC+7) = the previous US session,
    # so a manual/catch-up run earlier in that same session would otherwise be
    # re-processed here and append a duplicate row. Explicit --date or --force
    # overrides (a deliberate re-run is allowed; an accidental one is not).
    if not _force_run and not _date_override and already_ran_for_session(today):
        print(f"\n[Runner] A run is already logged for the {today} session "
              f"(market date). Skipping to avoid a duplicate record.")
        print(f"[Runner] Re-run deliberately with --force, or target another "
              f"session with --date=YYYY-MM-DD.")
        return

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

    # Update realized outcomes for past episodes (best-effort; no-op if no data)
    try:
        update_outcomes({})
    except Exception:
        pass

    # ── Step 0a: Start event agent background thread (market hours, weekdays) ──
    _event_thread = None
    try:
        from datetime import datetime as _dt
        import pytz as _pytz
        _et = _pytz.timezone("America/New_York")
        _now_et = _dt.now(_et)
        _hour, _minute = _now_et.hour, _now_et.minute
        _is_weekday = today.weekday() < 5
        _in_market_hours = _is_weekday and (9, 30) <= (_hour, _minute) <= (15, 45)
        if _in_market_hours:
            from agents.event_agent import start_event_agent_thread
            _event_thread = start_event_agent_thread()
            print("[EventAgent] Background thread started")
        else:
            print("[EventAgent] Outside market hours — thread not started")
    except Exception as _evt_e:
        print(f"[EventAgent] Failed to start: {_evt_e}")

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

    # ── OpenBB ingest: CBOE options, CFTC COT, Fama-French factors ─────────
    _today_str = today.isoformat()
    try:
        from ascent.data.ingest.cboe_options import update_options_cache
        _n_opts = update_options_cache(us_symbols, _today_str)
        if _n_opts:
            print(f"[Runner] CBOE options: {_n_opts} new rows added")
    except Exception as _opts_e:
        print(f"[Runner] CBOE options ingest skipped: {_opts_e}")

    try:
        from ascent.data.ingest.cftc_positioning import update_cot_cache
        _cot_added = update_cot_cache()
        if _cot_added:
            print("[Runner] CFTC COT: updated")
    except Exception as _cot_e:
        print(f"[Runner] CFTC COT ingest skipped: {_cot_e}")

    try:
        from ascent.data.ingest.famafrench_factors import update_ff_cache
        _ff_ok = update_ff_cache()
        if _ff_ok:
            print("[Runner] Fama-French factors: updated")
    except Exception as _ff_e:
        print(f"[Runner] Fama-French factors ingest skipped: {_ff_e}")

    # ── Sector profile coverage guard: backfill live-book gaps, warn <90% ────
    try:
        from ascent.data.ingest.supplementary import check_book_sector_coverage
        _book_w = _get_portfolio_symbols()
        if isinstance(_book_w, list):
            _book_w = {s: 1.0 / max(len(_book_w), 1) for s in _book_w}
        check_book_sector_coverage(_book_w or {})
    except Exception as _prof_e:
        print(f"[Runner] Profile coverage guard skipped: {_prof_e}")

    # ── Alt data collection (runs before agents; each source fails silently) ──
    _collect_altdata(
        portfolio_symbols=_get_portfolio_symbols(),
        all_symbols=us_symbols,
    )

    # ── Import agents (lazy, after startup validation) ───────────────────────
    # macro_agent/international_agent scored CUT on their real universes (proof
    # audit); alternatives_agent is still unmeasured (unexplained density issue,
    # excluded pending future re-measurement, not proven negative). Only
    # us_equities_agent allocates live capital — the other three agent modules
    # are kept on disk (with their run_*_agent() entry points intact) but are
    # no longer invoked from the daily orchestration flow.
    from agents.us_equities_agent import run_us_equities_agent
    from orchestrator.central_intelligence import run_orchestrator
    from ascent.monitoring.skill_tracker import export_skill_scores
    from ascent.monitoring.forward_pnl_tracker import run_forward_pnl_cycle
    from ascent.monitoring.pre_rebalance_checklist import run_checklist

    # ── Step 0b: Factor data + loadings update ───────────────────────────────
    try:
        from ascent.risk.factor_data import update_factor_data
        update_factor_data()
    except Exception as _fde:
        print(f"[FactorData] Update skipped: {_fde}")

    try:
        from ascent.risk.factor_model import update_factor_loadings
        update_factor_loadings()
    except Exception as _fle:
        print(f"[FactorModel] Loadings update skipped: {_fle}")

    # ── Step 1: Run agents (only us_equities allocates live capital) ─────────
    agent_tasks = [
        ("us_equities", run_us_equities_agent),
    ]

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

    # Early rebalance trigger: IC decay ≥30% since last rebalance after ≥5 bdays
    # Suppressed within 3 trading days of a scheduled rebalance — the scheduled
    # rebalance will recompute the book anyway; an early rotation is pure churn.
    if not is_rebalance:
        if _is_near_scheduled_rebalance(today, cal_path=cal_path):
            print("[Runner] Early rebalance trigger suppressed — within 3 trading days of scheduled rebalance")
        else:
            try:
                from ascent.monitoring.rebalance_trigger import is_triggered, check_ic_decay_trigger
                from ascent.monitoring.signal_health import compute_signal_health
                if is_triggered():
                    print("[Runner] Early rebalance triggered — IC decay flag detected.")
                    is_rebalance = True
                else:
                    _current_ics = {
                        s: d.get("ic_5d_avg", 0.0)
                        for s, d in compute_signal_health(today.isoformat()).items()
                    }
                    triggered = check_ic_decay_trigger(today.isoformat(), _current_ics)
                    if triggered:
                        print("[Runner] IC decay triggered early rebalance.")
                        is_rebalance = True
            except Exception as _te:
                print(f"[Runner] Rebalance trigger check skipped: {_te}")

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

            # Event trade IC — weekly measurement
            try:
                from ascent.execution.event_runner import compute_event_ic
                _eic = compute_event_ic(lookback_days=20)
                if _eic.get("n_trades", 0) > 0:
                    print(f"[EventIC] ic_5d={_eic.get('ic_5d')} ic_10d={_eic.get('ic_10d')} "
                          f"ic_20d={_eic.get('ic_20d')} n={_eic['n_trades']}")
            except Exception as _eic_e:
                print(f"[EventIC] Skipped: {_eic_e}")
    except Exception as e:
        print(f"[Runner] Self-improve failed: {type(e).__name__}: {e}")

    # Factor discovery — first Sunday of each month only
    try:
        from datetime import date as _fdate
        _ftoday = _fdate.today()
        if _ftoday.weekday() == 6 and _ftoday.day <= 7:
            from ascent.research.factor_discovery.discovery_runner import run_factor_discovery
            _disc_regime = _get_current_regime()
            print(f"[FactorDiscovery] Monthly run — regime={_disc_regime}")
            _disc = run_factor_discovery(n_candidates=5, regime=_disc_regime)
            print(
                f"[FactorDiscovery] Done: {_disc['n_accepted']} accepted, "
                f"{_disc['n_rejected']} rejected. "
                f"Proposals: outputs/factor_proposals/"
            )
    except Exception as _de:
        print(f"[FactorDiscovery] Monthly run skipped: {_de}")

    # Altdata validation — first Sunday of each month (same trigger as factor discovery)
    try:
        from datetime import date as _adate
        _atoday = _adate.today()
        if _atoday.weekday() == 6 and _atoday.day <= 7:
            from ascent.data.validate.altdata_validator import run_altdata_validation
            from ascent.data.ingest.sec_filings import load_sec_signals
            from ascent.data.ingest.earnings_transcripts import load_transcript_signals
            from ascent.data.ingest.google_trends import load_trends_signals
            from ascent.data.store.parquet import load_parquet as _lp_alt, has_data as _hd_alt
            import pandas as _pd_alt

            _alt_sources = {}
            for _src_name, _loader in [
                ("sec",         load_sec_signals),
                ("transcripts", load_transcript_signals),
                ("trends",      load_trends_signals),
            ]:
                try:
                    _panel = _loader()
                    if not _panel.empty:
                        _alt_sources[_src_name] = _panel
                except Exception:
                    pass

            if _alt_sources:
                _alt_prices = _lp_alt("prices_live") if _hd_alt("prices_live") else _pd_alt.DataFrame()
                try:
                    _alt_regime = _pd_alt.read_csv(
                        "dashboard/regime_labels.csv", index_col=0, parse_dates=True
                    ).iloc[:, 0]
                except Exception:
                    _alt_regime = _pd_alt.Series(dtype=str)
                _alt_results = run_altdata_validation(_alt_sources, _alt_prices, _alt_regime)
                _alt_accepted = [r["source"] for r in _alt_results if r["status"] == "accepted"]
                print(f"[AltdataValidation] {len(_alt_accepted)} accepted: {_alt_accepted}")
            else:
                print("[AltdataValidation] No cached altdata panels found — run ingest first")
    except Exception as _ae:
        print(f"[AltdataValidation] Monthly run skipped: {_ae}")

    # Google Trends weekly refresh — every Sunday (rate-limited; capped at 50 symbols)
    try:
        from datetime import date as _gdate
        if _gdate.today().weekday() == 6:
            from ascent.data.ingest.google_trends import update_trends_signals
            from ascent.config.settings import get_config as _gcfg
            _g_syms = list(_gcfg().universe.symbols)[:50]
            print(f"[GoogleTrends] Weekly refresh for {len(_g_syms)} symbols")
            update_trends_signals(_g_syms)
            print("[GoogleTrends] Weekly refresh complete")
    except Exception as _ge:
        print(f"[GoogleTrends] Weekly refresh skipped: {_ge}")

    # ── Monthly: audit integrity check ────────────────────────────────────────
    try:
        from datetime import date as _mdate
        _today_m = _mdate.today()
        if _today_m.weekday() == 6 and _today_m.day <= 7:  # first Sunday of month
            try:
                import subprocess as _sp, sys as _sys
                _audit_result = _sp.run(
                    [_sys.executable, "scripts/verify_audit_trail.py"],
                    capture_output=True, text=True, timeout=30,
                )
                print(f"[AuditIntegrity] {'PASS' if _audit_result.returncode == 0 else 'FAIL'}")
            except Exception as _ai_e:
                print(f"[AuditIntegrity] Skipped: {_ai_e}")
    except Exception as _monthly_e:
        print(f"[Monthly] Monthly tasks skipped: {_monthly_e}")

    # NOTE: alert checking (drawdown / factor breach / sleeve IC decay) and the
    # daily "system alive" proof-of-life ping used to be called here with zero
    # arguments, which made check_alerts() a permanent no-op (every threshold
    # derives from args that default to None) wrapped in a bare `except: pass`
    # that would have hidden even a real exception. Neither can be fixed at
    # this point in the run: merged_weights, factor exposures, and sleeve IC
    # are all computed later. See `_run_daily_alert_checks()`, called from
    # `_log_holdings()` below where equity, positions, factor exposures, and
    # sleeve IC actually exist.

    # ── Step 5: Run orchestrator (reads fresh skill scores written above) ─────
    merged_weights = run_orchestrator(agent_outputs)

    if not merged_weights:
        print("[Runner] Orchestrator returned empty weights — aborting execution")
        return

    # ── Step 5c: 130/30 long-short overlay (kill-switched) ───────────────────
    if LONG_SHORT_ENABLED:
        try:
            from ascent.portfolio.long_short import build_long_short_weights
            _us = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
            if _us is not None and _us.alpha_scores is not None and not _us.alpha_scores.empty:
                _alpha = _us.alpha_scores.iloc[-1].dropna()
                merged_weights = build_long_short_weights(
                    _alpha, long_n=15, short_n=5, long_pct=1.30, short_pct=0.30
                )
                print(f"[LongShort] 130/30 applied: "
                      f"{sum(1 for v in merged_weights.values() if v > 0)} longs, "
                      f"{sum(1 for v in merged_weights.values() if v < 0)} shorts")
        except Exception as _ls_e:
            print(f"[LongShort] Skipped: {_ls_e}")

    # ── Step 5d: Export factor exposures to dashboard ────────────────────────
    try:
        import pandas as _pd
        from ascent.risk.factor_exposure import export_factor_exposures
        _fw = _pd.Series({k: float(v) for k, v in merged_weights.items()})
        export_factor_exposures(_fw, today)
    except Exception as _fe:
        print(f"[FactorExposure] Export skipped: {_fe}")

    # ── Daily intelligence / discovery / falsifier / AI-PM synthesis layer
    # removed 2026-08-23 (noise-layer cut) — see CLAUDE.md constraint 5. The
    # merged quant weights from the orchestrator above are what gets written
    # and executed; nothing downstream mutates them anymore.

    # Log episode for regime-aware memory (quant-only; no AI PM layer exists).
    try:
        log_episode(
            run_date=today.isoformat(),
            regime=_get_current_regime(),
            quant_weights=merged_weights,
        )
    except Exception as _e:
        print(f"[Memory] Episode log failed: {_e}")

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


    # Audit trail: portfolio construction
    try:
        from compliance.audit_trail import record_event as _audit_rec
        _audit_rec("portfolio_constructed", {
            "date":         today.isoformat(),
            "n_positions":  len(merged_weights),
            "agents":       [ao.agent_id for ao in agent_outputs],
        })
    except Exception:
        pass

    # TimescaleDB: write portfolio state
    try:
        from ascent.data.store.timescale import write_portfolio_state, timescale_available
        if timescale_available():
            for _ao in agent_outputs:
                write_portfolio_state(today, _ao.agent_id, _ao.target_weights)
    except Exception:
        pass

    # ── Post-rebalance: update meta-learner from holding-period sleeve IC ────
    _ML_SNAP_PATH = Path("data_cache/meta_learner_rebalance_snapshot.json")
    if is_rebalance:
        try:
            from ascent.alpha.meta_learner import SleeveMetaLearner as _SML

            _sleeve_ic_log = Path("logs/sleeve_ic_log.jsonl")
            _realized_ic: dict = {}

            if _sleeve_ic_log.exists() and _ML_SNAP_PATH.exists():
                _prev_snap = json.loads(_ML_SNAP_PATH.read_text())
                _prev_date = _prev_snap.get("rebalance_date", "")
                if _prev_date:
                    from collections import defaultdict as _dd
                    _sleeve_sums: dict = _dd(list)
                    for _line in _sleeve_ic_log.read_text().splitlines():
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _e = json.loads(_line)
                            if _e.get("date", "") >= _prev_date:
                                for _sl, _st in _e.get("sleeves", {}).items():
                                    _ic = _st.get("mean_ic")
                                    if _ic is not None:
                                        _sleeve_sums[_sl].append(float(_ic))
                        except Exception:
                            continue
                    _realized_ic = {
                        _sl: sum(_ics) / len(_ics)
                        for _sl, _ics in _sleeve_sums.items() if _ics
                    }

            if _realized_ic:
                _current_regime = _get_current_regime()
                _ml = _SML()
                _ml.update_rebalance(_current_regime, _realized_ic)
                print(f"[Runner] Meta-learner updated: regime={_current_regime} "
                      f"sleeves={list(_realized_ic.keys())}")
            else:
                print("[Runner] Meta-learner: no IC data since prior rebalance — skipping")

            _ML_SNAP_PATH.write_text(json.dumps({
                "rebalance_date": today.isoformat(),
                "quant_weights":  merged_weights,
            }, indent=2))
        except Exception as _ml_upd_e:
            print(f"[Runner] Meta-learner update failed: {_ml_upd_e}")

    # ── Non-rebalance day: stop here ──────────────────────────────────────────
    if not is_rebalance:
        print("[Runner] Non-rebalance day — weights updated, no execution.")
        try:
            _log_holdings(today)
        except Exception as e:
            print(f"[Runner] Holdings log skipped: {e}")
        _log_run(today, merged_weights, agent_outputs, dry_run)
        return

    # ── Rebalance day: check for active halt before executing ────────────────
    if not check_halt_state(today=today):
        print("[Runner] Halted — agents ran, weights updated, execution skipped.")
        print("[Runner] Create execution/halt_override.json to resume.")
        try:
            _log_holdings(today)
        except Exception as e:
            print(f"[Runner] Holdings log skipped: {e}")
        _log_run(today, merged_weights, agent_outputs, dry_run)
        return

    # ── Step 5: Execute via eod_runner ────────────────────────────────────────
    if dry_run:
        print(f"\n[Runner] DRY RUN — would submit {len(merged_weights)} positions:")
        for sym, w in sorted(merged_weights.items(), key=lambda x: -x[1])[:15]:
            print(f"  {sym}: {w:.2%}")
    else:
        print("[Runner] LIVE MODE — calling eod_runner with merged weights")
        try:
            merged_weights, _stopped_syms = _apply_stop_loss_to_book(
                merged_weights, today.isoformat()
            )
            from ascent.execution.eod_runner import run_eod_with_weights
            run_eod_with_weights(merged_weights, run_date=today, dry_run=False)
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

    # Clear IC decay trigger flag after successful rebalance
    try:
        from ascent.monitoring.rebalance_trigger import consume_trigger
        consume_trigger()
    except Exception:
        pass


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
            ao.agent_id: round({"us_equities": 1.0}.get(ao.agent_id, 0.0), 2)
            for ao in agent_outputs
        },
        "merged_positions": len(merged_weights),
        "mode":             "dry_run" if dry_run else "live",
        "timestamp":        datetime.now().isoformat(),
    }

    if _CATCH_UP_STATE["active"]:
        # So a catch-up recovery run does not silently read as a normal, flat
        # day in the AI PM calibration and D-A*/B-A* counterfactual series.
        run_log["run_type"]      = "catch_up"
        run_log["skipped_dates"] = _CATCH_UP_STATE["missed_dates"]

    log_path = Path("logs/multi_agent_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(run_log) + "\n")

    print(f"\n[Runner] Run logged to {log_path}")

    try:
        from ascent.llm.client import log_costs
        log_costs(today.isoformat())
    except Exception as e:
        print(f"[Runner] Cost log skipped ({e})")

    print(f"[Runner] Done.\n")


def _compute_drawdown_from_holdings_log(
    current_equity: float,
    log_path: Path = Path("logs/holdings_log.jsonl"),
    lookback_entries: int = 90,
) -> Optional[float]:
    """
    Current drawdown from the trailing local peak, using the equity series
    already recorded in logs/holdings_log.jsonl. Returns None if there is not
    enough history to establish a peak (e.g. first run, or file missing) —
    callers should treat None as "no drawdown signal available", not zero.
    """
    if current_equity <= 0 or not log_path.exists():
        return None
    try:
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    except Exception:
        return None
    if not lines:
        return None

    equities = []
    for line in lines[-lookback_entries:]:
        try:
            e = json.loads(line).get("equity")
            if e:
                equities.append(float(e))
        except Exception:
            continue
    if not equities:
        return None

    peak = max(equities + [current_equity])
    if peak <= 0:
        return None
    return max(0.0, (peak - current_equity) / peak)


def _run_daily_alert_checks(today, equity: float, last_equity: float) -> None:
    """
    Wire up the alert system with real, currently-available data and fire the
    daily "system alive" proof-of-life ping.

    Split out of `_log_holdings()` (rather than inlined) so it can be
    unit-tested without a live Alpaca account: pass in equity/last_equity
    directly and monkeypatch the alert_system functions.

    Data sourced here, and why each is what it is (not invented):
      - portfolio_state["drawdown"]: computed from the equity series already
        written to logs/holdings_log.jsonl (local peak vs current equity).
        This is a real proxy for drawdown; it is NOT the same as LiveNAV's
        intraday drawdown (no streaming NAV is wired into this pipeline —
        documented gap, left as live_nav=None).
      - factor_exposures: read back from dashboard/factor_exposures.json,
        which export_factor_exposures() already wrote earlier this same run
        (see the "Export factor exposures to dashboard" step). Best-effort:
        file may not exist yet on a non-rebalance day.
      - sleeve_ic: from ascent.monitoring.signal_health.compute_signal_health(),
        which is already computed elsewhere in this run for the post-rebalance
        snapshot. Uses "ic_5d_avg" (a 5-day rolling average), not a literal
        21-day figure — check_alerts()'s docstring says 21d but the function
        only compares against a floor, so a 5d average is a legitimate,
        genuinely-available substitute rather than invented data.
    """
    import logging
    log = logging.getLogger(__name__)

    try:
        from ascent.monitoring.alert_system import check_alerts, send_system_alive_ping

        portfolio_state = None
        drawdown = _compute_drawdown_from_holdings_log(equity)
        if drawdown is not None:
            portfolio_state = {"drawdown": drawdown}

        factor_exposures = None
        try:
            fe_path = Path("dashboard/factor_exposures.json")
            if fe_path.exists():
                factor_exposures = json.loads(fe_path.read_text()).get("exposures") or None
        except Exception as e:
            log.warning("[Alerts] Could not read factor_exposures.json: %s", e)

        sleeve_ic = None
        try:
            from ascent.monitoring.signal_health import compute_signal_health
            _health = compute_signal_health(today.isoformat())
            if _health:
                sleeve_ic = {s: d.get("ic_5d_avg") for s, d in _health.items()
                             if d.get("ic_5d_avg") is not None}
        except Exception as e:
            log.warning("[Alerts] Could not compute sleeve IC: %s", e)

        alert_list = check_alerts(
            portfolio_state=portfolio_state,
            live_nav=None,  # no streaming LiveNAV wired into this pipeline yet
            factor_exposures=factor_exposures,
            sleeve_ic=sleeve_ic,
        )
        if alert_list:
            print(f"[Alerts] {len(alert_list)} new alert(s) fired.")
    except Exception:
        log.exception("[Alerts] check_alerts() failed")

    # Proof-of-life ping: once per run, so "no failure alerts today" can be
    # told apart from "the alert channel itself is dead" (the failure mode
    # behind the earlier multi-week silent outage).
    try:
        from ascent.monitoring.alert_system import send_system_alive_ping
        send_system_alive_ping(
            last_run=today.isoformat(),
            nav=equity if equity else None,
            nav_prior=last_equity if last_equity else None,
        )
    except Exception:
        log.exception("[Alerts] send_system_alive_ping() failed")


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

        # Day return: use Alpaca's own equity vs last session close — ground truth.
        # Attribution-derived returns are intraday estimates and miss after-hours moves.
        last_equity = float(acct.get("last_equity", 0))
        day_ret = (equity - last_equity) / last_equity if last_equity > 0 else 0.0

        # SPY return and position breakdown: still run attribution with actual positions.
        spy_ret = 0.0
        if positions:
            try:
                from ascent.monitoring.attribution import run_attribution
                attr = run_attribution(positions, today)
                if attr:
                    spy_ret = attr.get("spy_return", 0.0)
                    # attribution_log gets position-level breakdown; headline return comes from Alpaca above
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

        # ── Alerts + daily "system alive" proof-of-life ping ──────────────────
        # This is the actual wiring for the previously-dead alert path (see
        # `_run_daily_alert_checks` docstring): equity/last_equity are real
        # numbers from Alpaca at this point in the run, unlike the earlier
        # no-arg call site before the orchestrator had even run.
        _run_daily_alert_checks(today, equity, last_equity)

    except Exception as e:
        print(f"[Runner] Holdings log skipped ({e})")

    # ── Regenerate GitHub Pages dashboard ─────────────────────────────────────
    try:
        import os as _os
        _repo_root = str(Path(__file__).resolve().parent)
        # The generator lazily imports `ascent` (counterfactual chart). Running it as
        # `python scripts/...py` puts scripts/ on sys.path, not the repo root, so the
        # import fails unless the repo root is on PYTHONPATH. Make this independent of
        # the ambient environment (launchd sets it; a manual run does not).
        _env = {**_os.environ, "PYTHONPATH": _repo_root + _os.pathsep + _os.environ.get("PYTHONPATH", "")}
        result = subprocess.run(
            [sys.executable, "scripts/generate_performance_page.py", "--push"],
            capture_output=True, text=True, timeout=120,
            cwd=_repo_root, env=_env,
        )
        if result.returncode == 0:
            print("[Dashboard] GitHub Pages updated and pushed.")
        else:
            print(f"[Dashboard] Generator exited {result.returncode}: {result.stderr[-200:]}")
    except Exception as e:
        print(f"[Dashboard] Skipped ({e})")


def _is_near_scheduled_rebalance(today, window: int = 3, cal_path=None) -> bool:
    """
    Returns True if the next scheduled rebalance is within `window` trading days
    of `today` (inclusive). Off-calendar discovery must be suppressed in this
    window — the scheduled rebalance recomputes the whole book anyway, so an
    intra-period full rotation just ahead of it is pure churn.

    Returns False when no future rebalance is on the calendar, or the calendar
    file is missing/unreadable (fail-open: a missing calendar should not block
    discovery entirely).
    """
    from pathlib import Path as _P
    import pandas as _pd

    path = _P(cal_path) if cal_path is not None else _P("rebalance_calendar.csv")
    if not path.exists():
        return False
    try:
        cal = _pd.read_csv(path)
        today_ts = _pd.Timestamp(today)
        future = [
            _pd.Timestamp(d) for d in cal["rebalance_date"]
            if _pd.Timestamp(d) > today_ts
        ]
        if not future:
            return False
        next_reb = min(future)
        # trading days strictly between today and the next rebalance, inclusive
        trading_days = len(_pd.bdate_range(today_ts, next_reb)) - 1
        return trading_days <= window
    except Exception:
        return False


def already_ran_for_session(session_date, log_path=None) -> bool:
    """True if logs/eod_log.jsonl already holds a RUN record for `session_date`.

    The scheduled job fires at 09:00 local (UTC+7), which resolves to the
    previous US session — so a manual catch-up run earlier in that same session
    collides with it. Without this check the second run appends a duplicate row,
    which is how logs/ai_pm_decision_log.jsonl ended up with 9 rows across 2
    dates (2026-06-10 recorded 8 times), inflating every rate computed from it.

    Fails OPEN (returns False) on a missing or unreadable log: never block the
    first ever run. `_catch_up_guard` is the fail-CLOSED counterpart for
    staleness; this one only dedupes.

    Discovery-candidate objects are also written into eod_log (keys: symbol /
    trigger / conviction) and are not run records, so they do not count.
    """
    from pathlib import Path as _P

    p = _P(log_path) if log_path is not None else _P("logs/eod_log.jsonl")
    if not p.exists():
        return False
    target = session_date.isoformat() if hasattr(session_date, "isoformat") else str(session_date)
    try:
        with p.open("r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("trigger") == "discovery" or "conviction" in entry:
                    continue
                if str(entry.get("date") or entry.get("run_date") or "") == target:
                    return True
    except OSError:
        return False
    return False


def _catch_up_guard(today, threshold_days: int = CATCH_UP_STALE_TRADING_DAYS):
    """
    Determine whether the daily run should refuse to auto-execute because the
    last logged run is stale (W3 item 5 — the 27-day outage this guard exists
    to catch never printed a warning; the pipeline just silently didn't run).

    Returns (must_refuse: bool, missed_trading_days: list[str]).

    Fail-safe: any failure to read/parse the log, or no prior run found at
    all, returns must_refuse=True — refuse rather than silently proceed.

    Reuses the same stdlib-only trading-day arithmetic as
    scripts/heartbeat_check.py so the two staleness checks agree.
    """
    from datetime import timedelta as _timedelta
    try:
        from scripts.heartbeat_check import read_last_run_date, trading_days_between
        last_run = read_last_run_date(Path("logs/eod_log.jsonl"))
    except Exception as e:
        print(f"[CatchUpGuard] Could not read last run date ({e}) — refusing (fail-safe).")
        return True, []

    if last_run is None:
        print("[CatchUpGuard] No prior run found in logs/eod_log.jsonl — refusing (fail-safe).")
        return True, []

    missed = trading_days_between(last_run, today + _timedelta(days=1))
    missed_iso = [d.isoformat() for d in missed]
    must_refuse = len(missed) > threshold_days
    return must_refuse, missed_iso


def _live_book_or(fallback: dict) -> dict:
    """
    Return the LIVE Alpaca book as {symbol: weight}, normalized to sum to 1.0,
    falling back to `fallback` when the broker returns nothing usable or raises.

    This is the single implementation of the "read live positions, fall back to
    a recomputed/passed-in book" pattern used by both the falsifier-trim path
    and the discovery mini-rebalance path. Never raises.
    """
    try:
        from ascent.execution.alpaca_broker import get_positions
        pos = get_positions()
        if pos is not None and not pos.empty and {"symbol", "weight"}.issubset(pos.columns):
            book = {str(r["symbol"]): float(r["weight"]) for _, r in pos.iterrows()}
            total = sum(abs(w) for w in book.values())
            if total > 0:
                return {k: v / total for k, v in book.items()}
    except Exception:
        pass
    return dict(fallback or {})


def _insert_candidate_weights(
    current_weights: dict,
    symbol: str,
    max_weight: float = 0.10,
    target_weight: float | None = None,
) -> dict:
    """
    Add `symbol` to the book and trim existing holdings PRO-RATA to make room —
    without re-ranking or re-optimizing the rest of the portfolio.

    The candidate takes an equal-weight slot (1/(n+1)) by default; existing
    weights keep their relative ordering, scaled down to fit. The max-weight cap
    is then enforced via the same water-fill routine the optimizer uses, so the
    result satisfies the system's standing invariant (all weights <= max_weight,
    sum == 1.0).

    No-op (returns a copy of the input) if `symbol` is already held.
    """
    import pandas as _pd
    from ascent.portfolio.optimizer import _water_fill_cap

    if symbol in current_weights:
        return dict(current_weights)
    if not current_weights:
        return {symbol: 1.0}

    n = len(current_weights)
    if target_weight is None:
        target_weight = 1.0 / (n + 1)
    target_weight = min(target_weight, max_weight)

    # Existing weights sum to ~1.0; give the candidate a score that renormalizes
    # to `target_weight`, leaving the rest pro-rata. s/(1+s) = t  ->  s = t/(1-t).
    scores = _pd.Series(current_weights, dtype=float)
    scores = scores / scores.sum()
    scores[symbol] = target_weight / (1.0 - target_weight)
    capped = _water_fill_cap(scores, max_weight)
    return {k: float(v) for k, v in capped.items()}


def _check_mini_rebalance_cooldown() -> bool:
    """Returns True if a mini-rebalance ran < 5 trading days ago (cooldown active)."""
    import json as _j
    from pathlib import Path as _P
    import pandas as _pd

    cooldown_path = _P("data_cache/last_mini_rebalance.json")
    if not cooldown_path.exists():
        return False
    try:
        rec = _j.loads(cooldown_path.read_text())
        last = _pd.Timestamp(rec["date"])
        trading_days_elapsed = len(_pd.bdate_range(last, _pd.Timestamp.today())) - 1
        return trading_days_elapsed < 5
    except Exception:
        return False


def _write_mini_rebalance_log(symbol: str, conviction: float) -> None:
    """Write cooldown state after a mini-rebalance completes."""
    import json as _j
    from pathlib import Path as _P
    from datetime import date as _date

    path = _P("data_cache/last_mini_rebalance.json")
    path.write_text(_j.dumps({
        "date":       _date.today().isoformat(),
        "symbol":     symbol,
        "conviction": conviction,
    }))


def _apply_stop_loss_to_book(target_weights: dict, today: str) -> tuple:
    """
    Apply the position-level stop-loss to a target book, using entry prices
    from the live Alpaca positions.

    Must be called LAST, after every cap and overlay. Those caps
    (_water_fill_cap, enforce_cluster_cap, enforce_risk_budget_cap,
    apply_exposure_overlays) run upstream, in the agents and orchestrator,
    before `merged_weights` ever reaches this function — they are not invoked
    here or nearby. They all renormalize and would refill a stopped name, so
    this call must stay downstream of all of them.

    Fail-open: any failure (broker down, missing prices) returns the input
    unchanged. A monitoring failure must never liquidate the book.

    Returns (adjusted_weights, stopped_symbols).
    """
    # Local imports match this file's style: run_all_agents.py has no
    # module-level `logging` import (verified 2026-07-27 — the only use is a
    # function-local `import logging as _logging` at line 2616).
    import logging
    import pandas as pd
    from ascent.config.settings import get_config

    cfg = get_config()
    if not getattr(cfg.backtest, "stop_loss_enabled", False):
        return target_weights, []

    try:
        from ascent.portfolio.stop_loss import (
            DEFAULT_STATE_PATH, compute_stop_breaches, apply_stop_loss,
            load_stop_state, record_stops, blocked_symbols,
        )
        from ascent.execution import alpaca_broker

        w = pd.Series(target_weights, dtype=float)

        # 1. Names still inside their re-entry cooldown never get re-bought.
        state = load_stop_state(DEFAULT_STATE_PATH)
        blocked = blocked_symbols(
            state, today, cooldown_days=cfg.backtest.stop_loss_cooldown_days
        )
        if blocked:
            hit = [s for s in w.index if s in blocked]
            if hit:
                logging.info("[StopLoss] Cooldown blocks re-entry: %s", hit)
                w.loc[hit] = 0.0

        # 2. Evaluate live positions against their entry prices.
        pos = alpaca_broker.get_positions()
        stopped: list = []
        if pos is not None and not pos.empty and "avg_entry_price" in pos.columns:
            idx = pos.set_index("symbol")
            breached = compute_stop_breaches(
                idx["avg_entry_price"].astype(float),
                idx["current_price"].astype(float),
                threshold=cfg.backtest.stop_loss_threshold,
            )
            stopped = [s for s in breached.index if bool(breached[s])]
            if stopped:
                w = apply_stop_loss(
                    w,
                    pd.Series(True, index=[s for s in stopped if s in w.index]),
                    redistribute=cfg.backtest.stop_loss_redistribute,
                )
                record_stops(stopped, today, path=DEFAULT_STATE_PATH)
                logging.warning(
                    "[StopLoss] Stopped out %s at a %.0f%% threshold",
                    stopped, cfg.backtest.stop_loss_threshold * 100,
                )

        return w.to_dict(), stopped

    except Exception as exc:
        logging.warning(
            "[StopLoss] Skipped (%s) — book unchanged. Fail-open by design.",
            exc,
        )
        return target_weights, []


def _trigger_mini_rebalance(
    result,
    current_weights: dict,
    today,
    dry_run: bool = False,
    prior_agent_outputs: list | None = None,
) -> None:
    """
    ADD-ONLY mini-rebalance: insert the discovered ticker into the existing book
    and trim the rest pro-rata (NO full re-optimization of every holding), pass
    the resulting book through the debate gate, and execute. Writes cooldown log
    on completion.

    Rationale: re-running the full us_equities agent + orchestrator here churned
    the entire book for a single candidate (and could even drop the candidate it
    was triggered by). Add-only keeps discovery to what it claims to be — adding
    one name — and confines turnover to that insertion.
    """
    import json as _j
    from pathlib import Path as _P

    print(f"\n[Discovery] Mini-rebalance triggered: {result.symbol} "
          f"(conviction={result.conviction_score:.2f})")
    print(f"[Discovery] Catalyst: {result.catalyst_snippet}")

    try:
        # Base book MUST be the live Alpaca book, not the freshly recomputed
        # orchestrator target — otherwise the add-only insert below operates on
        # a book nobody is holding, and the downstream diff-against-live-positions
        # in run_eod_with_weights(force=True) turns every absent name into a full
        # exit (2026-06-30 incident: 27 orders, 7 complete exits from a "discovery"
        # trigger that only meant to add one name).
        base_book = _live_book_or(current_weights)
        if not base_book:
            print("[Discovery] Mini-rebalance: no book available (live + fallback both empty) — aborting")
            return
        if result.symbol in base_book:
            print(f"[Discovery] {result.symbol} already held — nothing to add, aborting")
            return

        try:
            from ascent.config.settings import get_config
            _max_w = float(get_config().backtest.max_weight)
        except Exception:
            _max_w = 0.10

        # Add-only insertion: candidate gets an equal-weight slot, the rest of the
        # book is trimmed pro-rata (relative ordering preserved), max-weight cap
        # enforced via the optimizer's water-fill routine.
        new_weights = _insert_candidate_weights(base_book, result.symbol, max_weight=_max_w)
        print(f"[Discovery] Add-only insertion: {result.symbol} @ "
              f"{new_weights.get(result.symbol, 0.0) * 100:.1f}% — "
              f"{len(new_weights)} positions (was {len(base_book)})")

        # ── Safety assertion (fail-safe, not fail-open) ──────────────────────
        # An add-only discovery insert must never produce a complete exit from
        # the base book, and must never introduce more than one new symbol
        # (the candidate itself). Either condition means the base book was
        # wrong (e.g. recomputed target instead of live positions) — abort
        # loudly rather than submit.
        _exited_syms = sorted(
            s for s, w in base_book.items()
            if w > 0.0 and new_weights.get(s, 0.0) <= 0.0
        )
        _new_syms = sorted(set(new_weights) - set(base_book))
        if _exited_syms or len(_new_syms) > 1:
            import logging as _logging
            _logging.error(
                "[Discovery] SAFETY ABORT: add-only insert would fully exit %s "
                "and/or introduce %d new symbol(s) %s vs base book (expected <=1, "
                "the candidate) — refusing to submit",
                _exited_syms, len(_new_syms), _new_syms,
            )
            print(f"[Discovery] SAFETY ABORT — full exits={_exited_syms}, "
                  f"new symbols={_new_syms} — mini-rebalance NOT submitted")
            return

        if dry_run:
            print(f"[Discovery] DRY RUN — would submit {len(new_weights)} positions "
                  f"including {result.symbol}")
        else:
            new_weights, _stopped_syms = _apply_stop_loss_to_book(
                new_weights, today.isoformat()
            )
            from ascent.execution.eod_runner import run_eod_with_weights
            run_eod_with_weights(
                new_weights,
                run_date=today,
                dry_run=False,
                force=True,  # mini-rebalance is intra-period by definition
            )

        _write_mini_rebalance_log(result.symbol, result.conviction_score)

        _P("logs/eod_log.jsonl").open("a").write(
            _j.dumps({
                "date":       today.isoformat(),
                "trigger":    "discovery",
                "symbol":     result.symbol,
                "conviction": result.conviction_score,
                "catalyst":   result.catalyst_snippet,
            }) + "\n"
        )
        print(f"[Discovery] Mini-rebalance complete — {result.symbol} added to pipeline")

    except Exception as exc:
        print(f"[Discovery] Mini-rebalance failed: {exc}")


if __name__ == "__main__":
    main()
