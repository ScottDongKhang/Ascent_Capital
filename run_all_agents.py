"""
run_all_agents.py
Top-level daily runner for the Ascent Capital multi-agent platform.

Non-rebalance day:  agents → orchestrator → write weights → log (no debate, no execution)
Rebalance day:      agents → orchestrator → write weights → debate → execute via eod_runner

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

from dotenv import load_dotenv
load_dotenv()

from ascent.data.store.parquet import has_data, load_parquet
from ascent.portfolio.optimizer import SectorDataError

from agents.ai_pm_agent import run_ai_pm, run_ai_pm_prethesis, AIPMResult, AIPreThesis
from ascent.risk.pm_risk_validator import validate as validate_pm_proposal
from memory.regime_memory import log_episode, update_outcomes
from ascent.strategy.earned_authority import blend as authority_blend, update_authority, get_state as get_authority_state
from ascent.strategy.thesis_formatter import format_thesis
from ascent.monitoring.ai_pm_counterfactual import (
    snapshot_quant_star, snapshot_quant, snapshot_ai_pm,
    score_daily as cf_score_daily, load_snapshots as cf_load_snapshots,
    print_cumulative_report as cf_print_report,
)
from ascent.strategy.ai_pm_perf_feedback import compute_feedback as compute_ai_feedback


SECTOR_OVERRIDE_LOG  = Path("logs/sector_override.jsonl")
HALT_STATE_PATH      = Path("execution/halt_state.json")
AI_PM_DECISION_LOG   = Path("logs/ai_pm_decision_log.jsonl")
AI_PM_DAILY_VIEWS    = Path("logs/ai_pm_daily_views.jsonl")


def _fetch_position_returns(symbols: list) -> dict:
    """Fetch today's price changes for held symbols. Returns {sym: pct_change}."""
    if not symbols:
        return {}
    try:
        import yfinance as yf
        df = yf.download(symbols, period="2d", auto_adjust=True,
                         progress=False, threads=True)
        if df.empty or len(df) < 2:
            return {}
        closes = df["Close"] if hasattr(df.columns, "levels") else df
        if closes.ndim == 1:
            closes = closes.to_frame(name=symbols[0])
        rets = closes.pct_change().iloc[-1].dropna()
        return {str(sym): round(float(r), 4) for sym, r in rets.items()}
    except Exception:
        return {}


def _run_daily_haiku_view(today, positions: list, feedback: dict) -> None:
    """Lightweight Haiku daily conviction update on non-rebalance days. ~$0.005/day."""
    try:
        from ascent.llm.client import HAIKU_MODEL
        import anthropic
        client = anthropic.Anthropic()

        level = feedback.get("level", 0)
        worst = feedback.get("worst_call_10d") or {}
        worst_str = (f"{worst.get('symbol')} ({worst.get('alpha', 0):+.1%} over 10d)"
                     if worst.get("symbol") else "none")

        # Fetch actual today's returns for held positions
        syms = [p["symbol"] for p in positions if p.get("symbol")]
        price_returns = _fetch_position_returns(syms)

        # Build position table with real price moves
        pos_lines = []
        for p in sorted(positions, key=lambda x: -x.get("weight", 0)):
            sym = p.get("symbol", "")
            w   = p.get("weight", 0)
            ret = price_returns.get(sym)
            ret_str = f"{ret:+.2%}" if ret is not None else "N/A"
            pos_lines.append(f"  {sym:6s} {w:.1%}  today: {ret_str}")

        pos_table = "\n".join(pos_lines) or "none"

        # Macro context
        spy_ret = price_returns.get("SPY")
        if not spy_ret:
            spy_prices = _fetch_position_returns(["SPY"])
            spy_ret = spy_prices.get("SPY")
        spy_str = f"{spy_ret:+.2%}" if spy_ret is not None else "N/A"

        prompt = f"""Today: {today.isoformat()} | AI PM Level {level} | SPY: {spy_str}
Worst recent call: {worst_str}

HELD POSITIONS (symbol | weight | today's return):
{pos_table}

You have the actual price data above. Give a concise daily update:
1. Biggest mover today and the most likely reason (sector news, macro, earnings follow-through)
2. Any position that changed your conviction — bullish, bearish, or watch-closely
3. One risk to monitor before the next rebalance

Be specific. Cite the return numbers from the table. No vague statements."""

        resp = client.messages.create(
            model=HAIKU_MODEL, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        view_text = resp.content[0].text if resp.content else ""

        AI_PM_DAILY_VIEWS.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_PM_DAILY_VIEWS, "a") as f:
            f.write(json.dumps({
                "date":          today.isoformat(),
                "level":         level,
                "price_returns": price_returns,
                "view":          view_text,
            }) + "\n")
        print(f"[Runner] AI PM daily view logged (Haiku, {len(view_text)} chars)")
        print(f"[Runner] AI PM view: {view_text[:200].strip()}...")
    except Exception as e:
        print(f"[Runner] AI PM daily view skipped: {e}")


def _write_decision_log(today, ai_pm_result, quant_weights: dict,
                        blended_weights: dict, authority_state: dict,
                        phase2_model: str = "claude-sonnet-4-6") -> None:
    """Write one entry to ai_pm_decision_log.jsonl per rebalance day."""
    try:
        AI_PM_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
        overrides = []
        if ai_pm_result and ai_pm_result.thesis:
            overrides = ai_pm_result.thesis.get("quant_overrides", [])
        entry = {
            "date":                  today.isoformat(),
            "level":                 authority_state.get("level", 0),
            "title":                 authority_state.get("title", "Shadow"),
            "ai_weight":             authority_state.get("ai_weight", 0.0),
            "phase2_model":          phase2_model,
            "perf_feedback_injected": Path("data_cache/ai_pm_perf_feedback.json").exists(),
            "quant_proposed":        {k: round(v, 6) for k, v in quant_weights.items()},
            "ai_pm_proposed":        {k: round(v, 6) for k, v in (ai_pm_result.portfolio if ai_pm_result and not ai_pm_result.fallback else {}).items()},
            "overrides_applied":     overrides,
            "final_blended":         {k: round(v, 6) for k, v in blended_weights.items()},
            "thesis_summary":        str((ai_pm_result.thesis or {}).get("market_view", ""))[:200] if ai_pm_result and ai_pm_result.thesis else "",
        }
        with open(AI_PM_DECISION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Runner] AI PM decision logged (Level {entry['level']}, {len(overrides)} overrides)")
    except Exception as e:
        print(f"[Runner] Decision log skipped: {e}")
HALT_OVERRIDE_PATH  = Path("execution/halt_override.json")
REGIME_SIGNAL_PATH  = Path("dashboard/regime_signal.json")
REGIME_STALE_DAYS   = 5

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


def _get_portfolio_symbols() -> list:
    """Return symbols with nonzero weight in the current merged portfolio."""
    try:
        p = Path("execution/merged_weights.json")
        if p.exists():
            weights = json.loads(p.read_text())
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


def _fill_wedge_and_decision_outcomes(as_of_date: str) -> None:
    """
    Fetch 21-day cumulative returns for symbols in pending alpha_wedge records,
    fill wedge_21d in alpha_wedge.jsonl, then propagate wedge to decision_memory.jsonl.

    Only runs for rebalances ≥30 calendar days old (giving market 21 trading days).
    No-op if alpha_wedge log doesn't exist or yfinance fetch fails.
    """
    from datetime import date as _date, timedelta as _td
    import json as _json
    from pathlib import Path as _Path

    wedge_log = _Path("logs/alpha_wedge.jsonl")
    if not wedge_log.exists():
        return

    rows = [_json.loads(l) for l in wedge_log.read_text().splitlines() if l.strip()]
    today = _date.fromisoformat(as_of_date)

    # Collect symbols from records ≥30 calendar days old with no wedge yet
    pending = [r for r in rows
               if r.get("wedge_21d") is None
               and (_date.fromisoformat(r["rebalance_date"]) + _td(days=30)) <= today]

    if not pending:
        return

    # Gather all symbols across pending records
    all_symbols: set = set()
    for r in pending:
        all_symbols.update(r.get("ai_pm_weights", {}).keys())
        all_symbols.update(r.get("quant_weights", {}).keys())

    if not all_symbols:
        return

    try:
        import yfinance as _yf
        syms = list(all_symbols)
        raw = _yf.download(syms, period="65d", auto_adjust=True, progress=False)
        if raw.empty:
            return
        import pandas as _pd
        closes = raw["Close"] if isinstance(raw.columns, _pd.MultiIndex) else raw
        if not isinstance(closes, _pd.DataFrame):
            closes = closes.to_frame()
    except Exception as _e:
        print(f"[WedgeFill] Price fetch failed: {_e}")
        return

    # Compute cumulative returns from each rebalance date
    from ascent.monitoring.alpha_wedge_tracker import update_outcomes as _aw_update
    from ascent.memory.decision_memory import update_outcomes as _dm_update

    for row in pending:
        rb_date_str = row["rebalance_date"]
        rb_date = _date.fromisoformat(rb_date_str)

        try:
            # Prices on rebalance day and 21 trading days later
            rb_close = closes[closes.index.date == rb_date]
            if rb_close.empty:
                # Try nearest date
                future = closes[closes.index.date >= rb_date]
                if future.empty:
                    continue
                rb_close = future.iloc[[0]]

            # Price 21+ calendar days after rebalance (use ~31cd as buffer)
            end_target = rb_date + _td(days=31)
            end_prices = closes[closes.index.date >= end_target]
            if end_prices.empty:
                end_target = rb_date + _td(days=28)
                end_prices = closes[closes.index.date >= end_target]
            if end_prices.empty:
                continue
            end_close = end_prices.iloc[[0]]

            price_rets = {}
            for sym in closes.columns:
                p0 = rb_close[sym].iloc[0] if sym in rb_close.columns else None
                p1 = end_close[sym].iloc[0] if sym in end_close.columns else None
                if p0 and p1 and float(p0) != 0:
                    price_rets[str(sym)] = float((p1 - p0) / p0)

            if not price_rets:
                continue

            # Fill alpha_wedge entry directly for this rebalance
            _aw_update(price_rets, as_of_date, lookback_days=60)

            # After fill, read back the wedge for this rebalance and propagate
            if wedge_log.exists():
                _updated = [_json.loads(l) for l in wedge_log.read_text().splitlines() if l.strip()]
                for _r in _updated:
                    if _r.get("rebalance_date") == rb_date_str and _r.get("wedge_21d") is not None:
                        _dm_update(rb_date_str, _r["wedge_21d"])
                        print(f"[WedgeFill] Propagated wedge {_r['wedge_21d']:+.3%} for {rb_date_str}")
                        break

        except Exception as _re:
            print(f"[WedgeFill] Could not fill {rb_date_str}: {_re}")


def main():
    dry_run             = "--dry-run" in sys.argv
    skip_sector_check   = "--skip-sector-check" in sys.argv
    today               = date.today()

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

    # Update calibration outcomes (best-effort)
    try:
        from ascent.strategy.calibration_tracker import update_outcomes as _update_cal
        _update_cal({}, str(date.today()))
    except Exception:
        pass

    # Fill 21d outcomes for alpha wedge + decision memory (best-effort)
    try:
        _fill_wedge_and_decision_outcomes(today.isoformat())
    except Exception as _fwd_e:
        print(f"[Runner] Wedge outcome fill failed: {_fwd_e}")

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

    # ── Alt data collection (runs before agents; each source fails silently) ──
    _collect_altdata(
        portfolio_symbols=_get_portfolio_symbols(),
        all_symbols=us_symbols,
    )

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

    # Early rebalance trigger: IC decay ≥30% since last rebalance after ≥5 bdays
    if not is_rebalance:
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

    def _get_current_regime() -> str:
        try:
            import json as _gj
            _gsig = _gj.loads(open("dashboard/regime_signal.json").read())
            if isinstance(_gsig, list):
                _gsig = _gsig[-1] if _gsig else {}
            return str(_gsig.get("label", "unknown")).lower()
        except Exception:
            return "unknown"

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

    # ── Monthly: investor report + audit integrity + methodology index ─────────
    try:
        from datetime import date as _mdate
        _today_m = _mdate.today()
        if _today_m.weekday() == 6 and _today_m.day <= 7:  # first Sunday of month
            # Monthly investor report
            try:
                from ascent.reporting.investor_report import schedule_monthly_report
                schedule_monthly_report()
                print("[InvestorReport] Monthly report generated.")
            except Exception as _ir_e:
                print(f"[InvestorReport] Skipped: {_ir_e}")

            # Audit trail integrity check
            try:
                import subprocess as _sp, sys as _sys
                _audit_result = _sp.run(
                    [_sys.executable, "scripts/verify_audit_trail.py"],
                    capture_output=True, text=True, timeout=30,
                )
                print(f"[AuditIntegrity] {'PASS' if _audit_result.returncode == 0 else 'FAIL'}")
            except Exception as _ai_e:
                print(f"[AuditIntegrity] Skipped: {_ai_e}")

            # Export methodology index
            try:
                from compliance.methodology_index import export_methodology_index
                export_methodology_index()
                print("[MethodologyIndex] Exported.")
            except Exception as _mi_e:
                print(f"[MethodologyIndex] Skipped: {_mi_e}")
    except Exception as _monthly_e:
        print(f"[Monthly] Plan 6/7 monthly tasks skipped: {_monthly_e}")

    # ── Daily: methodology index export + alert check ─────────────────────────
    try:
        from compliance.methodology_index import export_methodology_index as _export_mi
        _export_mi()
    except Exception:
        pass

    try:
        from ascent.monitoring.alert_system import check_alerts as _check_alerts
        _alert_list = _check_alerts()
        if _alert_list:
            print(f"[Alerts] {len(_alert_list)} new alert(s) fired.")
    except Exception:
        pass

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

    # ── Daily intelligence (non-rebalance days — feeds rebalance brief) ──────
    if not is_rebalance:
        try:
            from ascent.monitoring.daily_intelligence import run_daily_intelligence
            run_daily_intelligence(today.isoformat(), merged_weights, agent_outputs)
        except Exception as _di_e:
            print(f"[DailyIntel] Skipped: {_di_e}")

        # Adversarial monitor — lightweight non-rebalance scan
        try:
            from debate.adversarial_monitor import run_adversarial_monitor
            run_adversarial_monitor()
        except Exception as _am_e:
            print(f"[AdvMonitor] Skipped: {_am_e}")

        # Gate 4 — causal early exit check
        try:
            from ascent.causal.tracker import check_early_exits as _check_exits
            _early_exit_symbols = _check_exits()
            if _early_exit_symbols:
                print(f"[Causal] Early exit flagged for: {_early_exit_symbols}")
                import json as _json
                _shadow_path = Path("data_cache/ai_pm_shadow_returns.jsonl")
                _shadow_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_shadow_path, "a") as _sf:
                    for _sym in _early_exit_symbols:
                        _sf.write(_json.dumps({
                            "date": today.isoformat(),
                            "symbol": _sym,
                            "ai_pm_shadow_weight": 0.0,
                            "reason": "causal_mechanism_broken",
                        }) + "\n")
        except Exception as _ce:
            print(f"[Causal] Gate 4 early exit check failed: {_ce}")

    # ── Generate rebalance brief BEFORE AI PM so get_rebalance_brief tool reads current intel ──
    if is_rebalance:
        try:
            from ascent.monitoring.rebalance_brief import generate_rebalance_brief
            generate_rebalance_brief(today.isoformat())
            print("[RebalanceBrief] Brief generated for AI PM.")
        except Exception as _rb_e:
            print(f"[RebalanceBrief] Generation failed: {_rb_e}")

    # ── AI PM Phase 1: Pre-thesis (before quant agents on rebalance days) ────────
    # AI reads broadly and forms original investment thesis BEFORE seeing quant rankings.
    # This makes the fund genuinely AI-native: AI generates the thesis, quant validates it.
    _ai_prethesis: AIPreThesis | None = None
    if is_rebalance:
        # Inject pattern memory into environment so Phase 1 temporal context picks it up
        try:
            from ascent.strategy.ai_pm_learning import get_pattern_summary
            _pattern_summary = get_pattern_summary()
            if _pattern_summary:
                Path("data_cache/ai_pm_pattern_context.txt").write_text(_pattern_summary)
                print(f"[Runner] Pattern memory injected into Phase 1 context")
        except Exception:
            pass

        try:
            print("[Runner] AI PM Phase 1 — forming original thesis before quant runs...")
            _ai_prethesis = run_ai_pm_prethesis()
            if _ai_prethesis:
                syms = ", ".join(_ai_prethesis.conviction_symbols[:6])
                print(f"[Runner] Pre-thesis sealed: {len(_ai_prethesis.high_conviction_names)} "
                      f"conviction names ({syms}...)")
                # Write AI regime assessment + sleeve prior for main.py to pick up
                if _ai_prethesis.regime_assessment or _ai_prethesis.sleeve_weight_prior:
                    _assess_path = Path("data_cache/ai_regime_assessment.json")
                    try:
                        _assess_path.write_text(json.dumps({
                            **(_ai_prethesis.regime_assessment or {}),
                            "sleeve_weight_prior": _ai_prethesis.sleeve_weight_prior or {},
                            "as_of_date": today.isoformat(),
                        }))
                        print(f"[Runner] AI regime assessment written: "
                              f"{(_ai_prethesis.regime_assessment or {}).get('label', 'n/a')} "
                              f"sleeves={list((_ai_prethesis.sleeve_weight_prior or {}).keys())}")
                    except Exception as _ae:
                        print(f"[Runner] AI regime assessment write failed: {_ae}")
            else:
                print("[Runner] Pre-thesis returned None — synthesis will use standard mode")
        except Exception as _pt_e:
            print(f"[Runner] Pre-thesis failed ({_pt_e}) — continuing in standard mode")

    # ── AI PM Agent integration ────────────────────────────────────────────────
    # Track A★: snapshot BEFORE Phase 1 sleeve priors — true no-AI-PM baseline
    _quant_star_weights = dict(merged_weights)
    if is_rebalance:
        try:
            snapshot_quant_star(today, _quant_star_weights)
        except Exception as _cs_e:
            print(f"[Runner] Track A★ snapshot skipped: {_cs_e}")

    _quant_weights_snapshot = dict(merged_weights)  # updated after Phase 1 (Track A)
    _snap_ai_weights = None
    _phase2_model_used = "claude-sonnet-4-6"

    if not is_rebalance:
        # Non-rebalance: lightweight Haiku daily view
        try:
            _fb_data = json.loads(Path("data_cache/ai_pm_perf_feedback.json").read_text()) \
                if Path("data_cache/ai_pm_perf_feedback.json").exists() else {}
            _cur_pos = []
            try:
                from ascent.execution.alpaca_broker import get_positions as _gp
                _pos_df = _gp()
                if not _pos_df.empty:
                    _cur_pos = _pos_df[["symbol", "weight"]].to_dict("records")
            except Exception:
                pass
            # Fetch actual price returns for the intelligence brief
            _pos_returns = _fetch_position_returns([p["symbol"] for p in _cur_pos if p.get("symbol")])

            # Haiku view (lightweight, non-rebalance)
            _run_daily_haiku_view(today, _cur_pos, _fb_data)

            # Sonnet daily intelligence brief (richer analysis, thesis health, pattern memory)
            try:
                from ascent.strategy.ai_pm_learning import daily_intelligence_brief
                _brief = daily_intelligence_brief(
                    today=today,
                    positions=_cur_pos,
                    price_returns=_pos_returns,
                    feedback=_fb_data,
                )
                print(f"[Runner] AI PM intelligence brief written ({len(_brief)} chars, Sonnet)")
            except Exception as _ib_e:
                print(f"[Runner] Intelligence brief skipped: {_ib_e}")
        except Exception as _dv_e:
            print(f"[Runner] Daily view skipped: {_dv_e}")
    else:
        # Track A: snapshot AFTER Phase 1 sleeve priors applied, BEFORE Phase 2 blend
        _quant_weights_snapshot = dict(merged_weights)
        try:
            snapshot_quant(today, _quant_weights_snapshot)
        except Exception as _ca_e:
            print(f"[Runner] Track A snapshot skipped: {_ca_e}")

        # Smart Opus trigger: upgrade Phase 2 on high-stakes rebalances
        try:
            _current_regime = json.loads(Path("dashboard/regime_signal.json").read_text()).get("label", "") \
                if Path("dashboard/regime_signal.json").exists() else ""
            _last_regime = get_authority_state().get("last_regime", "")
            _use_opus = any([
                str(_current_regime).lower() in ("crisis",),          # always Opus in crisis
                _current_regime != _last_regime and _last_regime,      # regime change
                len(getattr(_ai_prethesis, "high_conviction_names", [])) >= 4,  # complex decision
                get_authority_state().get("in_cooldown") is False and
                get_authority_state().get("days_at_level", 99) == 0,   # first day post-promotion
            ])
            from ascent.llm.client import DEFAULT_MODEL, SONNET_MODEL
            _phase2_model_used = DEFAULT_MODEL if _use_opus else SONNET_MODEL
            if _use_opus:
                print(f"[Runner] Opus trigger: regime={_current_regime}, using {_phase2_model_used}")
        except Exception:
            from ascent.llm.client import SONNET_MODEL
            _phase2_model_used = SONNET_MODEL

        # Load causal track record for Phase 2 synthesis context
        _causal_track_record = None
        try:
            from ascent.causal.tracker import get_track_record as _get_track_record
            _causal_track_record = _get_track_record()
        except Exception:
            pass

        try:
            print("[Runner] AI PM Phase 2 — synthesising pre-thesis with quant validation...")
            ai_pm_result = run_ai_pm(
                quant_outputs=agent_outputs,
                merged_weights=merged_weights,
                prethesis=_ai_prethesis,
                causal_track_record=_causal_track_record,
            )

            ok = False
            violations = []
            if ai_pm_result.fallback:
                print("[Runner] AI PM fallback — using quant portfolio unchanged")
            else:
                ok, violations = validate_pm_proposal(ai_pm_result.portfolio)
                if ok:
                    ai_weight = get_authority_state().get("ai_weight", 0.0)
                    merged_weights = authority_blend(ai_pm_result.portfolio, merged_weights)
                    print(f"[Runner] AI PM blend applied (ai_weight={ai_weight * 100:.0f}%)")
                else:
                    print(f"[Runner] AI PM proposal rejected: {violations} — using quant 100%")
                _snap_ai_weights = dict(ai_pm_result.portfolio)  # capture for authority snapshot

                # Track D: snapshot pure AI PM portfolio (diagnostic)
                try:
                    snapshot_ai_pm(today, dict(ai_pm_result.portfolio))
                except Exception as _td_e:
                    print(f"[Runner] Track D snapshot skipped: {_td_e}")

                # Decision log: record what AI PM proposed, what was applied
                try:
                    _write_decision_log(
                        today, ai_pm_result, _quant_weights_snapshot,
                        merged_weights, get_authority_state(), _phase2_model_used,
                    )
                except Exception as _dl_e:
                    print(f"[Runner] Decision log skipped: {_dl_e}")

                format_thesis({**ai_pm_result.thesis, "ai_pm_portfolio": ai_pm_result.portfolio})

                # Log AI market character prediction for calibration tracking
                if _ai_prethesis and _ai_prethesis.market_character:
                    try:
                        from ascent.strategy.ai_calibration import log_thesis as _log_cal_thesis
                        _log_cal_thesis(
                            thesis_date=today.isoformat(),
                            regime=_get_current_regime(),
                            market_character=_ai_prethesis.market_character,
                            sleeve_weight_prior=_ai_prethesis.sleeve_weight_prior or {},
                        )
                        print(f"[Runner] Calibration: logged market_character="
                              f"{_ai_prethesis.market_character}")
                    except Exception as _cal_e:
                        print(f"[Runner] Calibration log failed: {_cal_e}")

                # Record AI PM vs quant wedge for feedback loop
                try:
                    from ascent.monitoring.alpha_wedge_tracker import record_rebalance as _record_wedge
                    _override_types = {
                        o.get("symbol", ""): o.get("override_type", "unknown")
                        for o in ai_pm_result.thesis.get("quant_overrides", [])
                        if o.get("symbol")
                    }
                    _record_wedge(
                        rebalance_date=today.isoformat(),
                        ai_pm_weights=ai_pm_result.portfolio,
                        quant_weights=_quant_weights_snapshot,
                        override_types=_override_types,
                    )
                    print("[Runner] Alpha wedge recorded")
                except Exception as _we:
                    print(f"[Runner] Alpha wedge record failed: {_we}")

                # Ingest each AI PM override into decision memory for future conviction gating
                try:
                    from ascent.memory.decision_memory import ingest_override as _ingest_dm
                    _dm_regime = _get_current_regime()
                    for _ov in ai_pm_result.thesis.get("quant_overrides", []):
                        _sym = _ov.get("symbol", "")
                        _ov_type = _ov.get("override_type", "")
                        if not _sym or not _ov_type:
                            continue
                        _ai_w = ai_pm_result.portfolio.get(_sym, 0.0)
                        _q_w = _quant_weights_snapshot.get(_sym, 0.0)
                        _mom = None
                        try:
                            from ascent.monitoring.conviction_tracker import get_position_momentum_safe
                            _mom = get_position_momentum_safe(_sym)
                        except Exception:
                            pass
                        _ingest_dm(
                            rebalance_date=today.isoformat(),
                            symbol=_sym,
                            override_type=_ov_type,
                            regime=_dm_regime,
                            ai_action=_ov.get("ai_action", ""),
                            ai_weight=_ai_w,
                            quant_weight=_q_w,
                            momentum_252d=_mom,
                        )
                    print("[Runner] Decision memory updated")
                except Exception as _dm_e:
                    print(f"[Runner] Decision memory update failed: {_dm_e}")

                try:
                    from compliance.audit_trail import record_event
                    record_event("ai_pm_proposal", {
                        "portfolio_size": len(ai_pm_result.portfolio),
                        "validated": ok if not ai_pm_result.fallback else False,
                        "violations": violations if not ai_pm_result.fallback and not ok else [],
                    })
                except Exception as ae:
                    print(f"[Runner] Audit trail write failed: {ae}")

        except Exception as exc:
            print(f"[Runner] AI PM agent failed: {exc} — using quant portfolio")

    # Log episode for regime-aware memory
    try:
        _episode_regime = _get_current_regime()
        _episode_ai_w = None
        try:
            if not ai_pm_result.fallback:
                _episode_ai_w = ai_pm_result.portfolio if ai_pm_result.portfolio else None
        except Exception:
            pass
        log_episode(
            run_date=today.isoformat(),
            regime=_episode_regime,
            quant_weights=merged_weights,
            ai_weights=_episode_ai_w,
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

    # ── Snapshot rebalance baseline for conviction tracker ───────────────────
    if is_rebalance:
        try:
            from ascent.monitoring.conviction_tracker import save_rebalance_alpha_state
            from ascent.monitoring.signal_health import compute_signal_health
            from ascent.monitoring.regime_trajectory import compute_regime_trajectory
            _sleeve_ics = {
                s: d.get("ic_5d_avg", 0.0)
                for s, d in compute_signal_health(today.isoformat()).items()
            }
            _traj = compute_regime_trajectory(today.isoformat())
            save_rebalance_alpha_state(
                date=today.isoformat(),
                merged_weights=merged_weights,
                agent_outputs=agent_outputs,
                sleeve_ics=_sleeve_ics,
                regime=_traj.get("current_label", "unknown"),
                regime_stability_10d=_traj.get("stability_10d", 0.5),
            )
        except Exception as _rs_e:
            print(f"[RebalanceState] Snapshot failed: {_rs_e}")

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

    # ── Post-rebalance: update meta-learner and calibration from holding-period sleeve IC ──
    if is_rebalance:
        try:
            from ascent.alpha.meta_learner import SleeveMetaLearner as _SML
            from ascent.strategy.ai_calibration import update_outcome as _update_cal_outcome

            _sleeve_ic_log = Path("logs/sleeve_ic_log.jsonl")
            _ml_snap_path = Path("data_cache/authority_rebalance_snapshot.json")
            _realized_ic: dict = {}

            if _sleeve_ic_log.exists() and _ml_snap_path.exists():
                _prev_snap = json.loads(_ml_snap_path.read_text())
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
                _update_cal_outcome(_realized_ic)
                print("[Runner] Calibration outcome updated")
            else:
                print("[Runner] Meta-learner: no IC data since prior rebalance — skipping")
        except Exception as _ml_upd_e:
            print(f"[Runner] Meta-learner/calibration update failed: {_ml_upd_e}")

    # ── Update earned authority (rebalance days only, full holding-period comparison) ──
    # Fair comparison: both AI PM and quant measured over the same holding period
    # on the same full multi-asset portfolio, not daily returns of stale weights.
    _AUTHORITY_SNAPSHOT = Path("data_cache/authority_rebalance_snapshot.json")
    if is_rebalance:
        try:
            import yfinance as _yf

            # Step 1: compute holding-period return vs previous rebalance snapshot
            if _AUTHORITY_SNAPSHOT.exists():
                _prev = json.loads(_AUTHORITY_SNAPSHOT.read_text())
                _prev_date = _prev["rebalance_date"]
                _prev_ai   = _prev["ai_weights"]
                _prev_qt   = _prev["quant_weights"]
                _all_syms  = list(set(_prev_ai) | set(_prev_qt))

                _px = _yf.download(_all_syms, start=_prev_date,
                                   end=today.isoformat(), auto_adjust=True, progress=False)
                if hasattr(_px.columns, "levels"):
                    _px = _px["Close"]

                if len(_px) >= 2:
                    _period_rets = (_px.iloc[-1] / _px.iloc[0] - 1).fillna(0)
                    # Clip per-symbol returns to ±50% to guard against bad price data
                    _period_rets = _period_rets.clip(-0.50, 0.50)

                    def _port_ret(weights):
                        tw = sum(weights.values()) or 1.0
                        return float(sum(
                            (w / tw) * float(_period_rets.get(s, 0))
                            for s, w in weights.items()
                        ))

                    _ai_ret = _port_ret(_prev_ai)
                    _qt_ret = _port_ret(_prev_qt)
                    update_authority(_ai_ret, _qt_ret)
                    print(f"[Runner] Authority updated: AI {_ai_ret*100:.2f}% vs Quant "
                          f"{_qt_ret*100:.2f}% ({_prev_date} → {today.isoformat()})")

            # Step 2: save snapshot for next rebalance comparison
            # Use AI PM portfolio if it ran successfully, else fall back to quant
            _snap = {
                "rebalance_date": today.isoformat(),
                "ai_weights":    _snap_ai_weights or _quant_weights_snapshot,
                "quant_weights": _quant_weights_snapshot,
            }
            _AUTHORITY_SNAPSHOT.write_text(json.dumps(_snap, indent=2))
            _ai_src = "AI PM" if _snap_ai_weights else "quant (AI PM unavailable)"
            print(f"[Runner] Authority snapshot saved — {_ai_src}, "
                  f"{len(_snap['ai_weights'])} AI / {len(_snap['quant_weights'])} quant positions")

        except Exception as exc:
            print(f"[Runner] Earned authority update failed: {exc}")
    else:
        print("[Runner] Authority update: waiting for next rebalance (rebalance-period comparison only)")

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
            "date":              today.isoformat(),
            "us_regime":         next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "us_equities" and ao.regime_signal), _saved_regime),
            "macro_regime":      next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "macro" and ao.regime_signal), "unknown"),
            "n_positions":       len(merged_weights),
            "allocation":        _orch_alloc or {ao.agent_id: round(_base_alloc.get(ao.agent_id, 0.0), 2)
                                 for ao in agent_outputs},
            "weights":           merged_weights,
            "causal_mechanisms": [
                (vars(m) if hasattr(m, "__dict__") else m)
                for m in (_ai_prethesis.causal_mechanisms if _ai_prethesis else [])
            ],
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

        # ── Apply ONE adversarial position change (if warranted and not halted) ──
        position_changes = verdict.get("position_changes", [])
        if position_changes:
            change   = position_changes[0]
            sym      = change.get("symbol", "")
            new_w    = float(change.get("new_weight", 0))
            old_w    = float(merged_weights.get(sym, 0))
            itype    = change.get("intervention_type", "adversarial_thesis")

            if sym in merged_weights and new_w < old_w and new_w >= 0.01:
                # Redistribute the freed weight proportionally to the other positions
                freed = old_w - new_w
                others = {s: w for s, w in merged_weights.items() if s != sym}
                other_total = sum(others.values())
                if other_total > 0:
                    for s in others:
                        merged_weights[s] += freed * (others[s] / other_total)
                merged_weights[sym] = new_w

                # Renormalize
                total = sum(w for w in merged_weights.values() if w > 0)
                if total > 0:
                    merged_weights = {s: max(0.0, w / total)
                                      for s, w in merged_weights.items()}

                print(f"\n[AdvInt] ADVERSARIAL INTERVENTION APPLIED:")
                print(f"  {sym}: {old_w:.1%} → {new_w:.1%} [{itype}]")
                print(f"  Reason: {change.get('reason', '')[:100]}")
                print(f"  10d Prediction: {change.get('prediction', '')[:100]}")

                # Log the intervention for authority tracking
                try:
                    from debate.adversarial_authority import record_intervention
                    record_intervention(
                        date_str=today.isoformat(),
                        symbol=sym,
                        intervention_type=itype,
                        from_weight=old_w,
                        to_weight=new_w,
                        prediction=change.get("prediction", ""),
                        regime=portfolio_state.get("us_regime", "unknown"),
                    )
                    print(f"[AdvInt] Intervention logged for 10-day outcome tracking")
                except Exception as _ai_e:
                    print(f"[AdvInt] Authority log failed: {_ai_e}")

                # Re-write merged_weights.json with the adjusted weights
                weights_path = Path("execution/merged_weights.json")
                with open(weights_path, "w") as f:
                    json.dump({
                        "date":         today.isoformat(),
                        "weights":      merged_weights,
                        "agents":       [ao.agent_id for ao in agent_outputs],
                        "adversarial_intervention": change,
                        "generated_at": datetime.now().isoformat(),
                    }, f, indent=2)
                print(f"[AdvInt] merged_weights.json updated with adversarial adjustment")
            else:
                print(f"[AdvInt] Position change for {sym} skipped "
                      f"(not in weights, not a reduction, or below 1% floor)")

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

    try:
        from ascent.llm.client import log_costs
        log_costs(today.isoformat())
    except Exception as e:
        print(f"[Runner] Cost log skipped ({e})")

    # Monthly investor letter — triggers on first trading day of each new month
    try:
        from ascent.reporting.investor_letter import generate_monthly_letter
        letter_path = generate_monthly_letter(today)
        if letter_path:
            print(f"[Letter] Monthly investor letter saved → {letter_path}")
    except Exception as _le:
        print(f"[Letter] Investor letter skipped: {_le}")

    # ── Post-rebalance post-mortem (fires ~21 days after each rebalance) ──────
    try:
        from ascent.strategy.ai_pm_learning import run_post_mortem, update_pattern_memory
        _fb_for_pm = json.loads(Path("data_cache/ai_pm_perf_feedback.json").read_text()) \
            if Path("data_cache/ai_pm_perf_feedback.json").exists() else {}
        _mortem = run_post_mortem(today, _fb_for_pm)
        if _mortem:
            print(f"[Runner] AI PM post-mortem written for past rebalance")
            update_pattern_memory(_mortem, today)
            print(f"[Runner] AI PM pattern memory updated")
    except Exception as _pm_e:
        print(f"[Runner] Post-mortem skipped: {_pm_e}")

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

        # ── Counterfactual daily scoring ─────────────────────────────────────
        try:
            _as_w, _a_w, _d_w = cf_load_snapshots()
            _cf_prices: dict = {}
            if _as_w:
                _cf_syms = list(set(_as_w) | set(_a_w or {}) | set(_d_w or {}))
                try:
                    import yfinance as _yf
                    _raw = _yf.download(_cf_syms, period="5d", auto_adjust=True, progress=False)
                    if not _raw.empty and len(_raw) >= 2:
                        _cls = _raw["Close"] if isinstance(_raw.columns, pd.MultiIndex) else _raw
                        for _sym in _cf_syms:
                            if _sym in _cls.columns:
                                _cf_prices[_sym] = {
                                    "prev": float(_cls[_sym].iloc[-2]),
                                    "curr": float(_cls[_sym].iloc[-1]),
                                }
                except Exception as _pfe:
                    pass  # price fetch failure — tracks computed without individual prices
            _cf_record = cf_score_daily(
                run_date=today,
                quant_star_weights=_as_w or None,
                quant_weights=_a_w or None,
                ai_pm_weights=_d_w or None,
                track_b_return=day_ret,
                spy_return=spy_ret,
                prices=_cf_prices,
            )
            cf_print_report()
        except Exception as _cfe:
            print(f"[Runner] Counterfactual scoring skipped: {_cfe}")

        # ── Daily learning brief ─────────────────────────────────────────────
        try:
            _fb = compute_ai_feedback()
            _auth_state = get_authority_state()
            # Update authority with today's Track D vs Track A★ returns
            _d_ret_today  = _cf_record.get("track_d_return", 0.0) if "_cf_record" in dir() else 0.0
            _as_ret_today = _cf_record.get("track_astar_return", 0.0) if "_cf_record" in dir() else 0.0
            update_authority(
                track_d_return=_d_ret_today,
                track_astar_return=_as_ret_today,
                n_decisions_evaluated=_fb.get("n_decisions_evaluated", 0),
                hit_rate=_fb.get("hit_rate_21d"),
                profit_factor=_fb.get("profit_factor"),
                fade_rate=_fb.get("fade_rate"),
            )
        except Exception as _fbe:
            print(f"[Runner] Feedback/authority update skipped: {_fbe}")

    except Exception as e:
        print(f"[Runner] Holdings log skipped ({e})")

    # ── Regenerate GitHub Pages dashboard ─────────────────────────────────────
    try:
        result = subprocess.run(
            [sys.executable, "scripts/generate_performance_page.py", "--push"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print("[Dashboard] GitHub Pages updated and pushed.")
        else:
            print(f"[Dashboard] Generator exited {result.returncode}: {result.stderr[-200:]}")
    except Exception as e:
        print(f"[Dashboard] Skipped ({e})")


if __name__ == "__main__":
    main()
