"""
ascent/monitoring/weekend_runner.py

Weekend intelligence pipeline orchestrator.

Runs once per weekend. Second call within the same ISO week exits immediately
with a clear message. Each job is independently wrapped — one failure never
blocks the rest.

Called by run_all_agents.py when date.today().weekday() >= 5.
"""

import json
import time
from datetime import date
from pathlib import Path
from typing import Callable, Optional

WEEKEND_STATE_PATH = Path("data_cache/weekend_run_state.json")
WEEKEND_LOG_PATH   = Path("logs/weekend_run.jsonl")

# Jobs that run at most once per weekend (expensive, idempotent within a weekend)
# Jobs not in this set run every time (data collection, debrief, scenarios)
_ONCE_PER_WEEKEND = {
    "ml_retrain",
    "factor_discovery",
    "self_improve",
    "ai_pm_research",
}


# ── State management ──────────────────────────────────────────────────────────

def _load_state() -> dict:
    if not WEEKEND_STATE_PATH.exists():
        return {}
    try:
        return json.loads(WEEKEND_STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    import tempfile, os
    WEEKEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=WEEKEND_STATE_PATH.parent, suffix=".tmp"))
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, WEEKEND_STATE_PATH)


def already_ran_this_weekend() -> Optional[str]:
    """
    Returns the run date string if this system already ran this weekend,
    None if it hasn't run yet this weekend.
    """
    today      = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    state = _load_state()
    if state.get("iso_year") == iso_year and state.get("iso_week") == iso_week:
        return state.get("run_date")
    return None


def _mark_ran(completed_jobs: list) -> None:
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    state = _load_state()
    _save_state({
        "run_date":       today.isoformat(),
        "iso_week":       iso_week,
        "iso_year":       iso_year,
        "completed_jobs": list(set(state.get("completed_jobs", []) + completed_jobs)),
    })


def _job_ran_this_weekend(job_name: str) -> bool:
    """True if this specific job completed during any run this weekend."""
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    state = _load_state()
    if state.get("iso_year") != iso_year or state.get("iso_week") != iso_week:
        return False
    return job_name in state.get("completed_jobs", [])


# ── Job runner ────────────────────────────────────────────────────────────────

def _run_job(name: str, fn: Callable, once_per_weekend: bool = False) -> bool:
    """
    Run a single weekend job. Returns True on success.
    If once_per_weekend=True and the job already ran this weekend, skip it.
    """
    if once_per_weekend and _job_ran_this_weekend(name):
        print(f"[Weekend] Skipping {name} — already ran this weekend")
        return True

    print(f"\n[Weekend] ── Starting: {name} ─────────────────────────")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"[Weekend] ✓ {name} completed in {elapsed:.0f}s")
        _log_job(name, "success", elapsed)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Weekend] ✗ {name} FAILED ({type(e).__name__}: {e})")
        _log_job(name, "failed", elapsed, str(e))
        return False


def _log_job(name: str, status: str, elapsed: float, error: str = "") -> None:
    WEEKEND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "date": date.today().isoformat(),
        "job": name,
        "status": status,
        "elapsed_s": round(elapsed, 1),
    }
    if error:
        entry["error"] = error
    with open(WEEKEND_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Individual job implementations ────────────────────────────────────────────

def _job_altdata_full_sweep(all_symbols: list, portfolio_symbols: list) -> None:
    """Full 901-symbol alt-data sweep across all 4 sources."""
    from ascent.data.ingest.sec_filings import update_sec_signals
    from ascent.data.ingest.earnings_transcripts import (
        fetch_recent_8k_transcripts, update_transcript_signals,
    )
    from ascent.data.ingest.google_trends import update_trends_signals

    _ETF_SET = {"ETF", "EWY", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "UUP",
                "HYG", "LQD", "TIP", "SGOV", "BIL", "DBB", "KMLM", "PDBC",
                "DBA", "IFRA", "VNQ", "VIXY", "EWT", "EWZ", "EWC", "EWJ",
                "EWG", "EWU", "EWY", "AAXJ", "INDA", "VWO"}
    equity_all = [s for s in all_symbols if s not in _ETF_SET]

    print(f"[AltData/Weekend] SEC sweep: {len(equity_all)} equity symbols")
    update_sec_signals(equity_all)

    print(f"[AltData/Weekend] Transcript sweep: {len(equity_all)} equity symbols")
    update_transcript_signals(fetch_recent_8k_transcripts(equity_all, lookback_days=14))

    print(f"[AltData/Weekend] Trends sweep: {len(all_symbols)} symbols "
          f"(priority: {len(portfolio_symbols)} portfolio, cap 30 min)")
    update_trends_signals(
        all_symbols,
        priority_symbols=portfolio_symbols,
        max_minutes=30.0,
    )


def _job_llm_fundamental_cache(all_symbols: list) -> None:
    """Update LLM fundamental cache for stale/missing symbols."""
    import pandas as pd
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    from ascent.data.store.parquet import load_parquet

    try:
        fundamentals = load_parquet("fundamentals")
    except Exception:
        fundamentals = pd.DataFrame()

    if fundamentals.empty:
        print("[LLMFundamental/Weekend] No fundamentals data — skipping cache update")
        return

    equity_symbols = set(all_symbols)
    if "symbol" in fundamentals.columns:
        fundamentals = fundamentals[fundamentals["symbol"].isin(equity_symbols)]

    n = len(fundamentals["symbol"].unique()) if "symbol" in fundamentals.columns else 0
    print(f"[LLMFundamental/Weekend] Updating cache for {n} symbols")
    llm_fundamental_alpha(fundamentals)


def _job_ml_retrain() -> None:
    """Force ML sleeve retrain with GridSearch hyperparameter tuning."""
    import os
    from pathlib import Path as _Path

    # Delete stale cache files to force retrain on next agent run
    cache_dir = _Path("data_cache")
    removed = []
    for f in cache_dir.glob("ml_model_*.pkl"):
        f.unlink()
        removed.append(f.name)

    if removed:
        print(f"[ML/Weekend] Cleared {len(removed)} cached models: {removed}")
        print("[ML/Weekend] Models will retrain with GridSearch on next agent run")
    else:
        print("[ML/Weekend] No cached models found — will train fresh on next run")

    # Write a retrain flag that the ML sleeve can detect
    flag_path = _Path("data_cache/ml_retrain_requested.json")
    flag_path.write_text(json.dumps({
        "requested": True,
        "requested_at": date.today().isoformat(),
        "use_gridsearch": True,
    }))
    print("[ML/Weekend] GridSearch retrain flag written")


def _job_factor_discovery() -> None:
    """Run factor discovery: PySR + LLM template proposals."""
    from ascent.research.factor_discovery.discovery_runner import run_factor_discovery
    run_factor_discovery()


def _job_self_improve(n_variants: int = 20) -> None:
    """Run self-improve loop with expanded variant search."""
    from ascent.research import self_improve
    # Temporarily expand variant count for weekend run
    original_n = self_improve.N_VARIANTS
    self_improve.N_VARIANTS = n_variants
    try:
        self_improve.run_self_improve()
    finally:
        self_improve.N_VARIANTS = original_n


def _job_conviction_retrain() -> None:
    """Warm conviction gate model on latest matured decision history."""
    import json
    from pathlib import Path as _Path
    from ascent.strategy.conviction_gate import _get_ml_model

    records_path = _Path("logs/conviction_outcomes.jsonl")
    if not records_path.exists():
        print("[ConvictionGate/Weekend] No outcomes log yet — skipping")
        return

    records = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()]
    matured = [r for r in records if r.get("outcome_realized")]
    model = _get_ml_model(matured)
    if model is not None:
        print(f"[ConvictionGate/Weekend] Model warmed on {len(matured)} matured records")
    else:
        print(f"[ConvictionGate/Weekend] Insufficient matured data ({len(matured)}/30) — skipping")


def _job_adversarial_calibration() -> None:
    """Score pending adversarial interventions and update earned authority per intervention type."""
    from debate.adversarial_authority import score_pending_interventions, get_calibration_report
    n_scored = score_pending_interventions()
    report   = get_calibration_report()
    print(f"[AdvAuth/Weekend] Scored {n_scored} intervention(s) against 10-day outcomes")
    print(f"[AdvAuth/Weekend] {report}")


def _job_ai_pm_research(all_symbols: list) -> None:
    """
    AI PM weekend deep research session.
    Runs pre-thesis to form a fresh thesis before Monday's quant run.
    Result cached so Monday's run_all_agents.py skips Phase 1 if fresh.
    """
    import tempfile, os
    from agents.ai_pm_agent import run_ai_pm_prethesis
    from dataclasses import asdict

    prethesis = run_ai_pm_prethesis()
    if prethesis is None:
        print("[AIPMResearch] Pre-thesis returned None — skipping memo write")
        return

    memo = {
        "as_of_date": date.today().isoformat(),
        "macro_view": prethesis.macro_view,
        "regime_interpretation": prethesis.regime_interpretation,
        "high_conviction_names": prethesis.high_conviction_names,
        "names_to_avoid": prethesis.names_to_avoid,
        "sector_tilts": prethesis.sector_tilts,
        "regime_assessment": prethesis.regime_assessment,
        "sleeve_weight_prior": prethesis.sleeve_weight_prior,
        "market_character": prethesis.market_character,
    }
    out_path = Path("data_cache/weekend_research.json")
    tmp = Path(tempfile.mktemp(dir=out_path.parent, suffix=".tmp"))
    tmp.write_text(json.dumps(memo, indent=2))
    os.replace(tmp, out_path)
    n = len(prethesis.high_conviction_names)
    print(f"[AIPMResearch] Pre-thesis written: {n} high-conviction names, "
          f"regime={prethesis.regime_assessment.get('label','?')}")


def _job_weekly_debrief() -> dict:
    from ascent.monitoring.weekly_debrief import run_weekly_debrief
    return run_weekly_debrief()


def _job_scenario_planning(debrief: Optional[dict] = None) -> None:
    from ascent.monitoring.scenario_planner import run_scenario_planning
    run_scenario_planning(debrief=debrief)


def _job_memory_ingestion() -> None:
    """Ingest week's logs into BM25/R2R semantic memory."""
    try:
        from memory.r2r_interface import ingest_logs
        ingest_logs()
        print("[Memory/Weekend] Log ingestion complete")
    except Exception as e:
        print(f"[Memory/Weekend] Ingestion skipped: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_weekend(dry_run: bool = False) -> None:
    """
    Main weekend pipeline. Called from run_all_agents.py when it's a weekend.
    """
    from ascent.config.settings import UniverseConfig

    today = date.today()
    day_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][today.weekday()]

    print(f"\n{'#'*60}")
    print(f"# ASCENT CAPITAL — Weekend Intelligence Run")
    print(f"# {day_name} {today}  |  {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'#'*60}\n")

    all_symbols       = UniverseConfig().symbols
    portfolio_symbols  = _load_portfolio_symbols()

    print(f"[Weekend] Universe: {len(all_symbols)} symbols | "
          f"Portfolio: {len(portfolio_symbols)} positions")
    print(f"[Weekend] Est. cost: ~$1.50 | Est. time: ~3.5 hours\n")

    completed = []

    # 1. Alt-data full sweep — every run (new filings accumulate)
    if _run_job("altdata_full_sweep",
                lambda: _job_altdata_full_sweep(all_symbols, portfolio_symbols),
                once_per_weekend=False):
        completed.append("altdata_full_sweep")

    # 2. LLM fundamental cache — every run (new earnings stale entries)
    if _run_job("llm_fundamental_cache",
                lambda: _job_llm_fundamental_cache(all_symbols),
                once_per_weekend=False):
        completed.append("llm_fundamental_cache")

    # 3. ML retrain — once per weekend
    if _run_job("ml_retrain", _job_ml_retrain, once_per_weekend=True):
        completed.append("ml_retrain")

    # 4. Factor discovery — once per weekend
    if _run_job("factor_discovery", _job_factor_discovery, once_per_weekend=True):
        completed.append("factor_discovery")

    # 5. Self-improve (20 variants) — once per weekend
    if _run_job("self_improve",
                lambda: _job_self_improve(n_variants=20),
                once_per_weekend=True):
        completed.append("self_improve")

    # 6. Conviction gate retrain — every run (new matured records may have arrived)
    if _run_job("conviction_retrain", _job_conviction_retrain, once_per_weekend=False):
        completed.append("conviction_retrain")

    # 6b. Adversarial intervention calibration — every run
    if _run_job("adversarial_calibration", _job_adversarial_calibration, once_per_weekend=False):
        completed.append("adversarial_calibration")

    # 7. AI PM deep research — once per weekend
    if _run_job("ai_pm_research",
                lambda: _job_ai_pm_research(all_symbols),
                once_per_weekend=True):
        completed.append("ai_pm_research")

    # 8. Weekly debrief — every run (captures any new data from earlier jobs)
    debrief = None
    if not dry_run:
        ok = _run_job("weekly_debrief",
                      lambda: _job_weekly_debrief(),
                      once_per_weekend=False)
        if ok:
            completed.append("weekly_debrief")
            try:
                debrief = json.loads(Path("data_cache/weekly_debrief.json").read_text())
            except Exception:
                pass

    # 9. Adversarial scenario planning — every run
    if not dry_run:
        if _run_job("scenario_planning",
                    lambda: _job_scenario_planning(debrief),
                    once_per_weekend=False):
            completed.append("scenario_planning")

    # 10. Memory ingestion — every run
    if _run_job("memory_ingestion", _job_memory_ingestion, once_per_weekend=False):
        completed.append("memory_ingestion")

    # Mark this weekend as done
    _mark_ran(completed)

    print(f"\n{'#'*60}")
    print(f"# Weekend run complete: {len(completed)}/{11} jobs succeeded")
    print(f"# Completed: {', '.join(completed)}")
    print(f"# Monday open: models retrained, 901-symbol alt-data fresh,")
    print(f"#              debrief + scenario plan ready in data_cache/")
    print(f"{'#'*60}\n")


def _load_portfolio_symbols() -> list:
    """Load current portfolio symbols from merged_weights.json."""
    try:
        p = Path("execution/merged_weights.json")
        if p.exists():
            weights = json.loads(p.read_text())
            return [s for s, w in weights.items()
                    if isinstance(w, (int, float)) and w > 0]
    except Exception:
        pass
    return []
