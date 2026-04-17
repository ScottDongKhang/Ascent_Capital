# Ascent Capital — System Hardening Design

**Date**: 2026-04-11  
**Status**: Approved, not yet implemented  
**Scope**: 8 architectural fixes across 4 phases  
**Priority order**: Quick wins first, then operational, then signal quality, then portfolio architecture  
**Implementation rule**: Do not implement Phase 4 until Phases 1–3 are complete and live for 30+ days

---

## Context

This spec was produced via full codebase audit (2026-04-11). The system is live on Alpaca paper trading with real portfolio tracking. All fixes must preserve existing integrity constraints (no look-ahead bias, no simulated data under live cache names, debate advisory only, approval gate for large trades).

Do not implement any of these changes until the self-improve / self-learning system is finalized separately. These fixes harden the existing system — they are not the self-improve roadmap.

---

## Phase 1 — Quick Wins

### Fix 1a: Skill Score Lag (1 hour)

**Problem**: `export_skill_scores()` runs after `run_orchestrator()` in `run_all_agents.py`. Orchestrator reads `dashboard/agent_skill_scores.json` written the previous day — always 1 day stale.

**Root cause**: execution order bug, not a data availability problem. The PnL logs needed are already on disk.

**Fix**: Reorder three calls in `run_all_agents.py`:

```python
# CORRECT ORDER — enforce strictly, do not parallelize these steps
run_forward_pnl_cycle(agent_outputs, today)   # step 1: writes today's PnL log entry
export_skill_scores()                          # step 2: reads 63-day window incl. today
run_orchestrator(agent_outputs)                # step 3: reads fresh scores
```

Step 2 depends on step 1. Step 3 depends on step 2. These are sequential dependencies — never run in a thread pool.

**Additional guard**: add `skill_score_as_of: str` (ISO date) field to `dashboard/agent_skill_scores.json`. In `run_orchestrator()`, before reading skill scores:

```python
scores = json.loads(Path("dashboard/agent_skill_scores.json").read_text())
as_of = scores.get("skill_score_as_of", "")
today_str = today.isoformat()

# Allow 1-day buffer for weekends (Friday scores valid on Monday)
if as_of and (pd.Timestamp(today_str) - pd.Timestamp(as_of)).days > 1:
    log.warning(
        f"[Orch] Skill scores are {(pd.Timestamp(today_str)-pd.Timestamp(as_of)).days}d stale "
        f"(as_of={as_of}). Falling back to base allocation."
    )
    return BASE_ALLOCATION  # stale scores worse than no scores
```

**Files touched**: `run_all_agents.py`, `ascent/monitoring/skill_tracker.py` (add `skill_score_as_of` field), `orchestrator/central_intelligence.py` (add stale check)

---

### Fix 1b: Sector Constraints Silent Failure (1 day)

**Problem**: When `profiles.parquet` is missing or sector coverage < 80%, `sector_constrained_weighted()` in `ascent/portfolio/optimizer.py` silently falls back to uncapped rank weighting. Portfolio can hold 40%+ in one sector with no alert.

**Fix — Part A**: Add startup validation to `run_all_agents.py`, runs before `ThreadPoolExecutor` spawns agents. If validation fails, the entire process aborts — not just one agent thread.

```python
# ascent/portfolio/optimizer.py — new exception
class SectorDataError(RuntimeError):
    pass

# run_all_agents.py — call at very start of main(), before ThreadPoolExecutor
def validate_sector_data(symbols: list[str], skip: bool = False) -> None:
    """
    Validates profiles.parquet exists and covers >= 80% of universe.
    Aborts entire process if check fails (not just one agent thread).
    
    Args:
        symbols: full US equities universe from UniverseConfig.symbols
        skip: if True (--skip-sector-check flag), log override and return
    """
    if skip:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "sector_check_skipped",
            "required_reason": "see CLI flag --skip-sector-check",
        }
        Path("logs/sector_override.jsonl").open("a").write(json.dumps(entry) + "\n")
        log.warning("[Startup] Sector check SKIPPED — override logged")
        return

    if not has_data("profiles"):
        raise SectorDataError(
            "profiles.parquet missing. Regenerate with:\n"
            "  .venv/bin/python -m ascent.data.ingest.profiles\n"
            "Or bypass with --skip-sector-check (logged)."
        )

    profiles = load_parquet("profiles")
    known = set(profiles["symbol"].dropna())

    # Count None / "Unknown" / NaN as unknown — no silent coercion
    unknown_sectors = profiles[
        profiles["sector"].isna() | profiles["sector"].isin(["Unknown", "unknown", ""])
    ]["symbol"].tolist()

    missing_from_profiles = [s for s in symbols if s not in known]
    total_unknown = len(set(missing_from_profiles + unknown_sectors))
    coverage = 1.0 - total_unknown / len(symbols)

    if coverage < 0.80:
        raise SectorDataError(
            f"Sector coverage {coverage:.1%} < 80% threshold.\n"
            f"Missing from profiles: {missing_from_profiles[:20]}"
            f"{'...' if len(missing_from_profiles) > 20 else ''}\n"
            f"Unknown sectors: {unknown_sectors[:10]}"
            f"{'...' if len(unknown_sectors) > 10 else ''}\n"
            f"Regenerate profiles or bypass with --skip-sector-check (logged)."
        )

    log.info(f"[Startup] Sector data valid — coverage {coverage:.1%} ({len(known)} symbols)")
```

**Fix — Part B**: Remove the silent fallback inside `sector_constrained_weighted()`. The existing fallback comment says "skip caps, log warning" — replace with hard raise.

```python
# ascent/portfolio/optimizer.py — inside sector_constrained_weighted()
# BEFORE (silent degradation):
if coverage < 0.80:
    log.warning("Low sector coverage — skipping sector caps")
    return rank_weighted(alpha, n=top_n, ...)

# AFTER (hard fail — startup check should have caught this):
if coverage < 0.80:
    raise SectorDataError(
        f"sector_constrained_weighted(): coverage {coverage:.1%} < 80% "
        f"at portfolio construction time. This should have been caught at startup."
    )
```

**Files touched**: `ascent/portfolio/optimizer.py`, `run_all_agents.py`  
**New log**: `logs/sector_override.jsonl`

---

### Fix 1c: Debate Halt Non-Blocking (1 day)

**Problem**: `halt_and_review` verdict stops execution on day T. On day T+1, a fresh debate runs with no memory of the halt. A bullish day T+1 produces `proceed` and execution continues — the halt was silently overridden by market conditions.

**Fix**: Persistent halt state requiring explicit human override file.

**New file: `execution/halt_state.json`** — written by `debate_runner.py` when halt issued:

```json
{
  "halted": true,
  "halt_date": "2026-04-15",
  "reason": "Bear agent flagged 40% sector concentration in energy",
  "key_risks": [
    "Energy sector at 38% of portfolio",
    "Oil supply shock risk elevated"
  ],
  "verdict_path": "outputs/debate_log/verdict_2026-04-15.json",
  "requires_override": true,
  "created_at": "2026-04-15T13:47:22"
}
```

**New file: `execution/halt_override.json`** — created manually by human to resume:

```json
{
  "override_date": "2026-04-16",
  "override_by": "scott",
  "reason": "Reviewed energy position — acceptable given Q1 earnings catalyst",
  "acknowledged_risks": [
    "Energy sector at 38% of portfolio",
    "Oil supply shock risk elevated"
  ]
}
```

**New function in `run_all_agents.py`** — called before debate on every rebalance day:

```python
def check_halt_state() -> bool:
    """
    Returns True if execution may proceed, False if halted.
    
    Edge cases handled:
    - No halt_state.json: return True (no halt active)
    - Halt present, no override: block, log, return False
    - Override predates halt: invalid, block, return False  
    - Override present and valid: clear both files, log, return True
    - Override clears old halt — does NOT immunize against new halts from today's debate
    """
    halt_path = Path("execution/halt_state.json")
    override_path = Path("execution/halt_override.json")

    if not halt_path.exists():
        return True  # no active halt

    halt = json.loads(halt_path.read_text())

    if not halt.get("requires_override", True):
        # Informational halt only (future use) — auto-clear
        halt_path.unlink(missing_ok=True)
        return True

    if not override_path.exists():
        log.error(
            f"[HALT] System halted since {halt['halt_date']}: {halt['reason']}\n"
            f"[HALT] Create execution/halt_override.json to resume trading.\n"
            f"[HALT] See verdict: {halt['verdict_path']}"
        )
        _log_halt_skip(halt)
        return False

    override = json.loads(override_path.read_text())

    # Validate override is newer than halt
    if override.get("override_date", "") < halt.get("halt_date", ""):
        log.error(
            f"[HALT] Override date {override['override_date']} predates "
            f"halt date {halt['halt_date']} — invalid override. Recreate the file."
        )
        return False

    # Valid override — clear both files and log
    _log_halt_override(halt, override)
    halt_path.unlink(missing_ok=True)
    override_path.unlink(missing_ok=True)
    log.info(f"[HALT] Override accepted by {override['override_by']} — halt cleared")
    return True  # proceed — NOTE: today's debate may still issue a new halt
```

**Behavior when halted**: agents still run (PnL tracked, skill scores updated), orchestrator still runs, only execution is blocked. This preserves the forward PnL record through the halt period.

**Files touched**: `run_all_agents.py`, `debate/debate_runner.py` (write `halt_state.json` on `halt_and_review`)  
**New files**: `execution/halt_state.json` (runtime), `execution/halt_override.json` (runtime, human-created)

---

## Phase 2 — Operational Improvements

### Fix 2a: Async Approval Gate + Resume on Restart (2 days)

**Problem**: Current approval polling is a blocking `while waited < max_wait: time.sleep(30)` loop. Holds main thread for up to 30 minutes. If the machine sleeps during the wait, the timer is wrong. If the process restarts (crash, launchd restart), the approval state is lost and orders are resubmitted as duplicates.

**Fix**: Non-blocking `threading.Event` gate + persistent approval state.

**New persistent state file: `execution/approval_pending.json`**:

```json
{
  "created_at": "2026-04-15T13:47:22",
  "expires_at": "2026-04-15T14:17:22",
  "trades": [
    {"symbol": "AAPL", "side": "buy", "dollar_amount": 5200.0, "pct_nav": 0.052}
  ],
  "status": "pending"
}
```

**New function in `ascent/execution/approval_server.py`**:

```python
def wait_for_approval_async(
    pending_trades: list,
    timeout_seconds: int = 1800,
    poll_interval: int = 30,
    resume: bool = False,
    pending: dict = None,
) -> ApprovalResult:
    """
    Non-blocking approval gate using threading.Event.
    Persists state to execution/approval_pending.json so process restarts
    don't cause duplicate order submission.
    
    Args:
        pending_trades: list of Order objects awaiting approval
        timeout_seconds: max wait time (default 30 min)
        poll_interval: seconds between status checks (default 30)
        resume: if True, resuming after process restart (use pending dict)
        pending: previously persisted pending state (used when resume=True)
    """
    pending_path = Path("execution/approval_pending.json")
    heartbeat_path = Path("logs/approval_heartbeat.jsonl")
    approval_event = threading.Event()
    result: dict = {"status": "pending"}

    if not resume:
        # Write persistent state before starting
        state = {
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat(),
            "trades": [order_to_dict(t) for t in pending_trades],
            "status": "pending",
        }
        pending_path.write_text(json.dumps(state, indent=2))
        write_pending_trades(pending_trades)
    else:
        # Resume: recalculate remaining timeout
        expires_at = pd.Timestamp(pending["expires_at"])
        timeout_seconds = max(0, int((expires_at - pd.Timestamp.now()).total_seconds()))
        if timeout_seconds == 0:
            return ApprovalResult(status="timeout", trades=pending_trades)

    def _watcher():
        elapsed = 0
        while elapsed < timeout_seconds:
            status = check_approval_status()
            if status in ("approved", "rejected"):
                result["status"] = status
                approval_event.set()
                return
            time.sleep(poll_interval)
            elapsed += poll_interval
            # Heartbeat every 5 minutes
            if elapsed % 300 == 0:
                heartbeat = {
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": timeout_seconds - elapsed,
                    "status": "waiting",
                }
                heartbeat_path.open("a").write(json.dumps(heartbeat) + "\n")
                log.info(f"[Approval] Waiting... {elapsed//60}m elapsed, "
                         f"{(timeout_seconds-elapsed)//60}m remaining")
        result["status"] = "timeout"
        approval_event.set()

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()
    approval_event.wait(timeout=timeout_seconds + 10)
    watcher.join(timeout=5)

    # Clear persistent state
    pending_path.unlink(missing_ok=True)

    return ApprovalResult(status=result["status"], trades=pending_trades)
```

**Resume check at start of `run_eod_with_weights()`**:

```python
# At top of run_eod_with_weights(), before computing new orders:
pending_path = Path("execution/approval_pending.json")
if pending_path.exists():
    pending = json.loads(pending_path.read_text())
    expires_at = pd.Timestamp(pending["expires_at"])
    if expires_at > pd.Timestamp.now():
        log.info("[EOD] Resuming pending approval from previous run")
        result = wait_for_approval_async(
            pending_trades=[], resume=True, pending=pending
        )
        if result.status != "approved":
            log.warning(f"[EOD] Resumed approval {result.status} — aborting execution")
            pending_path.unlink(missing_ok=True)
            return
        # Approved — proceed with the persisted trades (don't recompute)
        orders = [dict_to_order(t) for t in pending["trades"]]
        # Jump directly to order submission, skip recompute
        _submit_orders(orders)
        return
    else:
        log.warning("[EOD] Stale approval_pending.json found (expired) — clearing")
        pending_path.unlink(missing_ok=True)
```

**Files touched**: `ascent/execution/eod_runner.py`, `ascent/execution/approval_server.py`  
**New files**: `execution/approval_pending.json` (runtime), `logs/approval_heartbeat.jsonl`

---

### Fix 2b: Transaction Cost Model — Almgren-Chriss (3 days)

**Problem**: Orders sized as `shares = dollar_amount / price`. No spread cost applied in live execution (BacktestConfig has `spread_bps=5.0` but it is never used in `order_engine.py`). No market impact model. Large orders in illiquid names move the market against the trade.

**New module**: `ascent/execution/cost_model.py`

**Model**: Simplified Almgren-Chriss with separately calibrated temporary and permanent impact:

```
spread_cost_bps   = spread_bps / 2                          # half-spread on entry (~2.5 bps)
temporary_impact  = eta * sigma_daily * sqrt(X / V_daily)   # recovers post-trade
permanent_impact  = gamma * sigma_daily * (X / V_daily)     # does not recover

total_one_way_bps = spread_cost_bps + temporary_impact/2 + permanent_impact
```

Where:

- `X` = trade size in dollars
- `V_daily` = `dollar_vol_21d / 21` (21-day average daily dollar volume ÷ 21 trading days)
- `sigma_daily` = `vol_21d` feature (annualized) / sqrt(252)
- `eta` = temporary impact coefficient (default 0.10, calibrated weekly from slippage_log)
- `gamma` = permanent impact coefficient (default 0.05, calibrated weekly from slippage_log)

```python
# ascent/execution/cost_model.py

@dataclass
class CostEstimate:
    symbol: str
    spread_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_one_way_bps: float
    participation_rate: float       # X / V_daily
    flag: str | None                # "HIGH_IMPACT" | "SPLIT_RECOMMENDED" | "IMPACT_UNKNOWN" | None

@dataclass
class CostModelParams:
    eta: float = 0.10               # temporary impact coefficient
    gamma: float = 0.05             # permanent impact coefficient
    spread_bps: float = 5.0         # round-trip spread (from BacktestConfig)
    max_participation_rate: float = 0.10   # block orders > 10% of ADV
    split_participation_rate: float = 0.05 # recommend split if > 5% of ADV

PARAMS_PATH = Path("data_cache/cost_model_params.json")

def load_params() -> CostModelParams:
    if PARAMS_PATH.exists():
        return CostModelParams(**json.loads(PARAMS_PATH.read_text()))
    return CostModelParams()  # use defaults

def estimate(
    symbol: str,
    dollar_amount: float,
    features: dict,              # features dict from FeatureBuilder
    params: CostModelParams = None,
) -> CostEstimate:
    if params is None:
        params = load_params()

    # Get daily ADV
    dollar_vol_21d = features.get("dollar_vol_21d", {}).get(symbol)
    if dollar_vol_21d is None or dollar_vol_21d == 0:
        return CostEstimate(
            symbol=symbol,
            spread_bps=params.spread_bps / 2,
            temporary_impact_bps=float("nan"),
            permanent_impact_bps=float("nan"),
            total_one_way_bps=float("nan"),
            participation_rate=float("nan"),
            flag="IMPACT_UNKNOWN",
        )

    V_daily = dollar_vol_21d / 21
    X = abs(dollar_amount)
    participation = X / V_daily

    # Get daily vol
    vol_21d_annualized = features.get("vol_21d", {}).get(symbol, 0.20)
    sigma_daily = vol_21d_annualized / (252 ** 0.5)

    spread = params.spread_bps / 2
    temp_impact = params.eta * sigma_daily * (participation ** 0.5) * 10_000  # convert to bps
    perm_impact = params.gamma * sigma_daily * participation * 10_000

    total = spread + temp_impact / 2 + perm_impact

    flag = None
    if participation > params.max_participation_rate:
        flag = "HIGH_IMPACT"   # block this order
    elif participation > params.split_participation_rate:
        flag = "SPLIT_RECOMMENDED"

    return CostEstimate(
        symbol=symbol,
        spread_bps=spread,
        temporary_impact_bps=temp_impact,
        permanent_impact_bps=perm_impact,
        total_one_way_bps=total,
        participation_rate=participation,
        flag=flag,
    )
```

**Integration in `order_engine.py`**:

```python
# After computing orders list, before returning:
estimates = [cost_model.estimate(o.symbol, o.dollar_amount, features) for o in orders]

blocked = [e for e in estimates if e.flag == "HIGH_IMPACT"]
if blocked:
    log.warning(f"[Orders] {len(blocked)} orders blocked (participation > 10% ADV): "
                f"{[e.symbol for e in blocked]}")
    orders = [o for o, e in zip(orders, estimates) if e.flag != "HIGH_IMPACT"]
    estimates = [e for e in estimates if e.flag != "HIGH_IMPACT"]

total_cost_bps = sum(
    e.total_one_way_bps * abs(o.weight_delta)
    for e, o in zip(estimates, orders)
    if not math.isnan(e.total_one_way_bps)
)
log.info(f"[Orders] Estimated total round-trip cost: {total_cost_bps:.1f} bps")
```

**Weekly calibration** (add to `ascent/research/self_improve.py` or a new `calibrate_cost_model.py`):

Read `logs/slippage_log.jsonl`, fit OLS on `(X/V, sigma, realized_slippage_bps)` to update `eta` and `gamma`. Write updated params to `data_cache/cost_model_params.json`. Run weekly alongside self_improve.

**Files touched**: `ascent/execution/order_engine.py`  
**New files**: `ascent/execution/cost_model.py`, `data_cache/cost_model_params.json` (runtime)

---

## Phase 3 — Signal Quality

### Fix 3a: CPCV for ML Sleeve (1 week)

**Problem**: Current 80/20 chronological split in `stack.py`:

1. Single split — high variance estimate, one bad OOS period tanks the whole model
2. No purging — `mom_21d` feature uses 21 days of history; last 21 training days overlap with test period's feature inputs (leakage)

**Design**: Combinatorial Purged Cross-Validation (CPCV) from de Prado, *Advances in Financial Machine Learning*, Chapter 12.

**Parameters**:

- `n_splits = 6` (N folds)
- `n_test_splits = 2` (k test folds per combination)
- Produces `C(6,2) = 15` train/test splits
- Each date appears in exactly `C(N-1, k-1) = C(5,1) = 5` test folds
- Prediction for each date = mean of those 5 models
- `purge_days = 5` (covers `mom_5d` leakage — any feature with 5-day lookback)
- `embargo_days = 5` (additional buffer after each test fold)

**New module**: `ascent/research/cpcv.py`

```python
class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation splitter.
    
    Produces C(n_splits, n_test_splits) train/test date index pairs.
    Purge gap removes training observations whose forward returns overlap
    with the test period. Embargo gap prevents data leakage from slow features.
    
    Parameters:
        n_splits (int): N — number of folds. Default 6.
        n_test_splits (int): k — test folds per combination. Default 2.
        purge_days (int): days to remove from train end before test start.
                          Set to max feature lookback (mom_5d = 5 days).
        embargo_days (int): days to remove from train start after test end.
                            Set to 5 days (matches purge).
    
    Note on date coverage:
        Each date appears in C(N-1, k-1) = C(5,1) = 5 test folds.
        Predictions for each date are averaged across those 5 models.
        Sharpe is computed across all C(N,k) = 15 backtest paths, where
        each path covers the full date range via non-overlapping test folds.
    """
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_days: int = 5,
        embargo_days: int = 5,
    ): ...

    def split(
        self, dates: pd.DatetimeIndex
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yields (train_dates, test_dates) for each of C(N,k) combinations."""
        ...

    def backtest_paths(self) -> list[list[int]]:
        """
        Returns C(N,k) paths, each covering the full date range.
        Used to compute path-dependent Sharpe distribution.
        """
        ...
```

**Training procedure in `ml_sleeve.py`** (replaces current 80/20 split):

```python
def build_ml_alpha_cpcv(
    features: dict,
    targets: pd.DataFrame,
    agent_id: str,
    splitter: CPCVSplitter = None,
) -> pd.DataFrame:
    """
    Build ML alpha using CPCV. Returns OOS predictions for all dates
    covered by at least one test fold.
    
    Disables ML sleeve (returns empty DataFrame) if:
    - 5th percentile CPCV Sharpe < 0 (model unreliable OOS)
    - Fewer than 10 of 15 splits converge (insufficient data)
    """
    if splitter is None:
        splitter = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)

    dates = sorted(targets.index)
    all_predictions: dict[pd.Timestamp, list[float]] = {}
    split_sharpes: list[float] = []
    n_converged = 0

    for fold_idx, (train_dates, test_dates) in enumerate(splitter.split(pd.DatetimeIndex(dates))):
        try:
            model = _train_xgboost(features, targets, train_dates)
            preds = _predict(model, features, test_dates)
            fold_ic = _compute_ic(preds, targets.loc[test_dates])
            split_sharpes.append(fold_ic)
            n_converged += 1

            for date, pred in zip(test_dates, preds):
                all_predictions.setdefault(date, []).append(pred)

        except Exception as exc:
            log.warning(f"[ML-CPCV] Fold {fold_idx} failed: {exc}")
            continue

    # Guard: too many failed splits
    if n_converged < 10:
        log.warning(
            f"[ML-CPCV] Only {n_converged}/15 splits converged — "
            f"ML sleeve disabled (biased estimate from partial CPCV)"
        )
        return pd.DataFrame()

    # Guard: p5 Sharpe < 0 = unreliable OOS
    sharpe_p5 = float(np.percentile(split_sharpes, 5))
    sharpe_p50 = float(np.percentile(split_sharpes, 50))
    log.info(f"[ML-CPCV] {n_converged} folds — Sharpe p5={sharpe_p5:.3f} p50={sharpe_p50:.3f}")

    if sharpe_p5 < 0:
        log.warning(
            f"[ML-CPCV] p5 Sharpe {sharpe_p5:.3f} < 0 — "
            f"ML sleeve disabled this cycle (unreliable OOS)"
        )
        return pd.DataFrame()

    # Average predictions across folds that covered each date
    final_predictions = {
        d: float(np.mean(preds))
        for d, preds in all_predictions.items()
    }

    # Train final production model on most recent 80% for live prediction
    n_final_train = int(len(dates) * 0.80)
    final_train_dates = pd.DatetimeIndex(dates[:n_final_train])
    final_model = _train_xgboost(features, targets, final_train_dates)

    _cache_model(final_model, agent_id, meta={
        "sharpe_p5": sharpe_p5,
        "sharpe_p50": sharpe_p50,
        "n_converged": n_converged,
        "cpcv_params": {"n_splits": 6, "n_test_splits": 2, "purge": 5, "embargo": 5},
    })

    return _predict_latest(final_model, features)
```

**Files touched**: `ascent/alpha/ml_sleeve.py`, `ascent/alpha/stack.py` (remove old 80/20 split logic)  
**New files**: `ascent/research/cpcv.py`

---

### Fix 3b: Regime Particle Filter + Emergency Refit (1 week)

**Problem**: Regime HMM refit every 21 days. During a structural break (COVID March 2020, SVB March 2023), market structure changes in 2–3 days. 21-day lag means model calls "calm_bull" while portfolio is down 15%. `ascent/regime/breaks.py` detects breaks but its output is never wired to trigger a refit.

**Design**: Three-track regime update system.

**Track 1 — Scheduled refit (every 5 days)**  
Reduce `refit_every_days` from 21 to 5 in `RegimeConfig`. Full HMM walk-forward K selection. Cached to `data_cache/regime_engine.pkl`. Runtime ~45 seconds every 5 days.

**Track 2 — Emergency refit on structural break (same-day)**  
Wire `breaks.py` output into `engine.py`. Emergency refit fires when ANY of these are true:

```python
def check_emergency_refit_triggers(
    spy: pd.Series,
    tlt: pd.Series,
    vix: pd.Series,
    breaks_detector,  # existing ascent/regime/breaks.py
) -> tuple[bool, str]:
    """
    Returns (should_refit: bool, reason: str).
    
    Triggers:
    1. SPY 1-day return < -3% AND VIX > 30
    2. SPY crosses 200-day MA from above (bear market entry)
    3. SPY/TLT 5-day correlation > +0.30
       (normal correlation is -0.3 to -0.5; positive = both selling = liquidity crisis)
    4. Structural break z-score from breaks.py > 3.5σ
    """
    spy_ret_1d = spy.pct_change().iloc[-1]
    vix_current = vix.iloc[-1]
    spy_ma200 = spy.rolling(200).mean().iloc[-1]
    spy_prev_ma200 = spy.rolling(200).mean().iloc[-2]

    # Trigger 1: crash + fear
    if spy_ret_1d < -0.03 and vix_current > 30:
        return True, f"SPY -3%+ ({spy_ret_1d:.1%}) + VIX {vix_current:.0f}"

    # Trigger 2: bear market entry
    if spy.iloc[-2] > spy_prev_ma200 and spy.iloc[-1] < spy_ma200:
        return True, f"SPY crossed below 200MA ({spy.iloc[-1]:.2f} < {spy_ma200:.2f})"

    # Trigger 3: correlation flip — DIRECTION MATTERS
    # Normal: SPY and TLT negatively correlated (flight to safety)
    # Crisis: both sell off simultaneously = positive correlation = liquidity event
    spy_5d = spy.pct_change().tail(5)
    tlt_5d = tlt.pct_change().tail(5)
    corr = float(spy_5d.corr(tlt_5d))
    if corr > 0.30:
        return True, f"SPY/TLT 5-day correlation={corr:.2f} (risk-off: both falling)"

    # Trigger 4: statistical structural break
    break_zscore = breaks_detector.latest_zscore()
    if break_zscore > 3.5:
        return True, f"Structural break z-score={break_zscore:.1f}σ"

    return False, ""
```

Emergency refits logged to `logs/regime_emergency_refit.jsonl`:

```json
{"date": "2026-04-15", "trigger": "SPY -3.2% + VIX 38", "timestamp": "..."}
```

**Track 3 — Sequential Monte Carlo (Particle Filter) for daily online updates**  
Between batch refits, update regime posterior probabilities daily via particle filter.

New module: `ascent/regime/particle_filter.py`

```python
class RegimeParticleFilter:
    """
    Online Bayesian regime probability update via Sequential Monte Carlo.
    
    Does NOT refit HMM parameters. Updates the posterior distribution over
    hidden states given new daily observations using importance sampling
    with resampling (SIR algorithm).
    
    Algorithm (per new observation x_t):
      1. Propagate: each particle's state transitions via HMM transition matrix
      2. Reweight: multiply particle weight by emission likelihood p(x_t | state)
      3. Normalize weights
      4. Resample (SIR): when effective N = 1/sum(w^2) < N/2, resample uniformly
      5. State posterior = weighted histogram over current particle states
    
    CRITICAL: Must be fully reinitialized when HMM is batch-refit.
    State indices may change after refit (K=3 states re-labeled).
    Reinitialize from stationary distribution of new model.
    
    Parameters:
        hmm_model: fitted HMM (from RegimeEngine batch refit)
        n_particles (int): number of particles (default 500)
    """
    def __init__(self, hmm_model, n_particles: int = 500):
        self.hmm = hmm_model
        self.n_particles = n_particles
        self.particles = self._init_particles_from_stationary()
        self.weights = np.ones(n_particles) / n_particles

    def update(self, observation: np.ndarray) -> RegimeSignal:
        """
        Process one new observation. Returns updated RegimeSignal.
        Call once per trading day after new prices are available.
        """
        self._propagate()
        self._reweight(observation)
        if self._effective_n() < self.n_particles / 2:
            self._resample()  # SIR resampling
        return self._to_signal()

    def reinitialize(self, new_hmm_model):
        """
        Called after every batch HMM refit.
        Fully resets particles — do not carry over old particles
        calibrated to previous model's state space.
        """
        self.hmm = new_hmm_model
        self.particles = self._init_particles_from_stationary()
        self.weights = np.ones(self.n_particles) / self.n_particles
        log.info("[PF] Particle filter reinitialized after HMM refit")

    def _effective_n(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def _init_particles_from_stationary(self) -> np.ndarray:
        """Sample initial states from HMM stationary distribution."""
        stationary = self.hmm.get_stationary_distribution()
        return np.random.choice(
            len(stationary), size=self.n_particles, p=stationary
        )
    # ... _propagate, _reweight, _resample, _to_signal implementations
```

**Integration in `engine.py`**:

- After batch refit: call `particle_filter.reinitialize(new_model)`
- After emergency refit: same
- Each daily run: call `particle_filter.update(today_observation)` → use this as the live signal
- Batch refit signal and particle filter signal should agree within tolerance; log divergence > 2 states

**Files touched**: `ascent/regime/engine.py`, `ascent/regime/breaks.py`, `ascent/config/settings.py` (`refit_every_days: 21 → 5`)  
**New files**: `ascent/regime/particle_filter.py`, `logs/regime_emergency_refit.jsonl`

---

## Phase 4 — Portfolio Architecture (Research-Grade)

### Fix 4: Options-Based Hedge Overlay (2 weeks)

**Problem**: Portfolio is 100% long. In stressed/crisis regimes, gross exposure is scaled (SPY 200MA overlay → 70%) but remaining long positions still lose in a bear market. VIXY in alternatives provides some protection but alternatives get 10–35% of capital. No systematic hedge.

**Design**: Regime-conditional hedge overlay. Additive to equity portfolio — never modifies `target_weights`.

**New module**: `ascent/portfolio/hedge_overlay.py`

**New config**: `ascent/config/settings.py`

```python
@dataclass
class HedgeConfig:
    enabled: bool = False                    # OFF by default until paper-tested
    use_etf_fallback: bool = True            # start with ETFs, not options
    max_theta_budget_pct: float = 0.0002     # 2 bps/day max theta decay
    max_annual_cost_bps: float = 100.0       # kill hedge if cumulative cost exceeds 100 bps/year
    max_gross_exposure: float = 1.10         # cap total notional at 110% of NAV
    rebalance_delta_threshold: float = 0.10  # rebalance when hedge delta drifts ±0.10
    roll_dte_threshold: int = 14             # roll to next expiry when DTE < 14

    hedge_ratios: dict = field(default_factory=lambda: {
        "calm_bull": 0.00,    # no hedge (upside cost too high)
        "euphoric":  0.15,    # 15% of portfolio beta hedged
        "stressed":  0.30,    # 30%
        "crisis":    0.60,    # 60%
        "uncertain": 0.20,    # 20%
    })

    # Options structure by regime
    option_specs: dict = field(default_factory=lambda: {
        "euphoric": {
            "instrument": "SPY_PUT_SPREAD",
            "long_strike_pct": 0.95,    # buy put at 95% of SPY
            "short_strike_pct": 0.85,   # sell put at 85% of SPY
            "target_dte": 30,
        },
        "stressed": {
            "instrument": "SPY_PUT_SPREAD",
            "long_strike_pct": 0.92,
            "short_strike_pct": 0.80,
            "target_dte": 30,
        },
        "crisis": {
            "instrument": "SPY_PUT_ATM",
            "long_strike_pct": 1.00,    # ATM put
            "short_strike_pct": None,   # no short leg
            "target_dte": 30,
        },
        "uncertain": {
            "instrument": "SPY_PUT_SPREAD",
            "long_strike_pct": 0.95,
            "short_strike_pct": 0.85,
            "target_dte": 30,
        },
    })
```

**Portfolio beta computation**:

```python
def compute_portfolio_beta(
    target_weights: dict[str, float],
    price_history: pd.DataFrame,
    spy_history: pd.Series,
    lookback: int = 63,
) -> float:
    """
    Dollar-weighted portfolio beta vs SPY.
    beta_portfolio = sum(w_i * beta_i)
    beta_i = cov(r_i, r_spy) / var(r_spy) over lookback days.
    """
    returns = price_history.pct_change().tail(lookback)
    spy_returns = spy_history.pct_change().tail(lookback)
    var_spy = spy_returns.var()
    if var_spy == 0:
        return 1.0

    beta = 0.0
    for sym, w in target_weights.items():
        if sym not in returns.columns:
            continue
        sym_returns = returns[sym].dropna()
        aligned = pd.concat([sym_returns, spy_returns], axis=1).dropna()
        if len(aligned) < 20:
            beta += w * 1.0  # assume beta=1 if insufficient data
            continue
        cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        beta_i = cov / var_spy
        beta += w * beta_i

    return float(np.clip(beta, 0.3, 2.0))  # clip to reasonable range
```

**HedgeSpec and HedgePosition dataclasses**:

```python
@dataclass
class HedgeSpec:
    regime: str
    instrument: str           # "SPY_PUT_SPREAD" | "SPY_PUT_ATM" | "ETF_INVERSE" | "NONE"
    long_strike_pct: float    # % of current SPY (e.g. 0.92)
    short_strike_pct: float | None  # None for ATM
    expiry_dte: int
    hedge_ratio: float        # fraction of portfolio beta to hedge
    notional: float           # portfolio_value * beta * hedge_ratio

@dataclass
class HedgePosition:
    spec: HedgeSpec
    contracts: int            # option contracts (each = 100 shares)
    entry_date: str
    expiry_date: str
    long_strike: float        # actual dollar strike
    short_strike: float | None
    entry_cost: float         # total premium paid
    current_delta: float
    current_theta: float
    current_vega: float

@dataclass
class HedgeGreeks:
    delta: float
    gamma: float
    theta: float              # daily decay in dollars
    vega: float
```

**HedgeOverlay class**:

```python
class HedgeOverlay:
    def __init__(self, config: HedgeConfig, alpaca_broker, regime_signal: RegimeSignal):
        self.config = config
        self.broker = alpaca_broker
        self.regime = regime_signal

    def compute_spec(
        self,
        target_weights: dict[str, float],
        portfolio_value: float,
        price_history: pd.DataFrame,
        spy_history: pd.Series,
    ) -> HedgeSpec:
        """Compute the required hedge for current regime."""
        regime_label = self.regime.label.value

        if self.config.hedge_ratios.get(regime_label, 0) == 0:
            return HedgeSpec(regime=regime_label, instrument="NONE", ...)

        beta = compute_portfolio_beta(target_weights, price_history, spy_history)
        hedge_ratio = self.config.hedge_ratios[regime_label]
        notional = portfolio_value * beta * hedge_ratio

        # Check gross exposure constraint
        if portfolio_value + notional > portfolio_value * self.config.max_gross_exposure:
            notional = portfolio_value * (self.config.max_gross_exposure - 1.0)
            log.warning(f"[Hedge] Notional capped at gross_exposure limit "
                        f"({self.config.max_gross_exposure:.0%})")

        spec_dict = self.config.option_specs.get(regime_label, {})
        instrument = "ETF_INVERSE" if self.config.use_etf_fallback else spec_dict.get("instrument", "NONE")

        return HedgeSpec(
            regime=regime_label,
            instrument=instrument,
            long_strike_pct=spec_dict.get("long_strike_pct", 1.0),
            short_strike_pct=spec_dict.get("short_strike_pct"),
            expiry_dte=spec_dict.get("target_dte", 30),
            hedge_ratio=hedge_ratio,
            notional=notional,
        )

    def rebalance_needed(
        self, current_position: HedgePosition | None, new_spec: HedgeSpec
    ) -> bool:
        """
        Rebalance when:
        - No current position
        - Regime changed (new spec required)
        - Delta drifted > threshold from target
        - DTE < roll threshold
        - Cumulative cost exceeds annual budget
        """
        if current_position is None:
            return new_spec.instrument != "NONE"
        if current_position.spec.regime != new_spec.regime:
            return True
        if abs(current_position.current_delta - self._target_delta(new_spec)) > \
                self.config.rebalance_delta_threshold:
            return True
        days_to_expiry = (pd.Timestamp(current_position.expiry_date) -
                          pd.Timestamp.now()).days
        if days_to_expiry < self.config.roll_dte_threshold:
            return True
        return False

    def execute_hedge(self, spec: HedgeSpec, current_spy_price: float) -> HedgePosition | None:
        """
        Execute hedge via Alpaca (options or ETF fallback).
        Logs to logs/hedge_log.jsonl.
        """
        if spec.instrument == "NONE":
            return None
        if spec.instrument == "ETF_INVERSE":
            return self._execute_etf_hedge(spec)
        return self._execute_options_hedge(spec, current_spy_price)

    def track_greeks(self) -> HedgeGreeks:
        """Query Alpaca for current option positions, compute aggregate Greeks."""
        ...

    def check_theta_budget(self, greeks: HedgeGreeks, portfolio_value: float) -> bool:
        """
        Returns True if theta within budget.
        Logs warning if theta decay > max_theta_budget_pct * portfolio_value.
        """
        daily_theta_budget = self.config.max_theta_budget_pct * portfolio_value
        if abs(greeks.theta) > daily_theta_budget:
            log.warning(
                f"[Hedge] Theta decay ${abs(greeks.theta):.0f}/day exceeds "
                f"budget ${daily_theta_budget:.0f}/day — consider reducing hedge size"
            )
            return False
        return True
```

**P&L attribution** — new fields appended to every `eod_log.jsonl` entry:

```json
{
  "equity_pnl_bps": 34.0,
  "hedge_pnl_bps": -8.2,
  "hedge_cost_bps": 8.2,
  "hedge_protection_bps": 0.0,
  "net_pnl_bps": 25.8,
  "cumulative_hedge_cost_bps_ytd": 42.0,
  "hedge_spec": {
    "regime": "stressed",
    "instrument": "SPY_PUT_SPREAD",
    "hedge_ratio": 0.30,
    "notional": 28500.0
  },
  "hedge_greeks": {
    "delta": -0.18,
    "gamma": 0.004,
    "theta": -18.50,
    "vega": 142.0
  }
}
```

**Kill switch integration**: cumulative hedge cost is tracked in `eod_log.jsonl`. If `cumulative_hedge_cost_bps_ytd > HedgeConfig.max_annual_cost_bps`, hedge is automatically disabled and a `logs/hedge_cost_alert.jsonl` entry is written. Requires manual re-enable.

**Mandatory staged rollout — do not skip stages**:

```
Stage 1 (30 days min): ETF fallback (SH = 1x inverse SPY), paper account
  Verify: hedge_log.jsonl written correctly
  Verify: P&L attribution correct in eod_log.jsonl
  Verify: kill switch drawdown calculation includes hedge P&L
  Verify: regime transitions trigger correct hedge changes (calm→stressed→crisis)
  Verify: cumulative cost tracking accurate
  Stress-test: manually force regime="crisis" in halt_override, verify 60% hedge fires

Stage 2 (30 days min): ETF fallback continues, validate cost model
  Verify: theta budget enforcement (options stage prep)
  Verify: annual cost cap triggers correctly
  Verify: rebalance triggers fire on regime change

Stage 3 (30 days min): Options (if Alpaca options paper trading confirmed working)
  Verify: option order submission and fill
  Verify: Greeks tracking accuracy (compare Alpaca-reported vs Black-Scholes)
  Verify: delta drift rebalancing triggers

Stage 4: Enable on live paper account
  Only after Stages 1–3 pass with no anomalies
  Monitor for 30 days before any consideration of real money
```

**Files touched**: `ascent/execution/eod_runner.py` (call hedge overlay post-weights), `ascent/config/settings.py` (add `HedgeConfig`)  
**New files**: `ascent/portfolio/hedge_overlay.py`, `logs/hedge_log.jsonl`, `logs/hedge_cost_alert.jsonl`

---

## Integrity Constraints — Unchanged

These constraints from CLAUDE.md apply to all 8 fixes and cannot be violated:

1. **No look-ahead bias** — CPCV: purge=5 days, embargo=5 days, enforced strictly. Particle filter uses only past observations.
2. **No simulated data under live cache names** — unchanged.
3. **Max-weight hard cap** — `_water_fill_cap()` unchanged. Hedge overlay does not touch equity weights.
4. **Sector constraint fallback** — Fix 1b replaces fallback with hard fail. No silent degradation.
5. **Walk-forward runner not a production entrypoint** — unchanged.
6. **Debate is advisory only** — Fix 1c adds persistence to halts but debate still never writes to alpha/portfolio/execution directly.
7. **Approval layer for large trades** — Fix 2a preserves this, adds restart resilience.

**New constraints added by this spec**:

1. Transaction cost model warns only — never blocks a valid order unilaterally (HIGH_IMPACT flag blocks, but this is a safety guard on participation rate, not a cost block)
2. Hedge overlay is strictly additive — never modifies `target_weights` from equity pipeline
3. Particle filter reinitialized fully on every HMM batch refit — no carryover of old particles
4. `halt_state.json` cleared only by valid human override — never auto-cleared by debate or agent outputs
5. `HedgeConfig.enabled = False` by default — never enable in live until all 4 staged rollout phases pass with zero anomalies
6. Skill scores treated as stale if `skill_score_as_of` > 1 business day old — orchestrator falls back to base allocation
7. Approval pending state persisted to disk — no duplicate orders on process restart

---

## Implementation Notes for Future Claude Sessions

When implementing any phase:

1. Read the relevant section of this spec in full before touching any code
2. Read the current state of the file you're about to modify before editing
3. Use `ast.parse()` verification after each patch
4. Never implement Phase 4 until Phases 1–3 are live and stable
5. For Phase 4: follow the staged rollout exactly — ETF fallback first, options second, live last
6. The CPCV module (`cpcv.py`) is a standalone research utility — unit test it independently before wiring into `ml_sleeve.py`
7. The particle filter must be tested against the batch HMM signal before replacing it — run both in parallel for 5 days and log divergence before switching to particle filter as the primary signal

---

## Files Summary


| Phase | New Files                                                                                  | Modified Files                                                     |
| ----- | ------------------------------------------------------------------------------
------------ | ------------------------------------------------------------------ |
| 1a    | —                                                                                          | `run_all_agents.py`, `skill_tracker.py`, `central_intelligence.py` |
| 1b    | `logs/sector_override.jsonl`                                                               | `optimizer.py`, `run_all_agents.py`                                |
| 1c    | `execution/halt_state.json`, `execution/halt_override.json`                                | `run_all_agents.py`, `debate_runner.py`                            |
| 2a    | `execution/approval_pending.json`, `logs/approval_heartbeat.jsonl`                         | `eod_runner.py`, `approval_server.py`                              |
| 2b    | `ascent/execution/cost_model.py`, `data_cache/cost_model_params.json`                      | `order_engine.py`                                                  |
| 3a    | `ascent/research/cpcv.py`                                                                  | `ml_sleeve.py`, `stack.py`                                         |
| 3b    | `ascent/regime/particle_filter.py`, `logs/regime_emergency_refit.jsonl`                    | `engine.py`, `breaks.py`, `settings.py`                            |
| 4     | `ascent/portfolio/hedge_overlay.py`, `logs/hedge_log.jsonl`, `logs/hedge_cost_alert.jsonl` | `eod_runner.py`, `settings.py`                                     |


