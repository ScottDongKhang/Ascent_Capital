"""
debate/adversarial_authority.py

Earned authority tracking for the Adversarial Intelligence layer.

Every weight change the judge makes is a falsifiable 10-day prediction.
After 30 scored interventions per type, authority levels are computed:
  - win_rate > 70%: up to 4% weight change allowed
  - win_rate > 50%: up to 2% weight change allowed
  - win_rate < 40%: suspended (intervention type makes things worse)

Intervention types:
  adversarial_thesis  — high adversarial score, position trimmed
  regime_sizing       — position oversized vs regime-optimal target
  coherence_risk      — position trimmed due to narrative concentration risk
  event_risk          — binary event (earnings <7d) triggered trim

State:      data_cache/adversarial_authority.json
Log:        logs/adversarial_interventions.jsonl
"""

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

AUTHORITY_PATH      = Path("data_cache/adversarial_authority.json")
INTERVENTION_LOG    = Path("logs/adversarial_interventions.jsonl")
OUTCOME_WINDOW_DAYS = 10   # trading-day target → use 14 calendar days as proxy
MIN_SAMPLE_SUSPEND  = 30   # need 30 scored to be eligible for suspension
SUSPEND_WIN_RATE    = 0.40  # suspend if win_rate falls below this after MIN_SAMPLE_SUSPEND
HIGH_WIN_RATE       = 0.70  # allow up to 4% change
MED_WIN_RATE        = 0.50  # allow up to 2% change

# Authority limits per intervention type
_MAX_CHANGE_PCTS = {
    "high":    0.04,   # win_rate > 70%
    "medium":  0.02,   # win_rate > 50%
    "low":     0.01,   # win_rate unknown (< 30 scored) or borderline
    "suspended": 0.0,
}

_ALL_TYPES = ["adversarial_thesis", "regime_sizing", "coherence_risk", "event_risk", "conviction_press"]


# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_authority() -> dict:
    if not AUTHORITY_PATH.exists():
        return {t: _empty_type_state() for t in _ALL_TYPES}
    try:
        data = json.loads(AUTHORITY_PATH.read_text())
        # Add any missing intervention types
        for t in _ALL_TYPES:
            if t not in data:
                data[t] = _empty_type_state()
        return data
    except Exception:
        return {t: _empty_type_state() for t in _ALL_TYPES}


def _save_authority(state: dict) -> None:
    AUTHORITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=AUTHORITY_PATH.parent, suffix=".tmp"))
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, AUTHORITY_PATH)


def _empty_type_state() -> dict:
    return {
        "n_interventions":  0,
        "n_scored":         0,
        "win_rate":         None,
        "allowed_change_pct": _MAX_CHANGE_PCTS["low"],
        "suspended":        False,
    }


# ── Log I/O ───────────────────────────────────────────────────────────────────

def _load_interventions() -> list:
    if not INTERVENTION_LOG.exists():
        return []
    rows = []
    for line in INTERVENTION_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _append_intervention(record: dict) -> None:
    INTERVENTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERVENTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def _rewrite_interventions(rows: list) -> None:
    INTERVENTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=INTERVENTION_LOG.parent, suffix=".tmp"))
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.replace(tmp, INTERVENTION_LOG)


# ── Public API ────────────────────────────────────────────────────────────────

def get_authority(intervention_type: str) -> dict:
    """
    Return current authority for an intervention type.
    Returns {"allowed_change_pct": float, "suspended": bool, "win_rate": Optional[float]}
    """
    state = _load_authority()
    ts = state.get(intervention_type, _empty_type_state())
    return {
        "allowed_change_pct": ts["allowed_change_pct"],
        "suspended":          ts["suspended"],
        "win_rate":           ts.get("win_rate"),
        "n_scored":           ts.get("n_scored", 0),
    }


def record_intervention(
    date_str:          str,
    symbol:            str,
    intervention_type: str,
    from_weight:       float,
    to_weight:         float,
    prediction:        str,
    regime:            str,
) -> None:
    """Log a new intervention. Called immediately after the judge applies a change."""
    record = {
        "date":              date_str,
        "symbol":            symbol,
        "intervention_type": intervention_type,
        "from_weight":       round(from_weight, 6),
        "to_weight":         round(to_weight, 6),
        "prediction":        prediction,
        "regime":            regime,
        "outcome_measured":  False,
        "outcome_date":      None,
        "outcome_correct":   None,
        "symbol_10d_return": None,
        "spy_10d_return":    None,
    }
    _append_intervention(record)

    # Increment n_interventions in authority state
    state = _load_authority()
    if intervention_type not in state:
        state[intervention_type] = _empty_type_state()
    state[intervention_type]["n_interventions"] += 1
    _save_authority(state)


def score_pending_interventions() -> int:
    """
    Scan intervention log. For any entries ≥14 calendar days old without an outcome,
    fetch prices and determine if the trim was correct.

    Correct = trimmed symbol underperformed SPY over the next 10 trading days
    (we reduced it, so if it lagged the market, our risk call was right).

    Returns count of newly scored interventions.
    """
    rows = _load_interventions()
    if not rows:
        return 0

    today     = date.today()
    cutoff    = today - timedelta(days=OUTCOME_WINDOW_DAYS + 4)  # calendar buffer
    pending   = [r for r in rows
                 if not r.get("outcome_measured")
                 and date.fromisoformat(r["date"]) <= cutoff]
    if not pending:
        return 0

    # Fetch price data for all symbols + SPY
    symbols_needed = list({r["symbol"] for r in pending} | {"SPY"})
    try:
        import yfinance as yf
        raw = yf.download(symbols_needed, period="60d", auto_adjust=True, progress=False)
        if raw.empty:
            return 0
        import pandas as pd
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if not isinstance(closes, pd.DataFrame):
            closes = closes.to_frame()
    except Exception as e:
        print(f"[AdvAuth] Price fetch failed: {e}")
        return 0

    scored = 0
    for row in pending:
        intervention_date = date.fromisoformat(row["date"])
        sym               = row["symbol"]

        try:
            # Find price on intervention date and 14 calendar days later
            def _price_on_or_after(target_date):
                for delta in range(5):
                    d = target_date + timedelta(days=delta)
                    mask = closes.index.date == d
                    if mask.any():
                        return closes.loc[mask].iloc[0]
                return None

            p0 = _price_on_or_after(intervention_date)
            p1 = _price_on_or_after(intervention_date + timedelta(days=14))
            if p0 is None or p1 is None:
                continue

            sym_ret = (float(p1.get(sym, float("nan"))) /
                       float(p0.get(sym, float("nan"))) - 1)
            spy_ret = (float(p1.get("SPY", float("nan"))) /
                       float(p0.get("SPY", float("nan"))) - 1)

            if any(v != v for v in [sym_ret, spy_ret]):  # NaN check
                continue

            # Correct if the trimmed position lagged SPY (our concern was right)
            outperformed = sym_ret > spy_ret + 0.01  # >1pp above SPY = wrong call
            underperformed = sym_ret < spy_ret - 0.01  # >1pp below SPY = right call

            if underperformed:
                outcome_correct = True
            elif outperformed:
                outcome_correct = False
            else:
                outcome_correct = None  # neutral / too close to call

            row["outcome_measured"]  = True
            row["outcome_date"]      = (intervention_date + timedelta(days=14)).isoformat()
            row["outcome_correct"]   = outcome_correct
            row["symbol_10d_return"] = round(sym_ret, 6)
            row["spy_10d_return"]    = round(spy_ret, 6)
            scored += 1

        except Exception as e:
            print(f"[AdvAuth] Outcome scoring failed for {sym} ({row['date']}): {e}")

    if scored:
        _rewrite_interventions(rows)
        _rebuild_authority(rows)

    return scored


def _rebuild_authority(rows: list) -> None:
    """Recompute authority levels from all scored interventions."""
    from collections import defaultdict
    buckets: dict = defaultdict(list)

    for row in rows:
        if not row.get("outcome_measured"):
            continue
        oc = row.get("outcome_correct")
        if oc is None:
            continue  # skip neutral outcomes
        itype = row.get("intervention_type", "unknown")
        buckets[itype].append(1 if oc else 0)

    state = _load_authority()
    for itype in _ALL_TYPES:
        scores = buckets.get(itype, [])
        n      = len(scores)
        wr     = sum(scores) / n if n > 0 else None

        if wr is None or n < MIN_SAMPLE_SUSPEND:
            tier = "low"
            suspended = False
        elif wr >= HIGH_WIN_RATE:
            tier = "high"
            suspended = False
        elif wr >= MED_WIN_RATE:
            tier = "medium"
            suspended = False
        elif wr < SUSPEND_WIN_RATE:
            tier = "suspended"
            suspended = True
        else:
            tier = "low"
            suspended = False

        state[itype]["n_scored"]          = n
        state[itype]["win_rate"]          = round(wr, 3) if wr is not None else None
        state[itype]["allowed_change_pct"] = _MAX_CHANGE_PCTS[tier]
        state[itype]["suspended"]         = suspended

    _save_authority(state)


def get_calibration_report() -> str:
    """Return human-readable calibration summary for weekend logs and AI PM context."""
    state = _load_authority()
    rows  = _load_interventions()
    total = len(rows)
    scored = sum(1 for r in rows if r.get("outcome_measured"))

    lines = [f"Adversarial Authority Calibration ({scored}/{total} scored):"]
    for itype in _ALL_TYPES:
        ts  = state.get(itype, _empty_type_state())
        wr  = ts.get("win_rate")
        n   = ts.get("n_scored", 0)
        max_pct = ts["allowed_change_pct"]
        susp = ts["suspended"]

        wr_str = f"{wr:.0%}" if wr is not None else "n/a"
        status = "SUSPENDED" if susp else f"max_change={max_pct:.0%}"
        lines.append(f"  {itype:<20} win_rate={wr_str} (n={n}) — {status}")

    return "\n".join(lines)


def format_authority_for_judge(regime: str) -> str:
    """Return authority state as text for injection into judge prompt."""
    state = _load_authority()
    lines = ["ADVERSARIAL AUTHORITY (earned based on past intervention accuracy):"]
    for itype in _ALL_TYPES:
        ts   = state.get(itype, _empty_type_state())
        wr   = ts.get("win_rate")
        n    = ts.get("n_scored", 0)
        pct  = ts["allowed_change_pct"]
        susp = ts["suspended"]

        if susp:
            lines.append(f"  {itype}: SUSPENDED (win_rate below 40% threshold) — do NOT use")
        elif wr is None or n < 10:
            lines.append(f"  {itype}: unproven (n={n}) — max change {pct:.0%}")
        else:
            lines.append(f"  {itype}: win_rate={wr:.0%} (n={n}) — max change {pct:.0%}")

    return "\n".join(lines)
