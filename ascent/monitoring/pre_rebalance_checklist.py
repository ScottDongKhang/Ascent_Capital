"""
ascent/monitoring/pre_rebalance_checklist.py
Pre-rebalance sanity checklist.

Runs automatically on rebalance days before the debate and orchestrator fire.
Catches silent failures, stale data, and environmental issues before
they propagate into execution.

Checks:
  1. Market open / trading day confirmation
  2. Regime signal freshness (not stale)
  3. Agent weight sanity (no agent returned empty)
  4. Kill switch status (not tripped)
  5. API key presence (Alpaca + Anthropic)
  6. Earnings today for held positions (warning only)
  7. Price data freshness (not using stale cache)
  8. Portfolio NAV reasonableness (not zero, not crashed)

Returns a ChecklistResult with pass/warn/fail per check.
FAIL = block execution. WARN = proceed but log. PASS = clean.

Called by run_all_agents.py before debate fires.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

REGIME_JSON_PATH  = Path("dashboard/regime_signal.json")
EOD_LOG_PATH      = Path("logs/eod_log.jsonl")
KILL_SWITCH_PATH  = Path("logs/kill_switch_state.json")
MERGED_W_PATH     = Path("execution/merged_weights.json")
PRICES_CACHE_PATH = Path("data_cache/prices_live.parquet")
CHECKLIST_LOG     = Path("logs/checklist_log.jsonl")

REGIME_STALE_DAYS   = 3    # regime signal older than this = warn
PRICE_STALE_DAYS    = 2    # price cache older than this = warn
NAV_CRASH_THRESHOLD = 0.50 # NAV below 50% of starting = fail
STARTING_NAV_PATH   = Path("data_cache/starting_nav.json")


def _get_starting_nav() -> float:
    """
    Returns the baseline NAV for drawdown calculation.
    On first call, fetches live NAV from Alpaca and stores it.
    On subsequent calls, reads the stored value.
    This way STARTING_NAV always matches the actual account seed amount.
    """
    if STARTING_NAV_PATH.exists():
        try:
            data = json.loads(STARTING_NAV_PATH.read_text())
            return float(data["starting_nav"])
        except Exception:
            pass

    # First run — fetch from Alpaca and persist
    try:
        from ascent.execution.alpaca_broker import get_portfolio_value
        nav = get_portfolio_value()
        if nav > 0:
            STARTING_NAV_PATH.parent.mkdir(parents=True, exist_ok=True)
            STARTING_NAV_PATH.write_text(json.dumps({
                "starting_nav": nav,
                "recorded_at": datetime.now().isoformat(),
            }))
            print(f"[Checklist] Starting NAV recorded: ${nav:,.2f}")
            return nav
    except Exception:
        pass

    return 100_000.0  # safe fallback if Alpaca unreachable on first run


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class CheckItem:
    name:    str
    status:  str   # "PASS" | "WARN" | "FAIL"
    message: str
    blocking: bool = False  # FAIL items with blocking=True halt execution


@dataclass
class ChecklistResult:
    run_date:  str
    timestamp: str
    checks:    List[CheckItem] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status != "FAIL" or not c.blocking for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "WARN" for c in self.checks)

    @property
    def blocking_failures(self) -> List[CheckItem]:
        return [c for c in self.checks if c.status == "FAIL" and c.blocking]

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"[Checklist] Pre-rebalance checklist | {self.run_date}")
        print(f"{'='*60}")
        for c in self.checks:
            icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(c.status, "?")
            print(f"  {icon} {c.name}: {c.message}")
        print()
        if self.blocking_failures:
            print(f"[Checklist] ✗ BLOCKED — {len(self.blocking_failures)} blocking failure(s)")
        elif self.has_warnings:
            print(f"[Checklist] ⚠ PROCEED WITH CAUTION — {sum(1 for c in self.checks if c.status == 'WARN')} warning(s)")
        else:
            print(f"[Checklist] ✓ ALL CLEAR — proceeding to debate and execution")
        print()


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_market_open() -> CheckItem:
    """Check if today is a trading day and market is open or will open."""
    try:
        from ascent.execution.alpaca_broker import is_market_open
        if is_market_open():
            return CheckItem("Market", "PASS", "Market is open")
        # Market might not be open yet — check if it's a weekday
        today = date.today()
        if today.weekday() >= 5:
            return CheckItem("Market", "FAIL", "Today is a weekend — not a trading day", blocking=True)
        return CheckItem("Market", "WARN", "Market not yet open (pre-market or holiday)")
    except Exception as e:
        return CheckItem("Market", "WARN", f"Could not verify market status: {e}")


def _check_regime_freshness() -> CheckItem:
    """Check that the regime signal is recent."""
    if not REGIME_JSON_PATH.exists():
        return CheckItem("Regime signal", "WARN", "No regime signal file found — will use fallback")

    try:
        data = json.loads(REGIME_JSON_PATH.read_text())
        if isinstance(data, list) and data:
            latest = data[-1]
        elif isinstance(data, dict):
            latest = data
        else:
            return CheckItem("Regime signal", "WARN", "Regime signal file is empty")

        signal_date_str = latest.get("date") or latest.get("as_of") or latest.get("d")
        if not signal_date_str:
            return CheckItem("Regime signal", "WARN", "Regime signal has no date field")

        signal_date = date.fromisoformat(str(signal_date_str)[:10])
        age_days    = (date.today() - signal_date).days
        regime      = latest.get("regime") or latest.get("label") or "unknown"

        if age_days > REGIME_STALE_DAYS:
            return CheckItem(
                "Regime signal", "WARN",
                f"Signal is {age_days} days old (regime={regime}) — may be stale"
            )
        return CheckItem("Regime signal", "PASS", f"Fresh signal: {regime} ({age_days}d old)")

    except Exception as e:
        return CheckItem("Regime signal", "WARN", f"Could not parse regime signal: {e}")


def _check_agent_weights(agent_outputs: list) -> CheckItem:
    """Check that agents returned non-empty weights."""
    if not agent_outputs:
        return CheckItem("Agent weights", "FAIL", "No agent outputs received", blocking=True)

    empty  = [ao.agent_id for ao in agent_outputs if not ao.target_weights]
    active = [ao.agent_id for ao in agent_outputs if ao.target_weights]

    if not active:
        return CheckItem("Agent weights", "FAIL", "All agents returned empty weights", blocking=True)
    if empty:
        return CheckItem(
            "Agent weights", "WARN",
            f"{len(active)} agents OK, {len(empty)} empty: {empty}"
        )
    total_positions = sum(ao.n_positions for ao in agent_outputs)
    return CheckItem("Agent weights", "PASS", f"{len(active)} agents, {total_positions} total positions")


def _check_kill_switch() -> CheckItem:
    """Check that kill switch is not tripped."""
    if not KILL_SWITCH_PATH.exists():
        return CheckItem("Kill switch", "PASS", "No kill switch state file — clean")

    try:
        state = json.loads(KILL_SWITCH_PATH.read_text())
        if state.get("halted") or state.get("triggered"):
            reason = state.get("reason", "unknown reason")
            return CheckItem("Kill switch", "FAIL", f"KILL SWITCH IS ACTIVE: {reason}", blocking=True)
        return CheckItem("Kill switch", "PASS", "Kill switch clear")
    except Exception as e:
        return CheckItem("Kill switch", "WARN", f"Could not read kill switch state: {e}")


def _check_api_keys() -> CheckItem:
    """Check that required API keys are present."""
    missing = []
    present = []

    for key in ["ALPACA_KEY", "ALPACA_SECRET", "ANTHROPIC_API_KEY"]:
        if os.environ.get(key):
            present.append(key)
        else:
            missing.append(key)

    if "ALPACA_KEY" in missing or "ALPACA_SECRET" in missing:
        return CheckItem("API keys", "FAIL", f"Missing critical keys: {missing}", blocking=True)
    if missing:
        return CheckItem("API keys", "WARN", f"Missing optional keys: {missing}")
    return CheckItem("API keys", "PASS", f"All {len(present)} keys present")


def _check_earnings_today(held_symbols: List[str]) -> CheckItem:
    """
    Warn if any held position has earnings today.
    Uses yfinance calendar — advisory only, never blocks.
    """
    if not held_symbols:
        return CheckItem("Earnings today", "PASS", "No held positions to check")

    try:
        import yfinance as yf
        today_str = date.today().isoformat()
        flagged   = []

        for sym in held_symbols[:10]:  # cap at 10 to avoid slow startup
            try:
                ticker   = yf.Ticker(sym)
                calendar = ticker.calendar
                if calendar is None or calendar.empty:
                    continue
                # calendar index includes 'Earnings Date'
                if "Earnings Date" in calendar.index:
                    earn_dates = calendar.loc["Earnings Date"]
                    if hasattr(earn_dates, '__iter__'):
                        for d in earn_dates:
                            if str(d)[:10] == today_str:
                                flagged.append(sym)
                                break
                    elif str(earn_dates)[:10] == today_str:
                        flagged.append(sym)
            except Exception:
                pass

        if flagged:
            return CheckItem(
                "Earnings today", "WARN",
                f"Earnings today for held positions: {flagged} — review before rebalancing"
            )
        return CheckItem("Earnings today", "PASS", f"No earnings today for {len(held_symbols)} held positions")

    except Exception as e:
        return CheckItem("Earnings today", "WARN", f"Could not check earnings calendar: {e}")


def _check_price_data_freshness() -> CheckItem:
    """Check that price cache is not stale."""
    if not PRICES_CACHE_PATH.exists():
        return CheckItem("Price data", "WARN", "No live price cache found — will fetch fresh")

    try:
        mod_time  = datetime.fromtimestamp(PRICES_CACHE_PATH.stat().st_mtime)
        age_days  = (datetime.now() - mod_time).days

        if age_days > PRICE_STALE_DAYS:
            return CheckItem(
                "Price data", "WARN",
                f"Price cache is {age_days} days old — may fetch fresh data"
            )
        return CheckItem("Price data", "PASS", f"Price cache is {age_days}d old")
    except Exception as e:
        return CheckItem("Price data", "WARN", f"Could not check price cache: {e}")


def _check_portfolio_nav() -> CheckItem:
    """Check that portfolio NAV is reasonable — not zero or crashed."""
    try:
        from ascent.execution.alpaca_broker import get_portfolio_value
        nav = get_portfolio_value()

        if nav <= 0:
            return CheckItem("Portfolio NAV", "FAIL", f"NAV is ${nav:,.0f} — something is wrong", blocking=True)

        starting_nav = _get_starting_nav()
        drawdown = (starting_nav - nav) / starting_nav
        if drawdown > NAV_CRASH_THRESHOLD:
            return CheckItem(
                "Portfolio NAV", "FAIL",
                f"NAV ${nav:,.0f} is {drawdown:.0%} below starting — check kill switch",
                blocking=True
            )
        if drawdown > 0.15:
            return CheckItem(
                "Portfolio NAV", "WARN",
                f"NAV ${nav:,.0f} ({drawdown:.0%} drawdown from start)"
            )
        return CheckItem("Portfolio NAV", "PASS", f"NAV ${nav:,.0f}")

    except Exception as e:
        return CheckItem("Portfolio NAV", "WARN", f"Could not fetch NAV: {e}")


def _check_anthropic_balance() -> CheckItem:
    """
    Soft check — warn if API calls have been failing recently.
    We don't have a balance endpoint, so we check recent LLM call failures in logs.
    """
    try:
        if not EOD_LOG_PATH.exists():
            return CheckItem("API health", "PASS", "No EOD logs yet — assuming healthy")

        lines   = EOD_LOG_PATH.read_text().splitlines()
        recent  = lines[-5:] if len(lines) >= 5 else lines
        errors  = sum(1 for l in recent if "api" in l.lower() and "fail" in l.lower())

        if errors >= 3:
            return CheckItem("API health", "WARN", f"{errors} recent API failures in EOD log")
        return CheckItem("API health", "PASS", "No recent API failures")
    except Exception:
        return CheckItem("API health", "PASS", "Could not check — assuming healthy")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_checklist(
    agent_outputs: list = None,
    held_symbols: List[str] = None,
) -> ChecklistResult:
    """
    Run all pre-rebalance checks.

    Args:
        agent_outputs: List of AgentOutput from today's agent runs
        held_symbols:  List of currently held symbols for earnings check

    Returns:
        ChecklistResult — call .passed to know if execution should proceed
    """
    today  = date.today().isoformat()
    result = ChecklistResult(run_date=today, timestamp=datetime.now().isoformat())

    result.checks.append(_check_market_open())
    result.checks.append(_check_kill_switch())
    result.checks.append(_check_api_keys())
    result.checks.append(_check_portfolio_nav())
    result.checks.append(_check_regime_freshness())
    result.checks.append(_check_price_data_freshness())

    if agent_outputs is not None:
        result.checks.append(_check_agent_weights(agent_outputs))

    if held_symbols:
        result.checks.append(_check_earnings_today(held_symbols))

    result.print_summary()

    # Write to log
    log_entry = {
        "date":      today,
        "timestamp": result.timestamp,
        "passed":    result.passed,
        "checks":    [{"name": c.name, "status": c.status, "message": c.message}
                      for c in result.checks],
    }
    CHECKLIST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKLIST_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return result


if __name__ == "__main__":
    result = run_checklist()
    if not result.passed:
        exit(1)
