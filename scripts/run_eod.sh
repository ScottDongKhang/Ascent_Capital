#!/bin/bash
REPO="/Users/scott/IdeaProjects/ascent-capital"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

# Trading-day check. Uses the stdlib holiday calendar in heartbeat_check.py,
# NOT pandas_market_calendars — that package is not installed in .venv and
# never was, so the old `except ImportError: print("open")` branch meant this
# gate silently passed on every market holiday. Fails open (prints "open") on
# any unexpected error, matching the previous behaviour: a spurious run is
# recoverable, a silently skipped one is what caused the outage.
# Gated on the US MARKET date, not dt.date.today(): this host is UTC+7 and the
# job fires at 09:00 local, which is the previous evening in PT. The local date
# would name tomorrow's session, so the gate would test the wrong day entirely —
# skipping real trading days and permitting runs on closed ones.
IS_HOLIDAY=$("$REPO/.venv/bin/python" - << 'PYEOF'
import sys
try:
    sys.path.insert(0, "/Users/scott/IdeaProjects/ascent-capital")
    sys.path.insert(0, "/Users/scott/IdeaProjects/ascent-capital/scripts")
    from ascent.utils.market_time import market_today
    from heartbeat_check import _holiday_set, is_trading_day
    today = market_today()
    holidays = _holiday_set(today.year - 1, today.year + 1)
    print("open" if is_trading_day(today, holidays) else "closed")
except Exception:
    print("open")
PYEOF
)

if [ "$IS_HOLIDAY" = "closed" ]; then
    echo "$(date): Market closed today — skipping." >> "$LOG_DIR/run_eod.log"
    exit 0
fi

# Secrets come from .env, never from this file (previously hardcoded here —
# rotate those paper-trading credentials; see remediation report).
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
fi

# Naming reconciliation: .env stores ALPACA_KEY / ALPACA_SECRET, but
# ascent/config/settings.py's APIKeys.from_env() reads
# ALPACA_API_KEY / ALPACA_SECRET_KEY, while other call sites
# (alpaca_stream.py, generate_performance_page.py) check both spellings.
# Export both here so either convention resolves, without editing either
# source of truth.
export ALPACA_API_KEY="${ALPACA_API_KEY:-$ALPACA_KEY}"
export ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY:-$ALPACA_SECRET}"
export ALPACA_KEY="${ALPACA_KEY:-$ALPACA_API_KEY}"
export ALPACA_SECRET="${ALPACA_SECRET:-$ALPACA_SECRET_KEY}"
export ALPACA_BASE_URL="${ALPACA_BASE_URL:-https://paper-api.alpaca.markets/v2}"

echo "$(date): Starting multi-agent EOD run..." >> "$LOG_DIR/run_eod.log"
cd "$REPO" && "$REPO/.venv/bin/python" run_all_agents.py >> "$LOG_DIR/run_eod.log" 2>&1
echo "$(date): Run complete." >> "$LOG_DIR/run_eod.log"

# Heartbeat: record that a run attempt happened, regardless of outcome, and
# surface WARN/CRITICAL immediately rather than waiting for the next
# 6-hourly heartbeat interval.
"$REPO/.venv/bin/python" "$REPO/scripts/heartbeat_check.py" --quiet
