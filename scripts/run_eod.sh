#!/bin/bash
REPO="/Users/scott/IdeaProjects/ascent-capital"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

IS_HOLIDAY=$("$REPO/.venv/bin/python" - << 'PYEOF'
try:
    import pandas_market_calendars as mcal
    from datetime import date
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date.today(), end_date=date.today())
    print("open" if not schedule.empty else "closed")
except ImportError:
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
