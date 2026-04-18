#!/bin/bash
REPO="/Users/kdong/Downloads/ascent capital v2 up to phase 5.1"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

IS_HOLIDAY=$(python3 - << 'PYEOF'
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

export ALPACA_API_KEY="PK24I444KXOU4UG5YFGJSMR7VI"
export ALPACA_SECRET_KEY="GRRBqNy3P7rMwxmzKntQXF5RTuQwd6xDhuKFiTjhaahG"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2"

echo "$(date): Starting multi-agent EOD run..." >> "$LOG_DIR/run_eod.log"
cd "$REPO" && python3 run_all_agents.py >> "$LOG_DIR/run_eod.log" 2>&1
echo "$(date): D run complete." >> "$LOG_DIR/run_eod.log"
