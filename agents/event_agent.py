"""agents/event_agent.py

Background event agent — polls EDGAR, Capitol Trades, and options during market hours.
Runs as a daemon thread launched by run_all_agents.py.

Schedule:
  EDGAR:          every 5 minutes
  Capitol Trades: every 60 minutes
  Options:        every 15 minutes
  Loop exits at 16:10 ET or when stop_event is set.
"""
from __future__ import annotations
import logging
import threading
import time
from datetime import date, datetime, timezone

import pytz

log = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
MARKET_OPEN  = (9, 30)
MARKET_CLOSE = (16, 10)

EDGAR_INTERVAL     = 5 * 60    # 5 min
CAPTRADES_INTERVAL = 60 * 60   # 60 min
OPTIONS_INTERVAL   = 15 * 60   # 15 min

# Conviction thresholds per source
EDGAR_MIN_CONVICTION    = 0.6
CAPTRADES_MIN_CONVICTION = 0.4
OPTIONS_MIN_CONVICTION  = 0.5


def _now_et() -> datetime:
    return datetime.now(ET)


def _is_market_hours() -> bool:
    now = _now_et()
    h, m = now.hour, now.minute
    after_open  = (h, m) >= MARKET_OPEN
    before_close = (h, m) <= MARKET_CLOSE
    return after_open and before_close


def _load_universe_symbols() -> set:
    try:
        from ascent.config.settings import get_config
        cfg = get_config()
        return set(cfg.universe.symbols)
    except Exception:
        return set()


def _get_nav() -> float:
    """Approximate current NAV from Alpaca account."""
    try:
        import alpaca_trade_api as tradeapi
        from ascent.config.settings import APIKeys
        keys = APIKeys.from_env()
        api  = tradeapi.REST(keys.alpaca_key, keys.alpaca_secret, base_url="https://paper-api.alpaca.markets")
        account = api.get_account()
        return float(account.portfolio_value)
    except Exception:
        return 100_000.0  # fallback


def _already_traded_today(symbol: str) -> bool:
    """Return True if we already have an event trade for this symbol today."""
    from pathlib import Path
    import json
    log_path = Path("logs/event_trades.jsonl")
    if not log_path.exists():
        return False
    today_str = date.today().isoformat()
    try:
        with open(log_path) as f:
            for line in f:
                t = json.loads(line)
                if t.get("symbol") == symbol and t.get("timestamp", "")[:10] == today_str:
                    return True
    except Exception:
        pass
    return False


def _process_signal(signal: dict, source: str, nav: float) -> None:
    """Classify, size, submit, and log a single event signal."""
    from ascent.execution.event_runner import compute_event_size, submit_event_trade, log_event_trade, make_event_id
    direction = signal.get("direction", "neutral")
    if direction == "neutral":
        return
    symbol    = signal.get("symbol", "")
    if not symbol:
        return
    if _already_traded_today(symbol):
        log.debug("[EventAgent] %s already traded today — skipping", symbol)
        return

    event_id = make_event_id()
    size = compute_event_size(signal.get("conviction", 0.0), signal.get("urgency", "low"), nav)
    if size <= 0:
        log.debug("[EventAgent] %s: size below minimum — skipping", symbol)
        return

    fill = submit_event_trade(symbol, direction, size, signal.get("rationale", ""), event_id, nav=nav)
    log_event_trade(event_id, signal, size, fill, nav, source=source)
    log.info("[EventAgent] %s %s $%.0f [%s] — %s", direction, symbol, size, source, signal.get("rationale", ""))


def _run_edgar_cycle(universe_symbols: set, nav: float) -> None:
    """Poll EDGAR for new 8-K filings and classify them."""
    try:
        from ascent.data.ingest.edgar_listener import get_recent_filings, extract_8k_text, is_universe_company
        from ascent.alpha.event_alpha import classify_event
        filings = get_recent_filings(lookback_minutes=6)  # slight overlap for safety
        for filing in filings:
            if not is_universe_company(filing["company_name"], filing["cik"]):
                continue
            text   = extract_8k_text(filing["url"])
            signal = classify_event(text, filing["company_name"], filing["filing_type"])
            if signal.get("conviction", 0) < EDGAR_MIN_CONVICTION:
                continue
            # Need symbol — try CIK map
            from ascent.data.ingest.edgar_listener import _load_cik_map
            cik_map = _load_cik_map()
            symbol  = cik_map.get(filing["cik"], "")
            if not symbol:
                continue
            signal["symbol"] = symbol
            _process_signal(signal, "edgar", nav)
    except Exception as e:
        log.warning("[EventAgent] EDGAR cycle failed: %s", e)


def _run_captrades_cycle(universe_symbols: set, nav: float) -> None:
    """Poll Capitol Trades and classify disclosures."""
    try:
        from ascent.data.ingest.capitol_trades import fetch_recent_disclosures
        from ascent.alpha.event_alpha import classify_congressional_trade
        disclosures = fetch_recent_disclosures(universe_symbols=universe_symbols)
        for d in disclosures:
            signal = classify_congressional_trade(d)
            if signal.get("conviction", 0) < CAPTRADES_MIN_CONVICTION:
                continue
            _process_signal(signal, "capitol_trades", nav)
    except Exception as e:
        log.warning("[EventAgent] Capitol Trades cycle failed: %s", e)


def _run_options_cycle(universe_symbols: set, nav: float) -> None:
    """Scan options anomalies and classify."""
    try:
        from ascent.data.ingest.options_scanner import scan_options_anomalies
        from ascent.alpha.event_alpha import classify_options_anomaly
        anomalies = scan_options_anomalies(list(universe_symbols))
        for a in anomalies:
            signal = classify_options_anomaly(a["symbol"], a["iv_zscore"], a["pc_ratio_zscore"])
            if signal.get("conviction", 0) < OPTIONS_MIN_CONVICTION:
                continue
            _process_signal(signal, "options", nav)
    except Exception as e:
        log.warning("[EventAgent] Options cycle failed: %s", e)


def run_event_agent(stop_event: threading.Event | None = None) -> None:
    """
    Main polling loop. Runs until 16:10 ET or stop_event is set.
    Each source has its own interval counter.
    """
    log.info("[EventAgent] Starting — EDGAR/%dm, CapTrades/%dm, Options/%dm",
             EDGAR_INTERVAL // 60, CAPTRADES_INTERVAL // 60, OPTIONS_INTERVAL // 60)

    universe_symbols = _load_universe_symbols()
    last_edgar    = 0.0
    last_captrade = 0.0
    last_options  = 0.0

    while True:
        if stop_event and stop_event.is_set():
            log.info("[EventAgent] Stop event set — exiting")
            break
        if not _is_market_hours():
            log.info("[EventAgent] Outside market hours — stopping")
            break

        nav  = _get_nav()
        now  = time.monotonic()

        if now - last_edgar >= EDGAR_INTERVAL:
            _run_edgar_cycle(universe_symbols, nav)
            last_edgar = now

        if now - last_captrade >= CAPTRADES_INTERVAL:
            _run_captrades_cycle(universe_symbols, nav)
            last_captrade = now

        if now - last_options >= OPTIONS_INTERVAL:
            _run_options_cycle(universe_symbols, nav)
            last_options = now

        time.sleep(30)  # check every 30 seconds


def start_event_agent_thread(stop_event: threading.Event | None = None) -> threading.Thread:
    """Launch event agent as a non-blocking daemon thread."""
    t = threading.Thread(
        target=run_event_agent,
        args=(stop_event,),
        name="event-agent",
        daemon=True,
    )
    t.start()
    log.info("[EventAgent] Thread started (daemon=True)")
    return t
