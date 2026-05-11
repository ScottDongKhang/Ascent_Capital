"""
ascent/data/store/timescale.py
TimescaleDB read/write interface.

All functions return False / None / empty DataFrame on failure — never raise.
The system operates in parquet-only mode when TimescaleDB is unavailable.

Set TIMESCALEDB_URL in .env:
    postgresql://postgres:ascent@localhost:5432/ascent
"""

import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

_connection = None  # lazy singleton


def get_connection():
    """
    Lazy singleton psycopg2 connection. Returns None if DB unavailable.
    Call timescale_available() at startup to decide operating mode.
    """
    global _connection
    if _connection is not None:
        try:
            _connection.cursor().execute("SELECT 1")
            return _connection
        except Exception:
            _connection = None

    url = os.environ.get("TIMESCALEDB_URL", "")
    if not url:
        return None

    try:
        import psycopg2
        _connection = psycopg2.connect(url)
        _connection.autocommit = True
        return _connection
    except Exception as exc:
        log.debug("[TimescaleDB] Connection failed: %s", exc)
        return None


def timescale_available() -> bool:
    """Liveness check. Returns True/False, never raises."""
    try:
        conn = get_connection()
        if conn is None:
            return False
        conn.cursor().execute("SELECT 1")
        return True
    except Exception:
        return False


def write_prices(df: pd.DataFrame, table: str = "prices_1d") -> bool:
    """
    Bulk-insert price rows. DataFrame must have columns:
        [time, symbol, open, high, low, close, volume]
    Returns True on success, False on failure.
    """
    conn = get_connection()
    if conn is None:
        return False

    required = {"time", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        log.warning("[TimescaleDB] write_prices: missing columns %s", required - set(df.columns))
        return False

    try:
        from psycopg2.extras import execute_values
        rows = [
            (
                row["time"],
                row["symbol"],
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
            )
            for _, row in df.iterrows()
        ]
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {table} (time, symbol, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
        return True
    except Exception as exc:
        log.error("[TimescaleDB] write_prices failed: %s", exc)
        return False


def read_prices(
    symbols: List[str],
    start: date,
    end: date,
    table: str = "prices_1d",
) -> pd.DataFrame:
    """
    Read OHLCV for symbols between start and end (inclusive).
    Returns DataFrame with DatetimeIndex and symbol columns (close prices).
    Returns empty DataFrame on failure.
    """
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        placeholders = ",".join(["%s"] * len(symbols))
        sql = f"""
            SELECT time, symbol, close
            FROM {table}
            WHERE symbol IN ({placeholders})
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        with conn.cursor() as cur:
            cur.execute(sql, symbols + [start, end])
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["time", "symbol", "close"])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        return df.pivot(index="time", columns="symbol", values="close")
    except Exception as exc:
        log.error("[TimescaleDB] read_prices failed: %s", exc)
        return pd.DataFrame()


def write_factor_scores(
    run_date: date,
    agent_id: str,
    scores_df: pd.DataFrame,
) -> bool:
    """
    Insert factor scores. scores_df must have columns [symbol, sleeve, score].
    """
    conn = get_connection()
    if conn is None:
        return False
    try:
        from psycopg2.extras import execute_values
        ts = datetime.combine(run_date, datetime.min.time())
        rows = [
            (ts, agent_id, row["symbol"], row["sleeve"], row["score"])
            for _, row in scores_df.iterrows()
        ]
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO factor_scores (time, agent_id, symbol, sleeve, score) VALUES %s ON CONFLICT DO NOTHING",
                rows,
            )
        return True
    except Exception as exc:
        log.error("[TimescaleDB] write_factor_scores failed: %s", exc)
        return False


def write_portfolio_state(
    run_date: date,
    agent_id: str,
    weights_dict: Dict[str, float],
    alpha_scores_df: Optional[pd.DataFrame] = None,
) -> bool:
    """Insert portfolio weights for an agent on a given date."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        from psycopg2.extras import execute_values
        ts = datetime.combine(run_date, datetime.min.time())

        alpha_map: Dict[str, float] = {}
        if alpha_scores_df is not None and not alpha_scores_df.empty:
            col = "composite" if "composite" in alpha_scores_df.columns else alpha_scores_df.columns[0]
            alpha_map = alpha_scores_df[col].to_dict()

        rows = [
            (ts, agent_id, sym, w, alpha_map.get(sym))
            for sym, w in weights_dict.items()
        ]
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO portfolio_state (time, agent_id, symbol, weight, alpha_score) VALUES %s ON CONFLICT DO NOTHING",
                rows,
            )
        return True
    except Exception as exc:
        log.error("[TimescaleDB] write_portfolio_state failed: %s", exc)
        return False


def write_nav(
    run_date: date,
    agent_id: str,
    nav: float,
    daily_return: float,
    spy_return: Optional[float] = None,
) -> bool:
    """Insert a NAV data point."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        ts = datetime.combine(run_date, datetime.min.time())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO nav_series (time, agent_id, nav, daily_return, spy_return)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (ts, agent_id, nav, daily_return, spy_return),
            )
        return True
    except Exception as exc:
        log.error("[TimescaleDB] write_nav failed: %s", exc)
        return False


def write_factor_exposures(
    run_date: date,
    exposures_series: "pd.Series",
) -> bool:
    """Insert factor exposure snapshot."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        from psycopg2.extras import execute_values
        ts = datetime.combine(run_date, datetime.min.time())
        rows = [(ts, factor, float(exp), None) for factor, exp in exposures_series.items()]
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO factor_exposures (time, factor, exposure, variance_explained) VALUES %s ON CONFLICT DO NOTHING",
                rows,
            )
        return True
    except Exception as exc:
        log.error("[TimescaleDB] write_factor_exposures failed: %s", exc)
        return False


def get_nav_series(
    agent_id: str = "us_equities",
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Read NAV series from TimescaleDB or fall back to pnl JSONL log.
    Returns DataFrame with columns [nav, daily_return, spy_return].
    """
    conn = get_connection()
    if conn is not None:
        try:
            sql = """
                SELECT time, nav, daily_return, spy_return
                FROM nav_series
                WHERE agent_id = %s
                  AND time >= NOW() - INTERVAL '%s days'
                ORDER BY time ASC
            """
            with conn.cursor() as cur:
                cur.execute(sql, (agent_id, lookback_days))
                rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["time", "nav", "daily_return", "spy_return"])
                df.set_index("time", inplace=True)
                return df
        except Exception as exc:
            log.warning("[TimescaleDB] get_nav_series DB read failed, falling back: %s", exc)

    # Fallback: read from JSONL log
    import json
    from pathlib import Path
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()
    log_path = Path("logs/us_equities_pnl.jsonl")
    if not log_path.exists():
        return pd.DataFrame()

    records = []
    with open(log_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("date", "") >= cutoff:
                    records.append({
                        "time":         pd.Timestamp(row["date"]),
                        "nav":          row.get("nav"),
                        "daily_return": row.get("daily_return"),
                        "spy_return":   row.get("spy_return"),
                    })
            except Exception:
                pass

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("time")
    return df
