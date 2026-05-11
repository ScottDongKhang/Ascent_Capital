"""
ascent/monitoring/live_nav.py
Real-time NAV computation using streamed prices.

Holds last-rebalance positions and computes intraday NAV from live prices.
Falls back to last daily close when stream prices are unavailable.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_5MIN_SECONDS = 300
_nav_write_lock = threading.Lock()

# Tracks the open NAV at market open for intraday drawdown computation
_open_nav: Optional[float] = None


class LiveNAV:
    """
    Computes real-time NAV from streamed prices and last-known positions.

    positions_dict: {symbol: shares}  — from last Alpaca fill records
    cost_basis:     {symbol: avg_fill_price} — for P&L calculation
    """

    def __init__(
        self,
        positions_dict: Dict[str, float],
        cost_basis: Optional[Dict[str, float]] = None,
        cash: float = 0.0,
    ):
        self.positions = positions_dict
        self.cost_basis = cost_basis or {}
        self.cash = cash
        self._last_nav_write = datetime.min

    def _get_price(self, symbol: str) -> Optional[float]:
        """Live price from stream; fall back to latest daily close."""
        try:
            from ascent.data.streaming.alpaca_stream import get_live_price
            price = get_live_price(symbol)
            if price:
                return price
        except Exception:
            pass

        try:
            from ascent.data.store.timescale import read_prices
            from datetime import date
            df = read_prices([symbol], date.today() - timedelta(days=5), date.today())
            if not df.empty and symbol in df.columns:
                return float(df[symbol].dropna().iloc[-1])
        except Exception:
            pass

        return None

    def compute_current_nav(self) -> dict:
        """
        Returns:
            {
              "nav":          float,
              "daily_pnl":    float,
              "daily_return": float,
              "positions":    {symbol: {"price", "value", "pnl"}}
            }
        """
        global _open_nav

        position_detail: Dict[str, dict] = {}
        total_value = self.cash

        for symbol, shares in self.positions.items():
            price = self._get_price(symbol)
            if price is None:
                continue
            value = price * shares
            basis = self.cost_basis.get(symbol, price)
            pnl   = (price - basis) * shares
            position_detail[symbol] = {
                "price": round(price, 4),
                "value": round(value, 2),
                "pnl":   round(pnl, 2),
            }
            total_value += value

        if _open_nav is None:
            _open_nav = total_value

        daily_pnl    = total_value - (_open_nav or total_value)
        daily_return = daily_pnl / _open_nav if _open_nav else 0.0

        return {
            "nav":          round(total_value, 2),
            "daily_pnl":    round(daily_pnl, 2),
            "daily_return": round(daily_return, 6),
            "positions":    position_detail,
        }

    def compute_intraday_drawdown(self) -> float:
        """Current drawdown from today's open NAV. Negative means loss."""
        nav_data = self.compute_current_nav()
        current  = nav_data["nav"]
        if not _open_nav or _open_nav == 0:
            return 0.0
        return (current - _open_nav) / _open_nav

    def maybe_write_to_timescale(self, agent_id: str = "us_equities") -> None:
        """Write NAV to TimescaleDB at most once per 5 minutes."""
        now = datetime.now()
        if (now - self._last_nav_write).total_seconds() < _5MIN_SECONDS:
            return

        try:
            from ascent.data.store.timescale import write_nav
            nav_data = self.compute_current_nav()
            write_nav(
                run_date=now.date(),
                agent_id=agent_id,
                nav=nav_data["nav"],
                daily_return=nav_data["daily_return"],
                spy_return=None,
            )
            self._last_nav_write = now
        except Exception as exc:
            log.debug("[LiveNAV] TimescaleDB write failed: %s", exc)


def get_nav_series(lookback_days: int = 252, agent_id: str = "us_equities"):
    """Read NAV series from TimescaleDB or fall back to pnl JSONL log."""
    from ascent.data.store.timescale import get_nav_series as _ts_get
    return _ts_get(agent_id=agent_id, lookback_days=lookback_days)


def load_positions_from_alpaca() -> Dict[str, float]:
    """
    Load current positions from Alpaca as {symbol: shares}.
    Returns empty dict on failure.
    """
    try:
        from ascent.execution.alpaca_broker import ALPACA_BASE_URL, _headers
        import requests
        r = requests.get(
            f"{ALPACA_BASE_URL}/positions",
            headers=_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return {}
        return {p["symbol"]: float(p["qty"]) for p in r.json()}
    except Exception as exc:
        log.warning("[LiveNAV] Could not load positions from Alpaca: %s", exc)
        return {}
