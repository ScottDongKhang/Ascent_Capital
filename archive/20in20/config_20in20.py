"""
Ascent Intel for 20in20 — configuration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PostureLabel = Literal["constructive", "selective", "neutral", "defensive", "crisis", "uncertain"]

@dataclass(frozen=True)
class Config20in20:
    asof: str                          # YYYY-MM-DD
    lookback_days: int = 252
    universe_name: str = "20in20_watchlist"
    outputs_dir: Path = Path("outputs/20in20")
    memo_horizon_days: int = 7
    risk_free_rate_annual: float = 0.03

    # Regime
    regime_symbol: str = "SPY"
    posture_min_conf: float = 0.50

    # Tables
    top_n_themes: int = 6
    top_n_relative_value: int = 12

    # Feature toggles
    enable_public_comps: bool = False   # phase 2
    enable_scenarios: bool = True
    enable_dashboard: bool = True

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "memos").mkdir(exist_ok=True)
        (self.outputs_dir / "tables").mkdir(exist_ok=True)


def load_config_20in20(
    asof: str,
    outputs_dir: str = "outputs/20in20",
    lookback_days: int = 252,
) -> Config20in20:
    """Validate asof format and return a frozen config."""
    import re
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", asof):
        raise ValueError(f"asof must be YYYY-MM-DD, got: {asof!r}")
    return Config20in20(
        asof=asof,
        outputs_dir=Path(outputs_dir),
        lookback_days=lookback_days,
    )
