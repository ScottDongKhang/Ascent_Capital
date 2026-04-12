"""
Ascent Capital — Configuration
Loads from YAML with environment variable overrides for secrets.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data_cache"
RUNS_DIR = ROOT_DIR / "runs"


@dataclass
class APIKeys:
    polygon: str = ""
    tiingo: str = "b208387103b5cac95a5bb5bceed29132bca8fb03"
    fred: str = "f4ff5c2e94544a9d49c3e1a84fa64339"
    fmp: str = "bFsrpeZ0gtfZMsy0vZVGsJhmqzLtHivV"
    alpaca_key: str = "PKRFXINL4QJ4AWW5OSNGJ5ITZK"
    alpaca_secret: str = "ZhVS7Q6x7jDqibByUfQL2N8pt76go4HHzNVJJUhFPUc"
    openai: str = ""

    @classmethod
    def from_env(cls) -> "APIKeys":
        """Load keys from environment; fall back to dataclass defaults if env not set."""
        fields = cls.__dataclass_fields__

        def _get(name: str, env_key: str) -> str:
            val = (os.getenv(env_key) or "").strip()
            if val:
                return val
            default = fields[name].default
            return default if isinstance(default, str) else ""

        return cls(
            polygon=_get("polygon", "POLYGON_API_KEY"),
            tiingo=_get("tiingo", "TIINGO_API_KEY"),
            fred=_get("fred", "FRED_API_KEY"),
            fmp=_get("fmp", "FMP_API_KEY"),
            alpaca_key=_get("alpaca_key", "ALPACA_API_KEY"),
            alpaca_secret=_get("alpaca_secret", "ALPACA_SECRET_KEY"),
            openai=_get("openai", "OPENAI_API_KEY"),
        )


@dataclass
class UniverseConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD",
        "ABBV", "MRK", "KO", "PEP", "COST", "AVGO", "LLY",
        "WMT", "TMO", "CRM", "ACN", "MCD", "LIN", "ADBE", "NFLX",
    ])
    benchmark: str = "SPY"


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    start_date: str = "2020-01-01"
    end_date: str = ""
    top_n: int = 8
    max_weight: float = 0.15
    min_weight: float = 0.02
    rebalance_freq_days: int = 21  # monthly
    execution_delay_days: int = 1  # signal at close t, execute at open t+1
    spread_bps: float = 5.0       # half-spread cost per side in bps
    impact_bps: float = 5.0       # market impact estimate in bps
    commission_per_share: float = 0.0
    slippage_vol_mult: float = 0.1  # slippage = mult * vol * sqrt(participation)


@dataclass
class FeatureConfig:
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63, 126, 252])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 21, 63])
    volume_windows: List[int] = field(default_factory=lambda: [10, 21])
    target_horizons: List[int] = field(default_factory=lambda: [1, 5, 21])


@dataclass
class WalkForwardConfig:
    train_days: int = 252
    test_days: int = 63
    step_days: int = 21
    purge_days: int = 5
    min_train_days: int = 126


@dataclass
class Config:
    keys: APIKeys = field(default_factory=APIKeys.from_env)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    data_dir: Path = DATA_DIR
    runs_dir: Path = RUNS_DIR

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


# Singleton
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
