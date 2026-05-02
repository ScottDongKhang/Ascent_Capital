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


def _load_us_equity_symbols() -> List[str]:
    """Load US equity universe from JSON. Falls back to a minimal seed list if file missing."""
    import json
    universe_path = Path(__file__).parent / "us_equity_universe.json"
    if universe_path.exists():
        try:
            with open(universe_path) as f:
                data = json.load(f)
            syms = data.get("symbols", [])
            if syms:
                return syms
        except Exception:
            pass
    # Minimal seed — should not be hit in normal operation
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA",
        "CAT", "HD", "CVX", "LIN", "COST",
    ]


@dataclass
class UniverseConfig:
    symbols: List[str] = field(default_factory=_load_us_equity_symbols)
    benchmark: str = "SPY"

    # Macro agent universe — rate-sensitive ETFs and commodities
    macro_symbols: List[str] = field(default_factory=lambda: [
        "TLT",   # 20+ year treasury bonds
        "IEF",   # 7-10 year treasury bonds
        "UUP",   # US dollar index
        "GLD",   # gold
        "PDBC",  # broad commodities
        "HYG",   # high-yield corporate bonds (credit risk proxy)
        "LQD",   # investment-grade corporate bonds
        "TIP",   # TIPS (inflation-linked)
        "SGOV",  # ultra-short treasury / cash proxy
        "BIL",   # T-bills (0-3 month)
        "DBB",   # base metals (copper/aluminum/zinc — different from broad PDBC)
        "KMLM",  # managed futures trend (diversifying macro signal)
    ])
    macro_benchmark: str = "SPY"

    # International/EM agent universe
    international_symbols: List[str] = field(default_factory=lambda: [
        # Broad EM
        "EEM", "VWO", "AAXJ",
        # Asia
        "EWT",   # Taiwan
        "EWY",   # South Korea
        "EWJ",   # Japan
        "INDA",  # India
        # Latin America
        "EWZ",   # Brazil
        "EWC",   # Canada (developed, low correlation to US)
        # Europe
        "EWG",   # Germany
        "EWU",   # UK
        # Developed markets broad
        "EFA",   # MSCI EAFE (developed ex-US)
    ])

    # Alternatives agent universe
    alternatives_symbols: List[str] = field(default_factory=lambda: [
        "VNQ",   # US REITs
        "GLD",   # gold
        "IAU",   # gold (alternative, lower expense ratio)
        "PDBC",  # broad commodities
        "DBA",   # agriculture
        "IFRA",  # US infrastructure
        "PAVE",  # US infrastructure (domestic focus)
        "WOOD",  # timber / forestry
        "BIL",   # T-bills (cash-like, crisis allocation)
        "VIXY",  # long volatility (crisis hedge)
        # Note: SVXY (short vol) removed — dangerous in paper trading context
    ])


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    start_date: str = "2020-01-02"
    end_date: str = ""
    top_n: int = 15
    max_weight: float = 0.10
    min_weight: float = 0.02
    rebalance_freq_days: int = 10
    execution_delay_days: int = 1
    spread_bps: float = 5.0
    impact_bps: float = 5.0
    commission_per_share: float = 0.0
    slippage_vol_mult: float = 0.1


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


# ── NEW: Regime config ────────────────────────────────────────────────────
@dataclass
class RegimeConfig:
    # Feature layer
    spy_ticker: str = "SPY"
    vix_ticker: str = "^VIX"
    feature_lookbacks: List[int] = field(default_factory=lambda: [5, 21, 63])

    # Model selection
    n_candidates: List[int] = field(default_factory=lambda: [2, 3, 4])
    training_min_days: int = 252
    refit_every_days: int = 21

    # Walk-forward validation
    wf_train_days: int = 252
    wf_test_days: int = 63
    wf_step_days: int = 63

    # Hysteresis
    enter_threshold: float = 0.55
    exit_threshold: float = 0.35
    min_dwell_days: int = 3
    entropy_uncertain_threshold: float = 0.90

    # Gross exposure multipliers per regime
    risk_multiplier: dict = field(default_factory=lambda: {
        "calm_bull": 1.00,
        "euphoric":  0.85,
        "stressed":  0.65,
        "crisis":    0.40,
        "uncertain": 0.75,
    })

    # Sleeve weight adjustments per regime
    sleeve_adjustments: dict = field(default_factory=lambda: {
        "calm_bull": {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
        "euphoric":  {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
        "stressed":  {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
        "crisis":    {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
        "uncertain": {"trend": 0.00, "statarb": 0.00, "meanrev": 0.00, "volatility": 0.00},
    })

    # Per-name max weight by regime
    max_weight_overrides: dict = field(default_factory=lambda: {
        "calm_bull": 0.15,
        "euphoric":  0.12,
        "stressed":  0.10,
        "crisis":    0.08,
        "uncertain": 0.12,
    })

    # Structural break detection
    break_min_size: int = 63
    break_penalty_multiplier: float = 2.0

    def to_engine_dict(self) -> dict:
        """Convert to the flat dict that RegimeEngine expects."""
        return {
            "regime_spy_ticker": self.spy_ticker,
            "regime_vix_ticker": self.vix_ticker,
            "regime_feature_lookbacks": self.feature_lookbacks,
            "regime_n_candidates": self.n_candidates,
            "regime_training_min_days": self.training_min_days,
            "regime_refit_every_days": self.refit_every_days,
            "regime_wf_train_days": self.wf_train_days,
            "regime_wf_test_days": self.wf_test_days,
            "regime_wf_step_days": self.wf_step_days,
            "regime_enter_threshold": self.enter_threshold,
            "regime_exit_threshold": self.exit_threshold,
            "regime_min_dwell_days": self.min_dwell_days,
            "regime_entropy_uncertain_threshold": self.entropy_uncertain_threshold,
            "regime_risk_multiplier": self.risk_multiplier,
            "regime_sleeve_adjustments": self.sleeve_adjustments,
            "regime_max_weight_overrides": self.max_weight_overrides,
            "regime_break_min_size": self.break_min_size,
            "regime_break_penalty_multiplier": self.break_penalty_multiplier,
        }


@dataclass
class Config:
    keys: APIKeys = field(default_factory=APIKeys.from_env)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)  # ← NEW
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