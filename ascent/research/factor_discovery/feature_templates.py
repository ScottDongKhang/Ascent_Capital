"""
ascent/research/factor_discovery/feature_templates.py

Pre-defined factor template families. Each template accepts a parameter dict
from the LLM and a price DataFrame, and returns a cross-sectionally z-scored
pd.Series.

Template families:
    MomentumTemplate    — skip-adjusted momentum
    ReversionTemplate   — short-term mean reversion
    VolatilityTemplate  — volatility regime (low-vol or vol-trend)
    QualityTemplate     — growth/stability metrics from price series
    CorrelationTemplate — market beta / idiosyncratic component

The LLM (llm_suggester.py) fills parameters into these templates via JSON.
No code is generated; the template logic is trusted and human-reviewed.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NORM_METHODS = frozenset({"zscore", "rank", "minmax"})


def _normalize(s: pd.Series, method: str) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    if method == "zscore":
        std = s.std()
        if std < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    if method == "rank":
        return s.rank(pct=True) - 0.5
    if method == "minmax":
        rng = s.max() - s.min()
        if rng < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.min()) / rng - 0.5
    return s


class MomentumTemplate:
    """
    Skip-adjusted momentum: return(lookback) - return(skip_days).
    skip_days > 0 avoids 1-month reversal contamination.
    """

    PARAM_SCHEMA = {
        "lookback":      {"type": int, "min": 21,  "max": 252, "default": 120},
        "skip_days":     {"type": int, "min": 0,   "max": 63,  "default": 21},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, lookback: int = 120, skip_days: int = 21,
                 normalization: str = "zscore"):
        self.lookback      = lookback
        self.skip_days     = skip_days
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback + 1:
            return pd.Series(0.0, index=df.columns)
        ret_long = df.pct_change(self.lookback).iloc[-1]
        if self.skip_days > 0 and len(df) > self.skip_days:
            ret_short = df.pct_change(self.skip_days).iloc[-1]
            signal    = ret_long - ret_short
        else:
            signal = ret_long
        return _normalize(signal, self.normalization)

    def to_dict(self) -> dict:
        return {
            "template": "MomentumTemplate",
            "lookback": self.lookback, "skip_days": self.skip_days,
            "normalization": self.normalization,
        }


class ReversionTemplate:
    """Short-term mean reversion: -return(lookback), optionally smoothed."""

    PARAM_SCHEMA = {
        "lookback":      {"type": int, "min": 2, "max": 21,  "default": 5},
        "smooth_window": {"type": int, "min": 1, "max": 10,  "default": 1},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, lookback: int = 5, smooth_window: int = 1,
                 normalization: str = "zscore"):
        self.lookback      = lookback
        self.smooth_window = smooth_window
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback + self.smooth_window:
            return pd.Series(0.0, index=df.columns)
        rets = df.pct_change(self.lookback)
        if self.smooth_window > 1:
            rets = rets.rolling(self.smooth_window).mean()
        signal = -rets.iloc[-1]
        return _normalize(signal, self.normalization)

    def to_dict(self) -> dict:
        return {"template": "ReversionTemplate", "lookback": self.lookback,
                "smooth_window": self.smooth_window, "normalization": self.normalization}


class VolatilityTemplate:
    """
    Volatility regime signal.
    direction="low"   → long low-vol names
    direction="trend" → long names with declining vol
    """

    PARAM_SCHEMA = {
        "vol_window":    {"type": int, "min": 10,  "max": 63,  "default": 21},
        "vov_window":    {"type": int, "min": 21,  "max": 126, "default": 63},
        "direction":     {"type": str, "choices": {"low", "trend"},  "default": "low"},
        "normalization": {"type": str, "choices": _NORM_METHODS,     "default": "zscore"},
    }

    def __init__(self, vol_window: int = 21, vov_window: int = 63,
                 direction: str = "low", normalization: str = "zscore"):
        self.vol_window    = vol_window
        self.vov_window    = vov_window
        self.direction     = direction
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.vov_window:
            return pd.Series(0.0, index=df.columns)
        rets = df.pct_change()
        vol  = rets.rolling(self.vol_window).std().iloc[-1]
        if self.direction == "low":
            signal = -vol
        else:
            vov       = rets.rolling(self.vol_window).std().rolling(self.vov_window).std()
            trend_col = rets.rolling(self.vol_window).std().diff(5)
            vov_last   = vov.iloc[-1]
            trend_last = trend_col.iloc[-1]
            signal     = -(trend_last / (vov_last + 1e-8))
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "VolatilityTemplate", "vol_window": self.vol_window,
                "vov_window": self.vov_window, "direction": self.direction,
                "normalization": self.normalization}


class QualityTemplate:
    """
    Price-implied quality: consistency, drawdown, or trend strength.
    """

    PARAM_SCHEMA = {
        "metric":        {"type": str, "choices": {"consistency", "drawdown", "trend_strength"},
                          "default": "consistency"},
        "window":        {"type": int, "min": 21, "max": 252, "default": 63},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, metric: str = "consistency", window: int = 63,
                 normalization: str = "zscore"):
        self.metric        = metric
        self.window        = window
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.window:
            return pd.Series(0.0, index=df.columns)
        rets = df.pct_change()
        w    = rets.tail(self.window)
        if self.metric == "consistency":
            mean_ret = w.mean()
            std_ret  = w.std().replace(0, np.nan)
            signal   = mean_ret / std_ret
        elif self.metric == "drawdown":
            prices_w    = df.tail(self.window)
            running_max = prices_w.cummax()
            dd = ((prices_w - running_max) / running_max.replace(0, np.nan)).min()
            signal = -dd
        else:
            signal = (w > 0).mean()
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "QualityTemplate", "metric": self.metric,
                "window": self.window, "normalization": self.normalization}


class CorrelationTemplate:
    """
    Market correlation / idiosyncratic component.
    mode="beta"          → cross-sectional beta-rank (long low-beta)
    mode="idiosyncratic" → residual return after removing market component
    """

    PARAM_SCHEMA = {
        "window":        {"type": int, "min": 21, "max": 126, "default": 63},
        "mode":          {"type": str, "choices": {"beta", "idiosyncratic"}, "default": "beta"},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, window: int = 63, mode: str = "beta",
                 normalization: str = "zscore"):
        self.window        = window
        self.mode          = mode
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.window + 1:
            return pd.Series(0.0, index=df.columns)
        rets   = df.pct_change().tail(self.window)
        mkt    = rets.mean(axis=1)
        betas  = {}
        resids = {}
        for col in rets.columns:
            s = rets[col].dropna()
            m = mkt.reindex(s.index)
            cov   = np.cov(s.values, m.values)
            var_m = cov[1, 1]
            beta  = cov[0, 1] / var_m if var_m > 1e-8 else 1.0
            betas[col]  = beta
            resids[col] = (s - beta * m).mean()
        if self.mode == "beta":
            signal = -pd.Series(betas)
        else:
            signal = pd.Series(resids)
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "CorrelationTemplate", "window": self.window,
                "mode": self.mode, "normalization": self.normalization}


# ── Registry ───────────────────────────────────────────────────────────────────

TEMPLATE_REGISTRY: Dict[str, type] = {
    "MomentumTemplate":    MomentumTemplate,
    "ReversionTemplate":   ReversionTemplate,
    "VolatilityTemplate":  VolatilityTemplate,
    "QualityTemplate":     QualityTemplate,
    "CorrelationTemplate": CorrelationTemplate,
}


def instantiate_template(template_name: str, params: dict):
    cls = TEMPLATE_REGISTRY.get(template_name)
    if cls is None:
        raise ValueError(f"Unknown template: {template_name}. Valid: {list(TEMPLATE_REGISTRY)}")
    schema = cls.PARAM_SCHEMA
    validated = {}
    for key, spec in schema.items():
        val = params.get(key, spec["default"])
        if spec["type"] is int:
            val = int(val)
            val = max(spec["min"], min(spec["max"], val))
        elif spec["type"] is str and "choices" in spec:
            if val not in spec["choices"]:
                val = spec["default"]
        validated[key] = val
    return cls(**validated)
