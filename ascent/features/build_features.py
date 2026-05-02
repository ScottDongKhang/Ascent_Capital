"""
Ascent Capital — Feature Builder
Orchestrates feature computation from normalized price data.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ascent.data.normalize.prices import pivot_prices, pivot_macro
from ascent.features.feature_defs import build_all_features
from ascent.features.targets import build_targets


class FeatureBuilder:
    """Builds feature matrices and targets from normalized data."""

    def __init__(self, price_df: pd.DataFrame, macro_df: pd.DataFrame | None = None,
                 fundamentals_df: pd.DataFrame | None = None,
                 earnings_df: pd.DataFrame | None = None,
                 analyst_df: pd.DataFrame | None = None):
        self.price_df = price_df
        self.macro_df = macro_df

        # Pivot to wide format
        self.close = pivot_prices(price_df, "close")
        self.open = pivot_prices(price_df, "open")
        self.high = pivot_prices(price_df, "high")
        self.low = pivot_prices(price_df, "low")
        self.volume = pivot_prices(price_df, "volume")

        # Dollar volume
        dv = price_df.copy()
        dv["dollar_volume"] = dv["close"] * dv["volume"]
        self.dollar_volume = dv.pivot_table(
            index="date", columns="symbol", values="dollar_volume", aggfunc="last"
        ).sort_index()

        # Macro
        self.macro_pivot = None
        if macro_df is not None and not macro_df.empty:
            self.macro_pivot = pivot_macro(macro_df)
        self.fundamentals_df = fundamentals_df
        self.earnings_df = earnings_df
        self.analyst_df = analyst_df

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def compute_features(self) -> dict[str, pd.DataFrame]:
        """Compute all features. Returns {name: DataFrame(dates × symbols)}."""
        features = build_all_features(
            self.close, self.volume, self.dollar_volume, self.macro_pivot,
            earnings_df=self.earnings_df,
            analyst_df=self.analyst_df,
        )
        if self.fundamentals_df is not None and not self.fundamentals_df.empty:
            try:
                from ascent.features.feature_defs import build_fundamental_panel
                panel = build_fundamental_panel(
                    self.fundamentals_df, self.close.index, list(self.close.columns)
                )
                features.update(panel)
            except Exception as e:
                import logging as _log
                _log.getLogger(__name__).warning("fundamental panel failed: %s", e)
        return features

    def compute_targets(self, horizons: list[int] = [1, 5, 21]) -> dict[str, pd.DataFrame]:
        """Compute forward return targets."""
        return build_targets(self.close, self.open, horizons)

    def to_panel(self, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert features dict to long-format panel: (date, symbol, feature1, feature2, ...).
        Useful for cross-sectional analysis and model training.
        """
        frames = []
        for name, feat_df in features.items():
            melted = feat_df.reset_index().melt(
                id_vars="date", var_name="symbol", value_name=name
            )
            if not frames:
                frames.append(melted)
            else:
                frames[0] = frames[0].merge(melted, on=["date", "symbol"], how="outer")

        if not frames:
            return pd.DataFrame()
        return frames[0].sort_values(["date", "symbol"]).reset_index(drop=True)
