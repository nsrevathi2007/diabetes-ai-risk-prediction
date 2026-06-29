"""Feature scaling utilities for preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FeatureScaler:
    """Scale continuous features while leaving binary features untouched."""

    def __init__(self, scaler_type: str = "StandardScaler", logger: Any = None) -> None:
        """Initialize the scaler with a supported sklearn scaler name."""
        self.scaler_type = scaler_type
        self.logger = logger
        self.scaler = self._build_scaler()
        self.continuous_features: list[str] = []

    def _build_scaler(self):
        """Create the underlying sklearn scaler."""
        if self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        if self.scaler_type == "RobustScaler":
            return RobustScaler()
        return StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and transform the dataframe."""
        scaled_df = df.copy()
        self.continuous_features = [
            column
            for column in scaled_df.columns
            if column != "Diabetes_binary"
            and pd.api.types.is_numeric_dtype(scaled_df[column])
            and scaled_df[column].nunique(dropna=True) > 2
        ]

        for column in self.continuous_features:
            scaled_values = self.scaler.fit_transform(scaled_df[[column]])
            scaled_df[column] = scaled_values.flatten()

        if self.logger is not None:
            self.logger.info("Scaled continuous features: %s", self.continuous_features)

        return scaled_df
