"""Feature classification utilities for preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureClassifier:
    """Classify dataset features into reusable preprocessing categories."""

    def classify_features(
        self, df: pd.DataFrame, target_column: str = "Diabetes_binary"
    ) -> dict[str, str]:
        """Classify each feature in a dataframe.

        Args:
            df: Input dataframe containing features and the target column.
            target_column: Name of the target column to exclude from classification.

        Returns:
            Dictionary mapping each feature name to a category.
        """
        feature_types: dict[str, str] = {}

        for column in df.columns:
            if column == target_column:
                continue
            feature_types[column] = self._classify_feature(df[column])

        return feature_types

    def _classify_feature(self, series: pd.Series) -> str:
        """Assign a category to an individual feature series."""
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            return "Categorical"

        if not pd.api.types.is_numeric_dtype(series):
            return "Categorical"

        values = series.dropna()
        if values.empty:
            return "Categorical"

        unique_values = values.nunique(dropna=True)
        numeric_values = values.astype(float)

        if unique_values <= 2 and set(numeric_values.unique()).issubset({0.0, 1.0}):
            return "Binary"

        if self._is_integer_like(numeric_values) and unique_values <= 6:
            return "Ordinal"

        if self._is_integer_like(numeric_values) and unique_values <= 10:
            return "Ordinal"

        return "Continuous"

    def _is_integer_like(self, values: pd.Series) -> bool:
        """Check whether values resemble discrete ordinal integers."""
        return bool(np.all(np.isclose(values, np.round(values))))
