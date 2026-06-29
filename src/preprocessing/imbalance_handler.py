"""Imbalance handling utilities for preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler


class ImbalanceHandler:
    """Provide configurable oversampling and undersampling strategies."""

    def __init__(self, strategy: str = "SMOTE", logger: Any = None) -> None:
        """Initialize the imbalance handler."""
        self.strategy = strategy
        self.logger = logger

    def compare_strategies(self, df: pd.DataFrame, target_column: str = "Diabetes_binary") -> dict[str, Any]:
        """Compare supported imbalance strategies without applying them permanently."""
        strategies = {
            "SMOTE": SMOTE(random_state=42),
            "SMOTEENN": SMOTEENN(random_state=42),
            "RandomUnderSampler": RandomUnderSampler(random_state=42),
        }

        reports: dict[str, Any] = {}
        for name, sampler in strategies.items():
            if name != self.strategy and self.strategy != "ALL":
                continue
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            reports[name] = {
                "rows": int(len(X_resampled)),
                "class_counts": {str(key): int(value) for key, value in y_resampled.value_counts().to_dict().items()},
            }

        return reports
