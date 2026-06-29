"""Utilities for comparing imbalance handling strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def compare_imbalance_strategies(
    df: pd.DataFrame, target_column: str = "Diabetes_binary"
) -> dict[str, Any]:
    """Compare original data and the supported oversampling/undersampling strategies."""
    strategies = {
        "Original": None,
        "SMOTE": SMOTE(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }

    results: dict[str, Any] = {}
    for name, sampler in strategies.items():
        if sampler is None:
            X_resampled = df.drop(columns=[target_column])
            y_resampled = df[target_column]
        else:
            X_resampled, y_resampled = sampler.fit_resample(
                df.drop(columns=[target_column]), df[target_column]
            )

        results[name] = {
            "rows": int(len(X_resampled)),
            "class_counts": {
                str(key): int(value)
                for key, value in y_resampled.value_counts().to_dict().items()
            },
        }

    return results
