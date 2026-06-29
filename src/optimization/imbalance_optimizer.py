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
    results: dict[str, Any] = {}
    for name in get_imbalance_strategy_names():
        X_resampled, y_resampled = apply_imbalance_strategy(
            df.drop(columns=[target_column]), df[target_column], name
        )

        results[name] = {
            "rows": int(len(X_resampled)),
            "class_counts": {
                str(key): int(value)
                for key, value in y_resampled.value_counts().to_dict().items()
            },
        }

    return results


def get_imbalance_strategy_names() -> list[str]:
    """Return supported imbalance strategy names."""
    return ["Original", "SMOTE", "SMOTEENN", "RandomUnderSampler"]


def apply_imbalance_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    strategy_name: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply an imbalance strategy to a feature matrix and target vector."""
    if strategy_name == "Original":
        return X.copy(), y.copy()

    sampler = _build_sampler(strategy_name, random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)


def _build_sampler(strategy_name: str, random_state: int) -> Any:
    """Build an imbalanced-learn sampler by name."""
    if strategy_name == "SMOTE":
        return SMOTE(random_state=random_state)
    if strategy_name == "SMOTEENN":
        return SMOTEENN(random_state=random_state)
    if strategy_name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=random_state)
    raise ValueError(f"Unsupported imbalance strategy: {strategy_name}")
