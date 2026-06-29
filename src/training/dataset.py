"""Dataset loading and preparation utilities for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = "Diabetes_binary",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataframe into stratified train and test sets.

    Args:
        df: Dataframe that already contains the target column.
        target_column: Name of the target column.
        test_size: Fraction to reserve for testing.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of training features, testing features, training targets, testing targets.
    """
    feature_frame = df.drop(columns=[target_column], errors="ignore")
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return X_train, X_test, y_train, y_test


def load_dataset(path: str | Path, target_column: str = "Diabetes_binary") -> pd.DataFrame:
    """Load the processed dataset from disk and ensure the target column exists."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    return dataset
