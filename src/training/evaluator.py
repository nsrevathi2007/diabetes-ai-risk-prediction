"""Model evaluation utilities and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metrics import build_classification_report, compute_metrics


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train a model and evaluate it on a held-out test set.

    Args:
        model: Scikit-learn compatible model instance.
        X_train: Training feature dataframe.
        X_test: Testing feature dataframe.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        Dictionary with evaluation metrics, confusion matrix, and report.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": build_classification_report(y_test, y_pred),
    }
