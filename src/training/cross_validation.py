"""Cross-validation utilities for baseline model assessment."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


def compute_cross_validation_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> dict[str, float]:
    """Compute cross-validated metrics for a model.

    Args:
        model: Scikit-learn compatible estimator.
        X: Feature matrix.
        y: Target vector.
        cv: Number of folds.
        scoring: Scoring metric for CV.

    Returns:
        Dictionary containing mean score and standard deviation.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=None)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }
