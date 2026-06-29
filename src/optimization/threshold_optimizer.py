"""Probability threshold optimization for healthcare classification."""

from __future__ import annotations

from typing import Any

import numpy as np


def optimize_threshold(y_true: Any, y_proba: Any, start: float = 0.10, end: float = 0.90) -> dict[str, Any]:
    """Search a threshold range to maximize recall while maintaining precision.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities for the positive class.
        start: Lower bound of the threshold search range.
        end: Upper bound of the threshold search range.

    Returns:
        Dictionary with the selected threshold and associated metrics.
    """
    y_true_array = np.asarray(y_true)
    y_proba_array = np.asarray(y_proba)

    best_result: dict[str, Any] = {"threshold": 0.5, "recall": -1.0, "precision": 0.0}
    for threshold in np.linspace(start, end, 81):
        y_pred = (y_proba_array >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true_array == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_array == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_array == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if recall > best_result["recall"] and precision >= 0.2:
            best_result = {
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": float(precision),
            }

    return best_result
