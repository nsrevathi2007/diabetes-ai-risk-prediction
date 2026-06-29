"""Healthcare-focused evaluation metrics for classification models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing the requested metrics.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    precision = precision_score(y_true_array, y_pred_array, zero_division=0)
    recall = recall_score(y_true_array, y_pred_array, zero_division=0)
    specificity = _specificity_score(y_true_array, y_pred_array)
    f1 = f1_score(y_true_array, y_pred_array, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_array, y_pred_array)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true_array, y_pred_array)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else float("nan"),
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "mcc": float(matthews_corrcoef(y_true_array, y_pred_array)),
    }


def _specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity as true negatives over all negatives."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] != 2:
        return 0.0
    tn, fp = cm[0][0], cm[0][1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def build_classification_report(y_true: Any, y_pred: Any) -> str:
    """Create a human-readable classification report."""
    return classification_report(y_true, y_pred, zero_division=0)
