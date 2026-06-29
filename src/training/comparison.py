"""Comparison utilities for ranking trained models."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Rank models by ROC-AUC, recall, F1 score, and accuracy.

    Args:
        results: List of dictionaries containing model_name and metrics.

    Returns:
        Ranked dataframe suitable for reporting.
    """
    normalized_results: list[dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", result)
        if not isinstance(metrics, dict):
            metrics = {}

        normalized_result = {key: value for key, value in result.items() if key != "metrics"}
        normalized_result.update(metrics)
        normalized_results.append(normalized_result)

    comparison_df = pd.DataFrame(normalized_results)
    score_columns = [
        column
        for column in ["roc_auc", "recall", "f1", "accuracy"]
        if column in comparison_df.columns
    ]

    if "rank" in comparison_df.columns:
        comparison_df = comparison_df.drop(columns=["rank"])

    if not score_columns:
        raise ValueError("No scoring columns available for model comparison")

    comparison_df = comparison_df.sort_values(
        by=score_columns,
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    comparison_df.insert(0, "rank", range(1, len(comparison_df) + 1))
    return comparison_df
