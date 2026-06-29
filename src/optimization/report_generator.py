"""Report generation for optimization results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_optimization_report(output_path: str | Path, content: str) -> Path:
    """Write the optimization markdown report to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    return output_file


def build_optimization_report(comparison_df: pd.DataFrame) -> str:
    """Build a markdown summary for optimized model performance."""
    best_row = comparison_df.sort_values(
        by=["Optimized ROC-AUC", "Optimized Recall", "Optimized Accuracy"],
        ascending=False,
    ).iloc[0]

    comparison_lines = [
        "| Model | Baseline ROC-AUC | Optimized ROC-AUC | Baseline Recall | Optimized Recall | Improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison_df.iterrows():
        comparison_lines.append(
            "| {model} | {base_auc:.4f} | {opt_auc:.4f} | {base_recall:.4f} | {opt_recall:.4f} | {improvement:.2f}% |".format(
                model=row["Model Name"],
                base_auc=row["Baseline ROC-AUC"],
                opt_auc=row["Optimized ROC-AUC"],
                base_recall=row["Baseline Recall"],
                opt_recall=row["Optimized Recall"],
                improvement=row["Percentage Improvement"],
            )
        )

    return "\n".join(
        [
            "# Optimization Report",
            "",
            "## Summary",
            (
                "The best optimized model was "
                f"{best_row['Model Name']} with ROC-AUC {best_row['Optimized ROC-AUC']:.4f}, "
                f"recall {best_row['Optimized Recall']:.4f}, and threshold {best_row['Best Threshold']:.2f}."
            ),
            "",
            "Percentage improvement is calculated from baseline ROC-AUC to optimized ROC-AUC.",
            "",
            "## Optimized Comparison",
            *comparison_lines,
            "",
            "## Selected Strategies",
            *[
                (
                    f"- {row['Model Name']}: {row['Best Imbalance Strategy']} imbalance handling, "
                    f"threshold {row['Best Threshold']:.2f}, hyperparameters `{row['Best Hyperparameters']}`"
                )
                for _, row in comparison_df.iterrows()
            ],
        ]
    )
