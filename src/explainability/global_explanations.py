"""Global SHAP explanation generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .shap_explainer import ShapExplainer
from .visualization import ShapVisualizer


class GlobalExplanationGenerator:
    """Generate global model behavior explanations from SHAP values."""

    def __init__(
        self,
        explainer: ShapExplainer,
        visualizer: ShapVisualizer,
        output_dir: str | Path = "reports/shap",
        logger: Any | None = None,
    ) -> None:
        """Initialize global explanation generation."""
        self.explainer = explainer
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def generate(
        self,
        X: pd.DataFrame,
        max_rows: int = 500,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Generate global SHAP plots and feature importance.

        Args:
            X: Feature dataframe.
            max_rows: Maximum rows for SHAP calculation.
            top_n: Number of top features to include.

        Returns:
            Dictionary with generated paths and top feature metadata.
        """
        start_time = time.perf_counter()
        explanation = self.explainer.explain_dataframe(X, max_rows=max_rows)
        importance_df = self.compute_feature_importance(
            explanation.values,
            list(explanation.feature_names),
            top_n=top_n,
        )

        summary_path = self.visualizer.save_summary_plot(explanation, self.output_dir / "summary_plot.png")
        bar_path = self.visualizer.save_bar_plot(explanation, self.output_dir / "bar_plot.png")
        importance_path = self.output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        self._log(
            "Generated global SHAP explanations in %.2fs",
            time.perf_counter() - start_time,
        )
        return {
            "summary_plot": summary_path,
            "bar_plot": bar_path,
            "feature_importance": importance_path,
            "top_features": importance_df.head(top_n).to_dict(orient="records"),
            "explanation": explanation,
        }

    def compute_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP importance."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        )
        return importance_df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
