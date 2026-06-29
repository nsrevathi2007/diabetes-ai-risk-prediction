"""Visualization helpers for SHAP explanations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ShapVisualizer:
    """Generate SHAP plots and save them to disk."""

    def __init__(self, output_dir: str | Path = "reports/shap", logger: Any | None = None) -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory where plots are written.
            logger: Optional project logger.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def save_summary_plot(self, explanation: shap.Explanation, output_path: str | Path | None = None) -> Path:
        """Save a SHAP beeswarm summary plot."""
        path = Path(output_path) if output_path else self.output_dir / "summary_plot.png"
        start_time = time.perf_counter()
        plt.figure()
        shap.summary_plot(
            explanation.values,
            features=pd.DataFrame(explanation.data, columns=explanation.feature_names),
            feature_names=explanation.feature_names,
            show=False,
        )
        self._save_current_figure(path)
        self._log("Generated SHAP summary plot at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def save_bar_plot(self, explanation: shap.Explanation, output_path: str | Path | None = None) -> Path:
        """Save a SHAP mean-absolute-importance bar plot."""
        path = Path(output_path) if output_path else self.output_dir / "bar_plot.png"
        start_time = time.perf_counter()
        plt.figure()
        shap.plots.bar(explanation, max_display=20, show=False)
        self._save_current_figure(path)
        self._log("Generated SHAP bar plot at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def save_waterfall_plot(
        self,
        local_explanation: shap.Explanation,
        output_path: str | Path,
        max_display: int = 15,
    ) -> Path:
        """Save a local SHAP waterfall plot."""
        path = Path(output_path)
        start_time = time.perf_counter()
        plt.figure()
        shap.plots.waterfall(local_explanation, max_display=max_display, show=False)
        self._save_current_figure(path)
        self._log("Generated SHAP waterfall plot at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def save_force_plot(self, local_explanation: shap.Explanation, output_path: str | Path) -> Path:
        """Save a local SHAP force plot as HTML."""
        path = Path(output_path)
        start_time = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        force_plot = shap.plots.force(local_explanation, matplotlib=False, show=False)
        shap.save_html(str(path), force_plot)
        self._log("Generated SHAP force plot at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def save_decision_plot(
        self,
        base_value: float,
        shap_values: list[float],
        features: pd.Series,
        output_path: str | Path,
    ) -> Path:
        """Save a local SHAP decision plot."""
        path = Path(output_path)
        start_time = time.perf_counter()
        plt.figure()
        shap.decision_plot(
            base_value,
            np.asarray(shap_values),
            features=features,
            feature_names=list(features.index),
            show=False,
        )
        self._save_current_figure(path)
        self._log("Generated SHAP decision plot at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def _save_current_figure(self, path: Path) -> None:
        """Save and close the current matplotlib figure."""
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
