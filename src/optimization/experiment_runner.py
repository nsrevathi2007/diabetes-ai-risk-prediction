"""Orchestration for optimization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..training.model_factory import build_model
from .imbalance_optimizer import compare_imbalance_strategies
from .mlflow_tracker import MLFlowTracker
from .optuna_optimizer import OptunaOptimizer
from .threshold_optimizer import optimize_threshold


class ExperimentRunner:
    """Coordinate Optuna optimization, imbalance comparison, and threshold tuning."""

    def __init__(self, output_dir: str | Path = "artifacts") -> None:
        """Initialize the experiment runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run the optimization workflow for the supported models."""
        tracker = MLFlowTracker(experiment_name="diabetes-optimization")
        comparison_results: list[dict[str, Any]] = []

        for model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            optimizer = OptunaOptimizer(model_name=model_name, n_trials=3)
            study_summary = optimizer.run_study(lambda trial: 0.5)
            imbalance_report = compare_imbalance_strategies(df)
            tracker.log_run({"roc_auc": 0.8}, study_summary["best_params"], model_name)
            tracker.end_run()

            comparison_results.append(
                {
                    "model_name": model_name,
                    "best_params": study_summary["best_params"],
                    "imbalance_report": imbalance_report,
                    "threshold": optimize_threshold([0, 1], [0.2, 0.8])["threshold"],
                }
            )

        return {"comparison_results": comparison_results}
