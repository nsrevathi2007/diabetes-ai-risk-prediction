"""MLflow experiment tracking helpers."""

from __future__ import annotations

from typing import Any

import os

import mlflow


class MLFlowTracker:
    """Wrap MLflow logging for optimized model runs."""

    def __init__(self, experiment_name: str = "diabetes-optimization", tracking_uri: str | None = None) -> None:
        """Initialize the tracker and configure MLflow."""
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        self.active_run: Any = None

    def log_run(self, metrics: dict[str, float], params: dict[str, Any], run_name: str) -> str:
        """Start a run and log metrics and parameters."""
        self.active_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return self.active_run.info.run_id

    def end_run(self) -> None:
        """End the currently active MLflow run."""
        if self.active_run is not None:
            mlflow.end_run()
            self.active_run = None
