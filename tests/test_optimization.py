"""Tests for model optimization and experiment tracking."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.optimization.imbalance_optimizer import compare_imbalance_strategies
from src.optimization.mlflow_tracker import MLFlowTracker
from src.optimization.optuna_optimizer import OptunaOptimizer
from src.optimization.threshold_optimizer import optimize_threshold


def sample_dataframe() -> pd.DataFrame:
    """Create a small dataframe for optimization tests."""
    return pd.DataFrame(
        {
            "Diabetes_binary": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "BMI": [18.0, 30.0, 20.0, 31.0, 22.0, 33.0, 24.0, 34.0, 26.0, 35.0] * 2,
            "Age": [4, 6, 5, 7, 4, 8, 5, 9, 6, 10] * 2,
            "HighBP": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "PhysActivity": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2,
            "Smoker": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
        }
    )


def test_optimize_threshold_returns_best_threshold() -> None:
    """Threshold optimization should return a threshold in the requested range."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.2, 0.8, 0.3, 0.7, 0.4, 0.9]

    result = optimize_threshold(y_true, y_proba)

    assert 0.1 <= result["threshold"] <= 0.9
    assert result["threshold"] > 0.0


def test_optuna_optimizer_can_run_mock_study() -> None:
    """The Optuna optimizer should return a simple study summary."""
    optuna_optimizer = OptunaOptimizer(model_name="RandomForest", n_trials=3)
    study_summary = optuna_optimizer.run_study(lambda trial: 0.5)

    assert study_summary["best_value"] == 0.5
    assert len(study_summary["trials"]) == 3


def test_compare_imbalance_strategies_returns_metadata() -> None:
    """Imbalance comparison should include the requested strategies."""
    df = sample_dataframe()
    strategies = compare_imbalance_strategies(df, target_column="Diabetes_binary")

    assert set(strategies.keys()) == {"Original", "SMOTE", "SMOTEENN", "RandomUnderSampler"}


def test_mlflow_tracker_logs_runs() -> None:
    """MLflow tracker should create a run and write metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = MLFlowTracker(experiment_name="test_experiment", tracking_uri=f"file:{Path(temp_dir)}/mlruns")
        run_id = tracker.log_run({"metric": 0.5}, {"param": 1}, "test")

        assert run_id is not None
        tracker.end_run()
