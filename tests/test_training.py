"""Tests for the model training and evaluation framework."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.training.comparison import compare_models
from src.training.dataset import prepare_training_data
from src.training.evaluator import evaluate_model
from src.training.metrics import compute_metrics
from src.training.model_factory import build_model
from src.training.trainer import DiabetesModelTrainer


def sample_dataframe() -> pd.DataFrame:
    """Create a small binary classification dataset for training tests."""
    return pd.DataFrame(
        {
            "Diabetes_binary": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "BMI": [18.0, 30.0, 20.0, 31.0, 22.0, 33.0, 24.0, 34.0, 26.0, 35.0] * 2,
            "Age": [4, 6, 5, 7, 4, 8, 5, 9, 6, 10] * 2,
            "HighBP": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "PhysActivity": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2,
            "Smoker": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            "GenHlth": [2, 4, 3, 5, 2, 4, 3, 5, 2, 4] * 2,
        }
    )


def test_prepare_training_data_returns_expected_splits() -> None:
    """Training data preparation should return stratified train/test splits."""
    df = sample_dataframe()

    X_train, X_test, y_train, y_test = prepare_training_data(df, target_column="Diabetes_binary")

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_compute_metrics_returns_required_keys() -> None:
    """Metric computation should include the healthcare-focused metrics."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics


def test_evaluate_model_returns_training_outputs() -> None:
    """Model evaluation should return metrics, classification report, and confusion matrix."""
    df = sample_dataframe()
    X_train, X_test, y_train, y_test = prepare_training_data(df, target_column="Diabetes_binary")
    model = build_model("LogisticRegression")

    evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)

    assert evaluation["metrics"]["accuracy"] >= 0.0
    assert evaluation["confusion_matrix"] is not None
    assert "classification_report" in evaluation


def test_compare_models_ranks_results() -> None:
    """Comparison should produce a ranked table for model selection."""
    comparison_df = compare_models(
        [
            {
                "model_name": "LogisticRegression",
                "metrics": {"roc_auc": 0.82, "recall": 0.80, "f1": 0.78, "accuracy": 0.74},
            },
            {
                "model_name": "RandomForest",
                "metrics": {"roc_auc": 0.86, "recall": 0.84, "f1": 0.80, "accuracy": 0.76},
            },
        ]
    )

    assert comparison_df.iloc[0]["model_name"] == "RandomForest"
    assert "rank" in comparison_df.columns


def test_trainer_saves_models_and_results() -> None:
    """The trainer should save models and reports to disk."""
    df = sample_dataframe()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = DiabetesModelTrainer(
            config={
                "dataset_path": None,
                "target_column": "Diabetes_binary",
                "output_dir": Path(temp_dir),
                "models_dir": Path(temp_dir) / "models",
                "results_dir": Path(temp_dir) / "results",
                "reports_dir": Path(temp_dir) / "reports",
                "logger": None,
            }
        )
        result = trainer.train_from_dataframe(df)

        assert result["comparison_path"].exists()
        assert result["models"]["LogisticRegression"].exists()
