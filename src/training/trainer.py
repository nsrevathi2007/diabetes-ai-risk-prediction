"""Training orchestration and estimator persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..utils.logging import configure_logger
from .comparison import compare_models
from .cross_validation import compute_cross_validation_metrics
from .dataset import load_dataset, prepare_training_data
from .evaluator import evaluate_model
from .model_factory import build_model, get_supported_models
from .report_generator import write_model_report


class DiabetesModelTrainer:
    """Encapsulates diabetes risk model training and evaluation lifecycle."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the trainer with configuration."""
        self.config = config
        self.logger = config.get("logger") or configure_logger({"log_dir": "logs", "log_file": "training.log"})
        self.output_dir = Path(self.config.get("output_dir", "artifacts"))
        self.models_dir = Path(self.config.get("models_dir", self.output_dir / "models"))
        self.results_dir = Path(self.config.get("results_dir", self.output_dir / "results"))
        self.reports_dir = Path(self.config.get("reports_dir", Path("reports/model_reports")))
        self.target_column = self.config.get("target_column", "Diabetes_binary")

    def train_from_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train baseline models from an in-memory dataframe."""
        self.logger.info("Training start")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_test, y_train, y_test = prepare_training_data(df, target_column=self.target_column)

        model_results: list[dict[str, Any]] = []
        saved_models: dict[str, Path] = {}

        for model_name in get_supported_models():
            model = build_model(model_name)
            evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)
            metrics = evaluation["metrics"]
            metrics["model_name"] = model_name

            cv_metrics = compute_cross_validation_metrics(model, X_train, y_train)
            metrics["cv_mean_roc_auc"] = cv_metrics["mean"]
            metrics["cv_std_roc_auc"] = cv_metrics["std"]

            model_results.append(metrics)

            model_path = self.models_dir / f"{model_name.lower()}.joblib"
            joblib.dump(evaluation["model"], model_path)
            saved_models[model_name] = model_path
            self.logger.info("Saved model: %s", model_path)

        comparison_df = compare_models(model_results)
        comparison_path = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        importance_path = self.results_dir / "feature_importance.json"
        importance_path.write_text("{}", encoding="utf-8")

        report_path = self.reports_dir / "model_training_report.md"
        write_model_report(report_path, self._build_markdown_report(comparison_df))

        self.logger.info("Training completion")
        return {
            "comparison_path": comparison_path,
            "importance_path": importance_path,
            "report_path": report_path,
            "models": saved_models,
        }

    def _build_markdown_report(self, comparison_df: pd.DataFrame) -> str:
        """Build a markdown summary report from the comparison dataframe."""
        best_model = comparison_df.iloc[0]["model_name"]
        comparison_lines = [
            "| rank | model_name | roc_auc | recall | f1 | accuracy |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in comparison_df.iterrows():
            comparison_lines.append(
                f"| {int(row['rank'])} | {row['model_name']} | {row.get('roc_auc', '')} | {row.get('recall', '')} | {row.get('f1', '')} | {row.get('accuracy', '')} |"
            )

        return "\n".join(
            [
                "# Model Training Report",
                "",
                "## Model Comparison",
                *comparison_lines,
                "",
                "## Best Performing Model",
                f"The best baseline model was {best_model} based on ROC-AUC, recall, F1 score, and accuracy.",
                "",
                "## Healthcare Considerations",
                "Healthcare classification should prioritize ROC-AUC, recall, and F1 score because false negatives can delay treatment.",
            ]
        )

    def train_from_file(self, dataset_path: str | Path) -> dict[str, Any]:
        """Train models from a dataset file on disk."""
        dataset = load_dataset(dataset_path, target_column=self.target_column)
        return self.train_from_dataframe(dataset)
