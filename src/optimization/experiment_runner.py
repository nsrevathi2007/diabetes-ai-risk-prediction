"""Orchestration for optimization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..training.dataset import prepare_training_data
from ..training.metrics import compute_metrics
from ..training.model_factory import build_model
from .imbalance_optimizer import (
    apply_imbalance_strategy,
    get_imbalance_strategy_names,
)
from .mlflow_tracker import MLFlowTracker
from .optuna_optimizer import OptunaOptimizer
from .report_generator import build_optimization_report, write_optimization_report
from .threshold_optimizer import optimize_threshold


class ExperimentRunner:
    """Coordinate Optuna optimization, imbalance comparison, and threshold tuning."""

    def __init__(
        self,
        output_dir: str | Path = "artifacts",
        reports_dir: str | Path = "reports/model_reports",
        target_column: str = "Diabetes_binary",
        n_trials: int = 3,
        random_state: int = 42,
        max_tuning_rows: int = 30000,
    ) -> None:
        """Initialize the experiment runner."""
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.reports_dir = Path(reports_dir)
        self.target_column = target_column
        self.n_trials = n_trials
        self.random_state = random_state
        self.max_tuning_rows = max_tuning_rows
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run the optimization workflow for the supported models."""
        tracker = MLFlowTracker(
            experiment_name="diabetes-optimization",
            tracking_uri="file:mlruns",
        )
        X_train, X_test, y_train, y_test = prepare_training_data(
            df,
            target_column=self.target_column,
            random_state=self.random_state,
        )
        X_tune, y_tune = self._sample_tuning_data(X_train, y_train)
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_tune,
            y_tune,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y_tune,
        )

        baseline_metrics = self._load_or_compute_baseline_metrics(df)
        comparison_results: list[dict[str, Any]] = []

        for model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            optimizer = OptunaOptimizer(model_name=model_name, n_trials=self.n_trials)
            study_summary = optimizer.run_study(
                lambda trial, name=model_name: self._objective(name, trial, X_fit, X_valid, y_fit, y_valid)
            )
            best_params = study_summary["best_params"]
            best_strategy = self._select_imbalance_strategy(
                model_name,
                best_params,
                X_fit,
                X_valid,
                y_fit,
                y_valid,
            )

            X_strategy, y_strategy = apply_imbalance_strategy(X_fit, y_fit, best_strategy, self.random_state)
            validation_model = self._build_configured_model(model_name, best_params)
            validation_model.fit(X_strategy, y_strategy)
            validation_proba = self._predict_positive_probability(validation_model, X_valid)
            threshold_result = optimize_threshold(y_valid, validation_proba)

            X_final, y_final = apply_imbalance_strategy(X_train, y_train, best_strategy, self.random_state)
            final_model = self._build_configured_model(model_name, best_params)
            final_model.fit(X_final, y_final)

            test_proba = self._predict_positive_probability(final_model, X_test)
            test_pred = (test_proba >= threshold_result["threshold"]).astype(int)
            optimized_metrics = compute_metrics(y_test, test_pred)
            optimized_metrics["roc_auc"] = self._safe_roc_auc(y_test, test_proba)

            tracker.log_run(
                {
                    "accuracy": optimized_metrics["accuracy"],
                    "recall": optimized_metrics["recall"],
                    "roc_auc": optimized_metrics["roc_auc"],
                    "best_threshold": threshold_result["threshold"],
                },
                {**best_params, "imbalance_strategy": best_strategy},
                model_name,
            )
            tracker.end_run()

            baseline = baseline_metrics[model_name]
            comparison_results.append(
                self._build_comparison_row(
                    model_name=model_name,
                    baseline=baseline,
                    optimized=optimized_metrics,
                    threshold=threshold_result["threshold"],
                    strategy=best_strategy,
                    best_params=best_params,
                )
            )

        comparison_df = pd.DataFrame(comparison_results)
        comparison_path = self.results_dir / "optimized_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        report_path = self.reports_dir / "optimization_report.md"
        write_optimization_report(report_path, build_optimization_report(comparison_df))

        return {
            "comparison_results": comparison_results,
            "comparison_path": comparison_path,
            "report_path": report_path,
        }

    def _objective(
        self,
        model_name: str,
        trial: Any,
        X_fit: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_fit: pd.Series,
        y_valid: pd.Series,
    ) -> float:
        """Optuna objective that evaluates sampled hyperparameters on validation ROC-AUC."""
        params = self._suggest_params(model_name, trial)
        model = self._build_configured_model(model_name, params)
        model.fit(X_fit, y_fit)
        y_proba = self._predict_positive_probability(model, X_valid)
        return self._safe_roc_auc(y_valid, y_proba)

    def _select_imbalance_strategy(
        self,
        model_name: str,
        best_params: dict[str, Any],
        X_fit: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_fit: pd.Series,
        y_valid: pd.Series,
    ) -> str:
        """Choose the imbalance strategy that gives the best validation ROC-AUC."""
        best_strategy = "Original"
        best_score = -np.inf
        for strategy_name in get_imbalance_strategy_names():
            X_resampled, y_resampled = apply_imbalance_strategy(
                X_fit,
                y_fit,
                strategy_name,
                self.random_state,
            )
            model = self._build_configured_model(model_name, best_params)
            model.fit(X_resampled, y_resampled)
            y_proba = self._predict_positive_probability(model, X_valid)
            score = self._safe_roc_auc(y_valid, y_proba)
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        return best_strategy

    def _build_configured_model(self, model_name: str, params: dict[str, Any]) -> Any:
        """Build a model and apply optimized hyperparameters."""
        model = build_model(model_name, random_state=self.random_state)
        if model_name == "CatBoost":
            model.set_params(allow_writing_files=False)
        if params:
            model.set_params(**params)
        return model

    def _suggest_params(self, model_name: str, trial: Any) -> dict[str, Any]:
        """Suggest a compact hyperparameter search space for each optimized model."""
        if model_name == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 140),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            }
        if model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 140),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            }
        if model_name == "LightGBM":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 140),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            }
        if model_name == "CatBoost":
            return {
                "iterations": trial.suggest_int("iterations", 50, 140),
                "depth": trial.suggest_int("depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
            }
        raise ValueError(f"Unsupported model for optimization: {model_name}")

    def _predict_positive_probability(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for a fitted classifier."""
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        decision = np.asarray(model.decision_function(X))
        return 1 / (1 + np.exp(-decision))

    def _safe_roc_auc(self, y_true: Any, y_score: Any) -> float:
        """Compute ROC-AUC while handling degenerate label inputs."""
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return float("nan")

    def _load_or_compute_baseline_metrics(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Load baseline metrics from training output or compute them when unavailable."""
        comparison_path = self.results_dir / "model_comparison.csv"
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            return {
                str(row["model_name"]): {
                    "accuracy": float(row["accuracy"]),
                    "recall": float(row["recall"]),
                    "roc_auc": float(row["roc_auc"]),
                }
                for _, row in comparison_df.iterrows()
            }

        X_train, X_test, y_train, y_test = prepare_training_data(
            df,
            target_column=self.target_column,
            random_state=self.random_state,
        )
        metrics_by_model: dict[str, dict[str, float]] = {}
        for model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            model = build_model(model_name, random_state=self.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics_by_model[model_name] = compute_metrics(y_test, y_pred)
        return metrics_by_model

    def _sample_tuning_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a bounded stratified sample for optimization selection work."""
        if len(X_train) <= self.max_tuning_rows:
            return X_train, y_train

        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=self.max_tuning_rows,
            random_state=self.random_state,
            stratify=y_train,
        )
        return X_tune, y_tune

    def _build_comparison_row(
        self,
        model_name: str,
        baseline: dict[str, float],
        optimized: dict[str, float],
        threshold: float,
        strategy: str,
        best_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Create one row for the optimized model comparison artifact."""
        baseline_roc_auc = baseline["roc_auc"]
        optimized_roc_auc = optimized["roc_auc"]
        percentage_improvement = (
            ((optimized_roc_auc - baseline_roc_auc) / baseline_roc_auc) * 100
            if baseline_roc_auc
            else 0.0
        )
        return {
            "Model Name": model_name,
            "Baseline Accuracy": baseline["accuracy"],
            "Optimized Accuracy": optimized["accuracy"],
            "Baseline Recall": baseline["recall"],
            "Optimized Recall": optimized["recall"],
            "Baseline ROC-AUC": baseline_roc_auc,
            "Optimized ROC-AUC": optimized_roc_auc,
            "Best Threshold": threshold,
            "Best Imbalance Strategy": strategy,
            "Best Hyperparameters": json.dumps(best_params, sort_keys=True),
            "Percentage Improvement": percentage_improvement,
        }
