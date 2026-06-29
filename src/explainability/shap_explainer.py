"""SHAP explainer construction and value calculation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Load a trained model and calculate SHAP values without retraining."""

    TREE_MODEL_MARKERS = (
        "RandomForest",
        "DecisionTree",
        "XGB",
        "XGBoost",
        "LGBM",
        "LightGBM",
        "CatBoost",
        "GradientBoosting",
    )

    def __init__(
        self,
        model_path: str | Path = "artifacts/models/best_model.joblib",
        fallback_comparison_path: str | Path = "artifacts/results/optimized_model_comparison.csv",
        models_dir: str | Path = "artifacts/models",
        metadata_dir: str | Path = "artifacts",
        logger: Any | None = None,
    ) -> None:
        """Initialize the SHAP explainer service.

        Args:
            model_path: Preferred path to the saved best model.
            fallback_comparison_path: Optimized comparison used to find a model
                artifact when ``best_model.joblib`` is not available.
            models_dir: Directory containing saved model artifacts.
            metadata_dir: Directory that may contain preprocessing metadata.
            logger: Optional project logger.
        """
        self.model_path = Path(model_path)
        self.fallback_comparison_path = Path(fallback_comparison_path)
        self.models_dir = Path(models_dir)
        self.metadata_dir = Path(metadata_dir)
        self.logger = logger
        self.model: Any | None = None
        self.loaded_model_path: Path | None = None
        self.explainer: Any | None = None
        self.background: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.expected_value: float = 0.0

    def load_model(self) -> Any:
        """Load the preferred saved model artifact.

        Returns:
            A fitted model loaded from disk.

        Raises:
            FileNotFoundError: If neither the preferred model nor a fallback
                model artifact can be found.
        """
        start_time = time.perf_counter()
        resolved_path = self._resolve_model_path()
        self.model = joblib.load(resolved_path)
        self.loaded_model_path = resolved_path
        self._log(
            "Loaded model from %s in %.2fs",
            resolved_path,
            time.perf_counter() - start_time,
        )
        return self.model

    def load_feature_metadata(self) -> dict[str, Any]:
        """Load feature metadata when a known metadata file is available."""
        candidates = [
            self.metadata_dir / "preprocessing_metadata.json",
            self.metadata_dir / "metadata.json",
            self.metadata_dir / "feature_metadata.json",
        ]
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def create_explainer(
        self,
        X_background: pd.DataFrame,
        max_background_rows: int = 500,
    ) -> Any:
        """Create a SHAP explainer appropriate for the loaded model.

        Args:
            X_background: Feature data used as explainer background.
            max_background_rows: Maximum rows to use for background data.

        Returns:
            A SHAP explainer instance.
        """
        if self.model is None:
            self.load_model()

        self.feature_names = list(X_background.columns)
        self.background = self._sample_frame(X_background, max_background_rows)
        model_type = self.detect_model_type()
        start_time = time.perf_counter()

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            masker = shap.maskers.Independent(self.background)
            self.explainer = shap.Explainer(self._predict_positive_probability, masker)

        self.expected_value = self._extract_expected_value()
        self._log(
            "Created %s SHAP explainer in %.2fs",
            model_type,
            time.perf_counter() - start_time,
        )
        return self.explainer

    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        max_rows: int | None = None,
    ) -> np.ndarray:
        """Calculate positive-class SHAP values for a feature dataframe.

        Args:
            X: Feature dataframe to explain.
            max_rows: Optional maximum rows to explain.

        Returns:
            A two-dimensional array of SHAP values.
        """
        if self.explainer is None:
            self.create_explainer(X)

        X_to_explain = self._sample_frame(X, max_rows) if max_rows else X
        start_time = time.perf_counter()
        if hasattr(self.explainer, "shap_values"):
            raw_values = self.explainer.shap_values(X_to_explain)
        else:
            raw_values = self.explainer(X_to_explain)
        shap_values = self._normalize_shap_values(raw_values)
        self._log(
            "Calculated SHAP values for %s rows in %.2fs",
            len(X_to_explain),
            time.perf_counter() - start_time,
        )
        return shap_values

    def explain_dataframe(
        self,
        X: pd.DataFrame,
        max_rows: int | None = None,
    ) -> shap.Explanation:
        """Return a SHAP Explanation object for plotting."""
        X_to_explain = self._sample_frame(X, max_rows) if max_rows else X
        values = self.calculate_shap_values(X_to_explain)
        return shap.Explanation(
            values=values,
            base_values=np.repeat(self.expected_value, len(X_to_explain)),
            data=X_to_explain.to_numpy(),
            feature_names=list(X_to_explain.columns),
        )

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive-class diabetes risk probabilities."""
        if self.model is None:
            self.load_model()
        return self._predict_positive_probability(X)

    def detect_model_type(self) -> str:
        """Detect whether the loaded model is tree-based."""
        if self.model is None:
            return "unknown"
        class_name = self.model.__class__.__name__
        module_name = self.model.__class__.__module__
        if any(marker in class_name or marker in module_name for marker in self.TREE_MODEL_MARKERS):
            return "tree"
        return "model_agnostic"

    def _resolve_model_path(self) -> Path:
        """Resolve best model path with a no-retraining fallback."""
        if self.model_path.exists():
            return self.model_path

        fallback_path = self._resolve_fallback_model_path()
        if fallback_path is not None:
            self._log(
                "Preferred model %s not found; using fallback artifact %s",
                self.model_path,
                fallback_path,
            )
            return fallback_path

        raise FileNotFoundError(
            f"Model artifact not found: {self.model_path}. "
            "Expected artifacts/models/best_model.joblib or a saved model named in optimized_model_comparison.csv."
        )

    def _resolve_fallback_model_path(self) -> Path | None:
        """Find a saved model artifact from the optimized comparison file."""
        if not self.fallback_comparison_path.exists():
            return None

        comparison_df = pd.read_csv(self.fallback_comparison_path)
        if comparison_df.empty or "Model Name" not in comparison_df.columns:
            return None

        sort_columns = [
            column
            for column in ["Optimized ROC-AUC", "Optimized Recall", "Optimized Accuracy"]
            if column in comparison_df.columns
        ]
        if sort_columns:
            comparison_df = comparison_df.sort_values(sort_columns, ascending=False)

        for model_name in comparison_df["Model Name"]:
            candidate = self.models_dir / f"{str(model_name).lower()}.joblib"
            if candidate.exists():
                return candidate
        return None

    def _predict_positive_probability(self, X: Any) -> np.ndarray:
        """Return positive-class probabilities for SHAP model-agnostic calls."""
        frame = pd.DataFrame(X, columns=self.feature_names) if not isinstance(X, pd.DataFrame) else X
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(frame))[:, 1]
        decision = np.asarray(self.model.decision_function(frame))
        return 1 / (1 + np.exp(-decision))

    def _extract_expected_value(self) -> float:
        """Extract the positive-class expected value from a SHAP explainer."""
        expected_value = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            values = np.asarray(expected_value).reshape(-1)
            return float(values[1] if len(values) > 1 else values[0])
        return float(expected_value)

    def _normalize_shap_values(self, raw_values: Any) -> np.ndarray:
        """Normalize SHAP outputs to a two-dimensional positive-class array."""
        if isinstance(raw_values, list):
            values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
        else:
            values = raw_values

        if hasattr(values, "base_values"):
            base_values = np.asarray(values.base_values).reshape(-1)
            if len(base_values) > 0:
                self.expected_value = float(base_values[0])

        if hasattr(values, "values"):
            values = values.values

        values_array = np.asarray(values)
        if values_array.ndim == 3:
            values_array = values_array[:, :, 1] if values_array.shape[2] > 1 else values_array[:, :, 0]
        if values_array.ndim == 1:
            values_array = values_array.reshape(1, -1)
        return values_array

    def _sample_frame(self, X: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
        """Return a deterministic sample when the dataframe is larger than needed."""
        if max_rows is None or len(X) <= max_rows:
            return X.copy()
        return X.sample(n=max_rows, random_state=42).sort_index()

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
