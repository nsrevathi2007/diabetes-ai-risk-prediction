"""Reusable preprocessing pipeline for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data_cleaner import DataCleaner
from .feature_classifier import FeatureClassifier
from .feature_engineering import FeatureEngineering
from .feature_selector import FeatureSelector
from .scaler import FeatureScaler


class PreprocessingPipeline:
    """Compose feature classification, cleaning, engineering, selection, and scaling."""

    def __init__(self, target_column: str = "Diabetes_binary", logger: Any = None) -> None:
        """Initialize the preprocessing pipeline."""
        self.target_column = target_column
        self.logger = logger
        self.feature_classifier = FeatureClassifier()
        self.data_cleaner = DataCleaner(logger=logger)
        self.feature_engineering = FeatureEngineering(logger=logger)
        self.feature_selector = FeatureSelector(logger=logger)
        self.scaler = FeatureScaler(logger=logger)
        self.feature_types: dict[str, str] = {}
        self.cleaning_report: dict[str, Any] = {}
        self.feature_selection_report: dict[str, Any] = {}
        self.engineered_features: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and return the transformed dataframe."""
        self.feature_types = self.feature_classifier.classify_features(df, self.target_column)
        self.cleaning_report = self.data_cleaner.validate_and_report(df, self.target_column)

        engineered_df = self.feature_engineering.engineer_features(df)
        self.engineered_features = self.feature_engineering.engineered_features
        self.feature_selection_report = self.feature_selector.rank_features(engineered_df, self.target_column)

        transformed_df = self.scaler.fit_transform(engineered_df)
        return transformed_df

    def save_artifacts(self, output_dir: str | Path = "artifacts") -> dict[str, Path]:
        """Persist preprocessing artifacts to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_path = output_path / "feature_metadata.json"
        metadata_path.write_text(
            str({
                "feature_types": self.feature_types,
                "cleaning_report": self.cleaning_report,
                "feature_selection": self.feature_selection_report,
                "engineered_features": self.engineered_features,
            }),
            encoding="utf-8",
        )

        pipeline_path = output_path / "preprocessing_pipeline.joblib"
        joblib.dump(self, pipeline_path)

        scaler_path = output_path / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)

        engineered_features_path = output_path / "engineered_features.json"
        engineered_features_path.write_text(
            str(self.engineered_features),
            encoding="utf-8",
        )

        return {
            "metadata": metadata_path,
            "pipeline": pipeline_path,
            "scaler": scaler_path,
            "engineered_features": engineered_features_path,
        }
