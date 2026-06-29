"""Tests for preprocessing pipeline components."""

import numpy as np
import pandas as pd

from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_classifier import FeatureClassifier
from src.preprocessing.feature_engineering import FeatureEngineering
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.preprocessing.scaler import FeatureScaler


def sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic dataset for preprocessing tests."""
    return pd.DataFrame(
        {
            "Diabetes_binary": [0, 1, 0, 1],
            "HighBP": [0, 1, 1, 0],
            "HighChol": [0, 1, 1, 1],
            "BMI": [18.0, 31.5, 22.0, 35.4],
            "Smoker": [0, 1, 0, 1],
            "PhysActivity": [1, 0, 1, 0],
            "Age": [7, 9, 4, 10],
            "GenHlth": [2, 4, 3, 5],
            "Education": [4, 2, 3, 1],
            "Income": [7, 3, 5, 2],
            "Sex": [0, 1, 0, 1],
            "Fruits": [1, 0, 1, 0],
        }
    )


def test_feature_classification_assigns_expected_types():
    """Features should be classified into supported categories."""
    df = sample_dataframe()
    classifier = FeatureClassifier()

    feature_types = classifier.classify_features(df)

    assert feature_types["BMI"] == "Continuous"
    assert feature_types["HighBP"] == "Binary"
    assert feature_types["GenHlth"] == "Ordinal"
    assert feature_types["Fruits"] == "Binary"


def test_data_cleaner_generates_issue_report():
    """The cleaner should detect invalid values and report them."""
    df = sample_dataframe().copy()
    df.loc[0, "BMI"] = 0.0
    df.loc[1, "Age"] = -1
    df.loc[2, "HighBP"] = -1
    df.loc[3, "Diabetes_binary"] = 2
    df.loc[4] = df.loc[0]

    cleaner = DataCleaner()
    report = cleaner.validate_and_report(df)

    assert report["duplicate_rows"] > 0
    assert report["issue_count"] > 0
    assert len(report["issues"]) > 0


def test_feature_engineering_adds_healthcare_features():
    """Engineered features should be added to the dataset."""
    df = sample_dataframe()
    engineer = FeatureEngineering()

    transformed = engineer.engineer_features(df)

    expected_features = {
        "BMI_Category",
        "Age_Group",
        "Lifestyle_Risk_Score",
        "Cardiovascular_Risk_Score",
        "Health_Burden_Score",
    }

    assert expected_features.issubset(transformed.columns)


def test_feature_scaler_only_scales_continuous_features():
    """Binary features should remain unchanged while continuous ones are scaled."""
    df = sample_dataframe()[["BMI", "HighBP", "Age", "Diabetes_binary"]].copy()
    scaler = FeatureScaler(scaler_type="StandardScaler")

    transformed = scaler.fit_transform(df)

    assert np.isclose(transformed["HighBP"].mean(), 0.5, atol=1e-9)
    assert np.isclose(transformed["BMI"].mean(), 0.0, atol=1e-9)
    assert "BMI" in transformed.columns


def test_feature_selector_returns_rankings():
    """The selector should produce rankings for each requested test."""
    df = sample_dataframe()
    selector = FeatureSelector()

    ranking = selector.rank_features(df, target_column="Diabetes_binary")

    assert "mutual_information" in ranking
    assert "chi_square" in ranking
    assert "anova_f_score" in ranking
    assert "correlation_filtering" in ranking


def test_preprocessing_pipeline_builds_and_transforms_data():
    """The end-to-end preprocessing pipeline should transform the data."""
    df = sample_dataframe()
    pipeline = PreprocessingPipeline(target_column="Diabetes_binary")

    transformed = pipeline.fit_transform(df)

    assert "BMI_Category" in transformed.columns
    assert "Lifestyle_Risk_Score" in transformed.columns
    assert transformed.shape[0] == df.shape[0]
