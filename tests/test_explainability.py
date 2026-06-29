"""Tests for SHAP explainability utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.explainability.global_explanations import GlobalExplanationGenerator
from src.explainability.local_explanations import LocalExplanationGenerator
from src.explainability.report_generator import ShapReportGenerator
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.visualization import ShapVisualizer


def sample_dataframe() -> pd.DataFrame:
    """Create a small dataset for explainability tests."""
    return pd.DataFrame(
        {
            "BMI": [18.0, 30.0, 20.0, 31.0, 22.0, 33.0, 24.0, 34.0, 26.0, 35.0],
            "Age": [4, 6, 5, 7, 4, 8, 5, 9, 6, 10],
            "HighBP": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "PhysActivity": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def sample_target() -> list[int]:
    """Create a binary target for explainability tests."""
    return [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def save_test_model(path: Path) -> None:
    """Save a fitted test model artifact."""
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(sample_dataframe(), sample_target())
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def test_shap_explainer_loads_model(tmp_path: Path) -> None:
    """The SHAP explainer should load a saved model artifact."""
    model_path = tmp_path / "models" / "best_model.joblib"
    save_test_model(model_path)

    explainer = ShapExplainer(model_path=model_path)
    model = explainer.load_model()

    assert model is not None
    assert explainer.loaded_model_path == model_path


def test_shap_generation_returns_expected_shape(tmp_path: Path) -> None:
    """SHAP generation should return one value per row and feature."""
    model_path = tmp_path / "models" / "best_model.joblib"
    save_test_model(model_path)
    X = sample_dataframe()

    explainer = ShapExplainer(model_path=model_path)
    explainer.load_model()
    explainer.create_explainer(X)
    shap_values = explainer.calculate_shap_values(X.head(3))

    assert shap_values.shape == (3, X.shape[1])


def test_global_report_generation(tmp_path: Path) -> None:
    """Global report generation should write markdown and feature importance."""
    model_path = tmp_path / "models" / "best_model.joblib"
    save_test_model(model_path)
    X = sample_dataframe()

    explainer = ShapExplainer(model_path=model_path)
    explainer.load_model()
    explainer.create_explainer(X)
    visualizer = ShapVisualizer(output_dir=tmp_path / "plots")
    generator = GlobalExplanationGenerator(explainer, visualizer, output_dir=tmp_path / "plots")
    global_results = generator.generate(X, max_rows=5, top_n=3)

    report_path = ShapReportGenerator(output_dir=tmp_path / "reports").write_global_report(global_results)

    assert report_path.exists()
    assert global_results["feature_importance"].exists()


def test_local_explanation_generation(tmp_path: Path) -> None:
    """Local explanations should include contributors and generated plots."""
    model_path = tmp_path / "models" / "best_model.joblib"
    save_test_model(model_path)
    X = sample_dataframe()

    explainer = ShapExplainer(model_path=model_path)
    explainer.load_model()
    explainer.create_explainer(X)
    visualizer = ShapVisualizer(output_dir=tmp_path / "plots")
    generator = LocalExplanationGenerator(explainer, visualizer, output_dir=tmp_path / "explanations")
    result = generator.explain_patient(X.iloc[0], patient_id="patient_01")
    report_path = ShapReportGenerator(output_dir=tmp_path / "reports").write_local_report(result)

    assert result["prediction"] in {"High Diabetes Risk", "Low Diabetes Risk"}
    assert "top_positive_contributors" in result
    assert "top_negative_contributors" in result
    assert result["plots"]["waterfall_plot"].exists()
    assert result["plots"]["force_plot"].exists()
    assert result["plots"]["decision_plot"].exists()
    assert report_path.exists()
