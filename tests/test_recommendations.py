"""Tests for personalized recommendation generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.recommendations import (
    PatientProfileBuilder,
    RecommendationEngine,
    RecommendationPayload,
    RecommendationReportGenerator,
    RecommendationRuleEngine,
)
from src.recommendations.patient_profile import PatientProfile, ShapFactor


DISCLAIMER = (
    "This information is generated for educational purposes only and should not be considered medical advice. "
    "Please consult a qualified healthcare professional for diagnosis or treatment."
)


def sample_config() -> dict[str, object]:
    """Return a compact recommendation config for tests."""
    return {
        "risk_thresholds": {"low": 0.25, "moderate": 0.50, "high": 0.75},
        "recommendation_settings": {
            "top_factor_limit": 5,
            "priority_action_limit": 5,
            "min_abs_shap_value": 0.02,
            "priority_thresholds": {
                "high_abs_shap": 0.30,
                "medium_abs_shap": 0.10,
                "high_probability": 0.75,
                "medium_probability": 0.50,
            },
        },
        "disclaimer": DISCLAIMER,
    }


def sample_profile() -> PatientProfile:
    """Create a patient profile with meaningful SHAP factors."""
    return PatientProfile(
        patient_id="patient_01",
        indicators={
            "BMI": 33.0,
            "HighBP": 1.0,
            "PhysActivity": 0.0,
            "Fruits": 1.0,
            "Veggies": 1.0,
            "Smoker": 0.0,
            "HvyAlcoholConsump": 0.0,
            "HighChol": 0.0,
        },
        prediction="High Diabetes Risk",
        prediction_probability=0.82,
        risk_factors=[
            ShapFactor("BMI", 0.42, "increases_risk", 33.0),
            ShapFactor("HighBP", 0.28, "increases_risk", 1.0),
            ShapFactor("Education", 0.01, "increases_risk", 4.0),
        ],
        positive_factors=[
            ShapFactor("PhysActivity", -0.18, "reduces_risk", 1.0),
        ],
    )


def test_risk_categorization_uses_config() -> None:
    """Risk categorization should use YAML-style thresholds."""
    engine = RecommendationEngine(config=sample_config())

    assert engine.categorize_risk(0.10) == "Low Risk"
    assert engine.categorize_risk(0.30) == "Moderate Risk"
    assert engine.categorize_risk(0.60) == "High Risk"
    assert engine.categorize_risk(0.90) == "Very High Risk"


def test_rule_engine_generates_prioritized_recommendations() -> None:
    """Rule execution should create relevant recommendations with priorities."""
    settings = sample_config()["recommendation_settings"]
    rule_engine = RecommendationRuleEngine(priority_thresholds=settings["priority_thresholds"])

    recommendations = rule_engine.evaluate(sample_profile(), top_factor_limit=2)

    assert "weight_management" in recommendations
    assert recommendations["weight_management"][0]["priority"] == "High"
    assert recommendations["blood_pressure"][0]["priority"] == "Medium"
    assert "preventive_screening" in recommendations


def test_engine_returns_llm_ready_schema() -> None:
    """Recommendation output should validate against the Pydantic schema."""
    engine = RecommendationEngine(config=sample_config())

    payload = engine.generate(sample_profile())
    validated = RecommendationPayload.model_validate(payload)

    assert validated.patient_id == "patient_01"
    assert validated.priority_actions[0].priority == "High"
    assert "protective_factors" in payload
    assert "positive_observations" in payload
    assert "positive_lifestyle_observations" not in payload
    assert all(abs(factor["shap_value"]) >= 0.02 for factor in payload["risk_factors"])


def test_profile_builder_parses_markdown_fallback(tmp_path: Path) -> None:
    """Profile builder should parse markdown when JSON SHAP artifacts are absent."""
    report_dir = tmp_path / "reports" / "shap"
    report_dir.mkdir(parents=True)
    (report_dir / "patient_01_explanation.md").write_text(
        "\n".join(
            [
                "# Patient Explanation: patient_01",
                "",
                "## Prediction",
                "Low Diabetes Risk",
                "",
                "## Main contributing factors",
                "- BMI increased risk (SHAP 0.4200)",
                "- Income increased risk (SHAP 0.0100)",
                "",
                "## Protective factors",
                "- HighBP reduced risk (SHAP -0.3000)",
                "",
                "## Confidence",
                "High",
                "",
                "## Risk Probability",
                "0.2182",
            ]
        ),
        encoding="utf-8",
    )
    builder = PatientProfileBuilder(
        shap_json_dir=tmp_path / "missing",
        shap_report_dir=report_dir,
        min_abs_shap_value=0.02,
    )

    profile = builder.build_profile("patient_01", {"BMI": 31.0, "HighBP": 0.0})

    assert profile.prediction_probability == 0.2182
    assert [factor.feature for factor in profile.risk_factors] == ["BMI"]
    assert profile.risk_factors[0].value == 31.0
    assert profile.positive_factors[0].direction == "reduces_risk"


def test_report_generator_writes_json_and_markdown(tmp_path: Path) -> None:
    """JSON and markdown reports should be generated with disclaimer and priorities."""
    payload = RecommendationEngine(config=sample_config()).generate(sample_profile())
    generator = RecommendationReportGenerator(
        json_dir=tmp_path / "artifacts" / "recommendations",
        report_dir=tmp_path / "reports" / "recommendations",
    )

    json_path = generator.save_json(payload)
    markdown_path = generator.write_markdown(payload)

    assert json_path.exists()
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "[High]" in markdown
    assert DISCLAIMER in markdown


def test_profile_builder_prefers_json_artifact(tmp_path: Path) -> None:
    """Structured SHAP JSON should be loaded before markdown fallback."""
    patient_dir = tmp_path / "artifacts" / "explanations" / "patient_01"
    patient_dir.mkdir(parents=True)
    (patient_dir / "local_explanation.json").write_text(
        """
        {
          "prediction": "High Diabetes Risk",
          "risk_probability": 0.81,
          "top_positive_contributors": [
            {"feature": "BMI", "value": 35.0, "shap_value": 0.51}
          ],
          "top_negative_contributors": [
            {"feature": "PhysActivity", "value": 1.0, "shap_value": -0.21}
          ]
        }
        """,
        encoding="utf-8",
    )
    builder = PatientProfileBuilder(
        shap_json_dir=tmp_path / "artifacts" / "explanations",
        shap_report_dir=tmp_path / "reports" / "shap",
    )

    profile = builder.build_profile("patient_01", pd.Series({"BMI": 30.0, "PhysActivity": 1.0}))

    assert profile.prediction_probability == 0.81
    assert profile.risk_factors[0].value == 35.0
