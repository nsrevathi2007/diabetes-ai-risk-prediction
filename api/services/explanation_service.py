"""Explanation service wrapper for API endpoint use."""

from __future__ import annotations

import pandas as pd

from src.explainability.local_explanations import LocalExplanationGenerator
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.visualization import ShapVisualizer
from src.recommendations import PatientProfile, PatientProfileBuilder, RecommendationEngine


class ExplanationService:
    """Service responsible for SHAP explanation generation."""

    def __init__(self, shap_explainer: ShapExplainer, recommendation_engine: RecommendationEngine) -> None:
        self.shap_explainer = shap_explainer
        visualizer = ShapVisualizer(output_dir="artifacts/explanations")
        self.local_generator = LocalExplanationGenerator(shap_explainer, visualizer)
        self.recommendation_engine = recommendation_engine
        self.profile_builder = PatientProfileBuilder(logger=None)

    def explain(self, features: dict[str, object]) -> dict[str, object]:
        explanation = self.local_generator.explain_patient(features, patient_id="patient_01", top_n=5)
        return {
            "prediction": explanation["prediction"],
            "probability": float(explanation["risk_probability"]),
            "risk_level": self.recommendation_engine.categorize_risk(float(explanation["risk_probability"])),
            "top_risk_factors": explanation["top_positive_contributors"],
            "top_protective_factors": explanation["top_negative_contributors"],
        }
