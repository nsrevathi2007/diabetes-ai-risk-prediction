"""Recommendation service wrapper for API endpoint use."""

from __future__ import annotations

from src.explainability.shap_explainer import ShapExplainer
from src.recommendations import PatientProfile, PatientProfileBuilder, RecommendationEngine


class RecommendationService:
    """Service responsible for building recommendations from patient features."""

    def __init__(
        self,
        shap_explainer: ShapExplainer,
        recommendation_engine: RecommendationEngine,
        prediction_service: "PredictionService",
    ) -> None:
        self.shap_explainer = shap_explainer
        self.recommendation_engine = recommendation_engine
        self.prediction_service = prediction_service
        self.profile_builder = PatientProfileBuilder(logger=None)

    def recommend(self, features: dict[str, object]) -> dict[str, object]:
        patient_profile = self._build_patient_profile(features)
        return self.recommendation_engine.generate(patient_profile)

    def _build_patient_profile(self, features: dict[str, object]) -> PatientProfile:
        prediction_result = self.prediction_service.predict(features)
        prediction = prediction_result["prediction"]
        prediction_probability = float(prediction_result["probability"])
        return PatientProfile(
            patient_id="patient_01",
            indicators=features,
            prediction=prediction,
            prediction_probability=prediction_probability,
            risk_factors=[],
            positive_factors=[],
        )
