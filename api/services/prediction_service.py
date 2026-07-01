"""Prediction service wrapper for API endpoint use."""

from __future__ import annotations

import pandas as pd

from src.explainability.shap_explainer import ShapExplainer
from src.recommendations.recommendation_engine import RecommendationEngine


class PredictionService:
    """Service responsible for model risk prediction."""

    def __init__(self, shap_explainer: ShapExplainer, recommendation_engine: RecommendationEngine) -> None:
        self.shap_explainer = shap_explainer
        self.recommendation_engine = recommendation_engine

    def predict(self, features: dict[str, object]) -> dict[str, object]:
        patient_df = pd.DataFrame([features])
        probability = float(self.shap_explainer.predict_risk(patient_df)[0])
        risk_level = self.recommendation_engine.categorize_risk(probability)
        prediction_label = "High Diabetes Risk" if probability >= 0.5 else "Low Diabetes Risk"
        return {
            "prediction": prediction_label,
            "probability": probability,
            "risk_level": risk_level,
        }
