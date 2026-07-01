"""Full analysis orchestration service for API endpoint use."""

from __future__ import annotations

from .prediction_service import PredictionService
from .recommendation_service import RecommendationService
from .explanation_service import ExplanationService
from src.llm import LLMExplanationGenerator


class AnalysisService:
    """Service that composes prediction, recommendation, SHAP, and LLM outputs."""

    def __init__(
        self,
        prediction_service: PredictionService,
        recommendation_service: RecommendationService,
        explanation_service: ExplanationService,
        llm_generator: LLMExplanationGenerator,
    ) -> None:
        self.prediction_service = prediction_service
        self.recommendation_service = recommendation_service
        self.explanation_service = explanation_service
        self.llm_generator = llm_generator

    def full_analysis(self, features: dict[str, object]) -> dict[str, object]:
        prediction = self.prediction_service.predict(features)
        recommendation = self.recommendation_service.recommend(features)
        explanation = self.explanation_service.explain(features)
        llm_result = self.llm_generator.generate(recommendation)
        return {
            "prediction": prediction,
            "recommendation": recommendation,
            "explanation": explanation,
            "llm_explanation": llm_result.payload.model_dump(mode="json"),
            "fallback_reason": llm_result.fallback_reason,
        }
