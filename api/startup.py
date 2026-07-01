"""Application startup initialization for API dependencies."""

from __future__ import annotations

from fastapi import FastAPI
from src.explainability.shap_explainer import ShapExplainer
from src.llm import LLMExplanationGenerator
from src.recommendations import RecommendationEngine
from src.utils.logging import configure_logger

from .config import APISettings
from .services import (
    AnalysisService,
    ExplanationService,
    PredictionService,
    RecommendationService,
)


def initialize_app_state(app: FastAPI, settings: APISettings) -> None:
    """Initialize the shared application state and dependencies."""
    logger = configure_logger({"log_dir": "logs", "log_file": "api.log"})
    logger.info("Initializing API application state")

    shap_explainer = ShapExplainer(model_path=settings.model_path, logger=logger)
    shap_explainer.load_model()

    recommendation_engine = RecommendationEngine.from_config_file(
        settings.recommendation_config_path,
        logger=logger,
    )

    llm_generator = LLMExplanationGenerator(logger=logger)

    app.state.settings = settings
    app.state.logger = logger
    app.state.shap_explainer = shap_explainer
    app.state.recommendation_engine = recommendation_engine
    app.state.llm_generator = llm_generator
    app.state.prediction_service = PredictionService(shap_explainer, recommendation_engine)
    app.state.recommendation_service = RecommendationService(
        shap_explainer,
        recommendation_engine,
        app.state.prediction_service,
    )
    app.state.explanation_service = ExplanationService(shap_explainer, recommendation_engine)
    app.state.analysis_service = AnalysisService(
        app.state.prediction_service,
        app.state.recommendation_service,
        app.state.explanation_service,
        llm_generator,
    )
