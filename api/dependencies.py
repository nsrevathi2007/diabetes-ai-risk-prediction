"""Dependency injection helpers for API routes."""

from __future__ import annotations

from fastapi import Depends, Request

from .config import APISettings
from .services import (
    AnalysisService,
    ExplanationService,
    PredictionService,
    RecommendationService,
)


def get_settings(request: Request) -> APISettings:
    """Return runtime settings stored on the application state."""
    return request.app.state.settings


def get_prediction_service(request: Request) -> PredictionService:
    """Return the prediction service from application state."""
    return request.app.state.prediction_service


def get_recommendation_service(request: Request) -> RecommendationService:
    """Return the recommendation service from application state."""
    return request.app.state.recommendation_service


def get_explanation_service(request: Request) -> ExplanationService:
    """Return the explanation service from application state."""
    return request.app.state.explanation_service


def get_analysis_service(request: Request) -> AnalysisService:
    """Return the full analysis service from application state."""
    return request.app.state.analysis_service
