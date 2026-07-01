"""Pydantic response models for API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RootResponse(BaseModel):
    """Root API metadata."""

    api_name: str
    version: str
    status: str


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    llm_available: bool
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    training_date: str
    metrics: dict[str, float]
    feature_count: int
    model_version: str


class PredictionResponse(BaseModel):
    """Prediction response."""

    prediction: str
    probability: float
    risk_level: str


class ShapFactorResponse(BaseModel):
    """SHAP factor response item."""

    feature: str
    value: Any
    shap_value: float


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""

    prediction: str
    probability: float
    risk_level: str
    top_risk_factors: list[ShapFactorResponse]
    top_protective_factors: list[ShapFactorResponse]


class RecommendationResponse(BaseModel):
    """Structured recommendation response."""

    model_config = ConfigDict(extra="allow")


class FullAnalysisResponse(BaseModel):
    """Combined full analysis response."""

    prediction: PredictionResponse
    recommendation: dict[str, Any]
    explanation: dict[str, Any]
    llm_explanation: dict[str, Any]
    fallback_reason: str | None = None
