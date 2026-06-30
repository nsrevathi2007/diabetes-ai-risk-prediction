"""Pydantic schemas for structured recommendation payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Priority = Literal["High", "Medium", "Low"]


class ShapFactorSchema(BaseModel):
    """A meaningful SHAP factor used by the recommendation engine."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    shap_value: float
    direction: Literal["increases_risk", "reduces_risk"]
    value: Any | None = None


class RecommendationEntry(BaseModel):
    """One educational recommendation tied to a risk signal."""

    model_config = ConfigDict(extra="forbid")

    category: str
    title: str
    recommendation: str
    reason: str
    priority: Priority
    feature: str | None = None
    feature_value: Any | None = None
    shap_value: float | None = None


class PriorityAction(BaseModel):
    """A concise action item for frontend and LLM consumption."""

    model_config = ConfigDict(extra="forbid")

    category: str
    title: str
    reason: str
    priority: Priority


class RecommendationPayload(BaseModel):
    """LLM-ready recommendation output for one patient."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    prediction: str
    risk_level: str
    prediction_probability: float = Field(ge=0.0, le=1.0)
    priority_actions: list[PriorityAction]
    risk_factors: list[ShapFactorSchema]
    protective_factors: list[ShapFactorSchema]
    recommendations: dict[str, list[RecommendationEntry]]
    positive_observations: list[str]
    preventive_suggestions: list[str]
    disclaimer: str
