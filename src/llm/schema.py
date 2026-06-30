"""Pydantic schemas for LLM explanation outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Priority = Literal["High", "Medium", "Low"]


class ExplainedAction(BaseModel):
    """A source-backed priority action explanation."""

    model_config = ConfigDict(extra="forbid")

    title: str
    priority: Priority
    explanation: str


class LLMExplanationPayload(BaseModel):
    """Structured natural-language explanation generated from recommendations."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    risk_level: str
    summary: str
    why_this_prediction: str
    positive_observations: list[str]
    priority_actions: list[ExplainedAction]
    recommendation_explanation: str
    next_steps: str
    disclaimer: str
    generated_at: datetime
    generation_mode: Literal["llm", "template"]
    model: str
    source_recommendation_count: int = Field(ge=0)


class LLMGenerationResult(BaseModel):
    """Internal generation result with metadata."""

    model_config = ConfigDict(extra="forbid")

    payload: LLMExplanationPayload
    raw_response: str | None = None
    fallback_reason: str | None = None
