"""Deterministic offline explanation generator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schema import LLMExplanationPayload


class TemplateExplanationGenerator:
    """Generate natural-language explanations without an LLM provider."""

    def generate(
        self,
        recommendation_payload: dict[str, Any],
        model_name: str = "template-mode",
    ) -> LLMExplanationPayload:
        """Create a deterministic explanation from authoritative recommendations."""
        actions = [
            {
                "title": action["title"],
                "priority": action["priority"],
                "explanation": action["reason"],
            }
            for action in recommendation_payload.get("priority_actions", [])
        ]
        risk_factors = recommendation_payload.get("risk_factors", [])[:5]
        protective_factors = recommendation_payload.get("protective_factors", [])[:5]
        recommendations = recommendation_payload.get("recommendations", {})
        recommendation_count = sum(len(entries) for entries in recommendations.values())

        return LLMExplanationPayload.model_validate(
            {
                "patient_id": recommendation_payload["patient_id"],
                "risk_level": recommendation_payload["risk_level"],
                "summary": self._summary(recommendation_payload),
                "why_this_prediction": self._why_prediction(risk_factors, protective_factors),
                "positive_observations": recommendation_payload.get("positive_observations", []),
                "priority_actions": actions,
                "recommendation_explanation": self._recommendation_explanation(recommendations),
                "next_steps": self._next_steps(recommendation_payload),
                "disclaimer": recommendation_payload["disclaimer"],
                "generated_at": datetime.now(timezone.utc),
                "generation_mode": "template",
                "model": model_name,
                "source_recommendation_count": recommendation_count,
            }
        )

    def _summary(self, payload: dict[str, Any]) -> str:
        """Build a concise patient summary."""
        probability = float(payload["prediction_probability"])
        return (
            f"The model estimated this patient's diabetes risk as {payload['risk_level']} "
            f"with a prediction probability of {probability:.4f}. This explanation summarizes "
            "the existing structured recommendation output."
        )

    def _why_prediction(
        self,
        risk_factors: list[dict[str, Any]],
        protective_factors: list[dict[str, Any]],
    ) -> str:
        """Explain SHAP factors without adding new recommendations."""
        risk_names = ", ".join(factor["feature"] for factor in risk_factors) or "no strong increasing factors"
        protective_names = ", ".join(factor["feature"] for factor in protective_factors) or "no strong protective factors"
        return (
            f"The main SHAP factors increasing the prediction were {risk_names}. "
            f"Protective SHAP factors reducing the prediction were {protective_names}."
        )

    def _recommendation_explanation(self, recommendations: dict[str, list[dict[str, Any]]]) -> str:
        """Summarize existing recommendation categories."""
        categories = [category.replace("_", " ") for category, entries in recommendations.items() if entries]
        if not categories:
            return "No specific recommendation categories were triggered by the meaningful SHAP factors."
        return (
            "The structured recommendation engine produced guidance in these categories: "
            f"{', '.join(categories)}. These items are explanatory and educational only."
        )

    def _next_steps(self, payload: dict[str, Any]) -> str:
        """Convert preventive suggestions to prose."""
        suggestions = payload.get("preventive_suggestions", [])
        if not suggestions:
            return "Continue routine wellness habits and review risk-related changes over time."
        return " ".join(str(suggestion) for suggestion in suggestions)
