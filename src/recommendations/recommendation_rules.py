"""Rule engine for personalized health recommendations."""

from __future__ import annotations

import time
from typing import Any

from .patient_profile import PatientProfile, ShapFactor
from .recommendation_templates import (
    FEATURE_CATEGORY_MAP,
    POSITIVE_OBSERVATION_TEMPLATES,
    RECOMMENDATION_TEMPLATES,
)


class RecommendationRuleEngine:
    """Evaluate feature-driven recommendation rules."""

    def __init__(self, logger: Any | None = None) -> None:
        """Initialize the rule engine."""
        self.logger = logger

    def evaluate(
        self,
        profile: PatientProfile,
        top_factor_limit: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Evaluate recommendation categories for important risk factors.

        Args:
            profile: Patient profile with SHAP factors and indicators.
            top_factor_limit: Maximum number of risk factors to use.

        Returns:
            Mapping of recommendation category to recommendation entries.
        """
        start_time = time.perf_counter()
        recommendations: dict[str, list[dict[str, Any]]] = {}
        important_factors = profile.risk_factors[:top_factor_limit]

        for factor in important_factors:
            for category in FEATURE_CATEGORY_MAP.get(factor.feature, []):
                if not self._is_relevant(category, factor, profile):
                    continue
                recommendations.setdefault(category, [])
                entry = self._build_recommendation(category, factor, profile)
                if entry not in recommendations[category]:
                    recommendations[category].append(entry)

        if profile.prediction_probability >= 0.5:
            recommendations.setdefault("preventive_screening", [])
            screening = self._build_general_recommendation("preventive_screening")
            if screening not in recommendations["preventive_screening"]:
                recommendations["preventive_screening"].append(screening)

        self._log(
            "Evaluated recommendation rules for %s in %.2fs",
            profile.patient_id,
            time.perf_counter() - start_time,
        )
        return recommendations

    def positive_observations(self, profile: PatientProfile) -> list[str]:
        """Generate positive lifestyle observations from indicators and SHAP factors."""
        observations: list[str] = []
        for feature, template in POSITIVE_OBSERVATION_TEMPLATES.items():
            value = profile.get_indicator(feature)
            if self._is_positive_indicator(feature, value):
                observations.append(template)

        for factor in profile.positive_factors[:5]:
            message = f"{factor.feature} appears protective in this prediction."
            if message not in observations:
                observations.append(message)
        return observations or ["No clear protective lifestyle observations were identified from the available data."]

    def _build_recommendation(
        self,
        category: str,
        factor: ShapFactor,
        profile: PatientProfile,
    ) -> dict[str, Any]:
        """Build a structured recommendation entry."""
        template = RECOMMENDATION_TEMPLATES[category]
        return {
            "category": category,
            "title": template["title"],
            "recommendation": template["text"],
            "reason": f"{factor.feature} was an important contributor to the prediction.",
            "feature": factor.feature,
            "feature_value": profile.get_indicator(factor.feature),
            "shap_value": factor.shap_value,
        }

    def _build_general_recommendation(self, category: str) -> dict[str, Any]:
        """Build a recommendation entry not tied to one feature."""
        template = RECOMMENDATION_TEMPLATES[category]
        return {
            "category": category,
            "title": template["title"],
            "recommendation": template["text"],
            "reason": "The predicted risk probability supports preventive follow-up.",
            "feature": None,
            "feature_value": None,
            "shap_value": None,
        }

    def _is_relevant(self, category: str, factor: ShapFactor, profile: PatientProfile) -> bool:
        """Return whether a recommendation category applies to patient data."""
        value = profile.get_indicator(factor.feature)
        if category == "exercise" and factor.feature == "PhysActivity":
            return float(value or 0) == 0.0
        if category == "smoking" and factor.feature == "Smoker":
            return float(value or 0) == 1.0
        if category == "alcohol" and factor.feature == "HvyAlcoholConsump":
            return float(value or 0) == 1.0
        if category == "blood_pressure" and factor.feature == "HighBP":
            return float(value or 0) == 1.0
        if category == "diet" and factor.feature in {"Fruits", "Veggies"}:
            return float(value or 0) == 0.0
        return True

    def _is_positive_indicator(self, feature: str, value: Any) -> bool:
        """Detect clearly positive lifestyle indicators."""
        if value is None:
            return False
        numeric_value = float(value)
        if feature in {"PhysActivity", "Fruits", "Veggies"}:
            return numeric_value == 1.0
        if feature in {"Smoker", "HvyAlcoholConsump", "HighBP", "HighChol"}:
            return numeric_value == 0.0
        return False

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
