"""Personalized recommendation engine for diabetes risk support."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from .patient_profile import PatientProfile
from .recommendation_rules import RecommendationRuleEngine


class RecommendationEngine:
    """Generate structured educational recommendations for a patient."""

    DEFAULT_DISCLAIMER = (
        "This information is generated for educational purposes only and should not be considered medical advice. "
        "Please consult a qualified healthcare professional for diagnosis or treatment."
    )

    def __init__(
        self,
        config: dict[str, Any],
        rule_engine: RecommendationRuleEngine | None = None,
        logger: Any | None = None,
    ) -> None:
        """Initialize the recommendation engine.

        Args:
            config: Recommendation configuration loaded from YAML.
            rule_engine: Optional rule engine implementation.
            logger: Optional project logger.
        """
        self.config = config
        self.rule_engine = rule_engine or RecommendationRuleEngine(logger=logger)
        self.logger = logger

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "configs/recommendation_config.yaml",
        logger: Any | None = None,
    ) -> "RecommendationEngine":
        """Create an engine from a YAML configuration file."""
        path = Path(config_path)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(config=config, logger=logger)

    def categorize_risk(self, probability: float) -> str:
        """Categorize risk using configured probability thresholds."""
        thresholds = self.config["risk_thresholds"]
        if probability < float(thresholds["low"]):
            return "Low Risk"
        if probability < float(thresholds["moderate"]):
            return "Moderate Risk"
        if probability < float(thresholds["high"]):
            return "High Risk"
        return "Very High Risk"

    def generate(self, profile: PatientProfile) -> dict[str, Any]:
        """Generate a structured recommendation payload.

        Args:
            profile: Patient profile containing prediction and SHAP factors.

        Returns:
            JSON-serializable recommendation payload.
        """
        start_time = time.perf_counter()
        settings = self.config.get("recommendation_settings", {})
        top_factor_limit = int(settings.get("top_factor_limit", 5))
        priority_limit = int(settings.get("priority_action_limit", 5))
        recommendations = self.rule_engine.evaluate(profile, top_factor_limit=top_factor_limit)
        risk_level = self.categorize_risk(profile.prediction_probability)
        priority_actions = self._build_priority_actions(recommendations, priority_limit)
        positive_observations = self.rule_engine.positive_observations(profile)

        payload = {
            "patient_id": profile.patient_id,
            "prediction": profile.prediction,
            "risk_level": risk_level,
            "prediction_probability": profile.prediction_probability,
            "priority_actions": priority_actions,
            "recommendations": recommendations,
            "positive_factors": [factor.__dict__ for factor in profile.positive_factors],
            "risk_factors": [factor.__dict__ for factor in profile.risk_factors],
            "positive_lifestyle_observations": positive_observations,
            "preventive_suggestions": self._preventive_suggestions(risk_level),
            "disclaimer": self.config.get("disclaimer", self.DEFAULT_DISCLAIMER),
        }
        self._log(
            "Generated recommendations for %s in %.2fs",
            profile.patient_id,
            time.perf_counter() - start_time,
        )
        return payload

    def _build_priority_actions(
        self,
        recommendations: dict[str, list[dict[str, Any]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Flatten recommendations into a concise priority action list."""
        actions: list[dict[str, Any]] = []
        for entries in recommendations.values():
            for entry in entries:
                actions.append(
                    {
                        "category": entry["category"],
                        "title": entry["title"],
                        "reason": entry["reason"],
                    }
                )
        return actions[:limit]

    def _preventive_suggestions(self, risk_level: str) -> list[str]:
        """Return non-diagnostic preventive suggestions by risk level."""
        if risk_level in {"High Risk", "Very High Risk"}:
            return [
                "Consider discussing diabetes risk screening with a qualified healthcare professional.",
                "Review blood pressure, cholesterol, activity, nutrition, and weight-related risk factors.",
            ]
        if risk_level == "Moderate Risk":
            return [
                "Maintain routine preventive checkups and monitor changes in health indicators.",
                "Focus on sustainable lifestyle habits that support metabolic health.",
            ]
        return [
            "Continue protective lifestyle habits and routine preventive care.",
        ]

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
