"""Patient profile models for recommendation generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShapFactor:
    """A SHAP factor used by the recommendation engine."""

    feature: str
    shap_value: float
    direction: str
    value: Any | None = None


@dataclass(frozen=True)
class PatientProfile:
    """Structured patient profile assembled from dataset and SHAP artifacts."""

    patient_id: str
    indicators: dict[str, Any]
    prediction: str
    prediction_probability: float
    risk_factors: list[ShapFactor] = field(default_factory=list)
    positive_factors: list[ShapFactor] = field(default_factory=list)

    def get_indicator(self, feature_name: str, default: Any | None = None) -> Any:
        """Return a patient health indicator value by name."""
        return self.indicators.get(feature_name, default)
