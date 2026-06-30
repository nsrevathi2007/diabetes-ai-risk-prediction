"""Build patient profiles from existing dataset and SHAP artifacts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .patient_profile import PatientProfile, ShapFactor


class PatientProfileBuilder:
    """Assemble recommendation-ready profiles without regenerating SHAP values."""

    FACTOR_PATTERN = re.compile(
        r"^- (?P<feature>.+?) (?P<phrase>increased risk|reduced risk) "
        r"\(SHAP (?P<shap_value>[-+]?\d*\.?\d+)\)$"
    )

    def __init__(
        self,
        shap_json_dir: str | Path = "artifacts/explanations",
        shap_report_dir: str | Path = "reports/shap",
        min_abs_shap_value: float = 0.02,
        logger: Any | None = None,
    ) -> None:
        """Initialize profile assembly from existing SHAP artifacts."""
        self.shap_json_dir = Path(shap_json_dir)
        self.shap_report_dir = Path(shap_report_dir)
        self.min_abs_shap_value = min_abs_shap_value
        self.logger = logger

    def build_profiles(self, dataset: pd.DataFrame, patient_count: int = 10) -> list[PatientProfile]:
        """Build profiles for the same deterministic patient sample used by SHAP."""
        feature_data = dataset.drop(columns=["Diabetes_binary"], errors="ignore")
        sample = feature_data.sample(n=min(patient_count, len(feature_data)), random_state=42)
        profiles: list[PatientProfile] = []

        for number, (_, row) in enumerate(sample.iterrows(), start=1):
            patient_id = f"patient_{number:02d}"
            profiles.append(self.build_profile(patient_id, row))
        return profiles

    def build_profile(self, patient_id: str, indicators: pd.Series | dict[str, Any]) -> PatientProfile:
        """Build one patient profile from indicators and saved SHAP output."""
        start_time = time.perf_counter()
        indicator_dict = self._to_plain_dict(indicators)
        shap_data = self._load_shap_data(patient_id)
        profile = PatientProfile(
            patient_id=patient_id,
            indicators=indicator_dict,
            prediction=str(shap_data["prediction"]),
            prediction_probability=float(shap_data["prediction_probability"]),
            risk_factors=self._attach_values(shap_data["risk_factors"], indicator_dict),
            positive_factors=self._attach_values(shap_data["protective_factors"], indicator_dict),
        )
        self._log("Built recommendation profile for %s in %.2fs", patient_id, time.perf_counter() - start_time)
        return profile

    def _load_shap_data(self, patient_id: str) -> dict[str, Any]:
        """Load structured SHAP JSON when available, otherwise parse markdown."""
        json_path = self._find_json_artifact(patient_id)
        if json_path is not None:
            self._log("Loading structured SHAP JSON for %s from %s", patient_id, json_path)
            return self._load_json_artifact(json_path)

        markdown_path = self.shap_report_dir / f"{patient_id}_explanation.md"
        self._log("Structured SHAP JSON missing for %s; parsing %s", patient_id, markdown_path)
        return self._load_markdown_artifact(markdown_path)

    def _find_json_artifact(self, patient_id: str) -> Path | None:
        """Return the first known structured SHAP JSON artifact path."""
        candidates = [
            self.shap_json_dir / patient_id / "local_explanation.json",
            self.shap_json_dir / patient_id / "shap_values.json",
            self.shap_report_dir / f"{patient_id}_explanation.json",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _load_json_artifact(self, path: Path) -> dict[str, Any]:
        """Normalize a structured SHAP JSON artifact."""
        data = json.loads(path.read_text(encoding="utf-8"))
        positive = data.get("top_positive_contributors", data.get("risk_factors", []))
        negative = data.get("top_negative_contributors", data.get("protective_factors", []))
        return {
            "prediction": data["prediction"],
            "prediction_probability": data.get("risk_probability", data.get("prediction_probability")),
            "risk_factors": self._normalize_factors(positive, "increases_risk"),
            "protective_factors": self._normalize_factors(negative, "reduces_risk"),
        }

    def _load_markdown_artifact(self, path: Path) -> dict[str, Any]:
        """Parse the current Phase 6 markdown SHAP report format."""
        if not path.exists():
            raise FileNotFoundError(f"SHAP artifact not found for recommendations: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        prediction = self._section_value(lines, "Prediction")
        probability_text = self._section_value(lines, "Risk Probability")
        risk_factors = self._section_factors(lines, "Main contributing factors", "increases_risk")
        protective_factors = self._section_factors(lines, "Protective factors", "reduces_risk")

        return {
            "prediction": prediction,
            "prediction_probability": float(probability_text),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
        }

    def _section_value(self, lines: list[str], heading: str) -> str:
        """Read the first non-empty value under a markdown H2 heading."""
        start = self._heading_index(lines, heading)
        for line in lines[start + 1:]:
            if line.startswith("## "):
                break
            if line.strip():
                return line.strip()
        raise ValueError(f"Missing value for SHAP markdown section: {heading}")

    def _section_factors(
        self,
        lines: list[str],
        heading: str,
        direction: str,
    ) -> list[ShapFactor]:
        """Parse SHAP factor bullets under a markdown H2 heading."""
        start = self._heading_index(lines, heading)
        factors: list[ShapFactor] = []
        for line in lines[start + 1:]:
            if line.startswith("## "):
                break
            match = self.FACTOR_PATTERN.match(line.strip())
            if match is None:
                continue
            shap_value = float(match.group("shap_value"))
            if abs(shap_value) < self.min_abs_shap_value:
                continue
            factors.append(
                ShapFactor(
                    feature=match.group("feature"),
                    shap_value=shap_value,
                    direction=direction,
                )
            )
        return factors

    def _heading_index(self, lines: list[str], heading: str) -> int:
        """Find a markdown H2 heading index."""
        target = f"## {heading}"
        for index, line in enumerate(lines):
            if line.strip() == target:
                return index
        raise ValueError(f"Missing SHAP markdown section: {heading}")

    def _normalize_factors(self, factors: list[dict[str, Any]], direction: str) -> list[ShapFactor]:
        """Convert JSON factor dictionaries into dataclass factors."""
        normalized: list[ShapFactor] = []
        for factor in factors:
            shap_value = float(factor["shap_value"])
            if abs(shap_value) < self.min_abs_shap_value:
                continue
            normalized.append(
                ShapFactor(
                    feature=str(factor["feature"]),
                    shap_value=shap_value,
                    direction=direction,
                    value=factor.get("value"),
                )
            )
        return normalized

    def _attach_values(
        self,
        factors: list[ShapFactor],
        indicators: dict[str, Any],
    ) -> list[ShapFactor]:
        """Add patient indicator values when SHAP artifacts do not include them."""
        return [
            ShapFactor(
                feature=factor.feature,
                shap_value=factor.shap_value,
                direction=factor.direction,
                value=factor.value if factor.value is not None else indicators.get(factor.feature),
            )
            for factor in factors
        ]

    def _to_plain_dict(self, indicators: pd.Series | dict[str, Any]) -> dict[str, Any]:
        """Convert pandas scalar values to JSON-friendly Python values."""
        raw = indicators.to_dict() if isinstance(indicators, pd.Series) else dict(indicators)
        return {
            key: value.item() if hasattr(value, "item") else value
            for key, value in raw.items()
        }

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
