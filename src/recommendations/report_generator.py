"""Recommendation JSON and markdown report generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RecommendationReportGenerator:
    """Persist recommendation outputs as JSON and markdown."""

    def __init__(
        self,
        json_dir: str | Path = "artifacts/recommendations",
        report_dir: str | Path = "reports/recommendations",
        logger: Any | None = None,
    ) -> None:
        """Initialize report generation."""
        self.json_dir = Path(json_dir)
        self.report_dir = Path(report_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def save_json(self, payload: dict[str, Any]) -> Path:
        """Save a recommendation payload as JSON."""
        start_time = time.perf_counter()
        path = self.json_dir / f"{payload['patient_id']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log("Generated recommendation JSON at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def write_markdown(self, payload: dict[str, Any]) -> Path:
        """Write a patient recommendation markdown report."""
        start_time = time.perf_counter()
        path = self.report_dir / f"{payload['patient_id']}.md"
        lines = [
            f"# Personalized Health Recommendations: {payload['patient_id']}",
            "",
            "## Prediction",
            str(payload["prediction"]),
            "",
            "## Risk Level",
            str(payload["risk_level"]),
            "",
            "## Prediction Probability",
            f"{payload['prediction_probability']:.4f}",
            "",
            "## Top SHAP Factors Increasing Risk",
            *self._factor_lines(payload.get("risk_factors", [])),
            "",
            "## Top Protective SHAP Factors",
            *self._factor_lines(payload.get("protective_factors", [])),
            "",
            "## Priority Actions",
            *self._priority_lines(payload.get("priority_actions", [])),
            "",
            "## Personalized Recommendations",
            *self._recommendation_lines(payload.get("recommendations", {})),
            "",
            "## Positive Lifestyle Observations",
            *[f"- {item}" for item in payload.get("positive_observations", [])],
            "",
            "## Preventive Suggestions",
            *[f"- {item}" for item in payload.get("preventive_suggestions", [])],
            "",
            "## Disclaimer",
            str(payload["disclaimer"]),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        self._log("Created recommendation report at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def _factor_lines(self, factors: list[dict[str, Any]]) -> list[str]:
        """Format SHAP factors for markdown."""
        if not factors:
            return ["- No important factors identified."]
        return [f"- {factor['feature']}: SHAP {float(factor['shap_value']):.4f}" for factor in factors]

    def _priority_lines(self, actions: list[dict[str, Any]]) -> list[str]:
        """Format priority actions for markdown."""
        if not actions:
            return ["- Continue general wellness habits and preventive care."]
        return [
            f"- [{action['priority']}] {action['title']}: {action['reason']}"
            for action in actions
        ]

    def _recommendation_lines(self, recommendations: dict[str, list[dict[str, Any]]]) -> list[str]:
        """Format grouped recommendations for markdown."""
        if not recommendations:
            return ["- No specific lifestyle recommendation was triggered by the top SHAP factors."]

        lines: list[str] = []
        for category, entries in recommendations.items():
            lines.append(f"### {category.replace('_', ' ').title()}")
            for entry in entries:
                lines.append(f"- [{entry['priority']}] {entry['title']}: {entry['recommendation']}")
        return lines

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
