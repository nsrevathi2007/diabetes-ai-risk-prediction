"""Markdown report generation for SHAP explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ShapReportGenerator:
    """Create human-readable SHAP markdown reports."""

    def __init__(self, output_dir: str | Path = "reports/shap") -> None:
        """Initialize report generation."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_global_report(
        self,
        global_results: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Path:
        """Write the global explainability report."""
        path = Path(output_path) if output_path else self.output_dir / "global_explainability_report.md"
        top_features = global_results.get("top_features", [])
        lines = [
            "# Global SHAP Explainability Report",
            "",
            "## Model Behavior",
            "Global SHAP analysis estimates how each feature contributes to diabetes risk predictions across the sampled dataset.",
            "",
            "## Top 20 Features",
            "| Rank | Feature | Mean Absolute SHAP |",
            "| ---: | --- | ---: |",
        ]
        for rank, feature in enumerate(top_features, start=1):
            lines.append(f"| {rank} | {feature['feature']} | {feature['mean_abs_shap']:.6f} |")

        lines.extend(
            [
                "",
                "## Generated Artifacts",
                f"- Summary plot: `{global_results.get('summary_plot')}`",
                f"- Bar plot: `{global_results.get('bar_plot')}`",
                f"- Feature importance: `{global_results.get('feature_importance')}`",
            ]
        )
        return self._write(path, lines)

    def write_local_report(
        self,
        local_result: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Path:
        """Write a local patient explanation report."""
        path = (
            Path(output_path)
            if output_path
            else self.output_dir / f"{local_result['patient_id']}_explanation.md"
        )
        positive = local_result.get("top_positive_contributors", [])
        negative = local_result.get("top_negative_contributors", [])

        lines = [
            f"# Patient Explanation: {local_result['patient_id']}",
            "",
            "## Prediction",
            str(local_result["prediction"]),
            "",
            "## Main contributing factors",
            *self._factor_lines(positive, "increased risk"),
            "",
            "## Protective factors",
            *self._factor_lines(negative, "reduced risk"),
            "",
            "## Confidence",
            str(local_result["confidence"]),
            "",
            "## Risk Probability",
            f"{local_result['risk_probability']:.4f}",
            "",
            "## Plots",
            *[f"- {name}: `{path}`" for name, path in local_result.get("plots", {}).items()],
        ]
        return self._write(path, lines)

    def write_batch_report(
        self,
        local_results: list[dict[str, Any]],
        output_path: str | Path | None = None,
    ) -> Path:
        """Write a summary report for batch local explanations."""
        path = Path(output_path) if output_path else self.output_dir / "batch_explainability_report.md"
        rows = [
            {
                "patient_id": result["patient_id"],
                "prediction": result["prediction"],
                "risk_probability": result["risk_probability"],
                "confidence": result["confidence"],
            }
            for result in local_results
        ]
        table_lines = [
            "| Patient | Prediction | Risk Probability | Confidence |",
            "| --- | --- | ---: | --- |",
        ]
        for row in rows:
            table_lines.append(
                f"| {row['patient_id']} | {row['prediction']} | {row['risk_probability']:.4f} | {row['confidence']} |"
            )
        lines = [
            "# Batch SHAP Explainability Report",
            "",
            "## Patient Predictions",
            *(table_lines if rows else ["No local explanations generated."]),
        ]
        return self._write(path, lines)

    def _factor_lines(self, factors: list[dict[str, Any]], phrase: str) -> list[str]:
        """Create human-readable factor bullets."""
        if not factors:
            return ["- No strong factors identified."]
        return [
            f"- {factor['feature']} {phrase} (SHAP {factor['shap_value']:.4f})"
            for factor in factors
        ]

    def _write(self, path: Path, lines: list[str]) -> Path:
        """Write markdown lines to a path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
