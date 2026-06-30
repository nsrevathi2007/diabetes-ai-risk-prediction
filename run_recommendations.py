"""Runner for personalized recommendation generation."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.explainability.shap_explainer import ShapExplainer
from src.recommendations import (
    PatientProfileBuilder,
    RecommendationEngine,
    RecommendationReportGenerator,
)
from src.utils.logging import configure_logger


def main() -> None:
    """Generate structured recommendation artifacts for SHAP patient cohort."""
    start_time = time.perf_counter()
    logger = configure_logger({"log_dir": "logs", "log_file": "recommendations.log"})

    dataset_path = Path("data/processed/diabetes_binary_health_indicators_cleaned.csv")
    config_path = Path("configs/recommendation_config.yaml")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading optimized model artifact for recommendation context")
    ShapExplainer(logger=logger).load_model()

    logger.info("Loading processed dataset from %s", dataset_path)
    dataset = pd.read_csv(dataset_path)

    engine = RecommendationEngine.from_config_file(config_path, logger=logger)
    settings = engine.config.get("recommendation_settings", {})
    profile_builder = PatientProfileBuilder(
        min_abs_shap_value=float(settings.get("min_abs_shap_value", 0.02)),
        logger=logger,
    )
    report_generator = RecommendationReportGenerator(logger=logger)

    profiles = profile_builder.build_profiles(dataset, patient_count=10)
    payloads = []
    for profile in profiles:
        logger.info("Recommendation generation started for %s", profile.patient_id)
        payload = engine.generate(profile)
        report_generator.save_json(payload)
        report_generator.write_markdown(payload)
        payloads.append(payload)

    docs_path = write_engine_report(payloads, docs_dir / "recommendation_engine_report.md")
    logger.info("Generated recommendation engine report: %s", docs_path)
    logger.info("Recommendation pipeline completed in %.2fs", time.perf_counter() - start_time)
    print("RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY")


def write_engine_report(payloads: list[dict[str, object]], output_path: Path) -> Path:
    """Write a compact Phase 7 recommendation engine summary."""
    risk_counts: dict[str, int] = {}
    for payload in payloads:
        risk_level = str(payload["risk_level"])
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

    risk_lines = [f"- {risk}: {count}" for risk, count in sorted(risk_counts.items())]
    patient_lines = [
        (
            f"| {payload['patient_id']} | {payload['risk_level']} | "
            f"{float(payload['prediction_probability']):.4f} | {len(payload['priority_actions'])} |"
        )
        for payload in payloads
    ]

    lines = [
        "# Recommendation Engine Report",
        "",
        "## Summary",
        "The Phase 7 recommendation engine generated informational, non-diagnostic recommendations from existing model predictions, saved SHAP explanations, and patient health indicators.",
        "",
        "## SHAP Artifact Strategy",
        "Structured SHAP JSON artifacts are preferred when present. Existing SHAP markdown reports are parsed as a fallback, so SHAP values are not regenerated.",
        "",
        "## Risk Level Counts",
        *(risk_lines or ["- No patients processed."]),
        "",
        "## Generated Patients",
        "| Patient | Risk Level | Probability | Priority Actions |",
        "| --- | --- | ---: | ---: |",
        *(patient_lines or ["| None | N/A | 0.0000 | 0 |"]),
        "",
        "## Disclaimer",
        (
            "This information is generated for educational purposes only and should not be considered medical advice. "
            "Please consult a qualified healthcare professional for diagnosis or treatment."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()
