"""Runner for SHAP explainability generation."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.explainability import (
    GlobalExplanationGenerator,
    LocalExplanationGenerator,
    ShapExplainer,
    ShapReportGenerator,
    ShapVisualizer,
)
from src.utils.logging import configure_logger


def main() -> None:
    """Run the explainability pipeline using saved model artifacts."""
    start_time = time.perf_counter()
    logger = configure_logger({"log_dir": "logs", "log_file": "explainability.log"})

    dataset_path = Path("data/processed/diabetes_binary_health_indicators_cleaned.csv")
    target_column = "Diabetes_binary"
    reports_dir = Path("reports/shap")
    explanations_dir = Path("artifacts/explanations")
    docs_dir = Path("docs")

    reports_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed dataset from %s", dataset_path)
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=[target_column])

    shap_explainer = ShapExplainer(logger=logger)
    shap_explainer.load_model()
    shap_explainer.load_feature_metadata()
    shap_explainer.create_explainer(X)

    visualizer = ShapVisualizer(output_dir=reports_dir, logger=logger)
    global_generator = GlobalExplanationGenerator(
        explainer=shap_explainer,
        visualizer=visualizer,
        output_dir=reports_dir,
        logger=logger,
    )
    local_generator = LocalExplanationGenerator(
        explainer=shap_explainer,
        visualizer=visualizer,
        output_dir=explanations_dir,
        logger=logger,
    )
    report_generator = ShapReportGenerator(output_dir=reports_dir)

    global_results = global_generator.generate(X, max_rows=500, top_n=20)
    global_report = report_generator.write_global_report(global_results)

    local_results = local_generator.explain_batch(X, n_patients=10)
    local_reports = [
        report_generator.write_local_report(result)
        for result in local_results
    ]
    batch_report = report_generator.write_batch_report(local_results)

    logger.info("Generated global report: %s", global_report)
    logger.info("Generated %s local reports", len(local_reports))
    logger.info("Generated batch report: %s", batch_report)
    logger.info("Explainability pipeline completed in %.2fs", time.perf_counter() - start_time)
    print("EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
