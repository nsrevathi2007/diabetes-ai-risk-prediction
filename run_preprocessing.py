"""Runner for the preprocessing pipeline stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.utils.logging import configure_logger


def main() -> None:
    """Run the preprocessing workflow end to end."""
    logger = configure_logger({"log_dir": "logs", "log_file": "preprocessing.log"})
    dataset_path = Path("data/processed/diabetes_binary_health_indicators_cleaned.csv")

    df = pd.read_csv(dataset_path)
    pipeline = build_preprocessing_pipeline({"target_column": "Diabetes_binary", "logger": logger})

    transformed_df = pipeline.fit_transform(df)
    artifact_paths = pipeline.save_artifacts("artifacts")

    report_dir = Path("reports/statistics")
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "feature_types.json").write_text(
        str(pipeline.feature_types),
        encoding="utf-8",
    )
    (report_dir / "feature_selection.json").write_text(
        str(pipeline.feature_selection_report),
        encoding="utf-8",
    )

    markdown_report = Path("docs/preprocessing_report.md")
    markdown_report.write_text(
        "\n".join(
            [
                "# Preprocessing Report",
                "",
                "## Cleaning Summary",
                f"- Duplicate rows: {pipeline.cleaning_report.get('duplicate_rows', 0)}",
                f"- Issues found: {pipeline.cleaning_report.get('issue_count', 0)}",
                "",
                "## Engineered Features",
                f"- {', '.join(pipeline.engineered_features)}",
                "",
                "## Feature Types",
                f"- {pipeline.feature_types}",
                "",
                "## Selected Features",
                f"- {pipeline.feature_selection_report.get('mutual_information', {})}",
                "",
                "## Scaling Strategy",
                "- Continuous features scaled using StandardScaler; binary features left unchanged.",
                "",
                "## Imbalance Strategy",
                "- SMOTE, SMOTEENN, and RandomUnderSampler are supported and configurable.",
                "",
                "## Recommendations",
                "- Preserve the preprocessing artifacts for future training and deployment."
                "- Review cleaning issues before model training.",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Preprocessing pipeline completed successfully")
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
