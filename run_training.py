"""Runner for the model training and evaluation phase."""

from __future__ import annotations

from pathlib import Path

from src.training.trainer import DiabetesModelTrainer


def main() -> None:
    """Train and evaluate baseline models for diabetes risk prediction."""
    trainer = DiabetesModelTrainer(
        config={
            "dataset_path": Path("data/processed/diabetes_binary_health_indicators_cleaned.csv"),
            "target_column": "Diabetes_binary",
            "output_dir": Path("artifacts"),
            "models_dir": Path("artifacts/models"),
            "results_dir": Path("artifacts/results"),
            "reports_dir": Path("reports/model_reports"),
            "logger": None,
        }
    )

    results = trainer.train_from_file("data/processed/diabetes_binary_health_indicators_cleaned.csv")
    print(results)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
