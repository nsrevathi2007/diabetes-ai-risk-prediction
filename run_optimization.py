"""Runner for the model optimization and experiment tracking phase."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.optimization.experiment_runner import ExperimentRunner


def main() -> None:
    """Run model optimization experiments and save the results."""
    dataset_path = Path("data/processed/diabetes_binary_health_indicators_cleaned.csv")
    df = pd.read_csv(dataset_path)
    runner = ExperimentRunner(output_dir="artifacts")
    runner.run(df)
    print("MODEL OPTIMIZATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
