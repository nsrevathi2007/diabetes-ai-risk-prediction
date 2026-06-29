"""Data cleaning and validation helpers for preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DataCleaner:
    """Validate data quality and log issues without removing duplicates."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize the cleaner with an optional logger."""
        self.logger = logger

    def validate_and_report(
        self, df: pd.DataFrame, target_column: str = "Diabetes_binary"
    ) -> dict[str, Any]:
        """Validate the dataset and produce a cleaning report.

        Args:
            df: Input dataframe to validate.
            target_column: Name of the target column.

        Returns:
            Dictionary containing duplicate counts, issue counts, and logged issues.
        """
        issues: list[dict[str, Any]] = []

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows > 0:
            issues.append(
                {
                    "type": "duplicate_rows",
                    "message": f"Found {duplicate_rows} duplicate rows",
                    "count": duplicate_rows,
                }
            )

        if "BMI" in df.columns:
            invalid_bmi = df["BMI"].dropna()
            invalid_mask = invalid_bmi.between(10, 100, inclusive="neither")
            if not invalid_mask.all():
                issues.append(
                    {
                        "type": "impossible_bmi",
                        "message": "BMI values outside the expected range were found",
                        "count": int((~invalid_mask).sum()),
                    }
                )

        if "Age" in df.columns:
            invalid_age = df["Age"].dropna()
            invalid_mask = invalid_age.between(1, 13, inclusive="both")
            if not invalid_mask.all():
                issues.append(
                    {
                        "type": "impossible_age",
                        "message": "Age values outside the expected range were found",
                        "count": int((~invalid_mask).sum()),
                    }
                )

        healthcare_columns = [
            "HighBP",
            "HighChol",
            "CholCheck",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "DiffWalk",
            "Sex",
        ]
        for column in healthcare_columns:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
                invalid_mask = values.isna() | values.isin([0, 1])
                if not invalid_mask.all():
                    issues.append(
                        {
                            "type": "invalid_healthcare_value",
                            "message": f"Unexpected values found in {column}",
                            "count": int((~invalid_mask).sum()),
                        }
                    )

        if target_column in df.columns:
            target_values = pd.to_numeric(df[target_column], errors="coerce")
            invalid_target_mask = target_values.isna() | target_values.isin([0, 1])
            if not invalid_target_mask.all():
                issues.append(
                    {
                        "type": "invalid_target_value",
                        "message": f"Unexpected target values found in {target_column}",
                        "count": int((~invalid_target_mask).sum()),
                    }
                )

        negative_values = df.select_dtypes(include=[np.number]).lt(0).sum()
        for column, count in negative_values.items():
            if count > 0:
                issues.append(
                    {
                        "type": "negative_values",
                        "message": f"Negative values found in {column}",
                        "count": int(count),
                    }
                )

        if self.logger is not None:
            for issue in issues:
                self.logger.warning("Data cleaning issue: %s", issue["message"])

        return {
            "duplicate_rows": duplicate_rows,
            "issue_count": len(issues),
            "issues": issues,
            "status": "issues_found" if issues else "clean",
        }
