"""Feature engineering utilities for healthcare datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Create clinically meaningful engineered features."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize the feature engineering component."""
        self.logger = logger
        self.engineered_features: list[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features and return an augmented dataframe."""
        working_df = df.copy()

        bmi_category = pd.cut(
            working_df["BMI"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
            include_lowest=True,
        )
        working_df["BMI_Category"] = bmi_category.astype(object).where(
            bmi_category.notna(), "Unknown"
        )

        age_group = []
        for value in working_df.get("Age", pd.Series([0] * len(working_df))):
            numeric_value = float(value)
            if numeric_value <= 3:
                age_group.append("Young")
            elif numeric_value <= 6:
                age_group.append("Middle")
            elif numeric_value <= 9:
                age_group.append("Older")
            else:
                age_group.append("Senior")
        working_df["Age_Group"] = age_group

        zero_series = pd.Series(0, index=working_df.index, dtype=float)

        smoker = working_df.get("Smoker")
        if smoker is None:
            smoker = zero_series
        else:
            smoker = smoker.astype(float)

        heavy_alcohol = working_df.get("HvyAlcoholConsump")
        if heavy_alcohol is None:
            heavy_alcohol = zero_series
        else:
            heavy_alcohol = heavy_alcohol.astype(float)

        phys_activity = working_df.get("PhysActivity")
        if phys_activity is None:
            phys_activity = pd.Series(1, index=working_df.index, dtype=float)
        else:
            phys_activity = phys_activity.astype(float)

        fruits = working_df.get("Fruits")
        if fruits is None:
            fruits = pd.Series(1, index=working_df.index, dtype=float)
        else:
            fruits = fruits.astype(float)

        veggies = working_df.get("Veggies")
        if veggies is None:
            veggies = pd.Series(1, index=working_df.index, dtype=float)
        else:
            veggies = veggies.astype(float)

        working_df["Lifestyle_Risk_Score"] = (
            smoker.fillna(0)
            + heavy_alcohol.fillna(0)
            + (1 - phys_activity.fillna(1))
            + (1 - fruits.fillna(1))
            + (1 - veggies.fillna(1))
        ).clip(0, 5)

        high_bp = working_df.get("HighBP")
        if high_bp is None:
            high_bp = zero_series
        else:
            high_bp = high_bp.astype(float)

        high_chol = working_df.get("HighChol")
        if high_chol is None:
            high_chol = zero_series
        else:
            high_chol = high_chol.astype(float)

        stroke = working_df.get("Stroke")
        if stroke is None:
            stroke = zero_series
        else:
            stroke = stroke.astype(float)

        heart_disease = working_df.get("HeartDiseaseorAttack")
        if heart_disease is None:
            heart_disease = zero_series
        else:
            heart_disease = heart_disease.astype(float)

        working_df["Cardiovascular_Risk_Score"] = (
            high_bp.fillna(0)
            + high_chol.fillna(0)
            + stroke.fillna(0)
            + heart_disease.fillna(0)
        )

        bmi = working_df.get("BMI")
        if bmi is None:
            bmi = zero_series
        else:
            bmi = bmi.astype(float)

        working_df["Health_Burden_Score"] = (
            working_df["Lifestyle_Risk_Score"]
            + working_df["Cardiovascular_Risk_Score"]
            + bmi.fillna(0).div(20).round(0)
        )

        self.engineered_features = [
            "BMI_Category",
            "Age_Group",
            "Lifestyle_Risk_Score",
            "Cardiovascular_Risk_Score",
            "Health_Burden_Score",
        ]

        if self.logger is not None:
            self.logger.info("Engineered features: %s", self.engineered_features)

        return working_df
