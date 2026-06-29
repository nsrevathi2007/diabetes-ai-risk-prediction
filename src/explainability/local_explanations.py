"""Local SHAP explanations for individual patients."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from .shap_explainer import ShapExplainer
from .visualization import ShapVisualizer


class LocalExplanationGenerator:
    """Generate SHAP explanations for individual patient predictions."""

    def __init__(
        self,
        explainer: ShapExplainer,
        visualizer: ShapVisualizer,
        output_dir: str | Path = "artifacts/explanations",
        logger: Any | None = None,
    ) -> None:
        """Initialize local explanation generation."""
        self.explainer = explainer
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def explain_patient(
        self,
        patient_record: dict[str, Any] | pd.Series | pd.DataFrame,
        patient_id: str = "patient",
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Explain one patient prediction.

        Args:
            patient_record: One patient record as dict, Series, or one-row DataFrame.
            patient_id: Stable identifier used for plot filenames.
            top_n: Number of positive and negative contributors to return.

        Returns:
            Dictionary containing prediction, contributors, SHAP values, and plots.
        """
        start_time = time.perf_counter()
        patient_df = self._coerce_patient_frame(patient_record)
        shap_values = self.explainer.calculate_shap_values(patient_df)
        values = shap_values[0]
        probability = float(self.explainer.predict_risk(patient_df)[0])
        prediction_label = "High Diabetes Risk" if probability >= 0.5 else "Low Diabetes Risk"
        confidence = self._confidence_label(probability)

        contribution_df = pd.DataFrame(
            {
                "feature": patient_df.columns,
                "value": patient_df.iloc[0].to_numpy(),
                "shap_value": values,
            }
        ).sort_values("shap_value", ascending=False)

        positive = contribution_df[contribution_df["shap_value"] > 0].head(top_n)
        negative = contribution_df[contribution_df["shap_value"] < 0].sort_values("shap_value").head(top_n)

        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        local_explanation = shap.Explanation(
            values=values,
            base_values=self.explainer.expected_value,
            data=patient_df.iloc[0].to_numpy(),
            feature_names=list(patient_df.columns),
        )

        plots = {
            "waterfall_plot": self.visualizer.save_waterfall_plot(
                local_explanation,
                patient_dir / "waterfall_plot.png",
            ),
            "force_plot": self.visualizer.save_force_plot(
                local_explanation,
                patient_dir / "force_plot.html",
            ),
            "decision_plot": self.visualizer.save_decision_plot(
                self.explainer.expected_value,
                values.tolist(),
                patient_df.iloc[0],
                patient_dir / "decision_plot.png",
            ),
        }

        result = {
            "patient_id": patient_id,
            "prediction": prediction_label,
            "risk_probability": probability,
            "confidence": confidence,
            "top_positive_contributors": positive.to_dict(orient="records"),
            "top_negative_contributors": negative.to_dict(orient="records"),
            "shap_values": contribution_df.sort_values("feature").to_dict(orient="records"),
            "plots": plots,
            "output_dir": patient_dir,
        }
        self._log(
            "Generated local SHAP explanation for %s in %.2fs",
            patient_id,
            time.perf_counter() - start_time,
        )
        return result

    def explain_batch(
        self,
        X: pd.DataFrame,
        n_patients: int = 10,
        random_state: int = 42,
    ) -> list[dict[str, Any]]:
        """Generate local explanations for a random patient batch."""
        sample = X.sample(n=min(n_patients, len(X)), random_state=random_state)
        explanations: list[dict[str, Any]] = []
        for batch_number, (_, row) in enumerate(sample.iterrows(), start=1):
            explanations.append(self.explain_patient(row, patient_id=f"patient_{batch_number:02d}"))
        return explanations

    def _coerce_patient_frame(self, patient_record: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Normalize patient input to a one-row dataframe."""
        if isinstance(patient_record, pd.DataFrame):
            if len(patient_record) != 1:
                raise ValueError("Local explanations require exactly one patient record")
            patient_df = patient_record.copy()
        elif isinstance(patient_record, pd.Series):
            patient_df = patient_record.to_frame().T
        else:
            patient_df = pd.DataFrame([patient_record])

        expected_features = self.explainer.feature_names
        if expected_features:
            missing = [feature for feature in expected_features if feature not in patient_df.columns]
            if missing:
                raise ValueError(f"Patient record is missing required features: {missing}")
            patient_df = patient_df[expected_features]
        return patient_df

    def _confidence_label(self, probability: float) -> str:
        """Convert probability distance from threshold into a confidence label."""
        distance = abs(probability - 0.5)
        if distance >= 0.25:
            return "High"
        if distance >= 0.10:
            return "Medium"
        return "Low"

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
