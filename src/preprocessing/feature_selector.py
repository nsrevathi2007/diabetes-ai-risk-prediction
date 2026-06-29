"""Feature selection ranking utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


class FeatureSelector:
    """Compute feature rankings using several statistical methods."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize the selector with an optional logger."""
        self.logger = logger

    def rank_features(
        self, df: pd.DataFrame, target_column: str = "Diabetes_binary"
    ) -> dict[str, Any]:
        """Rank features with mutual information, chi-square, ANOVA, and correlation."""
        feature_frame = df.drop(columns=[target_column], errors="ignore")
        target = df[target_column]

        encoded_features = pd.get_dummies(feature_frame, dummy_na=False)
        encoded_features = encoded_features.astype(float)

        mutual_information = mutual_info_classif(encoded_features, target, random_state=42)
        mutual_information_scores = {
            feature: float(score)
            for feature, score in zip(encoded_features.columns, mutual_information)
        }

        chi_square_scores = {}
        try:
            chi_values, _ = chi2(encoded_features, target)
            chi_square_scores = {
                feature: float(score)
                for feature, score in zip(encoded_features.columns, chi_values)
            }
        except ValueError:
            chi_square_scores = {feature: 0.0 for feature in encoded_features.columns}

        anova_scores, _ = f_classif(encoded_features, target)
        anova_f_score = {
            feature: float(score)
            for feature, score in zip(encoded_features.columns, anova_scores)
        }

        correlation_filtering = {
            feature: float(abs(encoded_features[feature].corr(target)))
            for feature in encoded_features.columns
        }

        return {
            "mutual_information": dict(
                sorted(mutual_information_scores.items(), key=lambda item: item[1], reverse=True)
            ),
            "chi_square": dict(
                sorted(chi_square_scores.items(), key=lambda item: item[1], reverse=True)
            ),
            "anova_f_score": dict(
                sorted(anova_f_score.items(), key=lambda item: item[1], reverse=True)
            ),
            "correlation_filtering": dict(
                sorted(correlation_filtering.items(), key=lambda item: item[1], reverse=True)
            ),
        }
