"""Model factory for baseline machine learning pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODEL_REGISTRY: dict[str, Any] = {}


def build_model(model_name: str, random_state: int = 42) -> Any:
    """Construct a model instance for a supported name.

    Args:
        model_name: Name of the requested model.
        random_state: Random state for reproducibility.

    Returns:
        Initialized model instance.
    """
    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=random_state, max_iter=1000)
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=random_state)
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=random_state, n_estimators=200, n_jobs=-1)
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=200,
            random_state=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def get_supported_models() -> list[str]:
    """Return the supported model names."""
    return [
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
