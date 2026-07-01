"""API configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class APISettings:
    """Runtime settings for the FastAPI backend."""

    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    model_path: Path = Path("artifacts/models/best_model.joblib")
    processed_data_path: Path = Path("data/processed/diabetes_binary_health_indicators_cleaned.csv")
    recommendation_config_path: Path = Path("configs/recommendation_config.yaml")
    openai_api_key: str | None = None
    api_version: str = "1.0.0"


def load_settings(env_path: str | Path = ".env") -> APISettings:
    """Load API settings from .env and process environment variables."""
    load_dotenv(env_path)
    return APISettings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"},
        model_path=Path(os.getenv("MODEL_PATH", "artifacts/models/best_model.joblib")),
        processed_data_path=Path(
            os.getenv("PROCESSED_DATA_PATH", "data/processed/diabetes_binary_health_indicators_cleaned.csv")
        ),
        recommendation_config_path=Path(
            os.getenv("RECOMMENDATION_CONFIG_PATH", "configs/recommendation_config.yaml")
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION", "1.0.0"),
    )
