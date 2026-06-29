"""Preprocessing pipeline builder.

This module exposes a reusable entry point for composing the preprocessing steps
used during model training and inference.
"""

from __future__ import annotations

from typing import Any, Dict

from .preprocessing_pipeline import PreprocessingPipeline


def build_preprocessing_pipeline(config: Dict[str, Any] | None = None) -> PreprocessingPipeline:
    """Build and return a configured preprocessing pipeline.

    Args:
        config: Optional dictionary describing preprocessing options.

    Returns:
        A configured preprocessing pipeline instance.
    """
    config = config or {}
    target_column = config.get("target_column", "Diabetes_binary")
    logger = config.get("logger")
    return PreprocessingPipeline(target_column=target_column, logger=logger)
