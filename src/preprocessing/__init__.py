"""Data preprocessing package for feature engineering and pipeline construction.

This package will include modules for loading raw data, cleaning clinical variables, encoding
categorical features, and building reproducible transformation pipelines.
"""

from .pipeline import build_preprocessing_pipeline

__all__ = ["build_preprocessing_pipeline"]
