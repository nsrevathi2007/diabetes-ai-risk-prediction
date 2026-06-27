"""Training package for model orchestration and experiment workflows.

This package will include modules to configure training jobs, manage datasets, and persist
trained estimators and metadata.
"""

from .trainer import DiabetesModelTrainer

__all__ = ["DiabetesModelTrainer"]
