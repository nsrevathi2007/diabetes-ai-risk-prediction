"""AI-Powered Diabetes Risk Prediction & Personalized Health Recommendation System.

This package exposes the core application modules and provides a clean entry point for
training, inference, explainability, recommendation, and monitoring subsystems.
"""

from .preprocessing import preprocessing
from .training import training
from .inference import inference
from .explainability import explainability
from .recommendations import recommendations
from .monitoring import monitoring
from .utils import utils

__all__ = [
    "preprocessing",
    "training",
    "inference",
    "explainability",
    "recommendations",
    "monitoring",
    "utils",
]
