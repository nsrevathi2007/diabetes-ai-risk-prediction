"""AI-Powered Diabetes Risk Prediction & Personalized Health Recommendation System.

This package exposes the core application modules and provides a clean entry point for
data ingestion, training, inference, explainability, recommendation, and monitoring subsystems.
"""

try:
    from .preprocessing import preprocessing
except ImportError:
    pass

try:
    from .training import training
except ImportError:
    pass

try:
    from .inference import inference
except ImportError:
    pass

try:
    from .explainability import explainability
except ImportError:
    pass

try:
    from .recommendations import recommendations
except ImportError:
    pass

try:
    from .monitoring import monitoring
except ImportError:
    pass

try:
    from .utils import utils
except ImportError:
    pass

from . import data_ingestion

__all__ = [
    "data_ingestion",
    "monitoring",
    "utils",
]
