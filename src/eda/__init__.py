"""
EDA (Exploratory Data Analysis) pipeline module.

This package provides comprehensive automated exploratory data analysis
including dataset overview, feature analysis, correlation analysis, and
statistical feature ranking.
"""

from .correlation_analysis import CorrelationAnalysis
from .data_overview import DataOverview
from .feature_analysis import FeatureAnalysis
from .healthcare_analysis import HealthcareAnalysis
from .report_generator import ReportGenerator
from .statistical_analysis import StatisticalAnalysis
from .target_analysis import TargetAnalysis
from .visualization import VisualizationUtils

__all__ = [
    "DataOverview",
    "TargetAnalysis",
    "FeatureAnalysis",
    "CorrelationAnalysis",
    "HealthcareAnalysis",
    "StatisticalAnalysis",
    "VisualizationUtils",
    "ReportGenerator",
]
