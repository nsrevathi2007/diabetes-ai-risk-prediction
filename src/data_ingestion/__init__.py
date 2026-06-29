"""
Data ingestion module for diabetes dataset.

This module provides a complete data ingestion and validation pipeline including:
- Data loading and validation
- Schema compliance checking
- Report generation
- Structured logging
"""

from .data_loader import DataLoader
from .report_generator import ReportGenerator
from .schema import DataSchemaConfig
from .validator import DataValidator

__all__ = [
    "DataLoader",
    "DataValidator",
    "DataSchemaConfig",
    "ReportGenerator",
]
