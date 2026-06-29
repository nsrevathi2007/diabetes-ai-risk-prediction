"""
Data overview module for EDA pipeline.

Provides comprehensive overview statistics of the dataset including
shape, data types, memory usage, and basic statistics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class DataOverview:
    """
    Generates comprehensive dataset overview statistics.
    
    This class computes and provides access to high-level dataset statistics
    including shape, memory usage, data types, and descriptive statistics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataOverview analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset overview.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary containing overview statistics.
        """
        self.logger.info("Generating dataset overview statistics")
        
        overview = {
            "shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            },
            "feature_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "duplicates": {
                "total": int(df.duplicated().sum()),
                "percentage": float(
                    (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
                ),
            },
            "missing_values": {
                "total": int(df.isnull().sum().sum()),
                "percentage": float(
                    (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    if len(df) > 0 else 0
                ),
                "by_column": {
                    col: int(count) for col, count in df.isnull().sum().items()
                },
            },
            "descriptive_statistics": self._compute_descriptive_stats(df),
        }
        
        self.logger.info(f"Dataset overview generated: {overview['shape']}")
        return overview
    
    def _compute_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute descriptive statistics for all numerical columns.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary with descriptive statistics.
        """
        numeric_df = df.select_dtypes(include=['number'])
        
        stats = {}
        for col in numeric_df.columns:
            stats[col] = {
                "count": int(numeric_df[col].count()),
                "mean": float(numeric_df[col].mean()),
                "median": float(numeric_df[col].median()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "25%": float(numeric_df[col].quantile(0.25)),
                "50%": float(numeric_df[col].quantile(0.50)),
                "75%": float(numeric_df[col].quantile(0.75)),
                "max": float(numeric_df[col].max()),
                "skewness": float(numeric_df[col].skew()),
                "kurtosis": float(numeric_df[col].kurtosis()),
            }
        
        return stats
    
    def save_overview(self, overview: Dict[str, Any], output_path: str) -> Path:
        """
        Save overview statistics to JSON file.
        
        Args:
            overview: Overview statistics dictionary.
            output_path: Path to save the JSON file.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_file.write_text(json.dumps(overview, indent=2), encoding='utf-8')
        self.logger.info(f"Dataset overview saved to {output_file}")
        
        return output_file
