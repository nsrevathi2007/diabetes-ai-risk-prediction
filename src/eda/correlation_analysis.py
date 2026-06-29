"""
Correlation analysis module for EDA pipeline.

Analyzes correlations between features including Pearson correlation matrix,
heatmap generation, and identification of strongest correlations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CorrelationAnalysis:
    """
    Analyzes correlations between features in the dataset.
    
    This class computes Pearson correlation matrices, identifies strong
    correlations, and generates correlation heatmaps.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the CorrelationAnalysis analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        sns.set_style("whitegrid")
    
    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for numerical features.
        
        Args:
            df: DataFrame containing numerical features.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            
        Returns:
            Correlation matrix as DataFrame.
        """
        self.logger.info(f"Computing {method} correlation matrix")
        
        numerical_df = df.select_dtypes(include=['number'])
        corr_matrix = numerical_df.corr(method=method)
        
        self.logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
        return corr_matrix
    
    def analyze_correlations(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze correlations in the dataset.
        
        Args:
            df: DataFrame to analyze.
            target_column: Name of the target column.
            top_n: Number of top correlations to report.
            
        Returns:
            Dictionary containing correlation analysis results.
        """
        self.logger.info("Analyzing correlations")
        
        corr_matrix = self.compute_correlation_matrix(df)
        
        # Get correlations with target variable
        target_corrs = corr_matrix[target_column].drop(target_column)
        target_corrs_sorted = target_corrs.abs().sort_values(ascending=False)
        
        # Find strongest positive and negative correlations
        positive_corrs = target_corrs[target_corrs > 0].sort_values(ascending=False)
        negative_corrs = target_corrs[target_corrs < 0].sort_values(ascending=True)
        
        # Find strongest feature-to-feature correlations (excluding diagonal)
        feature_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feature_corrs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j]),
                })
        
        feature_corrs_sorted = sorted(
            feature_corrs,
            key=lambda x: abs(x['correlation']),
            reverse=True
        )
        
        analysis = {
            "correlation_matrix": corr_matrix.to_dict(),
            "target_correlations": {
                str(k): float(v) for k, v in target_corrs.items()
            },
            "top_positive_correlations_with_target": {
                str(k): float(v) for k, v in positive_corrs.head(top_n).items()
            },
            "top_negative_correlations_with_target": {
                str(k): float(v) for k, v in negative_corrs.head(top_n).items()
            },
            "strongest_feature_correlations": [
                {
                    "feature1": item["feature1"],
                    "feature2": item["feature2"],
                    "correlation": item["correlation"],
                }
                for item in feature_corrs_sorted[:top_n]
            ],
        }
        
        self.logger.info("Correlation analysis completed")
        return analysis
    
    def generate_heatmap(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Generate correlation heatmap.
        
        Args:
            df: DataFrame containing numerical features.
            output_path: Path to save the figure. If None, figure is not saved.
            
        Returns:
            Path object pointing to the saved figure, or None if not saved.
        """
        self.logger.info("Generating correlation heatmap")
        
        corr_matrix = self.compute_correlation_matrix(df)
        
        fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Correlation heatmap saved to {output_file}")
            plt.close(fig)
            return output_file
        
        plt.close(fig)
        return None
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: str
    ) -> Path:
        """
        Save correlation analysis to JSON file.
        
        Args:
            analysis: Correlation analysis dictionary.
            output_path: Path to save the JSON file.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_analysis = convert_to_serializable(analysis)
        output_file.write_text(json.dumps(serializable_analysis, indent=2), encoding='utf-8')
        self.logger.info(f"Correlation analysis saved to {output_file}")
        
        return output_file
