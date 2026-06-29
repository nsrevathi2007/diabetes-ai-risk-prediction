"""
Feature analysis module for EDA pipeline.

Analyzes individual features including numerical and categorical variables,
generating statistics and visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FeatureAnalysis:
    """
    Analyzes individual features in the dataset.
    
    This class provides analysis for both numerical and categorical features
    including distribution statistics and visualization generation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureAnalysis analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        sns.set_style("whitegrid")
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze all features in the dataset.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary containing feature analysis results.
        """
        self.logger.info("Analyzing all features")
        
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis = {
            "total_features": len(df.columns),
            "numerical_features": len(numerical_features),
            "categorical_features": len(categorical_features),
            "feature_list": list(df.columns),
            "numerical_feature_list": numerical_features,
            "categorical_feature_list": categorical_features,
            "feature_statistics": {},
        }
        
        # Analyze numerical features
        for col in numerical_features:
            analysis["feature_statistics"][col] = self._analyze_numerical_feature(df, col)
        
        # Analyze categorical features
        for col in categorical_features:
            analysis["feature_statistics"][col] = self._analyze_categorical_feature(df, col)
        
        self.logger.info(f"Feature analysis completed: {analysis['total_features']} features")
        return analysis
    
    def _analyze_numerical_feature(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """
        Analyze a numerical feature.
        
        Args:
            df: DataFrame containing the feature.
            feature: Name of the feature.
            
        Returns:
            Dictionary containing feature statistics.
        """
        col_data = df[feature].dropna()
        
        return {
            "type": "numerical",
            "count": int(col_data.count()),
            "missing": int(df[feature].isnull().sum()),
            "unique": int(df[feature].nunique()),
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "25%": float(col_data.quantile(0.25)),
            "75%": float(col_data.quantile(0.75)),
            "max": float(col_data.max()),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
        }
    
    def _analyze_categorical_feature(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """
        Analyze a categorical feature.
        
        Args:
            df: DataFrame containing the feature.
            feature: Name of the feature.
            
        Returns:
            Dictionary containing feature statistics.
        """
        value_counts = df[feature].value_counts()
        
        return {
            "type": "categorical",
            "count": int(df[feature].count()),
            "missing": int(df[feature].isnull().sum()),
            "unique": int(df[feature].nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        }
    
    def generate_numerical_visualizations(
        self,
        df: pd.DataFrame,
        feature: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Generate visualizations for a numerical feature.
        
        Args:
            df: DataFrame containing the feature.
            feature: Name of the feature.
            output_dir: Directory to save figures. If None, figures are not saved.
            
        Returns:
            Dictionary with paths to saved figures.
        """
        self.logger.info(f"Generating visualizations for numerical feature: {feature}")
        
        paths = {}
        
        # Histogram
        paths['histogram'] = self._generate_histogram(df, feature, output_dir)
        
        # Boxplot
        paths['boxplot'] = self._generate_boxplot(df, feature, output_dir)
        
        return paths
    
    def _generate_histogram(
        self,
        df: pd.DataFrame,
        feature: str,
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """Generate histogram for a numerical feature."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        df[feature].hist(bins=50, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f'{feature}_histogram.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Histogram saved: {output_file}")
            plt.close(fig)
            return output_file
        
        plt.close(fig)
        return None
    
    def _generate_boxplot(
        self,
        df: pd.DataFrame,
        feature: str,
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """Generate boxplot for a numerical feature."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        df[[feature]].boxplot(ax=ax, patch_artist=True)
        ax.set_title(f'Boxplot of {feature}', fontsize=14, fontweight='bold')
        ax.set_ylabel(feature, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color the box
        for patch in ax.artists:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f'{feature}_boxplot.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Boxplot saved: {output_file}")
            plt.close(fig)
            return output_file
        
        plt.close(fig)
        return None
    
    def generate_categorical_visualizations(
        self,
        df: pd.DataFrame,
        feature: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Generate visualizations for a categorical feature.
        
        Args:
            df: DataFrame containing the feature.
            feature: Name of the feature.
            output_dir: Directory to save figures. If None, figures are not saved.
            
        Returns:
            Dictionary with paths to saved figures.
        """
        self.logger.info(f"Generating visualizations for categorical feature: {feature}")
        
        paths = {}
        paths['count_plot'] = self._generate_count_plot(df, feature, output_dir)
        
        return paths
    
    def _generate_count_plot(
        self,
        df: pd.DataFrame,
        feature: str,
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """Generate count plot for a categorical feature."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        value_counts = df[feature].value_counts().sort_index()
        colors = plt.cm.Set2(range(len(value_counts)))
        
        value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f'{feature}_count_plot.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Count plot saved: {output_file}")
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
        Save feature analysis to JSON file.
        
        Args:
            analysis: Feature analysis dictionary.
            output_path: Path to save the JSON file.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_file.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
        self.logger.info(f"Feature analysis saved to {output_file}")
        
        return output_file
