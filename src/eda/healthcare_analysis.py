"""
Healthcare-specific analysis module for EDA pipeline.

Generates publication-quality visualizations for healthcare-related features
and their relationships with the target variable.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class HealthcareAnalysis:
    """
    Generates healthcare-specific visualizations and analyses.
    
    This class creates publication-quality figures for healthcare features
    including BMI, blood pressure, physical activity, and other health indicators.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the HealthcareAnalysis analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def generate_feature_target_visualizations(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        output_dir: Optional[str] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Generate visualizations for healthcare features vs target.
        
        Args:
            df: DataFrame containing features and target.
            target_column: Name of the target column.
            output_dir: Directory to save figures.
            
        Returns:
            Dictionary mapping feature names to saved figure paths.
        """
        self.logger.info("Generating healthcare feature-target visualizations")
        
        healthcare_features = [
            'BMI',
            'Age',
            'HighBP',
            'HighChol',
            'PhysActivity',
            'Smoker',
            'HeartDiseaseorAttack',
            'GenHlth',
            'Income',
            'Education',
        ]
        
        paths = {}
        for feature in healthcare_features:
            if feature in df.columns:
                paths[feature] = self._generate_feature_target_plot(
                    df, feature, target_column, output_dir
                )
        
        return paths
    
    def _generate_feature_target_plot(
        self,
        df: pd.DataFrame,
        feature: str,
        target_column: str,
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """
        Generate a visualization for a feature vs target.
        
        Args:
            df: DataFrame containing the feature and target.
            feature: Name of the feature.
            target_column: Name of the target column.
            output_dir: Directory to save the figure.
            
        Returns:
            Path object pointing to the saved figure, or None if not saved.
        """
        self.logger.info(f"Generating {feature} vs {target_column} visualization")
        
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        
        # Check if feature is numerical or categorical
        if df[feature].dtype in ['float64', 'int64']:
            # Numerical feature - use boxplot
            data_to_plot = [df[df[target_column] == c][feature].dropna() for c in sorted(df[target_column].unique())]
            
            bp = ax.boxplot(
                data_to_plot,
                patch_artist=True,
                widths=0.6,
            )
            
            # Set labels using set_xticklabels (compatible with Matplotlib 3.11)
            label_list = [f'Class {int(c)}' for c in sorted(df[target_column].unique())]
            ax.set_xticklabels(label_list)
            
            # Color the boxes
            colors = ['#2ecc71', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(feature, fontsize=12, fontweight='bold')
        else:
            # Categorical feature - use grouped bar plot
            cross_tab = pd.crosstab(df[feature], df[target_column])
            cross_tab.plot(kind='bar', ax=ax, width=0.8, color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.legend(title=target_column, labels=[f'Class {int(c)}' for c in sorted(df[target_column].unique())])
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(f'{feature} Distribution by {target_column}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(target_column if df[feature].dtype not in ['float64', 'int64'] else 'Class', 
                      fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f'healthcare_{feature.lower()}_vs_target.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Healthcare visualization saved: {output_file}")
            plt.close(fig)
            return output_file
        
        plt.close(fig)
        return None
    
    def generate_summary_statistics(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary"
    ) -> Dict[str, Dict]:
        """
        Generate summary statistics for healthcare features by target class.
        
        Args:
            df: DataFrame containing features and target.
            target_column: Name of the target column.
            
        Returns:
            Dictionary containing summary statistics.
        """
        self.logger.info("Generating healthcare summary statistics")
        
        healthcare_features = [
            'BMI', 'Age', 'HighBP', 'HighChol', 'PhysActivity', 'Smoker',
            'HeartDiseaseorAttack', 'GenHlth', 'Income', 'Education'
        ]
        
        summary_stats = {}
        
        for feature in healthcare_features:
            if feature in df.columns:
                summary_stats[feature] = {}
                for target_class in sorted(df[target_column].unique()):
                    class_data = df[df[target_column] == target_class][feature].dropna()
                    
                    if len(class_data) > 0:
                        summary_stats[feature][f'class_{int(target_class)}'] = {
                            'count': int(len(class_data)),
                            'mean': float(class_data.mean()),
                            'median': float(class_data.median()),
                            'std': float(class_data.std()),
                            'min': float(class_data.min()),
                            'max': float(class_data.max()),
                        }
        
        return summary_stats
