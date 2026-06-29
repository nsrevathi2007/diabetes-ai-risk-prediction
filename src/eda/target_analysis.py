"""
Target variable analysis module for EDA pipeline.

Analyzes the target variable (Diabetes_binary) including distribution,
class imbalance, and visualization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TargetAnalysis:
    """
    Analyzes the target variable distribution and class imbalance.
    
    This class provides comprehensive analysis of the target variable including
    class distribution, imbalance ratios, and visualization generation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the TargetAnalysis analyzer.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        sns.set_style("whitegrid")
    
    def analyze_target(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary"
    ) -> Dict[str, Any]:
        """
        Analyze the target variable.
        
        Args:
            df: DataFrame containing the target variable.
            target_column: Name of the target column.
            
        Returns:
            Dictionary containing target analysis results.
        """
        self.logger.info(f"Analyzing target variable: {target_column}")
        
        value_counts = df[target_column].value_counts().sort_index()
        percentages = (value_counts / len(df) * 100).round(2)
        
        analysis = {
            "target_column": target_column,
            "class_distribution": {
                int(idx): int(count) for idx, count in value_counts.items()
            },
            "class_percentages": {
                int(idx): float(pct) for idx, pct in percentages.items()
            },
            "class_counts": {
                int(idx): int(count) for idx, count in value_counts.items()
            },
            "class_imbalance_ratio": float(
                value_counts.max() / value_counts.min()
                if len(value_counts) > 1 else 1.0
            ),
            "majority_class": int(value_counts.idxmax()),
            "minority_class": int(value_counts.idxmin()) if len(value_counts) > 1 else int(value_counts.index[0]),
            "majority_count": int(value_counts.max()),
            "minority_count": int(value_counts.min()) if len(value_counts) > 1 else int(value_counts.iloc[0]),
        }
        
        self.logger.info(f"Target analysis completed: {analysis['class_distribution']}")
        return analysis
    
    def generate_count_plot(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        output_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Generate count plot for target variable.
        
        Args:
            df: DataFrame containing the target variable.
            target_column: Name of the target column.
            output_path: Path to save the figure. If None, figure is not saved.
            
        Returns:
            Path object pointing to the saved figure, or None if not saved.
        """
        self.logger.info(f"Generating count plot for {target_column}")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        counts = df[target_column].value_counts().sort_index()
        colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in counts.index]
        
        counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Distribution of {target_column}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        
        # Add count labels on bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Count plot saved to {output_file}")
            plt.close(fig)
            return output_file
        
        plt.close(fig)
        return None
    
    def generate_pie_chart(
        self,
        df: pd.DataFrame,
        target_column: str = "Diabetes_binary",
        output_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Generate pie chart for target variable distribution.
        
        Args:
            df: DataFrame containing the target variable.
            target_column: Name of the target column.
            output_path: Path to save the figure. If None, figure is not saved.
            
        Returns:
            Path object pointing to the saved figure, or None if not saved.
        """
        self.logger.info(f"Generating pie chart for {target_column}")
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        counts = df[target_column].value_counts().sort_index()
        labels = [f'Class {int(i)}\n({v:,} samples)' for i, v in zip(counts.index, counts.values)]
        colors = ['#2ecc71', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax.set_title(f'{target_column} Class Distribution', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Pie chart saved to {output_file}")
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
        Save target analysis to JSON file.
        
        Args:
            analysis: Target analysis dictionary.
            output_path: Path to save the JSON file.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_file.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
        self.logger.info(f"Target analysis saved to {output_file}")
        
        return output_file
