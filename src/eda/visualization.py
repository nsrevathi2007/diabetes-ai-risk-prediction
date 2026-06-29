"""
Visualization utilities module for EDA pipeline.

Contains utility functions for creating consistent, publication-quality visualizations.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class VisualizationUtils:
    """
    Utility class for creating consistent visualizations.
    
    This class provides helper methods for styling and saving figures
    with consistent formatting.
    """
    
    # Color palettes
    COLORS_BINARY = ['#2ecc71', '#e74c3c']  # Green for 0, Red for 1
    COLORS_QUALITATIVE = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Figure style
    DPI = 300
    DEFAULT_FIGSIZE = (12, 7)
    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 12
    
    @staticmethod
    def setup_style():
        """Set up matplotlib and seaborn style for consistent appearance."""
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    @staticmethod
    def save_figure(
        fig: plt.Figure,
        output_path: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> Path:
        """
        Save a matplotlib figure to disk.
        
        Args:
            fig: Matplotlib figure object.
            output_path: Path to save the figure.
            dpi: Resolution in dots per inch.
            bbox_inches: Bounding box setting.
            
        Returns:
            Path object pointing to the saved file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_file, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        
        return output_file
    
    @staticmethod
    def apply_formatting(
        ax,
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: bool = True,
    ):
        """
        Apply consistent formatting to an axes object.
        
        Args:
            ax: Matplotlib axes object.
            title: Title for the plot.
            xlabel: Label for x-axis.
            ylabel: Label for y-axis.
            grid: Whether to show grid.
        """
        ax.set_title(title, fontsize=VisualizationUtils.TITLE_FONTSIZE, fontweight='bold', pad=15)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=VisualizationUtils.LABEL_FONTSIZE, fontweight='bold')
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=VisualizationUtils.LABEL_FONTSIZE, fontweight='bold')
        
        if grid:
            ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def add_value_labels(ax, values, offset=500):
        """
        Add value labels on top of bars in a bar plot.
        
        Args:
            ax: Matplotlib axes object.
            values: Array of values to label.
            offset: Vertical offset for labels.
        """
        for i, v in enumerate(values):
            ax.text(i, v + offset, str(int(v)), ha='center', va='bottom', fontweight='bold')
