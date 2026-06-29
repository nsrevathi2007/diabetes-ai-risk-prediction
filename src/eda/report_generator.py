"""
Report generation module for EDA pipeline.

Generates comprehensive markdown reports summarizing all EDA findings.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ReportGenerator:
    """
    Generates comprehensive EDA reports in markdown format.
    
    This class creates professional, detailed markdown reports that summarize
    all findings from the EDA pipeline.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_report(
        self,
        dataset_overview: Dict[str, Any],
        target_analysis: Dict[str, Any],
        feature_analysis: Dict[str, Any],
        correlation_analysis: Dict[str, Any],
        feature_ranking: Dict[str, Any],
        healthcare_summary: Dict[str, Any],
        output_path: str
    ) -> Path:
        """
        Generate comprehensive EDA report.
        
        Args:
            dataset_overview: Dataset overview statistics.
            target_analysis: Target variable analysis.
            feature_analysis: Feature analysis results.
            correlation_analysis: Correlation analysis results.
            feature_ranking: Feature ranking results.
            healthcare_summary: Healthcare summary statistics.
            output_path: Path to save the report.
            
        Returns:
            Path object pointing to the saved report.
        """
        self.logger.info("Generating comprehensive EDA report")
        
        report_lines = []
        
        # Header
        report_lines.append("# Exploratory Data Analysis Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Table of Contents
        report_lines.append("## Table of Contents\n")
        report_lines.append("1. [Dataset Overview](#dataset-overview)")
        report_lines.append("2. [Target Variable Analysis](#target-variable-analysis)")
        report_lines.append("3. [Feature Analysis](#feature-analysis)")
        report_lines.append("4. [Correlation Analysis](#correlation-analysis)")
        report_lines.append("5. [Healthcare Insights](#healthcare-insights)")
        report_lines.append("6. [Feature Ranking](#feature-ranking)")
        report_lines.append("7. [Recommendations](#recommendations)\n")
        
        # Dataset Overview
        report_lines.extend(self._generate_dataset_overview_section(dataset_overview))
        
        # Target Analysis
        report_lines.extend(self._generate_target_analysis_section(target_analysis))
        
        # Feature Analysis
        report_lines.extend(self._generate_feature_analysis_section(feature_analysis))
        
        # Correlation Analysis
        report_lines.extend(self._generate_correlation_analysis_section(correlation_analysis))
        
        # Healthcare Insights
        report_lines.extend(self._generate_healthcare_insights_section(healthcare_summary))
        
        # Feature Ranking
        report_lines.extend(self._generate_feature_ranking_section(feature_ranking))
        
        # Recommendations
        report_lines.extend(self._generate_recommendations_section())
        
        # Save report
        report_content = "\n".join(report_lines)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_file.write_text(report_content, encoding='utf-8')
        self.logger.info(f"Report saved to {output_file}")
        
        return output_file
    
    def _generate_dataset_overview_section(self, overview: Dict[str, Any]) -> list:
        """Generate dataset overview section."""
        lines = []
        lines.append("## Dataset Overview\n")
        
        shape = overview.get('shape', {})
        lines.append(f"- **Number of Rows:** {shape.get('rows', 0):,}")
        lines.append(f"- **Number of Columns:** {shape.get('columns', 0)}")
        lines.append(f"- **Memory Usage:** {overview.get('memory_usage_mb', 0):.2f} MB")
        lines.append(f"- **Duplicate Rows:** {overview.get('duplicates', {}).get('total', 0)} ({overview.get('duplicates', {}).get('percentage', 0):.2f}%)")
        
        missing = overview.get('missing_values', {})
        lines.append(f"- **Missing Values:** {missing.get('total', 0)} ({missing.get('percentage', 0):.2f}%)\n")
        
        lines.append("### Descriptive Statistics\n")
        lines.append("| Feature | Count | Mean | Median | Std | Min | Max |")
        lines.append("|---------|-------|------|--------|-----|-----|-----|")
        
        stats = overview.get('descriptive_statistics', {})
        for feature, feature_stats in stats.items():
            if isinstance(feature_stats, dict):
                lines.append(
                    f"| {feature} | {feature_stats.get('count', 0)} | "
                    f"{feature_stats.get('mean', 0):.2f} | "
                    f"{feature_stats.get('median', 0):.2f} | "
                    f"{feature_stats.get('std', 0):.2f} | "
                    f"{feature_stats.get('min', 0):.2f} | "
                    f"{feature_stats.get('max', 0):.2f} |"
                )
        
        lines.append("")
        return lines
    
    def _generate_target_analysis_section(self, analysis: Dict[str, Any]) -> list:
        """Generate target analysis section."""
        lines = []
        lines.append("## Target Variable Analysis\n")
        
        dist = analysis.get('class_distribution', {})
        pct = analysis.get('class_percentages', {})
        
        lines.append("### Class Distribution\n")
        lines.append("| Class | Count | Percentage |")
        lines.append("|-------|-------|-----------|")
        
        for class_id in sorted(dist.keys()):
            lines.append(f"| {class_id} | {dist[class_id]:,} | {pct.get(class_id, 0):.2f}% |")
        
        lines.append("")
        lines.append(f"### Class Imbalance Ratio\n")
        lines.append(f"- **Ratio:** {analysis.get('class_imbalance_ratio', 1.0):.2f}:1")
        lines.append(f"- **Majority Class:** {analysis.get('majority_class')} ({analysis.get('majority_count', 0):,} samples)")
        lines.append(f"- **Minority Class:** {analysis.get('minority_class')} ({analysis.get('minority_count', 0):,} samples)\n")
        
        lines.append("**Note:** Class imbalance detected. Consider using techniques like SMOTE, class weights, or stratified sampling during model training.\n")
        
        return lines
    
    def _generate_feature_analysis_section(self, analysis: Dict[str, Any]) -> list:
        """Generate feature analysis section."""
        lines = []
        lines.append("## Feature Analysis\n")
        
        lines.append(f"- **Total Features:** {analysis.get('total_features', 0)}")
        lines.append(f"- **Numerical Features:** {analysis.get('numerical_features', 0)}")
        lines.append(f"- **Categorical Features:** {analysis.get('categorical_features', 0)}\n")
        
        lines.append("### Feature Summary\n")
        lines.append("| Feature | Type | Count | Unique | Mean/Top |")
        lines.append("|---------|------|-------|--------|----------|")
        
        stats = analysis.get('feature_statistics', {})
        for feature, feature_stats in sorted(stats.items()):
            feature_type = feature_stats.get('type', 'unknown')
            count = feature_stats.get('count', 0)
            unique = feature_stats.get('unique', 0)
            
            if feature_type == 'numerical':
                mean = feature_stats.get('mean', 0)
                lines.append(f"| {feature} | Numerical | {count} | {unique} | {mean:.2f} |")
            else:
                top = feature_stats.get('top_value', 'N/A')
                lines.append(f"| {feature} | Categorical | {count} | {unique} | {top} |")
        
        lines.append("")
        return lines
    
    def _generate_correlation_analysis_section(self, analysis: Dict[str, Any]) -> list:
        """Generate correlation analysis section."""
        lines = []
        lines.append("## Correlation Analysis\n")
        
        lines.append("### Top Positive Correlations with Target\n")
        pos_corrs = analysis.get('top_positive_correlations_with_target', {})
        for feature, corr in list(pos_corrs.items())[:5]:
            lines.append(f"- **{feature}:** {corr:.4f}")
        lines.append("")
        
        lines.append("### Top Negative Correlations with Target\n")
        neg_corrs = analysis.get('top_negative_correlations_with_target', {})
        for feature, corr in list(neg_corrs.items())[:5]:
            lines.append(f"- **{feature}:** {corr:.4f}")
        lines.append("")
        
        lines.append("### Strongest Feature-to-Feature Correlations\n")
        feature_corrs = analysis.get('strongest_feature_correlations', [])
        for item in feature_corrs[:5]:
            lines.append(
                f"- **{item.get('feature1')} - {item.get('feature2')}:** {item.get('correlation', 0):.4f}"
            )
        lines.append("")
        
        return lines
    
    def _generate_healthcare_insights_section(self, summary: Dict[str, Any]) -> list:
        """Generate healthcare insights section."""
        lines = []
        lines.append("## Healthcare Insights\n")
        
        lines.append("Key health indicators have been visualized against the target variable:\n")
        lines.append("- **BMI vs Diabetes**")
        lines.append("- **Age vs Diabetes**")
        lines.append("- **High Blood Pressure vs Diabetes**")
        lines.append("- **High Cholesterol vs Diabetes**")
        lines.append("- **Physical Activity vs Diabetes**")
        lines.append("- **Smoking vs Diabetes**")
        lines.append("- **Heart Disease/Attack vs Diabetes**")
        lines.append("- **General Health vs Diabetes**")
        lines.append("- **Income vs Diabetes**")
        lines.append("- **Education vs Diabetes**\n")
        
        lines.append("These visualizations reveal the relationship between health indicators and diabetes risk.\n")
        
        return lines
    
    def _generate_feature_ranking_section(self, ranking: Dict[str, Any]) -> list:
        """Generate feature ranking section."""
        lines = []
        lines.append("## Feature Ranking\n")
        
        lines.append("### Top 10 Features by Mutual Information\n")
        lines.append("| Rank | Feature | MI Score |")
        lines.append("|------|---------|----------|")
        
        mi_scores = ranking.get('mutual_information_scores', {})
        for rank, (feature, score) in enumerate(list(mi_scores.items())[:10], 1):
            lines.append(f"| {rank} | {feature} | {score:.4f} |")
        
        lines.append("")
        return lines
    
    def _generate_recommendations_section(self) -> list:
        """Generate recommendations section."""
        lines = []
        lines.append("## Recommendations Before Preprocessing\n")
        
        lines.append("### Data Quality\n")
        lines.append("- ✓ Verify that duplicate rows are intentional or remove if needed")
        lines.append("- ✓ Investigate any missing values and decide on handling strategy")
        lines.append("- ✓ Check for outliers and decide on treatment strategy\n")
        
        lines.append("### Feature Engineering\n")
        lines.append("- ✓ Consider creating interaction terms for highly correlated features")
        lines.append("- ✓ Evaluate polynomial features for non-linear relationships")
        lines.append("- ✓ Consider domain-specific feature engineering based on healthcare knowledge\n")
        
        lines.append("### Class Imbalance\n")
        lines.append("- ✓ Class imbalance detected - consider SMOTE, class weights, or stratification")
        lines.append("- ✓ Use appropriate metrics (precision, recall, F1) instead of accuracy\n")
        
        lines.append("### Feature Selection\n")
        lines.append("- ✓ Use mutual information scores to guide feature selection")
        lines.append("- ✓ Consider removing features with near-zero variance")
        lines.append("- ✓ Evaluate multicollinearity and remove highly correlated features if needed\n")
        
        lines.append("### Modeling\n")
        lines.append("- ✓ Use stratified k-fold cross-validation")
        lines.append("- ✓ Consider ensemble methods to handle class imbalance")
        lines.append("- ✓ Use appropriate evaluation metrics for imbalanced classification\n")
        
        return lines
