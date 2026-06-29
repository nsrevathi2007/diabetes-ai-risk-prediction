#!/usr/bin/env python
"""
Main runner script for the Exploratory Data Analysis (EDA) pipeline.

This script orchestrates the complete EDA workflow:
1. Load the processed dataset
2. Generate dataset overview statistics
3. Analyze target variable
4. Analyze individual features
5. Perform correlation analysis
6. Generate healthcare-specific visualizations
7. Compute statistical feature rankings
8. Generate comprehensive markdown report
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_ingestion.logging_config import setup_logging
from src.eda import (
    CorrelationAnalysis,
    DataOverview,
    FeatureAnalysis,
    HealthcareAnalysis,
    ReportGenerator,
    StatisticalAnalysis,
    TargetAnalysis,
    VisualizationUtils,
)


def main() -> int:
    """
    Main entry point for the EDA pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        start_time = time.time()
        
        # ============================================================================
        # Setup
        # ============================================================================
        print("=" * 80)
        print("EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
        print("=" * 80)
        print("\n[SETUP] Initializing logging and configuration...")
        
        logger = setup_logging(
            log_dir="logs",
            log_file="eda_pipeline.log",
            log_level="INFO",
        )
        
        logger.info("=" * 80)
        logger.info("EDA PIPELINE STARTED")
        logger.info("=" * 80)
        
        VisualizationUtils.setup_style()
        
        # ============================================================================
        # Load Dataset
        # ============================================================================
        print("[1/8] Loading processed dataset...")
        logger.info("Loading processed dataset")
        
        dataset_path = "data/processed/diabetes_binary_health_indicators_cleaned.csv"
        df = pd.read_csv(dataset_path)
        target_column = "Diabetes_binary"
        
        logger.info(f"Dataset loaded: {df.shape}")
        print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # ============================================================================
        # Dataset Overview
        # ============================================================================
        print("[2/8] Generating dataset overview...")
        logger.info("Generating dataset overview")
        
        overview_analyzer = DataOverview(logger)
        dataset_overview = overview_analyzer.generate_overview(df)
        overview_analyzer.save_overview(
            dataset_overview,
            "reports/statistics/dataset_overview.json"
        )
        print("✓ Dataset overview generated")
        
        # ============================================================================
        # Target Analysis
        # ============================================================================
        print("[3/8] Analyzing target variable...")
        logger.info("Analyzing target variable")
        
        target_analyzer = TargetAnalysis(logger)
        target_analysis = target_analyzer.analyze_target(df, target_column)
        target_analyzer.save_analysis(target_analysis, "reports/statistics/target_analysis.json")
        
        # Generate target visualizations
        target_analyzer.generate_count_plot(
            df, target_column, "reports/figures/target_count_plot.png"
        )
        target_analyzer.generate_pie_chart(
            df, target_column, "reports/figures/target_pie_chart.png"
        )
        print("✓ Target analysis completed with visualizations")
        
        # ============================================================================
        # Feature Analysis
        # ============================================================================
        print("[4/8] Analyzing features...")
        logger.info("Analyzing features")
        
        feature_analyzer = FeatureAnalysis(logger)
        feature_analysis = feature_analyzer.analyze_features(df)
        feature_analyzer.save_analysis(feature_analysis, "reports/statistics/feature_analysis.json")
        
        # Generate feature visualizations
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        for feature in numerical_features:
            if feature != target_column:
                feature_analyzer.generate_numerical_visualizations(
                    df, feature, "reports/figures"
                )
        
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        for feature in categorical_features:
            feature_analyzer.generate_categorical_visualizations(
                df, feature, "reports/figures"
            )
        
        print(f"✓ Feature analysis completed: {feature_analysis['total_features']} features analyzed")
        
        # ============================================================================
        # Correlation Analysis
        # ============================================================================
        print("[5/8] Computing correlations...")
        logger.info("Computing correlations")
        
        corr_analyzer = CorrelationAnalysis(logger)
        correlation_analysis = corr_analyzer.analyze_correlations(df, target_column)
        corr_analyzer.save_analysis(correlation_analysis, "reports/statistics/correlation_report.json")
        
        # Generate correlation heatmap
        corr_analyzer.generate_heatmap(df, "reports/figures/correlation_heatmap.png")
        print("✓ Correlation analysis completed with heatmap")
        
        # ============================================================================
        # Healthcare Analysis
        # ============================================================================
        print("[6/8] Generating healthcare visualizations...")
        logger.info("Generating healthcare visualizations")
        
        healthcare_analyzer = HealthcareAnalysis(logger)
        healthcare_summary = healthcare_analyzer.generate_summary_statistics(df, target_column)
        healthcare_analyzer.generate_feature_target_visualizations(
            df, target_column, "reports/figures"
        )
        print("✓ Healthcare visualizations generated")
        
        # ============================================================================
        # Statistical Feature Ranking
        # ============================================================================
        print("[7/8] Computing feature rankings...")
        logger.info("Computing feature rankings")
        
        stat_analyzer = StatisticalAnalysis(logger)
        feature_ranking = stat_analyzer.rank_features(df, target_column)
        stat_analyzer.save_ranking(feature_ranking, "reports/statistics/feature_ranking.json")
        print("✓ Feature ranking computed")
        
        # ============================================================================
        # Generate Markdown Report
        # ============================================================================
        print("[8/8] Generating comprehensive report...")
        logger.info("Generating comprehensive report")
        
        report_gen = ReportGenerator(logger)
        report_gen.generate_report(
            dataset_overview=dataset_overview,
            target_analysis=target_analysis,
            feature_analysis=feature_analysis,
            correlation_analysis=correlation_analysis,
            feature_ranking=feature_ranking,
            healthcare_summary=healthcare_summary,
            output_path="docs/eda_report.md"
        )
        print("✓ Comprehensive EDA report generated")
        
        # ============================================================================
        # Summary
        # ============================================================================
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("EDA PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"✓ Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"✓ Features Analyzed: {feature_analysis['total_features']}")
        print(f"✓ Numerical Features: {feature_analysis['numerical_features']}")
        print(f"✓ Categorical Features: {feature_analysis['categorical_features']}")
        print(f"✓ Target Classes: {len(target_analysis['class_distribution'])}")
        print(f"✓ Class Imbalance Ratio: {target_analysis['class_imbalance_ratio']:.2f}:1")
        print(f"\n✓ Execution Time: {elapsed_time:.2f} seconds")
        print("\n✓ EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        if 'logger' in locals():
            logger.exception("EDA pipeline execution failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
