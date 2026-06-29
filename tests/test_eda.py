"""
Pytest tests for EDA pipeline modules.

Tests cover:
- Dataset overview generation
- Target variable analysis
- Feature analysis
- Correlation analysis
- Statistical analysis
- Visualization generation
- Report generation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    CorrelationAnalysis,
    DataOverview,
    FeatureAnalysis,
    HealthcareAnalysis,
    ReportGenerator,
    StatisticalAnalysis,
    TargetAnalysis,
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "Diabetes_binary": np.random.binomial(1, 0.2, 100),
        "BMI": np.random.uniform(15, 50, 100),
        "Age": np.random.randint(1, 14, 100),
        "HighBP": np.random.binomial(1, 0.4, 100),
        "PhysActivity": np.random.binomial(1, 0.7, 100),
        "Category": np.random.choice(['A', 'B', 'C'], 100),
    })
    return df


class TestDataOverview:
    """Tests for DataOverview class."""
    
    def test_generate_overview(self, sample_dataset):
        """Test overview generation."""
        analyzer = DataOverview()
        overview = analyzer.generate_overview(sample_dataset)
        
        assert overview['shape']['rows'] == 100
        assert overview['shape']['columns'] == 6
        assert 'data_types' in overview
        assert 'descriptive_statistics' in overview
    
    def test_save_overview(self, sample_dataset):
        """Test saving overview to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = DataOverview()
            overview = analyzer.generate_overview(sample_dataset)
            
            output_path = Path(temp_dir) / "overview.json"
            analyzer.save_overview(overview, str(output_path))
            
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert data['shape']['rows'] == 100


class TestTargetAnalysis:
    """Tests for TargetAnalysis class."""
    
    def test_analyze_target(self, sample_dataset):
        """Test target variable analysis."""
        analyzer = TargetAnalysis()
        analysis = analyzer.analyze_target(sample_dataset, "Diabetes_binary")
        
        assert 'class_distribution' in analysis
        assert 'class_imbalance_ratio' in analysis
        assert 'majority_class' in analysis
        assert 'minority_class' in analysis
    
    def test_generate_count_plot(self, sample_dataset):
        """Test count plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = TargetAnalysis()
            output_path = Path(temp_dir) / "count_plot.png"
            
            result = analyzer.generate_count_plot(
                sample_dataset, "Diabetes_binary", str(output_path)
            )
            
            assert result is not None
            assert result.exists()
    
    def test_generate_pie_chart(self, sample_dataset):
        """Test pie chart generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = TargetAnalysis()
            output_path = Path(temp_dir) / "pie_chart.png"
            
            result = analyzer.generate_pie_chart(
                sample_dataset, "Diabetes_binary", str(output_path)
            )
            
            assert result is not None
            assert result.exists()


class TestFeatureAnalysis:
    """Tests for FeatureAnalysis class."""
    
    def test_analyze_features(self, sample_dataset):
        """Test feature analysis."""
        analyzer = FeatureAnalysis()
        analysis = analyzer.analyze_features(sample_dataset)
        
        assert analysis['total_features'] == 6
        assert 'numerical_features' in analysis
        assert 'categorical_features' in analysis
        assert 'feature_statistics' in analysis
    
    def test_generate_numerical_visualizations(self, sample_dataset):
        """Test numerical feature visualizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = FeatureAnalysis()
            paths = analyzer.generate_numerical_visualizations(
                sample_dataset, "BMI", temp_dir
            )
            
            assert 'histogram' in paths
            assert 'boxplot' in paths
    
    def test_generate_categorical_visualizations(self, sample_dataset):
        """Test categorical feature visualizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = FeatureAnalysis()
            paths = analyzer.generate_categorical_visualizations(
                sample_dataset, "Category", temp_dir
            )
            
            assert 'count_plot' in paths


class TestCorrelationAnalysis:
    """Tests for CorrelationAnalysis class."""
    
    def test_compute_correlation_matrix(self, sample_dataset):
        """Test correlation matrix computation."""
        analyzer = CorrelationAnalysis()
        corr_matrix = analyzer.compute_correlation_matrix(sample_dataset)
        
        assert corr_matrix.shape[0] > 0
        assert corr_matrix.shape[1] > 0
    
    def test_analyze_correlations(self, sample_dataset):
        """Test correlation analysis."""
        analyzer = CorrelationAnalysis()
        analysis = analyzer.analyze_correlations(sample_dataset, "Diabetes_binary")
        
        assert 'correlation_matrix' in analysis
        assert 'target_correlations' in analysis
        assert 'strongest_feature_correlations' in analysis
    
    def test_generate_heatmap(self, sample_dataset):
        """Test heatmap generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CorrelationAnalysis()
            output_path = Path(temp_dir) / "heatmap.png"
            
            result = analyzer.generate_heatmap(sample_dataset, str(output_path))
            
            assert result is not None
            assert result.exists()


class TestStatisticalAnalysis:
    """Tests for StatisticalAnalysis class."""
    
    def test_compute_mutual_information(self, sample_dataset):
        """Test mutual information computation."""
        analyzer = StatisticalAnalysis()
        mi_scores = analyzer.compute_mutual_information(
            sample_dataset, "Diabetes_binary"
        )
        
        assert len(mi_scores) > 0
        assert all(isinstance(v, float) for v in mi_scores.values())
    
    def test_rank_features(self, sample_dataset):
        """Test feature ranking."""
        analyzer = StatisticalAnalysis()
        ranking = analyzer.rank_features(sample_dataset, "Diabetes_binary")
        
        assert 'mutual_information_scores' in ranking
        assert 'combined_ranking' in ranking
        assert 'top_10_features' in ranking


class TestHealthcareAnalysis:
    """Tests for HealthcareAnalysis class."""
    
    def test_generate_feature_target_visualizations(self, sample_dataset):
        """Test feature-target visualization generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = HealthcareAnalysis()
            paths = analyzer.generate_feature_target_visualizations(
                sample_dataset, "Diabetes_binary", temp_dir
            )
            
            assert len(paths) > 0
    
    def test_generate_summary_statistics(self, sample_dataset):
        """Test summary statistics generation."""
        analyzer = HealthcareAnalysis()
        stats = analyzer.generate_summary_statistics(
            sample_dataset, "Diabetes_binary"
        )
        
        assert len(stats) > 0


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    def test_generate_report(self, sample_dataset):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare required data
            overview_analyzer = DataOverview()
            dataset_overview = overview_analyzer.generate_overview(sample_dataset)
            
            target_analyzer = TargetAnalysis()
            target_analysis = target_analyzer.analyze_target(
                sample_dataset, "Diabetes_binary"
            )
            
            feature_analyzer = FeatureAnalysis()
            feature_analysis = feature_analyzer.analyze_features(sample_dataset)
            
            corr_analyzer = CorrelationAnalysis()
            correlation_analysis = corr_analyzer.analyze_correlations(
                sample_dataset, "Diabetes_binary"
            )
            
            stat_analyzer = StatisticalAnalysis()
            feature_ranking = stat_analyzer.rank_features(
                sample_dataset, "Diabetes_binary"
            )
            
            healthcare_analyzer = HealthcareAnalysis()
            healthcare_summary = healthcare_analyzer.generate_summary_statistics(
                sample_dataset, "Diabetes_binary"
            )
            
            # Generate report
            generator = ReportGenerator()
            output_path = Path(temp_dir) / "report.md"
            
            generator.generate_report(
                dataset_overview=dataset_overview,
                target_analysis=target_analysis,
                feature_analysis=feature_analysis,
                correlation_analysis=correlation_analysis,
                feature_ranking=feature_ranking,
                healthcare_summary=healthcare_summary,
                output_path=str(output_path)
            )
            
            assert output_path.exists()
            content = output_path.read_text()
            assert "Dataset Overview" in content
            assert "Target Variable Analysis" in content


class TestIntegration:
    """Integration tests for EDA pipeline."""
    
    def test_complete_eda_pipeline(self, sample_dataset):
        """Test complete EDA pipeline execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run all analyses
            overview_analyzer = DataOverview()
            dataset_overview = overview_analyzer.generate_overview(sample_dataset)
            
            target_analyzer = TargetAnalysis()
            target_analysis = target_analyzer.analyze_target(
                sample_dataset, "Diabetes_binary"
            )
            target_analyzer.generate_count_plot(
                sample_dataset, "Diabetes_binary",
                f"{temp_dir}/count_plot.png"
            )
            
            feature_analyzer = FeatureAnalysis()
            feature_analysis = feature_analyzer.analyze_features(sample_dataset)
            
            corr_analyzer = CorrelationAnalysis()
            correlation_analysis = corr_analyzer.analyze_correlations(
                sample_dataset, "Diabetes_binary"
            )
            
            stat_analyzer = StatisticalAnalysis()
            feature_ranking = stat_analyzer.rank_features(
                sample_dataset, "Diabetes_binary"
            )
            
            healthcare_analyzer = HealthcareAnalysis()
            healthcare_summary = healthcare_analyzer.generate_summary_statistics(
                sample_dataset, "Diabetes_binary"
            )
            
            # Verify results
            assert dataset_overview['shape']['rows'] == 100
            assert len(target_analysis['class_distribution']) == 2
            assert feature_analysis['total_features'] == 6
            assert len(correlation_analysis['target_correlations']) > 0
            assert len(feature_ranking['top_10_features']) > 0
            assert len(healthcare_summary) > 0
