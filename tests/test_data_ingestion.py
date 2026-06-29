"""
Pytest tests for data ingestion pipeline.

Tests cover:
- Data loading
- Validation (duplicates, missing values, infinite values, data types)
- Schema compliance
- Report generation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.data_loader import DataLoader
from src.data_ingestion.report_generator import ReportGenerator
from src.data_ingestion.schema import DataSchemaConfig
from src.data_ingestion.validator import DataValidator


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_load_dataset_success(self):
        """Test loading a valid dataset."""
        # Create a temporary CSV file
        df_temp = pd.DataFrame({
            "Diabetes_binary": [0.0, 1.0],
            "HighBP": [1.0, 0.0],
            "HighChol": [1.0, 1.0],
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_temp.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load_dataset(temp_path)
            assert df.shape == (2, 3)
            assert list(df.columns) == ["Diabetes_binary", "HighBP", "HighChol"]
        finally:
            Path(temp_path).unlink()
    
    def test_load_dataset_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_dataset("nonexistent_file.csv")
    
    def test_load_dataset_empty_file(self):
        """Test that ValueError is raised for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            loader = DataLoader()
            with pytest.raises(ValueError):
                loader.load_dataset(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_verify_file_exists(self):
        """Test file existence verification."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            loader = DataLoader()
            assert loader.verify_file_exists(temp_path) is True
            assert loader.verify_file_exists("nonexistent_file.txt") is False
        finally:
            Path(temp_path).unlink()
    
    def test_check_empty_dataset(self):
        """Test empty dataset detection."""
        loader = DataLoader()
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        assert loader.check_empty_dataset(empty_df) is True
        
        # Non-empty dataframe
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        assert loader.check_empty_dataset(df) is False


class TestDataValidator:
    """Tests for DataValidator class."""
    
    @pytest.fixture
    def valid_df(self):
        """Create a valid test dataframe."""
        return pd.DataFrame({
            "Diabetes_binary": [0.0, 1.0, 0.0],
            "HighBP": [1.0, 0.0, 1.0],
            "HighChol": [1.0, 1.0, 0.0],
            "CholCheck": [1.0, 0.0, 1.0],
            "BMI": [30.5, 25.0, 28.3],
            "Smoker": [0.0, 1.0, 0.0],
            "Stroke": [0.0, 0.0, 1.0],
            "HeartDiseaseorAttack": [0.0, 1.0, 0.0],
            "PhysActivity": [1.0, 0.0, 1.0],
            "Fruits": [0.0, 1.0, 0.0],
            "Veggies": [1.0, 0.0, 1.0],
            "HvyAlcoholConsump": [0.0, 0.0, 0.0],
            "AnyHealthcare": [1.0, 1.0, 1.0],
            "NoDocbcCost": [0.0, 0.0, 0.0],
            "GenHlth": [3.0, 2.0, 4.0],
            "MentHlth": [0.0, 5.0, 10.0],
            "PhysHlth": [0.0, 0.0, 5.0],
            "DiffWalk": [0.0, 0.0, 1.0],
            "Sex": [0.0, 1.0, 0.0],
            "Age": [9.0, 5.0, 7.0],
            "Education": [4.0, 2.0, 3.0],
            "Income": [3.0, 2.0, 4.0],
        })
    
    def test_check_duplicates_no_duplicates(self, valid_df):
        """Test duplicate detection with no duplicates."""
        validator = DataValidator()
        result = validator.check_duplicates(valid_df)
        assert result["is_valid"] is True
        assert result["num_duplicates"] == 0
    
    def test_check_duplicates_with_duplicates(self, valid_df):
        """Test duplicate detection with duplicates."""
        df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        validator = DataValidator()
        result = validator.check_duplicates(df)
        assert result["is_valid"] is False
        assert result["num_duplicates"] > 0
    
    def test_check_missing_values_no_missing(self, valid_df):
        """Test missing value detection with no missing values."""
        validator = DataValidator()
        result = validator.check_missing_values(valid_df)
        assert result["is_valid"] is True
        assert result["total_missing"] == 0
    
    def test_check_missing_values_with_missing(self, valid_df):
        """Test missing value detection with missing values."""
        df = valid_df.copy()
        df.loc[0, "Diabetes_binary"] = np.nan
        df.loc[1, "HighBP"] = np.nan
        
        validator = DataValidator()
        result = validator.check_missing_values(df)
        assert result["is_valid"] is False
        assert result["total_missing"] == 2
        assert "Diabetes_binary" in result["columns_with_missing"]
        assert "HighBP" in result["columns_with_missing"]
    
    def test_check_infinite_values_no_infinite(self, valid_df):
        """Test infinite value detection with no infinite values."""
        validator = DataValidator()
        result = validator.check_infinite_values(valid_df)
        assert result["is_valid"] is True
        assert result["total_infinite"] == 0
    
    def test_check_infinite_values_with_infinite(self, valid_df):
        """Test infinite value detection with infinite values."""
        df = valid_df.copy()
        df.loc[0, "BMI"] = np.inf
        df.loc[1, "Age"] = -np.inf
        
        validator = DataValidator()
        result = validator.check_infinite_values(df)
        assert result["is_valid"] is False
        assert result["total_infinite"] == 2
    
    def test_check_data_types(self, valid_df):
        """Test data type validation."""
        schema = DataSchemaConfig()
        validator = DataValidator(schema)
        result = validator.check_data_types(valid_df)
        assert result["is_valid"] is True
        assert len(result["mismatches"]) == 0
    
    def test_validate_all(self, valid_df):
        """Test comprehensive validation."""
        validator = DataValidator()
        is_valid, results = validator.validate_all(valid_df)
        assert is_valid is True
        assert "duplicates" in results
        assert "missing_values" in results
        assert "infinite_values" in results
        assert "data_types" in results


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for reports."""
        df = pd.DataFrame({
            "Diabetes_binary": [0.0, 1.0, 0.0],
            "HighBP": [1.0, 0.0, 1.0],
            "BMI": [30.5, 25.0, 28.3],
        })
        
        validation_results = {
            "duplicates": {"is_valid": True, "num_duplicates": 0, "percentage": 0.0},
            "missing_values": {"is_valid": True, "total_missing": 0, "total_missing_percentage": 0.0},
            "infinite_values": {"is_valid": True, "total_infinite": 0},
            "data_types": {"is_valid": True, "mismatches": {}},
        }
        
        return df, validation_results
    
    def test_generate_markdown_report(self, test_data):
        """Test markdown report generation."""
        df, validation_results = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator()
            generator._generate_markdown_report(
                df,
                validation_results,
                Path(temp_dir) / "report.md"
            )
            
            report_file = Path(temp_dir) / "report.md"
            assert report_file.exists()
            content = report_file.read_text()
            assert "Data Quality Report" in content
            assert "Dataset Overview" in content
    
    def test_generate_json_report(self, test_data):
        """Test JSON report generation."""
        df, validation_results = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator()
            generator._generate_json_report(
                df,
                validation_results,
                Path(temp_dir) / "report.json"
            )
            
            report_file = Path(temp_dir) / "report.json"
            assert report_file.exists()
            
            report_data = json.loads(report_file.read_text())
            assert "dataset_overview" in report_data
            assert report_data["dataset_overview"]["num_rows"] == 3
            assert report_data["dataset_overview"]["num_columns"] == 3
    
    def test_generate_reports(self, test_data):
        """Test generating both reports."""
        df, validation_results = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator()
            paths = generator.generate_reports(df, validation_results, output_dir=temp_dir)
            
            assert paths["markdown"].exists()
            assert paths["json"].exists()


class TestSchemaConfig:
    """Tests for DataSchemaConfig class."""
    
    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        schema = DataSchemaConfig()
        df = pd.DataFrame({
            col: [0.0, 1.0] for col in schema.expected_columns
        })
        
        is_valid, errors = schema.validate_schema(df)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        schema = DataSchemaConfig()
        df = pd.DataFrame({"Diabetes_binary": [0.0, 1.0]})
        
        is_valid, errors = schema.validate_schema(df)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validate_schema_extra_columns(self):
        """Test schema validation with extra columns."""
        schema = DataSchemaConfig()
        df = pd.DataFrame({
            col: [0.0, 1.0] for col in schema.expected_columns
        })
        df["ExtraColumn"] = [0.0, 1.0]
        
        is_valid, errors = schema.validate_schema(df)
        assert is_valid is False
        assert len(errors) > 0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from load to report."""
        # Create test data
        df_temp = pd.DataFrame({
            col: [0.0, 1.0, 0.0] for col in [
                "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI",
                "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
                "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
                "Sex", "Age", "Education", "Income"
            ]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            test_csv = Path(temp_dir) / "test.csv"
            df_temp.to_csv(test_csv, index=False)
            
            # Load data
            loader = DataLoader()
            df = loader.load_dataset(str(test_csv))
            
            # Validate data
            validator = DataValidator()
            is_valid, results = validator.validate_all(df)
            
            # Generate reports
            report_dir = Path(temp_dir) / "reports"
            generator = ReportGenerator()
            generator.generate_reports(df, results, output_dir=str(report_dir))
            
            # Verify
            assert df.shape[0] == 3
            assert is_valid is True
            assert (report_dir / "data_quality_report.md").exists()
            assert (report_dir / "data_quality_report.json").exists()
