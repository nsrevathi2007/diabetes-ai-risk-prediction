"""
Data schema definitions for diabetes dataset.

This module defines the expected schema for the diabetes binary health indicators
dataset from BRFSS 2015. It includes expected columns, data types, and validation rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataSchemaConfig:
    """
    Configuration class for diabetes dataset schema.
    
    Attributes:
        target_column: Name of the target variable column.
        expected_columns: List of expected column names.
        expected_dtypes: Dictionary mapping column names to expected data types.
        min_rows: Minimum expected number of rows.
        max_missing_percentage: Maximum allowed percentage of missing values per column.
    """
    
    target_column: str = "Diabetes_binary"
    expected_columns: List[str] = field(
        default_factory=lambda: [
            "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI",
            "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Sex", "Age", "Education", "Income"
        ]
    )
    expected_dtypes: Dict[str, str] = field(
        default_factory=lambda: {
            col: "float64" for col in [
                "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI",
                "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
                "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
                "Sex", "Age", "Education", "Income"
            ]
        }
    )
    min_rows: int = 1
    max_missing_percentage: float = 100.0  # No missing values allowed
    
    def validate_schema(self, df) -> tuple[bool, List[str]]:
        """
        Validate that a dataframe matches the schema.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        
        # Check columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            errors.append(f"Extra columns: {extra_cols}")
        
        # Check data types for existing columns
        for col in self.expected_columns:
            if col in df.columns:
                if str(df[col].dtype) != self.expected_dtypes[col]:
                    errors.append(
                        f"Column '{col}': expected {self.expected_dtypes[col]}, "
                        f"got {df[col].dtype}"
                    )
        
        # Check minimum rows
        if len(df) < self.min_rows:
            errors.append(f"Insufficient rows: {len(df)} < {self.min_rows}")
        
        return len(errors) == 0, errors
