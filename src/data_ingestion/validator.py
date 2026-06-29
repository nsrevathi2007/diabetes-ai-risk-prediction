"""
Data validation module for diabetes dataset.

This module performs comprehensive validation checks on the dataset including
duplicate detection, missing value analysis, infinite value detection, and
data type validation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import DataSchemaConfig


class DataValidator:
    """
    Comprehensive data validation for diabetes dataset.
    
    This class performs validation checks including:
    - Duplicate row detection
    - Missing value analysis
    - Infinite value detection
    - Data type validation
    - Schema compliance
    """
    
    def __init__(
        self,
        schema: Optional[DataSchemaConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DataValidator.
        
        Args:
            schema: Schema configuration for validation. If not provided, default is created.
            logger: Optional logger instance. If not provided, a default logger is created.
        """
        self.schema = schema or DataSchemaConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results: Dict = {}
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Perform all validation checks on the dataset.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, validation_results_dict).
        """
        self.logger.info("Starting comprehensive data validation")
        
        # Run all validations
        self.validation_results = {
            "duplicates": self.check_duplicates(df),
            "missing_values": self.check_missing_values(df),
            "infinite_values": self.check_infinite_values(df),
            "data_types": self.check_data_types(df),
            "schema": self.check_schema_compliance(df),
        }
        
        # Determine overall validity - only critical checks cause failure
        # Critical: schema compliance and data types
        # Warnings: duplicates, missing values, infinite values
        critical_checks = ["schema", "data_types"]
        is_valid = all(
            self.validation_results[check].get("is_valid", False)
            for check in critical_checks
        )
        
        status = "PASSED" if is_valid else "FAILED"
        self.logger.info(f"Data validation {status}")
        
        return is_valid, self.validation_results
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Check for duplicate rows in the dataset.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary with duplicate analysis results.
        """
        self.logger.info("Checking for duplicate rows")
        
        num_duplicates = df.duplicated().sum()
        num_fully_duplicated = len(df[df.duplicated(keep=False)])
        
        result = {
            "is_valid": num_duplicates == 0,
            "num_duplicates": int(num_duplicates),
            "num_fully_duplicated": int(num_fully_duplicated),
            "percentage": float((num_duplicates / len(df) * 100) if len(df) > 0 else 0),
        }
        
        if result["is_valid"]:
            self.logger.info("[OK] No duplicate rows found")
        else:
            self.logger.warning(f"[WARNING] Found {num_duplicates} duplicate rows ({result['percentage']:.2f}%)")
        
        return result
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing values in the dataset.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary with missing value analysis results.
        """
        self.logger.info("Checking for missing values")
        
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df) * 100) if len(df) > 0 else 0
        
        total_missing = missing_counts.sum()
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        
        result = {
            "is_valid": total_missing == 0,
            "total_missing": int(total_missing),
            "columns_with_missing": {k: int(v) for k, v in cols_with_missing.items()},
            "missing_percentage_by_column": {
                col: float(pct)
                for col, pct in missing_percentage[missing_percentage > 0].items()
            },
            "total_missing_percentage": float(
                (total_missing / (len(df) * len(df.columns)) * 100)
                if len(df) > 0 else 0
            ),
        }
        
        if result["is_valid"]:
            self.logger.info("[OK] No missing values found")
        else:
            self.logger.warning(f"[WARNING] Found {total_missing} missing values")
            for col, count in cols_with_missing.items():
                self.logger.warning(f"  - {col}: {count} missing ({missing_percentage[col]:.2f}%)")
        
        return result
    
    def check_infinite_values(self, df: pd.DataFrame) -> Dict:
        """
        Check for infinite values in the dataset.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary with infinite value analysis results.
        """
        self.logger.info("Checking for infinite values")
        
        infinite_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = int(inf_count)
        
        total_infinite = sum(infinite_counts.values())
        
        result = {
            "is_valid": total_infinite == 0,
            "total_infinite": total_infinite,
            "columns_with_infinite": infinite_counts,
        }
        
        if result["is_valid"]:
            self.logger.info("[OK] No infinite values found")
        else:
            self.logger.warning(f"[WARNING] Found {total_infinite} infinite values")
            for col, count in infinite_counts.items():
                self.logger.warning(f"  - {col}: {count} infinite values")
        
        return result
    
    def check_data_types(self, df: pd.DataFrame) -> Dict:
        """
        Check data types against expected schema.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary with data type validation results.
        """
        self.logger.info("Checking data types")
        
        mismatches = {}
        for col in df.columns:
            if col in self.schema.expected_dtypes:
                expected = self.schema.expected_dtypes[col]
                actual = str(df[col].dtype)
                if expected != actual:
                    mismatches[col] = {
                        "expected": expected,
                        "actual": actual
                    }
        
        result = {
            "is_valid": len(mismatches) == 0,
            "mismatches": mismatches,
            "current_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        
        if result["is_valid"]:
            self.logger.info("[OK] All data types match expected schema")
        else:
            self.logger.warning(f"[WARNING] Found {len(mismatches)} data type mismatches")
            for col, mismatch in mismatches.items():
                self.logger.warning(
                    f"  - {col}: expected {mismatch['expected']}, got {mismatch['actual']}"
                )
        
        return result
    
    def check_schema_compliance(self, df: pd.DataFrame) -> Dict:
        """
        Check schema compliance.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary with schema compliance results.
        """
        self.logger.info("Checking schema compliance")
        
        is_valid, errors = self.schema.validate_schema(df)
        
        result = {
            "is_valid": is_valid,
            "errors": errors,
        }
        
        if result["is_valid"]:
            self.logger.info("[OK] Schema compliance check passed")
        else:
            self.logger.error("[FAIL] Schema compliance check failed")
            for error in errors:
                self.logger.error(f"  - {error}")
        
        return result
    
    def get_validation_summary(self) -> Dict:
        """
        Get a summary of validation results.
        
        Returns:
            Dictionary with validation summary.
        """
        if not self.validation_results:
            return {"status": "No validation performed"}
        
        all_valid = all(
            result.get("is_valid", False)
            for result in self.validation_results.values()
        )
        
        return {
            "status": "PASSED" if all_valid else "FAILED",
            "results": self.validation_results,
        }
