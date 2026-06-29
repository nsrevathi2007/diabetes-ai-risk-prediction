"""
Data loading module for diabetes dataset.

This module handles loading CSV data from disk, validating file existence,
and performing basic data integrity checks.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """
    Handles loading and basic validation of diabetes dataset.
    
    This class is responsible for:
    - Verifying data file existence
    - Loading CSV files into pandas DataFrames
    - Performing basic integrity checks
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataLoader.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger is created.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or cannot be read.
        """
        file_path = Path(filepath)
        
        # Check file existence
        if not file_path.exists():
            self.logger.error(f"Data file not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.logger.info(f"Found data file: {filepath}")
        
        # Check file is not empty
        if file_path.stat().st_size == 0:
            self.logger.error(f"Data file is empty: {filepath}")
            raise ValueError(f"Data file is empty: {filepath}")
        
        self.logger.info(f"File size: {file_path.stat().st_size} bytes")
        
        # Load CSV
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read CSV file: {str(e)}")
            raise ValueError(f"Failed to read CSV file: {str(e)}")
    
    def verify_file_exists(self, filepath: str) -> bool:
        """
        Verify that a file exists.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            True if file exists, False otherwise.
        """
        file_path = Path(filepath)
        exists = file_path.exists()
        
        if exists:
            self.logger.info(f"File verification passed: {filepath}")
        else:
            self.logger.warning(f"File verification failed: {filepath}")
        
        return exists
    
    def check_empty_dataset(self, df: pd.DataFrame) -> bool:
        """
        Check if dataset is empty.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            True if dataset is empty, False otherwise.
        """
        is_empty = df.empty
        
        if is_empty:
            self.logger.warning("Dataset is empty")
        else:
            self.logger.info(f"Dataset is not empty. Shape: {df.shape}")
        
        return is_empty
