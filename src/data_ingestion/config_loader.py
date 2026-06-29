"""
Configuration loader for data ingestion pipeline.

Loads configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """
    Loads and manages configuration for the data ingestion pipeline.
    """
    
    @staticmethod
    def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
        """
        file_path = Path(config_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def get_data_ingestion_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data ingestion configuration.
        
        Args:
            config: Full configuration dictionary.
            
        Returns:
            Data ingestion configuration subset.
        """
        return config.get("data_ingestion", {})
    
    @staticmethod
    def get_schema_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract schema configuration.
        
        Args:
            config: Full configuration dictionary.
            
        Returns:
            Schema configuration subset.
        """
        return config.get("schema", {})
