#!/usr/bin/env python
"""
Main runner script for the data ingestion and validation pipeline.

This script orchestrates the complete data ingestion workflow:
1. Load configuration
2. Setup logging
3. Load dataset
4. Validate dataset
5. Generate reports
6. Save cleaned dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.config_loader import ConfigLoader
from src.data_ingestion.data_loader import DataLoader
from src.data_ingestion.logging_config import setup_logging
from src.data_ingestion.report_generator import ReportGenerator
from src.data_ingestion.schema import DataSchemaConfig
from src.data_ingestion.validator import DataValidator


def main() -> int:
    """
    Main entry point for the data ingestion pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # ============================================================================
        # Step 1: Load Configuration
        # ============================================================================
        print("=" * 80)
        print("DATA INGESTION & VALIDATION PIPELINE")
        print("=" * 80)
        print("\n[1/5] Loading configuration...")
        
        config = ConfigLoader.load_config("configs/config.yaml")
        ingestion_config = ConfigLoader.get_data_ingestion_config(config)
        schema_config = ConfigLoader.get_schema_config(config)
        
        print("✓ Configuration loaded successfully")
        
        # ============================================================================
        # Step 2: Setup Logging
        # ============================================================================
        print("[2/5] Setting up logging...")
        
        logging_config = ingestion_config.get("logging", {})
        logger = setup_logging(
            log_dir=logging_config.get("log_dir", "logs"),
            log_file=logging_config.get("log_file", "data_ingestion.log"),
            log_level=logging_config.get("level", "INFO"),
            log_format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )
        
        logger.info("=" * 80)
        logger.info("DATA INGESTION & VALIDATION PIPELINE STARTED")
        logger.info("=" * 80)
        print("✓ Logging configured successfully")
        
        # ============================================================================
        # Step 3: Load Dataset
        # ============================================================================
        print("[3/5] Loading dataset...")
        
        data_loader = DataLoader(logger)
        raw_data_path = ingestion_config.get("raw_data_path", "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv")
        
        logger.info(f"Loading dataset from: {raw_data_path}")
        df = data_loader.load_dataset(raw_data_path)
        
        # Check for empty dataset
        if data_loader.check_empty_dataset(df):
            logger.error("Dataset is empty!")
            return 1
        
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Dataset loaded successfully: {df.shape}")
        
        # ============================================================================
        # Step 4: Validate Dataset
        # ============================================================================
        print("[4/5] Validating dataset...")
        
        # Create schema from config
        schema = DataSchemaConfig(
            target_column=schema_config.get("target_column", "Diabetes_binary"),
            expected_columns=schema_config.get("expected_columns", []),
            expected_dtypes=schema_config.get("expected_dtypes", {}),
        )
        
        validator = DataValidator(schema, logger)
        is_valid, validation_results = validator.validate_all(df)
        
        print(f"✓ Validation {'PASSED' if is_valid else 'FAILED'}")
        
        # ============================================================================
        # Step 5: Generate Reports
        # ============================================================================
        print("[5/5] Generating reports...")
        
        report_generator = ReportGenerator(logger)
        reporting_config = ingestion_config.get("reporting", {})
        
        report_paths = report_generator.generate_reports(
            df,
            validation_results,
            output_dir=reporting_config.get("output_dir", "docs"),
            markdown_filename=reporting_config.get("markdown_report", "data_quality_report.md"),
            json_filename=reporting_config.get("json_report", "data_quality_report.json"),
            is_valid=is_valid,
        )
        
        print(f"✓ Reports generated:")
        print(f"  - Markdown: {report_paths['markdown']}")
        print(f"  - JSON: {report_paths['json']}")
        
        # ============================================================================
        # Step 6: Save Cleaned Dataset
        # ============================================================================
        processed_data_path = ingestion_config.get("processed_data_path", "data/processed/diabetes_binary_health_indicators_cleaned.csv")
        
        logger.info(f"Saving cleaned dataset to: {processed_data_path}")
        
        # Create directory if it doesn't exist
        Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save without index
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Cleaned dataset saved successfully")
        
        print(f"✓ Cleaned dataset saved: {processed_data_path}")
        
        # ============================================================================
        # Summary
        # ============================================================================
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"✓ Dataset Shape: {df.shape}")
        print(f"✓ Validation Status: {'PASSED' if is_valid else 'FAILED'}")
        print(f"✓ Duplicates: {validation_results['duplicates']['num_duplicates']}")
        print(f"✓ Missing Values: {validation_results['missing_values']['total_missing']}")
        print(f"✓ Infinite Values: {validation_results['infinite_values']['total_infinite']}")
        print(f"✓ Data Type Mismatches: {len(validation_results['data_types']['mismatches'])}")
        print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        logger.info("DATA INGESTION & VALIDATION PIPELINE COMPLETED SUCCESSFULLY")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        if 'logger' in locals():
            logger.exception("Pipeline execution failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
