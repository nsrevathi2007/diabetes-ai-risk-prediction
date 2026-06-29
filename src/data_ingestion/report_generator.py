"""
Report generation module for data quality analysis.

This module generates comprehensive data quality reports in both
Markdown and JSON formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ReportGenerator:
    """
    Generates data quality reports in multiple formats.
    
    Creates comprehensive reports including:
    - Dataset statistics
    - Data types summary
    - Missing value analysis
    - Duplicate analysis
    - Validation status
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_reports(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        output_dir: str = "docs",
        markdown_filename: str = "data_quality_report.md",
        json_filename: str = "data_quality_report.json",
        is_valid: bool = True,
    ) -> Dict[str, Path]:
        """
        Generate both markdown and JSON reports.
        
        Args:
            df: DataFrame to analyze.
            validation_results: Results from data validation.
            output_dir: Directory to save reports.
            markdown_filename: Name of markdown report file.
            json_filename: Name of JSON report file.
            is_valid: Overall validation status.
            
        Returns:
            Dictionary with paths to generated reports.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        md_path = output_path / markdown_filename
        self._generate_markdown_report(df, validation_results, md_path, is_valid)
        
        # Generate JSON report
        json_path = output_path / json_filename
        self._generate_json_report(df, validation_results, json_path, is_valid)
        
        return {
            "markdown": md_path,
            "json": json_path,
        }
    
    def _generate_markdown_report(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        output_path: Path,
        is_valid: bool = True
    ) -> None:
        """
        Generate markdown report.
        
        Args:
            df: DataFrame to analyze.
            validation_results: Validation results.
            output_path: Path to save the report.
            is_valid: Overall validation status.
        """
        self.logger.info(f"Generating markdown report: {output_path}")
        
        report_lines = []
        
        # Header
        report_lines.append("# Data Quality Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset Overview
        report_lines.append("## Dataset Overview\n")
        report_lines.append(f"- **Number of Rows:** {len(df):,}")
        report_lines.append(f"- **Number of Columns:** {len(df.columns)}")
        report_lines.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        report_lines.append(f"- **Shape:** {df.shape}\n")
        
        # Validation Status
        report_lines.append("## Validation Status\n")
        status = "[PASSED]" if is_valid else "[FAILED]"
        report_lines.append(f"**Overall Status:** {status}\n")
        
        # Detailed Validation Results
        report_lines.append("### Duplicate Rows")
        dup_result = validation_results.get("duplicates", {})
        report_lines.append(f"- **Status:** {'[OK]' if dup_result.get('is_valid') else '[FAIL]'}")
        report_lines.append(f"- **Number of Duplicates:** {dup_result.get('num_duplicates', 0)}")
        report_lines.append(f"- **Percentage:** {dup_result.get('percentage', 0):.2f}%\n")
        
        # Missing Values
        report_lines.append("### Missing Values\n")
        miss_result = validation_results.get("missing_values", {})
        report_lines.append(f"- **Status:** {'[OK]' if miss_result.get('is_valid') else '[FAIL]'}")
        report_lines.append(f"- **Total Missing Values:** {miss_result.get('total_missing', 0)}")
        report_lines.append(f"- **Total Missing Percentage:** {miss_result.get('total_missing_percentage', 0):.2f}%")
        
        if miss_result.get("columns_with_missing"):
            report_lines.append("- **Missing by Column:**")
            for col, count in miss_result["columns_with_missing"].items():
                pct = miss_result["missing_percentage_by_column"].get(col, 0)
                report_lines.append(f"  - {col}: {count} ({pct:.2f}%)")
        report_lines.append("")
        
        # Infinite Values
        report_lines.append("### Infinite Values\n")
        inf_result = validation_results.get("infinite_values", {})
        report_lines.append(f"- **Status:** {'[OK]' if inf_result.get('is_valid') else '[FAIL]'}")
        report_lines.append(f"- **Total Infinite Values:** {inf_result.get('total_infinite', 0)}")
        
        if inf_result.get("columns_with_infinite"):
            report_lines.append("- **Infinite by Column:**")
            for col, count in inf_result["columns_with_infinite"].items():
                report_lines.append(f"  - {col}: {count}")
        report_lines.append("")
        
        # Data Types
        report_lines.append("### Data Types\n")
        dtype_result = validation_results.get("data_types", {})
        report_lines.append(f"- **Status:** {'[OK]' if dtype_result.get('is_valid') else '[FAIL]'}")
        
        if dtype_result.get("mismatches"):
            report_lines.append("- **Type Mismatches:**")
            for col, mismatch in dtype_result["mismatches"].items():
                report_lines.append(
                    f"  - {col}: Expected {mismatch['expected']}, Got {mismatch['actual']}"
                )
        else:
            report_lines.append("- **All data types match expected schema**")
        report_lines.append("")
        
        # Column Statistics
        report_lines.append("## Column Statistics\n")
        report_lines.append("| Column | Type | Unique | Min | Max | Mean |")
        report_lines.append("|--------|------|--------|-----|-----|------|")
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            unique = df[col].nunique()
            
            if df[col].dtype in ['float64', 'int64']:
                min_val = f"{df[col].min():.2f}"
                max_val = f"{df[col].max():.2f}"
                mean_val = f"{df[col].mean():.2f}"
            else:
                min_val = "-"
                max_val = "-"
                mean_val = "-"
            
            report_lines.append(
                f"| {col} | {col_type} | {unique} | {min_val} | {max_val} | {mean_val} |"
            )
        
        report_lines.append("")
        
        # Write report
        report_content = "\n".join(report_lines)
        output_path.write_text(report_content, encoding='utf-8')
        self.logger.info(f"Markdown report saved to {output_path}")
    
    def _generate_json_report(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        output_path: Path,
        is_valid: bool = True
    ) -> None:
        """
        Generate JSON report.
        
        Args:
            df: DataFrame to analyze.
            validation_results: Validation results.
            output_path: Path to save the report.
            is_valid: Overall validation status.
        """
        self.logger.info(f"Generating JSON report: {output_path}")
        
        # Calculate statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        stats = {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
            }
            for col in numeric_cols
        }
        
        report_dict: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "dataset_overview": {
                "num_rows": int(len(df)),
                "num_columns": len(df.columns),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
                "shape": list(df.shape),
            },
            "column_info": {
                col: {
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                }
                for col in df.columns
            },
            "statistics": stats,
            "validation_results": self._make_json_serializable(validation_results),
            "overall_status": "PASSED" if is_valid else "FAILED",
        }
        
        # Write JSON report
        output_path.write_text(json.dumps(report_dict, indent=2), encoding='utf-8')
        self.logger.info(f"JSON report saved to {output_path}")
    
    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """
        Convert numpy/pandas types to JSON-serializable types.
        
        Args:
            obj: Object to convert.
            
        Returns:
            JSON-serializable object.
        """
        if isinstance(obj, dict):
            return {k: ReportGenerator._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReportGenerator._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
