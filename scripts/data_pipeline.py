#!/usr/bin/env python3
"""
Data Pipeline Module for Phoenix Rent Growth Forecasting

Purpose: Handle data loading, validation, quality checks, and updates for
         the automated forecast system

Usage:
    from data_pipeline import DataPipeline

    pipeline = DataPipeline()
    df = pipeline.load_and_validate()
    pipeline.check_for_updates()

Author: Generated for automated forecast system
Version: 1.0
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
CONFIG_DIR = BASE_DIR / 'config'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data file path
PHOENIX_DATA_FILE = DATA_DIR / 'phoenix_modeling_dataset.csv'

# Pipeline configuration
PIPELINE_CONFIG_FILE = CONFIG_DIR / 'data_pipeline_config.json'
PIPELINE_EXAMPLE_FILE = CONFIG_DIR / 'data_pipeline_config.example.json'

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_PIPELINE_CONFIG = {
    'data_sources': {
        'primary': str(PHOENIX_DATA_FILE),
        'backup': None,
        'description': 'Primary data source is CoStar export CSV'
    },
    'data_quality_checks': {
        'max_missing_percentage': 30.0,
        'max_duplicate_dates': 0,
        'min_data_points': 30,
        'extreme_value_threshold': 30.0,
        'required_columns': [
            'date',
            'rent_growth_yoy'
        ]
    },
    'update_detection': {
        'enabled': True,
        'check_file_modified_time': True,
        'check_new_quarters': True,
        'expected_update_frequency_days': 90
    },
    'data_transformations': {
        'date_column': 'date',
        'date_format': '%Y-%m-%d',
        'sort_by_date': True,
        'filter_market': None
    }
}

# ============================================================================
# DATA PIPELINE CLASS
# ============================================================================

class DataPipeline:
    """
    Handle data loading, validation, and quality checks
    """

    def __init__(self, config_file=None, log_file=None):
        """
        Initialize data pipeline

        Args:
            config_file: Path to configuration JSON file
            log_file: Path to log file for pipeline operations
        """
        self.config_file = config_file or PIPELINE_CONFIG_FILE
        self.config = self._load_config()
        self.log_file = log_file or LOGS_DIR / f'data_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.quality_issues = []

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Error loading pipeline config: {e}")

        return DEFAULT_PIPELINE_CONFIG.copy()

    def log(self, message: str, level: str = 'INFO'):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"

        print(log_entry)

        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_data(self, data_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            data_file: Path to data file (uses config default if None)

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if data_file is None:
            data_file = Path(self.config['data_sources']['primary'])

        self.log(f"Loading data from {data_file}")

        if not data_file.exists():
            # Try backup source if available
            backup = self.config['data_sources'].get('backup')
            if backup and Path(backup).exists():
                self.log(f"Primary source not found, trying backup: {backup}", level='WARNING')
                data_file = Path(backup)
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load CSV
        df = pd.read_csv(data_file)
        self.log(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        return df

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to raw data

        Args:
            df: Raw DataFrame

        Returns:
            Transformed DataFrame
        """
        self.log("Applying data transformations...")

        transformations = self.config.get('data_transformations', {})

        # 1. Convert date column
        date_col = transformations.get('date_column', 'date')
        date_format = transformations.get('date_format', '%Y-%m-%d')

        if date_col in df.columns:
            # Always ensure date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                self.log(f"   ✅ Converted {date_col} to datetime")
            else:
                self.log(f"   ✅ {date_col} already in datetime format")

            # Ensure 'date' column exists for consistency
            if date_col != 'date':
                df['date'] = df[date_col]
        else:
            self.log(f"   ⚠️  Date column {date_col} not found", level='WARNING')

        # 2. Filter by market if specified
        filter_market = transformations.get('filter_market')
        if filter_market and 'market_name' in df.columns:
            original_len = len(df)
            df = df[df['market_name'] == filter_market].copy()
            self.log(f"   ✅ Filtered to {filter_market} market: {len(df)} rows (was {original_len})")

        # 3. Sort by date if requested
        if transformations.get('sort_by_date', True) and 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            self.log(f"   ✅ Sorted by date")

        # 4. Drop rows with missing critical data
        if 'rent_growth_yoy' in df.columns:
            original_len = len(df)
            df = df.dropna(subset=['rent_growth_yoy'])
            if len(df) < original_len:
                self.log(f"   ℹ️  Dropped {original_len - len(df)} rows with missing rent_growth_yoy")

        return df

    # ========================================================================
    # DATA QUALITY CHECKS
    # ========================================================================

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Perform comprehensive data quality checks

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, quality_issues)
        """
        self.log("Performing data quality checks...")
        self.quality_issues = []

        quality_config = self.config.get('data_quality_checks', {})

        # 1. Check required columns
        required_cols = quality_config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.quality_issues.append({
                'severity': 'CRITICAL',
                'check': 'required_columns',
                'message': f"Missing required columns: {missing_cols}"
            })

        # 2. Check minimum data points
        min_data_points = quality_config.get('min_data_points', 30)
        if len(df) < min_data_points:
            self.quality_issues.append({
                'severity': 'WARNING',
                'check': 'min_data_points',
                'message': f"Only {len(df)} data points (minimum: {min_data_points})"
            })

        # 3. Check for duplicate dates
        if 'date' in df.columns:
            duplicates = df['date'].duplicated().sum()
            max_duplicates = quality_config.get('max_duplicate_dates', 0)
            if duplicates > max_duplicates:
                self.quality_issues.append({
                    'severity': 'WARNING',
                    'check': 'duplicate_dates',
                    'message': f"Found {duplicates} duplicate dates (max allowed: {max_duplicates})"
                })

        # 4. Check missing value percentage
        if 'rent_growth_yoy' in df.columns:
            missing_pct = (df['rent_growth_yoy'].isna().sum() / len(df)) * 100
            max_missing = quality_config.get('max_missing_percentage', 30.0)
            if missing_pct > max_missing:
                self.quality_issues.append({
                    'severity': 'WARNING',
                    'check': 'missing_values',
                    'message': f"Missing values: {missing_pct:.1f}% (max allowed: {max_missing}%)"
                })

        # 5. Check for extreme values
        if 'rent_growth_yoy' in df.columns:
            extreme_threshold = quality_config.get('extreme_value_threshold', 30.0)
            extreme_values = df[df['rent_growth_yoy'].abs() > extreme_threshold]
            if len(extreme_values) > 0:
                self.quality_issues.append({
                    'severity': 'INFO',
                    'check': 'extreme_values',
                    'message': f"Found {len(extreme_values)} extreme values (>±{extreme_threshold}%)"
                })

        # 6. Check date range continuity
        if 'date' in df.columns:
            date_diffs = df['date'].diff().dt.days
            expected_diff = 90  # Quarterly data
            irregular = date_diffs[(date_diffs > 100) | (date_diffs < 80)]
            if len(irregular) > 0:
                self.quality_issues.append({
                    'severity': 'INFO',
                    'check': 'date_continuity',
                    'message': f"Found {len(irregular)} irregular date intervals"
                })

        # Report results
        is_valid = not any(issue['severity'] == 'CRITICAL' for issue in self.quality_issues)

        if is_valid:
            self.log(f"✅ Data quality checks passed ({len(self.quality_issues)} warnings/info)")
        else:
            self.log(f"❌ Data quality checks FAILED ({len(self.quality_issues)} issues)", level='ERROR')

        for issue in self.quality_issues:
            level = 'WARNING' if issue['severity'] != 'INFO' else 'INFO'
            self.log(f"   [{issue['severity']}] {issue['check']}: {issue['message']}", level=level)

        return is_valid, self.quality_issues

    # ========================================================================
    # UPDATE DETECTION
    # ========================================================================

    def check_for_updates(self, data_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Check if new data is available

        Args:
            data_file: Path to data file (uses config default if None)

        Returns:
            Dict with update status information
        """
        if data_file is None:
            data_file = Path(self.config['data_sources']['primary'])

        self.log("Checking for data updates...")

        update_info = {
            'updates_available': False,
            'last_modified': None,
            'days_since_update': None,
            'checks_performed': []
        }

        update_config = self.config.get('update_detection', {})

        if not update_config.get('enabled', True):
            self.log("   Update detection disabled in configuration")
            return update_info

        # 1. Check file modified time
        if update_config.get('check_file_modified_time', True):
            if data_file.exists():
                last_modified = datetime.fromtimestamp(data_file.stat().st_mtime)
                days_since_update = (datetime.now() - last_modified).days

                update_info['last_modified'] = last_modified.isoformat()
                update_info['days_since_update'] = days_since_update
                update_info['checks_performed'].append('file_modified_time')

                expected_freq = update_config.get('expected_update_frequency_days', 90)
                if days_since_update > expected_freq:
                    self.log(f"   ⚠️  Data file last modified {days_since_update} days ago (expected: every {expected_freq} days)", level='WARNING')
                    update_info['updates_available'] = True
                else:
                    self.log(f"   ✅ Data file modified {days_since_update} days ago (within expected {expected_freq} days)")

        # 2. Check for new quarters in data
        if update_config.get('check_new_quarters', True):
            df = self.load_data(data_file)
            df = self.transform_data(df)

            if 'date' in df.columns:
                latest_date = df['date'].max()
                quarters_old = (datetime.now() - latest_date).days / 90

                update_info['latest_data_date'] = latest_date.isoformat()
                update_info['quarters_old'] = round(quarters_old, 1)
                update_info['checks_performed'].append('new_quarters')

                if quarters_old > 1.5:
                    self.log(f"   ⚠️  Latest data is {quarters_old:.1f} quarters old (date: {latest_date.date()})", level='WARNING')
                    update_info['updates_available'] = True
                else:
                    self.log(f"   ✅ Latest data is {quarters_old:.1f} quarters old (date: {latest_date.date()})")

        return update_info

    # ========================================================================
    # MAIN PIPELINE METHODS
    # ========================================================================

    def load_and_validate(self, data_file: Optional[Path] = None) -> Tuple[pd.DataFrame, bool, List[Dict[str, Any]]]:
        """
        Main pipeline: Load, transform, and validate data

        Args:
            data_file: Path to data file (uses config default if None)

        Returns:
            Tuple of (dataframe, is_valid, quality_issues)
        """
        self.log("=" * 80)
        self.log("DATA PIPELINE EXECUTION")
        self.log("=" * 80)

        # Load data
        df = self.load_data(data_file)

        # Transform data
        df = self.transform_data(df)

        # Validate data quality
        is_valid, quality_issues = self.validate_data_quality(df)

        self.log("=" * 80)
        self.log(f"PIPELINE COMPLETE: {'SUCCESS' if is_valid else 'WARNINGS'}")
        self.log("=" * 80)

        return df, is_valid, quality_issues

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for loaded data

        Args:
            df: DataFrame to summarize

        Returns:
            Dict with summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns)
        }

        if 'date' in df.columns:
            summary['date_range'] = {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat(),
                'quarters': len(df['date'].unique())
            }

        if 'rent_growth_yoy' in df.columns:
            summary['rent_growth_stats'] = {
                'mean': float(df['rent_growth_yoy'].mean()),
                'std': float(df['rent_growth_yoy'].std()),
                'min': float(df['rent_growth_yoy'].min()),
                'max': float(df['rent_growth_yoy'].max()),
                'missing': int(df['rent_growth_yoy'].isna().sum())
            }

        return summary

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Data Pipeline for Phoenix Rent Growth Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and validate data
  python3 data_pipeline.py --validate

  # Check for updates
  python3 data_pipeline.py --check-updates

  # Get data summary
  python3 data_pipeline.py --summary

  # Run full pipeline with custom data file
  python3 data_pipeline.py --validate --file /path/to/data.csv
        """
    )

    parser.add_argument('--validate', action='store_true',
                       help='Load and validate data')
    parser.add_argument('--check-updates', action='store_true',
                       help='Check for data updates')
    parser.add_argument('--summary', action='store_true',
                       help='Generate data summary')
    parser.add_argument('--file', type=str,
                       help='Custom data file path')

    args = parser.parse_args()

    pipeline = DataPipeline()

    data_file = Path(args.file) if args.file else None

    if args.validate:
        df, is_valid, issues = pipeline.load_and_validate(data_file)
        print(f"\nValidation Result: {'PASS' if is_valid else 'WARNINGS'}")
        print(f"Quality Issues: {len(issues)}")

    elif args.check_updates:
        update_info = pipeline.check_for_updates(data_file)
        print("\n" + "=" * 80)
        print("UPDATE CHECK RESULTS")
        print("=" * 80)
        print(f"Updates Available: {update_info['updates_available']}")
        if update_info.get('last_modified'):
            print(f"Last Modified: {update_info['last_modified']}")
            print(f"Days Since Update: {update_info['days_since_update']}")
        if update_info.get('latest_data_date'):
            print(f"Latest Data Date: {update_info['latest_data_date']}")
            print(f"Quarters Old: {update_info['quarters_old']}")
        print("=" * 80)

    elif args.summary:
        df, _, _ = pipeline.load_and_validate(data_file)
        summary = pipeline.get_data_summary(df)

        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        print(json.dumps(summary, indent=2))
        print("=" * 80)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
