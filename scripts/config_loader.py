#!/usr/bin/env python3
"""
Configuration Loader for Phoenix Rent Growth Forecasting

Purpose: Load and manage custom configuration files for validation thresholds,
         email alerts, and other system settings

Usage:
    from config_loader import load_validation_thresholds, get_config

    # Load validation thresholds
    thresholds = load_validation_thresholds()

    # Access specific threshold
    sarima_max = get_config('validation_thresholds', 'sarima_max_prediction', default=10.0)

Author: Generated for automated forecast system
Version: 1.0
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'config'

# Configuration file paths
VALIDATION_CONFIG_FILE = CONFIG_DIR / 'validation_thresholds.json'
VALIDATION_EXAMPLE_FILE = CONFIG_DIR / 'validation_thresholds.example.json'
EMAIL_CONFIG_FILE = CONFIG_DIR / 'email_config.json'
EMAIL_EXAMPLE_FILE = CONFIG_DIR / 'email_config.example.json'

# ============================================================================
# DEFAULT PRODUCTION CONFIGURATION
# ============================================================================

DEFAULT_PRODUCTION_CONFIG = {
    'sarima': {
        'order': (1, 1, 2),
        'seasonal_order': (0, 0, 1, 4),
        'explanation': 'Production configuration - avoids explosive predictions'
    },
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 6,
        'min_data_in_leaf': 10,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    },
    'ridge': {
        'alphas': [0.1, 1.0, 10.0, 100.0, 1000.0],
        'cv': 5,
        'explanation': 'Production likely uses alpha=10.0 for strong regularization'
    },
    'early_stopping': {
        'rounds': 50,
        'explanation': 'Critical for preventing overfitting'
    },
    'validation_thresholds': {
        'sarima_max_prediction': 10.0,
        'component_correlation_min': -0.5,
        'test_train_rmse_ratio': 2.0,
        'ridge_alpha_min': 1.0,
        'forecast_revision_alert': 1.0
    }
}

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_json_config(config_file: Path, example_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load JSON configuration file with fallback to example

    Args:
        config_file: Primary configuration file path
        example_file: Example configuration file path (fallback)

    Returns:
        Configuration dictionary
    """
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Error loading {config_file}: {e}")
            if example_file and example_file.exists():
                print(f"Using example config from {example_file}")
                with open(example_file, 'r') as f:
                    return json.load(f)
    elif example_file and example_file.exists():
        print(f"Warning: Config not found at {config_file}")
        print(f"Copy {example_file} to {config_file} and customize")
        return {}

    return {}

def load_validation_thresholds() -> Dict[str, float]:
    """
    Load validation thresholds from config file or use defaults

    Returns:
        Dictionary of threshold values
    """
    config = load_json_config(VALIDATION_CONFIG_FILE, VALIDATION_EXAMPLE_FILE)

    if config:
        # Extract threshold values from config structure
        thresholds = {}
        for category in ['validation_thresholds', 'data_quality_thresholds',
                        'model_performance_thresholds', 'forecast_quality_thresholds']:
            if category in config:
                for key, value in config[category].items():
                    if isinstance(value, dict) and 'value' in value:
                        thresholds[key] = value['value']
                    else:
                        thresholds[key] = value

        # Merge with defaults and return
        default_thresholds = DEFAULT_PRODUCTION_CONFIG['validation_thresholds'].copy()
        default_thresholds.update(thresholds)
        return default_thresholds
    else:
        # Use default thresholds
        return DEFAULT_PRODUCTION_CONFIG['validation_thresholds'].copy()

def load_production_config() -> Dict[str, Any]:
    """
    Load complete production configuration with custom thresholds applied

    Returns:
        Complete production configuration dictionary
    """
    config = DEFAULT_PRODUCTION_CONFIG.copy()

    # Load custom validation thresholds if available
    custom_thresholds = load_validation_thresholds()
    config['validation_thresholds'] = custom_thresholds

    return config

def get_config(category: str, key: str, default: Any = None, config_file: str = 'validation') -> Any:
    """
    Get specific configuration value

    Args:
        category: Configuration category (e.g., 'validation_thresholds')
        key: Configuration key within category
        default: Default value if not found
        config_file: Which config file to load ('validation' or 'email')

    Returns:
        Configuration value or default
    """
    if config_file == 'validation':
        config = load_json_config(VALIDATION_CONFIG_FILE, VALIDATION_EXAMPLE_FILE)
    elif config_file == 'email':
        config = load_json_config(EMAIL_CONFIG_FILE, EMAIL_EXAMPLE_FILE)
    else:
        return default

    if not config:
        return default

    # Navigate nested config
    if category in config:
        category_config = config[category]
        if key in category_config:
            value = category_config[key]
            # Handle value dict format
            if isinstance(value, dict) and 'value' in value:
                return value['value']
            return value

    return default

def get_threshold_description(key: str) -> Optional[str]:
    """
    Get description for a specific threshold

    Args:
        key: Threshold key

    Returns:
        Description string or None
    """
    config = load_json_config(VALIDATION_CONFIG_FILE, VALIDATION_EXAMPLE_FILE)

    if not config:
        return None

    # Search all threshold categories
    for category in ['validation_thresholds', 'data_quality_thresholds',
                    'model_performance_thresholds', 'forecast_quality_thresholds']:
        if category in config and key in config[category]:
            threshold_config = config[category][key]
            if isinstance(threshold_config, dict):
                return threshold_config.get('description')

    return None

def print_config_summary():
    """Print summary of current configuration"""
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)

    # Production config
    print("\n📋 Production Configuration:")
    config = load_production_config()
    print(f"   SARIMA: {config['sarima']['order']} / {config['sarima']['seasonal_order']}")
    print(f"   Ridge Alphas: {config['ridge']['alphas']}")
    print(f"   Early Stopping: {config['early_stopping']['rounds']} rounds")

    # Validation thresholds
    print("\n🎯 Validation Thresholds:")
    thresholds = config['validation_thresholds']
    for key, value in thresholds.items():
        desc = get_threshold_description(key)
        if desc:
            print(f"   {key}: {value}")
            print(f"      → {desc}")
        else:
            print(f"   {key}: {value}")

    # Config file status
    print("\n📁 Configuration Files:")
    files = [
        ("Validation Thresholds", VALIDATION_CONFIG_FILE, VALIDATION_EXAMPLE_FILE),
        ("Email Alerts", EMAIL_CONFIG_FILE, EMAIL_EXAMPLE_FILE)
    ]

    for name, config_file, example_file in files:
        if config_file.exists():
            print(f"   ✅ {name}: {config_file.name}")
        else:
            print(f"   ⚠️  {name}: Not configured")
            print(f"      Copy {example_file.name} to {config_file.name}")

    print("=" * 80)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Configuration Loader for Phoenix Rent Growth Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--summary', action='store_true',
                       help='Print configuration summary')
    parser.add_argument('--test-load', action='store_true',
                       help='Test loading all configuration files')
    parser.add_argument('--get', nargs=2, metavar=('CATEGORY', 'KEY'),
                       help='Get specific configuration value')

    args = parser.parse_args()

    if args.summary:
        print_config_summary()
    elif args.test_load:
        print("Testing configuration loading...")
        config = load_production_config()
        print(f"✅ Loaded {len(config)} configuration categories")
        print(f"✅ Loaded {len(config['validation_thresholds'])} validation thresholds")
    elif args.get:
        category, key = args.get
        value = get_config(category, key)
        print(f"{category}.{key} = {value}")

        desc = get_threshold_description(key)
        if desc:
            print(f"Description: {desc}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
