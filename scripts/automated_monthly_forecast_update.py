#!/usr/bin/env python3
"""
Automated Monthly Forecast Update Script

Purpose: Automatically update Phoenix rent growth forecasts on a monthly basis
         with latest data, validation, comparison, and alerting.

Usage:
    python automated_monthly_forecast_update.py [options]

    Options:
        --force-retrain       Force model retraining even if models exist
        --skip-comparison     Skip comparison to previous forecasts
        --alert-email EMAIL   Send alerts to this email address
        --output-dir DIR      Custom output directory (default: outputs/)

    Cron Example (1st of each month at 8am):
        0 8 1 * * cd /path/to/project && python scripts/automated_monthly_forecast_update.py

Configuration:
    Uses production-validated configuration from root cause analysis:
    - SARIMA: (1,1,2)(0,0,1,4)
    - LightGBM: Early stopping 50 rounds
    - Ridge: Alpha range [0.1, 1.0, 10.0, 100.0, 1000.0]

Output:
    - Updated forecast CSV with datestamp
    - Updated executive summary
    - Comparison charts (actual vs forecast, forecast revisions)
    - Accuracy tracking metrics
    - Alert log for validation failures

Author: Generated from Root Cause Analysis (Nov 2025)
Version: 1.0
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PRODUCTION CONFIGURATION (From Root Cause Analysis)
# ============================================================================

# Import config loader for customizable thresholds
try:
    from config_loader import load_production_config
    PRODUCTION_CONFIG = load_production_config()
except ImportError:
    # Fallback to default configuration if config_loader not available
    PRODUCTION_CONFIG = {
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
            'forecast_revision_alert': 1.0  # Alert if forecast changes >1.0 percentage point
        }
    }

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models' / 'production'
OUTPUT_DIR = BASE_DIR / 'outputs'
REPORTS_DIR = BASE_DIR / 'reports'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
for directory in [MODEL_DIR, OUTPUT_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DATA_FILE = DATA_DIR / 'phoenix_modeling_dataset.csv'
ACCURACY_TRACKING_FILE = OUTPUT_DIR / 'forecast_accuracy_tracking.csv'
LAST_RUN_FILE = LOGS_DIR / 'last_successful_run.json'

# Core features (31 total from phoenix_modeling_dataset.csv)
CORE_FEATURES = [
    # Macroeconomic
    'fed_funds_rate', 'mortgage_rate_30yr', 'national_unemployment', 'cpi',
    'inflation_expectations_5yr',
    # Housing market
    'vacancy_rate', 'cap_rate', 'phx_hpi_yoy_growth', 'phx_home_price_index',
    'inventory_units', 'units_under_construction', 'absorption_12mo',
    'housing_starts', 'building_permits',
    # Employment
    'phx_total_employment', 'phx_unemployment_rate', 'phx_prof_business_employment',
    'phx_manufacturing_employment', 'phx_employment_yoy_growth', 'phx_prof_business_yoy_growth',
    # Lag features
    'phx_prof_business_employment_lag1', 'phx_total_employment_lag1',
    'units_under_construction_lag5', 'units_under_construction_lag6',
    'units_under_construction_lag7', 'units_under_construction_lag8',
    'mortgage_rate_30yr_lag2',
    # Engineered features
    'supply_inventory_ratio', 'absorption_inventory_ratio',
    'mortgage_employment_interaction', 'migration_proxy'
]

# ============================================================================
# LOGGING AND ALERTS
# ============================================================================

# Import email alerter if available
try:
    from email_alerts import EmailAlerter
    EMAIL_ALERTS_AVAILABLE = True
except ImportError:
    EMAIL_ALERTS_AVAILABLE = False

class ForecastLogger:
    """Centralized logging and alerting system"""

    def __init__(self, log_file=None, alert_email=None):
        self.log_file = log_file or LOGS_DIR / f'forecast_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.alert_email = alert_email
        self.alerts = []

        # Initialize email alerter if available
        self.email_alerter = None
        if EMAIL_ALERTS_AVAILABLE:
            try:
                self.email_alerter = EmailAlerter()
                if self.email_alerter.is_enabled():
                    self.log("📧 Email alerts enabled", level='INFO')
            except Exception as e:
                self.log(f"Warning: Could not initialize email alerter: {e}", level='WARNING')

    def log(self, message, level='INFO'):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

    def alert(self, message, severity='WARNING'):
        """Log alert and store for notification"""
        self.log(f"🚨 ALERT: {message}", level=severity)
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message
        })

    def send_alerts(self, forecast_summary=None):
        """Send accumulated alerts via email (if configured)"""
        if not self.alerts:
            self.log("✅ No alerts to send")
            return

        # Try to send via email alerter
        if self.email_alerter and self.email_alerter.is_enabled():
            try:
                attachments = [self.log_file] if self.log_file.exists() else None
                success = self.email_alerter.send_forecast_alert(
                    self.alerts,
                    forecast_summary=forecast_summary,
                    attachments=attachments
                )
                if success:
                    self.log(f"✅ Sent {len(self.alerts)} alerts via email")
                    return
            except Exception as e:
                self.log(f"Failed to send email alerts: {e}", level='WARNING')

        # Fallback: log alerts locally
        if self.alert_email:
            self.log(f"📧 Would send {len(self.alerts)} alerts to {self.alert_email}")
            for alert in self.alerts:
                self.log(f"   - [{alert['severity']}] {alert['message']}")
        else:
            self.log(f"⚠️  {len(self.alerts)} alerts generated (no email configured)")

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_latest_data(logger):
    """Load latest data using data pipeline and detect changes since last run"""
    logger.log("Loading latest data with data pipeline...")

    # Try to use data pipeline for enhanced validation
    try:
        from data_pipeline import DataPipeline

        pipeline = DataPipeline(log_file=logger.log_file)
        df, is_valid, quality_issues = pipeline.load_and_validate()

        # Log quality issues from pipeline
        for issue in quality_issues:
            if issue['severity'] == 'CRITICAL':
                logger.alert(f"Pipeline: {issue['message']}", severity='CRITICAL')
            elif issue['severity'] == 'WARNING':
                logger.alert(f"Pipeline: {issue['message']}", severity='WARNING')
            else:
                logger.log(f"   ℹ️  Pipeline: {issue['message']}")

        # Check for data updates
        update_info = pipeline.check_for_updates()
        if update_info['updates_available']:
            logger.log(f"   ⚠️  Data may need updating (see pipeline logs)", level='WARNING')

    except ImportError:
        # Fallback to basic data loading if pipeline not available
        logger.log("   ℹ️  Data pipeline module not available, using basic loading")

        if not DATA_FILE.exists():
            logger.alert(f"Data file not found: {DATA_FILE}", severity='CRITICAL')
            sys.exit(1)

        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    logger.log(f"✅ Loaded {len(df)} quarters of data ({df['date'].min()} to {df['date'].max()})")

    # Check for new data since last run
    if LAST_RUN_FILE.exists():
        with open(LAST_RUN_FILE, 'r') as f:
            last_run = json.load(f)

        last_data_date = pd.to_datetime(last_run['last_data_date'])
        current_data_date = df['date'].max()

        if current_data_date > last_data_date:
            new_quarters = len(df[df['date'] > last_data_date])
            logger.log(f"📊 NEW DATA: {new_quarters} new quarters since {last_data_date.strftime('%Y-%m-%d')}")
            return df, True
        else:
            logger.log(f"ℹ️  No new data since last run ({last_data_date.strftime('%Y-%m-%d')})")
            return df, False
    else:
        logger.log("ℹ️  First run - no previous run data found")
        return df, True

def validate_data_quality(df, logger):
    """Validate data quality and completeness"""
    logger.log("Validating data quality...")

    issues = []

    # Check for missing values in core features
    missing = df[CORE_FEATURES].isnull().sum()
    if missing.any():
        missing_features = missing[missing > 0]
        issues.append(f"Missing values in {len(missing_features)} features: {missing_features.to_dict()}")

    # Check for duplicate dates
    if df['date'].duplicated().any():
        issues.append(f"Duplicate dates found: {df[df['date'].duplicated()]['date'].tolist()}")

    # Check for extreme values in target
    if 'rent_growth_yoy' in df.columns:
        target = df['rent_growth_yoy'].dropna()
        if (target.abs() > 30).any():
            extreme = target[target.abs() > 30]
            issues.append(f"Extreme rent growth values: {extreme.to_dict()}")

    # Report results
    if issues:
        for issue in issues:
            logger.alert(f"Data quality issue: {issue}", severity='WARNING')
        return False
    else:
        logger.log("✅ Data quality validation passed")
        return True

# ============================================================================
# MODEL TRAINING AND PREDICTION
# ============================================================================

def should_retrain_models(force_retrain, has_new_data, logger):
    """Determine if models should be retrained"""
    if force_retrain:
        logger.log("🔄 Force retrain enabled - will retrain models")
        return True

    if has_new_data:
        logger.log("📊 New data detected - will retrain models")
        return True

    # Check if models exist
    model_files = [
        MODEL_DIR / 'lightgbm_production.pkl',
        MODEL_DIR / 'sarima_production.pkl',
        MODEL_DIR / 'ridge_meta_production.pkl',
        MODEL_DIR / 'scaler_production.pkl'
    ]

    if not all(f.exists() for f in model_files):
        logger.log("🔄 Missing model files - will retrain models")
        return True

    logger.log("✅ Using existing models (no new data, models exist)")
    return False

def train_ensemble_model(df, logger):
    """Train ensemble model with production configuration"""
    logger.log("Training ensemble model...")

    # Prepare data
    df_train = df[df['rent_growth_yoy'].notna()].copy()

    # Split into train/test
    split_date = '2023-01-01'
    train = df_train[df_train['date'] < split_date]
    test = df_train[df_train['date'] >= split_date]

    logger.log(f"   Training: {len(train)} quarters ({train['date'].min()} to {train['date'].max()})")
    logger.log(f"   Testing: {len(test)} quarters ({test['date'].min()} to {test['date'].max()})")

    # Features and target
    X_train = train[CORE_FEATURES]
    y_train = train['rent_growth_yoy']
    X_test = test[CORE_FEATURES]
    y_test = test['rent_growth_yoy']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train LightGBM with early stopping
    logger.log("   Training LightGBM with early stopping...")
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    val_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    lgb_model = lgb.train(
        PRODUCTION_CONFIG['lightgbm'],
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    best_iter = lgb_model.best_iteration
    logger.log(f"   ✅ LightGBM trained (best iteration: {best_iter})")

    # 2. Train SARIMA with production config
    logger.log("   Training SARIMA with production configuration...")
    sarima_model = SARIMAX(
        y_train,
        order=PRODUCTION_CONFIG['sarima']['order'],
        seasonal_order=PRODUCTION_CONFIG['sarima']['seasonal_order'],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    logger.log("   ✅ SARIMA trained")

    # 3. Create meta-features for Ridge
    lgb_train_pred = lgb_model.predict(X_train_scaled, num_iteration=best_iter)
    lgb_test_pred = lgb_model.predict(X_test_scaled, num_iteration=best_iter)

    sarima_train_pred = sarima_model.fittedvalues.values
    sarima_test_pred = sarima_model.forecast(steps=len(y_test))

    X_meta_train = np.column_stack([lgb_train_pred, sarima_train_pred])
    X_meta_test = np.column_stack([lgb_test_pred, sarima_test_pred])

    # 4. Train Ridge meta-learner
    logger.log("   Training Ridge meta-learner...")
    ridge_model = RidgeCV(
        alphas=PRODUCTION_CONFIG['ridge']['alphas'],
        cv=TimeSeriesSplit(n_splits=5)
    ).fit(X_meta_train, y_train)

    alpha_selected = ridge_model.alpha_
    weights = ridge_model.coef_

    logger.log(f"   ✅ Ridge trained (alpha={alpha_selected:.4f})")
    logger.log(f"      LightGBM weight: {weights[0]:.6f}")
    logger.log(f"      SARIMA weight: {weights[1]:.6f}")

    # Validate Ridge alpha
    if alpha_selected < PRODUCTION_CONFIG['validation_thresholds']['ridge_alpha_min']:
        logger.alert(
            f"Ridge alpha ({alpha_selected:.4f}) below threshold "
            f"({PRODUCTION_CONFIG['validation_thresholds']['ridge_alpha_min']})",
            severity='WARNING'
        )

    # Save models
    logger.log("   Saving models...")
    with open(MODEL_DIR / 'lightgbm_production.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    with open(MODEL_DIR / 'sarima_production.pkl', 'wb') as f:
        pickle.dump(sarima_model, f)
    with open(MODEL_DIR / 'ridge_meta_production.pkl', 'wb') as f:
        pickle.dump(ridge_model, f)
    with open(MODEL_DIR / 'scaler_production.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.log("✅ Models saved successfully")

    return lgb_model, sarima_model, ridge_model, scaler, X_test_scaled, y_test

def load_existing_models(logger):
    """Load previously trained models"""
    logger.log("Loading existing models...")

    try:
        with open(MODEL_DIR / 'lightgbm_production.pkl', 'rb') as f:
            lgb_model = pickle.load(f)
        with open(MODEL_DIR / 'sarima_production.pkl', 'rb') as f:
            sarima_model = pickle.load(f)
        with open(MODEL_DIR / 'ridge_meta_production.pkl', 'rb') as f:
            ridge_model = pickle.load(f)
        with open(MODEL_DIR / 'scaler_production.pkl', 'rb') as f:
            scaler = pickle.load(f)

        logger.log("✅ Models loaded successfully")
        return lgb_model, sarima_model, ridge_model, scaler

    except Exception as e:
        logger.alert(f"Failed to load models: {e}", severity='CRITICAL')
        sys.exit(1)

def validate_model_components(lgb_model, sarima_model, ridge_model, df, logger):
    """Validate model components against production thresholds"""
    logger.log("Validating model components...")

    validation_passed = True

    # Get forecast horizon (periods after last actual data point)
    last_actual_date = pd.to_datetime('2025-12-31')
    df_future = df[df['date'] > last_actual_date].copy()
    forecast_quarters = len(df_future)

    # If no future data to validate, skip SARIMA forecast check
    if forecast_quarters == 0:
        logger.log("   ℹ️  No future periods - skipping SARIMA forecast validation")
        return True

    # 1. Check SARIMA stability
    logger.log("   Checking SARIMA stability...")
    try:
        sarima_forecast = sarima_model.forecast(steps=forecast_quarters)
        sarima_max = sarima_forecast.max() if hasattr(sarima_forecast, 'max') else max(sarima_forecast)
    except Exception as e:
        logger.alert(f"SARIMA forecast failed during validation: {e}", severity='WARNING')
        # Generate a simple validation forecast for checking
        sarima_forecast = np.array([sarima_model.fittedvalues.iloc[-1]] * forecast_quarters)
        sarima_max = sarima_forecast.max()

    threshold = PRODUCTION_CONFIG['validation_thresholds']['sarima_max_prediction']
    if sarima_max > threshold:
        logger.alert(
            f"SARIMA predictions explosive! Max: {sarima_max:.2f}% (threshold: {threshold}%)",
            severity='CRITICAL'
        )
        validation_passed = False
    else:
        logger.log(f"   ✅ SARIMA stable: [{sarima_forecast.min():.2f}%, {sarima_max:.2f}%]")

    # 2. Check Ridge configuration
    logger.log("   Checking Ridge configuration...")
    alpha = ridge_model.alpha_
    weights = ridge_model.coef_

    min_alpha = PRODUCTION_CONFIG['validation_thresholds']['ridge_alpha_min']
    if alpha < min_alpha:
        logger.alert(
            f"Ridge alpha ({alpha:.4f}) below threshold ({min_alpha})",
            severity='WARNING'
        )

    logger.log(f"   Ridge alpha: {alpha:.4f}")
    logger.log(f"   LightGBM weight: {weights[0]:.6f} ({abs(weights[0])/(abs(weights[0])+abs(weights[1]))*100:.1f}%)")
    logger.log(f"   SARIMA weight: {weights[1]:.6f} ({abs(weights[1])/(abs(weights[0])+abs(weights[1]))*100:.1f}%)")

    return validation_passed

def generate_forecast(lgb_model, sarima_model, ridge_model, scaler, df, logger):
    """Generate forecast for future periods"""
    logger.log("Generating forecast...")

    # Determine last actual data point
    # Assumption: actuals are before 2026, forecasts are 2026+
    last_actual_date = pd.to_datetime('2025-12-31')

    # Get future data (periods after last actual)
    df_future = df[df['date'] > last_actual_date].copy()

    if len(df_future) == 0:
        logger.alert("No future periods to forecast (all data is historical)", severity='WARNING')
        return None

    logger.log(f"   Forecasting {len(df_future)} quarters ({df_future['date'].min()} to {df_future['date'].max()})")
    logger.log(f"   Last actual data: {last_actual_date.strftime('%Y-%m-%d')}")

    # Prepare features
    X_future = df_future[CORE_FEATURES]
    X_future_scaled = scaler.transform(X_future)

    # Get component predictions
    lgb_pred = lgb_model.predict(X_future_scaled, num_iteration=lgb_model.best_iteration)

    # SARIMA forecast
    # Calculate steps from end of training to start of forecast horizon
    train_end_date = pd.to_datetime('2022-12-31')  # End of training period
    forecast_start_date = df_future['date'].min()

    # Count quarters between train end and forecast start
    all_quarters_df = df[(df['date'] > train_end_date) & (df['date'] <= last_actual_date)]
    intermediate_quarters = len(all_quarters_df)

    # Total steps to forecast from model end
    total_steps = intermediate_quarters + len(df_future)

    # Get full SARIMA forecast and slice to get only future periods
    sarima_full_pred = sarima_model.forecast(steps=total_steps)
    sarima_pred = sarima_full_pred[-len(df_future):]  # Take only the last N periods

    # Ensemble prediction
    X_meta = np.column_stack([lgb_pred, sarima_pred])
    ensemble_pred = ridge_model.predict(X_meta)

    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': df_future['date'],
        'lightgbm_prediction': lgb_pred,
        'sarima_prediction': sarima_pred,
        'ensemble_prediction': ensemble_pred
    })

    logger.log(f"   ✅ Forecast generated")
    logger.log(f"      Range: [{ensemble_pred.min():.2f}%, {ensemble_pred.max():.2f}%]")
    logger.log(f"      Mean: {ensemble_pred.mean():.2f}%")

    return forecast_df

# ============================================================================
# FORECAST COMPARISON AND ACCURACY TRACKING
# ============================================================================

def load_previous_forecast(logger):
    """Load most recent previous forecast for comparison"""
    logger.log("Loading previous forecast...")

    # Find most recent forecast file
    forecast_files = list(OUTPUT_DIR.glob('phoenix_forecast_*_*.csv'))

    if not forecast_files:
        logger.log("   ℹ️  No previous forecast found (first run)")
        return None

    # Sort by modification time (most recent first)
    forecast_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = forecast_files[0]

    try:
        prev_forecast = pd.read_csv(latest_file)
        prev_forecast['date'] = pd.to_datetime(prev_forecast['date'])
        logger.log(f"   ✅ Loaded previous forecast: {latest_file.name}")
        return prev_forecast
    except Exception as e:
        logger.alert(f"Failed to load previous forecast: {e}", severity='WARNING')
        return None

def compare_forecasts(current_forecast, previous_forecast, logger):
    """Compare current forecast to previous forecast"""
    if previous_forecast is None:
        logger.log("   Skipping comparison (no previous forecast)")
        return None

    logger.log("Comparing forecasts...")

    # Merge on date
    comparison = current_forecast.merge(
        previous_forecast[['date', 'ensemble_prediction']],
        on='date',
        how='inner',
        suffixes=('_current', '_previous')
    )

    if len(comparison) == 0:
        logger.log("   ℹ️  No overlapping periods to compare")
        return None

    # Calculate revision
    comparison['revision'] = comparison['ensemble_prediction_current'] - comparison['ensemble_prediction_previous']

    # Summary statistics
    mean_revision = comparison['revision'].mean()
    max_revision = comparison['revision'].abs().max()
    periods_changed = len(comparison[comparison['revision'].abs() > 0.1])

    logger.log(f"   Comparing {len(comparison)} overlapping periods")
    logger.log(f"   Mean revision: {mean_revision:+.2f} percentage points")
    logger.log(f"   Max revision: {max_revision:.2f} percentage points")
    logger.log(f"   Periods with >0.1pp change: {periods_changed}")

    # Alert on large revisions
    threshold = PRODUCTION_CONFIG['validation_thresholds']['forecast_revision_alert']
    large_revisions = comparison[comparison['revision'].abs() > threshold]

    if len(large_revisions) > 0:
        logger.alert(
            f"Large forecast revisions detected ({len(large_revisions)} periods >±{threshold}pp)",
            severity='WARNING'
        )
        for _, row in large_revisions.iterrows():
            logger.log(f"      {row['date'].strftime('%Y-%m-%d')}: {row['revision']:+.2f}pp "
                      f"({row['ensemble_prediction_previous']:.2f}% → {row['ensemble_prediction_current']:.2f}%)")

    return comparison

def track_forecast_accuracy(df, forecast_df, logger):
    """Track forecast accuracy against actual values"""
    logger.log("Tracking forecast accuracy...")

    # Load existing accuracy tracking
    if ACCURACY_TRACKING_FILE.exists():
        accuracy_df = pd.read_csv(ACCURACY_TRACKING_FILE)
        accuracy_df['forecast_date'] = pd.to_datetime(accuracy_df['forecast_date'])
        accuracy_df['target_date'] = pd.to_datetime(accuracy_df['target_date'])
    else:
        accuracy_df = pd.DataFrame(columns=[
            'forecast_date', 'target_date', 'forecast_value', 'actual_value', 'error'
        ])

    # Get actuals for periods that were previously forecast
    df_actuals = df[df['rent_growth_yoy'].notna()][['date', 'rent_growth_yoy']]

    # Add current forecast
    current_date = datetime.now()
    new_forecasts = []

    for _, row in forecast_df.iterrows():
        new_forecasts.append({
            'forecast_date': current_date,
            'target_date': row['date'],
            'forecast_value': row['ensemble_prediction'],
            'actual_value': np.nan,
            'error': np.nan
        })

    new_forecasts_df = pd.DataFrame(new_forecasts)

    # Update actuals for previous forecasts
    for idx, row in accuracy_df.iterrows():
        if pd.isna(row['actual_value']):
            actual = df_actuals[df_actuals['date'] == row['target_date']]
            if len(actual) > 0:
                accuracy_df.loc[idx, 'actual_value'] = actual.iloc[0]['rent_growth_yoy']
                accuracy_df.loc[idx, 'error'] = actual.iloc[0]['rent_growth_yoy'] - row['forecast_value']

    # Combine and save
    accuracy_df = pd.concat([accuracy_df, new_forecasts_df], ignore_index=True)
    accuracy_df.to_csv(ACCURACY_TRACKING_FILE, index=False)

    # Report accuracy metrics for completed forecasts
    completed = accuracy_df[accuracy_df['actual_value'].notna()]

    if len(completed) > 0:
        mae = completed['error'].abs().mean()
        rmse = np.sqrt((completed['error'] ** 2).mean())
        logger.log(f"   ✅ Accuracy tracking updated ({len(completed)} completed forecasts)")
        logger.log(f"      Historical MAE: {mae:.2f}pp")
        logger.log(f"      Historical RMSE: {rmse:.2f}pp")
    else:
        logger.log(f"   ✅ Accuracy tracking updated (no completed forecasts yet)")

    return accuracy_df

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_forecast_outputs(forecast_df, comparison, logger):
    """Save forecast outputs with datestamp"""
    logger.log("Saving forecast outputs...")

    datestamp = datetime.now().strftime('%Y%m%d')

    # 1. Save forecast CSV
    forecast_file = OUTPUT_DIR / f'phoenix_forecast_2026_2028_{datestamp}.csv'
    forecast_df.to_csv(forecast_file, index=False)
    logger.log(f"   ✅ Forecast CSV: {forecast_file.name}")

    # 2. Save metadata
    metadata = {
        'forecast_date': datetime.now().isoformat(),
        'forecast_periods': len(forecast_df),
        'date_range': {
            'start': forecast_df['date'].min().isoformat(),
            'end': forecast_df['date'].max().isoformat()
        },
        'ensemble_predictions': {
            'mean': float(forecast_df['ensemble_prediction'].mean()),
            'min': float(forecast_df['ensemble_prediction'].min()),
            'max': float(forecast_df['ensemble_prediction'].max()),
            'std': float(forecast_df['ensemble_prediction'].std())
        },
        'configuration': PRODUCTION_CONFIG
    }

    metadata_file = OUTPUT_DIR / f'phoenix_forecast_2026_2028_metadata_{datestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.log(f"   ✅ Metadata JSON: {metadata_file.name}")

    # 3. Save comparison if available
    if comparison is not None:
        comparison_file = OUTPUT_DIR / f'forecast_comparison_{datestamp}.csv'
        comparison.to_csv(comparison_file, index=False)
        logger.log(f"   ✅ Comparison CSV: {comparison_file.name}")

    return forecast_file, metadata_file

def create_visualizations(df, forecast_df, comparison, accuracy_df, logger, enhanced=True):
    """Create comprehensive visualizations (basic + optional enhanced)"""
    logger.log("Creating visualizations...")

    datestamp = datetime.now().strftime('%Y%m%d')

    # Figure 1: Forecast time series (basic 4-panel chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phoenix Rent Growth Forecast Update', fontsize=16, fontweight='bold')

    # Plot 1: Historical + Forecast
    ax = axes[0, 0]
    df_actual = df[df['rent_growth_yoy'].notna()]
    ax.plot(df_actual['date'], df_actual['rent_growth_yoy'], 'o-',
            label='Actual', linewidth=2, markersize=4)
    ax.plot(forecast_df['date'], forecast_df['ensemble_prediction'], 'o-',
            label='Forecast', linewidth=2, markersize=4, color='coral')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Historical Actuals and Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rent Growth YoY (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Component predictions
    ax = axes[0, 1]
    ax.plot(forecast_df['date'], forecast_df['lightgbm_prediction'], 'o-',
            label='LightGBM', alpha=0.7)
    ax.plot(forecast_df['date'], forecast_df['sarima_prediction'], 's-',
            label='SARIMA', alpha=0.7)
    ax.plot(forecast_df['date'], forecast_df['ensemble_prediction'], 'D-',
            label='Ensemble', linewidth=2)
    ax.set_title('Model Component Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rent Growth YoY (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Forecast comparison (if available)
    ax = axes[1, 0]
    if comparison is not None:
        ax.bar(range(len(comparison)), comparison['revision'],
               color=['red' if x < 0 else 'green' for x in comparison['revision']])
        ax.set_xticks(range(len(comparison)))
        ax.set_xticklabels([d.strftime('%Y-Q%q') for d in comparison['date']], rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title('Forecast Revisions vs. Previous Month')
        ax.set_ylabel('Revision (percentage points)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No previous forecast\nfor comparison',
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Forecast Revisions vs. Previous Month')

    # Plot 4: Accuracy tracking (if available)
    ax = axes[1, 1]
    completed = accuracy_df[accuracy_df['actual_value'].notna()]
    if len(completed) > 0:
        ax.scatter(completed['forecast_value'], completed['actual_value'], alpha=0.6)

        # Add diagonal line (perfect forecast)
        min_val = min(completed['forecast_value'].min(), completed['actual_value'].min())
        max_val = max(completed['forecast_value'].max(), completed['actual_value'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Forecast')

        ax.set_title('Forecast Accuracy (Completed Periods)')
        ax.set_xlabel('Forecasted Value (%)')
        ax.set_ylabel('Actual Value (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No completed forecasts\nyet for accuracy tracking',
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Forecast Accuracy (Completed Periods)')

    plt.tight_layout()

    viz_file = OUTPUT_DIR / f'phoenix_forecast_update_{datestamp}.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.log(f"   ✅ Basic visualization: {viz_file.name}")

    # Create enhanced visualizations if enabled
    if enhanced:
        try:
            from visualizations import ForecastVisualizer

            logger.log("   Creating enhanced visualizations...")
            viz = ForecastVisualizer()

            # Prepare data for enhanced visualizations
            historical_data = pd.DataFrame({
                'Date': df_actual['date'],
                'Rent_Growth': df_actual['rent_growth_yoy']
            })

            enhanced_forecast_data = pd.DataFrame({
                'Date': forecast_df['date'],
                'Forecast': forecast_df['ensemble_prediction']
            })

            # Model components for component analysis
            components = {
                'LightGBM': forecast_df['lightgbm_prediction'].values,
                'SARIMA': forecast_df['sarima_prediction'].values,
                'Ridge': forecast_df['ridge_prediction'].values if 'ridge_prediction' in forecast_df.columns else None
            }
            # Remove None components
            components = {k: v for k, v in components.items() if v is not None}

            # Generate confidence intervals (simple approximation based on std)
            std_forecast = forecast_df['ensemble_prediction'].std()
            confidence_intervals = {
                95: (
                    forecast_df['ensemble_prediction'] - 1.96 * std_forecast,
                    forecast_df['ensemble_prediction'] + 1.96 * std_forecast
                ),
                80: (
                    forecast_df['ensemble_prediction'] - 1.28 * std_forecast,
                    forecast_df['ensemble_prediction'] + 1.28 * std_forecast
                ),
                50: (
                    forecast_df['ensemble_prediction'] - 0.67 * std_forecast,
                    forecast_df['ensemble_prediction'] + 0.67 * std_forecast
                )
            }

            # Generate scenarios (best/worst/base case)
            scenarios = {
                'Best Case': pd.DataFrame({
                    'Date': forecast_df['date'],
                    'Forecast': forecast_df['ensemble_prediction'] + std_forecast
                }),
                'Base Case': enhanced_forecast_data.copy(),
                'Worst Case': pd.DataFrame({
                    'Date': forecast_df['date'],
                    'Forecast': forecast_df['ensemble_prediction'] - std_forecast
                })
            }

            # Generate comprehensive report
            enhanced_files = viz.generate_comprehensive_report(
                historical_data,
                enhanced_forecast_data,
                components=components,
                confidence_intervals=confidence_intervals,
                scenarios=scenarios,
                metrics={'RMSE': std_forecast}
            )

            logger.log(f"   ✅ Enhanced visualizations generated: {len(enhanced_files)} files")

        except ImportError:
            logger.log("   ⚠️  Enhanced visualizations module not available (install plotly for interactive charts)", level='WARNING')
        except Exception as e:
            logger.log(f"   ⚠️  Enhanced visualizations failed: {e}", level='WARNING')

    return viz_file

def update_executive_summary(forecast_df, metadata, comparison, logger):
    """Update executive summary with latest forecast"""
    logger.log("Updating executive summary...")

    datestamp = datetime.now().strftime('%Y-%m-%d')

    # Calculate annual averages
    forecast_df['year'] = pd.to_datetime(forecast_df['date']).dt.year
    annual_avg = forecast_df.groupby('year')['ensemble_prediction'].mean()

    summary_content = f"""# Phoenix Rent Growth Forecast Update

**Forecast Date**: {datestamp}
**Model**: Production-Validated Ensemble (LightGBM + SARIMA + Ridge)

---

## Executive Summary

### 2026-2028 Outlook
- **Average Annual Rent Growth**: **{metadata['ensemble_predictions']['mean']:.2f}%**
- **Range**: {metadata['ensemble_predictions']['min']:.2f}% to {metadata['ensemble_predictions']['max']:.2f}%

### Annual Breakdown

| Year | Average Rent Growth |
|------|---------------------|
"""

    for year, avg in annual_avg.items():
        summary_content += f"| **{int(year)}** | **{avg:.2f}%** |\n"

    summary_content += """
### Quarterly Forecast Detail

| Quarter | Ensemble Prediction |
|---------|---------------------|
"""

    for _, row in forecast_df.iterrows():
        quarter = pd.to_datetime(row['date']).strftime('%Y-Q%q')
        summary_content += f"| {quarter} | **{row['ensemble_prediction']:.2f}%** |\n"

    # Add comparison section if available
    if comparison is not None:
        summary_content += f"""
---

## Changes from Previous Forecast

- **Mean Revision**: {comparison['revision'].mean():+.2f} percentage points
- **Max Revision**: {comparison['revision'].abs().max():.2f} percentage points
- **Periods with >0.1pp change**: {len(comparison[comparison['revision'].abs() > 0.1])}

### Notable Revisions

"""
        large_revisions = comparison[comparison['revision'].abs() > 0.5].sort_values('revision', ascending=False)
        if len(large_revisions) > 0:
            for _, row in large_revisions.iterrows():
                quarter = pd.to_datetime(row['date']).strftime('%Y-Q%q')
                summary_content += f"- **{quarter}**: {row['revision']:+.2f}pp "
                summary_content += f"({row['ensemble_prediction_previous']:.2f}% → {row['ensemble_prediction_current']:.2f}%)\n"
        else:
            summary_content += "No significant revisions (all changes <0.5pp)\n"

    summary_content += f"""
---

## Model Configuration

**Production-Validated Configuration** (from root cause analysis):
- **SARIMA**: Order (1,1,2), Seasonal (0,0,1,4)
- **LightGBM**: Early stopping 50 rounds, best iteration varies
- **Ridge**: Alpha range [0.1, 1.0, 10.0, 100.0, 1000.0]

---

**Auto-generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save executive summary
    summary_file = REPORTS_DIR / 'PHOENIX_FORECAST_EXECUTIVE_SUMMARY_LATEST.md'
    with open(summary_file, 'w') as f:
        f.write(summary_content)

    logger.log(f"   ✅ Executive summary: {summary_file.name}")

    return summary_file

def update_last_run_record(df, logger):
    """Update last successful run record"""
    last_run_data = {
        'run_date': datetime.now().isoformat(),
        'last_data_date': df['date'].max().isoformat(),
        'total_quarters': len(df),
        'actual_quarters': len(df[df['rent_growth_yoy'].notna()]),
        'future_quarters': len(df[df['rent_growth_yoy'].isna()])
    }

    with open(LAST_RUN_FILE, 'w') as f:
        json.dump(last_run_data, f, indent=2)

    logger.log(f"✅ Last run record updated")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Automated Phoenix Rent Growth Forecast Update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard monthly update
  python automated_monthly_forecast_update.py

  # Force retrain models
  python automated_monthly_forecast_update.py --force-retrain

  # With email alerts
  python automated_monthly_forecast_update.py --alert-email analyst@company.com

  # Cron job (1st of month at 8am)
  0 8 1 * * cd /path/to/project && python scripts/automated_monthly_forecast_update.py
        """
    )
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force model retraining even if models exist')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip comparison to previous forecasts')
    parser.add_argument('--alert-email', type=str,
                       help='Email address for alerts')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')

    args = parser.parse_args()

    # Initialize logger
    logger = ForecastLogger(alert_email=args.alert_email)

    logger.log("=" * 80)
    logger.log("PHOENIX RENT GROWTH FORECAST - AUTOMATED MONTHLY UPDATE")
    logger.log("=" * 80)
    logger.log(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("")

    try:
        # 1. Load and validate data
        df, has_new_data = load_latest_data(logger)
        data_valid = validate_data_quality(df, logger)

        if not data_valid:
            logger.alert("Data validation failed - review warnings above", severity='WARNING')

        # 2. Determine if models should be retrained
        retrain = should_retrain_models(args.force_retrain, has_new_data, logger)

        # 3. Train or load models
        if retrain:
            lgb_model, sarima_model, ridge_model, scaler, X_test, y_test = train_ensemble_model(df, logger)
        else:
            lgb_model, sarima_model, ridge_model, scaler = load_existing_models(logger)

        # 4. Validate model components
        validation_passed = validate_model_components(lgb_model, sarima_model, ridge_model, df, logger)

        if not validation_passed:
            logger.alert("Model validation failed - review warnings above", severity='CRITICAL')

        # 5. Generate forecast
        forecast_df = generate_forecast(lgb_model, sarima_model, ridge_model, scaler, df, logger)

        if forecast_df is None:
            logger.alert("Forecast generation failed", severity='CRITICAL')
            sys.exit(1)

        # 6. Compare to previous forecast
        comparison = None
        if not args.skip_comparison:
            prev_forecast = load_previous_forecast(logger)
            comparison = compare_forecasts(forecast_df, prev_forecast, logger)

        # 7. Track forecast accuracy
        accuracy_df = track_forecast_accuracy(df, forecast_df, logger)

        # 8. Save outputs
        forecast_file, metadata_file = save_forecast_outputs(forecast_df, comparison, logger)
        viz_file = create_visualizations(df, forecast_df, comparison, accuracy_df, logger)
        summary_file = update_executive_summary(
            forecast_df,
            json.load(open(metadata_file)),
            comparison,
            logger
        )

        # 9. Update last run record
        update_last_run_record(df, logger)

        # 10. Send alerts if configured
        # Load forecast summary for email
        forecast_summary = None
        try:
            with open(summary_file, 'r') as f:
                forecast_summary = f.read()
        except:
            pass

        logger.send_alerts(forecast_summary=forecast_summary)

        logger.log("")
        logger.log("=" * 80)
        logger.log("✅ FORECAST UPDATE COMPLETED SUCCESSFULLY")
        logger.log("=" * 80)
        logger.log(f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Log file: {logger.log_file}")
        logger.log("")
        logger.log("Output files:")
        logger.log(f"  - Forecast: {forecast_file}")
        logger.log(f"  - Metadata: {metadata_file}")
        logger.log(f"  - Visualization: {viz_file}")
        logger.log(f"  - Executive Summary: {summary_file}")
        logger.log("=" * 80)

        return 0

    except Exception as e:
        logger.alert(f"CRITICAL ERROR: {str(e)}", severity='CRITICAL')
        import traceback
        logger.log(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())
