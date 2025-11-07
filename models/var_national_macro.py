#!/usr/bin/env python3
"""
VAR National Macro Component - Phoenix Rent Growth Forecasting
Component 1 of 3 in Hierarchical Ensemble Model

Purpose: Vector Autoregression baseline using national macroeconomic variables
Weight: 30% in final ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'models/output'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VAR NATIONAL MACRO COMPONENT - PHOENIX RENT GROWTH FORECASTING")
print("=" * 80)
print(f"Component: 1 of 3 (Vector Autoregression)")
print(f"Weight in Ensemble: 30%")
print(f"Purpose: National macroeconomic baseline\n")

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PROCESSED DATASET")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')

print(f"✅ Loaded dataset: {len(df)} quarters")
print(f"   Date Range: {df.index.min()} to {df.index.max()}")

# Filter to historical period only (exclude future forecasts)
historical_df = df.loc[:'2025-09-30'].copy()  # Through 2025 Q3

print(f"\nFiltered to historical data: {len(historical_df)} quarters")
print(f"   Training Period: {historical_df.index.min()} to {historical_df.index.max()}")

# ============================================================================
# 2. Select National Macro Variables for VAR
# ============================================================================

print("\n" + "=" * 80)
print("SELECTING NATIONAL MACRO VARIABLES")
print("=" * 80)

# National macro variables (exogenous to Phoenix market)
var_columns = [
    'mortgage_rate_30yr',
    'fed_funds_rate',
    'national_unemployment',
    'cpi',
    'inflation_expectations_5yr',
    'housing_starts',
    'building_permits'
]

# Check availability
available_vars = [col for col in var_columns if col in historical_df.columns]
print(f"\nAvailable VAR variables: {len(available_vars)}/{len(var_columns)}")

for var in available_vars:
    non_null = historical_df[var].notna().sum()
    pct = (non_null / len(historical_df)) * 100
    print(f"  {var:35s}: {non_null:3d} non-null ({pct:5.1f}%)")

# Select data with minimal missing values
var_data = historical_df[available_vars].dropna()

print(f"\n✅ VAR dataset after dropping missing: {len(var_data)} quarters")
print(f"   Coverage: {var_data.index.min()} to {var_data.index.max()}")

# ============================================================================
# 3. Stationarity Testing
# ============================================================================

print("\n" + "=" * 80)
print("STATIONARITY TESTING (Augmented Dickey-Fuller)")
print("=" * 80)

print("\nTesting for unit roots (non-stationarity):")
print("H0: Series has unit root (non-stationary)")
print("If p-value < 0.05, reject H0 (series is stationary)\n")

adf_results = {}
non_stationary_vars = []

for col in var_data.columns:
    result = adfuller(var_data[col].dropna(), autolag='AIC')
    adf_results[col] = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'stationary': result[1] < 0.05
    }

    status = "✅ Stationary" if result[1] < 0.05 else "⚠️  Non-stationary"
    print(f"  {col:35s}: p={result[1]:.4f}  {status}")

    if result[1] >= 0.05:
        non_stationary_vars.append(col)

# Apply differencing to non-stationary variables
if non_stationary_vars:
    print(f"\n⚠️  Found {len(non_stationary_vars)} non-stationary variables")
    print("   Applying first-order differencing...")

    var_data_stationary = var_data.copy()
    for col in non_stationary_vars:
        var_data_stationary[col] = var_data[col].diff()

    # Drop first row (NaN from differencing)
    var_data_stationary = var_data_stationary.dropna()

    print(f"   ✅ Differenced data: {len(var_data_stationary)} quarters")
else:
    var_data_stationary = var_data.copy()
    print("\n✅ All variables are stationary")

# ============================================================================
# 4. Lag Order Selection
# ============================================================================

print("\n" + "=" * 80)
print("LAG ORDER SELECTION")
print("=" * 80)

# Fit VAR model for lag order selection
model_select = VAR(var_data_stationary)

# Test lag orders (conservative to ensure model can be estimated)
# Rule: need k*p + 20 observations, where k=variables, p=lags
max_lags = min(4, len(var_data_stationary) // 10)  # Conservative for VAR with 7 variables
lag_order_results = model_select.select_order(maxlags=max_lags)

print("\nLag Order Selection Criteria:")
print(f"  AIC (Akaike Information Criterion):    {lag_order_results.aic} lags")
print(f"  BIC (Bayesian Information Criterion):  {lag_order_results.bic} lags")
print(f"  FPE (Final Prediction Error):          {lag_order_results.fpe} lags")
print(f"  HQIC (Hannan-Quinn):                    {lag_order_results.hqic} lags")

# Use BIC (more conservative, penalizes complexity)
optimal_lag = lag_order_results.bic
print(f"\n✅ Selected lag order: {optimal_lag} (BIC criterion)")

# ============================================================================
# 5. Fit VAR Model
# ============================================================================

print("\n" + "=" * 80)
print("FITTING VAR MODEL")
print("=" * 80)

var_model = VAR(var_data_stationary)
var_fitted = var_model.fit(maxlags=optimal_lag, ic='bic')

print(f"✅ VAR model fitted with {optimal_lag} lags")
print(f"   Number of equations: {len(var_fitted.params.columns)}")
print(f"   Number of observations: {var_fitted.nobs}")
print(f"   Degrees of freedom: {var_fitted.df_model}")

# Model summary statistics
print("\nModel Summary:")
print(f"  AIC: {var_fitted.aic:.2f}")
print(f"  BIC: {var_fitted.bic:.2f}")
print(f"  FPE: {var_fitted.fpe:.2e}")
print(f"  Log Likelihood: {var_fitted.llf:.2f}")

# ============================================================================
# 6. Granger Causality Testing
# ============================================================================

print("\n" + "=" * 80)
print("GRANGER CAUSALITY TESTING")
print("=" * 80)

print("\nTesting if national macro variables Granger-cause each other:")
print("(Helps identify leading indicators)\n")

# Test key relationships
test_pairs = [
    ('fed_funds_rate', 'mortgage_rate_30yr'),
    ('inflation_expectations_5yr', 'fed_funds_rate'),
    ('national_unemployment', 'housing_starts')
]

for cause_var, effect_var in test_pairs:
    if cause_var in var_data_stationary.columns and effect_var in var_data_stationary.columns:
        print(f"Testing: Does '{cause_var}' Granger-cause '{effect_var}'?")
        try:
            test_result = grangercausalitytests(
                var_data_stationary[[effect_var, cause_var]],
                maxlag=optimal_lag,
                verbose=False
            )
            # Get p-value for lag 1
            p_value = test_result[1][0]['ssr_ftest'][1]
            result = "✅ Yes (p<0.05)" if p_value < 0.05 else "❌ No (p≥0.05)"
            print(f"  {result} (p={p_value:.4f})\n")
        except:
            print(f"  ⚠️  Unable to test (insufficient data)\n")

# ============================================================================
# 7. Time Series Cross-Validation Setup
# ============================================================================

print("\n" + "=" * 80)
print("TIME SERIES CROSS-VALIDATION SETUP")
print("=" * 80)

# Split data: train on 2010-2022, validate on 2023-2025
train_end_date = '2022-12-31'
train_data = var_data_stationary.loc[:train_end_date]
test_data = var_data_stationary.loc[train_end_date:]

print(f"\nTraining set: {len(train_data)} quarters ({train_data.index.min()} to {train_data.index.max()})")
print(f"Test set:     {len(test_data)} quarters ({test_data.index.min()} to {test_data.index.max()})")

# Refit model on training data only
var_model_train = VAR(train_data)
var_fitted_train = var_model_train.fit(maxlags=optimal_lag, ic='bic')

print(f"\n✅ VAR model refitted on training data")

# ============================================================================
# 8. Generate Forecasts
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VAR FORECASTS")
print("=" * 80)

# Forecast on test period
forecast_steps = len(test_data)
forecast = var_fitted_train.forecast(train_data.values[-optimal_lag:], steps=forecast_steps)

# Convert to DataFrame
forecast_df = pd.DataFrame(
    forecast,
    index=test_data.index,
    columns=var_data_stationary.columns
)

print(f"✅ Generated {forecast_steps}-step ahead forecasts")
print(f"   Forecast period: {forecast_df.index.min()} to {forecast_df.index.max()}")

# ============================================================================
# 9. Evaluate Forecast Performance
# ============================================================================

print("\n" + "=" * 80)
print("FORECAST PERFORMANCE EVALUATION")
print("=" * 80)

print("\nOut-of-Sample Forecast Accuracy (Test Period 2023-2025):\n")

metrics = {}
for col in var_data_stationary.columns:
    actual = test_data[col].values
    predicted = forecast_df[col].values

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # Directional accuracy
    actual_direction = np.sign(np.diff(actual))
    pred_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    metrics[col] = {
        'RMSE': rmse,
        'MAE': mae,
        'Directional_Accuracy': directional_accuracy
    }

    print(f"{col:35s}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
    print()

# ============================================================================
# 10. Generate Future Forecasts (2025 Q4 - 2027 Q4)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING FUTURE FORECASTS (2025 Q4 - 2027 Q4)")
print("=" * 80)

# Refit on all historical data for production forecasts
var_model_full = VAR(var_data_stationary)
var_fitted_full = var_model_full.fit(maxlags=optimal_lag, ic='bic')

# Forecast 8 quarters ahead (2 years)
future_steps = 8
future_forecast = var_fitted_full.forecast(
    var_data_stationary.values[-optimal_lag:],
    steps=future_steps
)

# Create future dates (quarterly)
last_date = var_data_stationary.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=3),
    periods=future_steps,
    freq='Q'
)

# Convert to DataFrame
future_forecast_df = pd.DataFrame(
    future_forecast,
    index=future_dates,
    columns=var_data_stationary.columns
)

print(f"✅ Generated {future_steps}-quarter ahead forecasts")
print(f"   Forecast period: {future_forecast_df.index.min()} to {future_forecast_df.index.max()}")

print("\nFuture National Macro Forecasts:")
for col in future_forecast_df.columns:
    current_val = var_data_stationary[col].iloc[-1]
    forecast_val = future_forecast_df[col].iloc[-1]
    print(f"  {col:35s}: {current_val:8.4f} → {forecast_val:8.4f}")

# ============================================================================
# 11. Save VAR Component Model and Predictions
# ============================================================================

print("\n" + "=" * 80)
print("SAVING VAR COMPONENT")
print("=" * 80)

# Save fitted model
model_path = OUTPUT_PATH / 'var_national_macro_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(var_fitted_full, f)

print(f"✅ Saved VAR model: {model_path}")

# Save forecasts
test_forecast_path = OUTPUT_PATH / 'var_test_forecasts.csv'
forecast_df.to_csv(test_forecast_path)
print(f"✅ Saved test forecasts: {test_forecast_path}")

future_forecast_path = OUTPUT_PATH / 'var_future_forecasts.csv'
future_forecast_df.to_csv(future_forecast_path)
print(f"✅ Saved future forecasts: {future_forecast_path}")

# Save performance metrics
metrics_df = pd.DataFrame(metrics).T
metrics_path = OUTPUT_PATH / 'var_performance_metrics.csv'
metrics_df.to_csv(metrics_path)
print(f"✅ Saved performance metrics: {metrics_path}")

# ============================================================================
# 12. Component Summary
# ============================================================================

print("\n" + "=" * 80)
print("VAR COMPONENT SUMMARY")
print("=" * 80)

print(f"""
Component: VAR National Macro Baseline
Status: ✅ COMPLETE

Model Specifications:
  - Variables: {len(var_data_stationary.columns)} national macro indicators
  - Lag Order: {optimal_lag} quarters (BIC criterion)
  - Training Period: {train_data.index.min()} to {train_data.index.max()}
  - Test Period: {test_data.index.min()} to {test_data.index.max()}
  - Future Forecast: {future_forecast_df.index.min()} to {future_forecast_df.index.max()}

Performance (Out-of-Sample):
  - Average RMSE: {np.mean([m['RMSE'] for m in metrics.values()]):.4f}
  - Average MAE: {np.mean([m['MAE'] for m in metrics.values()]):.4f}
  - Average Directional Accuracy: {np.mean([m['Directional_Accuracy'] for m in metrics.values()]):.1f}%

Ensemble Weight: 30%

Output Files:
  - Model: {model_path.name}
  - Test Forecasts: {test_forecast_path.name}
  - Future Forecasts: {future_forecast_path.name}
  - Performance Metrics: {metrics_path.name}

Next Steps:
  1. Build Phoenix-Specific GBM Component (45% weight)
  2. Build SARIMA Seasonal Component (25% weight)
  3. Combine all components with Ridge meta-learner
""")

print("=" * 80)
print("VAR COMPONENT BUILD COMPLETE")
print("=" * 80)
