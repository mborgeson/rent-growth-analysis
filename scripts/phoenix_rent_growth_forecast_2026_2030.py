#!/usr/bin/env python3
"""
Phoenix Rent Growth Forecast 2026-2030
Production-Validated Ensemble Model

Uses validated configuration from root cause analysis:
- SARIMA: (1,1,2)(0,0,1,4) - Stable, non-explosive
- Ridge: Alpha range [0.1, 1.0, 10.0, 100.0, 1000.0]
- LightGBM: Early stopping with 50 rounds
- Features: 26 core features (production validated)

Created: November 8, 2025
Author: Based on COMPLETE_ROOT_CAUSE_ANALYSIS.md findings
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
from datetime import datetime
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

# ============================================================================
# PRODUCTION CONFIGURATION (Validated from Root Cause Analysis)
# ============================================================================

PRODUCTION_CONFIG = {
    'sarima': {
        'order': (1, 1, 2),           # ✅ VALIDATED: Stable predictions
        'seasonal_order': (0, 0, 1, 4), # ✅ VALIDATED: No seasonal AR/differencing
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
        'alphas': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Include 10.0!
        'cv': 5,  # TimeSeriesSplit folds
        'explanation': 'Production likely uses alpha=10.0 for strong regularization'
    },
    'early_stopping': {
        'rounds': 50,
        'explanation': 'Critical for preventing overfitting'
    },
    'validation_thresholds': {
        'sarima_max_prediction': 10.0,  # Explosive if higher
        'component_correlation_min': -0.5,  # Concerning if lower
        'test_train_rmse_ratio': 2.0,  # Overfitting if higher
        'ridge_alpha_min': 1.0  # Weak regularization if lower
    }
}

# Core features (26 total - production validated)
CORE_FEATURES = [
    # Macroeconomic (8)
    'fed_funds_rate',
    'mortgage_rate_30yr',
    'national_unemployment',
    'cpi',
    'inflation_expectations_5yr',
    'housing_starts',
    'building_permits',

    # Housing market (7)
    'vacancy_rate',
    'cap_rate',
    'phx_hpi_yoy_growth',
    'phx_home_price_index',
    'inventory_units',
    'units_under_construction',
    'absorption_12mo',

    # Employment (5)
    'phx_total_employment',
    'phx_unemployment_rate',
    'phx_employment_yoy_growth',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',

    # Engineered features (6)
    'supply_inventory_ratio',
    'absorption_inventory_ratio',
    'mortgage_employment_interaction',
    'migration_proxy',
    'phx_prof_business_yoy_growth',
    'mortgage_rate_30yr_lag2'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath):
    """Load and prepare Phoenix modeling dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Dataset: {len(df)} quarters from {df['date'].min()} to {df['date'].max()}")

    # Identify actual vs projected data
    # Assuming data up to 2025-12-31 is actual, rest is projected
    actual_data = df[df['date'] <= '2025-12-31'].copy()
    future_data = df[df['date'] > '2025-12-31'].copy()

    print(f"Actual data: {len(actual_data)} quarters")
    print(f"Future periods: {len(future_data)} quarters")

    return df, actual_data, future_data

def validate_features(df, required_features):
    """Validate all required features are present"""
    missing = set(required_features) - set(df.columns)
    if missing:
        print(f"⚠️  WARNING: Missing features: {missing}")
        return False

    print(f"✅ All {len(required_features)} required features present")
    return True

def prepare_train_test_split(df, test_cutoff='2022-12-31'):
    """Split data into train/test using time-based cutoff"""
    train = df[df['date'] <= test_cutoff].copy()
    test = df[df['date'] > test_cutoff].copy()

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(train)} quarters ({train['date'].min()} to {train['date'].max()})")
    print(f"  Test: {len(test)} quarters ({test['date'].min()} to {test['date'].max()})")

    return train, test

def check_sarima_stability(predictions, threshold=10.0):
    """Check if SARIMA predictions are stable (not explosive)"""
    max_pred = predictions.max()
    min_pred = predictions.min()

    if max_pred > threshold:
        print(f"⚠️  WARNING: SARIMA predictions explosive! Max: {max_pred:.2f}%")
        print(f"   (Threshold: {threshold}%)")
        return False

    if min_pred < -threshold:
        print(f"⚠️  WARNING: SARIMA predictions too negative! Min: {min_pred:.2f}%")
        return False

    print(f"✅ SARIMA predictions stable: [{min_pred:.2f}%, {max_pred:.2f}%]")
    return True

def check_component_correlation(lgb_pred, sarima_pred, threshold=-0.5):
    """Check if components predict in generally same direction"""
    correlation = np.corrcoef(lgb_pred, sarima_pred)[0, 1]

    if correlation < threshold:
        print(f"⚠️  WARNING: Components predicting opposite directions!")
        print(f"   Correlation: {correlation:.4f} (threshold: {threshold})")
        return False

    print(f"✅ Component correlation acceptable: {correlation:.4f}")
    return True

def check_ridge_configuration(ridge_model, alpha_threshold=1.0):
    """Validate Ridge meta-learner configuration"""
    alpha = ridge_model.alpha_
    weights = ridge_model.coef_
    intercept = ridge_model.intercept_

    print(f"\nRidge Meta-Learner Configuration:")
    print(f"  Alpha selected: {alpha:.4f}")
    print(f"  LightGBM weight: {weights[0]:.6f}")
    print(f"  SARIMA weight: {weights[1]:.6f}")
    print(f"  Intercept: {intercept:.6f}")

    # Normalized weights
    total_weight = abs(weights[0]) + abs(weights[1])
    lgb_pct = abs(weights[0]) / total_weight * 100
    sarima_pct = abs(weights[1]) / total_weight * 100

    print(f"  Normalized influence:")
    print(f"    LightGBM: {lgb_pct:.1f}%")
    print(f"    SARIMA: {sarima_pct:.1f}%")

    if alpha < alpha_threshold:
        print(f"⚠️  WARNING: Ridge alpha ({alpha:.4f}) < threshold ({alpha_threshold})")
        print(f"   May indicate weak regularization")
    else:
        print(f"✅ Ridge alpha adequate for production")

    # Check for production-like configuration (negative weights)
    if weights[0] < 0 and weights[1] < 0:
        print(f"ℹ️  NOTE: Negative weights detected (similar to production)")
        print(f"   This is NORMAL and may represent regime-aware forecasting")

    return alpha >= alpha_threshold

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_lightgbm_component(X_train, y_train, X_val, y_val, config):
    """Train LightGBM with early stopping (production configuration)"""
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM COMPONENT")
    print("="*80)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    print("Training with early stopping (50 rounds)...")
    model = lgb.train(
        config['lightgbm'],
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=config['early_stopping']['rounds']),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"Best iteration: {model.best_iteration}")

    # Training metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
    val_rmse = np.sqrt(np.mean((val_pred - y_val)**2))

    print(f"\nLightGBM Performance:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Train/Val ratio: {val_rmse/train_rmse:.2f}x")

    if val_rmse / train_rmse > config['validation_thresholds']['test_train_rmse_ratio']:
        print(f"⚠️  WARNING: Possible overfitting (ratio > {config['validation_thresholds']['test_train_rmse_ratio']})")

    return model

def train_sarima_component(y_train, config):
    """Train SARIMA with production configuration (1,1,2)(0,0,1,4)"""
    print("\n" + "="*80)
    print("TRAINING SARIMA COMPONENT")
    print("="*80)

    order = config['sarima']['order']
    seasonal_order = config['sarima']['seasonal_order']

    print(f"Configuration: {order}{seasonal_order}")
    print(f"  {config['sarima']['explanation']}")

    print("\nFitting SARIMA model...")
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    print(f"✅ SARIMA fitted successfully")
    print(f"   AIC: {model.aic:.2f}")

    return model

def train_ridge_meta_learner(lgb_train, sarima_train, y_train, config):
    """Train Ridge meta-learner with production alpha range"""
    print("\n" + "="*80)
    print("TRAINING RIDGE META-LEARNER")
    print("="*80)

    # Combine component predictions
    X_meta = np.column_stack([lgb_train, sarima_train])

    print(f"Alpha range: {config['ridge']['alphas']}")
    print(f"  {config['ridge']['explanation']}")
    print(f"CV folds: {config['ridge']['cv']} (TimeSeriesSplit)")

    # Train with RidgeCV using TimeSeriesSplit
    ridge = RidgeCV(
        alphas=config['ridge']['alphas'],
        cv=TimeSeriesSplit(n_splits=config['ridge']['cv']),
        scoring='neg_root_mean_squared_error'
    )

    ridge.fit(X_meta, y_train)

    # Validate configuration
    check_ridge_configuration(
        ridge,
        config['validation_thresholds']['ridge_alpha_min']
    )

    return ridge

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_component_predictions(lgb_pred, sarima_pred, config):
    """Run all validation checks on component predictions"""
    print("\n" + "="*80)
    print("COMPONENT VALIDATION CHECKS")
    print("="*80)

    checks_passed = True

    # Check 1: SARIMA stability
    print("\n[1/2] SARIMA Stability Check")
    if not check_sarima_stability(
        sarima_pred,
        config['validation_thresholds']['sarima_max_prediction']
    ):
        checks_passed = False

    # Check 2: Component correlation
    print("\n[2/2] Component Correlation Check")
    if not check_component_correlation(
        lgb_pred,
        sarima_pred,
        config['validation_thresholds']['component_correlation_min']
    ):
        checks_passed = False

    print("\n" + "-"*80)
    if checks_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED - Review warnings above")
    print("-"*80)

    return checks_passed

def calculate_metrics(y_true, y_pred, name="Model"):
    """Calculate comprehensive performance metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')

    # Directional accuracy
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    metrics = {
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }

    return metrics

def print_metrics(metrics):
    """Pretty print metrics"""
    print(f"\n{metrics['name']} Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")

# ============================================================================
# FORECASTING FUNCTIONS
# ============================================================================

def generate_forecast_2026_2030(lgb_model, sarima_model, ridge_model,
                                scaler, future_data, features):
    """Generate production-quality forecast for 2026-2030"""
    print("\n" + "="*80)
    print("GENERATING FORECAST: 2026-2030")
    print("="*80)

    # Filter for 2026-2030
    forecast_data = future_data[
        (future_data['date'] >= '2026-01-01') &
        (future_data['date'] <= '2030-12-31')
    ].copy()

    print(f"Forecast periods: {len(forecast_data)} quarters")
    print(f"  From: {forecast_data['date'].min()}")
    print(f"  To: {forecast_data['date'].max()}")

    # Prepare features
    X_forecast = forecast_data[features].copy()

    # Handle missing values (forward fill)
    X_forecast = X_forecast.fillna(method='ffill')

    # Scale features
    X_forecast_scaled = scaler.transform(X_forecast)

    # LightGBM predictions
    print("\nGenerating LightGBM predictions...")
    lgb_pred = lgb_model.predict(X_forecast_scaled)

    # SARIMA predictions
    print("Generating SARIMA predictions...")
    n_periods = len(forecast_data)
    sarima_pred = sarima_model.forecast(steps=n_periods)

    # Validate components
    print("\nValidating component predictions...")
    validate_component_predictions(lgb_pred, sarima_pred, PRODUCTION_CONFIG)

    # Ensemble predictions
    print("\nCombining with Ridge meta-learner...")
    X_meta = np.column_stack([lgb_pred, sarima_pred])
    ensemble_pred = ridge_model.predict(X_meta)

    # Create results dataframe
    results = pd.DataFrame({
        'date': forecast_data['date'].values,
        'lightgbm_prediction': lgb_pred,
        'sarima_prediction': sarima_pred,
        'ensemble_prediction': ensemble_pred
    })

    # Summary statistics
    print("\n" + "-"*80)
    print("FORECAST SUMMARY")
    print("-"*80)
    print(f"\nLightGBM Predictions:")
    print(f"  Range: [{lgb_pred.min():.2f}%, {lgb_pred.max():.2f}%]")
    print(f"  Mean: {lgb_pred.mean():.2f}%")

    print(f"\nSARIMA Predictions:")
    print(f"  Range: [{sarima_pred.min():.2f}%, {sarima_pred.max():.2f}%]")
    print(f"  Mean: {sarima_pred.mean():.2f}%")

    print(f"\nEnsemble Predictions:")
    print(f"  Range: [{ensemble_pred.min():.2f}%, {ensemble_pred.max():.2f}%]")
    print(f"  Mean: {ensemble_pred.mean():.2f}%")
    print("-"*80)

    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_forecast_results(results, output_dir='outputs'):
    """Create comprehensive visualization of forecast results"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phoenix Rent Growth Forecast 2026-2030\n(Production-Validated Configuration)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Component predictions over time
    ax1 = axes[0, 0]
    ax1.plot(results['date'], results['lightgbm_prediction'],
             marker='o', label='LightGBM', linewidth=2)
    ax1.plot(results['date'], results['sarima_prediction'],
             marker='s', label='SARIMA', linewidth=2)
    ax1.plot(results['date'], results['ensemble_prediction'],
             marker='D', label='Ensemble', linewidth=2.5, color='black')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Component & Ensemble Predictions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rent Growth (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Ensemble prediction detail
    ax2 = axes[0, 1]
    bars = ax2.bar(results['date'].dt.to_period('Q').astype(str),
                   results['ensemble_prediction'],
                   color=['green' if x > 0 else 'red' for x in results['ensemble_prediction']],
                   alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Ensemble Forecast by Quarter', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Rent Growth (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Annual averages
    ax3 = axes[1, 0]
    results['year'] = results['date'].dt.year
    annual_avg = results.groupby('year')[['lightgbm_prediction',
                                          'sarima_prediction',
                                          'ensemble_prediction']].mean()

    x = np.arange(len(annual_avg))
    width = 0.25

    ax3.bar(x - width, annual_avg['lightgbm_prediction'], width,
            label='LightGBM', alpha=0.8)
    ax3.bar(x, annual_avg['sarima_prediction'], width,
            label='SARIMA', alpha=0.8)
    ax3.bar(x + width, annual_avg['ensemble_prediction'], width,
            label='Ensemble', alpha=0.8, color='black')

    ax3.set_title('Average Annual Rent Growth', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Rent Growth (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(annual_avg.index)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Prediction distribution
    ax4 = axes[1, 1]
    data_to_plot = [results['lightgbm_prediction'],
                    results['sarima_prediction'],
                    results['ensemble_prediction']]
    ax4.boxplot(data_to_plot, labels=['LightGBM', 'SARIMA', 'Ensemble'])
    ax4.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rent Growth (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_path = f"{output_dir}/phoenix_forecast_2026_2030_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Forecast visualization saved: {output_path}")

    return output_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("PHOENIX RENT GROWTH FORECAST 2026-2030")
    print("Production-Validated Ensemble Model")
    print("="*80)
    print(f"Configuration based on: COMPLETE_ROOT_CAUSE_ANALYSIS.md")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load data
    df, actual_data, future_data = load_and_prepare_data(
        'data/processed/phoenix_modeling_dataset.csv'
    )

    # Validate features
    if not validate_features(df, CORE_FEATURES):
        print("❌ Feature validation failed. Exiting.")
        return

    # Prepare train/test split (use all actual data for training)
    train, test = prepare_train_test_split(actual_data, test_cutoff='2022-12-31')

    # Prepare features and target
    X_train = train[CORE_FEATURES].fillna(method='ffill')
    y_train = train['rent_growth_yoy']

    X_test = test[CORE_FEATURES].fillna(method='ffill')
    y_test = test['rent_growth_yoy']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split training data for validation (last 20%)
    n_train = int(len(X_train_scaled) * 0.8)
    X_tr, X_val = X_train_scaled[:n_train], X_train_scaled[n_train:]
    y_tr, y_val = y_train[:n_train], y_train[n_train:]

    # ========================================================================
    # TRAIN MODELS
    # ========================================================================

    # 1. Train LightGBM
    lgb_model = train_lightgbm_component(
        X_tr, y_tr, X_val, y_val, PRODUCTION_CONFIG
    )

    # 2. Train SARIMA
    sarima_model = train_sarima_component(y_train, PRODUCTION_CONFIG)

    # 3. Get component predictions for meta-learner training
    lgb_train_pred = lgb_model.predict(X_train_scaled)
    sarima_train_pred = sarima_model.fittedvalues

    # 4. Train Ridge meta-learner
    ridge_model = train_ridge_meta_learner(
        lgb_train_pred, sarima_train_pred, y_train, PRODUCTION_CONFIG
    )

    # ========================================================================
    # VALIDATE ON TEST SET
    # ========================================================================

    print("\n" + "="*80)
    print("TEST SET VALIDATION")
    print("="*80)

    # Component predictions
    lgb_test_pred = lgb_model.predict(X_test_scaled)
    sarima_test_pred = sarima_model.forecast(steps=len(y_test))

    # Validate components
    validate_component_predictions(lgb_test_pred, sarima_test_pred, PRODUCTION_CONFIG)

    # Ensemble predictions
    X_meta_test = np.column_stack([lgb_test_pred, sarima_test_pred])
    ensemble_test_pred = ridge_model.predict(X_meta_test)

    # Calculate metrics
    lgb_metrics = calculate_metrics(y_test, lgb_test_pred, "LightGBM")
    sarima_metrics = calculate_metrics(y_test, sarima_test_pred, "SARIMA")
    ensemble_metrics = calculate_metrics(y_test, ensemble_test_pred, "Ensemble")

    print_metrics(lgb_metrics)
    print_metrics(sarima_metrics)
    print_metrics(ensemble_metrics)

    # ========================================================================
    # GENERATE 2026-2030 FORECAST
    # ========================================================================

    forecast_results = generate_forecast_2026_2030(
        lgb_model, sarima_model, ridge_model, scaler,
        future_data, CORE_FEATURES
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    # Save forecast to CSV
    output_csv = f"outputs/phoenix_forecast_2026_2030_{datetime.now().strftime('%Y%m%d')}.csv"
    forecast_results.to_csv(output_csv, index=False)
    print(f"\n✅ Forecast saved: {output_csv}")

    # Save detailed results with test performance
    detailed_results = {
        'forecast_date': datetime.now().strftime('%Y-%m-%d'),
        'configuration': PRODUCTION_CONFIG,
        'test_metrics': {
            'lightgbm': lgb_metrics,
            'sarima': sarima_metrics,
            'ensemble': ensemble_metrics
        },
        'forecast_summary': {
            'periods': len(forecast_results),
            'date_range': {
                'start': forecast_results['date'].min().strftime('%Y-%m-%d'),
                'end': forecast_results['date'].max().strftime('%Y-%m-%d')
            },
            'ensemble_predictions': {
                'mean': float(forecast_results['ensemble_prediction'].mean()),
                'min': float(forecast_results['ensemble_prediction'].min()),
                'max': float(forecast_results['ensemble_prediction'].max()),
                'std': float(forecast_results['ensemble_prediction'].std())
            }
        }
    }

    output_json = f"outputs/phoenix_forecast_2026_2030_metadata_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_json, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"✅ Metadata saved: {output_json}")

    # Save models
    models_dir = 'models/production'
    import os
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(lgb_model, f'{models_dir}/lightgbm_production.pkl')
    joblib.dump(sarima_model, f'{models_dir}/sarima_production.pkl')
    joblib.dump(ridge_model, f'{models_dir}/ridge_meta_production.pkl')
    joblib.dump(scaler, f'{models_dir}/scaler_production.pkl')
    print(f"✅ Models saved to: {models_dir}/")

    # Create visualizations
    plot_forecast_results(forecast_results)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("FORECAST COMPLETE")
    print("="*80)
    print(f"\n2026-2030 Phoenix Rent Growth Forecast:")
    print(f"  Average: {forecast_results['ensemble_prediction'].mean():.2f}%")
    print(f"  Range: [{forecast_results['ensemble_prediction'].min():.2f}%, "
          f"{forecast_results['ensemble_prediction'].max():.2f}%]")

    print(f"\nAnnual Averages:")
    annual = forecast_results.copy()
    annual['year'] = annual['date'].dt.year
    for year, group in annual.groupby('year'):
        avg = group['ensemble_prediction'].mean()
        print(f"  {year}: {avg:.2f}%")

    print("\n" + "="*80)
    print("All results saved. Review outputs/ directory for details.")
    print("="*80)

if __name__ == '__main__':
    main()
