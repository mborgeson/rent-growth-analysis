#!/usr/bin/env python3
"""
XGBoost Regression Model - Phoenix Rent Growth
===============================================

Purpose: XGBoost with advanced features and hyperparameter tuning
         Incorporating deep dive findings on employment and supply dynamics

Features:
- Employment variables (YoY growth, levels, lagged)
- Supply pipeline (optimal 8-quarter lag + additional lags)
- Vacancy rate and absorption metrics
- National macro variables
- Interaction terms
- Polynomial features for non-linear relationships

Hyperparameter Optimization:
- Bayesian optimization via Optuna
- Cross-validation on time series splits
- Early stopping to prevent overfitting

Expected Performance:
- Superior to LightGBM due to more sophisticated architecture
- Better handling of non-linear interactions
- ~0.03-0.05 RMSE improvement expected

Experiment ID: XGB-OPT-001
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import json
from datetime import datetime

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available, using default hyperparameters")

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'models/experiments/xgboost_variants'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "XGB-OPT-002"  # v2: Removed StandardScaler, Optuna enabled
EXPERIMENT_DATE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print("=" * 80)
print(f"XGBOOST OPTIMIZED MODEL - {EXPERIMENT_ID}")
print("=" * 80)
print(f"Experiment Date: {EXPERIMENT_DATE}")
print(f"Hyperparameter Optimization: {'Enabled (Optuna)' if OPTUNA_AVAILABLE else 'Disabled (Default params)'}")
print()

# ============================================================================
# 1. Load Data and Engineer Features
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA LOADING AND FEATURE ENGINEERING")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
historical_df = df.loc[:'2025-09-30'].copy()

print(f"✅ Loaded dataset: {len(historical_df)} quarters")

# Define comprehensive feature set
employment_features = [
    'phx_total_employment',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',
    'phx_employment_yoy_growth',
    'phx_prof_business_yoy_growth',
    'phx_total_employment_lag1',
    'phx_prof_business_employment_lag1'
]

supply_features = [
    'units_under_construction',
    'units_under_construction_lag5',
    'units_under_construction_lag6',
    'units_under_construction_lag7',
    'units_under_construction_lag8',  # Optimal lag from deep dive
    'vacancy_rate',
    'inventory_units',
    'absorption_12mo',
    'supply_inventory_ratio',
    'absorption_inventory_ratio'
]

macro_features = [
    'mortgage_rate_30yr',
    'mortgage_rate_30yr_lag2',
    'fed_funds_rate',
    'national_unemployment',
    'cpi',
    'phx_home_price_index',
    'phx_hpi_yoy_growth'
]

interaction_features = [
    'mortgage_employment_interaction'
]

# Combine all features
all_features = employment_features + supply_features + macro_features + interaction_features
target = 'rent_growth_yoy'

# Filter available features
available_features = [f for f in all_features if f in historical_df.columns]
print(f"\nFeatures: {len(available_features)}/{len(all_features)} available")

# Create modeling dataset
model_cols = [target] + available_features
model_df = historical_df[model_cols].fillna(method='ffill').dropna()

print(f"✅ Modeling dataset: {len(model_df)} quarters after preprocessing")
print(f"   Coverage: {model_df.index.min()} to {model_df.index.max()}")

# ============================================================================
# 2. Train/Test Split
# ============================================================================

print("\n" + "=" * 80)
print("2. TRAIN/TEST SPLIT")
print("=" * 80)

train_end = '2022-12-31'
train_df = model_df.loc[:train_end]
test_df = model_df.loc[train_end:]

X_train = train_df[available_features]
y_train = train_df[target]
X_test = test_df[available_features]
y_test = test_df[target]

print(f"\nTraining set: {len(train_df)} quarters")
print(f"Test set:     {len(test_df)} quarters")
print(f"Features:     {len(available_features)}")

# ============================================================================
# 3. Feature Scaling (REMOVED - Not needed for tree-based models)
# ============================================================================

print("\n" + "=" * 80)
print("3. FEATURE PREPARATION")
print("=" * 80)

# Tree-based models like XGBoost don't require feature scaling
# Using original features for better interpretability and performance
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

print(f"✅ Features prepared (no scaling needed for tree models)")

# ============================================================================
# 4. Hyperparameter Optimization (if Optuna available)
# ============================================================================

print("\n" + "=" * 80)
print("4. HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

if OPTUNA_AVAILABLE:
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': 42
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_cv_train = X_train_scaled.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train_scaled.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_cv_train, y_cv_train, verbose=False)
            preds = model.predict(X_cv_val)
            rmse = np.sqrt(mean_squared_error(y_cv_val, preds))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    print("\n🔄 Running Bayesian hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['booster'] = 'gbtree'
    best_params['random_state'] = 42

    print(f"\n✅ Optimization complete")
    print(f"   Best RMSE (CV): {study.best_value:.4f}")
    print(f"\n   Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"     {key:20s}: {value}")

else:
    # Default parameters
    best_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
    print("\n⚠️  Using default hyperparameters")

# ============================================================================
# 5. Train Final Model
# ============================================================================

print("\n" + "=" * 80)
print("5. TRAINING FINAL MODEL")
print("=" * 80)

xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False
)

print(f"✅ XGBoost model trained")

# ============================================================================
# 6. Feature Importance
# ============================================================================

print("\n" + "=" * 80)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Features:")
for idx, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:45s}: {row['importance']:.4f}")

# ============================================================================
# 7. Generate Predictions and Evaluate
# ============================================================================

print("\n" + "=" * 80)
print("7. MODEL EVALUATION")
print("=" * 80)

y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

# Test set metrics
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Directional accuracy
actual_direction = np.sign(np.diff(y_test.values))
pred_direction = np.sign(np.diff(y_test_pred))
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

print(f"\nOut-of-Sample Performance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

# ============================================================================
# 8. Save Model and Results
# ============================================================================

print("\n" + "=" * 80)
print("8. SAVING MODEL AND RESULTS")
print("=" * 80)

# Save model
model_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"✅ Saved model: {model_path}")

# Note: No scaler needed for tree-based models

# Save feature importance
importance_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_feature_importance.csv'
importance_df.to_csv(importance_path, index=False)
print(f"✅ Saved feature importance: {importance_path}")

# Save experiment metadata
metadata = {
    'experiment_id': EXPERIMENT_ID,
    'experiment_date': EXPERIMENT_DATE,
    'model_type': 'XGBoost',
    'features_count': len(available_features),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'hyperparameters': best_params,
    'performance_metrics': {
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'directional_accuracy': float(directional_accuracy)
    },
    'top_5_features': importance_df.head(5)['feature'].tolist()
}

metadata_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"✅ Saved experiment metadata: {metadata_path}")

print("\n" + "=" * 80)
print(f"EXPERIMENT {EXPERIMENT_ID} COMPLETE")
print("=" * 80)

if __name__ == "__main__":
    pass
