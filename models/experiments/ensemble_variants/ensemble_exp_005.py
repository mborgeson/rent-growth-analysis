#!/usr/bin/env python3
"""
ENSEMBLE-EXP-005: Early Stopping Hypothesis Test
Phoenix Rent Growth Forecasting

Purpose: Test whether production-style early stopping closes the performance gap
Key Change: Use lgb.train() with early stopping (like production) instead of LGBMRegressor()

Configuration:
  - LightGBM with lgb.train() + early_stopping(50 rounds) ⚡ KEY CHANGE
  - Pure SARIMA (same as EXP-003)
  - Ridge meta-learner (same as EXP-003)
  - StandardScaler (same as EXP-003)

Hypothesis: Early stopping will prevent overfitting and enable component success
Expected: Ensemble RMSE ~0.50-0.55 (vs EXP-003 0.5936, production 0.5046)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import ks_2samp
import json
import pickle
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'models/experiments/ensemble_variants'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ENSEMBLE-EXP-005: EARLY STOPPING HYPOTHESIS TEST")
print("=" * 80)
print("Architecture: LightGBM (lgb.train + early stopping) + Pure SARIMA + Ridge")
print("Hypothesis: Early stopping prevents overfitting and enables component success")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
historical_df = df.loc[:'2025-09-30'].copy()

print(f"✅ Loaded dataset: {len(historical_df)} quarters")
print(f"   Period: {historical_df.index.min()} to {historical_df.index.max()}")

# ============================================================================
# 2. Define Production Feature Set (26 Features)
# ============================================================================

print("\n" + "=" * 80)
print("PRODUCTION FEATURE SET (26 Features)")
print("=" * 80)

# Exact production GBM features
phoenix_features = [
    # Employment (7 features)
    'phx_total_employment',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',
    'phx_prof_business_employment_lag1',
    'phx_total_employment_lag1',
    'phx_employment_yoy_growth',
    'phx_prof_business_yoy_growth',

    # Supply Pipeline (4 features)
    'units_under_construction_lag5',
    'units_under_construction_lag6',
    'units_under_construction_lag7',
    'units_under_construction_lag8',

    # Market Conditions (4 features)
    'vacancy_rate',
    'inventory_units',
    'absorption_12mo',
    'cap_rate',

    # Supply/Demand Ratios (2 features)
    'supply_inventory_ratio',
    'absorption_inventory_ratio',

    # Phoenix Home Prices (2 features)
    'phx_home_price_index',
    'phx_hpi_yoy_growth',

    # Migration (1 feature)
    'migration_proxy',

    # National Factors (5 features)
    'mortgage_rate_30yr',
    'mortgage_rate_30yr_lag2',
    'fed_funds_rate',
    'national_unemployment',
    'cpi',

    # Interactions (1 feature)
    'mortgage_employment_interaction'
]

# Check availability
available_features = [col for col in phoenix_features if col in historical_df.columns]
print(f"\nAvailable features: {len(available_features)}/{len(phoenix_features)}")

# ============================================================================
# 3. Prepare Data Split
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA SPLIT")
print("=" * 80)

target = 'rent_growth_yoy'
feature_df = historical_df[available_features + [target]].copy()
feature_df = feature_df.fillna(method='ffill').dropna()

# Train/test split (same as EXP-003 and production)
train_end_date = '2022-12-31'
train_mask = feature_df.index <= train_end_date
test_mask = feature_df.index > train_end_date

train_df = feature_df.loc[train_mask]
test_df = feature_df.loc[test_mask]

print(f"\nData Split:")
print(f"  Train: {len(train_df)} quarters ({train_df.index.min()} to {train_df.index.max()})")
print(f"  Test:  {len(test_df)} quarters ({test_df.index.min()} to {test_df.index.max()})")

# Prepare component data
X_train_lgbm = train_df[available_features]
y_train_lgbm = train_df[target]
X_test_lgbm = test_df[available_features]
y_test_lgbm = test_df[target]

train_ts_sarima = train_df[target]
test_ts_sarima = test_df[target]

# ============================================================================
# 4. Component 1: LightGBM with Production-Style Early Stopping
# ============================================================================

print("\n" + "=" * 80)
print("COMPONENT 1: LIGHTGBM (PRODUCTION CONFIG WITH EARLY STOPPING)")
print("=" * 80)

# Production hyperparameters (same as production gbm_phoenix_specific.py)
lgb_params = {
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
}

print("\n⚡ KEY CHANGE: Using lgb.train() with early_stopping (like production)")
print("\nLightGBM Hyperparameters:")
for param, value in lgb_params.items():
    print(f"  {param:25s}: {value}")

# StandardScaler (like production and EXP-003)
print("\n🔄 Applying StandardScaler...")
scaler_lgbm = StandardScaler()
X_train_lgbm_scaled = scaler_lgbm.fit_transform(X_train_lgbm)
X_test_lgbm_scaled = scaler_lgbm.transform(X_test_lgbm)

# Convert to DataFrames for feature names
X_train_lgbm_scaled_df = pd.DataFrame(X_train_lgbm_scaled, columns=available_features, index=X_train_lgbm.index)
X_test_lgbm_scaled_df = pd.DataFrame(X_test_lgbm_scaled, columns=available_features, index=X_test_lgbm.index)

# Create LightGBM datasets
print("🔄 Creating LightGBM datasets...")
train_data = lgb.Dataset(X_train_lgbm_scaled_df, label=y_train_lgbm)
test_data = lgb.Dataset(X_test_lgbm_scaled_df, label=y_test_lgbm, reference=train_data)

# Train model with early stopping (EXACT production approach)
print("🔄 Training LightGBM with early stopping...")
gbm_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,  # Maximum iterations
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # Suppress iteration logs
    ]
)

print(f"\n✅ LightGBM Component Trained with Early Stopping")
print(f"   Best iteration: {gbm_model.best_iteration}")
print(f"   Training RMSE: {gbm_model.best_score['train']['rmse']:.4f}")
print(f"   Validation RMSE: {gbm_model.best_score['test']['rmse']:.4f}")

# Generate predictions
lgbm_train_pred = gbm_model.predict(X_train_lgbm_scaled_df, num_iteration=gbm_model.best_iteration)
lgbm_test_pred = gbm_model.predict(X_test_lgbm_scaled_df, num_iteration=gbm_model.best_iteration)

# Evaluate
lgbm_train_rmse = np.sqrt(mean_squared_error(y_train_lgbm, lgbm_train_pred))
lgbm_train_r2 = r2_score(y_train_lgbm, lgbm_train_pred)
lgbm_test_rmse = np.sqrt(mean_squared_error(y_test_lgbm, lgbm_test_pred))
lgbm_test_r2 = r2_score(y_test_lgbm, lgbm_test_pred)

print(f"\nLightGBM Performance:")
print(f"  Training RMSE: {lgbm_train_rmse:.4f}")
print(f"  Training R²:   {lgbm_train_r2:.4f}")
print(f"  Test RMSE:     {lgbm_test_rmse:.4f}")
print(f"  Test R²:       {lgbm_test_r2:.4f}")
print(f"  Train/Test RMSE Ratio: {lgbm_test_rmse/lgbm_train_rmse:.2f}x")

# Feature importance
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': gbm_model.feature_importance(importance_type='gain'),
    'split_importance': gbm_model.feature_importance(importance_type='split')
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features by Importance:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:45s}: {row['importance']:8.0f} gain ({row['split_importance']:4.0f} splits)")

# ============================================================================
# 5. Component 2: Pure SARIMA (Same as EXP-003)
# ============================================================================

print("\n" + "=" * 80)
print("COMPONENT 2: PURE SARIMA (GRID SEARCH)")
print("=" * 80)

# Grid search over SARIMA orders (same as EXP-003)
p_range = [0, 1, 2]
d_range = [1]
q_range = [0, 1, 2]
P_range = [0, 1]
D_range = [0, 1]
Q_range = [0, 1]
s = 4  # Quarterly seasonality

best_aic = np.inf
best_order = None
best_seasonal_order = None
best_model = None

print("\n🔄 Running grid search for best SARIMA configuration...")
print("   Testing configurations (will take a few minutes)...")

results = []
for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        try:
                            model = SARIMAX(
                                train_ts_sarima,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fitted = model.fit(disp=False, maxiter=200)

                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_seasonal_order = (P, D, Q, s)
                                best_model = fitted

                            results.append({
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, s),
                                'aic': fitted.aic
                            })
                        except:
                            continue

print(f"\n✅ Grid search complete. Tested {len(results)} configurations")
print(f"\nBest SARIMA Configuration:")
print(f"  Order: {best_order}")
print(f"  Seasonal Order: {best_seasonal_order}")
print(f"  AIC: {best_aic:.2f}")

# Generate SARIMA predictions
sarima_train_pred = best_model.fittedvalues
sarima_test_pred = best_model.forecast(steps=len(test_ts_sarima))

# Evaluate SARIMA
sarima_train_rmse = np.sqrt(mean_squared_error(train_ts_sarima, sarima_train_pred))
sarima_train_r2 = r2_score(train_ts_sarima, sarima_train_pred)
sarima_test_rmse = np.sqrt(mean_squared_error(test_ts_sarima, sarima_test_pred))
sarima_test_r2 = r2_score(test_ts_sarima, sarima_test_pred)

print(f"\nSARIMA Performance:")
print(f"  Training RMSE: {sarima_train_rmse:.4f}")
print(f"  Training R²:   {sarima_train_r2:.4f}")
print(f"  Test RMSE:     {sarima_test_rmse:.4f}")
print(f"  Test R²:       {sarima_test_r2:.4f}")

# ============================================================================
# 6. Meta-Learner: Ridge Regression (Same as EXP-003)
# ============================================================================

print("\n" + "=" * 80)
print("META-LEARNER: RIDGE REGRESSION")
print("=" * 80)

# Create meta-features (component predictions)
meta_train = pd.DataFrame({
    'lightgbm': lgbm_train_pred,
    'sarima': sarima_train_pred
}, index=train_df.index)

meta_test = pd.DataFrame({
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred
}, index=test_df.index)

# Ridge with cross-validation (same alpha range as EXP-003)
print("\n🔄 Training Ridge meta-learner with cross-validation...")
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_meta = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
ridge_meta.fit(meta_train, y_train_lgbm)

print(f"\n✅ Ridge Meta-Learner Trained")
print(f"   Best alpha: {ridge_meta.alpha_}")
print(f"\nLearned Weights:")
print(f"  LightGBM:  {ridge_meta.coef_[0]:8.4f}")
print(f"  SARIMA:    {ridge_meta.coef_[1]:8.4f}")
print(f"  Intercept: {ridge_meta.intercept_:8.4f}")

# Normalized percentage (excluding intercept)
total_coef = abs(ridge_meta.coef_[0]) + abs(ridge_meta.coef_[1])
if total_coef > 0:
    lgbm_pct = abs(ridge_meta.coef_[0]) / total_coef * 100
    sarima_pct = abs(ridge_meta.coef_[1]) / total_coef * 100
else:
    lgbm_pct = 0.0
    sarima_pct = 0.0

print(f"\nNormalized Component Weights:")
print(f"  LightGBM: {lgbm_pct:5.1f}%")
print(f"  SARIMA:   {sarima_pct:5.1f}%")

# Generate ensemble predictions
ensemble_train_pred = ridge_meta.predict(meta_train)
ensemble_test_pred = ridge_meta.predict(meta_test)

# ============================================================================
# 7. Ensemble Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE PERFORMANCE EVALUATION")
print("=" * 80)

# Calculate metrics
ensemble_test_rmse = np.sqrt(mean_squared_error(y_test_lgbm, ensemble_test_pred))
ensemble_test_mae = mean_absolute_error(y_test_lgbm, ensemble_test_pred)
ensemble_test_r2 = r2_score(y_test_lgbm, ensemble_test_pred)

# Directional accuracy
actual_direction = np.sign(np.diff(y_test_lgbm.values))
predicted_direction = np.sign(np.diff(ensemble_test_pred))
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

print(f"\nEnsemble Test Performance:")
print(f"  RMSE: {ensemble_test_rmse:.4f}")
print(f"  MAE:  {ensemble_test_mae:.4f}")
print(f"  R²:   {ensemble_test_r2:.4f}")
print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

# Compare to best component
best_component_rmse = min(lgbm_test_rmse, sarima_test_rmse)
improvement = (best_component_rmse - ensemble_test_rmse) / best_component_rmse * 100

print(f"\nImprovement vs Best Component:")
print(f"  Best component RMSE: {best_component_rmse:.4f}")
print(f"  Ensemble RMSE: {ensemble_test_rmse:.4f}")
print(f"  Improvement: {improvement:.1f}%")

# Compare to benchmarks
print(f"\nComparison to Benchmarks:")
print(f"  Production RMSE:     0.5046")
print(f"  EXP-003 RMSE:        0.5936")
print(f"  EXP-005 RMSE:        {ensemble_test_rmse:.4f}")
print(f"  Gap to Production:   {(ensemble_test_rmse - 0.5046) / 0.5046 * 100:+.1f}%")
print(f"  Gap to EXP-003:      {(ensemble_test_rmse - 0.5936) / 0.5936 * 100:+.1f}%")

# ============================================================================
# 8. Save Results
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save predictions
predictions_df = pd.DataFrame({
    'date': test_df.index,
    'actual': y_test_lgbm.values,
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred,
    'ensemble': ensemble_test_pred
})
predictions_path = OUTPUT_PATH / 'ENSEMBLE-EXP-005_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✅ Predictions saved: {predictions_path}")

# Save metadata
metadata = {
    'experiment_id': 'ENSEMBLE-EXP-005',
    'description': 'Early stopping hypothesis test - lgb.train() with early_stopping(50)',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'key_change': 'Using lgb.train() with early stopping instead of LGBMRegressor()',
    'architecture': {
        'component_1': 'LightGBM (lgb.train + early stopping)',
        'component_2': 'SARIMA (Pure)',
        'meta_learner': 'Ridge Regression'
    },
    'features': {
        'count': len(available_features),
    },
    'lightgbm': {
        'algorithm': 'LightGBM (lgb.train)',
        'n_features': len(available_features),
        'early_stopping': True,
        'stopping_rounds': 50,
        'best_iteration': int(gbm_model.best_iteration),
        'hyperparameters': lgb_params,
        'train_rmse': float(lgbm_train_rmse),
        'train_r2': float(lgbm_train_r2),
        'test_rmse': float(lgbm_test_rmse),
        'test_r2': float(lgbm_test_r2),
        'train_test_ratio': float(lgbm_test_rmse / lgbm_train_rmse)
    },
    'sarima': {
        'type': 'Pure SARIMA (no exogenous)',
        'order': best_order,
        'seasonal_order': best_seasonal_order,
        'aic': float(best_aic),
        'train_rmse': float(sarima_train_rmse),
        'train_r2': float(sarima_train_r2),
        'test_rmse': float(sarima_test_rmse),
        'test_r2': float(sarima_test_r2)
    },
    'ridge_meta': {
        'alpha': float(ridge_meta.alpha_),
        'weights': {
            'lightgbm': float(ridge_meta.coef_[0]),
            'sarima': float(ridge_meta.coef_[1]),
            'intercept': float(ridge_meta.intercept_)
        },
        'normalized_pct': {
            'lightgbm': float(lgbm_pct),
            'sarima': float(sarima_pct)
        }
    },
    'ensemble_performance': {
        'test_rmse': float(ensemble_test_rmse),
        'test_mae': float(ensemble_test_mae),
        'test_r2': float(ensemble_test_r2),
        'directional_accuracy': float(directional_accuracy),
        'improvement_vs_best_component': float(improvement)
    },
    'comparison': {
        'production_rmse': 0.5046,
        'exp003_rmse': 0.5936,
        'exp005_rmse': float(ensemble_test_rmse),
        'gap_to_production_pct': float((ensemble_test_rmse - 0.5046) / 0.5046 * 100),
        'improvement_vs_exp003_pct': float((0.5936 - ensemble_test_rmse) / 0.5936 * 100)
    }
}

metadata_path = OUTPUT_PATH / 'ENSEMBLE-EXP-005_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# Save models
lgbm_path = OUTPUT_PATH / 'ENSEMBLE-EXP-005_lgbm_component.pkl'
with open(lgbm_path, 'wb') as f:
    pickle.dump({
        'model': gbm_model,
        'scaler': scaler_lgbm,
        'feature_names': available_features
    }, f)
print(f"✅ LightGBM model saved: {lgbm_path}")

sarima_path = OUTPUT_PATH / 'ENSEMBLE-EXP-005_sarima_component.pkl'
with open(sarima_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ SARIMA model saved: {sarima_path}")

ridge_path = OUTPUT_PATH / 'ENSEMBLE-EXP-005_ridge_meta.pkl'
with open(ridge_path, 'wb') as f:
    pickle.dump(ridge_meta, f)
print(f"✅ Ridge meta-learner saved: {ridge_path}")

print("\n" + "=" * 80)
print("ENSEMBLE-EXP-005 COMPLETE")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  ⚡ Early Stopping Best Iteration: {gbm_model.best_iteration}")
print(f"  📊 LightGBM Test R²: {lgbm_test_r2:.4f} (vs EXP-003: -79.86)")
print(f"  🎯 Ensemble Test RMSE: {ensemble_test_rmse:.4f}")
print(f"  📈 Gap to Production: {(ensemble_test_rmse - 0.5046) / 0.5046 * 100:+.1f}%")
print(f"  ✨ Improvement vs EXP-003: {(0.5936 - ensemble_test_rmse) / 0.5936 * 100:+.1f}%")
print("\n" + "=" * 80)
