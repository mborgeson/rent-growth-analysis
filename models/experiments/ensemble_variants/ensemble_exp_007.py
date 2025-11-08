#!/usr/bin/env python3
"""
ENSEMBLE-EXP-007: SARIMA Configuration Hypothesis Test
Phoenix Rent Growth Forecasting

Purpose: Validate that SARIMA configuration difference is THE root cause of 1195% performance gap
Key Change: Use production's SARIMA order (1,1,2)(0,0,1,4) instead of experimental (2,1,2)(1,1,1,4)

ROOT CAUSE HYPOTHESIS:
The entire 1195% performance gap between production (0.5046 RMSE) and experimental (6.5338 RMSE)
is explained by a SINGLE difference in SARIMA configuration:

Production SARIMA:    (1,1,2)(0,0,1,4) → STABLE predictions  → RMSE 5.71  → Ensemble 0.50
Experimental SARIMA:  (2,1,2)(1,1,1,4) → EXPLOSIVE predictions → RMSE 16.91 → Ensemble 6.53

Key Differences:
1. AR Order: Production uses AR(1), Experimental uses AR(2) - extra lag causes instability
2. Seasonal Differencing: Production D=0, Experimental D=1 - seasonal differencing causes explosions
3. Seasonal AR: Production P=0, Experimental P=1 - seasonal AR creates feedback loops

All other factors are IDENTICAL:
- ✅ Same 26 features (including all macroeconomic regime indicators)
- ✅ Same data source and preprocessing
- ✅ Same LightGBM configuration and predictions (4.1058 RMSE - IDENTICAL)
- ✅ Same ensemble architecture (Ridge meta-learner)

Expected Outcome:
- SARIMA RMSE: ~5.71 (matching production)
- Ensemble RMSE: ~0.50-0.55 (closing gap to production's 0.5046)
- Gap to production: <10%

If successful, this validates that SARIMA configuration is THE root cause.

Comparison to Previous Experiments:
- EXP-003: 0.5936 RMSE (17.7% gap) - different ensemble architecture
- EXP-005: 6.5338 RMSE (1194.8% gap) - explosive SARIMA (2,1,2)(1,1,1,4)
- EXP-006: 6.5338 RMSE (identical to EXP-005) - proved features already aligned
- EXP-007: Testing production SARIMA config as THE root cause
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE-EXP-007: SARIMA CONFIGURATION HYPOTHESIS TEST")
print("=" * 80)
print("\nTesting ROOT CAUSE hypothesis: Production SARIMA configuration (1,1,2)(0,0,1,4)")
print("is THE single factor explaining the 1195% performance gap.")
print("\nKey Change: SARIMA order from (2,1,2)(1,1,1,4) → (1,1,2)(0,0,1,4)")
print("\nAll other factors IDENTICAL to EXP-005/006:")
print("  - Same 26 features (production feature set)")
print("  - Same LightGBM configuration with early stopping")
print("  - Same ensemble architecture (Ridge meta-learner)")

# Load processed data
historical_df = pd.read_csv('/home/mattb/Rent Growth Analysis/data/processed/phoenix_modeling_dataset.csv',
                           index_col='date', parse_dates=True)

print(f"\nLoaded data: {len(historical_df)} total quarters")
print(f"Date range: {historical_df.index.min()} to {historical_df.index.max()}")

# ============================================================================
# 2. Production Feature Set (EXACT MATCH to production model)
# ============================================================================

print("\n" + "=" * 80)
print("PRODUCTION FEATURE SET (26 features)")
print("=" * 80)

# EXACT production features (identical to EXP-005/006)
production_features = [
    # Employment Indicators (7 features)
    'phx_total_employment',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',
    'phx_prof_business_employment_lag1',
    'phx_total_employment_lag1',
    'phx_employment_yoy_growth',
    'phx_prof_business_yoy_growth',

    # Construction Pipeline (4 features)
    'units_under_construction_lag5',
    'units_under_construction_lag6',
    'units_under_construction_lag7',
    'units_under_construction_lag8',

    # Property Metrics (5 features)
    'vacancy_rate',
    'inventory_units',
    'absorption_12mo',
    'cap_rate',
    'supply_inventory_ratio',

    # Derived Ratios (1 feature)
    'absorption_inventory_ratio',

    # Housing Market (2 features)
    'phx_home_price_index',
    'phx_hpi_yoy_growth',

    # Migration (1 feature)
    'migration_proxy',

    # Interest Rates (3 features)
    'mortgage_rate_30yr',
    'mortgage_rate_30yr_lag2',
    'fed_funds_rate',

    # Macroeconomic (2 features)
    'national_unemployment',
    'cpi',

    # Interaction (1 feature)
    'mortgage_employment_interaction'
]

# Verify feature availability
available_features = [col for col in production_features if col in historical_df.columns]
missing_features = set(production_features) - set(available_features)

print(f"\nProduction features: {len(production_features)}")
print(f"Available features: {len(available_features)}/{len(production_features)}")

if missing_features:
    print(f"\n⚠️  MISSING FEATURES: {missing_features}")
    raise ValueError("Cannot proceed - missing critical production features")

print(f"\n✅ All production features available!")

# ============================================================================
# 3. Prepare Data Split
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA SPLIT")
print("=" * 80)

target = 'rent_growth_yoy'
feature_df = historical_df[available_features + [target]].copy()

# Forward fill nulls (same as production)
feature_df = feature_df.fillna(method='ffill').dropna()

print(f"\nData after preprocessing: {len(feature_df)} quarters")

# Train/test split (same as production and all experiments)
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

print(f"\nRegime Analysis:")
print(f"  Train Mean: {y_train_lgbm.mean():.4f}% (training regime)")
print(f"  Test Mean:  {y_test_lgbm.mean():.4f}% (test regime)")
print(f"  Regime Shift: Δ {y_test_lgbm.mean() - y_train_lgbm.mean():.4f}%")

# ============================================================================
# 4. Component 1: LightGBM with Production Config + Production Features
# ============================================================================

print("\n" + "=" * 80)
print("COMPONENT 1: LIGHTGBM (PRODUCTION CONFIG + PRODUCTION FEATURES)")
print("=" * 80)

# Production hyperparameters (identical to production and all experiments)
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

print("\n📋 LightGBM Configuration:")
print(f"  Features: {len(available_features)} (PRODUCTION SET)")
print(f"  Training Method: lgb.train() with early stopping")
print(f"  Early Stopping: 50 rounds")
print(f"  Hyperparameters: Production-matched")

# Apply StandardScaler (same as production)
scaler_lgbm = StandardScaler()
X_train_lgbm_scaled = scaler_lgbm.fit_transform(X_train_lgbm)
X_test_lgbm_scaled = scaler_lgbm.transform(X_test_lgbm)

# Convert to DataFrame (preserve feature names for LightGBM)
X_train_lgbm_scaled_df = pd.DataFrame(X_train_lgbm_scaled,
                                      index=X_train_lgbm.index,
                                      columns=X_train_lgbm.columns)
X_test_lgbm_scaled_df = pd.DataFrame(X_test_lgbm_scaled,
                                     index=X_test_lgbm.index,
                                     columns=X_test_lgbm.columns)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_lgbm_scaled_df, label=y_train_lgbm)
test_data = lgb.Dataset(X_test_lgbm_scaled_df, label=y_test_lgbm, reference=train_data)

# Train model with early stopping
print("\n🔄 Training LightGBM model...")
gbm_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

print(f"✅ LightGBM model trained")
print(f"   Best iteration: {gbm_model.best_iteration}")
print(f"   Training RMSE: {gbm_model.best_score['train']['rmse']:.4f}")
print(f"   Validation RMSE: {gbm_model.best_score['test']['rmse']:.4f}")

# Generate predictions
lgbm_train_pred = gbm_model.predict(X_train_lgbm_scaled_df, num_iteration=gbm_model.best_iteration)
lgbm_test_pred = gbm_model.predict(X_test_lgbm_scaled_df, num_iteration=gbm_model.best_iteration)

# Calculate metrics
lgbm_train_rmse = np.sqrt(mean_squared_error(y_train_lgbm, lgbm_train_pred))
lgbm_train_r2 = r2_score(y_train_lgbm, lgbm_train_pred)
lgbm_test_rmse = np.sqrt(mean_squared_error(y_test_lgbm, lgbm_test_pred))
lgbm_test_r2 = r2_score(y_test_lgbm, lgbm_test_pred)

print(f"\n📊 LightGBM Performance:")
print(f"   Train RMSE: {lgbm_train_rmse:.4f}")
print(f"   Train R²:   {lgbm_train_r2:.4f}")
print(f"   Test RMSE:  {lgbm_test_rmse:.4f}")
print(f"   Test R²:    {lgbm_test_r2:.4f}")
print(f"   Train/Test Ratio: {lgbm_test_rmse/lgbm_train_rmse:.2f}×")

# ============================================================================
# 5. Component 2: PRODUCTION SARIMA CONFIGURATION ✅ KEY CHANGE
# ============================================================================

print("\n" + "=" * 80)
print("COMPONENT 2: PRODUCTION SARIMA CONFIGURATION ✅ KEY CHANGE")
print("=" * 80)

# ✅ USE PRODUCTION SARIMA CONFIG (ROOT CAUSE TEST)
sarima_order = (1, 1, 2)           # ✅ PRODUCTION: AR(1) instead of AR(2)
sarima_seasonal = (0, 0, 1, 4)     # ✅ PRODUCTION: No seasonal AR or differencing

print(f"\n🎯 TESTING PRODUCTION SARIMA CONFIGURATION:")
print(f"  Order: {sarima_order}")
print(f"  Seasonal Order: {sarima_seasonal}")
print(f"  Exogenous: None (pure SARIMA)")
print(f"\n  Key Differences from EXP-005/006:")
print(f"    AR Order: (1) instead of (2) - simpler autoregressive structure")
print(f"    Seasonal AR (P): 0 instead of 1 - no multiplicative feedback")
print(f"    Seasonal Differencing (D): 0 instead of 1 - preserves level stability")

# Fit SARIMA
print("\n🔄 Training SARIMA model with PRODUCTION configuration...")
sarima_model = SARIMAX(train_ts_sarima,
                       order=sarima_order,
                       seasonal_order=sarima_seasonal,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_fit = sarima_model.fit(disp=False)

print(f"✅ SARIMA model trained")
print(f"   AIC: {sarima_fit.aic:.2f}")

# Generate predictions
sarima_train_pred = sarima_fit.fittedvalues
sarima_test_pred = sarima_fit.forecast(steps=len(test_ts_sarima))

# Calculate metrics
sarima_train_rmse = np.sqrt(mean_squared_error(train_ts_sarima, sarima_train_pred))
sarima_train_r2 = r2_score(train_ts_sarima, sarima_train_pred)
sarima_test_rmse = np.sqrt(mean_squared_error(test_ts_sarima, sarima_test_pred))
sarima_test_r2 = r2_score(test_ts_sarima, sarima_test_pred)

print(f"\n📊 SARIMA Performance:")
print(f"   Train RMSE: {sarima_train_rmse:.4f}")
print(f"   Train R²:   {sarima_train_r2:.4f}")
print(f"   Test RMSE:  {sarima_test_rmse:.4f}")
print(f"   Test R²:    {sarima_test_r2:.4f}")

# Compare to production and experimental SARIMA
production_sarima_rmse = 5.7122
exp006_sarima_rmse = 16.9081

print(f"\n📈 SARIMA Comparison:")
print(f"   Production SARIMA (1,1,2)(0,0,1,4): {production_sarima_rmse:.4f} RMSE")
print(f"   EXP-005/006 SARIMA (2,1,2)(1,1,1,4): {exp006_sarima_rmse:.4f} RMSE")
print(f"   EXP-007 SARIMA (1,1,2)(0,0,1,4):     {sarima_test_rmse:.4f} RMSE")
print(f"\n   Difference from Production: {((sarima_test_rmse - production_sarima_rmse) / production_sarima_rmse) * 100:+.1f}%")
print(f"   Improvement vs EXP-006: {((sarima_test_rmse - exp006_sarima_rmse) / exp006_sarima_rmse) * 100:+.1f}%")

# ============================================================================
# 6. Meta-Learner: Ridge Regression with Time Series CV
# ============================================================================

print("\n" + "=" * 80)
print("META-LEARNER: RIDGE REGRESSION")
print("=" * 80)

# Prepare meta-features (component predictions on training data)
meta_features_train = pd.DataFrame({
    'lightgbm': lgbm_train_pred,
    'sarima': sarima_train_pred.values
}, index=train_ts_sarima.index)

meta_features_test = pd.DataFrame({
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred.values
}, index=test_ts_sarima.index)

# Ridge with time series cross-validation
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                cv=TimeSeriesSplit(n_splits=5))

print("\n🔄 Training Ridge meta-learner...")
ridge.fit(meta_features_train, y_train_lgbm)

print(f"✅ Ridge meta-learner trained")
print(f"   Best alpha: {ridge.alpha_:.4f}")
print(f"   LightGBM weight: {ridge.coef_[0]:.4f}")
print(f"   SARIMA weight: {ridge.coef_[1]:.4f}")
print(f"   Intercept: {ridge.intercept_:.4f}")

# Normalize weights to percentages
total_weight = abs(ridge.coef_[0]) + abs(ridge.coef_[1])
lgbm_pct = (abs(ridge.coef_[0]) / total_weight) * 100
sarima_pct = (abs(ridge.coef_[1]) / total_weight) * 100

print(f"\n   Normalized weights:")
print(f"     LightGBM: {lgbm_pct:.1f}%")
print(f"     SARIMA:   {sarima_pct:.1f}%")

# Generate ensemble predictions
ensemble_train_pred = ridge.predict(meta_features_train)
ensemble_test_pred = ridge.predict(meta_features_test)

# ============================================================================
# 7. Evaluate Ensemble Performance
# ============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE PERFORMANCE")
print("=" * 80)

# Calculate metrics
ensemble_test_rmse = np.sqrt(mean_squared_error(y_test_lgbm, ensemble_test_pred))
ensemble_test_mae = mean_absolute_error(y_test_lgbm, ensemble_test_pred)
ensemble_test_r2 = r2_score(y_test_lgbm, ensemble_test_pred)

# Directional accuracy
actual_direction = np.sign(y_test_lgbm.values)
pred_direction = np.sign(ensemble_test_pred)
directional_accuracy = (actual_direction == pred_direction).mean() * 100

print(f"\n📊 Test Set Metrics:")
print(f"   RMSE: {ensemble_test_rmse:.4f}")
print(f"   MAE:  {ensemble_test_mae:.4f}")
print(f"   R²:   {ensemble_test_r2:.4f}")
print(f"   Directional Accuracy: {directional_accuracy:.1f}%")

# Compare to benchmarks
production_rmse = 0.5046
exp005_rmse = 6.5338
exp006_rmse = 6.5338  # Identical to EXP-005

gap_to_production_pct = ((ensemble_test_rmse - production_rmse) / production_rmse) * 100
improvement_vs_exp005_pct = ((ensemble_test_rmse - exp005_rmse) / exp005_rmse) * 100
improvement_vs_exp006_pct = ((ensemble_test_rmse - exp006_rmse) / exp006_rmse) * 100

print(f"\n📈 Performance Comparison:")
print(f"   Production RMSE:  {production_rmse:.4f}")
print(f"   EXP-005 RMSE:     {exp005_rmse:.4f} (explosive SARIMA 2,1,2)(1,1,1,4)")
print(f"   EXP-006 RMSE:     {exp006_rmse:.4f} (identical - features already aligned)")
print(f"   EXP-007 RMSE:     {ensemble_test_rmse:.4f} (PRODUCTION SARIMA 1,1,2)(0,0,1,4)")
print(f"\n   Gap to Production: {gap_to_production_pct:+.1f}%")
print(f"   Improvement vs EXP-005: {improvement_vs_exp005_pct:+.1f}%")
print(f"   Improvement vs EXP-006: {improvement_vs_exp006_pct:+.1f}%")

# ============================================================================
# 8. Save Results
# ============================================================================

print("\n" + "=" * 80)
print("SAVING EXPERIMENT RESULTS")
print("=" * 80)

# Save models
output_dir = '.'
with open(f'{output_dir}/ENSEMBLE-EXP-007_lgbm_component.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)
with open(f'{output_dir}/ENSEMBLE-EXP-007_sarima_component.pkl', 'wb') as f:
    pickle.dump(sarima_fit, f)
with open(f'{output_dir}/ENSEMBLE-EXP-007_ridge_meta.pkl', 'wb') as f:
    pickle.dump(ridge, f)

# Save predictions
predictions_df = pd.DataFrame({
    'date': test_df.index,
    'actual': y_test_lgbm.values,
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred.values,
    'ensemble': ensemble_test_pred
})
predictions_df.to_csv(f'{output_dir}/ENSEMBLE-EXP-007_predictions.csv', index=False)

# Save metadata
metadata = {
    'experiment_id': 'ENSEMBLE-EXP-007',
    'description': 'SARIMA configuration hypothesis test - production SARIMA order',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'key_change': 'Using production SARIMA configuration (1,1,2)(0,0,1,4) to validate root cause',
    'architecture': {
        'component_1': 'LightGBM (lgb.train + early stopping)',
        'component_2': 'SARIMA (Production Config)',
        'meta_learner': 'Ridge Regression'
    },
    'features': {
        'count': len(available_features),
        'production_match': True,
        'same_as_exp_005_006': True
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
        'order': list(sarima_order),
        'seasonal_order': list(sarima_seasonal),
        'configuration': 'PRODUCTION (1,1,2)(0,0,1,4)',
        'aic': float(sarima_fit.aic),
        'train_rmse': float(sarima_train_rmse),
        'train_r2': float(sarima_train_r2),
        'test_rmse': float(sarima_test_rmse),
        'test_r2': float(sarima_test_r2),
        'comparison': {
            'production_sarima_rmse': production_sarima_rmse,
            'exp006_sarima_rmse': exp006_sarima_rmse,
            'difference_from_production_pct': float(((sarima_test_rmse - production_sarima_rmse) / production_sarima_rmse) * 100),
            'improvement_vs_exp006_pct': float(((sarima_test_rmse - exp006_sarima_rmse) / exp006_sarima_rmse) * 100)
        }
    },
    'ridge_meta': {
        'alpha': float(ridge.alpha_),
        'weights': {
            'lightgbm': float(ridge.coef_[0]),
            'sarima': float(ridge.coef_[1]),
            'intercept': float(ridge.intercept_)
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
        'improvement_vs_best_component': float(((ensemble_test_rmse - min(lgbm_test_rmse, sarima_test_rmse)) / min(lgbm_test_rmse, sarima_test_rmse)) * 100)
    },
    'comparison': {
        'production_rmse': production_rmse,
        'exp005_rmse': exp005_rmse,
        'exp006_rmse': exp006_rmse,
        'exp007_rmse': float(ensemble_test_rmse),
        'gap_to_production_pct': float(gap_to_production_pct),
        'improvement_vs_exp005_pct': float(improvement_vs_exp005_pct),
        'improvement_vs_exp006_pct': float(improvement_vs_exp006_pct)
    }
}

with open(f'{output_dir}/ENSEMBLE-EXP-007_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Results saved:")
print(f"   - ENSEMBLE-EXP-007_lgbm_component.pkl")
print(f"   - ENSEMBLE-EXP-007_sarima_component.pkl")
print(f"   - ENSEMBLE-EXP-007_ridge_meta.pkl")
print(f"   - ENSEMBLE-EXP-007_predictions.csv")
print(f"   - ENSEMBLE-EXP-007_metadata.json")

# ============================================================================
# 9. Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("ROOT CAUSE HYPOTHESIS TEST SUMMARY")
print("=" * 80)

print(f"\n🎯 Hypothesis: SARIMA configuration (1,1,2)(0,0,1,4) vs (2,1,2)(1,1,1,4)")
print(f"   is THE root cause of the 1195% performance gap.")

print(f"\n📊 Results:")
print(f"   Production RMSE:  {production_rmse:.4f}")
print(f"   EXP-007 RMSE:     {ensemble_test_rmse:.4f}")
print(f"   Gap:              {gap_to_production_pct:+.1f}%")

print(f"\n📊 Component Breakdown:")
print(f"   LightGBM: {lgbm_test_rmse:.4f} RMSE (expected IDENTICAL to EXP-005/006)")
print(f"   SARIMA:   {sarima_test_rmse:.4f} RMSE (expected ~{production_sarima_rmse:.4f})")
print(f"   Ensemble: {ensemble_test_rmse:.4f} RMSE (expected ~0.50-0.55)")

if gap_to_production_pct < 10:
    print(f"\n✅ SUCCESS: Gap closed to <10% - ROOT CAUSE HYPOTHESIS VALIDATED!")
    print(f"   SARIMA configuration difference IS THE root cause of 1195% gap")
    print(f"   Production uses (1,1,2)(0,0,1,4) for stability")
    print(f"   Experimental (2,1,2)(1,1,1,4) causes explosive forecasts")
elif gap_to_production_pct < 20:
    print(f"\n⚠️  PARTIAL SUCCESS: Gap reduced but not closed - hypothesis PARTIALLY validated")
    print(f"   SARIMA config is major factor but other differences remain")
elif sarima_test_rmse < production_sarima_rmse * 1.2:
    print(f"\n⚠️  SARIMA MATCHED but ensemble gap remains - investigate meta-learner differences")
    print(f"   SARIMA component validated but ensemble methodology may differ")
else:
    print(f"\n❌ FAILURE: SARIMA and/or ensemble still diverge - hypothesis FALSIFIED")
    print(f"   Further investigation needed into hidden production logic")

print("\n" + "=" * 80)
