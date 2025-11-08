#!/usr/bin/env python3
"""
ENSEMBLE-EXP-006: Feature Set Alignment Hypothesis Test
Phoenix Rent Growth Forecasting

Purpose: Test whether adding production's 8 missing macroeconomic features closes the performance gap
Key Change: Use production's EXACT 26-feature set (including regime detection features)

Critical Missing Features Being Added:
1. fed_funds_rate - Federal Reserve policy (economic regime indicator)
2. national_unemployment - National unemployment (economic cycle indicator)
3. cpi - Consumer Price Index (inflation regime indicator)
4. cap_rate - Capitalization rate (investor sentiment)
5. phx_home_price_index - Phoenix home price index (local market strength)
6. phx_hpi_yoy_growth - Phoenix HPI growth (local housing momentum)
7. phx_manufacturing_employment - Manufacturing employment (economic diversity)
8. vacancy_rate - Vacancy rate (supply/demand balance)

Expected Outcome:
- LightGBM test RMSE: ~0.25-0.35 (regime-aware predictions)
- LightGBM test R²: Positive ~0.5-0.7 (meaningful fit)
- Ensemble RMSE: ~0.50-0.55 (close to production's 0.5046)
- Gap to production: <10%

Comparison to Previous Experiments:
- EXP-003: 0.5936 RMSE (17.7% gap) with 25 experimental features
- EXP-005: 6.5338 RMSE (1194.8% gap) with early stopping but wrong features
- EXP-006: Testing whether production feature set closes the gap
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
print("ENSEMBLE-EXP-006: FEATURE SET ALIGNMENT TEST")
print("=" * 80)
print("\nTesting hypothesis: Production's 26-feature set (with macroeconomic regime")
print("indicators) enables regime detection and closes the performance gap.")
print("\nAdding 8 critical features:")
print("  Macro: fed_funds_rate, national_unemployment, cpi")
print("  Housing: cap_rate, phx_home_price_index, phx_hpi_yoy_growth")
print("  Labor: phx_manufacturing_employment")
print("  Property: vacancy_rate")

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

# EXACT production features (from gbm_phoenix_specific_model.pkl inspection)
production_features = [
    # Employment Indicators (7 features)
    'phx_total_employment',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',  # ✅ ADDED (regime indicator)
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
    'vacancy_rate',  # ✅ ADDED (supply/demand regime indicator)
    'inventory_units',
    'absorption_12mo',
    'cap_rate',  # ✅ ADDED (investor sentiment regime indicator)
    'supply_inventory_ratio',

    # Derived Ratios (1 feature)
    'absorption_inventory_ratio',

    # Housing Market (2 features)
    'phx_home_price_index',  # ✅ ADDED (local housing regime indicator)
    'phx_hpi_yoy_growth',  # ✅ ADDED (local housing momentum indicator)

    # Migration (1 feature)
    'migration_proxy',

    # Interest Rates (3 features)
    'mortgage_rate_30yr',
    'mortgage_rate_30yr_lag2',
    'fed_funds_rate',  # ✅ ADDED (monetary policy regime indicator)

    # Macroeconomic (2 features)
    'national_unemployment',  # ✅ ADDED (economic cycle regime indicator)
    'cpi',  # ✅ ADDED (inflation regime indicator)

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

# Mark newly added regime detection features
new_features = ['fed_funds_rate', 'national_unemployment', 'cpi', 'cap_rate',
                'phx_home_price_index', 'phx_hpi_yoy_growth',
                'phx_manufacturing_employment', 'vacancy_rate']

print(f"\n✅ All production features available!")
print(f"\n🆕 Newly Added Regime Detection Features ({len(new_features)}):")
for feat in new_features:
    if feat in available_features:
        null_count = historical_df[feat].isnull().sum()
        null_pct = (null_count / len(historical_df)) * 100
        print(f"  ✅ {feat:30s} ({null_count} nulls, {null_pct:.1f}%)")

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

# Production hyperparameters (same as production and EXP-005)
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
print("\n🔄 Training LightGBM model with regime detection features...")
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
# 5. Component 2: Pure SARIMA (same as EXP-003/EXP-005)
# ============================================================================

print("\n" + "=" * 80)
print("COMPONENT 2: PURE SARIMA (NO EXOGENOUS VARIABLES)")
print("=" * 80)

# Use same SARIMA config as EXP-003/EXP-005
sarima_order = (2, 1, 2)
sarima_seasonal = (1, 1, 1, 4)

print(f"\nSARIMA Configuration:")
print(f"  Order: {sarima_order}")
print(f"  Seasonal Order: {sarima_seasonal}")
print(f"  Exogenous: None (pure SARIMA)")

# Fit SARIMA
print("\n🔄 Training SARIMA model...")
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
exp003_rmse = 0.5936
exp005_rmse = 6.5338

gap_to_production_pct = ((ensemble_test_rmse - production_rmse) / production_rmse) * 100
improvement_vs_exp003_pct = ((ensemble_test_rmse - exp003_rmse) / exp003_rmse) * 100
improvement_vs_exp005_pct = ((ensemble_test_rmse - exp005_rmse) / exp005_rmse) * 100

print(f"\n📈 Performance Comparison:")
print(f"   Production RMSE:  {production_rmse:.4f}")
print(f"   EXP-003 RMSE:     {exp003_rmse:.4f} (wrong features)")
print(f"   EXP-005 RMSE:     {exp005_rmse:.4f} (early stopping but wrong features)")
print(f"   EXP-006 RMSE:     {ensemble_test_rmse:.4f} (production features)")
print(f"\n   Gap to Production: {gap_to_production_pct:+.1f}%")
print(f"   Improvement vs EXP-003: {improvement_vs_exp003_pct:+.1f}%")
print(f"   Improvement vs EXP-005: {improvement_vs_exp005_pct:+.1f}%")

# Determine if improvement is from better components
lgbm_improvement_vs_exp005 = ((lgbm_test_rmse - 4.1058) / 4.1058) * 100
print(f"\n   LightGBM Improvement (vs EXP-005): {lgbm_improvement_vs_exp005:+.1f}%")

# ============================================================================
# 8. Save Results
# ============================================================================

print("\n" + "=" * 80)
print("SAVING EXPERIMENT RESULTS")
print("=" * 80)

# Save models
output_dir = '.'
with open(f'{output_dir}/ENSEMBLE-EXP-006_lgbm_component.pkl', 'wb') as f:
    pickle.dump(gbm_model, f)
with open(f'{output_dir}/ENSEMBLE-EXP-006_sarima_component.pkl', 'wb') as f:
    pickle.dump(sarima_fit, f)
with open(f'{output_dir}/ENSEMBLE-EXP-006_ridge_meta.pkl', 'wb') as f:
    pickle.dump(ridge, f)

# Save predictions
predictions_df = pd.DataFrame({
    'date': test_df.index,
    'actual': y_test_lgbm.values,
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred.values,
    'ensemble': ensemble_test_pred
})
predictions_df.to_csv(f'{output_dir}/ENSEMBLE-EXP-006_predictions.csv', index=False)

# Save metadata
metadata = {
    'experiment_id': 'ENSEMBLE-EXP-006',
    'description': 'Feature set alignment test - production 26-feature set with regime detection',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'key_change': 'Using production EXACT 26-feature set (added 8 regime detection features)',
    'architecture': {
        'component_1': 'LightGBM (lgb.train + early stopping)',
        'component_2': 'SARIMA (Pure)',
        'meta_learner': 'Ridge Regression'
    },
    'features': {
        'count': len(available_features),
        'production_match': True,
        'new_regime_features': new_features
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
        'aic': float(sarima_fit.aic),
        'train_rmse': float(sarima_train_rmse),
        'train_r2': float(sarima_train_r2),
        'test_rmse': float(sarima_test_rmse),
        'test_r2': float(sarima_test_r2)
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
        'exp003_rmse': exp003_rmse,
        'exp005_rmse': exp005_rmse,
        'exp006_rmse': float(ensemble_test_rmse),
        'gap_to_production_pct': float(gap_to_production_pct),
        'improvement_vs_exp003_pct': float(improvement_vs_exp003_pct),
        'improvement_vs_exp005_pct': float(improvement_vs_exp005_pct)
    }
}

with open(f'{output_dir}/ENSEMBLE-EXP-006_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Results saved:")
print(f"   - ENSEMBLE-EXP-006_lgbm_component.pkl")
print(f"   - ENSEMBLE-EXP-006_sarima_component.pkl")
print(f"   - ENSEMBLE-EXP-006_ridge_meta.pkl")
print(f"   - ENSEMBLE-EXP-006_predictions.csv")
print(f"   - ENSEMBLE-EXP-006_metadata.json")

# ============================================================================
# 9. Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)

print(f"\n🎯 Hypothesis: Production's 26-feature set (with macroeconomic regime")
print(f"   indicators) enables regime detection and closes the performance gap.")

print(f"\n📊 Results:")
print(f"   Production RMSE:  {production_rmse:.4f}")
print(f"   EXP-006 RMSE:     {ensemble_test_rmse:.4f}")
print(f"   Gap:              {gap_to_production_pct:+.1f}%")

if gap_to_production_pct < 10:
    print(f"\n✅ SUCCESS: Gap closed to <10% - hypothesis VALIDATED!")
    print(f"   Regime detection features enabled LightGBM to adapt to test regime")
elif gap_to_production_pct < 20:
    print(f"\n⚠️  PARTIAL SUCCESS: Gap reduced but not closed - hypothesis PARTIALLY validated")
    print(f"   Feature set helps but other factors remain (SARIMA config?)")
else:
    print(f"\n❌ FAILURE: Gap remains >20% - hypothesis FALSIFIED")
    print(f"   Feature set alone insufficient - investigate other differences")

print("\n" + "=" * 80)
