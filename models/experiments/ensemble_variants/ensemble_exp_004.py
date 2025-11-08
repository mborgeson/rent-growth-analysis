"""
ENSEMBLE-EXP-004: StandardScaler Ablation Study
================================================

Objective: Test if StandardScaler causes component failure amplification

Configuration: Identical to ENSEMBLE-EXP-003 but WITHOUT StandardScaler
- Same 26 features (production feature set)
- Same LightGBM hyperparameters
- Same pure SARIMA configuration
- Same Ridge meta-learner
- ONLY DIFFERENCE: No StandardScaler on LightGBM features

Hypothesis: StandardScaler amplifies regime-specific patterns, causing LightGBM to predict constant
Expected: Without scaling, LightGBM should produce variable predictions and better generalization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/mattb/Rent Growth Analysis/data")
OUTPUT_DIR = Path("/home/mattb/Rent Growth Analysis/models/experiments/ensemble_variants")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ENSEMBLE-EXP-004: StandardScaler Ablation Study")
print("="*80)
print("\nObjective: Test if StandardScaler causes component failure")
print("Change from EXP-003: Remove StandardScaler from LightGBM")
print("="*80)

# Load data
df = pd.read_csv(DATA_DIR / "processed/phoenix_modeling_dataset.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Production feature set (26 features - same as EXP-003)
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

# Create interaction feature
df['mortgage_employment_interaction'] = df['mortgage_rate_30yr'] * df['phx_total_employment']

# Split data: 2010-2022 train, 2023-2025 test (actual data only, not forecasts)
train_df = df[df['date'] < '2023-01-01'].copy()
test_df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-09-30')].copy()

print(f"\nTrain period: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} quarters)")
print(f"Test period:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} quarters)")

# Feature stability analysis (KS test)
print("\n" + "="*80)
print("FEATURE STABILITY ANALYSIS")
print("="*80)

feature_stability = []
for feature in phoenix_features:
    if feature in df.columns:
        train_values = train_df[feature].dropna()
        test_values = test_df[feature].dropna()

        if len(train_values) > 0 and len(test_values) > 0:
            ks_stat, ks_pvalue = stats.ks_2samp(train_values, test_values)

            train_mean = train_values.mean()
            test_mean = test_values.mean()
            pct_shift = abs((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0

            feature_stability.append({
                'feature': feature,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'stable_p05': ks_pvalue > 0.05,
                'stable_p01': ks_pvalue > 0.01,
                'train_mean': train_mean,
                'test_mean': test_mean,
                'pct_shift': pct_shift
            })

feature_stability_df = pd.DataFrame(feature_stability)
feature_stability_df = feature_stability_df.sort_values('ks_pvalue', ascending=False)

stable_p05 = feature_stability_df[feature_stability_df['stable_p05']].shape[0]
stable_p01 = feature_stability_df[feature_stability_df['stable_p01']].shape[0]

print(f"\nFeature Stability Summary:")
print(f"  Total features: {len(phoenix_features)}")
print(f"  Stable (p>0.05): {stable_p05}/{len(phoenix_features)} ({stable_p05/len(phoenix_features)*100:.1f}%)")
print(f"  Stable (p>0.01): {stable_p01}/{len(phoenix_features)} ({stable_p01/len(phoenix_features)*100:.1f}%)")

print(f"\nMost stable features (p>0.05):")
for _, row in feature_stability_df[feature_stability_df['stable_p05']].head(5).iterrows():
    print(f"  {row['feature']:40s} p={row['ks_pvalue']:.4f} shift={row['pct_shift']:.1f}%")

print(f"\nMost unstable features:")
for _, row in feature_stability_df.tail(5).iterrows():
    print(f"  {row['feature']:40s} p={row['ks_pvalue']:.4f} shift={row['pct_shift']:.1f}%")

# Component 1: LightGBM (WITHOUT StandardScaler - key difference!)
print("\n" + "="*80)
print("COMPONENT 1: LightGBM (NO SCALING)")
print("="*80)

X_train_lgbm = train_df[phoenix_features].values
y_train_lgbm = train_df['rent_growth_yoy'].values
X_test_lgbm = test_df[phoenix_features].values
y_test_lgbm = test_df['rent_growth_yoy'].values

print(f"\nLightGBM Training Set: {X_train_lgbm.shape}")
print(f"LightGBM Test Set:     {X_test_lgbm.shape}")
print("\n⚠️  KEY CHANGE: Using RAW features WITHOUT StandardScaler")

# LightGBM with production hyperparameters
lgbm_params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 10,
    'random_state': 42,
    'verbosity': -1
}

lgbm_component = LGBMRegressor(**lgbm_params)
lgbm_component.fit(X_train_lgbm, y_train_lgbm)

# LightGBM predictions (WITHOUT scaling)
lgbm_train_pred = lgbm_component.predict(X_train_lgbm)
lgbm_test_pred = lgbm_component.predict(X_test_lgbm)

lgbm_train_rmse = np.sqrt(np.mean((lgbm_train_pred - y_train_lgbm)**2))
lgbm_test_rmse = np.sqrt(np.mean((lgbm_test_pred - y_test_lgbm)**2))
lgbm_train_r2 = 1 - np.sum((y_train_lgbm - lgbm_train_pred)**2) / np.sum((y_train_lgbm - y_train_lgbm.mean())**2)
lgbm_test_r2 = 1 - np.sum((y_test_lgbm - lgbm_test_pred)**2) / np.sum((y_test_lgbm - y_test_lgbm.mean())**2)

print(f"\nLightGBM Performance (NO SCALING):")
print(f"  Train RMSE: {lgbm_train_rmse:.4f}")
print(f"  Test RMSE:  {lgbm_test_rmse:.4f}")
print(f"  Train R²:   {lgbm_train_r2:.4f}")
print(f"  Test R²:    {lgbm_test_r2:.4f}")

# Check prediction variability
print(f"\nPrediction Variability Check:")
print(f"  Test predictions: min={lgbm_test_pred.min():.4f}, max={lgbm_test_pred.max():.4f}, std={lgbm_test_pred.std():.4f}")
if lgbm_test_pred.std() < 0.01:
    print(f"  ⚠️  WARNING: Predictions nearly constant (std < 0.01)")
else:
    print(f"  ✓ Predictions show variation")

# Component 2: Pure SARIMA (same as EXP-003)
print("\n" + "="*80)
print("COMPONENT 2: Pure SARIMA (Grid Search)")
print("="*80)

train_ts_sarima = train_df.set_index('date')['rent_growth_yoy']
test_ts_sarima = test_df.set_index('date')['rent_growth_yoy']

print(f"\nSARIMA Training Set: {len(train_ts_sarima)} quarters")
print(f"SARIMA Test Set:     {len(test_ts_sarima)} quarters")

# Stationarity test for d parameter
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(train_ts_sarima.dropna())
d_order = 0 if adf_result[1] < 0.05 else 1
print(f"\nADF Test p-value: {adf_result[1]:.4f}")
print(f"Differencing order (d): {d_order}")

# Grid search for SARIMA parameters (same as EXP-003)
print("\nGrid search for optimal SARIMA parameters...")
p_range = range(0, 3)
d_range = [d_order]
q_range = range(0, 3)
P_range = range(0, 2)
D_range = [0, 1]
Q_range = range(0, 2)
seasonal_period = 4

best_aic = np.inf
best_params = None
best_seasonal_params = None

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
                                seasonal_order=(P, D, Q, seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False, maxiter=200)

                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_params = (p, d, q)
                                best_seasonal_params = (P, D, Q, seasonal_period)
                        except:
                            continue

print(f"\nBest SARIMA parameters:")
print(f"  order: {best_params}")
print(f"  seasonal_order: {best_seasonal_params}")
print(f"  AIC: {best_aic:.4f}")

# Train final SARIMA model
sarima_model = SARIMAX(
    train_ts_sarima,
    order=best_params,
    seasonal_order=best_seasonal_params,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fitted = sarima_model.fit(disp=False, maxiter=200)

# SARIMA predictions
sarima_train_pred = sarima_fitted.fittedvalues
sarima_test_pred = sarima_fitted.forecast(steps=len(test_ts_sarima))

sarima_train_rmse = np.sqrt(np.mean((sarima_train_pred - train_ts_sarima)**2))
sarima_test_rmse = np.sqrt(np.mean((sarima_test_pred.values - test_ts_sarima.values)**2))
sarima_train_r2 = 1 - np.sum((train_ts_sarima - sarima_train_pred)**2) / np.sum((train_ts_sarima - train_ts_sarima.mean())**2)
sarima_test_r2 = 1 - np.sum((test_ts_sarima.values - sarima_test_pred.values)**2) / np.sum((test_ts_sarima.values - test_ts_sarima.mean())**2)

print(f"\nSARIMA Performance:")
print(f"  Train RMSE: {sarima_train_rmse:.4f}")
print(f"  Test RMSE:  {sarima_test_rmse:.4f}")
print(f"  Train R²:   {sarima_train_r2:.4f}")
print(f"  Test R²:    {sarima_test_r2:.4f}")

# Meta-Learner: Ridge Regression (same as EXP-003)
print("\n" + "="*80)
print("META-LEARNER: Ridge Regression")
print("="*80)

# Align predictions for meta-learner training
ensemble_predictions = pd.DataFrame({
    'date': test_df['date'].values,
    'actual': y_test_lgbm,
    'lightgbm': lgbm_test_pred,
    'sarima': sarima_test_pred.values
})

# Meta-learner training data (from training period)
X_meta_train = pd.DataFrame({
    'lightgbm': lgbm_train_pred,
    'sarima': sarima_train_pred.values
})
y_meta_train = y_train_lgbm

print(f"\nMeta-learner training data: {X_meta_train.shape}")

# Ridge with cross-validation
ridge_meta = RidgeCV(
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_mean_squared_error'
)
ridge_meta.fit(X_meta_train, y_meta_train)

print(f"\nRidge CV Results:")
print(f"  Best alpha: {ridge_meta.alpha_}")
print(f"\nRidge Coefficients:")
print(f"  LightGBM:  {ridge_meta.coef_[0]:+.4f}")
print(f"  SARIMA:    {ridge_meta.coef_[1]:+.4f}")
print(f"  Intercept: {ridge_meta.intercept_:+.4f}")

# Normalized weights
total_weight = abs(ridge_meta.coef_[0]) + abs(ridge_meta.coef_[1])
lgbm_pct = abs(ridge_meta.coef_[0]) / total_weight * 100 if total_weight > 0 else 0
sarima_pct = abs(ridge_meta.coef_[1]) / total_weight * 100 if total_weight > 0 else 0

print(f"\nNormalized Weights:")
print(f"  LightGBM:  {lgbm_pct:.1f}%")
print(f"  SARIMA:    {sarima_pct:.1f}%")

# Count negative coefficients
negative_coefs = sum([1 for c in ridge_meta.coef_ if c < 0])
print(f"  Negative coefficients: {negative_coefs}/2")

# Ensemble predictions
X_meta_test = ensemble_predictions[['lightgbm', 'sarima']]
ensemble_predictions['ensemble'] = ridge_meta.predict(X_meta_test)

# Ensemble performance
ensemble_test_rmse = np.sqrt(np.mean((ensemble_predictions['ensemble'] - ensemble_predictions['actual'])**2))
ensemble_test_mae = np.mean(np.abs(ensemble_predictions['ensemble'] - ensemble_predictions['actual']))
ensemble_test_r2 = 1 - np.sum((ensemble_predictions['actual'] - ensemble_predictions['ensemble'])**2) / \
                      np.sum((ensemble_predictions['actual'] - ensemble_predictions['actual'].mean())**2)

# Directional accuracy
actual_direction = (ensemble_predictions['actual'].diff() > 0).astype(int)
pred_direction = (ensemble_predictions['ensemble'].diff() > 0).astype(int)
directional_accuracy = (actual_direction == pred_direction).sum() / (len(actual_direction) - 1) * 100

print("\n" + "="*80)
print("ENSEMBLE PERFORMANCE (EXP-004 - NO SCALING)")
print("="*80)
print(f"\nTest Period Metrics:")
print(f"  RMSE: {ensemble_test_rmse:.4f}")
print(f"  MAE:  {ensemble_test_mae:.4f}")
print(f"  R²:   {ensemble_test_r2:.4f}")
print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

# Improvement over best component
best_component_rmse = min(lgbm_test_rmse, sarima_test_rmse)
improvement = (best_component_rmse - ensemble_test_rmse) / best_component_rmse * 100
print(f"\nImprovement over best component: {improvement:.1f}%")

# Comparison to previous experiments
print("\n" + "="*80)
print("COMPARISON TO PREVIOUS EXPERIMENTS")
print("="*80)

production_rmse = 0.5046
exp001_rmse = 3.8113
exp002_rmse = 6.6447
exp003_rmse = 0.5936

print(f"\n{'Model':<30s} {'RMSE':<10s} {'vs EXP-004':<15s}")
print("-" * 60)
print(f"{'Production Ensemble':<30s} {production_rmse:<10.4f} {(production_rmse - ensemble_test_rmse) / production_rmse * 100:+.1f}%")
print(f"{'ENSEMBLE-EXP-004 (NO SCALE)':<30s} {ensemble_test_rmse:<10.4f} {'baseline':<15s}")
print(f"{'ENSEMBLE-EXP-003 (WITH SCALE)':<30s} {exp003_rmse:<10.4f} {(exp003_rmse - ensemble_test_rmse) / exp003_rmse * 100:+.1f}%")
print(f"{'ENSEMBLE-EXP-001':<30s} {exp001_rmse:<10.4f} {(exp001_rmse - ensemble_test_rmse) / exp001_rmse * 100:+.1f}%")
print(f"{'ENSEMBLE-EXP-002':<30s} {exp002_rmse:<10.4f} {(exp002_rmse - ensemble_test_rmse) / exp002_rmse * 100:+.1f}%")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save metadata
metadata = {
    'experiment_id': 'ENSEMBLE-EXP-004',
    'description': 'StandardScaler ablation study - same as EXP-003 but WITHOUT scaling',
    'date': '2025-11-07',
    'key_change': 'Removed StandardScaler from LightGBM features',
    'architecture': {
        'component_1': 'LightGBM (NO SCALING)',
        'component_2': 'SARIMA (Pure)',
        'meta_learner': 'Ridge Regression'
    },
    'features': {
        'count': len(phoenix_features),
        'stable_p05': int(stable_p05),
        'stable_p01': int(stable_p01),
        'stability_rate_p05': float(stable_p05 / len(phoenix_features))
    },
    'lightgbm': {
        'algorithm': 'LightGBM',
        'n_features': len(phoenix_features),
        'scaling': 'NONE (key change from EXP-003)',
        'hyperparameters': lgbm_params,
        'train_rmse': float(lgbm_train_rmse),
        'train_r2': float(lgbm_train_r2),
        'test_rmse': float(lgbm_test_rmse),
        'test_r2': float(lgbm_test_r2),
        'prediction_std': float(lgbm_test_pred.std())
    },
    'sarima': {
        'type': 'Pure SARIMA (no exogenous)',
        'order': list(best_params),
        'seasonal_order': list(best_seasonal_params),
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
        },
        'negative_coefficients': int(negative_coefs)
    },
    'ensemble_performance': {
        'test_rmse': float(ensemble_test_rmse),
        'test_mae': float(ensemble_test_mae),
        'test_r2': float(ensemble_test_r2),
        'directional_accuracy': float(directional_accuracy),
        'improvement_vs_best_component': float(improvement)
    },
    'comparison': {
        'production_rmse': float(production_rmse),
        'exp001_rmse': float(exp001_rmse),
        'exp002_rmse': float(exp002_rmse),
        'exp003_rmse': float(exp003_rmse),
        'exp004_rmse': float(ensemble_test_rmse),
        'improvement_vs_exp003_pct': float((exp003_rmse - ensemble_test_rmse) / exp003_rmse * 100)
    }
}

with open(OUTPUT_DIR / 'ENSEMBLE-EXP-004_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save predictions
ensemble_predictions.to_csv(OUTPUT_DIR / 'ENSEMBLE-EXP-004_predictions.csv', index=False)

# Save feature stability
feature_stability_df.to_csv(OUTPUT_DIR / 'ENSEMBLE-EXP-004_feature_stability.csv', index=False)

# Save models
import pickle
with open(OUTPUT_DIR / 'ENSEMBLE-EXP-004_lgbm_component.pkl', 'wb') as f:
    pickle.dump(lgbm_component, f)
with open(OUTPUT_DIR / 'ENSEMBLE-EXP-004_sarima_component.pkl', 'wb') as f:
    pickle.dump(sarima_fitted, f)
with open(OUTPUT_DIR / 'ENSEMBLE-EXP-004_ridge_meta.pkl', 'wb') as f:
    pickle.dump(ridge_meta, f)

print(f"\nSaved results:")
print(f"  - ENSEMBLE-EXP-004_metadata.json")
print(f"  - ENSEMBLE-EXP-004_predictions.csv")
print(f"  - ENSEMBLE-EXP-004_feature_stability.csv")
print(f"  - ENSEMBLE-EXP-004_lgbm_component.pkl")
print(f"  - ENSEMBLE-EXP-004_sarima_component.pkl")
print(f"  - ENSEMBLE-EXP-004_ridge_meta.pkl")

print("\n" + "="*80)
print("ENSEMBLE-EXP-004 COMPLETE")
print("="*80)
print(f"\n✅ StandardScaler ablation study complete")
print(f"\nKey Question: Did removing StandardScaler improve LightGBM?")
print(f"  EXP-003 (WITH scaling):   RMSE {exp003_rmse:.4f}, LightGBM stuck at constant")
print(f"  EXP-004 (WITHOUT scaling): RMSE {ensemble_test_rmse:.4f}, LightGBM std={lgbm_test_pred.std():.4f}")
print(f"\nConclusion: {'StandardScaler is harmful' if ensemble_test_rmse < exp003_rmse else 'StandardScaler is helpful' if ensemble_test_rmse > exp003_rmse else 'No significant difference'}")
