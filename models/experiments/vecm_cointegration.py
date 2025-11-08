#!/usr/bin/env python3
"""
VECM (Vector Error Correction Model) - Cointegration Analysis
==============================================================

Purpose: Model long-run equilibrium relationships between cointegrated variables
         - Rent growth
         - Employment growth
         - Construction pipeline
         - Home price index

Theory:
- If variables are cointegrated, they share a long-run equilibrium relationship
- Short-run deviations from equilibrium are corrected over time
- VECM captures both short-run dynamics and long-run adjustments

Based on Deep Dive Analysis:
- Employment Granger-causes rent growth (lag 6, p<0.0001)
- Construction Granger-causes rent growth (lag 1, p=0.021)
- Suggests potential cointegration relationships

Expected Insights:
- Long-run elasticities between variables
- Speed of adjustment back to equilibrium
- Better long-term forecasts than VAR

Experiment ID: VECM-COINT-001
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'models/experiments/vecm_variants'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ID = "VECM-COINT-001"
EXPERIMENT_DATE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print("=" * 80)
print(f"VECM COINTEGRATION MODEL - {EXPERIMENT_ID}")
print("=" * 80)
print(f"Experiment Date: {EXPERIMENT_DATE}")
print(f"Purpose: Model long-run equilibrium relationships")
print()

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA LOADING AND VARIABLE SELECTION")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
historical_df = df.loc[:'2025-09-30'].copy()

print(f"✅ Loaded dataset: {len(historical_df)} quarters")

# Select variables for VECM (theory: potentially cointegrated)
vecm_vars = [
    'rent_growth_yoy',
    'phx_employment_yoy_growth',
    'units_under_construction',
    'phx_hpi_yoy_growth'
]

available_vars = [v for v in vecm_vars if v in historical_df.columns]
print(f"\nVECM variables: {len(available_vars)}/{len(vecm_vars)}")
for var in available_vars:
    print(f"  {var}")

# Create VECM dataset
vecm_df = historical_df[available_vars].dropna()
print(f"\n✅ VECM dataset: {len(vecm_df)} quarters")
print(f"   Coverage: {vecm_df.index.min()} to {vecm_df.index.max()}")

# ============================================================================
# 2. Stationarity Testing
# ============================================================================

print("\n" + "=" * 80)
print("2. STATIONARITY TESTING")
print("=" * 80)

print("\nAugmented Dickey-Fuller Tests:")
print(f"{'Variable':>35} {'ADF Stat':>12} {'P-Value':>10} {'Stationary':>15}")
print("-" * 76)

stationarity_results = {}
for var in vecm_df.columns:
    adf_result = adfuller(vecm_df[var].dropna(), autolag='AIC')
    is_stationary = adf_result[1] < 0.05
    stationarity_results[var] = {
        'adf_stat': adf_result[0],
        'p_value': adf_result[1],
        'stationary': is_stationary
    }

    status = "✅ Yes" if is_stationary else "❌ No (I(1))"
    print(f"{var:>35} {adf_result[0]:>12.4f} {adf_result[1]:>10.4f} {status:>15}")

# Check if variables are I(1) (integrated of order 1)
i1_vars = [v for v, r in stationarity_results.items() if not r['stationary']]
if i1_vars:
    print(f"\n✅ Variables appear to be I(1): {', '.join(i1_vars)}")
    print("   VECM is appropriate (cointegration requires I(1) variables)")
else:
    print(f"\n⚠️  All variables are stationary - VECM may not be necessary")

# ============================================================================
# 3. Cointegration Testing (Johansen Test)
# ============================================================================

print("\n" + "=" * 80)
print("3. JOHANSEN COINTEGRATION TEST")
print("=" * 80)

# Johansen test for cointegration
print("\nTesting for cointegration relationships...")

# Determine lag order
lag_order = select_order(vecm_df, maxlags=8)
suggested_lag = lag_order.aic
print(f"Suggested lag order (AIC): {suggested_lag}")

# Johansen test
johansen_result = coint_johansen(vecm_df, det_order=0, k_ar_diff=suggested_lag)

# Trace statistic test
print("\nJohansen Trace Statistic Test:")
print(f"{'Rank':>10} {'Trace Stat':>15} {'5% Crit Value':>18} {'Reject H0':>15}")
print("-" * 62)

cointegration_rank = 0
for i in range(len(johansen_result.lr1)):
    trace_stat = johansen_result.lr1[i]
    crit_value = johansen_result.cvt[i, 1]  # 5% critical value
    rejects = "Yes" if trace_stat > crit_value else "No"

    print(f"{i:>10} {trace_stat:>15.2f} {crit_value:>18.2f} {rejects:>15}")

    if trace_stat > crit_value:
        cointegration_rank = i + 1

print(f"\n✅ Number of cointegrating relationships: {cointegration_rank}")

if cointegration_rank == 0:
    print("⚠️  No cointegration detected - VAR model may be more appropriate")
elif cointegration_rank == len(vecm_df.columns):
    print("⚠️  All variables cointegrated - series may be stationary")
else:
    print(f"✅ {cointegration_rank} cointegrating relationship(s) found - VECM appropriate")

# ============================================================================
# 4. Train/Test Split
# ============================================================================

print("\n" + "=" * 80)
print("4. TRAIN/TEST SPLIT")
print("=" * 80)

train_end = '2022-12-31'
train_df = vecm_df.loc[:train_end]
test_df = vecm_df.loc[train_end:]

print(f"\nTraining set: {len(train_df)} quarters ({train_df.index.min()} to {train_df.index.max()})")
print(f"Test set:     {len(test_df)} quarters ({test_df.index.min()} to {test_df.index.max()})")

# ============================================================================
# 5. Fit VECM Model
# ============================================================================

print("\n" + "=" * 80)
print("5. FITTING VECM MODEL")
print("=" * 80)

if cointegration_rank > 0:
    print(f"\n🔄 Fitting VECM with cointegration rank {cointegration_rank}...")

    vecm_model = VECM(
        train_df,
        k_ar_diff=suggested_lag,
        coint_rank=cointegration_rank,
        deterministic='ci'  # Constant in cointegration relation
    )

    vecm_fitted = vecm_model.fit()

    print(f"✅ VECM model fitted")
    print(f"   Lag order: {suggested_lag}")
    print(f"   Cointegration rank: {cointegration_rank}")
    print(f"   Log Likelihood: {vecm_fitted.llf:.2f}")

    # Display error correction coefficients (adjustment speeds)
    print(f"\nError Correction Coefficients (Speed of Adjustment):")
    ec_coeffs = vecm_fitted.alpha
    for i, var in enumerate(vecm_df.columns):
        coeff = ec_coeffs[i, 0] if ec_coeffs.shape[1] > 0 else 0
        print(f"  {var:35s}: {coeff:8.4f}")
        if abs(coeff) > 0.5:
            print(f"    (Fast adjustment back to equilibrium)")
        elif abs(coeff) > 0.1:
            print(f"    (Moderate adjustment)")
        else:
            print(f"    (Slow adjustment)")

    # ========================================================================
    # 6. Generate Forecasts
    # ========================================================================

    print("\n" + "=" * 80)
    print("6. GENERATING FORECASTS")
    print("=" * 80)

    # Forecast on test period
    forecast_steps = len(test_df)
    forecast = vecm_fitted.predict(steps=forecast_steps)

    print(f"✅ Generated {forecast_steps}-step ahead forecasts")

    # Extract rent growth forecasts (first column)
    rent_forecast = forecast[:, 0]
    rent_actual = test_df.iloc[:, 0].values  # First column is rent_growth_yoy

    # ========================================================================
    # 7. Evaluate Performance
    # ========================================================================

    print("\n" + "=" * 80)
    print("7. PERFORMANCE EVALUATION")
    print("=" * 80)

    rmse = np.sqrt(mean_squared_error(rent_actual, rent_forecast))
    mae = mean_absolute_error(rent_actual, rent_forecast)

    # Directional accuracy
    actual_direction = np.sign(np.diff(rent_actual))
    pred_direction = np.sign(np.diff(rent_forecast))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    print(f"\nOut-of-Sample Performance (Rent Growth):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

    # ========================================================================
    # 8. Refit and Generate Future Forecasts
    # ========================================================================

    print("\n" + "=" * 80)
    print("8. FUTURE FORECASTS (2025 Q4 - 2027 Q4)")
    print("=" * 80)

    print(f"\n🔄 Refitting on full historical data...")

    vecm_model_full = VECM(
        vecm_df,
        k_ar_diff=suggested_lag,
        coint_rank=cointegration_rank,
        deterministic='ci'
    )
    vecm_fitted_full = vecm_model_full.fit()

    future_steps = 8
    future_forecast = vecm_fitted_full.predict(steps=future_steps)

    # Create future dates
    last_date = vecm_df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        periods=future_steps,
        freq='Q'
    )

    future_rent_forecast = pd.Series(future_forecast[:, 0], index=future_dates)

    print(f"✅ Generated {future_steps}-quarter ahead forecasts")
    print(f"\nFuture Rent Growth Forecasts:")
    for date, value in future_rent_forecast.items():
        quarter = (date.month - 1) // 3 + 1
        print(f"  {date.year} Q{quarter}: {value:6.2f}%")

    # ========================================================================
    # 9. Save Model and Results
    # ========================================================================

    print("\n" + "=" * 80)
    print("9. SAVING MODEL AND RESULTS")
    print("=" * 80)

    # Save model
    model_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(vecm_fitted_full, f)
    print(f"✅ Saved model: {model_path}")

    # Save test forecasts
    test_forecast_df = pd.DataFrame({
        'date': test_df.index,
        'actual': rent_actual,
        'forecast': rent_forecast
    })
    test_forecast_df.set_index('date', inplace=True)

    test_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_test_forecasts.csv'
    test_forecast_df.to_csv(test_path)
    print(f"✅ Saved test forecasts: {test_path}")

    # Save future forecasts
    future_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_future_forecasts.csv'
    future_rent_forecast.to_csv(future_path, header=['forecast'])
    print(f"✅ Saved future forecasts: {future_path}")

    # Save metadata
    metadata = {
        'experiment_id': EXPERIMENT_ID,
        'experiment_date': EXPERIMENT_DATE,
        'model_type': 'VECM',
        'variables': available_vars,
        'lag_order': int(suggested_lag),
        'cointegration_rank': int(cointegration_rank),
        'training_period': f"{train_df.index.min()} to {train_df.index.max()}",
        'test_period': f"{test_df.index.min()} to {test_df.index.max()}",
        'performance_metrics': {
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        },
        'error_correction_coefficients': {
            var: float(vecm_fitted.alpha[i, 0]) if vecm_fitted.alpha.shape[1] > 0 else 0.0
            for i, var in enumerate(vecm_df.columns)
        }
    }

    metadata_path = OUTPUT_PATH / f'{EXPERIMENT_ID}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Saved experiment metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {EXPERIMENT_ID} COMPLETE")
    print("=" * 80)

    print(f"""
VECM Model Summary
==================

Cointegration Relationships: {cointegration_rank}
Lag Order: {suggested_lag}

Performance:
  - RMSE: {rmse:.4f}
  - MAE: {mae:.4f}
  - Directional Accuracy: {directional_accuracy:.1f}%

Key Insights:
  - Long-run equilibrium relationships identified
  - Error correction mechanisms quantified
  - Suitable for long-term forecasting

Output Files:
  - {model_path.name}
  - {test_path.name}
  - {future_path.name}
  - {metadata_path.name}
""")

else:
    print("\n⚠️  No cointegration found - skipping VECM estimation")
    print("   Recommendation: Use VAR model instead")

print("=" * 80)

if __name__ == "__main__":
    pass
