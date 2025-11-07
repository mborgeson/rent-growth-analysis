#!/usr/bin/env python3
"""
SARIMA Seasonal Component - Phoenix Rent Growth Forecasting
Component 3 of 3 in Hierarchical Ensemble Model

Purpose: Seasonal ARIMA for capturing quarterly patterns and pure time series dynamics
Weight: 25% in final ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
print("SARIMA SEASONAL COMPONENT - PHOENIX RENT GROWTH FORECASTING")
print("=" * 80)
print(f"Component: 3 of 3 (Seasonal ARIMA)")
print(f"Weight in Ensemble: 25%")
print(f"Purpose: Quarterly seasonal patterns and time series dynamics\n")

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("LOADING RENT GROWTH TIME SERIES")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')

print(f"✅ Loaded dataset: {len(df)} quarters")
print(f"   Date Range: {df.index.min()} to {df.index.max()}")

# Filter to historical period only
historical_df = df.loc[:'2025-09-30'].copy()

# Extract rent growth time series
rent_growth_ts = historical_df['rent_growth_yoy'].dropna()

print(f"\n✅ Rent growth time series: {len(rent_growth_ts)} quarters")
print(f"   Period: {rent_growth_ts.index.min()} to {rent_growth_ts.index.max()}")
print(f"   Mean: {rent_growth_ts.mean():.2f}%")
print(f"   Std: {rent_growth_ts.std():.2f}%")
print(f"   Min: {rent_growth_ts.min():.2f}% ({rent_growth_ts.idxmin()})")
print(f"   Max: {rent_growth_ts.max():.2f}% ({rent_growth_ts.idxmax()})")

# ============================================================================
# 2. Stationarity Testing
# ============================================================================

print("\n" + "=" * 80)
print("STATIONARITY TESTING")
print("=" * 80)

# Augmented Dickey-Fuller test
adf_result = adfuller(rent_growth_ts, autolag='AIC')

print("\nAugmented Dickey-Fuller Test:")
print(f"  Test Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  Lags Used: {adf_result[2]}")
print(f"  Observations: {adf_result[3]}")

if adf_result[1] < 0.05:
    print(f"\n✅ Series is stationary (p < 0.05)")
    d_order = 0  # No differencing needed
else:
    print(f"\n⚠️  Series is non-stationary (p >= 0.05)")
    print("   Will use differencing (d=1)")
    d_order = 1

# ============================================================================
# 3. ACF/PACF Analysis for Parameter Selection
# ============================================================================

print("\n" + "=" * 80)
print("ACF/PACF ANALYSIS FOR PARAMETER SELECTION")
print("=" * 80)

# Calculate ACF and PACF
acf_values = acf(rent_growth_ts, nlags=20)
pacf_values = pacf(rent_growth_ts, nlags=20)

# Find significant lags (|correlation| > 2/sqrt(n))
threshold = 2 / np.sqrt(len(rent_growth_ts))

print(f"\nSignificance threshold: ±{threshold:.3f}")

# AR order suggestion from PACF
pacf_sig = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > threshold]
ar_suggest = min(pacf_sig[:3]) if pacf_sig else 1
print(f"\nSuggested AR order (p) from PACF: {ar_suggest}")

# MA order suggestion from ACF
acf_sig = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > threshold]
ma_suggest = min(acf_sig[:3]) if acf_sig else 1
print(f"Suggested MA order (q) from ACF: {ma_suggest}")

# Seasonal period for quarterly data
seasonal_period = 4  # 4 quarters = 1 year
print(f"\nSeasonal period (s): {seasonal_period} quarters (1 year)")

# ============================================================================
# 4. SARIMA Model Selection via Grid Search
# ============================================================================

print("\n" + "=" * 80)
print("SARIMA MODEL SELECTION (Grid Search)")
print("=" * 80)

# Define parameter ranges to test
p_range = range(0, 3)  # AR order
d_range = [d_order]  # Differencing (from stationarity test)
q_range = range(0, 3)  # MA order
P_range = range(0, 2)  # Seasonal AR
D_range = [0, 1]  # Seasonal differencing
Q_range = range(0, 2)  # Seasonal MA

print(f"\nGrid search parameters:")
print(f"  p (AR): {list(p_range)}")
print(f"  d (I): {list(d_range)}")
print(f"  q (MA): {list(q_range)}")
print(f"  P (Seasonal AR): {list(P_range)}")
print(f"  D (Seasonal I): {list(D_range)}")
print(f"  Q (Seasonal MA): {list(Q_range)}")
print(f"  s (Seasonal period): {seasonal_period}")

# Grid search for best AIC
best_aic = np.inf
best_params = None
best_seasonal_params = None

print(f"\n🔄 Testing {len(p_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)} parameter combinations...")

for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        try:
                            model = SARIMAX(
                                rent_growth_ts,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False)

                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_params = (p, d, q)
                                best_seasonal_params = (P, D, Q, seasonal_period)
                        except:
                            continue

print(f"\n✅ Best SARIMA parameters found:")
print(f"   Order (p,d,q): {best_params}")
print(f"   Seasonal (P,D,Q,s): {best_seasonal_params}")
print(f"   AIC: {best_aic:.2f}")

# ============================================================================
# 5. Train/Test Split and Model Fitting
# ============================================================================

print("\n" + "=" * 80)
print("TRAIN/TEST SPLIT AND MODEL FITTING")
print("=" * 80)

# Split: train on 2010-2022, test on 2023-2025
train_end_date = '2022-12-31'
train_ts = rent_growth_ts.loc[:train_end_date]
test_ts = rent_growth_ts.loc[train_end_date:]

print(f"\nTraining set: {len(train_ts)} quarters ({train_ts.index.min()} to {train_ts.index.max()})")
print(f"Test set:     {len(test_ts)} quarters ({test_ts.index.min()} to {test_ts.index.max()})")

# Fit SARIMA on training data
print(f"\n🔄 Fitting SARIMA{best_params}x{best_seasonal_params} on training data...")
sarima_model = SARIMAX(
    train_ts,
    order=best_params,
    seasonal_order=best_seasonal_params,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fitted = sarima_model.fit(disp=False)

print(f"✅ SARIMA model fitted")
print(f"\nModel Summary:")
print(f"  AIC: {sarima_fitted.aic:.2f}")
print(f"  BIC: {sarima_fitted.bic:.2f}")
print(f"  Log Likelihood: {sarima_fitted.llf:.2f}")

# ============================================================================
# 6. Generate Forecasts
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING FORECASTS")
print("=" * 80)

# Test period forecasts
forecast_steps = len(test_ts)
test_forecast = sarima_fitted.forecast(steps=forecast_steps)

print(f"✅ Generated {forecast_steps}-step ahead forecast")
print(f"   Forecast period: {test_ts.index.min()} to {test_ts.index.max()}")

# ============================================================================
# 7. Evaluate Performance
# ============================================================================

print("\n" + "=" * 80)
print("FORECAST PERFORMANCE EVALUATION")
print("=" * 80)

# Calculate metrics
test_rmse = np.sqrt(mean_squared_error(test_ts, test_forecast))
test_mae = mean_absolute_error(test_ts, test_forecast)

# Directional accuracy
actual_direction = np.sign(np.diff(test_ts.values))
pred_direction = np.sign(np.diff(test_forecast.values))
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

# Compare to naive baseline
naive_forecast = test_ts.shift(1).dropna()
naive_actual = test_ts.iloc[1:]
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_forecast))

print(f"\nOut-of-Sample Performance (2023-2025):")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  Directional Accuracy: {directional_accuracy:.1f}%")

print(f"\nBaseline Comparison:")
print(f"  Naive RMSE (persistence): {naive_rmse:.4f}")
print(f"  SARIMA Improvement: {((naive_rmse - test_rmse) / naive_rmse * 100):.1f}%")

# ============================================================================
# 8. Generate Future Forecasts (2025 Q4 - 2027 Q4)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING FUTURE FORECASTS (2025 Q4 - 2027 Q4)")
print("=" * 80)

# Refit on full historical data for production forecasts
print(f"\n🔄 Refitting SARIMA on full historical data...")
sarima_model_full = SARIMAX(
    rent_growth_ts,
    order=best_params,
    seasonal_order=best_seasonal_params,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fitted_full = sarima_model_full.fit(disp=False)

# Forecast 8 quarters ahead
future_steps = 8
future_forecast = sarima_fitted_full.forecast(steps=future_steps)

# Create future dates
last_date = rent_growth_ts.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=3),
    periods=future_steps,
    freq='Q'
)
future_forecast.index = future_dates

print(f"✅ Generated {future_steps}-quarter ahead forecasts")
print(f"   Forecast period: {future_forecast.index.min()} to {future_forecast.index.max()}")

print(f"\nFuture Rent Growth Forecasts:")
for date, value in future_forecast.items():
    print(f"  {date.strftime('%Y Q%q')}: {value:6.2f}%")

# ============================================================================
# 9. Residual Diagnostics
# ============================================================================

print("\n" + "=" * 80)
print("RESIDUAL DIAGNOSTICS")
print("=" * 80)

# Get residuals from full model
residuals = sarima_fitted_full.resid

print(f"\nResidual Statistics:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# Ljung-Box test for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05]

if len(significant_lags) == 0:
    print(f"\n✅ Residuals show no significant autocorrelation (Ljung-Box test)")
else:
    print(f"\n⚠️  Residuals show autocorrelation at {len(significant_lags)} lags")

# ============================================================================
# 10. Save SARIMA Component
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SARIMA COMPONENT")
print("=" * 80)

# Save fitted model
model_path = OUTPUT_PATH / 'sarima_seasonal_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(sarima_fitted_full, f)
print(f"✅ Saved SARIMA model: {model_path}")

# Save test forecasts
test_forecast_df = pd.DataFrame({
    'date': test_ts.index,
    'actual': test_ts.values,
    'forecast': test_forecast.values
})
test_forecast_df.set_index('date', inplace=True)

test_forecast_path = OUTPUT_PATH / 'sarima_test_forecasts.csv'
test_forecast_df.to_csv(test_forecast_path)
print(f"✅ Saved test forecasts: {test_forecast_path}")

# Save future forecasts
future_forecast_path = OUTPUT_PATH / 'sarima_future_forecasts.csv'
future_forecast.to_csv(future_forecast_path, header=['forecast'])
print(f"✅ Saved future forecasts: {future_forecast_path}")

# Save performance metrics
metrics_df = pd.DataFrame({
    'metric': ['RMSE', 'MAE', 'Directional_Accuracy', 'Naive_RMSE', 'Improvement_Pct'],
    'value': [test_rmse, test_mae, directional_accuracy, naive_rmse, ((naive_rmse - test_rmse) / naive_rmse * 100)]
})
metrics_path = OUTPUT_PATH / 'sarima_performance_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"✅ Saved performance metrics: {metrics_path}")

# ============================================================================
# 11. Component Summary
# ============================================================================

print("\n" + "=" * 80)
print("SARIMA COMPONENT SUMMARY")
print("=" * 80)

print(f"""
Component: SARIMA Seasonal Model
Status: ✅ COMPLETE

Model Specifications:
  - Order (p,d,q): {best_params}
  - Seasonal (P,D,Q,s): {best_seasonal_params}
  - Training Period: {train_ts.index.min()} to {train_ts.index.max()}
  - Test Period: {test_ts.index.min()} to {test_ts.index.max()}
  - Future Forecast: {future_forecast.index.min()} to {future_forecast.index.max()}

Performance (Out-of-Sample):
  - RMSE: {test_rmse:.4f}
  - MAE: {test_mae:.4f}
  - Directional Accuracy: {directional_accuracy:.1f}%
  - Improvement over Naive: {((naive_rmse - test_rmse) / naive_rmse * 100):.1f}%

Ensemble Weight: 25%

Output Files:
  - Model: {model_path.name}
  - Test Forecasts: {test_forecast_path.name}
  - Future Forecasts: {future_forecast_path.name}
  - Performance Metrics: {metrics_path.name}

Next Steps:
  1. Combine VAR (30%) + GBM (45%) + SARIMA (25%) with Ridge meta-learner
  2. Generate final ensemble forecasts (2025 Q4 - 2027 Q4)
  3. Comprehensive validation and performance reporting
""")

print("=" * 80)
print("SARIMA COMPONENT BUILD COMPLETE")
print("=" * 80)
print("\n🎯 All 3 components ready for ensemble integration!")
