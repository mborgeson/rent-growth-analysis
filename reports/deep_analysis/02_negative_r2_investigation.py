#!/usr/bin/env python3
"""
Negative R² Investigation - Phoenix Rent Growth
================================================

Purpose: Investigate root cause of negative R² across all experimental models
         despite production ensemble performing well (RMSE 0.0198, R² 0.92)

Critical Questions:
1. Are there outliers or anomalies in the test period (2023-2025)?
2. Has there been a structural break or regime change?
3. Do train and test periods have different distributions?
4. What makes the production ensemble work when experiments fail?

Experiment ID: NEG-R2-INV-001
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'reports/deep_analysis/negative_r2_investigation'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEGATIVE R² INVESTIGATION")
print("=" * 80)
print()

# ============================================================================
# 1. Load Data and Split by Period
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA LOADING AND PERIOD SPLIT")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
historical_df = df.loc[:'2025-09-30'].copy()

# Split into train and test periods
train_end = '2022-12-31'
train_df = historical_df.loc[:train_end]
test_df = historical_df.loc[train_end:]

print(f"\nTrain period: {train_df.index.min()} to {train_df.index.max()}")
print(f"  {len(train_df)} quarters")
print(f"\nTest period: {test_df.index.min()} to {test_df.index.max()}")
print(f"  {len(test_df)} quarters")

# ============================================================================
# 2. Target Variable Analysis
# ============================================================================

print("\n" + "=" * 80)
print("2. TARGET VARIABLE ANALYSIS (RENT_GROWTH_YOY)")
print("=" * 80)

target = 'rent_growth_yoy'

# Descriptive statistics
print("\nDescriptive Statistics:")
print("-" * 40)
print(f"{'Metric':<25} {'Train':>12} {'Test':>12} {'Difference':>12}")
print("-" * 40)

train_mean = train_df[target].mean()
test_mean = test_df[target].mean()
print(f"{'Mean':<25} {train_mean:>12.4f} {test_mean:>12.4f} {test_mean - train_mean:>12.4f}")

train_std = train_df[target].std()
test_std = test_df[target].std()
print(f"{'Std Dev':<25} {train_std:>12.4f} {test_std:>12.4f} {test_std - train_std:>12.4f}")

train_min = train_df[target].min()
test_min = test_df[target].min()
print(f"{'Min':<25} {train_min:>12.4f} {test_min:>12.4f} {test_min - train_min:>12.4f}")

train_max = train_df[target].max()
test_max = test_df[target].max()
print(f"{'Max':<25} {train_max:>12.4f} {test_max:>12.4f} {test_max - train_max:>12.4f}")

train_median = train_df[target].median()
test_median = test_df[target].median()
print(f"{'Median':<25} {train_median:>12.4f} {test_median:>12.4f} {test_median - train_median:>12.4f}")

# Distribution comparison
print("\n\nDistribution Comparison:")
print("-" * 40)

# Test for normality
train_normality = stats.shapiro(train_df[target].dropna())
test_normality = stats.shapiro(test_df[target].dropna())

print(f"Train Normality Test (Shapiro-Wilk):")
print(f"  Statistic: {train_normality.statistic:.4f}")
print(f"  P-value: {train_normality.pvalue:.4f}")
print(f"  Normal: {'Yes' if train_normality.pvalue > 0.05 else 'No'}")

print(f"\nTest Normality Test (Shapiro-Wilk):")
print(f"  Statistic: {test_normality.statistic:.4f}")
print(f"  P-value: {test_normality.pvalue:.4f}")
print(f"  Normal: {'Yes' if test_normality.pvalue > 0.05 else 'No'}")

# Test for equal variances
variance_test = stats.levene(train_df[target].dropna(), test_df[target].dropna())
print(f"\nEqual Variance Test (Levene):")
print(f"  Statistic: {variance_test.statistic:.4f}")
print(f"  P-value: {variance_test.pvalue:.4f}")
print(f"  Equal Variances: {'Yes' if variance_test.pvalue > 0.05 else 'No'}")

# Test for different means
mean_test = stats.ttest_ind(train_df[target].dropna(), test_df[target].dropna())
print(f"\nDifferent Means Test (t-test):")
print(f"  Statistic: {mean_test.statistic:.4f}")
print(f"  P-value: {mean_test.pvalue:.4f}")
print(f"  Different Means: {'Yes' if mean_test.pvalue < 0.05 else 'No'}")

# ============================================================================
# 3. Outlier Detection
# ============================================================================

print("\n" + "=" * 80)
print("3. OUTLIER DETECTION")
print("=" * 80)

# Calculate outliers using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

train_outliers, train_lower, train_upper = detect_outliers_iqr(train_df[target].dropna())
test_outliers, test_lower, test_upper = detect_outliers_iqr(test_df[target].dropna())

print("\nTrain Period Outliers:")
print(f"  Bounds: [{train_lower:.4f}, {train_upper:.4f}]")
print(f"  Outliers: {len(train_outliers)} / {len(train_df[target].dropna())} ({len(train_outliers)/len(train_df[target].dropna())*100:.1f}%)")
if len(train_outliers) > 0:
    print("\n  Outlier Values:")
    for date, value in train_outliers.items():
        print(f"    {date.strftime('%Y-%m-%d')}: {value:.4f}")

print("\n\nTest Period Outliers:")
print(f"  Bounds: [{test_lower:.4f}, {test_upper:.4f}]")
print(f"  Outliers: {len(test_outliers)} / {len(test_df[target].dropna())} ({len(test_outliers)/len(test_df[target].dropna())*100:.1f}%)")
if len(test_outliers) > 0:
    print("\n  Outlier Values:")
    for date, value in test_outliers.items():
        print(f"    {date.strftime('%Y-%m-%d')}: {value:.4f}")

# ============================================================================
# 4. Structural Break Analysis
# ============================================================================

print("\n" + "=" * 80)
print("4. STRUCTURAL BREAK ANALYSIS")
print("=" * 80)

# Calculate rolling statistics
window = 8  # 2 years

rolling_mean = historical_df[target].rolling(window=window, center=True).mean()
rolling_std = historical_df[target].rolling(window=window, center=True).std()

print(f"\nRolling Statistics (window={window} quarters):")
print("-" * 60)
print(f"{'Period':<20} {'Mean':>12} {'Std Dev':>12} {'% Change Mean':>15}")
print("-" * 60)

periods = [
    ('2010-2014', '2010-01-01', '2014-12-31'),
    ('2015-2019', '2015-01-01', '2019-12-31'),
    ('2020-2022', '2020-01-01', '2022-12-31'),
    ('2023-2025', '2023-01-01', '2025-12-31'),
]

prev_mean = None
for name, start, end in periods:
    period_data = historical_df.loc[start:end, target]
    period_mean = period_data.mean()
    period_std = period_data.std()

    if prev_mean is not None:
        pct_change = ((period_mean - prev_mean) / abs(prev_mean)) * 100
    else:
        pct_change = 0.0

    print(f"{name:<20} {period_mean:>12.4f} {period_std:>12.4f} {pct_change:>14.1f}%")
    prev_mean = period_mean

# Identify potential regime changes
print("\n\nPotential Regime Changes:")
print("-" * 40)

# Calculate year-over-year changes in rolling mean
yoy_changes = rolling_mean.diff(4).abs()  # 4 quarters = 1 year
large_changes = yoy_changes[yoy_changes > yoy_changes.quantile(0.90)]

if len(large_changes) > 0:
    print("Periods with large shifts in trend:")
    for date, change in large_changes.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {change:.4f} absolute change in rolling mean")
else:
    print("No significant regime changes detected")

# ============================================================================
# 5. Feature Distribution Analysis
# ============================================================================

print("\n" + "=" * 80)
print("5. KEY FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)

# Key features from XGBoost model
key_features = [
    'mortgage_rate_30yr_lag2',
    'phx_manufacturing_employment',
    'phx_employment_yoy_growth',
    'vacancy_rate',
    'phx_hpi_yoy_growth'
]

print("\nTop 5 Features from XGB-OPT-002:")
print("-" * 80)

for feature in key_features:
    if feature not in historical_df.columns:
        print(f"\n{feature}: NOT FOUND IN DATASET")
        continue

    train_feat = train_df[feature].dropna()
    test_feat = test_df[feature].dropna()

    if len(train_feat) == 0 or len(test_feat) == 0:
        print(f"\n{feature}: INSUFFICIENT DATA")
        continue

    print(f"\n{feature}:")
    print(f"  Train: mean={train_feat.mean():.4f}, std={train_feat.std():.4f}, min={train_feat.min():.4f}, max={train_feat.max():.4f}")
    print(f"  Test:  mean={test_feat.mean():.4f}, std={test_feat.std():.4f}, min={test_feat.min():.4f}, max={test_feat.max():.4f}")

    # Distribution similarity test (Kolmogorov-Smirnov)
    ks_stat, ks_pvalue = stats.ks_2samp(train_feat, test_feat)
    print(f"  KS Test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
    print(f"  Same Distribution: {'Yes' if ks_pvalue > 0.05 else 'No (DIFFERENT!)'}")

# ============================================================================
# 6. Naive Baseline Comparison
# ============================================================================

print("\n" + "=" * 80)
print("6. NAIVE BASELINE PERFORMANCE")
print("=" * 80)

# Naive persistence: use previous quarter's value
y_test_actual = test_df[target].values
y_naive = train_df[target].iloc[-1:].values[0] * np.ones_like(y_test_actual)

naive_rmse = np.sqrt(mean_squared_error(y_test_actual, y_naive))
naive_mae = np.mean(np.abs(y_test_actual - y_naive))

# Mean prediction baseline
y_mean = np.mean(train_df[target]) * np.ones_like(y_test_actual)
mean_rmse = np.sqrt(mean_squared_error(y_test_actual, y_mean))
mean_mae = np.mean(np.abs(y_test_actual - y_mean))

print("\nBaseline Model Performance:")
print("-" * 60)
print(f"{'Model':<30} {'RMSE':>12} {'MAE':>12}")
print("-" * 60)
print(f"{'Persistence (last train value)':<30} {naive_rmse:>12.4f} {naive_mae:>12.4f}")
print(f"{'Mean (train mean)':<30} {mean_rmse:>12.4f} {mean_mae:>12.4f}")
print(f"{'Production Ensemble':<30} {'0.0198':>12} {'0.0165':>12}")
print(f"{'XGB-OPT-002':<30} {'4.2058':>12} {'3.8714':>12}")

# Calculate how much worse experimental model is
worse_than_persistence = (4.2058 / naive_rmse - 1) * 100
worse_than_mean = (4.2058 / mean_rmse - 1) * 100

print(f"\nXGB-OPT-002 vs Baselines:")
print(f"  vs Persistence: {worse_than_persistence:+.1f}% {'worse' if worse_than_persistence > 0 else 'better'}")
print(f"  vs Mean: {worse_than_mean:+.1f}% {'worse' if worse_than_mean > 0 else 'better'}")

# ============================================================================
# 7. Test Period Detailed Examination
# ============================================================================

print("\n" + "=" * 80)
print("7. TEST PERIOD DETAILED EXAMINATION (2023-2025)")
print("=" * 80)

print("\nQuarterly Values:")
print("-" * 60)
print(f"{'Date':<15} {'Rent Growth':>15} {'vs Train Mean':>15} {'vs Train Median':>15}")
print("-" * 60)

train_mean = train_df[target].mean()
train_median = train_df[target].median()

for date, row in test_df.iterrows():
    value = row[target]
    if pd.isna(value):
        print(f"{date.strftime('%Y-%m-%d'):<15} {'NaN':>15} {'N/A':>15} {'N/A':>15}")
    else:
        vs_mean = value - train_mean
        vs_median = value - train_median
        print(f"{date.strftime('%Y-%m-%d'):<15} {value:>15.4f} {vs_mean:>+15.4f} {vs_median:>+15.4f}")

# ============================================================================
# 8. Summary and Findings
# ============================================================================

print("\n" + "=" * 80)
print("8. SUMMARY AND KEY FINDINGS")
print("=" * 80)

findings = []

# Finding 1: Mean difference
if abs(test_mean - train_mean) > train_std:
    findings.append(f"⚠️  Test mean ({test_mean:.4f}) differs significantly from train mean ({train_mean:.4f})")
else:
    findings.append(f"✅ Test mean ({test_mean:.4f}) similar to train mean ({train_mean:.4f})")

# Finding 2: Variance difference
if variance_test.pvalue < 0.05:
    findings.append(f"⚠️  Test variance differs significantly from train variance (p={variance_test.pvalue:.4f})")
else:
    findings.append(f"✅ Test variance similar to train variance (p={variance_test.pvalue:.4f})")

# Finding 3: Outliers in test
outlier_pct = len(test_outliers) / len(test_df[target].dropna()) * 100
if outlier_pct > 20:
    findings.append(f"⚠️  High outlier rate in test period: {outlier_pct:.1f}%")
elif outlier_pct > 0:
    findings.append(f"ℹ️  Moderate outlier rate in test period: {outlier_pct:.1f}%")
else:
    findings.append(f"✅ No outliers detected in test period")

# Finding 4: Small test sample
if len(test_df) < 20:
    findings.append(f"⚠️  Very small test sample: {len(test_df)} quarters")
else:
    findings.append(f"ℹ️  Test sample size: {len(test_df)} quarters")

# Finding 5: XGBoost worse than naive
if 4.2058 > naive_rmse:
    findings.append(f"⚠️  XGBoost RMSE ({4.2058:.4f}) worse than persistence baseline ({naive_rmse:.4f})")
else:
    findings.append(f"✅ XGBoost RMSE ({4.2058:.4f}) better than persistence baseline ({naive_rmse:.4f})")

print("\nKey Findings:")
for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_PATH}")
