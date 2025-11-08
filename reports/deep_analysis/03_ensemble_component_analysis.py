#!/usr/bin/env python3
"""
Production Ensemble Component Analysis
========================================

Purpose: Decompose production ensemble performance to understand why it succeeds
         during regime change when all experimental models fail

Critical Questions:
1. How does each component (GBM, SARIMA) perform in train vs test periods?
2. Which component handles regime change best?
3. How does the Ridge meta-learner adapt weights?
4. What lessons can we extract for regime-adaptive modeling?

Context:
- Production ensemble: RMSE 0.0198, R² 0.92 ✅
- XGB-OPT-002: RMSE 4.2058, R² -18.53 ❌
- Regime change: +4.33% (train) → -1.34% (test)

Investigation ID: ENS-DECOMP-001
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'models/output'
ANALYSIS_OUTPUT = BASE_PATH / 'reports/deep_analysis/ensemble_decomposition'
ANALYSIS_OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRODUCTION ENSEMBLE COMPONENT ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("1. LOADING DATA AND PREDICTIONS")
print("=" * 80)

# Load ground truth
df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
actual_rent_growth = df['rent_growth_yoy'].dropna()

# Load ensemble predictions
ensemble_pred = pd.read_csv(OUTPUT_PATH / 'ensemble_predictions.csv',
                            parse_dates=['date'], index_col='date')
ensemble_weights = pd.read_csv(OUTPUT_PATH / 'ensemble_component_weights.csv')
ensemble_metrics = pd.read_csv(OUTPUT_PATH / 'ensemble_performance_metrics.csv')

# Load component predictions
gbm_pred = pd.read_csv(OUTPUT_PATH / 'gbm_predictions.csv',
                       parse_dates=['date'], index_col='date')
gbm_metrics = pd.read_csv(OUTPUT_PATH / 'gbm_performance_metrics.csv')

sarima_test = pd.read_csv(OUTPUT_PATH / 'sarima_test_forecasts.csv',
                          parse_dates=['date'], index_col='date')
sarima_metrics = pd.read_csv(OUTPUT_PATH / 'sarima_performance_metrics.csv')

print(f"\n✅ Loaded data:")
print(f"   Ensemble predictions: {len(ensemble_pred)} quarters")
print(f"   GBM predictions: {len(gbm_pred)} quarters")
print(f"   SARIMA test forecasts: {len(sarima_test)} quarters")
print(f"   Ensemble weights: {len(ensemble_weights)} components")

# ============================================================================
# 2. Period Split: Train vs Test
# ============================================================================

print("\n" + "=" * 80)
print("2. PERIOD SPLIT ANALYSIS")
print("=" * 80)

# Define periods based on regime change investigation
train_end = '2022-12-31'

# Ensemble predictions (test period only in saved file)
ensemble_test = ensemble_pred.copy()

print(f"\nTest Period (Regime Change):")
print(f"  Dates: {ensemble_test.index.min()} to {ensemble_test.index.max()}")
print(f"  Quarters: {len(ensemble_test)}")

# Get GBM train and test
gbm_train = gbm_pred[gbm_pred['split'] == 'train'].copy()
gbm_test = gbm_pred[gbm_pred['split'] == 'test'].copy()

print(f"\nGBM Component:")
print(f"  Train: {len(gbm_train)} quarters ({gbm_train.index.min()} to {gbm_train.index.max()})")
print(f"  Test:  {len(gbm_test)} quarters ({gbm_test.index.min()} to {gbm_test.index.max()})")

# SARIMA only has test forecasts
print(f"\nSARIMA Component:")
print(f"  Test forecasts: {len(sarima_test)} quarters")

# ============================================================================
# 3. Component Performance Analysis
# ============================================================================

print("\n" + "=" * 80)
print("3. COMPONENT PERFORMANCE ANALYSIS")
print("=" * 80)

# Helper function for metrics
def calculate_metrics(actual, predicted, period_name):
    """Calculate performance metrics for a prediction set"""
    # Remove NaN values
    valid_idx = ~(pd.isna(actual) | pd.isna(predicted))
    actual_clean = actual[valid_idx]
    predicted_clean = predicted[valid_idx]

    if len(actual_clean) < 2:
        return {
            'period': period_name,
            'n_samples': len(actual_clean),
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'directional_accuracy': np.nan
        }

    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mae = mean_absolute_error(actual_clean, predicted_clean)
    r2 = r2_score(actual_clean, predicted_clean)

    # Directional accuracy (requires at least 2 points)
    if len(actual_clean) >= 2:
        actual_direction = np.sign(np.diff(actual_clean))
        pred_direction = np.sign(np.diff(predicted_clean))
        dir_acc = np.mean(actual_direction == pred_direction) * 100
    else:
        dir_acc = np.nan

    return {
        'period': period_name,
        'n_samples': len(actual_clean),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': dir_acc
    }

# Analyze GBM component by period
print("\n" + "-" * 80)
print("GBM Component (Phoenix-Specific)")
print("-" * 80)

gbm_train_metrics = calculate_metrics(
    gbm_train['actual'].values,
    gbm_train['predicted'].values,
    'Train (2010-2022)'
)

gbm_test_metrics = calculate_metrics(
    gbm_test['actual'].values,
    gbm_test['predicted'].values,
    'Test (2023-2025)'
)

print(f"\nTrain Period (Growth Regime: +4.33% avg):")
print(f"  RMSE: {gbm_train_metrics['rmse']:.4f}")
print(f"  MAE:  {gbm_train_metrics['mae']:.4f}")
print(f"  R²:   {gbm_train_metrics['r2']:.4f}")
print(f"  Directional Accuracy: {gbm_train_metrics['directional_accuracy']:.1f}%")

print(f"\nTest Period (Decline Regime: -1.34% avg):")
print(f"  RMSE: {gbm_test_metrics['rmse']:.4f}")
print(f"  MAE:  {gbm_test_metrics['mae']:.4f}")
print(f"  R²:   {gbm_test_metrics['r2']:.4f}")
print(f"  Directional Accuracy: {gbm_test_metrics['directional_accuracy']:.1f}%")

# Performance degradation
if not np.isnan(gbm_test_metrics['rmse']) and not np.isnan(gbm_train_metrics['rmse']):
    gbm_degradation = ((gbm_test_metrics['rmse'] - gbm_train_metrics['rmse']) /
                       gbm_train_metrics['rmse'] * 100)
    print(f"\nPerformance Change (Train → Test): {gbm_degradation:+.1f}%")

# Analyze SARIMA component
print("\n" + "-" * 80)
print("SARIMA Component (Seasonal)")
print("-" * 80)

sarima_test_metrics = calculate_metrics(
    ensemble_test['actual'].values,
    ensemble_test['sarima'].values,
    'Test (2023-2025)'
)

print(f"\nTest Period Performance:")
print(f"  RMSE: {sarima_test_metrics['rmse']:.4f}")
print(f"  MAE:  {sarima_test_metrics['mae']:.4f}")
print(f"  R²:   {sarima_test_metrics['r2']:.4f}")
print(f"  Directional Accuracy: {sarima_test_metrics['directional_accuracy']:.1f}%")

# Analyze Ensemble
print("\n" + "-" * 80)
print("Ensemble (Ridge Meta-Learner)")
print("-" * 80)

ensemble_test_metrics = calculate_metrics(
    ensemble_test['actual'].values,
    ensemble_test['ensemble'].values,
    'Test (2023-2025)'
)

print(f"\nTest Period Performance:")
print(f"  RMSE: {ensemble_test_metrics['rmse']:.4f}")
print(f"  MAE:  {ensemble_test_metrics['mae']:.4f}")
print(f"  R²:   {ensemble_test_metrics['r2']:.4f}")
print(f"  Directional Accuracy: {ensemble_test_metrics['directional_accuracy']:.1f}%")

# ============================================================================
# 4. Component Comparison
# ============================================================================

print("\n" + "=" * 80)
print("4. COMPONENT COMPARISON (TEST PERIOD)")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Component': ['GBM', 'SARIMA', 'Ensemble', 'XGB-OPT-002 (Failed)'],
    'RMSE': [
        gbm_test_metrics['rmse'],
        sarima_test_metrics['rmse'],
        ensemble_test_metrics['rmse'],
        4.2058  # From investigation
    ],
    'R²': [
        gbm_test_metrics['r2'],
        sarima_test_metrics['r2'],
        ensemble_test_metrics['r2'],
        -18.53  # From investigation
    ],
    'Dir_Acc_%': [
        gbm_test_metrics['directional_accuracy'],
        sarima_test_metrics['directional_accuracy'],
        ensemble_test_metrics['directional_accuracy'],
        45.5  # From investigation
    ]
})

print("\n" + "-" * 80)
print(f"{'Component':<25} {'RMSE':>12} {'R²':>12} {'Dir. Acc.':>12}")
print("-" * 80)
for _, row in comparison_df.iterrows():
    print(f"{row['Component']:<25} {row['RMSE']:>12.4f} {row['R²']:>12.4f} {row['Dir_Acc_%']:>11.1f}%")

# Calculate improvements
gbm_improvement = ((gbm_test_metrics['rmse'] - ensemble_test_metrics['rmse']) /
                   gbm_test_metrics['rmse'] * 100)
sarima_improvement = ((sarima_test_metrics['rmse'] - ensemble_test_metrics['rmse']) /
                      sarima_test_metrics['rmse'] * 100)

print(f"\nEnsemble vs Components:")
print(f"  vs GBM:    {gbm_improvement:+.1f}% improvement")
print(f"  vs SARIMA: {sarima_improvement:+.1f}% improvement")

# ============================================================================
# 5. Meta-Learner Weight Analysis
# ============================================================================

print("\n" + "=" * 80)
print("5. RIDGE META-LEARNER WEIGHT ANALYSIS")
print("=" * 80)

print("\nComponent Weights:")
print("-" * 40)
for _, row in ensemble_weights.iterrows():
    component = row['component']
    weight = row['weight']
    pct = row['normalized_pct']

    if component != 'Intercept':
        print(f"  {component:<10}: weight={weight:>8.4f}, contribution={pct:>6.1f}%")
    else:
        print(f"  {component:<10}: {weight:>8.4f}")

# Analyze if weights correlate with performance
gbm_weight = ensemble_weights[ensemble_weights['component'] == 'GBM']['normalized_pct'].values[0]
sarima_weight = ensemble_weights[ensemble_weights['component'] == 'SARIMA']['normalized_pct'].values[0]

print(f"\nWeight vs Performance Analysis:")
print(f"  GBM:    {gbm_weight:.1f}% weight, RMSE {gbm_test_metrics['rmse']:.4f}")
print(f"  SARIMA: {sarima_weight:.1f}% weight, RMSE {sarima_test_metrics['rmse']:.4f}")

# Which component is better in test period?
if gbm_test_metrics['rmse'] < sarima_test_metrics['rmse']:
    print(f"\n✅ GBM performs better in test period (lower RMSE)")
    print(f"   Ridge meta-learner weights GBM at {gbm_weight:.1f}% (appropriate)")
else:
    print(f"\n✅ SARIMA performs better in test period (lower RMSE)")
    print(f"   Ridge meta-learner weights SARIMA at {sarima_weight:.1f}%")

# ============================================================================
# 6. Regime Adaptation Analysis
# ============================================================================

print("\n" + "=" * 80)
print("6. REGIME ADAPTATION ANALYSIS")
print("=" * 80)

print("\nHow does each component handle regime change?")
print("-" * 60)

# GBM: Train vs Test performance degradation
if not np.isnan(gbm_degradation):
    print(f"\nGBM (Phoenix-Specific):")
    print(f"  Train RMSE: {gbm_train_metrics['rmse']:.4f}")
    print(f"  Test RMSE:  {gbm_test_metrics['rmse']:.4f}")
    print(f"  Degradation: {gbm_degradation:+.1f}%")

    if gbm_degradation < 100:
        print(f"  ✅ MODERATE degradation - GBM adapts reasonably well")
    else:
        print(f"  ⚠️  SIGNIFICANT degradation - GBM struggles with regime")

# SARIMA: Only test period available
print(f"\nSARIMA (Seasonal):")
print(f"  Test RMSE: {sarima_test_metrics['rmse']:.4f}")
print(f"  Test R²:   {sarima_test_metrics['r2']:.4f}")

if sarima_test_metrics['r2'] > 0:
    print(f"  ✅ POSITIVE R² - SARIMA captures regime reasonably")
elif sarima_test_metrics['r2'] > -5:
    print(f"  ⚠️  NEGATIVE R² but modest - SARIMA struggles moderately")
else:
    print(f"  ❌ VERY NEGATIVE R² - SARIMA fails to capture regime")

# Ensemble: Regime adaptation success
print(f"\nEnsemble (Ridge Meta-Learner):")
print(f"  Test RMSE: {ensemble_test_metrics['rmse']:.4f}")
print(f"  Test R²:   {ensemble_test_metrics['r2']:.4f}")

if ensemble_test_metrics['r2'] > 0.85:
    print(f"  ✅ EXCELLENT R² - Ensemble adapts very well to regime change")
elif ensemble_test_metrics['r2'] > 0.5:
    print(f"  ✅ GOOD R² - Ensemble adapts well to regime change")
else:
    print(f"  ⚠️  MODERATE R² - Ensemble has some regime adaptation issues")

# ============================================================================
# 7. Quarterly Breakdown Analysis
# ============================================================================

print("\n" + "=" * 80)
print("7. QUARTERLY BREAKDOWN (TEST PERIOD)")
print("=" * 80)

print("\n" + "-" * 90)
print(f"{'Date':<15} {'Actual':>10} {'GBM':>10} {'SARIMA':>10} {'Ensemble':>10} {'GBM Err':>10} {'Ensemble Err':>12}")
print("-" * 90)

for date, row in ensemble_test.iterrows():
    actual = row['actual']
    gbm = row['gbm']
    sarima = row['sarima']
    ensemble = row['ensemble']

    gbm_error = abs(gbm - actual)
    ensemble_error = abs(ensemble - actual)

    print(f"{date.strftime('%Y-%m-%d'):<15} {actual:>10.2f} {gbm:>10.2f} "
          f"{sarima:>10.2f} {ensemble:>10.2f} {gbm_error:>10.2f} {ensemble_error:>12.2f}")

# ============================================================================
# 8. Key Insights Summary
# ============================================================================

print("\n" + "=" * 80)
print("8. KEY INSIGHTS - WHY ENSEMBLE SUCCEEDS")
print("=" * 80)

insights = []

# Insight 1: Component performance
if gbm_test_metrics['r2'] > 0:
    insights.append("✅ GBM maintains positive R² in test period despite regime change")
else:
    insights.append("⚠️  GBM has negative R² in test period but better than experimental models")

# Insight 2: SARIMA performance
if sarima_test_metrics['r2'] > 0:
    insights.append("✅ SARIMA maintains positive R² - seasonal patterns remain stable")
else:
    insights.append("ℹ️  SARIMA struggles but provides diversification")

# Insight 3: Ensemble advantage
ensemble_vs_gbm = ((gbm_test_metrics['rmse'] - ensemble_test_metrics['rmse']) /
                   gbm_test_metrics['rmse'] * 100)
if ensemble_vs_gbm > 5:
    insights.append(f"✅ Ensemble {ensemble_vs_gbm:.1f}% better than best component (diversification benefit)")
else:
    insights.append(f"ℹ️  Ensemble marginally better than components ({ensemble_vs_gbm:.1f}%)")

# Insight 4: Experimental model comparison
experimental_gap = ((4.2058 - ensemble_test_metrics['rmse']) / 4.2058 * 100)
insights.append(f"🎯 Ensemble {experimental_gap:.1f}% better than experimental XGBoost")

# Insight 5: Component weights
if gbm_weight > 70:
    insights.append(f"ℹ️  Ridge meta-learner heavily weights GBM ({gbm_weight:.1f}%) - relies on Phoenix dynamics")
elif sarima_weight > 70:
    insights.append(f"ℹ️  Ridge meta-learner heavily weights SARIMA ({sarima_weight:.1f}%) - relies on seasonality")
else:
    insights.append(f"✅ Ridge meta-learner balances components (GBM {gbm_weight:.1f}%, SARIMA {sarima_weight:.1f}%)")

print("\nKey Findings:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# ============================================================================
# 9. Lessons for Regime-Adaptive Modeling
# ============================================================================

print("\n" + "=" * 80)
print("9. LESSONS FOR REGIME-ADAPTIVE EXPERIMENTAL MODELS")
print("=" * 80)

lessons = []

# Lesson 1: Component diversity
if len(insights) > 0:
    lessons.append(
        "Component Diversity: Use multiple model types (tree-based, time series, etc.) "
        "to capture different aspects of regime dynamics"
    )

# Lesson 2: Meta-learning
lessons.append(
    "Adaptive Meta-Learning: Ridge regression learns optimal weights automatically "
    "rather than fixed weights"
)

# Lesson 3: Simpler individual components
if gbm_train_metrics['rmse'] < 1.0:  # If GBM performs well
    lessons.append(
        "Simpler Components: Individual GBM/SARIMA components may be simpler "
        "than experimental XGBoost (25 features)"
    )

# Lesson 4: Feature stability
lessons.append(
    "Feature Stability: GBM likely uses Phoenix-specific features that maintain "
    "relationships across regimes (employment YoY growth)"
)

# Lesson 5: Seasonal patterns
if sarima_test_metrics['r2'] > -10:
    lessons.append(
        "Seasonal Stability: Seasonal patterns (SARIMA captures) remain more "
        "stable than levels during regime changes"
    )

# Lesson 6: Regularization
lessons.append(
    "Regularization: Ridge regression provides implicit regularization, "
    "preventing overfitting to any single component"
)

print("\nLessons Learned:")
for i, lesson in enumerate(lessons, 1):
    print(f"\n{i}. {lesson}")

# ============================================================================
# 10. Recommendations for Experimental Models
# ============================================================================

print("\n" + "=" * 80)
print("10. RECOMMENDATIONS FOR EXPERIMENTAL MODELS")
print("=" * 80)

recommendations = [
    {
        'priority': 'HIGH',
        'title': 'Implement Ensemble Architecture',
        'description': 'Use ensemble of diverse models rather than single complex model',
        'action': 'Create XGBoost + SARIMA + LightGBM ensemble with Ridge meta-learner'
    },
    {
        'priority': 'HIGH',
        'title': 'Simplify Feature Sets',
        'description': 'Reduce from 25 to 5-10 most stable features',
        'action': 'Focus on employment_yoy_growth and seasonality-stable features'
    },
    {
        'priority': 'HIGH',
        'title': 'Add Seasonal Component',
        'description': 'SARIMA captures patterns that remain stable across regimes',
        'action': 'Implement SARIMAX experimental variant'
    },
    {
        'priority': 'MEDIUM',
        'title': 'Adaptive Weighting',
        'description': 'Let meta-learner adjust weights based on recent performance',
        'action': 'Implement Ridge or elastic net meta-learner'
    },
    {
        'priority': 'MEDIUM',
        'title': 'Component Specialization',
        'description': 'Different models capture different regime aspects',
        'action': 'XGBoost for Phoenix dynamics, SARIMA for seasonality, VAR for macro'
    },
    {
        'priority': 'LOWER',
        'title': 'Online Learning',
        'description': 'Update weights as new data arrives',
        'action': 'Implement rolling window meta-learner retraining'
    }
]

for rec in recommendations:
    print(f"\n[{rec['priority']}] {rec['title']}")
    print(f"  Why: {rec['description']}")
    print(f"  Action: {rec['action']}")

# ============================================================================
# 11. Save Analysis Results
# ============================================================================

print("\n" + "=" * 80)
print("11. SAVING ANALYSIS RESULTS")
print("=" * 80)

# Save component comparison
comparison_df.to_csv(ANALYSIS_OUTPUT / 'component_comparison.csv', index=False)
print(f"✅ Saved: component_comparison.csv")

# Save metrics by period
metrics_summary = pd.DataFrame([
    {'component': 'GBM', 'period': 'train', **gbm_train_metrics},
    {'component': 'GBM', 'period': 'test', **gbm_test_metrics},
    {'component': 'SARIMA', 'period': 'test', **sarima_test_metrics},
    {'component': 'Ensemble', 'period': 'test', **ensemble_test_metrics}
])
metrics_summary.to_csv(ANALYSIS_OUTPUT / 'metrics_by_period.csv', index=False)
print(f"✅ Saved: metrics_by_period.csv")

# Save insights and lessons
with open(ANALYSIS_OUTPUT / 'insights_and_lessons.txt', 'w') as f:
    f.write("ENSEMBLE COMPONENT ANALYSIS INSIGHTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("KEY INSIGHTS:\n")
    f.write("-" * 40 + "\n")
    for i, insight in enumerate(insights, 1):
        f.write(f"{i}. {insight}\n")

    f.write("\n\nLESSONS LEARNED:\n")
    f.write("-" * 40 + "\n")
    for i, lesson in enumerate(lessons, 1):
        f.write(f"{i}. {lesson}\n")

    f.write("\n\nRECOMMENDATIONS:\n")
    f.write("-" * 40 + "\n")
    for rec in recommendations:
        f.write(f"\n[{rec['priority']}] {rec['title']}\n")
        f.write(f"  {rec['description']}\n")
        f.write(f"  Action: {rec['action']}\n")

print(f"✅ Saved: insights_and_lessons.txt")

# Save quarterly breakdown
quarterly_analysis = ensemble_test.copy()
quarterly_analysis['gbm_error'] = abs(quarterly_analysis['gbm'] - quarterly_analysis['actual'])
quarterly_analysis['sarima_error'] = abs(quarterly_analysis['sarima'] - quarterly_analysis['actual'])
quarterly_analysis['ensemble_error'] = abs(quarterly_analysis['ensemble'] - quarterly_analysis['actual'])
quarterly_analysis.to_csv(ANALYSIS_OUTPUT / 'quarterly_breakdown.csv')
print(f"✅ Saved: quarterly_breakdown.csv")

print("\n" + "=" * 80)
print("ENSEMBLE COMPONENT ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {ANALYSIS_OUTPUT}")
print("\nKey Findings:")
print(f"  1. Ensemble RMSE: {ensemble_test_metrics['rmse']:.4f} (vs XGBoost {4.2058:.4f})")
print(f"  2. Ensemble R²: {ensemble_test_metrics['r2']:.4f} (vs XGBoost {-18.53:.2f})")
print(f"  3. Performance gap: {experimental_gap:.1f}% improvement")
print(f"  4. Component weights: GBM {gbm_weight:.1f}%, SARIMA {sarima_weight:.1f}%")
print("\nNext Steps:")
print("  1. Review insights_and_lessons.txt for detailed findings")
print("  2. Implement ensemble architecture for experimental models")
print("  3. Focus on feature stability and seasonal patterns")
print("  4. Use adaptive meta-learning approach")
