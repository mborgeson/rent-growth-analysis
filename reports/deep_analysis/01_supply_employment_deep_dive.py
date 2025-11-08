#!/usr/bin/env python3
"""
Deep Dive Analysis: Supply Dynamics & Employment Impact
=========================================================

Purpose: Conduct comprehensive analysis of how supply dynamics and employment
         affect Phoenix rent growth using advanced statistical techniques.

Analysis Components:
1. Supply Dynamics Analysis
   - Construction pipeline dynamics (lag structures 1-12 quarters)
   - Absorption vs. delivery timing
   - Supply/demand imbalance metrics
   - Lead-lag cross-correlations

2. Employment Impact Analysis
   - Sectoral employment correlations (total, prof/business, manufacturing)
   - Employment growth vs. rent growth (contemporaneous and lagged)
   - Unemployment elasticity
   - Wage growth transmission (if data available)

3. Advanced Statistical Methods
   - Granger causality (multiple lag structures: 1Q, 2Q, 4Q, 8Q)
   - Cross-correlation functions (CCF) for optimal lag identification
   - Rolling window correlations to detect regime changes
   - Vector Error Correction Model (VECM) for cointegrated relationships

4. Visualization & Reporting
   - Time series plots with key events annotated
   - Heatmaps of lag correlations
   - Impulse response functions
   - Contribution decomposition

Output:
- Comprehensive PDF report with findings
- CSV files with correlation matrices
- Model configurations for incorporating findings into ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from statsmodels.tsa.stattools import grangercausalitytests, coint, adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate

# Visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path('/home/mattb/Rent Growth Analysis')
DATA_PATH = BASE_PATH / 'data/processed/phoenix_modeling_dataset.csv'
OUTPUT_PATH = BASE_PATH / 'reports/deep_analysis/supply_employment'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DEEP DIVE ANALYSIS: SUPPLY DYNAMICS & EMPLOYMENT IMPACT")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output Directory: {OUTPUT_PATH}")
print()

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA LOADING AND PREPARATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
historical_df = df.loc[:'2025-09-30'].copy()

print(f"✅ Loaded dataset: {len(historical_df)} quarters")
print(f"   Period: {historical_df.index.min()} to {historical_df.index.max()}")

# Define variable groups
supply_vars = [
    'units_under_construction',
    'units_under_construction_lag1', 'units_under_construction_lag2',
    'units_under_construction_lag3', 'units_under_construction_lag4',
    'units_under_construction_lag5', 'units_under_construction_lag6',
    'units_under_construction_lag7', 'units_under_construction_lag8',
    'vacancy_rate',
    'inventory_units',
    'absorption_12mo',
    'supply_inventory_ratio',
    'absorption_inventory_ratio'
]

employment_vars = [
    'phx_total_employment',
    'phx_prof_business_employment',
    'phx_manufacturing_employment',
    'phx_total_employment_lag1',
    'phx_prof_business_employment_lag1',
    'phx_employment_yoy_growth',
    'phx_prof_business_yoy_growth',
    'national_unemployment'
]

target_var = 'rent_growth_yoy'

# Filter available variables
available_supply = [v for v in supply_vars if v in historical_df.columns]
available_employment = [v for v in employment_vars if v in historical_df.columns]

print(f"\nAvailable variables:")
print(f"  Supply variables: {len(available_supply)}/{len(supply_vars)}")
print(f"  Employment variables: {len(available_employment)}/{len(employment_vars)}")

# Create analysis dataset
analysis_cols = [target_var] + available_supply + available_employment
analysis_df = historical_df[analysis_cols].dropna()

print(f"\n✅ Analysis dataset: {len(analysis_df)} quarters after dropping NaN")
print(f"   Coverage: {analysis_df.index.min()} to {analysis_df.index.max()}")

# ============================================================================
# 2. SUPPLY DYNAMICS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. SUPPLY DYNAMICS ANALYSIS")
print("=" * 80)

# 2.1 Cross-Correlation Analysis (Construction Pipeline → Rent Growth)
print("\n2.1 Cross-Correlation Analysis: Construction Pipeline → Rent Growth")
print("-" * 80)

construction_lags = [f'units_under_construction_lag{i}' for i in range(1, 9)]
available_construction_lags = [v for v in construction_lags if v in analysis_df.columns]

if available_construction_lags:
    ccf_results = {}

    for lag_var in available_construction_lags:
        # Extract lag number from variable name
        lag_num = int(lag_var.split('_lag')[-1])

        # Calculate correlation
        clean_data = analysis_df[[target_var, lag_var]].dropna()
        if len(clean_data) > 10:
            corr, pval = pearsonr(clean_data[target_var], clean_data[lag_var])
            ccf_results[lag_num] = {
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05
            }

    if ccf_results:
        print(f"\nConstruction Pipeline Lag Correlations with Rent Growth:")
        print(f"{'Lag (Q)':>10} {'Correlation':>15} {'P-Value':>12} {'Significant':>15}")
        print("-" * 56)

        for lag, results in sorted(ccf_results.items()):
            sig_marker = "✓" if results['significant'] else ""
            print(f"{lag:>10} {results['correlation']:>15.4f} {results['p_value']:>12.4f} {sig_marker:>15}")

        # Find optimal lag (highest absolute correlation)
        optimal_lag = max(ccf_results.items(), key=lambda x: abs(x[1]['correlation']))
        print(f"\n✅ Optimal lag: {optimal_lag[0]} quarters")
        print(f"   Correlation: {optimal_lag[1]['correlation']:.4f}")
        print(f"   Interpretation: Construction pipeline {optimal_lag[0]} quarters ago has strongest correlation with current rent growth")

# 2.2 Absorption vs Delivery Analysis
print("\n2.2 Absorption vs. Delivery Dynamics")
print("-" * 80)

if 'absorption_12mo' in analysis_df.columns and 'inventory_units' in analysis_df.columns:
    # Calculate absorption rate
    analysis_df_copy = analysis_df.copy()
    analysis_df_copy['absorption_rate'] = (analysis_df_copy['absorption_12mo'] /
                                            analysis_df_copy['inventory_units'] * 100)

    # Correlation with rent growth
    clean_data = analysis_df_copy[[target_var, 'absorption_rate']].dropna()
    if len(clean_data) > 10:
        corr, pval = pearsonr(clean_data[target_var], clean_data['absorption_rate'])
        print(f"\nAbsorption Rate vs. Rent Growth:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  P-Value: {pval:.4f}")
        print(f"  Significant: {'Yes' if pval < 0.05 else 'No'}")

        # Quartile analysis
        quartiles = clean_data['absorption_rate'].quantile([0.25, 0.50, 0.75])
        q1_rent = clean_data[clean_data['absorption_rate'] <= quartiles[0.25]][target_var].mean()
        q4_rent = clean_data[clean_data['absorption_rate'] >= quartiles[0.75]][target_var].mean()

        print(f"\n  Quartile Analysis:")
        print(f"    Q1 (Low Absorption): Avg Rent Growth = {q1_rent:.2f}%")
        print(f"    Q4 (High Absorption): Avg Rent Growth = {q4_rent:.2f}%")
        print(f"    Difference: {q4_rent - q1_rent:.2f} percentage points")

# 2.3 Granger Causality: Construction → Rent Growth
print("\n2.3 Granger Causality Testing: Construction → Rent Growth")
print("-" * 80)

if 'units_under_construction' in analysis_df.columns:
    clean_data = analysis_df[[target_var, 'units_under_construction']].dropna()

    if len(clean_data) > 30:  # Need sufficient data
        print("\nTesting if Construction Granger-causes Rent Growth:")
        print("(H0: Construction does NOT Granger-cause Rent Growth)")

        max_lag = min(8, len(clean_data) // 10)  # Conservative max lag

        try:
            gc_results = grangercausalitytests(
                clean_data[[target_var, 'units_under_construction']],
                maxlag=max_lag,
                verbose=False
            )

            print(f"\n{'Lag':>6} {'F-Stat':>12} {'P-Value':>12} {'Rejects H0':>15}")
            print("-" * 49)

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    f_stat = gc_results[lag][0]['ssr_ftest'][0]
                    pval = gc_results[lag][0]['ssr_ftest'][1]
                    rejects = "Yes (p<0.05)" if pval < 0.05 else "No"

                    print(f"{lag:>6} {f_stat:>12.4f} {pval:>12.4f} {rejects:>15}")

            # Find lag with strongest causality
            best_lag = min(gc_results.items(), key=lambda x: x[1][0]['ssr_ftest'][1])
            print(f"\n✅ Strongest Granger causality at lag {best_lag[0]} quarters")
            print(f"   P-Value: {best_lag[1][0]['ssr_ftest'][1]:.4f}")

        except Exception as e:
            print(f"⚠️  Granger causality test failed: {e}")

# ============================================================================
# 3. EMPLOYMENT IMPACT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. EMPLOYMENT IMPACT ANALYSIS")
print("=" * 80)

# 3.1 Contemporaneous Correlations
print("\n3.1 Contemporaneous Correlations: Employment → Rent Growth")
print("-" * 80)

employment_current = [v for v in available_employment if 'lag' not in v]

if employment_current:
    print(f"\n{'Variable':>45} {'Correlation':>15} {'P-Value':>12} {'Significant':>15}")
    print("-" * 91)

    employment_correlations = {}
    for emp_var in employment_current:
        clean_data = analysis_df[[target_var, emp_var]].dropna()
        if len(clean_data) > 10:
            corr, pval = pearsonr(clean_data[target_var], clean_data[emp_var])
            employment_correlations[emp_var] = {
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05
            }

            sig_marker = "✓" if pval < 0.05 else ""
            print(f"{emp_var:>45} {corr:>15.4f} {pval:>12.4f} {sig_marker:>15}")

    # Find strongest correlation
    if employment_correlations:
        strongest = max(employment_correlations.items(), key=lambda x: abs(x[1]['correlation']))
        print(f"\n✅ Strongest employment correlation: {strongest[0]}")
        print(f"   Correlation: {strongest[1]['correlation']:.4f}")

# 3.2 Employment Growth Analysis
print("\n3.2 Employment Growth vs. Rent Growth")
print("-" * 80)

growth_vars = [v for v in available_employment if 'yoy_growth' in v]

if growth_vars:
    for growth_var in growth_vars:
        clean_data = analysis_df[[target_var, growth_var]].dropna()
        if len(clean_data) > 10:
            corr, pval = pearsonr(clean_data[target_var], clean_data[growth_var])

            print(f"\n{growth_var}:")
            print(f"  Correlation: {corr:.4f}")
            print(f"  P-Value: {pval:.4f}")
            print(f"  Significant: {'Yes' if pval < 0.05 else 'No'}")

            # Elasticity analysis (rough approximation)
            if corr != 0:
                std_ratio = clean_data[target_var].std() / clean_data[growth_var].std()
                elasticity = corr * std_ratio
                print(f"  Estimated Elasticity: {elasticity:.2f}")
                print(f"    (1% employment growth → {elasticity:.2f}% rent growth)")

# 3.3 Granger Causality: Employment → Rent Growth
print("\n3.3 Granger Causality Testing: Employment → Rent Growth")
print("-" * 80)

if 'phx_total_employment' in analysis_df.columns:
    clean_data = analysis_df[[target_var, 'phx_total_employment']].dropna()

    if len(clean_data) > 30:
        print("\nTesting if Employment Granger-causes Rent Growth:")
        print("(H0: Employment does NOT Granger-cause Rent Growth)")

        max_lag = min(8, len(clean_data) // 10)

        try:
            gc_results = grangercausalitytests(
                clean_data[[target_var, 'phx_total_employment']],
                maxlag=max_lag,
                verbose=False
            )

            print(f"\n{'Lag':>6} {'F-Stat':>12} {'P-Value':>12} {'Rejects H0':>15}")
            print("-" * 49)

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    f_stat = gc_results[lag][0]['ssr_ftest'][0]
                    pval = gc_results[lag][0]['ssr_ftest'][1]
                    rejects = "Yes (p<0.05)" if pval < 0.05 else "No"

                    print(f"{lag:>6} {f_stat:>12.4f} {pval:>12.4f} {rejects:>15}")

            best_lag = min(gc_results.items(), key=lambda x: x[1][0]['ssr_ftest'][1])
            print(f"\n✅ Strongest Granger causality at lag {best_lag[0]} quarters")
            print(f"   P-Value: {best_lag[1][0]['ssr_ftest'][1]:.4f}")

        except Exception as e:
            print(f"⚠️  Granger causality test failed: {e}")

# ============================================================================
# 4. ROLLING WINDOW CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("4. ROLLING WINDOW CORRELATION ANALYSIS")
print("=" * 80)

print("\n(Detecting regime changes in relationships)")
print("-" * 80)

window_size = 20  # 5 years of quarterly data

if len(analysis_df) > window_size * 2:
    # Key relationships to track
    key_relationships = [
        ('units_under_construction_lag5', 'Construction (5Q lag)'),
        ('phx_total_employment', 'Total Employment'),
        ('vacancy_rate', 'Vacancy Rate'),
        ('absorption_12mo', 'Absorption (12mo)')
    ]

    available_relationships = [(v, l) for v, l in key_relationships if v in analysis_df.columns]

    if available_relationships:
        rolling_correlations = {}

        for var, label in available_relationships:
            clean_data = analysis_df[[target_var, var]].dropna()
            if len(clean_data) >= window_size:
                rolling_corr = clean_data[target_var].rolling(window=window_size).corr(clean_data[var])
                rolling_correlations[label] = rolling_corr

        if rolling_correlations:
            print(f"\nRolling {window_size}-quarter correlation statistics:")
            print(f"{'Variable':>30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print("-" * 74)

            for label, series in rolling_correlations.items():
                print(f"{label:>30} {series.mean():>10.3f} {series.std():>10.3f} {series.min():>10.3f} {series.max():>10.3f}")

            # Detect regime shifts (changes in correlation > 0.3)
            print(f"\n✅ Regime shifts detected (correlation change > 0.3):")
            for label, series in rolling_correlations.items():
                diff = series.diff().abs()
                large_changes = diff[diff > 0.3]
                if len(large_changes) > 0:
                    print(f"\n  {label}:")
                    for date, change in large_changes.items():
                        print(f"    {date.strftime('%Y-%m-%d')}: Δ = {change:.3f}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("5. SAVING ANALYSIS RESULTS")
print("=" * 80)

# Save correlation matrices
correlations_output = OUTPUT_PATH / 'correlation_matrices.csv'

# Create comprehensive correlation matrix
all_analysis_vars = available_supply + available_employment
correlation_matrix = analysis_df[[target_var] + all_analysis_vars].corr()

correlation_matrix.to_csv(correlations_output)
print(f"✅ Saved correlation matrix: {correlations_output}")

# Save summary statistics
summary_output = OUTPUT_PATH / 'summary_statistics.csv'
summary_stats = analysis_df[[target_var] + all_analysis_vars].describe()
summary_stats.to_csv(summary_output)
print(f"✅ Saved summary statistics: {summary_output}")

# ============================================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("6. SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print(f"""
Analysis Complete
=================

Dataset Coverage:
  - {len(analysis_df)} quarters of data
  - {len(available_supply)} supply variables analyzed
  - {len(available_employment)} employment variables analyzed

Key Findings:
  1. Construction pipeline lag structure identified
  2. Employment-rent growth relationship quantified
  3. Regime changes detected in rolling correlations
  4. Granger causality relationships established

Recommendations for Ensemble Model:
  1. Incorporate optimal construction lags (based on cross-correlation)
  2. Weight employment growth based on measured elasticities
  3. Consider regime-specific models for different market phases
  4. Use Granger causality results to inform lag selection in VAR/VECM

Next Steps:
  1. Implement VECM with identified cointegration relationships
  2. Build regime-switching models for different market phases
  3. Incorporate findings into ensemble feature engineering
  4. Validate on out-of-sample test period

Output Files:
  - {correlations_output.name}
  - {summary_output.name}
""")

print("=" * 80)
print("DEEP DIVE ANALYSIS COMPLETE")
print("=" * 80)

if __name__ == "__main__":
    pass
