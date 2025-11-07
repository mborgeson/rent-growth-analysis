#!/usr/bin/env python3
"""
Analyze and summarize fetched Phoenix forecasting data
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("PHOENIX RENT GROWTH FORECASTING - DATA ANALYSIS")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# Load FRED National Macro Data
# ============================================================================

print("\n" + "="*80)
print("NATIONAL MACRO VARIABLES (FRED)")
print("="*80)

fred_df = pd.read_csv('data/raw/fred_national_macro.csv', index_col='date', parse_dates=True)

print(f"\nDataset Shape: {fred_df.shape}")
print(f"Date Range: {fred_df.index.min()} to {fred_df.index.max()}")
print(f"Total Observations: {len(fred_df):,}")

print("\nVariables Included:")
for col in fred_df.columns:
    non_null = fred_df[col].notna().sum()
    pct_complete = (non_null / len(fred_df)) * 100
    print(f"  {col:20s}: {non_null:6,} obs ({pct_complete:5.1f}% complete)")

print("\nSummary Statistics (Latest 12 Months):")
recent_data = fred_df.last('12M')
print("\nMortgage Rate (MORTGAGE30US):")
if 'MORTGAGE30US' in recent_data.columns:
    mort = recent_data['MORTGAGE30US'].dropna()
    if len(mort) > 0:
        print(f"  Current: {mort.iloc[-1]:.2f}%")
        print(f"  12-Mo Avg: {mort.mean():.2f}%")
        print(f"  12-Mo Range: {mort.min():.2f}% - {mort.max():.2f}%")

print("\nFed Funds Rate (DFF):")
if 'DFF' in recent_data.columns:
    dff = recent_data['DFF'].dropna()
    if len(dff) > 0:
        print(f"  Current: {dff.iloc[-1]:.2f}%")
        print(f"  12-Mo Avg: {dff.mean():.2f}%")
        print(f"  12-Mo Range: {dff.min():.2f}% - {dff.max():.2f}%")

print("\nUnemployment Rate (UNRATE):")
if 'UNRATE' in recent_data.columns:
    unrate = recent_data['UNRATE'].dropna()
    if len(unrate) > 0:
        print(f"  Current: {unrate.iloc[-1]:.1f}%")
        print(f"  12-Mo Avg: {unrate.mean():.1f}%")
        print(f"  12-Mo Range: {unrate.min():.1f}% - {unrate.max():.1f}%")

# ============================================================================
# Load Phoenix Home Price Data
# ============================================================================

print("\n" + "="*80)
print("PHOENIX HOME PRICES (FRED)")
print("="*80)

phx_df = pd.read_csv('data/raw/fred_phoenix_home_prices.csv', index_col='date', parse_dates=True)

print(f"\nDataset Shape: {phx_df.shape}")
print(f"Date Range: {phx_df.index.min()} to {phx_df.index.max()}")
print(f"Total Observations: {len(phx_df):,}")

# Calculate YoY growth
phx_df['yoy_growth'] = phx_df['PHXRNSA'].pct_change(12) * 100

print("\nPhoenix Home Price Index (PHXRNSA):")
recent_phx = phx_df.last('24M')
current_hpi = recent_phx['PHXRNSA'].iloc[-1]
yoy_growth = recent_phx['yoy_growth'].iloc[-1]

print(f"  Current Index: {current_hpi:.2f}")
print(f"  YoY Growth: {yoy_growth:+.1f}%")
print(f"  24-Mo Avg Growth: {recent_phx['yoy_growth'].mean():+.1f}%")

print("\nHistorical Price Growth Periods:")
phx_df_monthly = phx_df.resample('MS').last()

# 2010-2012 (Post-Crisis)
period_2010_2012 = phx_df_monthly.loc['2010':'2012', 'yoy_growth']
print(f"  2010-2012 (Post-Crisis): {period_2010_2012.mean():+.1f}% avg YoY")

# 2013-2019 (Recovery)
period_2013_2019 = phx_df_monthly.loc['2013':'2019', 'yoy_growth']
print(f"  2013-2019 (Recovery): {period_2013_2019.mean():+.1f}% avg YoY")

# 2020-2022 (Pandemic Boom)
period_2020_2022 = phx_df_monthly.loc['2020':'2022', 'yoy_growth']
print(f"  2020-2022 (Pandemic Boom): {period_2020_2022.mean():+.1f}% avg YoY")

# 2023-2025 (Normalization)
period_2023_2025 = phx_df_monthly.loc['2023':, 'yoy_growth']
print(f"  2023-2025 (Normalization): {period_2023_2025.mean():+.1f}% avg YoY")

# ============================================================================
# Data Gaps Analysis
# ============================================================================

print("\n" + "="*80)
print("DATA GAPS & MISSING VALUES")
print("="*80)

print("\nFRED National Macro - Missing Data Summary:")
for col in fred_df.columns:
    missing = fred_df[col].isna().sum()
    if missing > 0:
        pct_missing = (missing / len(fred_df)) * 100
        print(f"  {col:20s}: {missing:6,} missing ({pct_missing:5.1f}%)")

print("\nPhoenix Home Prices - Missing Data Summary:")
missing_phx = phx_df['PHXRNSA'].isna().sum()
print(f"  PHXRNSA: {missing_phx} missing values ({(missing_phx/len(phx_df)*100):.1f}%)")

# ============================================================================
# Variable Correlation Analysis
# ============================================================================

print("\n" + "="*80)
print("VARIABLE RELATIONSHIPS (Preliminary Analysis)")
print("="*80)

# Resample everything to monthly for correlation analysis
fred_monthly = fred_df.resample('MS').last()
phx_monthly = phx_df.resample('MS').last()

# Combine datasets
combined = pd.merge(fred_monthly, phx_monthly, left_index=True, right_index=True, how='inner')

# Calculate correlations with Phoenix home prices
print("\nCorrelation with Phoenix Home Price Growth:")
correlations = combined.corr()['yoy_growth'].sort_values(ascending=False)

for var, corr in correlations.items():
    if var != 'yoy_growth' and var != 'PHXRNSA':
        print(f"  {var:20s}: {corr:+.3f}")

# ============================================================================
# Export Summary
# ============================================================================

print("\n" + "="*80)
print("DATA EXPORT SUMMARY")
print("="*80)

output_summary = {
    'fred_national_macro': {
        'file': 'data/raw/fred_national_macro.csv',
        'size': '167 KB',
        'rows': len(fred_df),
        'columns': len(fred_df.columns),
        'date_range': f"{fred_df.index.min()} to {fred_df.index.max()}"
    },
    'phoenix_home_prices': {
        'file': 'data/raw/fred_phoenix_home_prices.csv',
        'size': '4.8 KB',
        'rows': len(phx_df),
        'columns': len(phx_df.columns),
        'date_range': f"{phx_df.index.min()} to {phx_df.index.max()}"
    }
}

print("\nSuccessfully Fetched Datasets:")
for dataset, info in output_summary.items():
    print(f"\n{dataset}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

print("\n" + "="*80)
print("CRITICAL NEXT STEPS")
print("="*80)
print("""
1. URGENT: Acquire rent growth data (dependent variable)
   - CoStar subscription or trial access
   - Contact commercial real estate brokers for market reports

2. Download BLS Phoenix employment data manually:
   https://data.bls.gov/cgi-bin/srgate
   Series: SMU04383400000000001, SMU04383406000000001

3. Download IRS migration data (CA → AZ):
   https://www.irs.gov/statistics/soi-tax-stats-migration-data

4. Download Zillow ZHVI for additional granularity:
   https://www.zillow.com/research/data/

5. Begin VAR model development with available FRED data

Estimated Model Readiness: 40%
Blocker: Dependent variable (rent growth) required for GBM and SARIMA training
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
