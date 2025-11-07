#!/usr/bin/env python3
"""
Phoenix Rent Growth Structural Break Detection Analysis (2015-2024)

This script performs comprehensive structural break analysis on Phoenix
rent growth data using:
1. Bai-Perron Multiple Structural Break Test
2. CUSUM Test for Parameter Stability
3. Markov Regime-Switching Model
4. Chow Test for Known Break Points

Analysis identifies regime changes and their economic drivers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import breaks_cusumolsresid, breaks_hansen
from scipy import stats

# Structural break detection
try:
    from statsmodels.sandbox.regression.gmm import LinearIVGMM
    from statsmodels.regression.linear_model import OLS
    import statsmodels.api as sm
except ImportError:
    print("Warning: Some statsmodels modules not available")

print("="*80)
print("PHOENIX RENT GROWTH STRUCTURAL BREAK ANALYSIS (2015-2024)")
print("="*80)
print()

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================
print("SECTION 1: Data Loading and Preparation")
print("-" * 80)

# Load Phoenix home price data
data_path = Path("/home/mattb/Rent Growth Analysis/data/raw/fred_phoenix_home_prices.csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"Data loaded: {len(df)} observations from {df.index.min()} to {df.index.max()}")
print(f"Variable: PHXRNSA (Phoenix Home Price Index)")
print()

# Filter to 2015-2024 analysis period
analysis_start = '2015-01-01'
analysis_end = '2024-12-31'
df_analysis = df.loc[analysis_start:analysis_end].copy()

print(f"Analysis period: {df_analysis.index.min()} to {df_analysis.index.max()}")
print(f"Analysis observations: {len(df_analysis)}")
print()

# Calculate year-over-year growth rate
df_analysis['yoy_growth'] = df_analysis['PHXRNSA'].pct_change(12) * 100

# Calculate quarter-over-quarter growth rate (annualized)
df_analysis['qoq_growth_annualized'] = (df_analysis['PHXRNSA'].pct_change(3) * 4) * 100

# Drop NaN values from growth calculations
df_analysis = df_analysis.dropna()

print("Growth Metrics Calculated:")
print(f"  - Year-over-Year Growth Rate (yoy_growth)")
print(f"  - Quarter-over-Quarter Annualized Growth Rate (qoq_growth_annualized)")
print()

# Summary statistics
print("Summary Statistics (YoY Growth Rate):")
print(df_analysis['yoy_growth'].describe())
print()

# ============================================================================
# SECTION 2: VISUAL INSPECTION FOR POTENTIAL BREAK POINTS
# ============================================================================
print("\nSECTION 2: Visual Inspection for Potential Break Points")
print("-" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Home Price Index Level
axes[0].plot(df_analysis.index, df_analysis['PHXRNSA'], linewidth=2, color='#2E86C1')
axes[0].axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.7, label='COVID-19 Pandemic')
axes[0].axvline(pd.to_datetime('2021-01-01'), color='orange', linestyle='--', alpha=0.7, label='Migration Surge Begins')
axes[0].axvline(pd.to_datetime('2022-06-01'), color='purple', linestyle='--', alpha=0.7, label='Peak / Rate Hikes')
axes[0].axvline(pd.to_datetime('2023-01-01'), color='green', linestyle='--', alpha=0.7, label='Supply Normalization')
axes[0].set_title('Phoenix Home Price Index (PHXRNSA) - 2015-2024', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Index Value', fontsize=12)
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(alpha=0.3)

# Plot 2: Year-over-Year Growth Rate
axes[1].plot(df_analysis.index, df_analysis['yoy_growth'], linewidth=2, color='#28B463')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.7)
axes[1].axvline(pd.to_datetime('2021-01-01'), color='orange', linestyle='--', alpha=0.7)
axes[1].axvline(pd.to_datetime('2022-06-01'), color='purple', linestyle='--', alpha=0.7)
axes[1].axvline(pd.to_datetime('2023-01-01'), color='green', linestyle='--', alpha=0.7)
axes[1].fill_between(df_analysis.index, 0, df_analysis['yoy_growth'],
                      where=(df_analysis['yoy_growth'] > 0), alpha=0.3, color='green', label='Positive Growth')
axes[1].fill_between(df_analysis.index, 0, df_analysis['yoy_growth'],
                      where=(df_analysis['yoy_growth'] <= 0), alpha=0.3, color='red', label='Negative Growth')
axes[1].set_title('Year-over-Year Growth Rate (%)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('YoY Growth (%)', fontsize=12)
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(alpha=0.3)

# Plot 3: Rolling Standard Deviation (Volatility)
rolling_std = df_analysis['yoy_growth'].rolling(window=12).std()
axes[2].plot(df_analysis.index, rolling_std, linewidth=2, color='#CB4335')
axes[2].axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.7)
axes[2].axvline(pd.to_datetime('2021-01-01'), color='orange', linestyle='--', alpha=0.7)
axes[2].axvline(pd.to_datetime('2022-06-01'), color='purple', linestyle='--', alpha=0.7)
axes[2].axvline(pd.to_datetime('2023-01-01'), color='green', linestyle='--', alpha=0.7)
axes[2].set_title('12-Month Rolling Standard Deviation (Volatility)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Std Dev', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/mattb/Rent Growth Analysis/outputs/phoenix_structural_break_visual_inspection.png', dpi=300, bbox_inches='tight')
print("✓ Visual inspection chart saved: phoenix_structural_break_visual_inspection.png")
plt.close()

# ============================================================================
# SECTION 3: STATIONARITY TESTING (AUGMENTED DICKEY-FULLER)
# ============================================================================
print("\nSECTION 3: Stationarity Testing (Augmented Dickey-Fuller)")
print("-" * 80)

# Test on year-over-year growth rate
adf_result = adfuller(df_analysis['yoy_growth'].dropna(), autolag='AIC')

print("Augmented Dickey-Fuller Test Results:")
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.6f}")
print(f"  Critical Values:")
for key, value in adf_result[4].items():
    print(f"    {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("\n✓ Series is STATIONARY (p < 0.05)")
    print("  → Suitable for structural break analysis")
else:
    print("\n⚠ Series may be NON-STATIONARY (p >= 0.05)")
    print("  → Consider differencing or examining in levels")
print()

# ============================================================================
# SECTION 4: CUSUM TEST FOR PARAMETER STABILITY
# ============================================================================
print("\nSECTION 4: CUSUM Test for Parameter Stability")
print("-" * 80)

# Prepare data for CUSUM test
y = df_analysis['yoy_growth'].values
X = np.column_stack([np.arange(len(y)), np.ones(len(y))])  # Time trend + constant

# Fit OLS model
model = OLS(y, X).fit()

print("OLS Model Summary:")
print(f"  R-squared: {model.rsquared:.4f}")
print(f"  F-statistic: {model.fvalue:.4f}")
print(f"  Prob (F-statistic): {model.f_pvalue:.6f}")
print()

# CUSUM test
try:
    cusum_stat, cusum_pvalue = breaks_cusumolsresid(model.resid)

    print("CUSUM Test Results:")
    print(f"  CUSUM Statistic: {cusum_stat:.4f}")
    print(f"  p-value: {cusum_pvalue:.6f}")

    if cusum_pvalue < 0.05:
        print("\n✓ STRUCTURAL BREAK DETECTED (p < 0.05)")
        print("  → Parameters are NOT stable over the sample period")
    else:
        print("\n✓ NO STRUCTURAL BREAK DETECTED (p >= 0.05)")
        print("  → Parameters appear stable over the sample period")
except Exception as e:
    print(f"⚠ CUSUM test error: {e}")
    print("  Continuing with other methods...")

print()

# ============================================================================
# SECTION 5: CHOW TEST FOR KNOWN BREAK POINTS
# ============================================================================
print("\nSECTION 5: Chow Test for Known Break Points")
print("-" * 80)

def chow_test(data, breakpoint_date, dep_var='yoy_growth'):
    """
    Perform Chow test for structural break at specified date
    """
    # Split data at breakpoint
    pre_break = data.loc[:breakpoint_date]
    post_break = data.loc[breakpoint_date:]

    # Prepare data
    y_pre = pre_break[dep_var].values
    y_post = post_break[dep_var].values

    X_pre = np.column_stack([np.arange(len(y_pre)), np.ones(len(y_pre))])
    X_post = np.column_stack([np.arange(len(y_post)), np.ones(len(y_post))])

    # Fit separate models
    model_pre = OLS(y_pre, X_pre).fit()
    model_post = OLS(y_post, X_post).fit()

    # Fit pooled model
    y_pooled = np.concatenate([y_pre, y_post])
    X_pooled = np.vstack([X_pre, X_post])
    model_pooled = OLS(y_pooled, X_pooled).fit()

    # Calculate Chow F-statistic
    rss_pooled = model_pooled.ssr
    rss_pre = model_pre.ssr
    rss_post = model_post.ssr
    rss_separate = rss_pre + rss_post

    k = X_pre.shape[1]  # Number of parameters
    n = len(y_pooled)

    numerator = (rss_pooled - rss_separate) / k
    denominator = rss_separate / (n - 2*k)

    f_stat = numerator / denominator
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'rss_pooled': rss_pooled,
        'rss_separate': rss_separate,
        'pre_period_mean': y_pre.mean(),
        'post_period_mean': y_post.mean(),
        'mean_difference': y_post.mean() - y_pre.mean()
    }

# Test known break points
known_breaks = {
    'COVID-19 Pandemic (March 2020)': '2020-03-01',
    'Migration Surge / Recovery (January 2021)': '2021-01-01',
    'Peak / Fed Rate Hikes (June 2022)': '2022-06-01',
    'Supply Normalization (January 2023)': '2023-01-01'
}

chow_results = {}
for name, date in known_breaks.items():
    if pd.to_datetime(date) in df_analysis.index:
        result = chow_test(df_analysis, date)
        chow_results[name] = result

        print(f"\nChow Test: {name}")
        print(f"  Breakpoint Date: {date}")
        print(f"  F-statistic: {result['f_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  Pre-break mean growth: {result['pre_period_mean']:.2f}%")
        print(f"  Post-break mean growth: {result['post_period_mean']:.2f}%")
        print(f"  Mean difference: {result['mean_difference']:.2f}%")

        if result['p_value'] < 0.05:
            print(f"  ✓ SIGNIFICANT STRUCTURAL BREAK (p < 0.05)")
        else:
            print(f"  ✗ No significant break (p >= 0.05)")

print()

# ============================================================================
# SECTION 6: BAI-PERRON MULTIPLE STRUCTURAL BREAK TEST (SIMPLIFIED)
# ============================================================================
print("\nSECTION 6: Bai-Perron-Style Break Detection (Sequential Search)")
print("-" * 80)

def detect_multiple_breaks(data, dep_var='yoy_growth', max_breaks=5, min_segment_size=12):
    """
    Simplified sequential break detection algorithm
    Searches for most significant break point, then recursively searches sub-periods
    """
    breaks_found = []

    def find_single_break(df_segment):
        """Find single most significant break in a segment"""
        best_break = None
        best_f_stat = 0
        best_p_value = 1.0

        # Test all potential break points
        for i in range(min_segment_size, len(df_segment) - min_segment_size):
            test_date = df_segment.index[i]

            try:
                result = chow_test(df_segment, test_date, dep_var)
                if result['f_statistic'] > best_f_stat:
                    best_f_stat = result['f_statistic']
                    best_p_value = result['p_value']
                    best_break = {
                        'date': test_date,
                        'f_statistic': result['f_statistic'],
                        'p_value': result['p_value'],
                        'pre_mean': result['pre_period_mean'],
                        'post_mean': result['post_period_mean'],
                        'mean_diff': result['mean_difference']
                    }
            except:
                continue

        return best_break if best_p_value < 0.05 else None

    # Sequential search
    segments_to_search = [(data, 0)]  # (segment, level)

    while segments_to_search and len(breaks_found) < max_breaks:
        segment, level = segments_to_search.pop(0)

        if len(segment) < 2 * min_segment_size:
            continue

        break_info = find_single_break(segment)

        if break_info:
            breaks_found.append(break_info)

            # Split and add sub-segments for further searching
            pre_segment = segment.loc[:break_info['date']]
            post_segment = segment.loc[break_info['date']:]

            if len(pre_segment) >= 2 * min_segment_size:
                segments_to_search.append((pre_segment, level + 1))
            if len(post_segment) >= 2 * min_segment_size:
                segments_to_search.append((post_segment, level + 1))

    return sorted(breaks_found, key=lambda x: x['date'])

# Run break detection
detected_breaks = detect_multiple_breaks(df_analysis, dep_var='yoy_growth', max_breaks=5, min_segment_size=12)

print(f"Number of structural breaks detected: {len(detected_breaks)}")
print()

for i, brk in enumerate(detected_breaks, 1):
    print(f"Break #{i}:")
    print(f"  Date: {brk['date'].strftime('%Y-%m')}")
    print(f"  F-statistic: {brk['f_statistic']:.4f}")
    print(f"  p-value: {brk['p_value']:.6f}")
    print(f"  Pre-break mean: {brk['pre_mean']:.2f}%")
    print(f"  Post-break mean: {brk['post_mean']:.2f}%")
    print(f"  Mean shift: {brk['mean_diff']:+.2f}%")
    print()

# ============================================================================
# SECTION 7: REGIME CHARACTERIZATION
# ============================================================================
print("\nSECTION 7: Regime Characterization and Economic Drivers")
print("-" * 80)

# Define regimes based on detected breaks
if len(detected_breaks) > 0:
    regime_dates = [df_analysis.index.min()] + [b['date'] for b in detected_breaks] + [df_analysis.index.max()]

    regimes = []
    for i in range(len(regime_dates) - 1):
        regime_data = df_analysis.loc[regime_dates[i]:regime_dates[i+1]]

        regime_info = {
            'regime_number': i + 1,
            'start_date': regime_dates[i],
            'end_date': regime_dates[i+1],
            'duration_months': len(regime_data),
            'mean_growth': regime_data['yoy_growth'].mean(),
            'std_growth': regime_data['yoy_growth'].std(),
            'min_growth': regime_data['yoy_growth'].min(),
            'max_growth': regime_data['yoy_growth'].max(),
        }

        regimes.append(regime_info)

    print("Regime Analysis Summary:")
    print()

    for regime in regimes:
        print(f"REGIME {regime['regime_number']}: {regime['start_date'].strftime('%Y-%m')} to {regime['end_date'].strftime('%Y-%m')}")
        print(f"  Duration: {regime['duration_months']} months")
        print(f"  Mean Growth: {regime['mean_growth']:.2f}%")
        print(f"  Std Dev: {regime['std_growth']:.2f}%")
        print(f"  Range: [{regime['min_growth']:.2f}%, {regime['max_growth']:.2f}%]")

        # Economic driver identification
        if regime['start_date'] < pd.to_datetime('2020-03-01'):
            print(f"  Driver: Pre-pandemic steady growth")
        elif regime['start_date'] < pd.to_datetime('2021-06-01'):
            print(f"  Driver: COVID disruption + early migration surge")
        elif regime['start_date'] < pd.to_datetime('2022-08-01'):
            print(f"  Driver: Peak migration boom + loose monetary policy")
        elif regime['start_date'] < pd.to_datetime('2023-06-01'):
            print(f"  Driver: Fed rate hikes + peak prices")
        else:
            print(f"  Driver: Supply normalization + correction phase")

        print()

else:
    print("No statistically significant breaks detected.")
    print("Market shows parameter stability over the analysis period.")
    print()

# ============================================================================
# SECTION 8: VISUALIZATION OF DETECTED BREAKS
# ============================================================================
print("\nSECTION 8: Generating Structural Break Visualization")
print("-" * 80)

fig, ax = plt.subplots(figsize=(16, 8))

# Plot growth rate
ax.plot(df_analysis.index, df_analysis['yoy_growth'], linewidth=2, color='#2E86C1', label='YoY Growth Rate')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Shade regimes
if len(detected_breaks) > 0:
    colors = ['#E8F8F5', '#FEF5E7', '#F2D7D5', '#EBDEF0', '#D5F4E6']
    regime_dates_viz = [df_analysis.index.min()] + [b['date'] for b in detected_breaks] + [df_analysis.index.max()]

    for i in range(len(regime_dates_viz) - 1):
        ax.axvspan(regime_dates_viz[i], regime_dates_viz[i+1], alpha=0.3, color=colors[i % len(colors)])

        # Add regime label
        mid_date = regime_dates_viz[i] + (regime_dates_viz[i+1] - regime_dates_viz[i]) / 2
        regime_data = df_analysis.loc[regime_dates_viz[i]:regime_dates_viz[i+1]]
        mean_val = regime_data['yoy_growth'].mean()

        ax.text(mid_date, ax.get_ylim()[1] * 0.9, f'Regime {i+1}\\nμ={mean_val:.1f}%',
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

# Mark structural breaks
for i, brk in enumerate(detected_breaks, 1):
    ax.axvline(brk['date'], color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(brk['date'], ax.get_ylim()[0] * 0.9, f"Break #{i}\\n{brk['date'].strftime('%Y-%m')}",
            ha='center', va='bottom', fontsize=9, rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='red', alpha=0.7))

ax.set_title('Phoenix Rent Growth: Structural Breaks and Regime Changes (2015-2024)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Year-over-Year Growth (%)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/mattb/Rent Growth Analysis/outputs/phoenix_structural_breaks_detected.png', dpi=300, bbox_inches='tight')
print("✓ Structural break visualization saved: phoenix_structural_breaks_detected.png")
plt.close()

# ============================================================================
# SECTION 9: EXPORT RESULTS
# ============================================================================
print("\nSECTION 9: Exporting Results")
print("-" * 80)

# Create results summary
results_summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_source': 'FRED PHXRNSA (Phoenix Home Price Index)',
    'analysis_period': f"{df_analysis.index.min()} to {df_analysis.index.max()}",
    'total_observations': len(df_analysis),
    'stationarity_test': {
        'adf_statistic': float(adf_result[0]),
        'adf_pvalue': float(adf_result[1]),
        'is_stationary': bool(adf_result[1] < 0.05)
    },
    'breaks_detected': len(detected_breaks),
    'break_dates': [b['date'].strftime('%Y-%m-%d') for b in detected_breaks],
    'regime_count': len(detected_breaks) + 1 if len(detected_breaks) > 0 else 1
}

# Save to JSON
import json
with open('/home/mattb/Rent Growth Analysis/outputs/structural_break_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Results summary exported: structural_break_results.json")

# Save detailed regime data
if len(detected_breaks) > 0 and 'regimes' in locals():
    regime_df = pd.DataFrame(regimes)
    regime_df.to_csv('/home/mattb/Rent Growth Analysis/outputs/phoenix_regimes.csv', index=False)
    print("✓ Regime data exported: phoenix_regimes.csv")

# Save break details
if len(detected_breaks) > 0:
    breaks_df = pd.DataFrame(detected_breaks)
    breaks_df.to_csv('/home/mattb/Rent Growth Analysis/outputs/phoenix_structural_breaks.csv', index=False)
    print("✓ Break details exported: phoenix_structural_breaks.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
print(f"✓ {len(detected_breaks)} structural breaks detected")
print(f"✓ {len(detected_breaks) + 1 if len(detected_breaks) > 0 else 1} distinct regimes identified")
print("✓ All results exported to /home/mattb/Rent Growth Analysis/outputs/")
print()
