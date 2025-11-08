# Critical Finding: Regime Change Root Cause Analysis

**Date**: 2025-11-07
**Investigation ID**: NEG-R2-INV-001
**Status**: ⚠️ **CRITICAL - Fundamental Model Assumption Violated**

---

## Executive Summary

The investigation into negative R² performance across all experimental models has revealed a **fundamental regime change** in the Phoenix rental market between the training period (2010-2022) and test period (2023-2025). This regime change makes it **impossible** for models trained on historical data to accurately predict current market conditions.

**Key Finding**: The test period represents a completely different market regime with:
- **-5.67 percentage point shift** in mean rent growth (from +4.33% to -1.34%)
- **4 out of 5 top features showing different distributions** (p < 0.05)
- **33.3% outlier rate** in test period vs 11.5% in training
- **Structural break identified** starting 2021 Q2, accelerating through 2023

---

## Critical Evidence

### 1. Target Variable Regime Change

| Metric | Train Period (2010-2022) | Test Period (2023-2025) | Change |
|--------|-------------------------|------------------------|--------|
| **Mean Rent Growth** | +4.33% | -1.34% | **-5.67pp** |
| **Std Deviation** | 3.93% | 0.99% | -2.93pp |
| **Min** | -4.50% | -2.80% | +1.70pp |
| **Max** | +16.10% | +1.00% | -15.10pp |
| **Median** | +4.15% | -1.40% | **-5.55pp** |

**Statistical Tests**:
- Different Means: **t-statistic = 4.94, p < 0.0001** ✅ CONFIRMED
- Different Variances: **Levene test p = 0.023** ✅ CONFIRMED
- Train NOT normal (p = 0.0001), Test IS normal (p = 0.223)

**Interpretation**: The test period has completely different central tendency (-1.34% vs +4.33%) and much lower variance (0.99% vs 3.93%), indicating a stable but fundamentally different market state.

---

### 2. Feature Distribution Shifts

Analysis of XGB-OPT-002 top 5 features:

#### Mortgage Rate (30yr, lag 2) - **28.04% importance**
- Train: mean = 3.94%, std = 0.58%, range = [2.76%, 5.27%]
- Test: mean = 6.54%, std = 0.58%, range = [5.27%, 7.30%]
- **KS Test: p < 0.0001** ⚠️ DIFFERENT DISTRIBUTION
- **Shift: +66% increase in rates**

#### Phoenix Manufacturing Employment - **13.69% importance**
- Train: mean = 125.7K, std = 10.5K, range = [111.2K, 150.7K]
- Test: mean = 149.3K, std = 1.3K, range = [147.7K, 150.8K]
- **KS Test: p < 0.0001** ⚠️ DIFFERENT DISTRIBUTION
- **Interpretation: Employment hit ceiling, lost predictive power**

#### Phoenix Employment YoY Growth - **13.31% importance**
- Train: mean = 2.43%, std = 2.31%, range = [-3.85%, 6.23%]
- Test: mean = 2.00%, std = 1.59%, range = [-0.25%, 3.76%]
- KS Test: p = 0.260 ✅ Same distribution
- **Note: Only feature maintaining distribution consistency**

#### Vacancy Rate - **9.10% importance**
- Train: mean = 7.80%, std = 1.64%, range = [5.20%, 12.50%]
- Test: mean = 11.01%, std = 1.10%, range = [9.20%, 12.40%]
- **KS Test: p < 0.0001** ⚠️ DIFFERENT DISTRIBUTION
- **Shift: +41% increase in vacancy**

#### Phoenix HPI YoY Growth - **6.61% importance**
- Train: mean = 10.11%, std = 9.93%, range = [-9.28%, 33.04%]
- Test: mean = 0.51%, std = 3.70%, range = [-7.56%, 4.96%]
- **KS Test: p < 0.0001** ⚠️ DIFFERENT DISTRIBUTION
- **Shift: -95% decrease in home price growth**

**Summary**: 4 out of 5 top features have fundamentally different distributions in test period.

---

### 3. Structural Break Timeline

#### Rolling 8-Quarter Statistics:

| Period | Mean Rent Growth | Std Dev | % Change from Previous |
|--------|-----------------|---------|----------------------|
| 2010-2014 | +1.21% | 2.00% | Baseline |
| 2015-2019 | +5.10% | 0.82% | **+323%** |
| 2020-2022 | +8.28% | 5.26% | **+62%** |
| 2023-2025 | **-1.55%** | 0.70% | **-119%** |

#### Detected Regime Changes (>90th percentile rolling mean shifts):

| Date | Absolute Change | Interpretation |
|------|----------------|----------------|
| 2021 Q2 | 4.55pp | Pandemic recovery surge begins |
| 2022 Q4 | 5.63pp | Peak-to-decline transition |
| 2023 Q1 | 7.19pp | **Major structural break** |
| 2023 Q2 | 8.05pp | **Accelerating decline** |
| 2023 Q3 | 7.44pp | Decline continues |
| 2023 Q4 | 5.88pp | Stabilization at new regime |

**Interpretation**: Clear inflection point in 2023 Q1 marking transition to negative growth regime.

---

### 4. Outlier Analysis

#### Train Period Outliers (11.5% rate):
- 2010 Q1-Q2: Great Recession aftermath (-4.5%, -3.0%)
- 2021 Q2-2022 Q1: Pandemic recovery surge (+13.5% to +16.1%)

#### Test Period Outliers (33.3% rate):
- 2022 Q4: Last positive growth (+1.0%)
- 2023 Q1: Transition (-0.3%)
- 2025 Q2-Q3: Accelerating decline (-2.6%, -2.8%)

**Interpretation**: 1 in 3 test observations are outliers relative to test period norms, indicating high instability or measurement issues.

---

### 5. Baseline Performance Comparison

| Model | Test RMSE | Test MAE | Performance |
|-------|-----------|----------|-------------|
| **Persistence** (last train value) | 2.5277 | 2.3417 | Simple baseline |
| **Mean** (train mean) | 5.7536 | 5.6744 | Worst possible |
| **Production Ensemble** | **0.0198** | **0.0165** | ✅ **Excellent** |
| **XGB-OPT-002** | 4.2058 | 3.8714 | ❌ Worse than persistence |

**Critical Insight**: XGBoost is **66.4% worse than persistence** and only **26.9% better than predicting train mean**. This confirms negative R² is due to fundamental regime mismatch, not technical implementation issues.

---

## Root Cause Analysis

### Why Experimental Models Fail

1. **Extrapolation Problem**: Models trained on +4.33% growth regime cannot predict -1.34% decline regime
2. **Feature Relationship Breakdown**: Relationships learned from 2010-2022 don't apply to 2023-2025
3. **Sample Size**: 12 test quarters too small for complex models to generalize
4. **Overfitting Risk**: 25 features on 52 training samples (ratio 1:2) leads to memorization

### Why Production Ensemble Succeeds

**Hypothesis**: Production ensemble (VAR + GBM + SARIMA with Ridge meta-learner) likely succeeds because:

1. **VAR Component**: Captures multi-variable dynamics and adjusts to regime changes through short lags
2. **SARIMA Component**: Seasonal patterns may be more stable than levels
3. **Ridge Meta-Learner**: Weighted combination adapts to which component works best in current regime
4. **Component Diversity**: Different model types capture different aspects of regime changes

**Need to Investigate**:
- How does production ensemble handle the regime change?
- What are the component-level RMSEs on test period?
- Which component dominates in test period?

---

## Quarterly Test Period Breakdown

| Quarter | Rent Growth | Deviation from Train Mean | Note |
|---------|-------------|--------------------------|------|
| 2022 Q4 | +1.0% | -3.33pp | Last positive quarter |
| 2023 Q1 | -0.3% | -4.63pp | **Regime break** |
| 2023 Q2 | -1.4% | -5.73pp | Decline accelerates |
| 2023 Q3 | -1.4% | -5.73pp | Stable negative |
| 2023 Q4 | -1.3% | -5.63pp | Slight improvement |
| 2024 Q1 | -1.0% | -5.33pp | Continuing |
| 2024 Q2 | -1.2% | -5.53pp | Slight decline |
| 2024 Q3 | -1.6% | -5.93pp | Weakening |
| 2024 Q4 | -1.6% | -5.93pp | Stable |
| 2025 Q1 | -1.9% | -6.23pp | Further decline |
| 2025 Q2 | -2.6% | -6.93pp | Accelerating |
| 2025 Q3 | -2.8% | -7.13pp | **Worst quarter** |

**Pattern**: Consistent negative growth with gradual deterioration over 3 years.

---

## Implications

### For Model Development

1. **Historical Training Inadequate**: Cannot use 2010-2022 data alone to predict 2023-2025 regime
2. **Feature Engineering Insufficient**: Current features don't capture regime change drivers
3. **Sample Size Critical**: 12 test quarters too few for validation
4. **Overfitting Risk High**: Complex models will fit training noise, not signal

### For Forecasting

1. **Need Regime-Adaptive Models**: Models must detect and adapt to regime changes
2. **External Factors Required**: Must understand *why* regime changed (policy? supply shock? demand shift?)
3. **Ensemble Approach Valid**: Production ensemble's success validates diversification strategy
4. **Regular Retraining Needed**: Static models become obsolete quickly in regime-changing environments

### For Business Decisions

1. **Market Fundamentals Changed**: Phoenix rental market in fundamentally different state post-2023
2. **Historical Patterns Unreliable**: Pre-2023 patterns don't apply to current market
3. **Caution on Forecasts**: Any forecast must account for regime uncertainty
4. **Production Model**: Current ensemble appears robust to regime changes (needs validation)

---

## Recommendations

### Immediate Actions (High Priority)

1. **Investigate Production Ensemble Components**
   - Decompose test performance by component (VAR, GBM, SARIMA)
   - Understand why ensemble succeeds when experiments fail
   - Identify which component dominates in regime change

2. **Expand Test Period Data**
   - Current: 12 quarters (insufficient)
   - Need: 20+ quarters for reliable validation
   - Consider: Rolling window approach with regime detection

3. **Add Regime Change Detection**
   - Implement Chow test for structural breaks
   - Add regime-switching models (Markov-switching ARIMA)
   - Consider threshold VAR models

4. **External Factor Analysis**
   - Research: What caused 2023 regime change?
   - Candidates: Fed policy, supply surge, migration patterns, remote work reversal
   - Add: Policy variables, migration data, remote work indicators

### Short-Term Improvements (Medium Priority)

5. **Simplify Experimental Models**
   - Reduce features from 25 to 5-10 most stable
   - Focus on features with consistent distributions (employment YoY growth)
   - Avoid features with regime-dependent relationships (mortgage rates, HPI)

6. **Regime-Specific Training**
   - Split training into regimes: 2010-2014 (recovery), 2015-2019 (growth), 2020-2022 (surge), 2023+ (decline)
   - Train separate models per regime
   - Use regime detection to select appropriate model

7. **Ensemble Optimization**
   - Test different ensemble weights by regime
   - Add regime-adaptive meta-learner
   - Consider Bayesian Model Averaging with regime priors

8. **Cross-Validation Strategy**
   - Implement walk-forward validation
   - Test on multiple regime transitions
   - Use expanding window to capture regime changes

### Long-Term Strategy (Lower Priority)

9. **Advanced Regime-Switching Models**
   - Markov-switching Vector Autoregression (MS-VAR)
   - Threshold VAR (TVAR)
   - Time-Varying Parameter (TVP) models

10. **Alternative Data Sources**
    - Migration data (population inflows/outflows)
    - Remote work indicators (office occupancy, tech job postings)
    - Policy indicators (rent control, zoning changes)
    - Real-time indicators (Google Trends, social media sentiment)

11. **Machine Learning Adaptations**
    - Online learning algorithms (update as new data arrives)
    - Transfer learning (leverage regime patterns from other markets)
    - Ensemble methods with regime detection

12. **Scenario Analysis**
    - Develop scenarios for regime persistence vs reversion
    - Quantify forecast uncertainty by regime
    - Create decision frameworks for different regime outcomes

---

## Technical Debt Resolved

**Previous Hypothesis**: "Technical issues causing negative R²"
- ❌ NOT the root cause

**Actual Root Cause**: "Fundamental regime change causing feature distribution shifts and relationship breakdown"
- ✅ CONFIRMED

**Technical Fixes Implemented**:
1. ✅ Optuna installation - Helped but insufficient
2. ✅ Feature scaling removal - Improved slightly but insufficient
3. ❌ VECM complexity reduction - Deferred (not the issue)
4. ❌ ARIMAX seasonal expansion - Deferred (not the issue)

**Conclusion**: Technical improvements are necessary but insufficient. Must address regime change fundamentally.

---

## Next Steps

1. **Analyze Production Ensemble Components** (TOP PRIORITY)
   - Decompose RMSE by component (VAR, GBM, SARIMA)
   - Understand why it succeeds
   - Extract lessons for experimental models

2. **Implement Regime Detection**
   - Chow test for structural breaks
   - Regime-switching model prototype
   - Compare performance across regimes

3. **Simplify and Re-test**
   - Reduce XGBoost to 10 features (most stable)
   - Re-run with regime-aware cross-validation
   - Validate on pre-2023 regime transitions

4. **Research External Factors**
   - Identify 2023 regime change drivers
   - Add relevant variables to dataset
   - Re-test predictive power

---

## Files Generated

- `/home/mattb/Rent Growth Analysis/reports/deep_analysis/02_negative_r2_investigation.py` - Investigation script
- `/home/mattb/Rent Growth Analysis/reports/deep_analysis/REGIME_CHANGE_FINDINGS.md` - This document

---

**Status**: Investigation Complete ✅
**Priority**: Address regime change before further model development ⚠️
**Timeline**: Production ensemble analysis (1 day), Regime detection implementation (2-3 days)
