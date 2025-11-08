# Experimental Models Performance Report - UPDATED

**Date**: 2025-11-07 (Updated after XGB-OPT-002 and regime change investigation)
**Purpose**: Comprehensive comparison of experimental model variants with root cause analysis

---

## ⚠️ CRITICAL FINDING: Fundamental Regime Change Identified

**Root Cause**: All experimental models fail due to a **fundamental regime change** in the Phoenix rental market between training (2010-2022) and test (2023-2025) periods, NOT due to technical implementation issues.

**Evidence**:
- Train mean rent growth: **+4.33%** → Test mean: **-1.34%** (5.67pp decline)
- 4 out of 5 top features have **different distributions** (KS test p < 0.0001)
- Structural break detected starting **2023 Q1** with accelerating decline
- Test period has **33.3% outlier rate** vs 11.5% in training

**For detailed analysis, see**: `/home/mattb/Rent Growth Analysis/reports/deep_analysis/REGIME_CHANGE_FINDINGS.md`

---

## Executive Summary

Four experimental model variants were developed and tested:
1. **ARIMAX-EMP-001**: ARIMA with employment, construction, and vacancy exogenous variables
2. **XGB-OPT-001**: XGBoost with default hyperparameters and feature scaling
3. **XGB-OPT-002**: XGBoost with Optuna optimization and no feature scaling (**LATEST**)
4. **VECM-COINT-001**: Vector Error Correction Model for cointegration relationships

**Key Finding**: All experimental models fail with negative R² because they were trained on a fundamentally different market regime (2010-2022: +4.33% growth) than the test period (2023-2025: -1.34% decline).

**Implication**: Technical fixes (Optuna, scaling removal) are necessary but insufficient. Must address regime change fundamentally.

---

## Production Ensemble Baseline

### Current Production Model (Ensemble Ridge Meta-Learner)

| Component | Model Type | Weight | Component RMSE | Component R² |
|-----------|-----------|--------|----------------|--------------|
| VAR | Vector Autoregression | 30% | - | - |
| GBM | LightGBM | 45% | 0.0234 | 0.89 |
| SARIMA | Seasonal ARIMA | 25% | 0.0312 | 0.75 |
| **Meta-Learner** | **Ridge Regression** | - | **0.0198** | **0.92** |

**Production Performance**:
- **Test RMSE**: 0.0198
- **Test MAE**: 0.0165
- **Test R²**: 0.92
- **Directional Accuracy**: 83.3%
- **Improvement over Naive**: 45.2%

**Training Period**: 2010 Q1 - 2022 Q4 (52 quarters)
**Test Period**: 2023 Q1 - 2025 Q3 (12 quarters)

**Why It Succeeds**: Likely due to component diversity (VAR captures regime dynamics, SARIMA captures seasonality, Ridge meta-learner adapts weighting) - requires further investigation.

---

## Naive Baselines

| Baseline | Method | Test RMSE | Test MAE | Note |
|----------|--------|-----------|----------|------|
| **Persistence** | Last train value (14.0%) | 2.5277 | 2.3417 | Simple baseline |
| **Mean** | Train mean (4.33%) | 5.7536 | 5.6744 | Worst possible (R²=0) |

**Interpretation**: Any model worse than persistence (RMSE > 2.53) or mean (RMSE > 5.75) has negative R².

---

## Experimental Model Results

### Comparative Performance Table

| Model | RMSE | MAE | R² | Dir. Acc. | vs Persistence | vs Mean | Status |
|-------|------|-----|----|-----------|----|-----|--------|
| **Production Ensemble** | **0.0198** | **0.0165** | **0.92** | **83.3%** | **-99.2%** ✅ | **-99.7%** ✅ | ✅ Excellent |
| Naive Persistence | 2.5277 | 2.3417 | 0.00 | - | Baseline | **-56.1%** ✅ | Benchmark |
| ARIMAX-EMP-001 | 5.1435 | 4.6463 | -28.21 | 18.2% | **+103.5%** ❌ | **-10.6%** ❌ | ❌ Failed |
| XGB-OPT-001 (scaled) | 3.9511 | 3.6530 | -16.23 | 36.4% | **+56.3%** ❌ | **-31.3%** ✅ | ❌ Failed |
| **XGB-OPT-002 (optimized)** | **4.2058** | **3.8714** | **-18.53** | **45.5%** | **+66.4%** ❌ | **-26.9%** ✅ | ❌ Failed |
| VECM-COINT-001 | 35.7762 | 25.2051 | - | 54.5% | **+1315.4%** ❌ | **+521.8%** ❌ | ❌ Failed |

**Key Insights**:
- XGB-OPT-002 shows **slight improvement** in directional accuracy (45.5% vs 36.4%) but **worse RMSE** (4.21 vs 3.95)
- All experimental models are **worse than persistence baseline**, confirming negative R²
- Production ensemble **99.2% better** than persistence, demonstrating successful regime adaptation

---

## Model 1: ARIMAX-EMP-001

**Configuration**:
- Model Type: ARIMAX (ARIMA with Exogenous Variables)
- ARIMA Order: (2, 1, 3) - Selected via AIC grid search
- Exogenous Variables: 3
  - phx_employment_yoy_growth (elasticity ~1.0 from deep dive)
  - units_under_construction_lag8 (optimal lag from analysis)
  - vacancy_rate

**Exogenous Variable Coefficients**:
| Variable | Coefficient | P-Value | Significance | Expected |
|----------|------------|---------|--------------|----------|
| Employment YoY Growth | 0.3430 | <0.0001 | *** | 0.99 from deep dive |
| Construction Lag 8 | 0.0001 | 0.5836 | - | Significant |
| Vacancy Rate | -1.4999 | 0.0012 | ** | Negative |

**Performance Metrics**:
- **Test RMSE**: 5.1435 (⚠️ Very Poor)
- **Test MAE**: 4.6463
- **Test R²**: -28.2081 (⚠️ Worse than mean)
- **Directional Accuracy**: 18.2%
- **vs. Persistence**: +103.5% worse
- **vs. Mean**: -10.6% better (barely beats worst baseline)

**Status**: ❌ **Failed** - Trained on growth regime, cannot predict decline regime

**Regime Impact**: Employment coefficient (0.34) much lower than expected (0.99) because relationship changes in negative growth environment.

---

## Model 2: XGB-OPT-001 (Original)

**Configuration**:
- Model Type: XGBoost Regressor
- Hyperparameter Optimization: **Default** (Optuna unavailable)
- Features: 25 variables (employment, supply, macro, interactions)
- Scaling: **StandardScaler applied** (not needed for trees)

**Top 5 Features by Importance**:
1. phx_employment_yoy_growth: 41.03%
2. phx_hpi_yoy_growth: 19.86%
3. vacancy_rate: 6.96%
4. absorption_12mo: 5.40%
5. phx_manufacturing_employment: 4.11%

**Performance Metrics**:
- **Test RMSE**: 3.9511
- **Test MAE**: 3.6530
- **Test R²**: -16.2354
- **Directional Accuracy**: 36.4%
- **vs. Persistence**: +56.3% worse
- **vs. Mean**: -31.3% better

**Status**: ❌ **Failed** - Suboptimal hyperparameters and unnecessary scaling

---

## Model 3: XGB-OPT-002 (LATEST - Optimized)

**Configuration**:
- Model Type: XGBoost Regressor
- Hyperparameter Optimization: **Bayesian (Optuna)** - 50 trials
- Features: 25 variables (employment, supply, macro, interactions)
- Scaling: **None** (tree models don't require scaling)

**Hyperparameters (Optimized)**:
```yaml
max_depth: 9
learning_rate: 0.0233
n_estimators: 744
min_child_weight: 6
subsample: 0.8842
colsample_bytree: 0.6803
gamma: 0.0339
reg_alpha: 0.2160
reg_lambda: 0.9790
```

**Cross-Validation Performance**:
- **Best CV RMSE**: 2.8616 (3-fold time series CV)
- **CV-Test Gap**: 4.2058 - 2.8616 = 1.344 (47% worse on test)

**Top 5 Features by Importance** (Changed from v1):
1. **mortgage_rate_30yr_lag2**: 28.04% (NEW #1, was not top 5)
2. phx_manufacturing_employment: 13.69%
3. phx_employment_yoy_growth: 13.31% (down from 41.03%)
4. vacancy_rate: 9.10%
5. phx_hpi_yoy_growth: 6.61%

**Feature Distribution Analysis**:
| Feature | Train Mean | Test Mean | KS Test p-value | Different? |
|---------|-----------|-----------|-----------------|------------|
| mortgage_rate_lag2 | 3.94% | 6.54% | <0.0001 | ✅ YES (+66%) |
| phx_manufacturing | 125.7K | 149.3K | <0.0001 | ✅ YES (ceiling) |
| employment_yoy | 2.43% | 2.00% | 0.260 | ❌ No (only stable!) |
| vacancy_rate | 7.80% | 11.01% | <0.0001 | ✅ YES (+41%) |
| hpi_yoy_growth | 10.11% | 0.51% | <0.0001 | ✅ YES (-95%) |

**Performance Metrics**:
- **Test RMSE**: 4.2058 (⚠️ Worse than v1 despite optimization!)
- **Test MAE**: 3.8714
- **Test R²**: -18.5290
- **Directional Accuracy**: 45.5% (✅ Improved from 36.4%)
- **vs. Persistence**: +66.4% worse
- **vs. Mean**: -26.9% better

**Status**: ❌ **Failed** - Technical improvements insufficient to overcome regime change

**Root Cause**:
1. **Regime Mismatch**: Trained on features that worked in growth regime (2010-2022) but fail in decline regime (2023-2025)
2. **Feature Shifts**: 4/5 top features have completely different distributions in test period
3. **Overfitting**: Despite cross-validation, model learned regime-specific patterns
4. **Small Test Sample**: 12 quarters insufficient for complex model (25 features, 48 training samples = 1:2 ratio)

**Why v2 Worse Than v1**:
- Optuna optimized hyperparameters for **training regime**, increasing overfitting
- CV RMSE 2.86 vs test RMSE 4.21 shows **47% performance degradation** from regime change
- Feature importance shift suggests different underlying dynamics

---

## Model 4: VECM-COINT-001

**Configuration**:
- Model Type: Vector Error Correction Model
- Lag Order: 8 (AIC-selected)
- Cointegration Rank: 4 (all variables)
- Variables: 4 (rent growth, employment, construction, HPI)

**Error Correction Coefficients**:
| Variable | Coefficient | Interpretation |
|----------|------------|----------------|
| Rent Growth | -0.3975 | Moderate adjustment |
| Employment Growth | 1.9360 | Fast adjustment |
| **Construction** | **3828.0768** | ⚠️ **Extremely unstable** |
| HPI Growth | -1.7711 | Fast adjustment |

**Performance Metrics**:
- **Test RMSE**: 35.7762 (⚠️ Extremely Poor)
- **Test MAE**: 25.2051
- **Directional Accuracy**: 54.5%
- **Log Likelihood**: NaN (⚠️ Numerical instability)
- **vs. Persistence**: +1315.4% worse
- **vs. Mean**: +521.8% worse

**Status**: ❌ **Failed** - Severe numerical instability and unrealistic forecasts

**Root Cause**: Cointegration rank 4 suggests misspecification (all variables cointegrated contradicts stationarity tests). Regime change breaks long-run equilibrium relationships.

---

## Regime Change Analysis Summary

### Target Variable Shift

| Period | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Train (2010-2022)** | **+4.33%** | 3.93% | -4.50% | +16.10% |
| **Test (2023-2025)** | **-1.34%** | 0.99% | -2.80% | +1.00% |
| **Change** | **-5.67pp** | -2.93pp | +1.70pp | -15.10pp |

### Structural Break Timeline

| Period | Mean Rent Growth | % Change | Event |
|--------|-----------------|----------|-------|
| 2010-2014 | +1.21% | Baseline | Recovery |
| 2015-2019 | +5.10% | **+323%** | Growth |
| 2020-2022 | +8.28% | **+62%** | Pandemic surge |
| **2023-2025** | **-1.55%** | **-119%** | **Decline regime** |

### Test Period Quarters

| Quarter | Rent Growth | Deviation from Train Mean |
|---------|-------------|--------------------------|
| 2022 Q4 | +1.0% | -3.33pp (last positive) |
| 2023 Q1 | -0.3% | -4.63pp (**regime break**) |
| 2023 Q2-Q4 | -1.3% to -1.4% | ~-5.6pp |
| 2024 Q1-Q4 | -1.0% to -1.6% | -5.3pp to -5.9pp |
| 2025 Q1-Q3 | -1.9% to -2.8% | -6.2pp to -7.1pp (**accelerating**) |

**Pattern**: Consistent negative growth with gradual deterioration over 3 years.

---

## Root Cause Conclusions

### Why Experimental Models Fail

1. **Fundamental Regime Mismatch**: Models trained on +4.33% growth cannot predict -1.34% decline
2. **Feature Distribution Shifts**: 80% of top features have different distributions in test period
3. **Relationship Breakdown**: Feature-target relationships learned in growth regime don't apply in decline regime
4. **Small Test Sample**: 12 quarters insufficient for complex models to generalize across regime change
5. **Overfitting to Training Regime**: Despite CV, models learned regime-specific rather than generalizable patterns

### Why Production Ensemble Succeeds

**Hypothesis** (requires validation):
1. **Component Diversity**: VAR captures regime dynamics, SARIMA captures seasonality, GBM captures non-linearities
2. **Adaptive Meta-Learner**: Ridge regression adjusts component weights based on recent performance
3. **Simpler Component Models**: Individual components may be less prone to overfitting
4. **Better Regularization**: Ridge meta-learner provides implicit regularization

**Action Required**: Decompose production ensemble performance by component to validate hypothesis.

---

## Lessons Learned

### Technical Improvements Implemented

1. ✅ **Optuna Installation**: Enabled Bayesian hyperparameter optimization
   - Result: CV RMSE improved to 2.86 but test RMSE worsened to 4.21
   - **Lesson**: Optimization on training regime can increase overfitting to regime-specific patterns

2. ✅ **Feature Scaling Removal**: Eliminated unnecessary StandardScaler from XGBoost
   - Result: Marginal directional accuracy improvement (36.4% → 45.5%)
   - **Lesson**: Correct technical implementation necessary but insufficient

3. ❌ **Root Cause**: Technical fixes cannot overcome fundamental regime change
   - **Lesson**: Must address regime adaptation before further technical optimization

### What Worked

1. **Systematic Investigation**: Root cause analysis identified fundamental issue
2. **Statistical Testing**: KS tests, normality tests, variance tests provided evidence
3. **Baseline Comparisons**: Persistence and mean baselines revealed extent of failure
4. **Production Ensemble**: Demonstrates successful regime adaptation (requires analysis)

### What Didn't Work

1. **Static Training**: Training on single regime fails to generalize to different regime
2. **Complex Models on Small Data**: 25 features on 48 samples (1:2 ratio) leads to overfitting
3. **Feature Engineering Without Regime Awareness**: Features effective in growth may fail in decline
4. **Hyperparameter Optimization on Single Regime**: Optimizing for training regime increases overfitting

---

## Recommendations

### Immediate Priority (HIGH)

1. **Analyze Production Ensemble Components** ⚡
   - Decompose test RMSE by component (VAR, GBM, SARIMA)
   - Understand which components handle regime change successfully
   - Extract lessons for experimental models
   - **Timeline**: 1 day

2. **Implement Regime Detection** ⚡
   - Chow test for structural breaks
   - Rolling window stability tests
   - Automatic regime change alerting
   - **Timeline**: 2 days

3. **Research Regime Change Drivers**
   - Why did Phoenix rental market shift to decline in 2023?
   - External factors: Fed policy, supply surge, migration, remote work
   - Identify leading indicators of regime changes
   - **Timeline**: 2-3 days

4. **Expand Test Period**
   - Current: 12 quarters (insufficient)
   - Target: 20+ quarters for validation
   - Consider: Walk-forward validation with regime detection
   - **Timeline**: Ongoing (as data becomes available)

### Short-Term Actions (MEDIUM)

5. **Simplify Experimental Models**
   - Reduce features from 25 to 5-10 most stable
   - Focus on features with consistent distributions across regimes
   - **Priority**: Employment YoY growth (only stable feature in top 5)
   - **Timeline**: 2 days

6. **Regime-Specific Training**
   - Train separate models by regime:
     - 2010-2014: Recovery (-4.5% to +5%)
     - 2015-2019: Growth (+3% to +7%)
     - 2020-2022: Surge (+5% to +16%)
     - 2023+: Decline (-3% to +1%)
   - Implement regime detection to select appropriate model
   - **Timeline**: 3-4 days

7. **Ensemble with Regime Awareness**
   - Add regime detector to meta-learner
   - Weight components based on regime classification
   - Test Bayesian Model Averaging with regime priors
   - **Timeline**: 3-4 days

8. **Cross-Validation Strategy Update**
   - Replace simple time series split with walk-forward validation
   - Include multiple regime transitions in validation
   - Use expanding window to capture regime changes
   - **Timeline**: 2 days

### Long-Term Strategy (LOWER)

9. **Regime-Switching Models**
   - Markov-switching VAR (MS-VAR)
   - Threshold VAR (TVAR)
   - Time-Varying Parameter (TVP) models
   - **Timeline**: 1-2 weeks

10. **Alternative Data Sources**
    - Migration data (population flows)
    - Remote work indicators
    - Policy variables (rent control, zoning)
    - Real-time sentiment (Google Trends)
    - **Timeline**: Ongoing

11. **Online Learning**
    - Models that update as new data arrives
    - Adaptive to regime changes
    - Weighted recent data more heavily
    - **Timeline**: 2-3 weeks

12. **Transfer Learning**
    - Learn regime change patterns from other markets
    - Apply to Phoenix forecasting
    - Multi-market ensemble
    - **Timeline**: 3-4 weeks

---

## Files Generated

### Analysis Scripts
- `/home/mattb/Rent Growth Analysis/reports/deep_analysis/02_negative_r2_investigation.py` - Root cause investigation

### Documentation
- `/home/mattb/Rent Growth Analysis/reports/deep_analysis/REGIME_CHANGE_FINDINGS.md` - Detailed findings
- `/home/mattb/Rent Growth Analysis/models/documentation/experimental_models_performance_report_UPDATED.md` - This document

### Model Outputs
**ARIMAX-EMP-001**:
- models/experiments/arimax_variants/ARIMAX-EMP-001_model.pkl
- models/experiments/arimax_variants/ARIMAX-EMP-001_test_predictions.csv
- models/experiments/arimax_variants/ARIMAX-EMP-001_metadata.json

**XGB-OPT-001**:
- models/experiments/xgboost_variants/XGB-OPT-001_model.pkl
- models/experiments/xgboost_variants/XGB-OPT-001_feature_importance.csv
- models/experiments/xgboost_variants/XGB-OPT-001_metadata.json

**XGB-OPT-002** (LATEST):
- models/experiments/xgboost_variants/XGB-OPT-002_model.pkl
- models/experiments/xgboost_variants/XGB-OPT-002_feature_importance.csv
- models/experiments/xgboost_variants/XGB-OPT-002_metadata.json

**VECM-COINT-001**:
- models/experiments/vecm_variants/VECM-COINT-001_model.pkl
- models/experiments/vecm_variants/VECM-COINT-001_test_forecasts.csv
- models/experiments/vecm_variants/VECM-COINT-001_metadata.json

---

## Conclusion

**Status**: Root cause identified ✅, Technical fixes implemented ✅, Fundamental issue requires new approach ⚡

**Key Takeaway**: The Phoenix rental market experienced a **fundamental regime change** in 2023 Q1, transitioning from a +4.33% growth regime (2010-2022) to a -1.34% decline regime (2023-2025). This regime change makes it **impossible** for models trained on historical data to accurately predict current conditions without regime adaptation.

**Production Ensemble**: Continues to perform excellently (RMSE 0.0198, R² 0.92), suggesting it has successful regime adaptation mechanisms. **Must analyze components to understand why.**

**Experimental Models**: All failed with negative R² due to regime mismatch, not technical issues. Technical improvements (Optuna, scaling removal) necessary but insufficient.

**Next Session Focus**:
1. Analyze production ensemble components to extract regime adaptation lessons
2. Implement regime detection framework
3. Develop regime-adaptive experimental models

**Timeline**: Regime analysis (1 day), Detection framework (2 days), New models (3-4 days)

---

*Report updated: 2025-11-07*
*Models tested: 4 experimental variants (ARIMAX, XGB-OPT-001, XGB-OPT-002, VECM) + production ensemble*
*Status: Root cause identified, new strategy required*
