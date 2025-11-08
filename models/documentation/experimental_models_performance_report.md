# Experimental Models Performance Report

**Date**: 2025-11-07
**Purpose**: Comprehensive comparison of experimental model variants against production ensemble

---

## Executive Summary

Three experimental model variants were developed based on deep dive analysis findings:
1. **ARIMAX-EMP-001**: ARIMA with employment, construction, and vacancy exogenous variables
2. **XGB-OPT-001**: XGBoost with comprehensive feature set (25 features)
3. **VECM-COINT-001**: Vector Error Correction Model for cointegration relationships

**Key Finding**: All experimental models significantly underperformed the production ensemble and naive baseline, indicating fundamental issues requiring investigation.

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

**Training Period**: 2010 Q1 - 2022 Q4
**Test Period**: 2023 Q1 - 2025 Q3

---

## Naive Persistence Baseline

The naive persistence model simply uses the previous quarter's value as the forecast.

**Naive Performance**:
- **RMSE**: 0.0363 (from ensemble report)
- **Benchmark**: All models should beat this baseline

---

## Experimental Model Results

### Model 1: ARIMAX-EMP-001

**Configuration**:
- Model Type: ARIMAX (ARIMA with Exogenous Variables)
- ARIMA Order: (2, 1, 3)
- Exogenous Variables: 3
  - phx_employment_yoy_growth (elasticity ~1.0 from deep dive)
  - units_under_construction_lag8 (optimal lag from analysis)
  - vacancy_rate

**Exogenous Variable Coefficients**:
| Variable | Coefficient | P-Value | Significance |
|----------|------------|---------|--------------|
| Employment YoY Growth | 0.3430 | <0.0001 | *** |
| Construction Lag 8 | 0.0001 | 0.5836 | - |
| Vacancy Rate | -1.4999 | 0.0012 | ** |

**Performance Metrics**:
- **Test RMSE**: 5.1435 (⚠️ Very Poor)
- **Test MAE**: 4.6463
- **Test R²**: -28.2081 (⚠️ Negative - Worse than mean)
- **Directional Accuracy**: 18.2% (⚠️ Much worse than random)
- **vs. Naive Baseline**: -772.8% (Significantly Worse)

**Model Diagnostics**:
- AIC: 148.35
- BIC: 165.00
- Log Likelihood: -65.17
- Convergence Warnings: Yes (some parameter combinations failed)

**Future Forecasts (2025 Q4 - 2027 Q3)**:
- Predicts consistent negative rent growth: -2.59% to -3.31%
- Pattern: Slight improvement over time from -3.31% to -2.59%

**Status**: ❌ **Failed** - Significantly worse than baseline

**Root Cause Analysis**:
1. Employment coefficient (0.34) much lower than expected elasticity (0.99 from deep dive)
2. Construction lag 8 not statistically significant despite deep dive findings
3. Negative R² indicates fundamental model mis-specification
4. May be overfitting to noise or suffering from multicollinearity

---

### Model 2: XGB-OPT-001

**Configuration**:
- Model Type: XGBoost Regressor
- Hyperparameter Optimization: Default (Optuna unavailable)
- Features: 25 variables (employment, supply, macro, interactions)
- Scaling: StandardScaler applied

**Top 5 Features by Importance**:
1. phx_employment_yoy_growth: 41.03% ✅ (Aligns with deep dive)
2. phx_hpi_yoy_growth: 19.86%
3. vacancy_rate: 6.96%
4. absorption_12mo: 5.40%
5. phx_manufacturing_employment: 4.11%

**Hyperparameters (Default)**:
- max_depth: 6
- learning_rate: 0.05
- n_estimators: 500
- min_child_weight: 3
- subsample: 0.8
- colsample_bytree: 0.8

**Performance Metrics**:
- **Test RMSE**: 3.9511 (⚠️ Very Poor)
- **Test MAE**: 3.6530
- **Test R²**: -16.2354 (⚠️ Negative - Worse than mean)
- **Directional Accuracy**: 36.4% (⚠️ Worse than random)

**Status**: ❌ **Failed** - Significantly worse than baseline

**Root Cause Analysis**:
1. No hyperparameter optimization (Optuna not available)
2. Possible overfitting to training data
3. Feature scaling may not be appropriate for tree-based model
4. May need different feature engineering approach
5. Default hyperparameters likely suboptimal

**Recommendation**:
- Install Optuna for Bayesian optimization
- Consider removing StandardScaler (not needed for tree models)
- Reduce model complexity (fewer estimators, shallower trees)
- Investigate feature interactions

---

### Model 3: VECM-COINT-001

**Configuration**:
- Model Type: Vector Error Correction Model
- Lag Order: 8 (AIC-selected)
- Cointegration Rank: 4
- Variables: 4 (rent growth, employment, construction, HPI)
- Deterministic: Constant in cointegration relation

**Stationarity Tests**:
| Variable | ADF Statistic | P-Value | Stationary |
|----------|--------------|---------|------------|
| Rent Growth YoY | -1.478 | 0.544 | ❌ I(1) |
| Employment YoY Growth | -3.956 | 0.002 | ✅ Stationary |
| Construction | -2.365 | 0.152 | ❌ I(1) |
| HPI YoY Growth | -2.508 | 0.114 | ❌ I(1) |

**Johansen Cointegration Test**:
- All 4 cointegration ranks rejected at 5% level
- ⚠️ Warning: All variables cointegrated suggests they may be stationary
- Contradicts ADF test results

**Error Correction Coefficients (Speed of Adjustment)**:
| Variable | Coefficient | Adjustment Speed |
|----------|------------|-----------------|
| Rent Growth | -0.3975 | Moderate |
| Employment Growth | 1.9360 | Fast |
| Construction | 3828.0768 | ⚠️ **Extremely Fast** (Unstable) |
| HPI Growth | -1.7711 | Fast |

**Performance Metrics**:
- **Test RMSE**: 35.7762 (⚠️ Extremely Poor)
- **Test MAE**: 25.2051
- **Directional Accuracy**: 54.5%
- **Log Likelihood**: NaN (⚠️ Numerical Instability)

**Future Forecasts (2025 Q4 - 2027 Q3)**:
- Predicts severe rent decline: -3.46% to -35.51%
- Pattern: Accelerating decline (highly unrealistic)

**Status**: ❌ **Failed** - Extreme numerical instability

**Root Cause Analysis**:
1. Error correction coefficient for construction (3828) indicates severe model instability
2. Log likelihood = NaN confirms numerical problems
3. Contradictory stationarity test results suggest data issues
4. All variables showing cointegration rank suggests misspecification
5. Model may be inappropriate for this data

**Recommendation**:
- Investigate stationarity issues (check for structural breaks)
- Consider lower cointegration rank
- Try different lag order
- Verify data quality and transformations

---

## Comparative Performance Table

| Model | Type | RMSE | MAE | R² | Dir. Acc. | Status |
|-------|------|------|-----|----|-----------| -------|
| **Production Ensemble** | Ridge Meta-Learner | **0.0198** | **0.0165** | **0.92** | **83.3%** | ✅ Excellent |
| Naive Persistence | Baseline | 0.0363 | - | - | - | Benchmark |
| ARIMAX-EMP-001 | ARIMAX(2,1,3) | 5.1435 | 4.6463 | -28.21 | 18.2% | ❌ Failed |
| XGB-OPT-001 | XGBoost | 3.9511 | 3.6530 | -16.23 | 36.4% | ❌ Failed |
| VECM-COINT-001 | VECM | 35.7762 | 25.2051 | - | 54.5% | ❌ Failed |
| GBM-PHX-001 (Component) | LightGBM | 0.0234 | 0.0189 | 0.89 | 75.0% | ✅ Good |
| SARIMA-001 (Component) | SARIMA | 0.0312 | 0.0245 | 0.75 | 66.7% | ✅ Good |

**Performance Gap**:
- Production ensemble outperforms experimental models by 100-1700%
- All experimental models worse than naive persistence baseline
- Experimental models show negative R² (worse than predicting the mean)

---

## Deep Dive Integration Analysis

### Employment Impact Integration

**Deep Dive Finding**: Employment elasticity of 0.99 (1% employment growth → 0.99% rent growth)

**Model Integration**:

| Model | Employment Feature | Coefficient/Importance | Alignment |
|-------|-------------------|----------------------|-----------|
| ARIMAX-EMP-001 | phx_employment_yoy_growth | 0.3430 (p<0.0001) | ⚠️ **Partial** - Significant but lower than expected |
| XGB-OPT-001 | phx_employment_yoy_growth | 41.03% importance | ✅ **Good** - Top feature |
| VECM-COINT-001 | phx_employment_yoy_growth | EC coef: 1.9360 | ⚠️ **Unclear** - Unstable model |

**Analysis**: While all models identified employment as important, only XGBoost assigned appropriate weight. ARIMAX coefficient much lower than expected elasticity.

### Construction Pipeline Integration

**Deep Dive Finding**: 8-quarter lag optimal (correlation -0.246, Granger causality at lag 1)

**Model Integration**:

| Model | Construction Feature | Coefficient/Importance | Alignment |
|-------|---------------------|----------------------|-----------|
| ARIMAX-EMP-001 | units_under_construction_lag8 | 0.0001 (p=0.584) | ❌ **Failed** - Not significant |
| XGB-OPT-001 | units_under_construction_lag5/6/7/8 | 2.22% (lag 5 highest) | ⚠️ **Weak** - Low importance |
| VECM-COINT-001 | units_under_construction | EC coef: 3828 | ❌ **Failed** - Unstable |

**Analysis**: Construction lag 8 not effectively captured by any experimental model. May need different transformation or feature engineering.

### Vacancy Rate Integration

**Deep Dive Finding**: Strong negative correlation with rent growth

**Model Integration**:

| Model | Vacancy Feature | Coefficient/Importance | Alignment |
|-------|----------------|----------------------|-----------|
| ARIMAX-EMP-001 | vacancy_rate | -1.50 (p=0.0012) | ✅ **Good** - Significant negative |
| XGB-OPT-001 | vacancy_rate | 6.96% importance | ✅ **Good** - Top 3 feature |
| VECM-COINT-001 | Not directly included | - | - |

**Analysis**: Vacancy rate effectively integrated where included, confirming deep dive findings.

---

## Issues Identified

### Critical Issues

1. **Negative R² Across All Experimental Models**
   - Indicates models worse than simply predicting the mean
   - Suggests fundamental mis-specification or data issues
   - May indicate overfitting or inappropriate model selection

2. **VECM Numerical Instability**
   - Log likelihood = NaN
   - Extreme error correction coefficients
   - Unrealistic future forecasts

3. **Hyperparameter Optimization Unavailable**
   - Optuna not installed
   - XGBoost using suboptimal default parameters
   - Significant performance impact

### Data Issues

1. **Stationarity Contradictions**
   - VECM: ADF tests vs. Johansen test show conflicts
   - Suggests potential structural breaks or data quality issues

2. **Small Sample Size**
   - 48-52 training quarters
   - May be insufficient for complex models
   - Time series may have regime changes not captured

3. **Feature Engineering**
   - Employment elasticity not properly captured
   - Construction lag 8 not significant despite deep dive findings
   - May need different transformations or interaction terms

### Model Architecture Issues

1. **XGBoost Feature Scaling**
   - StandardScaler applied to tree-based model (unnecessary)
   - May be hurting performance
   - Tree models don't require feature scaling

2. **VECM Model Specification**
   - Cointegration rank = 4 (all variables) suggests misspecification
   - Lag order 8 may be too high for small sample
   - Deterministic trend specification may be inappropriate

3. **ARIMAX Parameter Search**
   - Limited grid search (only 32 combinations)
   - May have missed optimal configuration
   - Seasonal component not included

---

## Recommendations

### Immediate Actions (High Priority)

1. **Install Optuna for Hyperparameter Optimization**
   ```bash
   pip install optuna
   ```
   - Re-run XGBoost with Bayesian optimization
   - Expected improvement: 0.5-1.5 RMSE reduction

2. **Fix XGBoost Feature Scaling**
   - Remove StandardScaler
   - Tree-based models don't benefit from scaling
   - Expected improvement: 0.3-0.8 RMSE reduction

3. **Investigate Data Quality**
   - Check for outliers in 2023-2025 period
   - Verify data transformations
   - Examine structural breaks

4. **Reduce VECM Complexity**
   - Try cointegration rank 0-2 instead of 4
   - Reduce lag order to 4-6
   - Test different deterministic specifications

### Short-Term Improvements (Medium Priority)

5. **Expand ARIMAX Search Space**
   - Include seasonal component: SARIMAX
   - Test broader parameter grid
   - Try different lag structures for exogenous variables

6. **Feature Engineering**
   - Create employment-vacancy interaction terms
   - Test log-transformed construction variables
   - Add momentum/change features

7. **Alternative ML Models**
   - LightGBM with hyperparameter tuning
   - Random Forest with grid search
   - Elastic Net for linear relationships

8. **Cross-Validation**
   - Implement rolling window cross-validation
   - Test on multiple time periods
   - Validate against holdout set

### Long-Term Strategy (Lower Priority)

9. **Regime-Switching Models**
   - Deep dive identified regime changes in 2015
   - Markov-switching ARIMA
   - Threshold VAR models

10. **External Data Integration**
    - Migration data (deep dive recommendation)
    - Demographic trends
    - Policy indicators

11. **Ensemble Optimization**
    - If experimental models improve, test ensemble combinations
    - Bayesian Model Averaging
    - Stacking with meta-learner

12. **Advanced Techniques**
    - LSTM/GRU neural networks
    - Prophet with custom seasonality
    - State-space models

---

## Conclusion

**Current Status**: None of the experimental models are ready for production integration.

**Key Takeaway**: The production ensemble (RMSE 0.0198, R² 0.92) significantly outperforms all experimental variants. Before adding complexity, we must:
1. Resolve fundamental data/specification issues causing negative R²
2. Implement proper hyperparameter optimization
3. Validate models on holdout periods

**Next Steps**:
1. Fix identified issues (Optuna, scaling, model specifications)
2. Re-run experiments with corrections
3. Compare revised performance
4. Only integrate models that beat naive baseline

**Timeline Estimate**:
- Issue fixes: 1-2 days
- Re-run and validation: 2-3 days
- Performance comparison: 1 day
- **Total**: 4-6 days to viable experimental models

---

## Appendix: Model Outputs

### File Locations

**ARIMAX-EMP-001**:
- Model: `models/experiments/arimax_variants/ARIMAX-EMP-001_model.pkl`
- Test Predictions: `models/experiments/arimax_variants/ARIMAX-EMP-001_test_predictions.csv`
- Future Forecasts: `models/experiments/arimax_variants/ARIMAX-EMP-001_future_forecasts.csv`
- Metadata: `models/experiments/arimax_variants/ARIMAX-EMP-001_metadata.json`

**XGB-OPT-001**:
- Model: `models/experiments/xgboost_variants/XGB-OPT-001_model.pkl`
- Scaler: `models/experiments/xgboost_variants/XGB-OPT-001_scaler.pkl`
- Feature Importance: `models/experiments/xgboost_variants/XGB-OPT-001_feature_importance.csv`
- Metadata: `models/experiments/xgboost_variants/XGB-OPT-001_metadata.json`

**VECM-COINT-001**:
- Model: `models/experiments/vecm_variants/VECM-COINT-001_model.pkl`
- Test Forecasts: `models/experiments/vecm_variants/VECM-COINT-001_test_forecasts.csv`
- Future Forecasts: `models/experiments/vecm_variants/VECM-COINT-001_future_forecasts.csv`
- Metadata: `models/experiments/vecm_variants/VECM-COINT-001_metadata.json`

---

*Report generated: 2025-11-07*
*Models tested: 3 experimental variants + production ensemble*
*Status: Experimental models require significant improvements before production integration*
