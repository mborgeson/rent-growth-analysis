# Model Registry

**Purpose**: Centralized tracking of all ensemble model experiments and configurations

**Last Updated**: 2025-11-07

---

## Active Production Models

### Ensemble Model (Current)

| Component | Model Type | Weight | RMSE | R² | Status |
|-----------|-----------|--------|------|-----|--------|
| VAR | Vector Autoregression | 30% | - | - | ✅ Production |
| GBM | LightGBM | 45% | 0.0234 | 0.89 | ✅ Production |
| SARIMA | Seasonal ARIMA | 25% | 0.0312 | 0.75 | ✅ Production |
| Meta-Learner | Ridge Regression | - | 0.0198 | 0.92 | ✅ Production |

**Overall Performance**:
- Test RMSE: 0.0198
- Test MAE: 0.0165
- Test R²: 0.92
- Directional Accuracy: 83.3%
- Improvement over Naive: 45.2%

---

## Experimental Models

### ARIMAX Variants

#### ARIMAX-EMP-001
- **Date**: 2025-11-07
- **Description**: ARIMAX with employment growth, construction lag 8Q, vacancy rate
- **ARIMA Order**: (2, 1, 3)
- **Exogenous Variables**: 3
  - phx_employment_yoy_growth (elasticity ~1.0)
  - units_under_construction_lag8
  - vacancy_rate
- **Status**: 🔄 Testing
- **Expected Performance**: RMSE ~0.025, R² ~0.85
- **Notes**: Based on deep dive analysis showing employment elasticity of 0.99

#### ARIMAX-FULL-002
- **Date**: TBD
- **Description**: ARIMAX with full feature set including macro variables
- **Status**: 📋 Planned

### XGBoost Variants

#### XGB-OPT-001
- **Date**: 2025-11-07
- **Description**: XGBoost with Bayesian hyperparameter optimization (Optuna)
- **Features**: 28 variables (employment, supply, macro, interactions)
- **Optimization**: 50 trials, 3-fold time series CV
- **Status**: 🔄 Ready to run
- **Expected Performance**: RMSE ~0.020, R² ~0.90
- **Notes**: Should outperform LightGBM due to better handling of interactions

#### XGB-SIMPLE-002
- **Date**: TBD
- **Description**: XGBoost with simplified feature set (top 10 features)
- **Status**: 📋 Planned

### Random Forest Variants

#### RF-OPT-001
- **Date**: TBD
- **Description**: Random Forest with hyperparameter tuning
- **Status**: 📋 Planned
- **Expected Features**: Same as XGB-OPT-001

### Elastic Net Variants

#### ENET-L1-001
- **Date**: TBD
- **Description**: Elastic Net with emphasis on L1 regularization (feature selection)
- **Status**: 📋 Planned
- **Alpha ratio**: 0.8 L1, 0.2 L2

#### ENET-L2-002
- **Date**: TBD
- **Description**: Elastic Net with emphasis on L2 regularization (ridge-like)
- **Status**: 📋 Planned
- **Alpha ratio**: 0.2 L1, 0.8 L2

### Prophet Variants

#### PROPHET-001
- **Date**: TBD
- **Description**: Facebook Prophet with employment and supply regressors
- **Status**: 📋 Planned
- **Seasonality**: Quarterly (4)
- **Regressors**: employment_yoy_growth, construction_lag8, vacancy_rate

### VECM Variants

#### VECM-COINT-001
- **Date**: TBD
- **Description**: Vector Error Correction Model for cointegrated relationships
- **Status**: 📋 Planned
- **Variables**: Rent growth, employment, construction, home prices
- **Cointegration Rank**: To be determined via Johansen test
- **Notes**: Based on deep dive finding of Granger causality at lag 6

---

## Deep Dive Analysis Results

### Supply Dynamics Analysis (2025-11-07)

**Key Findings**:
1. **Construction Pipeline**:
   - Optimal lag: 8 quarters (correlation: -0.246)
   - Granger causality: Lag 1 quarter (p=0.021)
   - Interpretation: Construction impacts rent growth with 1-2 year lag

2. **Absorption Dynamics**:
   - Positive correlation with rent growth (0.119, not significant)
   - High absorption quarters: +1.33pp higher rent growth
   - Suggests supply/demand balance matters

3. **Regime Changes**:
   - Detected in 2015 Q1-Q3
   - Rolling correlations show instability
   - Recommends regime-switching models

### Employment Impact Analysis (2025-11-07)

**Key Findings**:
1. **Employment Growth**:
   - YoY employment growth: Correlation 0.514 (highly significant)
   - Estimated elasticity: 0.99 (1% emp → 1% rent)
   - Professional/Business sector: Correlation 0.494, Elasticity 0.75

2. **Granger Causality**:
   - Strongest at lag 6 quarters (p<0.0001)
   - Also significant at lag 1 and 5 quarters
   - Suggests employment leads rent by 1-6 quarters

3. **Unemployment**:
   - National unemployment: Weak correlation (-0.228, p=0.072)
   - Local unemployment data limited

**Model Recommendations**:
- Include employment YoY growth in all models
- Use 1-quarter lag for immediate effects
- Consider 6-quarter lag for longer-term effects
- Weight employment variables heavily in feature importance

---

## Performance Tracking

### Model Comparison Matrix

| Model ID | Type | RMSE (Test) | MAE (Test) | R² (Test) | Dir. Acc. | Training Date | Status |
|----------|------|-------------|------------|-----------|-----------|---------------|--------|
| ENSEMBLE-CURRENT | Ridge Meta-learner | 0.0198 | 0.0165 | 0.92 | 83.3% | 2025-11-06 | ✅ Production |
| GBM-PHX-001 | LightGBM | 0.0234 | 0.0189 | 0.89 | 75.0% | 2025-11-06 | ✅ Component |
| SARIMA-001 | SARIMA(2,1,3)x(1,1,1,4) | 0.0312 | 0.0245 | 0.75 | 66.7% | 2025-11-06 | ✅ Component |
| VAR-MACRO-001 | VAR(2) | - | - | - | - | 2025-11-06 | ✅ Component |
| ARIMAX-EMP-001 | ARIMAX(2,1,3) | TBD | TBD | TBD | TBD | 2025-11-07 | 🔄 Testing |
| XGB-OPT-001 | XGBoost | TBD | TBD | TBD | TBD | 2025-11-07 | 📋 Ready |

### Benchmark: Naive Persistence Model
- RMSE: 0.0363
- All models should beat this baseline

---

## Ensemble Weight Optimization

### Current Weights (Ridge Meta-Learner)
- GBM: 45% (target)
- SARIMA: 25% (target)
- VAR: 30% (indirect via GBM features)

### Proposed Weight Updates

Based on new model experiments:

| Configuration | GBM | SARIMA | ARIMAX | XGB | Notes |
|--------------|-----|--------|--------|-----|-------|
| Current | 45% | 25% | - | - | VAR indirect via GBM |
| Option A | 30% | 20% | 25% | 25% | Add ARIMAX + XGB |
| Option B | 25% | 20% | - | 35% | Replace GBM with XGB if superior |
| Option C | 20% | 15% | 20% | 30% | Balanced 4-model ensemble + 15% VECM |

**Recommendation**: Test Option A first, then compare with Option B if XGB outperforms GBM significantly.

---

## Next Experiments

### Priority Queue

1. **✅ COMPLETED**: Deep dive analysis (supply + employment)
2. **🔄 IN PROGRESS**: ARIMAX-EMP-001
3. **📋 NEXT**: XGB-OPT-001 (run hyperparameter optimization)
4. **📋 QUEUE**: VECM-COINT-001 (cointegration analysis)
5. **📋 QUEUE**: PROPHET-001 (alternative time series approach)
6. **📋 QUEUE**: RF-OPT-001 (ensemble diversity)
7. **📋 QUEUE**: ENET-L1-001 (feature selection insights)

### Research Questions

1. Does XGBoost significantly outperform LightGBM? (Expected: Yes, by 0.002-0.005 RMSE)
2. Can ARIMAX match GBM performance with fewer features? (Expected: Close, within 0.005 RMSE)
3. Is there value in VECM for cointegrated relationships? (Expected: Moderate, helpful for long-term forecasts)
4. Does Prophet handle seasonality better than SARIMA? (Expected: Similar performance)
5. What are the truly important features? (ENET-L1 will answer this)

---

## Validation Framework

### Performance Thresholds

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| RMSE | <0.030 | <0.025 | <0.020 |
| MAE | <0.025 | <0.020 | <0.015 |
| R² | >0.70 | >0.80 | >0.90 |
| Directional Accuracy | >60% | >70% | >80% |

### Model Acceptance Criteria

To be included in ensemble:
1. Test RMSE < 0.030 (beats naive by >17%)
2. R² > 0.70
3. Adds diversity (low correlation with existing components)
4. Interpretable or provides unique insights

---

## Documentation Standards

Each model experiment must include:
1. **Experiment ID**: Unique identifier
2. **Date**: Training date
3. **Configuration**: Full hyperparameters
4. **Performance**: All metrics on train/test
5. **Metadata**: JSON file with all details
6. **Artifacts**: Saved model, scaler, predictions
7. **Notes**: Insights, lessons learned

---

*Registry maintained by: Automated model tracking system*
*Contact: Model development team*
