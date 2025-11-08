# Production Feature Engineering Pipeline Analysis

**Analysis Date**: 2025-11-08
**Analyst**: Claude (Continuation Session)
**Context**: Post-ENSEMBLE-EXP-004 investigation into production preprocessing differences

---

## Executive Summary

**Critical Finding**: Production feature engineering pipeline is **IDENTICAL** to experimental models. The preprocessing steps before StandardScaler application consist of simple transformations (lags, growth rates, ratios, interactions) with NO advanced techniques like log transforms, regime normalization, or complex differencing.

**Implication**: Feature engineering differences **DO NOT** explain why production components work (RMSE 0.5046) while experimental components fail (requiring intercept fallback for RMSE 0.5936). The 17.7% performance gap must originate from other sources.

**Key Insight**: The production advantage is NOT in "what features are engineered" but potentially in:
- Data quality/versions used
- Hyperparameter tuning refinements
- Training period differences
- SARIMA configuration details
- Component interaction dynamics

---

## Investigation Objective

After ENSEMBLE-EXP-004 falsified the "StandardScaler causes failure" hypothesis (showing 533.6% degradation when removed), the highest priority recommendation was to analyze production feature engineering to identify preprocessing differences before StandardScaler application.

**Research Question**: Does production use different feature transformations that enable component success?

**Answer**: **NO** - Production uses identical feature engineering to experimental models.

---

## Production Pipeline Architecture

### Data Flow

```
Raw Data Sources (CoStar, FRED, BLS)
  ↓
Quarterly Resampling
  ↓
Lagged Variables (1, 2, 5-8 quarters)
  ↓
Year-over-Year Growth Rates (4-quarter pct_change)
  ↓
Supply/Demand Ratios
  ↓
Interaction Terms (mortgage × employment)
  ↓
Migration Proxy (employment growth × 1000)
  ↓
Forward Fill Missing Values (ffill)
  ↓
StandardScaler (μ=0, σ=1)
  ↓
LightGBM Training
```

### Source Files

1. **`unified_data_loader.py`** (lines 266-324): Feature engineering transformations
2. **`gbm_phoenix_specific.py`** (lines 172-174): StandardScaler application

---

## Feature Engineering Transformations

### 1. Lagged Variables

**Employment Lags** (1 quarter = 3 months):
```python
merged_df['phx_prof_business_employment_lag1'] = merged_df['phx_prof_business_employment'].shift(1)
merged_df['phx_total_employment_lag1'] = merged_df['phx_total_employment'].shift(1)
```

**Supply Lags** (5-8 quarters = 15-24 months for construction delivery):
```python
for lag in [5, 6, 7, 8]:
    merged_df[f'units_under_construction_lag{lag}'] = merged_df['units_under_construction'].shift(lag)
```

**Mortgage Rate Lags** (2 quarters = 6 months):
```python
merged_df['mortgage_rate_30yr_lag2'] = merged_df['mortgage_rate_30yr'].shift(2)
```

**Rationale**: Simple `.shift()` operations to capture temporal dependencies. No differencing, no detrending, no stationarity transformations.

---

### 2. Year-over-Year Growth Rates

```python
# 4-quarter percentage change (annualized)
merged_df['phx_employment_yoy_growth'] = merged_df['phx_total_employment'].pct_change(4) * 100
merged_df['phx_prof_business_yoy_growth'] = merged_df['phx_prof_business_employment'].pct_change(4) * 100
```

**Transformation**: `pct_change(4)` calculates (value_t - value_t-4) / value_t-4, then multiplies by 100 for percentage.

**No Log-Differencing**: Does NOT use `log(x_t) - log(x_t-1)` or other log-space transformations.

---

### 3. Supply/Demand Ratios

```python
# Supply pressure indicator
merged_df['supply_inventory_ratio'] = (merged_df['units_under_construction'] / merged_df['inventory_units']) * 100

# Demand indicator
merged_df['absorption_inventory_ratio'] = (merged_df['absorption_12mo'] / merged_df['inventory_units']) * 100
```

**Transformation**: Simple division and percentage scaling. No normalization beyond ratio calculation.

---

### 4. Interaction Terms

```python
# Mortgage rate × employment growth interaction
merged_df['mortgage_employment_interaction'] = merged_df['mortgage_rate_30yr'] * merged_df['phx_employment_yoy_growth']
```

**Rationale**: Captures non-linear relationship between borrowing costs and job market conditions. Simple multiplication, no polynomial expansions.

---

### 5. Migration Proxy

```python
# Employment growth as migration proxy (highly correlated)
merged_df['migration_proxy'] = merged_df['phx_employment_yoy_growth'] * 1000  # Scale to approximate people
```

**Transformation**: Linear scaling of employment growth. No complex population modeling.

---

### 6. Missing Value Handling

**Method**: Forward Fill
```python
feature_df = feature_df.fillna(method='ffill')  # Forward-fill small gaps
```

**Scope**: Applied to handle minor discontinuities in quarterly data. No interpolation, no imputation models.

---

### 7. StandardScaler Normalization

**Applied After All Feature Engineering**:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Effect**: Transforms each feature to μ=0, σ=1 using training set statistics.

**Critical Detail**: This is the ONLY normalization step. No regime-specific scaling, no min-max normalization, no robust scaling.

---

## What Production Does NOT Use

### Advanced Transformations (Absent)

❌ **Log Transformations**
- No `np.log()`, `np.log1p()`, or log-differencing
- All features remain in original scale (after pct_change and ratios)

❌ **Regime Normalization**
- No regime-specific mean subtraction
- No segmented scaling by market conditions
- StandardScaler applied uniformly across all time periods

❌ **Differencing Beyond pct_change**
- No first-difference transformations: `x_t - x_t-1`
- No seasonal differencing: `x_t - x_t-4`
- Only pct_change for growth rates

❌ **Box-Cox or Power Transformations**
- No variance stabilization transformations
- No automatic λ selection for optimal normality

❌ **Outlier Treatment**
- No winsorization (capping extreme values)
- No outlier removal or flagging
- Forward fill handles missing values without outlier detection

❌ **Polynomial Features**
- Interaction term is simple multiplication, not polynomial expansion
- No quadratic, cubic, or higher-order features

❌ **Custom Scaling Strategies**
- No RobustScaler (median/IQR based)
- No MinMaxScaler
- No QuantileTransformer
- Only StandardScaler

---

## Comparison: Production vs Experimental Preprocessing

### Feature Engineering Steps

| Step | Production | Experimental (EXP-003, EXP-004) | Match? |
|------|------------|----------------------------------|--------|
| Lagged variables | 1, 2, 5-8 quarters | 1, 2, 5-8 quarters | ✅ IDENTICAL |
| Growth rates | 4-quarter pct_change | 4-quarter pct_change | ✅ IDENTICAL |
| Supply/demand ratios | Division × 100 | Division × 100 | ✅ IDENTICAL |
| Interaction terms | Mortgage × employment | Mortgage × employment | ✅ IDENTICAL |
| Migration proxy | Employment growth × 1000 | Employment growth × 1000 | ✅ IDENTICAL |
| Missing values | Forward fill | Forward fill | ✅ IDENTICAL |
| Scaling | StandardScaler | StandardScaler (EXP-003) / None (EXP-004) | ✅ IDENTICAL (EXP-003) |

### Feature Count

- **Production**: 26 features (documented in `gbm_phoenix_specific.py` lines 67-112)
- **EXP-003**: 26 features (identical list)
- **EXP-004**: 26 features (identical list, no scaling)

**Conclusion**: Feature engineering is **100% IDENTICAL** between production and experimental models.

---

## Implications

### 1. Feature Engineering Hypothesis Eliminated

**Original Hypothesis**: Production might use advanced feature transformations (log transforms, regime normalization, complex differencing) that enable component success.

**Evidence**: Production uses only simple transformations (lags, pct_change, ratios, interactions, scaling).

**Verdict**: **FALSIFIED** - Feature engineering differences DO NOT explain the 17.7% gap between EXP-003 (RMSE 0.5936, intercept-only) and production (RMSE 0.5046, working components).

---

### 2. Remaining Hypotheses for Production Advantage

Since preprocessing is identical, the performance gap must originate from:

#### A. Data Quality/Versions
- **Hypothesis**: Production may use different or more recent data sources
- **Test**: Compare data snapshots between production and experimental runs
- **Evidence Needed**: Data file timestamps, source versions, update frequencies

#### B. Hyperparameter Tuning
- **Hypothesis**: Production LightGBM parameters may differ from experiments
- **Test**: Compare production vs experimental hyperparameters line-by-line
- **Current Status**: Both use n_estimators=1000, learning_rate=0.05, max_depth=6, but subtle differences may exist

#### C. Training Period Differences
- **Hypothesis**: Production may train on different time periods or use different train/test splits
- **Test**: Document exact training/test date ranges for both systems
- **Impact**: Different training periods could expose models to different regime characteristics

#### D. SARIMA Configuration
- **Hypothesis**: Production pure SARIMA may use different (p,d,q)(P,D,Q,s) parameters
- **Test**: Compare production SARIMA order to experimental (2,1,2)(0,0,1,4)
- **Potential**: Better SARIMA parameters could improve ensemble even with negative coefficients

#### E. Component Interaction Dynamics
- **Hypothesis**: Production's negative coefficient strategy may extract signal experimental Ridge cannot
- **Test**: Analyze production meta-learner weights and decision logic
- **Theory**: Production may use custom meta-learning beyond simple Ridge regression

---

### 3. StandardScaler Remains Essential

**ENSEMBLE-EXP-004 Finding**: Removing StandardScaler degraded performance by 533.6% (RMSE 3.761 vs 0.5936).

**Production Confirmation**: Production applies StandardScaler after feature engineering, identical to EXP-003.

**Conclusion**: StandardScaler is **NOT** the cause of component failure. It is essential for LightGBM performance with diverse feature scales.

---

## Revised Investigation Priorities

### Priority 1: Data Source Investigation ⚡
**Objective**: Compare production vs experimental data files
**Actions**:
- Identify production data file locations and timestamps
- Compare FRED series IDs, CoStar exports, BLS datasets
- Check for data updates or revisions between runs
- Validate data consistency across experiments

**Timeline**: 1 day
**Risk**: High - Different data versions could invalidate all comparisons

---

### Priority 2: Hyperparameter Comparison 🔍
**Objective**: Document exact parameter differences
**Actions**:
- Extract production LightGBM hyperparameters from source code
- Compare to experimental parameters (EXP-001 through EXP-004)
- Test whether production parameters improve experimental models
- Quantify sensitivity to parameter choices

**Timeline**: 0.5 days
**Risk**: Medium - May reveal overlooked tuning opportunities

---

### Priority 3: SARIMA Configuration Analysis 📊
**Objective**: Compare production vs experimental SARIMA orders
**Actions**:
- Extract production SARIMA (p,d,q)(P,D,Q,s) parameters
- Compare to experimental (2,1,2)(0,0,1,4)
- Test whether production SARIMA parameters improve experiments
- Analyze AIC/BIC differences and forecast quality

**Timeline**: 1 day
**Risk**: Low - SARIMA changes are testable independently

---

### Priority 4: Training Period Documentation 📅
**Objective**: Document exact training/test splits
**Actions**:
- Record production training start/end dates
- Compare to experimental splits (likely different due to data availability)
- Test whether production training periods improve experimental models
- Analyze regime representation in different splits

**Timeline**: 0.5 days
**Risk**: Low - Informational, helps contextualize results

---

### Priority 5: Meta-Learner Investigation 🧠
**Objective**: Understand production meta-learning strategy
**Actions**:
- Extract production meta-learner code (if different from Ridge)
- Compare weights, regularization, validation strategies
- Test whether production meta-learner improves experimental ensembles
- Analyze negative coefficient handling

**Timeline**: 1 day
**Risk**: Medium - May reveal ensemble architecture advantages

---

## Experimental Recommendations

### ENSEMBLE-EXP-005: Production Data Replication
**Objective**: Test whether using exact production data improves experimental models

**Configuration**:
- Use production data files (if accessible)
- Apply EXP-003 architecture (StandardScaler + Ridge meta-learner)
- Train on production training period
- Validate on production test period

**Hypothesis**: Data quality/versions explain the gap

**Success Criteria**: Experimental RMSE ≈ production RMSE (0.5046)

---

### ENSEMBLE-EXP-006: Production Hyperparameter Adoption
**Objective**: Test whether production hyperparameters improve experimental models

**Configuration**:
- Use experimental data (for controlled comparison)
- Apply production LightGBM hyperparameters exactly
- Keep EXP-003 meta-learner (Ridge)
- Same train/test split as experiments

**Hypothesis**: Hyperparameter tuning explains the gap

**Success Criteria**: RMSE improvement vs EXP-003 (0.5936)

---

### ENSEMBLE-EXP-007: Production SARIMA Integration
**Objective**: Test whether production SARIMA parameters improve ensemble

**Configuration**:
- Use production SARIMA (p,d,q)(P,D,Q,s) order
- Combine with EXP-003 LightGBM
- Ridge meta-learner
- Same data as EXP-003

**Hypothesis**: Better SARIMA improves ensemble even with negative weights

**Success Criteria**: RMSE improvement vs EXP-003, reduced directional accuracy gap

---

## Conclusions

1. **Feature Engineering is Identical**: Production uses simple transformations (lags, growth rates, ratios, interactions, StandardScaler) identical to experimental models.

2. **No Advanced Preprocessing**: Production does NOT use log transforms, regime normalization, complex differencing, or outlier treatment.

3. **StandardScaler is Standard**: Applied uniformly after feature engineering, same as EXP-003.

4. **Gap Explanation Shifts**: The 17.7% performance gap (EXP-003 0.5936 vs production 0.5046) must come from data quality, hyperparameters, SARIMA configuration, training periods, or meta-learning strategy.

5. **Investigation Continues**: Next priority is data source comparison, followed by hyperparameter analysis and SARIMA configuration investigation.

---

## Appendix: Production Feature Engineering Code

### Lagged Variables (unified_data_loader.py, lines 266-287)

```python
# Employment lags (3-month lag)
merged_df['phx_prof_business_employment_lag1'] = merged_df['phx_prof_business_employment'].shift(1)
merged_df['phx_total_employment_lag1'] = merged_df['phx_total_employment'].shift(1)

# Supply lags (15-24 month delivery time)
for lag in [5, 6, 7, 8]:
    merged_df[f'units_under_construction_lag{lag}'] = merged_df['units_under_construction'].shift(lag)

# Mortgage rate lag (6-month lag)
merged_df['mortgage_rate_30yr_lag2'] = merged_df['mortgage_rate_30yr'].shift(2)
```

### Growth Rates (unified_data_loader.py, lines 297-301)

```python
# Year-over-year employment growth
merged_df['phx_employment_yoy_growth'] = merged_df['phx_total_employment'].pct_change(4) * 100
merged_df['phx_prof_business_yoy_growth'] = merged_df['phx_prof_business_employment'].pct_change(4) * 100
```

### Ratios (unified_data_loader.py, lines 303-306)

```python
# Supply/inventory pressure
merged_df['supply_inventory_ratio'] = (merged_df['units_under_construction'] / merged_df['inventory_units']) * 100

# Demand indicator
merged_df['absorption_inventory_ratio'] = (merged_df['absorption_12mo'] / merged_df['inventory_units']) * 100
```

### Interactions (unified_data_loader.py, lines 309-311)

```python
# Mortgage rate × employment interaction
merged_df['mortgage_employment_interaction'] = merged_df['mortgage_rate_30yr'] * merged_df['phx_employment_yoy_growth']
```

### Migration Proxy (unified_data_loader.py, lines 318-324)

```python
# Employment growth as migration proxy
merged_df['migration_proxy'] = merged_df['phx_employment_yoy_growth'] * 1000  # Scale to approximate people
```

### StandardScaler (gbm_phoenix_specific.py, lines 172-174)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

**Analysis Complete**: 2025-11-08
**Next Step**: Data source investigation (Priority 1)
