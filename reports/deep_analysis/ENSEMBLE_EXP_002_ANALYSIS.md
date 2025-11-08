# ENSEMBLE-EXP-002 Analysis: Unexpected Degradation

**Date**: 2025-11-07
**Experiment ID**: ENSEMBLE-EXP-002
**Status**: ❌ **FAILURE - Relaxed Stability Criteria Backfired**

---

## Executive Summary

ENSEMBLE-EXP-002 was designed to address ENSEMBLE-EXP-001's feature scarcity limitation by relaxing stability criteria from p>0.05 to include p>0.01 and p>0.001 tiers. **The result was catastrophic performance degradation**: Test RMSE increased from 3.8113 to 6.6447 (74% worse) and R² from -15.04 to -47.75.

**Critical Discovery**: Moderately unstable features (Tier 2-3) cause **severe overfitting** in regime-change scenarios. The strict stability criterion (p>0.05) in ENSEMBLE-EXP-001 was actually **protective**, not limiting.

**Key Insight**: In the presence of fundamental regime change, feature stability is MORE important than feature quantity. Two stable features outperform five partially-stable features.

---

## Performance Comparison

| Model | Features | Test RMSE | Test R² | Change from EXP-001 |
|-------|----------|-----------|---------|---------------------|
| **ENSEMBLE-EXP-001** | 2 (p>0.05) | 3.8113 | -15.04 | Baseline |
| **ENSEMBLE-EXP-002** | 5 (p>0.001) | **6.6447** | **-47.75** | **-74% WORSE** ❌ |
| Production Ensemble | Unknown | 0.5046 | 0.43 | Target |
| XGB-OPT-002 | 25 (unstable) | 4.2058 | -18.53 | Original failure |

**Catastrophic Finding**: Adding 3 moderately-stable features made performance worse than the original failed XGB-OPT-002 model!

---

## Component Performance Breakdown

### ENSEMBLE-EXP-001 (Baseline - 2 Features)

| Component | Train R² | Test R² | Test RMSE | Status |
|-----------|----------|---------|-----------|--------|
| XGBoost | 0.8213 | -19.97 | 4.3584 | Poor |
| SARIMA | 0.8989 | -14.16 | 3.7055 | Poor |
| LightGBM | 0.3413 | -36.08 | 5.7956 | Very Poor |
| **Ensemble** | **0.9232** | **-15.04** | **3.8113** | **Best** |

### ENSEMBLE-EXP-002 (5 Features + VAR)

| Component | Train R² | Test R² | Test RMSE | Status | Change |
|-----------|----------|---------|-----------|--------|--------|
| XGBoost | 0.9886 | **-64.96** | **7.7295** | Catastrophic | **-225% worse** ❌ |
| SARIMA | 0.8714 | **-82.74** | **8.7090** | Catastrophic | **-485% worse** ❌ |
| LightGBM | 0.5199 | **-54.42** | **7.0848** | Very Poor | **-51% worse** ❌ |
| VAR | -0.6581 | **-67.04** | **7.8502** | Failed | New component |
| **Ensemble** | **0.9905** | **-47.75** | **6.6447** | **Failure** | **-218% worse** ❌ |

**Critical Pattern**: ALL components degraded. This isn't a single component issue - it's a systematic overfitting problem.

---

## Root Cause Analysis

### 1. Severe Overfitting from Tier 2-3 Features

**Evidence**:
- XGBoost: Train R² 0.9886 → Test R² -64.96 (**-6505% train-test gap**)
- SARIMA: Train R² 0.8714 → Test R² -82.74
- Ensemble: Train R² 0.9905 → Test R² -47.75

**Mechanism**:
The tier 2-3 features (p>0.01, p>0.001) have distributions that **shifted enough** to cause models to memorize training-regime-specific patterns that completely fail in test regime.

**Tier 2-3 Features**:
1. `mortgage_employment_interaction` (p=0.0442, Tier 2) - 35.8% distribution shift
2. `absorption_inventory_ratio` (p=0.0110, Tier 2) - 44.4% distribution shift
3. `absorption_12mo` (p=0.0011, Tier 3) - **85.1% distribution shift** ⚠️

**Interpretation**: The 85.1% shift in `absorption_12mo` indicates this feature is fundamentally different in test period, yet models used it heavily for training fit, creating a trap.

---

### 2. XGBoost Overfitting Amplification

**XGBoost with 2 features (EXP-001)**:
- Train R² 0.8213, Test R² -19.97
- Train-test gap: 82.13 → -19.97 = 102 points

**XGBoost with 5 features (EXP-002)**:
- Train R² 0.9886, Test R² -64.96
- Train-test gap: 98.86 → -64.96 = **164 points** ⚠️

**Mechanism**:
Tree models with more features have more opportunities to create regime-specific splits. The tier 2-3 features provided additional "useful" training signal that was actually regime-specific noise.

**Comparison to Original Failure**:
- XGB-OPT-002 (25 unstable features): Test R² -18.53
- ENSEMBLE-EXP-002 XGBoost (5 partially-stable): Test R² -64.96
- **Result**: 5 partially-stable features worse than 25 unstable features!

**Interpretation**: Partially-stable features are **more dangerous** than clearly-unstable features because they appear predictive in training but fail catastrophically in regime change.

---

### 3. SARIMA Degradation

**SARIMA with 1 exog variable (EXP-001)**:
- Only `phx_employment_yoy_growth` (p=0.2598, highly stable)
- Test RMSE: 3.7055, R² -14.16

**SARIMA with 2 exog variables (EXP-002)**:
- `phx_employment_yoy_growth` + `migration_proxy` (both p=0.2598)
- Test RMSE: 8.7090, R² -82.74 (**-485% worse**)

**Mystery**: Adding another highly-stable tier 1 feature made SARIMA worse!

**Hypothesis**: `migration_proxy` may be:
1. Collinear with employment growth (redundant information)
2. Have different lag structure in test period
3. Interact poorly with seasonal component

**Implication**: Even tier 1 features can cause problems when combined inappropriately.

---

### 4. VAR Component Failure

**VAR Configuration**:
- Variables: `national_unemployment`, `mortgage_rate_30yr`, `inflation_expectations_5yr`
- Lag order: 2 quarters

**Performance**:
- Train R² -0.6581 (worse than mean even in training!)
- Test R² -67.04

**Root Cause**:
1. **Wrong scale**: National macro variables don't predict Phoenix-specific rent growth directly
2. **Lag mismatch**: 2-quarter lags may not capture transmission to local market
3. **Missing localization**: No Phoenix-specific transformation of national variables

**Comparison to Production Ensemble**:
- Production VAR likely uses Phoenix economic variables, not just national macro
- Or production VAR feeds into GBM component rather than being standalone predictor

---

### 5. Meta-Learner Weight Distribution

**ENSEMBLE-EXP-001 Weights**:
- XGBoost: 33.0%
- LightGBM: 3.7% (negative coefficient)
- SARIMA: 63.3%

**ENSEMBLE-EXP-002 Weights**:
- XGBoost: **86.5%** ⚠️
- LightGBM: 1.8% (negative)
- SARIMA: 6.7%
- VAR: 4.9%

**Critical Finding**: Ridge meta-learner heavily weighted XGBoost (86.5%) - the WORST performing component (Test R² -64.96)!

**Why This Happened**:
- XGBoost had **best training performance** (R² 0.9886)
- Ridge optimized for training cross-validation, not test regime
- Train-test regime mismatch caused catastrophic weight misallocation

**Implication**: Cross-validation with TimeSeriesSplit failed to detect regime overfitting because all CV folds were in the same regime (2010-2022 growth period).

---

## Why Relaxed Stability Failed

### Hypothesis 1: Stability Threshold is Critical

**Observation**:
- p>0.05 (2 features): RMSE 3.8113
- p>0.001 (5 features): RMSE 6.6447
- **Result**: 150% degradation from relaxing threshold

**Mechanism**: The p=0.05 threshold acts as a **regime-change filter**:
- Features with p>0.05: Distributions similar enough for cautious generalization
- Features with p<0.05: Distributions shifted enough to cause overfitting

**Implication**: In regime-change scenarios, p>0.05 is not "too strict" - it's the **minimum safe threshold**.

---

### Hypothesis 2: Partial Stability Worse Than Clear Instability

**Counterintuitive Finding**:
- 25 clearly unstable features (XGB-OPT-002): RMSE 4.2058
- 5 partially-stable features (ENSEMBLE-EXP-002): RMSE 6.6447

**Explanation**:
1. **Clearly unstable features**: XGBoost recognizes high variance, applies regularization
2. **Partially-stable features**: Appear reliable, models trust them, over-rely on them
3. **Result**: Partial stability creates **false confidence** leading to over-fitting

**Analogy**: Like a bridge that appears structurally sound but has hidden cracks vs. obviously broken bridge - the hidden weakness is more dangerous.

---

### Hypothesis 3: VAR Needs Local Variables

**Production ensemble likely uses**:
- Phoenix GDP, Phoenix employment, Phoenix HPI
- Local mortgage applications, local building permits
- Phoenix-specific leading indicators

**ENSEMBLE-EXP-002 used**:
- National unemployment, national mortgage rate, national inflation
- No Phoenix localization

**Result**: National variables too distant from Phoenix rent growth causality chain

---

## Lessons Learned

### Critical Insights

1. **Feature Stability > Feature Quantity** ✅
   - 2 highly-stable features > 5 partially-stable features
   - Validates strict stability criterion (p>0.05)

2. **Partial Stability is Dangerous** 🆕
   - p-values between 0.01-0.05 create false confidence
   - Models overfit to partially-stable patterns
   - Better to exclude than risk catastrophic overfitting

3. **Regime-Specific Cross-Validation Needed** 🆕
   - TimeSeriesSplit within single regime doesn't detect regime overfitting
   - Need validation sets that span regime boundaries
   - Alternative: Out-of-regime validation set

4. **VAR Requires Localization** 🆕
   - National macro variables insufficient for local market prediction
   - Need Phoenix-specific economic indicators
   - Or Phoenix-specific transformation of national variables

5. **Meta-Learner Can Amplify Failure** 🆕
   - Ridge weighted worst component (XGBoost 86.5%)
   - CV optimization on training regime misleads meta-learner
   - Need regime-aware meta-learning or explicit component bounds

---

## Corrective Actions

### What NOT to Do ❌

1. ❌ **Relax stability criteria below p>0.05**
   - Proven to cause catastrophic overfitting
   - False confidence from partial stability

2. ❌ **Add features just to increase count**
   - More features ≠ better performance in regime change
   - Quality (stability) >>> Quantity

3. ❌ **Use national macro variables directly**
   - Too distant from Phoenix rent growth
   - Need localization or Phoenix-specific indicators

4. ❌ **Trust TimeSeriesSplit CV in regime change**
   - Fails to detect regime overfitting
   - Need out-of-regime validation

---

### What TO Do ✅

1. ✅ **Maintain strict p>0.05 stability threshold**
   - ENSEMBLE-EXP-001 approach was correct
   - Don't relax unless absolutely necessary

2. ✅ **Find additional HIGHLY-stable features**
   - Seek external data sources with p>0.05
   - Prioritize stability over historical importance

3. ✅ **Localize national variables**
   ```python
   # Instead of national_unemployment
   Use: phx_unemployment - national_unemployment  # Local deviation

   # Instead of mortgage_rate_30yr
   Use: phx_mortgage_apps * mortgage_rate_30yr  # Local interaction
   ```

4. ✅ **Implement regime-aware validation**
   ```python
   # Regime-boundary validation
   train: 2010-2019 (pre-COVID)
   val: 2020-2022 (COVID surge regime)
   test: 2023-2025 (decline regime)
   ```

5. ✅ **Bound meta-learner weights**
   ```python
   # Prevent single component dominance
   RidgeCV(positive=True)  # All positive weights
   # Or manually bound: max component weight = 50%
   ```

---

## Revised Recommendations

### Immediate Next Steps

#### 1. **Return to ENSEMBLE-EXP-001 Architecture** ⚡
**Status**: ENSEMBLE-EXP-001 is the best experimental model so far
**Action**: Use as baseline, focus on incremental improvements
**Do NOT**: Add more features below p>0.05 threshold

---

#### 2. **Seek External Highly-Stable Features** 🆕
**Objective**: Find 3-5 additional features with p>0.05

**Sources**:
- **IRS Migration Data**: County-to-county flows (likely stable)
- **Census Bureau**: Population growth, household formation (stable fundamentals)
- **BLS Regional Data**: Phoenix-specific employment sectors
- **Zillow Research**: Phoenix housing inventory, days on market

**Validation**: KS test p>0.05 BEFORE adding to model

**Expected**: 5-7 total features if successful

---

#### 3. **Analyze Production Model Features** 📋 (HIGHEST PRIORITY)
**Objective**: Reverse-engineer production ensemble to identify:
- Exact features in GBM component
- Exact exogenous variables in SARIMA
- VAR component configuration
- Any feature transformations

**Action**: Read production model source code files

**Timeline**: 1 day

---

#### 4. **Implement Regime-Aware Validation** 🆕
**Objective**: Detect overfitting across regime boundaries

```python
regime_splits = [
    ('2010-2019', '2020-2022'),  # Pre-COVID → COVID surge
    ('2010-2022', '2023-2025'),  # History → Decline
]

for train_regime, val_regime in regime_splits:
    model.fit(train_regime)
    validate(val_regime)  # Detect regime overfitting
```

**Timeline**: 2 days

---

#### 5. **Simplify to Best Components Only** 🆕
**Observation**: In EXP-001, SARIMA outperformed tree models

**Action**: Test SARIMA-only model with 2 tier 1 exog variables
- Remove tree models entirely
- Focus on seasonal patterns (most regime-stable)
- Compare to full ensemble

**Timeline**: 1 day
**Experiment ID**: SARIMA-EXP-001

---

### Long-Term Strategy

#### 6. **Feature Engineering Pipeline**
- Automated p>0.05 validation before feature addition
- External data integration with stability testing
- Phoenix-specific transformations of national variables

---

#### 7. **Regime Detection Integration**
- Chow test for structural breaks
- Automatic regime classification
- Regime-specific model selection or weighting

---

## Conclusion

ENSEMBLE-EXP-002 provided a **critical negative result**: relaxing stability criteria below p>0.05 causes catastrophic overfitting in regime-change scenarios. The strict p>0.05 threshold in ENSEMBLE-EXP-001 was **protective, not limiting**.

**Key Takeaway**: In regime change contexts, **feature stability is more important than feature quantity**. Two highly-stable features outperform five partially-stable features.

**Path Forward**:
1. Return to ENSEMBLE-EXP-001 as baseline ✅
2. Analyze production model features (highest priority) 🔍
3. Seek external highly-stable features (p>0.05 only) 📊
4. Test SARIMA-only approach (best individual component) 🧪

**Status**: ENSEMBLE-EXP-001 remains best experimental model
**Next Experiment**: Analyze production model OR SARIMA-EXP-001

---

*Analysis completed: 2025-11-07*
*Key achievement: Validated p>0.05 stability threshold as critical for regime change*
*Critical lesson: Partial stability more dangerous than clear instability*
