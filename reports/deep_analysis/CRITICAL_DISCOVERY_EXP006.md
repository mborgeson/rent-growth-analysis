# CRITICAL DISCOVERY: Feature Set Hypothesis Falsified
## EXP-005 Already Had All 26 Production Features

**Date**: November 8, 2025
**Discovery**: EXP-005 and EXP-006 used IDENTICAL feature sets
**Impact**: Entire feature set hypothesis was based on flawed analysis

---

## The Error

### What I Thought
After analyzing production model feature names, I concluded that production used 8 critical features that experimental models were missing:
1. `fed_funds_rate`
2. `national_unemployment`
3. `cpi`
4. `cap_rate`
5. `phx_home_price_index`
6. `phx_hpi_yoy_growth`
7. `phx_manufacturing_employment`
8. `vacancy_rate`

### What Was Actually True
**EXP-005 ALREADY HAD ALL 8 FEATURES** (lines 79, 92, 95, 102-103, 111-113):
```python
phoenix_features = [
    # Employment
    'phx_manufacturing_employment',  # ✅ ALREADY HAD THIS

    # Market Conditions
    'vacancy_rate',                   # ✅ ALREADY HAD THIS
    'cap_rate',                       # ✅ ALREADY HAD THIS

    # Phoenix Home Prices
    'phx_home_price_index',           # ✅ ALREADY HAD THIS
    'phx_hpi_yoy_growth',             # ✅ ALREADY HAD THIS

    # National Factors
    'fed_funds_rate',                 # ✅ ALREADY HAD THIS
    'national_unemployment',          # ✅ ALREADY HAD THIS
    'cpi',                            # ✅ ALREADY HAD THIS
]
```

### Proof: Identical Predictions
EXP-005 and EXP-006 produced **EXACTLY IDENTICAL predictions**:
- LightGBM RMSE: 4.1058 (both experiments, 0.0% difference)
- LightGBM predictions: IDENTICAL (mean 2.4699, std 0.2194)
- Ensemble RMSE: 6.53 (both experiments, 0.0% difference)

---

## How This Happened

### My Flawed Analysis Process
1. **Step 1** ✅ Correctly extracted production model's 26 feature names
2. **Step 2** ❌ Compared against WRONG experimental feature list
3. **Step 3** ❌ Concluded 8 features were "missing"
4. **Step 4** ❌ Created EXP-006 to "add" features that were already there

### What I Compared Against
I compared production features against the EXP-003 feature list which had 25 features and WAS genuinely missing some production features. But EXP-005 (created AFTER the hyperparameter comparison) had already been updated to match production's 26 features.

### Timeline of Confusion
1. **EXP-003**: Used 25 experimental features (genuinely different from production)
2. **Hyperparameter Investigation**: Found production uses 26 features
3. **EXP-005 Creation**: Updated to 26 features (MATCHING production)
4. **My Analysis Error**: Compared production to EXP-003 instead of EXP-005
5. **EXP-006 Creation**: "Added" features that EXP-005 already had

---

## What We Actually Eliminated

### Hypotheses Successfully Tested and FALSIFIED

1. ❌ **Ensemble Architecture** (EXP-001, EXP-002)
   - Pure SARIMA vs SARIMAX with exog
   - VAR inclusion
   - Result: Not the issue

2. ❌ **StandardScaler Impact** (EXP-004)
   - Scaling vs no scaling
   - Result: Not the primary issue

3. ❌ **Early Stopping Alone** (EXP-005)
   - lgb.train() with early_stopping(50)
   - Result: Necessary but not sufficient

4. ❌ **Feature Set** (EXP-005 = EXP-006)
   - Both used identical 26 production features
   - Result: Not the issue (features were already aligned)

### What This Means

**Production and Experimental Models Share**:
- ✅ Same 26 features (including macroeconomic regime indicators)
- ✅ Same data source (`phoenix_modeling_dataset.csv`)
- ✅ Same hyperparameters (num_leaves, learning_rate, regularization, etc.)
- ✅ Same early stopping (50 rounds)
- ✅ Same preprocessing (forward fill → StandardScaler)
- ✅ Same ensemble architecture (LightGBM + Pure SARIMA + Ridge)

**Yet Production Achieves**:
- Production RMSE: **0.5046**
- Experimental RMSE: **6.5338** (EXP-005/006)
- Gap: **1194.8%**

---

## The Remaining Mystery

### What Could Explain the Gap?

Since we've eliminated:
- Feature engineering
- Data source
- Hyperparameters
- Early stopping approach
- Feature set

**The remaining possibilities are**:

1. **Training Regime Differences**
   - Production may train on DIFFERENT time period
   - Production may use different train/test split
   - Production may use different validation strategy

2. **SARIMA Implementation Differences**
   - Production might use SARIMAX with exogenous variables (not pure SARIMA)
   - Production might use different SARIMA order
   - Production might use different forecast strategy

3. **Meta-Learner Differences**
   - Production might use different stacking approach
   - Production might not use Ridge (might use different meta-learner)
   - Production might use different cross-validation strategy

4. **Hidden Production Logic**
   - Production code might have additional preprocessing not visible in code
   - Production might have ensemble voting/averaging not documented
   - Production might use model selection logic we haven't found

5. **Test Set Usage Error**
   - Experimental models might be using test set INCORRECTLY
   - Validation set strategy might differ fundamentally
   - Data leakage might be happening differently

---

## Critical Questions to Answer

### Immediate Investigation Needed

1. **What is production's actual train/test split?**
   - Check production model training logs
   - Verify training period vs test period
   - Confirm experimental split matches production

2. **What is production's SARIMA configuration?**
   - Pure SARIMA vs SARIMAX with exog
   - Order and seasonal order
   - Forecast strategy

3. **What is production's ensemble strategy?**
   - Is Ridge the actual meta-learner?
   - How are component predictions combined?
   - What validation strategy is used?

4. **Are we loading the right production model?**
   - Confirm `gbm_phoenix_specific_model.pkl` is actually used
   - Check if there's a more recent model
   - Verify model artifact metadata

---

## Corrected Understanding

### What We Know for Certain

**SHARED BETWEEN PRODUCTION & EXPERIMENTAL**:
```yaml
Features:
  Count: 26
  List: [IDENTICAL - all macroeconomic features included]

Data:
  Source: phoenix_modeling_dataset.csv (Nov 7, 2025 05:57)
  Preprocessing: forward fill + StandardScaler

LightGBM Config:
  Training: lgb.train() with early_stopping(50)
  Hyperparameters: IDENTICAL (num_leaves, learning_rate, etc.)

Architecture:
  Component 1: LightGBM
  Component 2: SARIMA (configuration UNKNOWN)
  Meta-Learner: Ridge (ASSUMED, not verified)
```

**DIFFERENT (UNKNOWN)**:
```yaml
Training Process:
  Train/Test Split: ❓ UNKNOWN if identical
  Validation Strategy: ❓ UNKNOWN if identical
  SARIMA Config: ❓ Pure vs Exog, order, seasonal_order
  Ensemble Method: ❓ Ridge vs other, CV strategy
```

---

## Next Steps

### Priority 1: Verify Train/Test Split
Check if production uses the same 2022-12-31 cutoff or a different split.

### Priority 2: Extract Production SARIMA Config
Determine if production uses:
- Pure SARIMA or SARIMAX with exog
- Same (2,1,2)(1,1,1,4) order
- Same forecast strategy

### Priority 3: Verify Production Ensemble
Confirm if production actually uses Ridge or a different meta-learner.

### Priority 4: Production Model Provenance
Verify we're comparing against the CORRECT production model artifact.

---

## Lessons Learned

1. **Always verify experimental baselines** before claiming features are "missing"
2. **Compare apples-to-apples** - EXP-005 vs production, not EXP-003 vs production
3. **Feature importance doesn't prove causation** - features can be present but ineffective
4. **Identical code can behave differently** if training/validation differs
5. **Document assumptions explicitly** to catch errors early

---

## Conclusion

The feature set hypothesis was **COMPLETELY FALSIFIED**. EXP-005 and EXP-006 used identical features, confirming that:

1. ✅ Experimental models HAVE all macroeconomic regime indicators
2. ✅ Features are used significantly (35% importance)
3. ❌ Features alone DO NOT explain the production gap
4. ❓ **Unknown factors in training/validation/SARIMA remain**

The investigation must now focus on:
- **Training regime differences**
- **SARIMA configuration**
- **Ensemble methodology**
- **Hidden production logic**

We are back to square one on explaining the 1194.8% gap to production.
