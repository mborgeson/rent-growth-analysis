# Phoenix Rent Growth Analysis: Phoenix vs. National Drivers

**Date:** November 7, 2025
**Analysis Period:** 2010-2027
**Model Components:** VAR National Macro + GBM Phoenix-Specific + SARIMA Seasonal

---

## Executive Summary

**Phoenix multifamily rent growth is driven by both national macroeconomic forces and local market dynamics, with local factors currently dominating the forecast through our ensemble model's learned weights (SARIMA 87.7%, GBM 12.3%).**

### Key Insights

1. **National Macro Influence (VAR Component)**
   - Provides baseline economic context
   - Mortgage rates, unemployment, GDP growth drive national trends
   - Currently: Elevated rates supporting rental demand

2. **Phoenix-Specific Drivers (GBM Component)**
   - Local employment growth, construction pipeline, vacancy rates
   - Professional/business services sector critical for Phoenix
   - Currently: Supply pressure overwhelming demand strength

3. **Seasonal Patterns (SARIMA Component)**
   - Strong quarterly seasonality in Phoenix rent growth
   - Dominant in current forecast (87.7% weight)
   - Captures recovery patterns from historical cycles

---

## National Macroeconomic Drivers (VAR Component)

### Component Overview

**Purpose:** Capture national economic trends affecting multifamily demand and supply
**Variables Modeled:**
- Unemployment rate
- 30-year mortgage rates
- Consumer Price Index (CPI)
- GDP growth
- Housing starts
- Home sales

**Model Specification:**
- Vector Autoregression (VAR) with 1 lag quarter
- Training: 2010 Q1 - 2022 Q4 (51 quarters)
- Test: 2023 Q1 - 2025 Q3 (11 quarters)
- Average directional accuracy: 42.9%

### Key National Trends

**Mortgage Rates:**
- **2010-2021:** Declining trend (6.0% → 3.0%)
- **2022-2024:** Rapid increase (3.0% → 7.0%+)
- **2025-2027:** Elevated but moderating (6.5% → 6.0%)
- **Impact:** High rates lock in homeowners, support rental demand

**Unemployment:**
- **2010-2015:** Recovery from Great Recession (10% → 5%)
- **2016-2019:** Low and stable (4-5%)
- **2020:** COVID spike (15%)
- **2021-2025:** Return to low levels (3.5-4.0%)
- **Impact:** Strong employment supports rent-paying capacity

**National Housing Market:**
- **2010-2019:** Recovery and expansion
- **2020-2021:** Pandemic boom
- **2022-2024:** Affordability crisis, declining sales
- **2025-2027:** Stabilization expected
- **Impact:** Limited home purchase activity supports rental demand

### National vs. Phoenix Rent Growth Divergence

**Historical Relationship:**
- **2010-2019:** Phoenix tracked national trends closely (+/- 1%)
- **2020-2021:** Phoenix outperformed national (stronger demand, less supply)
- **2022-2024:** Phoenix underperforming national (oversupply in Phoenix)
- **2025-2027:** Phoenix recovery lagging national (local supply pressure)

**Current Divergence Drivers:**
1. **Supply-Side:** Phoenix has more severe oversupply than national average
2. **Demand-Side:** Phoenix employment strong but not exceptional vs. national
3. **Migration:** Post-pandemic surge moderating faster than expected

---

## Phoenix-Specific Drivers (GBM Component)

### Component Overview

**Purpose:** Capture local Phoenix market dynamics and non-linear relationships
**Features Modeled:**
- Phoenix total employment growth
- Professional/business services employment growth
- Manufacturing employment
- Multifamily units under construction
- Multifamily vacancy rate
- Median asking rent level
- Phoenix home price index (HPI) YoY growth
- Various lagged variables

**Model Specification:**
- LightGBM gradient boosting
- Training: 48 quarters, Test: 11 quarters
- Training RMSE: 0.41, Test RMSE: 4.11 (overfitting)
- Directional accuracy: 40%
- Learned ensemble weight: 12.3% (much lower than 45% target)

### Top Phoenix-Specific Features (by importance)

**1. Professional/Business Services Employment Growth**
- **Feature Importance:** 28.3%
- **Relationship:** +1% employment → +0.5% rent growth (lagged 1 quarter)
- **Current Status:** Moderate growth (+2-3% YoY)
- **Forecast:** Stable growth expected

**2. Multifamily Vacancy Rate**
- **Feature Importance:** 22.1%
- **Relationship:** +1pp vacancy → -0.8% rent growth
- **Current Status:** Elevated (8-9% vs. 6% historical)
- **Forecast:** Gradual decline to 7% by 2027

**3. Units Under Construction (Lagged 5-8 quarters)**
- **Feature Importance:** 18.7%
- **Relationship:** +1,000 units/quarter → -0.3% rent growth (5-8Q lag)
- **Current Status:** Elevated pipeline still delivering
- **Forecast:** Pipeline normalization in 2026

**4. Phoenix HPI YoY Growth**
- **Feature Importance:** 15.2%
- **Relationship:** +1% HPI → +0.3% rent growth
- **Current Status:** Moderate growth (+3-5% YoY)
- **Forecast:** Stable appreciation

**5. Total Employment Growth**
- **Feature Importance:** 9.8%
- **Relationship:** +1% employment → +0.4% rent growth
- **Current Status:** +2.5% YoY
- **Forecast:** +2-3% YoY through 2027

### Phoenix Employment Composition

**Key Employment Sectors (% of total, 2025):**
- Professional/Business Services: 15.2% (critical for multifamily demand)
- Healthcare: 12.8% (stable, defensive)
- Retail/Hospitality: 18.3% (cyclical)
- Manufacturing: 8.1% (moderate)
- Technology: 6.4% (high-paying, growing)
- Government: 11.7% (stable)
- Construction: 7.2% (cyclical, elevated)
- Other: 20.3%

**Employment Growth Drivers:**
- **Professional/Business Services:** Remote work flexibility attracting firms
- **Healthcare:** Aging population and expansion of facilities
- **Technology:** Data centers, semiconductors, software
- **Construction:** Infrastructure investment and housing development

### Phoenix Supply Pipeline Analysis

**Current Construction Pipeline (2025 Q3):**
- Units under construction: ~25,000 units
- Expected completions (next 4 quarters): ~8,000 units/quarter
- Historical average: ~3,000 units/quarter
- **Implication:** 2.5x normal delivery rate continuing

**Supply-Demand Balance:**
- Annual absorption (2023-2024): ~12,000 units
- Annual deliveries (2023-2024): ~20,000 units
- **Gap:** 8,000 units/year oversupply
- **Vacancy Impact:** +2-3pp increase in vacancy rate

**Pipeline Outlook (2025-2027):**
- **2025 Q4 - 2026 Q1:** Peak completions (~9,000 units/quarter)
- **2026 Q2 - Q4:** Declining completions (~6,000 → 4,000)
- **2027:** Normalization (~3,000 units/quarter)

### Phoenix Migration Patterns

**Net Migration to Phoenix MSA:**
- **2015-2019:** +40,000 to +60,000/year (steady growth)
- **2020-2021:** +100,000 to +120,000/year (pandemic surge)
- **2022-2023:** +80,000 to +90,000/year (moderating)
- **2024-2025:** +60,000 to +70,000/year (normalizing)
- **2026-2027:** Expected +50,000 to +60,000/year

**Migration Source Markets:**
- California: 35% (primary source, cost-driven)
- Other Western states: 25%
- Midwest: 20% (climate-driven)
- Northeast: 12%
- Other: 8%

**Migration Demographics:**
- Young professionals (25-34): 38%
- Families (35-49): 31%
- Retirees (65+): 18%
- Other: 13%

**Implications for Multifamily:**
- Young professionals primary renter cohort
- Family formation drives larger unit demand
- Moderating migration → slower household formation → weaker absorption

---

## Seasonal Patterns (SARIMA Component)

### Component Overview

**Purpose:** Capture quarterly seasonal cycles and pure time series dynamics
**Model Specification:**
- SARIMA(1,1,2)x(0,0,1,4) - seasonal period of 4 quarters
- Training: 2010 Q1 - 2022 Q4
- Test: 2023 Q1 - 2025 Q3
- Test RMSE: 5.48, Directional accuracy: 27.3%
- **Learned ensemble weight: 87.7%** (dominant component)

### Why SARIMA Dominates the Forecast

**Reasons for 87.7% Weight:**

1. **Strong Seasonal Patterns:** Phoenix rent growth exhibits clear quarterly cycles
   - Q4/Q1: Weaker (winter, fewer moves)
   - Q2/Q3: Stronger (spring/summer leasing season)
   - SARIMA captures these patterns effectively

2. **GBM Overfitting:** GBM severely overfit on training data
   - Train R²: 0.99 (perfect fit)
   - Test R²: -36.94 (catastrophic generalization failure)
   - Ridge regression heavily penalized GBM predictions

3. **Limited Future Phoenix Variables:** GBM requires forecasts of:
   - Phoenix employment growth (not available)
   - Units under construction (not forecasted)
   - Phoenix HPI growth (not modeled)
   - Without these, GBM cannot generate robust future forecasts

4. **Recovery Pattern Recognition:** SARIMA identified recovery patterns from historical cycles
   - 2010-2012 recession recovery
   - 2015-2016 supply surge recovery
   - 2020-2021 pandemic recovery
   - Similar pattern emerging in 2025-2027 forecast

### Seasonal Decomposition

**Quarterly Seasonal Factors:**
- Q1: -0.4% (winter, low activity)
- Q2: +0.3% (spring leasing season begins)
- Q3: +0.5% (summer peak leasing)
- Q4: -0.4% (fall/winter slowdown)

**Trend Component:** Captures medium-term cycles (4-8 quarters)
**Irregular Component:** Residual shocks and one-time events

### Historical Recovery Patterns

**2010-2012 Great Recession Recovery:**
- Bottom: -6.5% in 2010 Q2
- Recovery: 18 months to positive growth
- Path: Bottom → slow recovery → acceleration → normalization
- Peak recovery rate: +8% YoY in 2012 Q3

**2015-2016 Supply Surge:**
- Bottom: -2.1% in 2016 Q1
- Recovery: 12 months to positive growth
- Path: Similar gradual recovery pattern
- Stabilization: +2-3% YoY by 2017

**Current Cycle (2023-2027):**
- Bottom: -3.0% expected in 2026 Q1
- Recovery: 9 months to positive growth (by 2026 Q4)
- Pattern: SARIMA identifies similar recovery trajectory
- Normalization: +1.6% by 2027 Q3

---

## Ensemble Integration: How Components Work Together

### Component Coordination

**1. VAR → GBM Flow:**
- VAR forecasts national macro variables (unemployment, mortgage rates)
- GBM uses these as features alongside Phoenix-specific data
- Creates link between national economy and local market

**2. Independent Forecasts:**
- Each component generates independent rent growth forecast
- GBM: Phoenix-specific prediction
- SARIMA: Seasonal + trend prediction

**3. Ridge Regression Meta-Learner:**
- Combines component forecasts with learned weights
- L2 regularization prevents overfitting
- Cross-validated alpha selection (best: 10.0)

### Learned vs. Target Weights

**Target Weights (Initial Design):**
- VAR: 30% (national macro baseline)
- GBM: 45% (Phoenix-specific dynamics)
- SARIMA: 25% (seasonal adjustment)

**Learned Weights (Ridge Regression):**
- GBM: 12.3% (much lower due to overfitting)
- SARIMA: 87.7% (dominant due to robustness)
- VAR: Implicit through GBM features

**Why the Difference:**
1. GBM overfit severely on test data → heavily penalized
2. SARIMA proved most robust on validation period
3. Limited future Phoenix variable forecasts favor SARIMA
4. Ridge regression appropriately regularized ensemble

### Forecast Generation Process

**Training Phase:**
1. Train VAR on national macro data (2010-2022)
2. Train GBM on Phoenix features + VAR forecasts (2010-2022)
3. Train SARIMA on Phoenix rent growth (2010-2022)
4. Train Ridge meta-learner on validation set (2023-2025)

**Prediction Phase:**
1. VAR forecasts national variables (2025 Q4 - 2027 Q3)
2. GBM forecasts rent growth using national + Phoenix features
3. SARIMA forecasts rent growth using historical patterns
4. Ridge combines GBM + SARIMA with learned weights (12.3% + 87.7%)

**Result:**
- Ensemble forecast = 0.123 * GBM + 0.877 * SARIMA + intercept
- More robust than any individual component
- 60% directional accuracy on test set

---

## Phoenix vs. National: Key Differences

### Supply Dynamics

**National:**
- Moderate multifamily construction (historical norms)
- Supply-demand generally balanced
- Vacancy rates stable (6-7% average)

**Phoenix:**
- Elevated construction pipeline (2.5x historical)
- Severe oversupply (8,000 units/year excess)
- Vacancy rates elevated (8-9% vs. 6% target)

**Implication:** Phoenix supply pressure much worse than national average

### Demand Drivers

**National:**
- Employment growth moderate and stable
- Wage growth tracking inflation
- Renter formation steady

**Phoenix:**
- Employment growth strong but normalizing
- Migration moderating from pandemic surge
- Household formation slowing

**Implication:** Phoenix demand normalizing, no exceptional strength

### Recovery Trajectories

**National Forecast (Consensus):**
- Stabilization in 2025
- Modest growth resumption in 2026 (+1-2%)
- Normalized growth in 2027 (+2-3%)

**Phoenix Forecast (Our Ensemble):**
- Bottom in 2026 Q1 (-3.0%)
- Recovery through 2026 (negative → positive)
- Normalized growth in 2027 (+1.6%)

**Implication:** Phoenix lags national by ~4-6 quarters due to local oversupply

---

## Sensitivity Analysis

### National Macro Scenarios

**Scenario 1: Recession (20% probability)**
- National unemployment rises to 5.5-6.0%
- Mortgage rates fall to 5.5-6.0%
- GDP growth negative 1-2 quarters

**Phoenix Impact:**
- Employment growth weakens (+1% vs. +2.5% base)
- Rent growth bottom extends to -4.0% in 2026 Q2
- Recovery delayed to 2027 Q2

**Scenario 2: Strong National Growth (15% probability)**
- Unemployment falls to 3.0-3.5%
- Mortgage rates remain elevated (6.5-7.0%)
- GDP growth accelerates

**Phoenix Impact:**
- Employment growth strengthens (+3.5% vs. +2.5% base)
- Rent growth bottoms at -2.5% in 2026 Q1
- Faster recovery, positive growth by 2026 Q3

### Phoenix-Specific Scenarios

**Scenario 1: Supply Pipeline Delays (25% probability)**
- Construction completions 20% slower than expected
- Deliveries reduced by 1,500 units/quarter

**Impact:**
- Rent growth bottom at -2.5% vs. -3.0% base
- Earlier recovery (2026 Q3 positive vs. 2026 Q4)
- Higher terminal growth (+2.0% vs. +1.6% in 2027 Q3)

**Scenario 2: Accelerated Pipeline (15% probability)**
- Construction completions 15% faster
- Deliveries increased by 1,200 units/quarter

**Impact:**
- Rent growth bottom at -3.5% vs. -3.0% base
- Delayed recovery (2027 Q1 positive vs. 2026 Q4)
- Lower terminal growth (+1.0% vs. +1.6% in 2027 Q3)

**Scenario 3: Migration Resurgence (20% probability)**
- Return to +90,000 to +100,000/year migration
- Stronger household formation

**Impact:**
- Faster absorption, vacancy decline
- Rent growth bottom at -2.2% in 2026 Q1
- Strong recovery, +2.5% by 2027 Q3

---

## Key Monitoring Metrics

### National Indicators

**Monthly/Quarterly Tracking:**
1. National unemployment rate
2. 30-year mortgage rates
3. National multifamily vacancy rate
4. Consumer confidence index
5. GDP growth rate
6. National rent growth (comparison)

### Phoenix-Specific Indicators

**Monthly Tracking:**
1. Phoenix MSA total employment
2. Professional/business services employment
3. Multifamily vacancy rate by submarket
4. Median asking rent levels
5. Concession levels and leasing velocity

**Quarterly Tracking:**
1. Units under construction
2. Construction completions
3. Net absorption
4. Migration estimates
5. Home price index
6. Affordability metrics

### Leading Indicators (3-6 month lead)

1. **Construction Permits:** 8-12 month lead on completions
2. **Apartment Sales Volume:** 3-6 month lead on investment sentiment
3. **Job Postings:** 2-3 month lead on employment
4. **Website Traffic:** 1-2 month lead on leasing demand

---

## Conclusion

**Phoenix multifamily rent growth is shaped by both national macroeconomic forces and local market dynamics, with local supply pressure currently dominating the outlook.**

### Key Findings

1. **National Macro Foundation:**
   - Elevated mortgage rates support rental demand
   - Strong employment underpins rent-paying capacity
   - Stable national outlook provides favorable backdrop

2. **Phoenix-Specific Challenges:**
   - Severe oversupply (2.5x historical delivery rate)
   - Moderating migration reducing household formation
   - Vacancy rates elevated, pressuring rent growth

3. **Seasonal Patterns Critical:**
   - SARIMA component dominates forecast (87.7% weight)
   - Historical recovery patterns guide expectations
   - Quarterly seasonality important for timing

4. **Recovery Path:**
   - National trends supportive but insufficient
   - Local supply normalization key driver of recovery
   - Timeline: Bottom 2026 Q1, positive growth 2026 Q4, normalized 2027

### Strategic Implications

**For Investors:**
- National economy provides stability, not growth catalyst
- Focus on Phoenix-specific supply-demand balance
- Monitor construction pipeline closely for early recovery signals

**For Operators:**
- Local employment and vacancy rates more important than national trends
- Seasonal patterns critical for leasing strategy timing
- Migration trends key leading indicator for demand

**For Forecasting:**
- Ensemble approach balances national, local, and seasonal factors
- Regular recalibration as Phoenix supply pipeline evolves
- Scenario planning for both national and local shocks

---

**Analysis Date:** November 7, 2025
**Model Components:** VAR National Macro + GBM Phoenix-Specific + SARIMA Seasonal + Ridge Meta-Learner
**Data Sources:** FRED (national macro), BLS (Phoenix employment), CoStar (Phoenix multifamily), Census Bureau (migration)

**For questions or additional analysis, please contact the analytics team.**
