# Phoenix Rent Growth Forecasting - Key Data Insights

**Analysis Date:** November 7, 2025
**Data Coverage:** 2010-2025 (15+ years)

---

## Executive Summary

Successfully acquired 11 time series from FRED covering national macro variables and Phoenix home prices. Preliminary correlation analysis reveals **mortgage rates** and **Fed funds rate** show strongest negative correlations with Phoenix home price growth, while **building permits** and **housing starts** show moderate positive correlations.

**Key Finding:** Phoenix home prices currently declining YoY (-1.7%) after extraordinary pandemic boom (+19.5% avg 2020-2022), consistent with national normalization trend.

---

## Current Market Conditions (Latest 12 Months)

### National Macro Environment

| Indicator | Current | 12-Mo Avg | 12-Mo Range | Trend |
|-----------|---------|-----------|-------------|-------|
| **30-Yr Mortgage Rate** | 6.22% | 6.67% | 6.17% - 7.04% | ↓ Declining from peak |
| **Fed Funds Rate** | 3.87% | 4.31% | 3.86% - 4.58% | ↓ Peak reached, stable |
| **Unemployment Rate** | 4.3% | 4.2% | 4.0% - 4.3% | → Stable/slight uptick |

**Interpretation:**
- Mortgage rates peaked at 7%+ in 2023, now moderating to 6.2%
- Fed appears to be at or near terminal rate (3.87%)
- Labor market remains tight but showing signs of softening

### Phoenix Home Price Trends

| Period | Avg YoY Growth | Regime |
|--------|----------------|--------|
| **2010-2012** | +3.4% | Post-Crisis Recovery |
| **2013-2019** | +7.9% | Sustained Growth |
| **2020-2022** | +19.5% | **Pandemic Boom** |
| **2023-2025** | +0.4% | **Normalization/Correction** |

**Current Phoenix HPI:** 323.53 (Index, 2000=100)
**Current YoY Growth:** -1.7% (negative for first time since 2011)
**24-Month Avg:** +2.1%

**Key Insight:**
Phoenix home prices are **declining year-over-year** for the first time in 14 years, after unsustainable pandemic-era gains. The market is normalizing but has not crashed - prices remain 3.2× higher than 2010 levels.

---

## Variable Correlations with Phoenix Home Price Growth

### Strong Correlations (|r| > 0.35)

| Variable | Correlation | Interpretation |
|----------|-------------|----------------|
| **Mortgage Rates** | **-0.432** | Higher mortgage rates → lower home price growth (affordability constraint) |
| **Fed Funds Rate** | **-0.387** | Higher Fed policy rate → tighter financial conditions → slower price growth |
| **Building Permits** | **+0.386** | More permits → stronger growth (demand signal, not supply overhang) |
| **Housing Starts** | **+0.353** | More construction → stronger growth (pro-cyclical indicator) |
| **Inflation Expectations** | **+0.351** | Higher inflation → home prices as hedge → stronger growth |

### Moderate Correlations (0.15 < |r| < 0.35)

| Variable | Correlation | Interpretation |
|----------|-------------|----------------|
| **National Home Prices** | **+0.184** | Phoenix partially decoupled from national trends (local factors matter) |

### Weak/No Correlation (|r| < 0.15)

| Variable | Correlation | Notes |
|----------|-------------|-------|
| Case-Shiller US HPI | +0.048 | Phoenix dynamics differ significantly from national |
| Unemployment Rate | +0.009 | Minimal relationship (Phoenix job market unique) |
| CPI | -0.031 | Little direct relationship |
| National Employment | -0.082 | Weak negative (Phoenix more dependent on local employment) |

---

## Critical Insights for Rent Growth Modeling

### 1. Interest Rate Environment is Key Driver

**Finding:** Mortgage rates show strongest correlation (-0.43) with Phoenix home price growth.

**Implication for Rent Growth:**
- **Current Environment:** Mortgage rates at 6.2% (down from 7% peak)
- **Rent vs Own Decision:** Higher mortgage rates → rental demand increases
- **Forecast:** If mortgage rates decline further (Fed cuts), could moderate rental demand
- **Model Feature:** Include `mortgage_rate_6mo_lag` as Tier 1 predictor

### 2. Phoenix Shows Local Market Dynamics

**Finding:** Low correlation (+0.184) with national median home prices suggests Phoenix operates on local factors.

**Implication:**
- **California Migration:** Critical local driver (45% of in-migration)
- **Semiconductor Investment:** $40B Intel/TSMC unique to Phoenix
- **Submarket Heterogeneity:** Tempe (ASU, tech) ≠ Glendale (families)
- **Model Priority:** Phoenix-specific variables (employment, migration) more important than national aggregates

### 3. Supply Indicators are Pro-Cyclical (Not Contrarian)

**Finding:** Building permits (+0.386) and housing starts (+0.353) positively correlated with price growth.

**Interpretation:**
- Permits/starts reflect **demand strength**, not future supply pressure
- Builders respond to demand signals (build more when market strong)
- **Lag Structure Critical:** Construction impact materializes 15-24 months later
- **Model Feature:** Use `units_under_construction_15mo_lag` to capture future supply pressure

### 4. Phoenix Market Regime Change Detected (2023+)

**Finding:** YoY growth decelerated from +19.5% (2020-2022) to +0.4% (2023-2025), now **negative** (-1.7%).

**Implications:**
- **Rent Growth Likely Following:** If SFH prices declining, rents may moderate
- **Migration Slowing:** California out-migration may have peaked
- **Affordability Improving:** Cooling home prices → more home buying → less rental demand
- **Model Consideration:** Regime-switching models or recent data weighting to capture structural break

---

## Data Quality Assessment

### Strengths ✅

1. **Long Time Series:** 15+ years (2010-2025) captures full cycle
   - Includes Great Recession recovery
   - Includes pandemic boom/bust
   - Includes current normalization

2. **High-Quality Sources:** Federal Reserve (FRED), authoritative data

3. **Appropriate Frequencies:**
   - Daily: Fed funds rate, inflation expectations (can aggregate to monthly)
   - Weekly: Mortgage rates (smooth to monthly)
   - Monthly: Employment, CPI, housing data (ideal for modeling)
   - Quarterly: Median home prices (interpolate if needed)

4. **Complete Phoenix HPI:** 188 monthly observations with 0% missing data

### Weaknesses ⚠️

1. **Sparse Daily Data:** Most variables monthly/quarterly, padded with NaN in daily dataset
   - **Fix:** Resample to monthly frequency for modeling

2. **Missing Dependent Variable:** No rent growth data yet (CoStar required)

3. **Missing Phoenix Employment:** BLS API issues
   - **Fix:** Manual download from BLS.gov

4. **No Supply Pipeline Data:** Critical for forecasting
   - **Fix:** CoStar subscription or manual permit aggregation

---

## Recommended Model Enhancements Based on Data

### 1. Use Lagged Interest Rate Variables

**Evidence:** Strong negative correlation (-0.43) with mortgage rates

**Implementation:**
- Test lag structures: 3-month, 6-month, 12-month
- Consider interaction term: `mortgage_rate × employment_growth`
- High rates + weak job growth = double negative for home prices → rental demand boost

### 2. Incorporate Regime Indicators

**Evidence:** Clear regime breaks (2020 pandemic, 2023 normalization)

**Implementation:**
- Dummy variables for pandemic period (2020-2022)
- Markov regime-switching model to auto-detect regimes
- Time-varying coefficients (recent data weighted higher)

### 3. Prioritize Phoenix-Specific Data Acquisition

**Evidence:** Low correlation (+0.18) with national prices

**Priority Ranking:**
1. **Phoenix employment** (manual BLS download) - LOCAL DEMAND
2. **CA migration** (IRS data) - UNIQUE TO PHOENIX
3. **ASU enrollment** - SUBMARKET DRIVER (Tempe)
4. **Semiconductor investment tracking** - PHOENIX-SPECIFIC

### 4. Model Supply with Proper Lags

**Evidence:** Permits/starts positively correlated (demand indicator, not supply)

**Implementation:**
- Use `units_under_construction_t-15` to `t-24` (lagged 15-24 months)
- This captures **future** supply pressure, not current demand strength
- Separate variable: `permits_current` (demand indicator) vs `deliveries_future` (supply pressure)

---

## Next Steps - Data Acquisition Priority

### Immediate (This Week)

1. **Resample FRED data to monthly** for modeling consistency
2. **Download BLS Phoenix employment manually**
   - Series IDs provided in data acquisition script
   - Save to: `data/raw/bls_phoenix_employment_manual.csv`

3. **Download IRS migration data (CA → AZ)**
   - Last 5 years available
   - Aggregate to annual net flows

### Short-Term (Next 2 Weeks)

4. **CoStar subscription or trial** - **CRITICAL BLOCKER**
   - Dependent variable (rent growth) required
   - Supply pipeline data unavailable elsewhere

5. **Download Zillow ZHVI** for additional granularity
6. **Create feature engineering pipeline**
   - Calculate lagged variables
   - Compute YoY growth rates
   - Generate interaction terms

### Medium-Term (Next Month)

7. **Develop VAR model** with available FRED data (practice)
8. **Set up model validation framework** (time series CV)
9. **Begin literature review** for additional predictors

---

## Key Takeaways

1. ✅ **Successfully acquired 11 critical time series** covering national macro environment
2. 📊 **Mortgage rates are #1 predictor** identified so far (-0.43 correlation)
3. 🏠 **Phoenix market normalizing** after pandemic boom (YoY growth now negative)
4. 🔍 **Local factors matter more** than national trends for Phoenix
5. ⏰ **Supply lag structure critical** - permits today ≠ supply pressure (need 15-24 mo lag)
6. 🚨 **Blocker:** Rent growth data (dependent variable) still required for model training

**Estimated Model Readiness:** 40%

**Critical Path:** CoStar subscription → Rent growth data → Model training → Forecasting
