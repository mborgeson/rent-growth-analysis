# Phoenix Multifamily Rent Growth Projection Report (2026-2030)

**Prepared:** 2025-11-06
**Market:** Phoenix-Mesa-Scottsdale, AZ MSA
**Methodology:** SARIMA/XGBoost Ensemble Model

---

## Executive Summary

Phoenix multifamily market faces **continued rent decline through 2026-2028** before stabilizing in 2029-2030 as excess supply is absorbed. Current market conditions show elevated vacancy (12.4%), substantial construction pipeline (22K units), and negative rent growth momentum (-2.8% YoY Q3 2025).

### Key Projections (Base Case)

| Metric | 2025 | 2030 | Change |
|--------|------|------|--------|
| **Avg Asking Rent** | $1,569 | $1,298 | -17.3% |
| **Vacancy Rate** | 12.4% | 8.5% | -3.9 pp |
| **Avg Annual Rent Growth** | -2.8% | -1.2% | -3.5% (5-yr avg) |

---

## Market Context & Historical Analysis

### Historical Performance (2000-2025)

**Long-Term Trends:**
- **Average Annual Rent Growth:** 2.32%
- **Median Growth:** 2.41%
- **Range:** -6.87% (2009 Q4) to +16.09% (2021 Q3)
- **Volatility:** ±4.10% standard deviation

**Market Cycles:**

1. **Pre-GFC Boom (2000-2007)**
   - Avg Rent Growth: 1.79%
   - Avg Vacancy: 9.14%
   - Rent Range: $903 → $1,045

2. **GFC/Recovery (2008-2012)**
   - Avg Rent Growth: -1.26%
   - Peak Vacancy: 12.9% (Q4 2009)
   - Rent decline period

3. **Post-GFC Expansion (2013-2019)**
   - Avg Rent Growth: 4.39%
   - Avg Vacancy: 7.23%
   - Cumulative Growth: +35.3%

4. **COVID Boom (2020-2021)**
   - Avg Rent Growth: 9.01%
   - Peak Growth: 16.09% (Q3 2021)
   - Trough Vacancy: 5.2% (Q2 2021)
   - Cumulative Growth: +21.0%

5. **Post-COVID Normalization (2022-2025)**
   - Avg Rent Growth: 0.68%
   - Current Growth: -2.79% (Q3 2025)
   - Avg Vacancy: 10.33%
   - Current Vacancy: 12.4%

---

## Statistical Analysis

### Stationarity Testing (Per Protocol)

**Augmented Dickey-Fuller (ADF) Test:**
- **Result:** Series is non-stationary (p=0.90)
- **First Difference:** Stationary (p<0.05)
- **Conclusion:** Rent series is I(1) - integrated of order 1

**KPSS Test:**
- **Result:** Series is non-stationary (p=0.01)
- **Conclusion:** Confirms ADF findings

**Recommendation:** Use d=1 (first differencing) in SARIMA model

### Structural Break Detection

**COVID-19 Break (March 2020):**
- **Chow Test F-statistic:** 117.05
- **p-value:** <0.000001
- **Conclusion:** Highly significant structural break

**Growth Rate Shifts:**
- Pre-COVID QoQ Growth: 0.50%
- COVID Period QoQ Growth: 2.53%
- Post-COVID QoQ Growth: -0.25%
- **Acceleration:** +2.03 percentage points during COVID

**Level Shift:**
- Pre-COVID Mean Rent: $1,026
- Post-COVID Mean Rent: $1,574
- **Increase:** +$549 (+53.5%)

### Autocorrelation Analysis

**Ljung-Box Test Results:**
- Lag 1 p-value: <0.0001
- Lag 4 p-value: <0.0001
- Lag 8 p-value: <0.0001

**Conclusion:** Significant autocorrelation → SARIMA models appropriate

---

## Current Market Conditions (Q3 2025)

### Supply Metrics

| Metric | Current | Pre-COVID Avg | Change |
|--------|---------|---------------|--------|
| **Inventory** | 424,896 units | 302,857 units | +40.3% |
| **Under Construction** | 22,114 units | 7,716 units | +186.5% |
| **Construction % of Inventory** | 5.2% | 2.8% | +86% |
| **Vacancy Rate** | 12.4% | 8.9% | +39% |

**Key Observations:**
- Construction pipeline 135% above historical average
- Vacancy 37% above pre-COVID levels
- ~30K excess vacant units above 5% equilibrium

### Demand Metrics

**Employment (August 2025):**
- Total Nonfarm: 2,461.6K jobs
- YoY Growth: +1.56%
- 5-Year Growth: +17.32%

**Key Sectors:**
- Professional & Business Services: 386.3K
- Education & Health: 426.9K
- Leisure & Hospitality: 265.2K

**Demographics (2023 ACS):**
- Total Population: 5,070,110
- Total Households: 1,922,068
- Renter Households: 636,378 (33.1%)
- Median HH Income: $85,700
- Median Gross Rent: $1,760

### Supply/Demand Balance

**Absorption Analysis:**
- Recent Quarterly Avg: 4,104 units
- Implied Annual: 16,414 units/year
- Pipeline/Annual Absorption: **1.6x**
- Months to Absorb Pipeline: **6.6 months**

**Vacancy Analysis:**
- Current: 12.4%
- Equilibrium: 5.0%
- **Excess Supply:** 7.2 percentage points
- **Excess Units:** ~30,099

### Rent Growth Correlation Drivers

| Driver | Correlation | Strength |
|--------|-------------|----------|
| **Vacancy Rate** | -0.886 | Strong inverse |
| **Unemployment Rate** | -0.275 | Weak inverse |
| **Employment Level** | +0.219 | Weak positive |
| **Construction Pipeline** | +0.158 | Weak positive |
| **Net Absorption** | +0.014 | Weak positive |

**Key Insight:** Vacancy rate is the dominant driver of rent growth (r=-0.89)

---

## Migration Impact Analysis

### 2021 Migration Data (IRS)

- **Net Migration:** 145,871 people
- **Implied Household Formation:** ~58,348 households
- **Renter Household Demand:** ~19,319 units
- **Share of COVID-Era Absorption:** 89%

### Migration Projections (2026-2030)

| Scenario | Annual Migration | Renter Demand | 5-Year Total |
|----------|------------------|---------------|--------------|
| **Bear** | 50,000 people | ~6,622 units/yr | 33,109 units |
| **Base** | 100,000 people | ~13,244 units/yr | 66,218 units |
| **Bull** | 150,000 people | ~19,865 units/yr | 99,327 units |

**Key Insights:**
- Migration historically accounted for 89% of COVID-era absorption
- Base case migration supports ~81% of current absorption rate
- Sunbelt migration trends remain strong but moderating from 2021 peak

---

## Model Methodology

### SARIMA + XGBoost Ensemble

**SARIMA Model:**
- Configuration: SARIMA(1,1,1)x(1,1,1,4)
- AIC: 675.29
- Validation MAE: $22.62
- Captures time series patterns and seasonality

**XGBoost Model:**
- Features: vacancy, under_construction, employment, unemployment, COVID dummy
- Validation MAE: $47.89
- Top Feature Importance:
  - Under Construction: 0.853
  - Employment: 0.142
  - Unemployment: 0.004

**Ensemble Combination:**
- Weighting: 60% SARIMA + 40% XGBoost
- **Validation MAE: $12.16** (best performance)
- Combines time series patterns with economic drivers

---

## Scenario Projections (2026-2030)

### Bear Case: Economic Slowdown, Oversupply

**Assumptions:**
- Employment Growth: 0.5% annual
- Migration: 50,000 people/year
- Construction Pipeline: 20% above current
- Vacancy Path: Remains elevated (12.5%-13.2%)

**Projections:**

| Year | Rent | YoY Growth | Vacancy |
|------|------|------------|---------|
| 2026 | $1,509 | -3.81% | 12.5% |
| 2027 | $1,437 | -4.81% | 13.0% |
| 2028 | $1,368 | -4.77% | 13.2% |
| 2029 | $1,315 | -3.88% | 13.0% |
| 2030 | $1,270 | -3.40% | 12.5% |

**5-Year Summary:**
- **Total Change:** -19.0%
- **Avg Annual:** -3.8%
- **2030 Rent:** $1,270

### Base Case: Moderate Growth, Gradual Absorption

**Assumptions:**
- Employment Growth: 1.5% annual
- Migration: 100,000 people/year
- Construction Pipeline: 20% below current
- Vacancy Path: Gradual improvement (12.0% → 8.5%)

**Projections:**

| Year | Rent | YoY Growth | Vacancy |
|------|------|------------|---------|
| 2026 | $1,465 | -6.62% | 12.0% |
| 2027 | $1,389 | -5.21% | 11.5% |
| 2028 | $1,344 | -3.25% | 10.5% |
| 2029 | $1,314 | -2.20% | 9.5% |
| 2030 | $1,298 | -1.22% | 8.5% |

**5-Year Summary:**
- **Total Change:** -17.3%
- **Avg Annual:** -3.5%
- **2030 Rent:** $1,298

### Bull Case: Strong Economy, High Absorption

**Assumptions:**
- Employment Growth: 2.5% annual
- Migration: 150,000 people/year
- Construction Pipeline: 40% below current
- Vacancy Path: Rapid improvement (11.5% → 6.0%)

**Projections:**

| Year | Rent | YoY Growth | Vacancy |
|------|------|------------|---------|
| 2026 | $1,470 | -6.33% | 11.5% |
| 2027 | $1,408 | -4.20% | 10.0% |
| 2028 | $1,373 | -2.47% | 8.5% |
| 2029 | $1,357 | -1.16% | 7.0% |
| 2030 | $1,345 | -0.89% | 6.0% |

**5-Year Summary:**
- **Total Change:** -14.3%
- **Avg Annual:** -2.9%
- **2030 Rent:** $1,345

### Scenario Comparison

| Scenario | 2030 Rent | 5-Yr Growth | Avg Annual |
|----------|-----------|-------------|------------|
| **Bear Case** | $1,270 | -19.0% | -3.8% |
| **Base Case** | $1,298 | -17.3% | -3.5% |
| **Bull Case** | $1,345 | -14.3% | -2.9% |

**Range:** $75 spread between bear and bull ($1,270 - $1,345)

---

## Key Findings & Conclusions

### Market Dynamics

1. **Oversupply Correction Phase (2026-2028)**
   - Current vacancy (12.4%) significantly above equilibrium (5%)
   - ~30K excess vacant units must be absorbed
   - Construction pipeline (22K units) = 1.6x annual absorption
   - Continued negative rent growth inevitable until supply absorbed

2. **Stabilization Period (2029-2030)**
   - Vacancy approaches equilibrium levels
   - Rent declines moderate significantly
   - Market prepares for next growth cycle
   - New construction likely to slow substantially

3. **Structural Drivers**
   - **Vacancy** is dominant rent growth driver (r=-0.89)
   - Strong inverse relationship: -1 pp vacancy ≈ +1% rent growth
   - Current 7.2 pp excess vacancy suggests multi-year absorption period
   - Employment growth (+1.6% YoY) provides demand foundation

4. **Migration Impact**
   - 2021 peak migration (146K people) drove 89% of COVID-era absorption
   - Base case (100K/yr) supports ~81% of current absorption
   - Migration remains key demand driver but moderating from peak
   - Continued sunbelt appeal vs. affordability concerns

### Risk Factors

**Downside Risks (Bear Case):**
- Economic recession reducing employment growth
- Higher interest rates suppressing homebuyer demand (limiting renter conversion)
- Continued elevated construction pipeline
- Faster-than-expected migration slowdown
- Increased single-family rental supply competing for renters

**Upside Risks (Bull Case):**
- Stronger employment growth from tech sector expansion
- Higher migration from continued WFH flexibility
- Faster construction slowdown reducing new supply
- Build-to-rent converting to condos reducing rental supply
- Interest rate cuts accelerating renter-to-owner conversion (reducing supply faster)

### Model Confidence

**High Confidence:**
- Near-term (2026-2027): Strong correlation between current vacancy and rent trajectory
- Direction: All scenarios show rent decline given current supply/demand imbalance
- Vacancy normalization timeline: 3-5 years based on absorption patterns

**Moderate Confidence:**
- Mid-term (2028-2029): Dependent on construction pipeline and migration trends
- Magnitude of decline: -14% to -19% range reflects scenario uncertainty
- Vacancy recovery pace: Absorption rates may vary with economic conditions

**Lower Confidence:**
- Long-term (2030+): Policy changes, development trends, economic shocks
- Construction behavior: Developer response to market signals
- Migration sustainability: Remote work policies, relative affordability

---

## Investment Implications

### Market Timing

**Not Recommended (2026-2027):**
- Continued rent decline phase
- Elevated vacancy absorption period
- Better entry points likely ahead
- Value-add strategies challenged by negative rent growth

**Potential Entry Points (2028-2029):**
- Stabilization phase beginning
- Vacancy approaching equilibrium
- Rent declines moderating
- Distressed asset opportunities from oversupplied assets

**Recommended (2029-2030):**
- Market stabilized at new equilibrium
- Next growth cycle preparation
- Development pipeline cleared
- Rent growth returning to positive territory

### Asset Strategy

**Value-Add:**
- **Timing:** Wait for stabilization (2029+)
- **Risk:** Negative rent growth limits value creation
- **Opportunity:** Distressed acquisitions in 2028-2029

**Core/Core-Plus:**
- **Focus:** High-quality assets with strong retention
- **Strategy:** Defensive positioning through downturn
- **Target:** Submarkets with strongest employment/migration

**Development:**
- **Recommendation:** Avoid new starts 2026-2028
- **Pipeline:** Reassess 2029-2030 for 2031+ delivery
- **Focus:** Differentiated product capturing next cycle

### Underwriting Guidance

**Rent Growth Assumptions:**
- 2026-2027: -4% to -6% annual
- 2028-2029: -2% to -3% annual
- 2030+: -1% to +1% annual
- **Avoid:** Pro forma rent growth assumptions until 2030+

**Vacancy Assumptions:**
- Use economic vacancy vs. physical vacancy
- Budget 10-12% effective vacancy 2026-2028
- Budget 8-10% effective vacancy 2029-2030
- **Critical:** Include concession costs in underwriting

**Exit Assumptions:**
- **5-Year Hold:** Cap rate expansion likely
- **7-Year Hold:** Better positioning for stabilized exit
- **10-Year Hold:** Captures next growth cycle
- **Conservative:** +50-100 bps cap rate expansion 2026-2029

---

## Conclusion

Phoenix multifamily market is entering a **multi-year rent correction phase** driven by elevated vacancy (12.4%) and substantial construction pipeline (22K units). All scenarios project **negative rent growth through 2026-2028** before gradual stabilization in 2029-2030.

**Base case projects:**
- **-17.3% total rent decline** (2025-2030)
- **-3.5% average annual** growth
- **2030 rent: $1,298** (vs. $1,569 in 2025)

Market fundamentals remain intact with **1.6% employment growth** and **moderate migration** (+100K/yr), but near-term supply/demand imbalance requires **3-5 year absorption period** before returning to balanced growth.

**Strategic recommendation:** Patient capital positioned for **2028-2029 entry** will capture best risk-adjusted returns as market transitions from correction to next growth cycle.

---

## Appendix: Model Validation

### Performance Metrics (8-Quarter Validation)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| SARIMA | $22.62 | - | - |
| XGBoost | $47.89 | - | - |
| **Ensemble** | **$12.16** | - | - |

**Conclusion:** Ensemble significantly outperforms individual models

### Data Sources

- **CoStar:** Quarterly market data (Q1 2000 - Q3 2025)
- **FRED:** Employment data (1990 - 2025)
- **US Census ACS:** Demographics (2023)
- **IRS Migration:** Net migration flows (2021)

### Methodology Notes

- 103 quarters of historical data (25+ years)
- SARIMA(1,1,1)x(1,1,1,4) with first differencing
- XGBoost with 5 features (vacancy, construction, employment, unemployment, COVID)
- 60/40 ensemble weighting (SARIMA/XGBoost)
- 8-quarter out-of-sample validation
- Scenarios based on economic drivers and historical precedent

---

**Report Prepared by:** Claude Code SPARC Agent
**Model:** SARIMA/XGBoost Ensemble
**Validation MAE:** $12.16
**Confidence Level:** High (near-term), Moderate (mid-term), Lower (long-term)
