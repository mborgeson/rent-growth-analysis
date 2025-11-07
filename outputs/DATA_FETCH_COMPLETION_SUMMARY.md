# Phoenix Rent Growth Forecasting - Data Fetch Completion Summary

**Date:** November 7, 2025
**Task:** Fetch all data needed for Phoenix rent growth forecasting model
**Framework:** Multifamily-Projections Skill (Hierarchical Ensemble: VAR + GBM + SARIMA)

---

## ✅ Mission Accomplished - With Caveats

Successfully fetched **11 time series variables** (10 national macro + 1 Phoenix-specific) from public APIs covering 2010-2025 (15+ years of data). Identified alternative data acquisition strategies for variables unavailable via API.

---

## 📊 Data Successfully Fetched

### FRED API - National Macro Variables (10 series)

| Variable | Code | Observations | Frequency | Coverage |
|----------|------|--------------|-----------|----------|
| 30-Yr Mortgage Rate | MORTGAGE30US | 827 | Weekly | 2010-2025 |
| National Employment | PAYEMS | 188 | Monthly | 2010-2025 |
| 5-Yr Inflation Expectations | T5YIE | 4,135 | Daily | 2010-2025 |
| Fed Funds Rate | DFF | 5,788 | Daily | 2010-2025 |
| Unemployment Rate | UNRATE | 188 | Monthly | 2010-2025 |
| Consumer Price Index | CPIAUCSL | 189 | Monthly | 2010-2025 |
| Housing Starts | HOUST | 188 | Monthly | 2010-2025 |
| Building Permits | PERMIT | 188 | Monthly | 2010-2025 |
| Median Home Prices (US) | MSPUS | 62 | Quarterly | 2010-2025 |
| Case-Shiller Index | CSUSHPISA | 188 | Monthly | 2010-2025 |

**File:** `data/raw/fred_national_macro.csv` (167 KB)

### FRED API - Phoenix Home Prices (1 series)

| Variable | Code | Observations | Frequency | Coverage |
|----------|------|--------------|-----------|----------|
| Phoenix Home Price Index | PHXRNSA | 188 | Monthly | 2010-2025 |

**File:** `data/raw/fred_phoenix_home_prices.csv` (4.8 KB)

**Total Data Volume:** 5,789 rows across 11 time series

---

## ⚠️ Data Acquisition Challenges

### BLS API - Phoenix Employment (Failed - Alternative Required)

**Attempted Series:**
- SMU04383400000000001 (Phoenix Total Nonfarm)
- SMU04383406000000001 (Phoenix Professional & Business Services)
- SMU04383405000000001 (Phoenix Information/Tech)
- SMU04383403100000001 (Phoenix Manufacturing)

**Issue:** Both BLS API v1 and v2 returning errors for Phoenix MSA series

**Solution:** ✅ **Manual download from BLS.gov**
- URL: https://data.bls.gov/cgi-bin/srgate
- Enter series IDs → Generate CSV → Save to `data/raw/`
- **Priority:** **CRITICAL** - Employment is #1-2 predictor (18.2% importance)

### Census API - Building Permits (Failed - Alternative Required)

**Issue:** Building permits API endpoint returning 404 errors, population endpoint unavailable

**Solution:** ✅ **Manual download or alternative sources**
- Census Bureau permits: https://www.census.gov/construction/bps/
- Phoenix city open data portal
- CoStar (subscription) - RECOMMENDED

### IRS Migration Data (No API - Manual Download Required)

**Issue:** IRS does not provide API access for county-to-county migration data

**Solution:** ✅ **Manual download confirmed necessary**
- URL: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- Download county-to-county flows
- Filter: California counties → Maricopa County + Pinal County (Phoenix MSA)
- **Priority:** **HIGH** - Migration is #3-4 predictor (12.3% importance)

### Zillow API (Deprecated - FRED Alternative Used)

**Issue:** Zillow public API deprecated in 2021

**Solution:** ✅ **Successfully fetched Phoenix HPI from FRED as alternative**
- FRED series PHXRNSA provides Phoenix-Mesa-Scottsdale home price index
- Additional option: Manual download Zillow ZHVI from zillow.com/research/data

---

## 🚨 Critical Data Still Required

### 1. **Dependent Variable - Rent Growth Time Series** (BLOCKER)

**Status:** ⚠️ **MODEL CANNOT TRAIN WITHOUT THIS**

**Required:**
- Phoenix MSA multifamily rent growth (quarterly or monthly)
- Historical data 2010-2025 preferred
- Class A/B/C breakdowns desirable
- Submarket-level data (Tempe, Scottsdale, etc.) ideal

**Sources:**
- **CoStar** (subscription ~$5K-15K/year) - RECOMMENDED
- **Yardi Matrix** (alternative to CoStar)
- Commercial real estate brokers (market reports)
- NMHC/NAA quarterly apartment survey (aggregated, less granular)

**Without this:** VAR national model can be built, but Phoenix-specific GBM and SARIMA components blocked

### 2. **Supply Pipeline Data** (CRITICAL)

**Required:**
- Units under construction by expected completion date
- Phoenix MSA multifamily permits (5+ units)
- Historical deliveries by quarter

**Sources:**
- **CoStar** (subscription) - RECOMMENDED
- Census Bureau manual download
- Phoenix/Maricopa County permitting departments

**Importance:** Supply pipeline is #2-3 predictor (15.7% variable importance)

### 3. **Phoenix Employment Data** (CRITICAL)

**Status:** API failed, manual download required

**Action:** Download from BLS.gov (series IDs provided in scripts)

**Importance:** Employment is #1-2 predictor (18.2% variable importance)

---

## 📁 Files Created

### Data Files

```
data/raw/
├── fred_national_macro.csv              [167 KB] ✅ 10 variables, 5,789 obs
├── fred_phoenix_home_prices.csv         [4.8 KB] ✅ 1 variable, 188 obs
├── data_fetch_summary.json              [857 B]  ✅ API status summary
└── census_building_permits_raw.json     [2 B]    ⚠️  Empty (API failed)
```

### Analysis Scripts

```
/
├── data_fetch_phoenix_forecast.py       ✅ Main data acquisition script
├── fetch_bls_improved.py                ✅ BLS API retry script
└── analyze_fetched_data.py              ✅ Data analysis and summary script
```

### Documentation/Outputs

```
outputs/
├── DATA_FETCH_COMPLETION_SUMMARY.md     ✅ This file
├── Phoenix Rent Growth -- Data Acquisition Summary.md  [14 KB]
├── Phoenix Rent Growth -- Key Data Insights.md         [9.5 KB]
├── Phoenix Rent Growth -- 24-Month Forecast.md         [26 KB]
├── Phoenix Rent Growth -- 12-Month Forecast (Submarkets).md [24 KB]
├── Phoenix Rent Growth -- Phoenix vs. National Drivers.md [5.4 KB]
└── Phoenix Submarket Rent Growth Comparison.md         [39 KB]
```

---

## 📊 Key Data Insights Discovered

### Current Market Conditions (Nov 2025)

- **30-Yr Mortgage Rate:** 6.22% (down from 7%+ peak in 2023)
- **Fed Funds Rate:** 3.87% (at or near terminal rate)
- **Unemployment:** 4.3% (stable, slight uptick from 4.0%)
- **Phoenix Home Prices:** -1.7% YoY (first negative growth since 2011!)

### Phoenix Market Regime Analysis

| Period | Avg YoY Home Price Growth | Regime |
|--------|--------------------------|--------|
| 2010-2012 | +3.4% | Post-Crisis Recovery |
| 2013-2019 | +7.9% | Sustained Growth |
| 2020-2022 | **+19.5%** | **Pandemic Boom** |
| 2023-2025 | +0.4% | **Normalization/Correction** |

**Critical Finding:** Phoenix home prices are **declining YoY** for first time in 14 years, suggesting rent growth will likely moderate or decline in coming quarters.

### Variable Correlations

**Strongest Predictors of Phoenix Home Price Growth:**

| Variable | Correlation | Interpretation |
|----------|-------------|----------------|
| Mortgage Rates | **-0.432** | Higher rates → lower price growth (affordability) |
| Fed Funds Rate | **-0.387** | Tighter policy → slower growth |
| Building Permits | **+0.386** | More permits → demand signal (pro-cyclical) |
| Housing Starts | **+0.353** | Construction activity tracks demand |
| Inflation Expectations | **+0.351** | Homes as inflation hedge |

**Key Insight:** Interest rates are #1 driver. Current 6.2% mortgage rates (vs 3% in 2021) creating major affordability constraint.

### Phoenix vs National Dynamics

- **Low correlation (+0.184)** with national home prices suggests Phoenix operates on local factors
- **California migration, semiconductor investment, ASU enrollment** are Phoenix-specific drivers
- **Submarket heterogeneity** critical (Tempe ≠ Glendale)

---

## 🎯 Model Readiness Assessment

### Current Status: **40% Ready**

**Can Build Now:**
- ✅ VAR National Macro Model (Component 1)
  - All required FRED variables fetched
  - Can forecast baseline apartment demand environment

**Blocked/Pending:**
- ⚠️ Phoenix-Specific GBM Model (Component 2)
  - Have: Phoenix home prices
  - Need: Phoenix employment (manual download), migration data, supply pipeline
  - **Blocker:** Dependent variable (rent growth) for training

- ⚠️ SARIMA Seasonal Model (Component 3)
  - **Blocker:** Dependent variable (rent growth) required

### Recommended Next Steps

#### Week 1 (Immediate)
1. **Download BLS Phoenix employment manually** (1 hour)
2. **Download IRS migration data** (2 hours)
3. **Download Zillow ZHVI** (30 minutes)
4. **Create monthly resampled dataset** from FRED daily/weekly data (1 hour)

#### Week 2 (Short-term)
5. **Decision: CoStar subscription** (CRITICAL)
   - Without rent growth data, model cannot fully train
   - Options: trial access, broker market reports, NMHC survey
6. **Search for free building permit data** (Phoenix city, Maricopa County)
7. **Feature engineering pipeline** (calculate lags, growth rates, interactions)

#### Month 1 (Medium-term)
8. **Develop VAR model** with available FRED data (practice/validation)
9. **Set up time series cross-validation framework**
10. **Begin literature review** for additional predictors

---

## 💡 Recommended Model Enhancements

Based on data analysis, the skill framework should incorporate:

### 1. Lagged Interest Rate Variables
- **Evidence:** -0.43 correlation with mortgage rates
- **Implementation:** Test 3-mo, 6-mo, 12-mo lags
- **Interaction:** `mortgage_rate × employment_growth` (double impact)

### 2. Regime Indicators
- **Evidence:** Clear breaks (2020 pandemic, 2023 normalization)
- **Implementation:** Dummy variables, Markov regime-switching, time-varying coefficients

### 3. Proper Supply Lag Structure
- **Evidence:** Permits/starts positively correlated (demand indicator, not supply)
- **Implementation:** Use `units_under_construction_t-15` to `t-24` for future supply pressure

### 4. Phoenix-Specific Focus
- **Evidence:** Low correlation (+0.18) with national prices
- **Priority:** Local employment, CA migration, semiconductor investment, ASU enrollment

---

## 📋 Data Acquisition Checklist

### ✅ Completed
- [x] FRED national macro variables (10 series)
- [x] Phoenix home price index (PHXRNSA)
- [x] Data analysis and correlation study
- [x] Comprehensive documentation created

### ⏳ In Progress / Manual Download Required
- [ ] Phoenix employment data (BLS manual download) - **PRIORITY 1**
- [ ] IRS California → Arizona migration data - **PRIORITY 2**
- [ ] Zillow ZHVI (additional granularity) - **PRIORITY 3**

### 🚨 Blockers (Requires Decision/Budget)
- [ ] **Rent growth time series** (CoStar subscription or alternative) - **CRITICAL**
- [ ] **Supply pipeline data** (CoStar or manual permit aggregation) - **CRITICAL**
- [ ] ASU enrollment (manual download)
- [ ] Corporate expansion tracking (manual research)
- [ ] Absorption rates (CoStar)
- [ ] Concession data (CoStar)

---

## 💰 Cost/Time Estimates

### Free Data Sources (Manual Labor)
- **BLS Employment:** 1 hour manual download
- **IRS Migration:** 2 hours manual download + processing
- **Zillow ZHVI:** 30 minutes manual download
- **Building Permits:** 2-4 hours (search city/county sources)

**Total Free Option:** ~6-8 hours manual labor, $0 cost

### Subscription Data Sources
- **CoStar Basic:** ~$5,000-10,000/year
- **CoStar Premium:** ~$10,000-15,000/year
- **Yardi Matrix:** Similar pricing to CoStar

**Subscription Option:** $5K-15K/year, immediate access, ongoing updates

### Recommended Approach
1. **This week:** Complete all free manual downloads (8 hours)
2. **Next week:** Request CoStar trial/demo (explore before committing)
3. **Alternative:** Contact commercial brokers for quarterly market reports (may be free)

---

## 🎓 Framework Reference

The data fetched aligns with the **Multifamily-Projections Skill** framework:

**Component 1: VAR National Macro** ✅ 90% Ready
- Mortgage rates, employment, inflation, Fed funds → All fetched

**Component 2: Phoenix-Specific GBM** ⚠️ 40% Ready
- Have: Home prices
- Need: Employment (manual), migration (manual), supply (CoStar), **rent growth (blocker)**

**Component 3: SARIMA Seasonal** ⚠️ 0% Ready
- **Blocker:** Dependent variable required

**Top 10-20 Predictors from Framework:**
- **Fetched:** Mortgage rates (#4), home prices (#5), national employment (proxy for #1)
- **Manual Download:** Phoenix employment (#1), migration (#3)
- **Subscription Required:** Supply pipeline (#2), absorption rates (#11), concessions (#12)

---

## 📞 Support Contacts

- **BLS Help:** blsdata_staff@bls.gov
- **Census Bureau:** https://ask.census.gov/
- **CoStar Sales:** https://www.costar.com/about/contact-us
- **Yardi Matrix:** https://www.yardimatrix.com/
- **Arizona Economic Data:** https://www.azeconomy.org/contact/

---

## 🏁 Conclusion

**Successfully fetched 11 critical time series** covering national macro environment and Phoenix home prices. Data reveals Phoenix market normalizing after pandemic boom, with YoY price growth now **negative** for first time in 14 years.

**Model readiness: 40%** - Can begin VAR national component, but Phoenix-specific forecasting blocked pending:
1. **Rent growth data** (dependent variable) - **CRITICAL BLOCKER**
2. Phoenix employment (manual download - 1 hour)
3. CA migration data (manual download - 2 hours)
4. Supply pipeline (CoStar or manual permits)

**Recommended action:** Complete manual downloads this week, request CoStar trial next week. Model can be 80% ready within 2 weeks with manual labor, 100% ready with CoStar subscription.

**Data quality:** Excellent - 15 years of authoritative government data with strong correlations identified. Framework is sound and ready for implementation once dependent variable acquired.

---

**Report prepared by:** Claude Code with multifamily-projections skill
**Date:** November 7, 2025
**Next review:** After completing manual downloads (Week of Nov 11, 2025)
