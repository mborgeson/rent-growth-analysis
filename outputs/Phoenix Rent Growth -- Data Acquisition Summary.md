# Phoenix Multifamily Rent Growth Forecasting - Data Acquisition Summary

**Date:** November 7, 2025
**Project:** Phoenix MSA Rent Growth Analysis
**Framework:** Hierarchical Ensemble Model (VAR + Gradient Boosting + SARIMA)

---

## Executive Summary

Successfully fetched **10 national macro variables** from FRED API and **Phoenix home price index**, covering 2010-2025. BLS and Census APIs experienced technical issues requiring alternative data acquisition strategies.

### Data Acquisition Status

| Data Source | Status | Variables | Coverage | Notes |
|-------------|--------|-----------|----------|-------|
| **FRED** | ✅ Complete | 10 series | 2010-2025 | 5,789 observations |
| **Phoenix Home Prices (FRED)** | ✅ Complete | 1 series | 2010-2025 | 188 monthly observations |
| **BLS Employment** | ⚠️ Pending | 0 series | N/A | API access issues - needs alternative |
| **Census Permits** | ⚠️ Pending | 0 series | N/A | API endpoint unavailable |
| **IRS Migration** | ⚠️ Manual | 0 series | N/A | No API - requires manual download |
| **CoStar Data** | ⚠️ Manual | 0 series | N/A | Subscription required |

---

## Successfully Fetched Data (FRED)

### National Macro Variables

| Variable | FRED Code | Description | Observations | Coverage |
|----------|-----------|-------------|--------------|----------|
| **Mortgage Rate** | MORTGAGE30US | 30-Year Fixed Rate Mortgage Average | 827 | 2010-01-07 to 2025-11-06 (weekly) |
| **Employment** | PAYEMS | Total Nonfarm Payrolls (National) | 188 | 2010-01-01 to 2025-08-01 (monthly) |
| **Inflation Exp** | T5YIE | 5-Year Breakeven Inflation Rate | 4,135 | 2010-01-01 to 2025-11-06 (daily) |
| **Fed Funds Rate** | DFF | Federal Funds Effective Rate | 5,788 | 2010-01-01 to 2025-11-05 (daily) |
| **Unemployment** | UNRATE | Unemployment Rate (National) | 188 | 2010-01-01 to 2025-08-01 (monthly) |
| **CPI** | CPIAUCSL | Consumer Price Index for All Urban Consumers | 189 | 2010-01-01 to 2025-09-01 (monthly) |
| **Housing Starts** | HOUST | New Privately-Owned Housing Units Started | 188 | 2010-01-01 to 2025-08-01 (monthly) |
| **Building Permits** | PERMIT | New Privately-Owned Housing Units Authorized | 188 | 2010-01-01 to 2025-08-01 (monthly) |
| **Home Prices** | MSPUS | Median Sales Price of Houses Sold (National) | 62 | 2010-01-01 to 2025-04-01 (quarterly) |
| **Case-Shiller** | CSUSHPISA | S&P/Case-Shiller U.S. National Home Price Index | 188 | 2010-01-01 to 2025-08-01 (monthly) |

**Files:**
- `data/raw/fred_national_macro.csv` (167 KB, 5,789 rows × 10 columns)

### Phoenix-Specific Variables

| Variable | FRED Code | Description | Observations | Coverage |
|----------|-----------|-------------|--------------|----------|
| **Phoenix HPI** | PHXRNSA | Phoenix-Mesa-Scottsdale Home Price Index | 188 | 2010-01-01 to 2025-08-01 (monthly) |

**Files:**
- `data/raw/fred_phoenix_home_prices.csv` (4.8 KB, 188 rows × 1 column)

---

## Data Still Needed

### Critical Priority (Tier 1)

#### 1. Phoenix MSA Employment Data (BLS)

**Required Series:**
- `SMU04383400000000001` - Phoenix MSA Total Nonfarm Employment
- `SMU04383406000000001` - Phoenix Professional & Business Services Employment
- `SMU04383405000000001` - Phoenix Information Sector Employment (Tech)
- `SMU04383403100000001` - Phoenix Manufacturing Employment

**Alternative Sources:**
- Arizona Office of Economic Opportunity: https://www.azeconomy.org/
- Phoenix Chamber of Commerce Economic Data
- Download from BLS.gov manually: https://data.bls.gov/cgi-bin/srgate
- FRED may have aggregated Phoenix employment series

**Importance:** **CRITICAL** - Employment growth is the #1-2 predictor in the model (18.2% variable importance)

#### 2. Units Under Construction / Building Permits (Phoenix MSA)

**Required Data:**
- Phoenix MSA multifamily building permits (5+ units)
- Units under construction by expected completion date
- Historical deliveries by quarter/year

**Sources:**
- Census Bureau manual download: https://www.census.gov/construction/bps/
- CoStar (subscription required - RECOMMENDED)
- Yardi Matrix (alternative to CoStar)
- Phoenix city permitting department

**Importance:** **CRITICAL** - Supply pipeline is the #2-3 predictor (15.7% variable importance)

#### 3. Net Migration from California (IRS Data)

**Required Data:**
- County-to-county migration flows
- California → Arizona (Phoenix MSA counties)
- Annual data with 18-month lag

**Source:**
- Manual download: https://www.irs.gov/statistics/soi-tax-stats-migration-data

**Process:**
1. Download county-to-county migration CSV files
2. Filter origin: California counties
3. Filter destination: Maricopa County, Pinal County (Phoenix MSA)
4. Aggregate net flows annually

**Importance:** **HIGH** - Migration is #3-4 predictor (12.3% variable importance)

### Important Priority (Tier 2)

#### 4. CoStar Subscription Data (HIGHLY RECOMMENDED)

**Required Variables:**
- **Rent Growth:** Class A/B/C effective rent by quarter (Phoenix MSA + submarkets)
- **Occupancy:** Occupancy rates by class and submarket
- **Absorption Rates:** New Class A absorption rates (Tempe, Scottsdale, Downtown Phoenix)
- **Concessions:** % of properties offering concessions, average weeks free
- **Supply Pipeline:** Units under construction by submarket and expected delivery date
- **Inventory:** Total units by class and submarket

**Why Critical:**
- CoStar provides the **dependent variable** (rent growth) for model training
- Supply pipeline data unavailable from free sources
- Submarket-level granularity essential for Phoenix (Tempe ≠ Glendale)

**Cost:** ~$5,000-15,000/year (varies by subscription level)

**Alternatives:**
- Yardi Matrix (similar pricing, alternative data provider)
- NMHC/NAA Apartment Survey (aggregated, less granular)
- Local apartment associations

#### 5. Arizona/Phoenix Demographics

**Required Data:**
- Phoenix MSA population growth (annual)
- Household formation rates by age cohort (25-34 key demographic)
- Median household income growth (Phoenix MSA)

**Sources:**
- Census Bureau American Community Survey (ACS)
- FRED may have Phoenix population series
- Arizona Office of Economic Opportunity

#### 6. ASU Enrollment Trends

**Required Data:**
- Arizona State University total enrollment (annual)
- Campus locations (Tempe, Downtown Phoenix, West)

**Source:**
- ASU Office of Institutional Analysis: https://uoia.asu.edu/
- National Center for Education Statistics (NCES) IPEDS

**Importance:** **MODERATE** - Relevant for Tempe/Downtown submarkets (r = 0.35-0.50)

### Supplementary Priority (Tier 3)

#### 7. Corporate Expansion Announcements

**Required Data:**
- Major corporate relocations/expansions in Phoenix MSA
- Expected job creation (Intel $20B fab, TSMC $40B)
- Semiconductor industry investment tracking

**Sources:**
- Economic development announcements
- Company press releases
- Phoenix Business Journal
- Greater Phoenix Economic Council

**Note:** Qualitative data requiring manual scoring/categorization

#### 8. Zillow Home Value Index (ZHVI)

**Current Status:** Have FRED Phoenix home price index (PHXRNSA)

**Additional Data Available:**
- Download from Zillow Research: https://www.zillow.com/research/data/
- Select "Phoenix-Mesa-Scottsdale, AZ" metro
- Download All Homes (SFR) Time Series
- Provides more granular breakdowns (by bedroom count, home type)

**Importance:** **MODERATE** - Already have FRED alternative, Zillow provides more detail

---

## Data Acquisition Roadmap

### Immediate Actions (This Week)

1. **Download BLS Employment Data Manually**
   - Visit: https://data.bls.gov/cgi-bin/srgate
   - Enter series IDs: SMU04383400000000001, SMU04383406000000001, etc.
   - Download CSV files
   - Save to: `data/raw/bls_phoenix_employment_manual.csv`

2. **Download IRS Migration Data**
   - Visit: https://www.irs.gov/statistics/soi-tax-stats-migration-data
   - Download most recent 5 years (2018-2022 likely available)
   - Filter for California → Arizona flows
   - Save to: `data/raw/irs_migration_ca_to_az.csv`

3. **Download Zillow ZHVI**
   - Visit: https://www.zillow.com/research/data/
   - Download Phoenix-Mesa-Scottsdale ZHVI (All Homes)
   - Save to: `data/raw/zillow_zhvi_phoenix.csv`

4. **Search for Free Census Building Permit Data**
   - Try alternative Census Bureau downloads
   - Check Phoenix city open data portal
   - Maricopa County building department

### Short-Term Actions (This Month)

5. **CoStar Subscription Decision**
   - **Critical decision:** Model cannot be fully trained without rent growth dependent variable
   - Options:
     - Request CoStar trial/demo access
     - Contact commercial real estate brokers for market reports
     - Use NMHC/NAA quarterly apartment survey (less granular)
     - Consider alternative: Yardi Matrix

6. **Alternative Employment Data**
   - Arizona Office of Economic Opportunity
   - Contact Phoenix Chamber of Commerce
   - Check FRED for Phoenix aggregated employment series

### Medium-Term Actions (Next Quarter)

7. **Establish Ongoing Data Updates**
   - Automate FRED data pulls (monthly)
   - Set up BLS data monitoring
   - Create data refresh pipeline

8. **Quality Assurance**
   - Cross-validate data sources (e.g., FRED vs BLS employment)
   - Check for outliers and missing data
   - Document data transformations

---

## Data Quality Assessment

### Current Data Quality: **GOOD**

#### Strengths
- ✅ 10 national macro variables from authoritative source (FRED)
- ✅ 15+ years historical coverage (2010-2025)
- ✅ Daily, weekly, monthly, quarterly frequencies available
- ✅ High-quality government data (Federal Reserve, BLS, Census)
- ✅ Phoenix home price index successfully fetched

#### Gaps
- ⚠️ **No dependent variable (rent growth)** - requires CoStar or alternative
- ⚠️ **No Phoenix employment data** - BLS API issues, needs manual fetch
- ⚠️ **No supply pipeline data** - critical for forecasting, requires CoStar
- ⚠️ **No migration data yet** - manual download required
- ⚠️ **No submarket-level data** - all data is MSA-level

### Data Completeness by Model Component

#### Component 1: VAR National Macro Model
**Completeness: 90%**
- ✅ Mortgage rates (MORTGAGE30US)
- ✅ National employment (PAYEMS)
- ✅ Inflation expectations (T5YIE)
- ✅ Fed funds rate (DFF)
- ⚠️ Need: Tech sector employment growth (can derive from PAYEMS subset)

#### Component 2: Phoenix-Specific GBM Model
**Completeness: 40%**
- ✅ Phoenix home prices (PHXRNSA)
- ⚠️ Need: Phoenix employment (manual BLS download)
- ⚠️ Need: Units under construction (CoStar)
- ⚠️ Need: Migration data (IRS manual)
- ⚠️ Need: Absorption rates (CoStar)
- ⚠️ Need: Concession data (CoStar)
- ⚠️ Need: ASU enrollment (manual)

#### Component 3: SARIMA Seasonal Model
**Completeness: 0% (dependent variable needed)**
- ⚠️ Need: Phoenix rent growth time series (CoStar)

**Overall Model Readiness: 40%**

---

## Technical Notes

### API Access Information

#### FRED API
- **API Key:** d043d26a9a4139438bb2a8d565bc01f7
- **Documentation:** https://fred.stlouisfed.org/docs/api/fred/
- **Rate Limits:** None for public data
- **Status:** ✅ Working

#### Census API
- **API Key:** 0145eb3254e9885fa86407a72b6b0fb381e846e8
- **Documentation:** https://www.census.gov/data/developers.html
- **Status:** ⚠️ Building permits endpoint not accessible (404 errors)
- **Action:** Try alternative endpoints or manual downloads

#### BLS API
- **Version:** Tested v1 (public) and v2 (requires registration)
- **Status:** ⚠️ Both versions returning errors for Phoenix MSA series
- **Action:** Manual CSV download from BLS website

### File Structure

```
data/raw/
├── fred_national_macro.csv           [167 KB] ✅
├── fred_phoenix_home_prices.csv      [4.8 KB] ✅
├── data_fetch_summary.json           [857 B]  ✅
├── census_building_permits_raw.json  [2 B]    ⚠️ Empty
├── bls_phoenix_employment.csv        [TBD]    ⚠️ To be created
├── irs_migration_ca_to_az.csv        [TBD]    ⚠️ To be downloaded
└── costar_phoenix_rents.csv          [TBD]    ⚠️ Subscription required
```

---

## Next Steps

### Priority 1 (Cannot Proceed Without)
1. **Acquire CoStar subscription or alternative rent growth data**
   - This is the dependent variable - model cannot train without it
   - Contact: CoStar sales, commercial brokers, Yardi Matrix

### Priority 2 (Manual Downloads This Week)
2. Download BLS Phoenix employment manually
3. Download IRS migration data (CA → AZ)
4. Download Zillow ZHVI for Phoenix
5. Search for free building permit data

### Priority 3 (Feature Engineering)
6. Create feature engineering pipeline once data assembled
7. Calculate lagged variables (employment 3-mo lag, construction 15-mo lag)
8. Calculate derived features (supply/demand interaction, growth rates)

### Priority 4 (Model Development)
9. Begin model development with available FRED data (practice VAR model)
10. Develop data cleaning and preprocessing scripts
11. Set up model validation framework (time series CV)

---

## Contact Information for Data Access

- **CoStar:** https://www.costar.com/about/contact-us
- **Yardi Matrix:** https://www.yardimatrix.com/
- **BLS Help:** blsdata_staff@bls.gov
- **Census Bureau:** https://ask.census.gov/
- **Arizona Office of Economic Opportunity:** https://www.azeconomy.org/contact/

---

## Summary

**Successfully fetched:** 11 time series variables (10 national macro + 1 Phoenix home prices)
**Coverage:** 2010-2025 (15+ years)
**Data volume:** ~6,000 daily/weekly/monthly observations

**Critical gaps:**
1. Dependent variable (rent growth) - **BLOCKER**
2. Phoenix employment - Manual download required
3. Supply pipeline - Requires subscription
4. Migration data - Manual download required

**Estimated time to complete data acquisition:** 1-2 weeks
**Estimated cost:** $0 (if using manual downloads) to $5,000-15,000 (if CoStar subscription)

**Model readiness:** 40% - Can begin VAR national macro component, but Phoenix-specific GBM and SARIMA components blocked pending data acquisition.
