# Phoenix MSA FRED Series Discovery Results

**Discovery Date:** 2025-01-07
**Total Series Found:** 718 unique Phoenix MSA series

## Top 10 Most Popular Phoenix Series (by FRED popularity metric)

| Rank | Series ID | Description | Frequency | Popularity | Relevance to Rent Growth |
|------|-----------|-------------|-----------|------------|-------------------------|
| 1 | **ACTLISCOU38060** | Housing Inventory: Active Listing Count | Monthly | 54 | ⭐⭐⭐ HIGH - Supply indicator |
| 2 | **ATNHPIUS38060Q** | All-Transactions House Price Index | Quarterly | 49 | ⭐⭐⭐ HIGH - Affordability driver |
| 3 | **PHXRNSA** | S&P Case-Shiller Home Price Index | Monthly | 46 | ⭐⭐⭐ HIGH - Already in model |
| 4 | **PHXPOP** | Resident Population | Annual | 40 | ⭐⭐ MEDIUM - Demand driver |
| 5 | **APUS48A74714** | Average Gasoline Price | Monthly | 36 | ⭐ LOW - Cost of living proxy |
| 6 | **MEDDAYONMAR38060** | Median Days on Market | Monthly | 32 | ⭐⭐⭐ HIGH - Market velocity |
| 7 | **NGMP38060** | Total Gross Domestic Product | Annual | 24 | ⭐⭐ MEDIUM - Economic health |
| 8 | **PHXRSA** | S&P Case-Shiller (Seasonally Adjusted) | Monthly | 23 | ⭐⭐⭐ HIGH - Already in model |
| 9 | **MEDLISPRI38060** | Median Listing Price | Monthly | 22 | ⭐⭐⭐ HIGH - Price trend |
| 10 | **PHOE004UR** | Unemployment Rate | Monthly | 21 | ⭐⭐⭐ HIGH - Labor market |

## Newly Discovered High-Value Series (Not in Original Framework)

### Housing Market Indicators

1. **ACTLISCOU38060** - Housing Inventory: Active Listing Count
   - **Frequency:** Monthly
   - **Use Case:** Supply pressure indicator; low inventory → rental demand
   - **Theory:** Tight for-sale inventory drives renters into market
   - **Expected Correlation:** r = -0.55 to -0.65 (negative: low inventory → higher rents)

2. **MEDDAYONMAR38060** - Median Days on Market
   - **Frequency:** Monthly
   - **Use Case:** Market velocity; fast sales → strong demand
   - **Theory:** Rapid home sales indicate strong buyer demand, potential rental competition
   - **Expected Correlation:** r = -0.40 to -0.50

3. **MEDLISPRI38060** - Median Listing Price
   - **Frequency:** Monthly
   - **Use Case:** Alternative to Case-Shiller; more current pricing
   - **Theory:** Rising list prices → affordability constraint → rental demand
   - **Expected Correlation:** r = 0.50 to 0.60

4. **ATNHPIUS38060Q** - All-Transactions House Price Index (FHFA)
   - **Frequency:** Quarterly
   - **Use Case:** Alternative to Case-Shiller, broader coverage
   - **Theory:** Comprehensive price measure including all mortgage transactions
   - **Expected Correlation:** r = 0.55 to 0.65

### Economic Indicators

5. **NGMP38060** - Phoenix MSA Total GDP
   - **Frequency:** Annual (with lag)
   - **Use Case:** Overall economic health indicator
   - **Theory:** Economic growth → employment → rental demand
   - **Expected Correlation:** r = 0.45 to 0.55

6. **PHOE004UR** - Phoenix MSA Unemployment Rate
   - **Frequency:** Monthly
   - **Use Case:** Labor market slack indicator
   - **Theory:** Lower unemployment → higher employment → rental demand
   - **Expected Correlation:** r = -0.60 to -0.70 (negative: high unemployment → lower rents)

### Building Permits (Construction Pipeline)

7. **PHOE004BP1FH** - Building Permits: 1-Unit Structures
   - **Frequency:** Monthly
   - **Use Case:** Single-family construction indicator
   - **Theory:** More SF construction → reduces rental demand (substitution)
   - **Expected Correlation:** r = -0.35 to -0.45

8. **PHOE004BPPRIVSA** - Total Private Housing Permits
   - **Frequency:** Monthly (Seasonally Adjusted)
   - **Use Case:** Overall construction activity
   - **Theory:** Construction boom → future supply pressure
   - **Expected Correlation:** r = -0.40 to -0.50 (lagged 18-24 months)

### Population & Demographics

9. **PHXPOP** - Phoenix MSA Population
   - **Frequency:** Annual
   - **Use Case:** Long-term demand driver
   - **Theory:** Population growth → household formation → rental demand
   - **Expected Correlation:** r = 0.50 to 0.60

## Series Breakdown by Category

| Category | Count | Top Series | Frequency |
|----------|-------|------------|-----------|
| Employment | 321 | Various sector employment | Monthly |
| Housing | 291 | Inventory, prices, permits | Monthly/Quarterly |
| Price/Inflation | 156 | CPI components, price indices | Monthly |
| Income/Wages | 103 | Median income, wage indices | Annual/Quarterly |
| Construction | 36 | Building permits by type | Monthly |
| Population | 116 | Population by demographics | Annual |

## Recommended Additions to Rent Growth Model

Based on discovery, recommend adding these to the existing variable set:

### Tier 1 - High Priority (Add Immediately)

1. **ACTLISCOU38060** - Active Listing Count
   - **Rationale:** Real-time supply indicator, monthly frequency
   - **Expected Rank:** Top 10-12 predictor
   - **Integration:** Feature in GBM Phoenix-specific model

2. **MEDDAYONMAR38060** - Median Days on Market
   - **Rationale:** Market velocity, forward-looking demand indicator
   - **Expected Rank:** Top 12-15 predictor
   - **Integration:** Interaction term with listing price

3. **PHOE004UR** - Phoenix Unemployment Rate
   - **Rationale:** Direct labor market indicator, monthly updates
   - **Expected Rank:** Top 8-10 predictor
   - **Integration:** VAR national macro model component

### Tier 2 - Medium Priority (Test in Variable Selection)

4. **MEDLISPRI38060** - Median Listing Price
   - **Rationale:** Alternative to Case-Shiller, more current
   - **Note:** May be collinear with existing home price variables

5. **ATNHPIUS38060Q** - FHFA House Price Index
   - **Rationale:** Broader coverage than Case-Shiller
   - **Note:** Quarterly vs monthly Case-Shiller

6. **NGMP38060** - Phoenix GDP
   - **Rationale:** Overall economic health
   - **Note:** Annual frequency limits usefulness for short-term forecasts

### Tier 3 - Low Priority (Research/Experimental)

7. **PHOE004BP1FH** - Single-Family Permits
   - **Rationale:** Substitution effect (SF vs MF)
   - **Note:** May be weak signal relative to MF construction

## Data Export

Full discovery results saved to:
- **All Series:** `fred_phoenix_all_series.csv` (718 series)
- **Recommended Series:** `fred_phoenix_recommended_series.csv` (15 series)

## Next Steps

1. **Fetch Historical Data:** Use `phoenix_api_fetcher.py` to retrieve time series
2. **Elastic Net Screening:** Test new variables in elastic net variable selection
3. **Permutation Importance:** Rank new variables against existing predictors
4. **Model Integration:** Add top performers to GBM Phoenix-specific model

## Notes

- FRED series discovery returned 718 unique Phoenix MSA series
- Many series are CPI components (gasoline, electricity, food) - low relevance
- Housing inventory and labor market series show highest promise
- Construction permit data available at detailed level (1-unit, 5+ units, total)

---

**Generated by:** FRED Series Discovery Tool
**API Version:** FRED API v2.0
**Search Terms:** Phoenix, Phoenix-Mesa-Scottsdale, Phoenix MSA, Arizona Phoenix, Maricopa County
