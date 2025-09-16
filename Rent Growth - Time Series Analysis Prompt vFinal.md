# Time Series Economic Relationship Analysis Prompt
## Context & Objective
I need to analyze time series relationships between variables included within each category of the "Data Specification by Category", or real estate data, economic indicators, demographic factors, financial metrics, and demographic factors over the previous 40 years with monthly or quarterly data.

## Data Specifications by Category
Variables of Interest:
- Primary dependent variable: multifamily market rent growth
- Real estate data: multifamily inventory (current),
    multifamily deliveries - trailing (36 Months), multifamily deliveries - trailing (24 Months), multifamily deliveries - trailing (12 Months), multifamily deliveries - Forward-Looking (12 Months), multifamily absorption - trailing (36 Months), multifamily absorption - trailing (24 Months), multifamily absorption - trailing (12 Months)
- Economic indicators: GDP growth,
    fed funds effective  rate, 2-year U.S. Treasury yield, 5-year U.S. Treasury yield, 10-year U.S. Treasury yield, 30-year mortgage rate, unemployment rate, CPI, PPI
- Financial metrics: S&P 500, bond yields, credit spreads
- Demographic factors: population growth, median age,
    migration patterns
Data characteristics:
- Temporal range: 1/01/1985 to 9/12/2025
- Geographic scope: national and MSA level (Phoenix, Austin, Dallas, Denver, Salt Lake City, Nashville, Miami)
- Data sources:
  - National:
      - URLS: fred.stlouisfed.org, data.census.gov,
            bls.gov/data, bea.gov/data, fhfa.gov/DataTools, finance.yahoo.com/quotes/API, datashop.cboe.com, redfin.com/news/data-center, corelogic.com/intelligence/reports, apartmentlist.com/research/category/data-rent-estimates, nar.realtor/research-and-statistics, costar.com, data.nasdaq.com, ceicdata.com
      - APIs: {"FRED": "api.stlouisfed.org/fred/series",
            "Census": "api.census.gov/data",
            "BLS": "api.bls.gov/publicAPI/v2",
            "Quandl": "data.nasdaq.com/api/v3",
            "Alpha Vantage": "alphavantage.co/query"}
  - MSA (Phoenix):
      - URLS: oeo.az.gov,
            azmag.gov/Programs/Maps-and-Data, housing.az.gov/data-and-reports, armls.com/stats, cromfordreport.com, realestate.wpcarey.asu.edu/reports, gpec.org/data, frbsf.org/economic-research/regional, azcommerce.com/data, mcassessor.maricopa.gov, phoenixopendata.com, edpco.com/economic-trends, rlbrownreports.com, zondahome.com/phoenix, costar.com/markets/phoenix
      - APIs (API Endpoints Specific to Phoenix): {
            FRED: Add geographic series modifier ?geography=PHOE for Phoenix-Mesa-Scottsdale,
            Census: Use MSA code 38060 for Phoenix-Mesa-Chandler,
            BLS: Area code 04060 for Phoenix MSA employment data} 

## Analysis Requirements
### Phase 1: Exploratory Analysis
- Perform stationarity tests (ADF, KPSS) on all series
- Calculate correlation matrices at various lags (0-12 periods)
- Identify seasonal patterns and structural breaks
- Generate ACF/PACF plots for initial model selection

### Phase 2: Relationship Mapping
```python
relationships_to_test = {
    "contemporaneous": ["pearson", "spearman", "kendall"],
    "lagged": [1, 3, 6, 12],  # periods
    "rolling_windows": [30, 90, 365],  # days
    "regime_dependent": ["expansion", "recession", "recovery"]
}
```

### Phase 3: Predictive Modeling
Implement and compare:
1. **VAR/VECM Models** - for multivariate relationships
2. **Granger Causality Tests** - directional influence assessment
3. **Transfer Function Models** - lead-lag relationships
4. **Machine Learning Approaches**:
   - Random Forest with lagged features
   - LSTM for non-linear temporal dependencies
   - XGBoost with time-based validation

### Phase 4: Validation Framework
- Out-of-sample testing (walk-forward analysis)
- Cross-validation strategy: `TimeSeriesSplit` with 5 folds
- Performance metrics: RMSE, MAE, directional accuracy, Diebold-Mariano test
- Stability analysis: recursive estimation, parameter constancy tests

## Output Requirements
### Statistical Summary
```
For each variable pair:
- Correlation coefficient with confidence intervals
- Optimal lag structure
- Granger causality p-values (both directions)
- Cointegration test results
- Impulse response functions
```

### Predictive Power Assessment
- Feature importance rankings
- Out-of-sample R² by forecast horizon
- Prediction intervals at 80% and 95% confidence
- Model combination weights if ensemble approach used

### Visualization Requirements
1. Correlation heatmap with significance stars
2. Lead-lag correlation profiles
3. Rolling correlation time series
4. Forecast vs actual with confidence bands
5. Residual diagnostics panel

## Code Structure Preferences
- Use `statsmodels` for econometric tests
- `pandas` for data manipulation with method chaining
- `scikit-learn` for ML models with Pipeline architecture
- Parallel processing for multiple model estimation
- Type hints and docstrings for all functions

## Risk Factors & Limitations to Address
- Spurious correlations in non-stationary data
- Multiple testing correction (Bonferroni/FDR)
- Parameter instability over time
- Confounding variables not captured
- Data frequency mismatches

## Specific Questions to Answer
1. Which variables show strongest predictive power for multifamily rent growth (National) & multifamily rent growth (PHoenix MSA) during 3-5 year windows over the previous 40 years?
2. Are relationships stable across different market regimes?
3. What is the optimal combination of predictors for forecasting?
4. Do demographic variables add value beyond traditional economic indicators?
5. What are the confidence bounds on our predictions?

---

**Additional Context**: [Add domain-specific requirements, regulatory constraints, or business objectives]

**Computational Constraints**: [Specify if real-time processing needed, memory limitations, etc.]