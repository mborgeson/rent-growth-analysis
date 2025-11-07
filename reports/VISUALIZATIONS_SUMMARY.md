# Phoenix Rent Growth Analysis - Visualizations Summary

**Generated:** 2025-11-06
**Project:** Phoenix Multifamily Market Rent Growth Projection (2026-2030)
**Location:** `/home/mattb/Rent Growth Analysis/outputs/`

---

## Overview

This document summarizes the comprehensive set of visualizations created to accompany the Phoenix Multifamily Rent Growth Projection Report (2026-2030). All visualizations are high-resolution PNG files (300 DPI) suitable for presentations and reports.

---

## Visualization Files

### 1. `phoenix_historical_analysis.png` (16" x 12")

**Purpose**: Comprehensive historical market analysis covering 2000-2025

**Contents**:
- **Top Panel**: Historical rent and vacancy trends with market cycle shading
  - 5 market cycles clearly identified (Pre-GFC, GFC/Recovery, Post-GFC, COVID, Post-COVID)
  - Dual-axis showing rent ($) and vacancy rate (%)
  - COVID-19 structural break marker (March 2020)
  - Equilibrium vacancy line at 5%

- **Middle Left**: Year-over-year rent growth volatility
  - Positive/negative growth shading
  - Long-term average trend line (2.32%)
  - Peak annotation: +16.09% (Q3 2021)
  - Trough annotation: -6.87% (Q4 2009)

- **Middle Right**: Supply metrics - inventory and construction pipeline
  - Total inventory growth (265K → 425K units, +60%)
  - Construction pipeline fluctuations
  - Dual-axis visualization

- **Bottom Left**: Construction as percentage of inventory
  - Historical average: 2.8%
  - Current level: 5.2% (86% above average)
  - Trend analysis highlighting current oversupply

- **Bottom Right**: Employment growth trends
  - Total nonfarm employment (1,710K → 2,462K)
  - +43.9% total growth (2000-2025)
  - Foundation for demand projections

**Key Insights**:
- Rent Range: $903 - $1,677
- Vacancy Range: 5.2% - 12.9%
- Growth Range: -6.87% to +16.09%

---

### 2. `phoenix_scenario_projections.png` (16" x 12")

**Purpose**: Future scenario analysis and projections (2026-2030)

**Contents**:
- **Top Left**: Rent projections by scenario
  - Bear Case: $1,270 in 2030 (-19.0% total)
  - Base Case: $1,298 in 2030 (-17.3% total)
  - Bull Case: $1,345 in 2030 (-14.3% total)
  - Scenario range shading showing uncertainty band
  - Current 2025 baseline marker ($1,569)

- **Top Right**: Annual rent growth rates
  - All scenarios show negative growth 2026-2028
  - Base case average: -3.5% annually
  - Gradual recovery toward 2030
  - Negative growth zone shading

- **Bottom Left**: Vacancy rate projections
  - Bear Case: 12.5% → 12.5% (persistent oversupply)
  - Base Case: 12.0% → 8.5% (gradual normalization)
  - Bull Case: 11.5% → 6.0% (rapid absorption)
  - Equilibrium line at 5%
  - Oversupply zone shading

- **Bottom Right**: Summary metrics table
  - 2030 rent levels by scenario
  - 5-year total change percentages
  - Average annual growth rates
  - 2030 vacancy rates
  - Employment growth assumptions
  - Migration assumptions

**Key Insights**:
- All scenarios project negative rent growth through 2026-2028
- Base case: -17.3% cumulative decline over 5 years
- Scenario spread: $75 between bear ($1,270) and bull ($1,345)

---

### 3. `phoenix_correlation_analysis.png` (16" x 10")

**Purpose**: Statistical analysis of rent growth drivers and relationships

**Contents**:
- **Top Panel**: Correlation heatmap matrix
  - 6 variables: rent_yoy_growth, vacancy, unemployment, employment, under_construction, net_absorption
  - Color-coded strength (-1 to +1)
  - Key finding: Vacancy has -0.886 correlation (strongest driver)
  - Employment shows +0.219 (weak positive)
  - Under construction shows +0.158 (weak positive)

- **Bottom Left**: Vacancy vs rent growth scatter plot
  - 103 quarterly observations color-coded by market cycle
  - Linear regression trend line: y = -1.06x + 11.18
  - Correlation: -0.886 (highly significant)
  - Quadrant markers at 5% vacancy and 0% growth
  - Strong inverse relationship visible

- **Bottom Right**: Employment vs rent growth scatter
  - Same 103 observations with cycle color coding
  - Linear regression trend line
  - Correlation: +0.219 (weak positive)
  - Demonstrates employment provides demand foundation but vacancy dominates

**Color Legend**:
- Light blue: Pre-GFC (2000-2007)
- Light red: GFC/Recovery (2008-2012)
- Light green: Post-GFC Expansion (2013-2019)
- Light orange: COVID Boom (2020-2021)
- Light purple: Post-COVID Normalization (2022-2025)

**Key Insights**:
- Vacancy is dominant driver (r = -0.886)
- -1 pp vacancy change ≈ +1% rent growth
- Employment has weak positive influence
- Construction pipeline has minimal direct impact on growth

---

### 4. `phoenix_migration_validation.png` (16" x 11")

**Purpose**: Migration impact analysis and model performance validation

**Contents**:
- **Top Panel**: Migration impact on rental demand
  - Three scenario bars (Bear/Base/Bull)
  - Annual renter demand: 6,622 / 13,244 / 19,865 units
  - 5-year total demand: 33,109 / 66,218 / 99,327 units
  - Reference line: Current annual absorption (16,414 units)
  - Assumptions box: 2.5 ppl/HH, 33.1% renter rate
  - 2021 peak context: 145,871 people = 19,319 HH = 89% of COVID absorption

- **Middle Panel**: Model validation performance
  - 8-quarter out-of-sample validation (Q4 2023 - Q3 2025)
  - Actual rent vs. three model predictions
  - SARIMA: MAE $22.62
  - XGBoost: MAE $47.89
  - Ensemble (60/40): MAE $12.16 (best)
  - Clear visualization of ensemble superiority

- **Bottom Left**: Model prediction errors by quarter
  - Grouped bar chart showing error for each model
  - Q1-Q8 validation period
  - Visual demonstration of ensemble stability
  - Error reduction patterns clear

- **Bottom Right**: Cumulative absolute error
  - Accumulated error over validation period
  - Final cumulative errors:
    - SARIMA: $181
    - XGBoost: $383
    - Ensemble: $97 (best)
  - 74.6% improvement over XGBoost
  - Demonstrates consistent ensemble advantage

**Key Insights**:
- Base case migration (100K/yr) supports 81% of current absorption
- Ensemble model outperforms individual models by 46-75%
- Model validation confirms projection reliability
- Migration remains critical demand driver (historically 89% of COVID absorption)

---

## Technical Specifications

**Format**: PNG (Portable Network Graphics)
**Resolution**: 300 DPI
**Color Space**: RGB
**Dimensions**:
- Landscape format: 16" x 12" (4800 x 3600 pixels)
- Special format: 16" x 11" (4800 x 3300 pixels), 16" x 10" (4800 x 3000 pixels)

**Software Used**:
- Python 3.12
- Matplotlib 3.x (visualization framework)
- Seaborn 0.x (statistical visualizations)
- Pandas 2.x (data processing)

**Style**:
- Seaborn darkgrid theme
- Husl color palette for consistency
- Professional typography with bold titles
- High-contrast color schemes for accessibility

---

## Usage Recommendations

### For Presentations
- All images are 300 DPI and suitable for projection
- Landscape format optimized for 16:9 aspect ratio
- Clear legends and annotations readable from distance
- Professional color schemes appropriate for business context

### For Reports
- High resolution suitable for print reproduction
- Self-contained with comprehensive legends
- Annotations provide context without requiring separate caption
- Consistent branding and color scheme across all visualizations

### For Analysis
- All data points visible and traceable
- Statistical relationships clearly demonstrated
- Historical context provided through color coding
- Quantitative annotations support data-driven decisions

---

## Data Sources

All visualizations derive from:
- **CoStar Market Data**: Quarterly Phoenix MSA multifamily metrics (Q1 2000 - Q3 2025)
- **FRED Employment Data**: Phoenix MSA employment statistics (1990-2025)
- **US Census ACS**: 2023 demographic data
- **IRS Migration Data**: 2021 net migration flows
- **Custom Models**: SARIMA + XGBoost ensemble projections

---

## Interpretation Guide

### Historical Analysis (Visualization 1)
- **Market Cycles**: Color-coded shading identifies distinct market periods
- **COVID Break**: Vertical red line shows March 2020 structural break
- **Equilibrium**: Horizontal lines at 5% vacancy and historical averages
- **Peak/Trough**: Annotated extreme values for historical context

### Scenario Projections (Visualization 2)
- **Scenario Spread**: Shaded area between bear and bull shows uncertainty range
- **Current Baseline**: Black diamond marker shows 2025 starting point
- **Negative Zone**: Shaded areas highlight projected decline periods
- **Metrics Table**: Comprehensive scenario assumptions and outcomes

### Correlation Analysis (Visualization 3)
- **Heatmap Colors**: Green = positive correlation, Red = negative correlation
- **Scatter Colors**: Time periods show how relationships evolved across cycles
- **Trend Lines**: Red dashed lines show linear regression fits
- **R-values**: Correlation coefficients quantify relationship strength

### Migration & Validation (Visualization 4)
- **Bar Heights**: Direct comparison of migration scenario impacts
- **Error Bars**: Prediction accuracy across validation period
- **Cumulative Lines**: Running total shows consistent model advantage
- **Reference Lines**: Current absorption provides context for demand

---

## File Locations

```
/home/mattb/Rent Growth Analysis/outputs/
├── phoenix_historical_analysis.png       (2.1 MB)
├── phoenix_scenario_projections.png      (1.8 MB)
├── phoenix_correlation_analysis.png      (1.6 MB)
└── phoenix_migration_validation.png      (1.9 MB)
```

**Total Size**: ~7.4 MB
**Combined Coverage**: 25+ years historical + 5 years projected = 30 years total analysis

---

## Visualization Integration with Report

These visualizations are designed to complement the comprehensive Phoenix Multifamily Rent Growth Projection Report (2026-2030):

**Report Section** → **Recommended Visualization**
- Executive Summary → Scenario Projections (#2)
- Historical Analysis → Historical Analysis (#1)
- Current Market Conditions → Historical Analysis (#1)
- Statistical Analysis → Correlation Analysis (#3)
- Migration Impact → Migration & Validation (#4)
- Model Methodology → Migration & Validation (#4)
- Scenario Projections → Scenario Projections (#2)
- Investment Implications → All visualizations

---

## Next Steps & Enhancements

### Potential Additional Visualizations
1. **Interactive Dashboard**: Convert static images to interactive HTML with Plotly
2. **Submarket Analysis**: Break down projections by Phoenix submarkets
3. **Sensitivity Analysis**: Tornado charts showing assumption impact
4. **Construction Pipeline**: Detailed delivery timeline visualization
5. **Cap Rate Trends**: Historical and projected cap rate analysis

### Export Formats
- PDF (vector format for scalability)
- SVG (web-ready vector graphics)
- Excel (interactive charts with source data)
- PowerPoint (presentation-ready slide deck)

---

## Conclusion

This comprehensive visualization suite provides:
- **Historical Context**: 25 years of market data across 5 distinct cycles
- **Statistical Rigor**: Correlation analysis and model validation
- **Future Scenarios**: Three projection scenarios with uncertainty bands
- **Investment Insights**: Clear visual evidence for decision-making

All visualizations maintain professional quality standards suitable for:
- Investor presentations
- Analytical reports
- Market research publications
- Strategic planning documents
- Academic research

---

**Report Prepared by:** Claude Code SPARC Agent
**Visualization Engine:** Matplotlib + Seaborn
**Data Quality:** 103 quarterly observations, 8-quarter validation
**Projection Confidence:** High (near-term), Moderate (mid-term), Lower (long-term)
