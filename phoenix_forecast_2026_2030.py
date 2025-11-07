import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading Phoenix market data...")
costar = pd.read_csv('/home/mattb/Documents/multifamily-data/costar-exports/phoenix/market_submarket_data/CoStar Market Data (Quarterly) - Phoenix (AZ) MSA Market.csv')
employment = pd.read_csv('/home/mattb/Documents/multifamily-data/msa-data/phoenix/phoenix_fred_employment.csv')
cycle_data = pd.read_csv('phoenix_cycle_analysis.csv')

# Clean CoStar data
costar['Period'] = costar['Period'].str.strip('\ufeff')
costar = costar.sort_values('Period').reset_index(drop=True)

# Separate actual vs forecast data
actual_data = costar[costar['Period'] <= '2025 Q3'].copy()

# Clean numeric columns
for col in actual_data.columns:
    if col != 'Period':
        actual_data[col] = actual_data[col].astype(str).str.replace(',', '').str.replace('%', '').str.replace('$', '').str.replace(' ', '')
        actual_data[col] = pd.to_numeric(actual_data[col], errors='coerce')

print("\n=== PHOENIX MARKET CONTEXT (Q3 2025) ===")
latest = actual_data[actual_data['Period'] == '2025 Q3'].iloc[0]
print(f"Current Rent: ${latest['Market Asking Rent/Unit']:.0f}")
print(f"Vacancy Rate: {latest['Vacancy Rate']:.1f}%")
print(f"YoY Rent Growth: {latest['Annual Rent Growth']:.1f}%")
print(f"Under Construction: {latest['Under Constr Units']:.0f} units ({latest['Under Constr % of Inventory']:.1f}% of inventory)")
print(f"Inventory: {latest['Inventory Units']:.0f} units")

# MSA-SPECIFIC ADJUSTMENT PROTOCOL FOR PHOENIX
print("\n=== MSA-SPECIFIC ADJUSTMENTS ===")

# Phoenix characteristics:
# 1. HIGH supply elasticity - flat terrain, abundant land
# 2. MIGRATION-DEPENDENT - California outmigration key driver
# 3. MODERATE employment diversity - weighted toward services
# 4. HYPERSUPPLY phase - needs mean reversion, not trend extrapolation

# Calculate key metrics
recent_4q = actual_data.tail(4)
recent_12q = actual_data.tail(12)

supply_surplus_pct = ((recent_4q['Net Delivered Units 12 Mo'].mean() -
                       recent_4q['12 Mo Absorp Units'].mean()) /
                      recent_4q['12 Mo Absorp Units'].mean() * 100)

print(f"\nSupply Surplus: {supply_surplus_pct:.1f}%")
print(f"Construction Pipeline: {latest['Under Constr Units']:.0f} units")
print(f"Average Deliveries (L12Q): {recent_12q['Net Delivered Units 12 Mo'].mean():.0f} units/year")
print(f"Average Absorption (L12Q): {recent_12q['12 Mo Absorp Units'].mean():.0f} units/year")

# FORECAST MODEL SETUP
print("\n=== BUILDING ENSEMBLE FORECAST MODELS ===")

# Historical data for modeling (2015-2025 Q3)
model_data = actual_data[actual_data['Period'] >= '2015 Q1'].copy()

# Feature engineering
model_data['quarter'] = model_data.index % 4
model_data['year'] = model_data.index // 4 + 2015
model_data['rent_yoy'] = model_data['Market Asking Rent/Unit'].pct_change(4) * 100
model_data['occupancy'] = 100 - model_data['Vacancy Rate']
model_data['constr_pct'] = model_data['Under Constr Units'] / model_data['Inventory Units'] * 100
model_data['supply_demand_ratio'] = model_data['Net Delivered Units 12 Mo'] / model_data['12 Mo Absorp Units']

# Calculate equilibrium metrics
equilibrium_occupancy = model_data['occupancy'].quantile(0.5)  # Median occupancy
equilibrium_rent_growth = 3.5  # Long-run equilibrium for Phoenix (inflation + real growth)

print(f"\nEquilibrium Occupancy: {equilibrium_occupancy:.1f}%")
print(f"Equilibrium Rent Growth: {equilibrium_rent_growth:.1f}%")
print(f"Current Occupancy: {100 - latest['Vacancy Rate']:.1f}%")

# MEAN REVERSION MODEL (Primary for Hypersupply)
print("\n=== MEAN REVERSION PROJECTION ===")

# Reversion speed based on supply surplus
reversion_speed = 0.25 if supply_surplus_pct > 30 else 0.20  # Slower if extreme surplus
print(f"Reversion Speed: {reversion_speed:.2f} (25% = fast, 15% = slow)")

# Project occupancy recovery
current_occupancy = 100 - latest['Vacancy Rate']
occupancy_gap = equilibrium_occupancy - current_occupancy
print(f"Occupancy Gap: {occupancy_gap:.1f}pp")

# Project construction slowdown
construction_start = latest['Under Constr Units']
construction_decline_rate = 0.15  # 15% quarterly decline in pipeline

# Generate quarterly projections
projection_quarters = 20  # 2025 Q4 through 2030 Q4
projections = []

for q in range(projection_quarters):
    quarter_date = f"{2025 + ((3 + q) // 4)} Q{((3 + q) % 4) + 1}"

    # Occupancy recovery (mean reversion)
    quarters_out = q + 1
    projected_occupancy = current_occupancy + (occupancy_gap * (1 - (1 - reversion_speed) ** quarters_out))

    # Construction pipeline decline
    projected_construction = construction_start * ((1 - construction_decline_rate) ** quarters_out)
    constr_pct = projected_construction / latest['Inventory Units'] * 100

    # Rent growth forecast based on occupancy position
    occupancy_impact = (projected_occupancy - equilibrium_occupancy) * 0.6  # 60 bps per pp
    supply_impact = -1.5 if constr_pct > 7 else (-0.8 if constr_pct > 5 else 0)

    # Mean reversion toward equilibrium
    current_growth_estimate = occupancy_impact + supply_impact
    reversion_to_equilibrium = equilibrium_rent_growth - current_growth_estimate
    projected_rent_growth = current_growth_estimate + (reversion_to_equilibrium * reversion_speed * quarters_out / 8)

    # Seasonal adjustment
    seasonal_factors = [0.2, 0.8, 0.5, -0.5]  # Q1, Q2, Q3, Q4
    seasonal_adj = seasonal_factors[((3 + q) % 4)]
    projected_rent_growth += seasonal_adj

    projections.append({
        'period': quarter_date,
        'quarter': q + 1,
        'occupancy': projected_occupancy,
        'construction_units': projected_construction,
        'construction_pct': constr_pct,
        'rent_growth_yoy': projected_rent_growth,
        'model': 'mean_reversion'
    })

# Convert to DataFrame
forecast_df = pd.DataFrame(projections)

# Calculate annual averages
print("\n=== PHOENIX RENT GROWTH PROJECTIONS (2026-2030) ===")
print("\nANNUAL AVERAGE RENT GROWTH:")

for year in range(2026, 2031):
    year_data = forecast_df[forecast_df['period'].str.startswith(str(year))]
    avg_growth = year_data['rent_growth_yoy'].mean()
    avg_occupancy = year_data['occupancy'].mean()
    avg_constr_pct = year_data['construction_pct'].mean()

    print(f"\n{year}:")
    print(f"  Base Case Rent Growth: {avg_growth:.1f}%")
    print(f"  Avg Occupancy: {avg_occupancy:.1f}%")
    print(f"  Avg Construction Pipeline: {avg_constr_pct:.1f}% of inventory")

# RISK SCENARIOS
print("\n=== RISK SCENARIOS (2026-2030 Average) ===")

# Calculate 5-year average
forecast_2026_2030 = forecast_df[forecast_df['quarter'] <= 20]
base_avg = forecast_2026_2030['rent_growth_yoy'].mean()

# Bull scenario: Faster occupancy recovery, stronger demand
bull_boost = 1.5  # 150 bps above base
bull_avg = base_avg + bull_boost

# Bear scenario: Prolonged oversupply, slower recovery
bear_penalty = -2.0  # 200 bps below base
bear_avg = base_avg + bear_penalty

print(f"\nBASE CASE:  {base_avg:.1f}% average annual rent growth")
print(f"  - Assumptions: Gradual occupancy recovery, construction slowdown")
print(f"  - Occupancy reaches ~{forecast_2026_2030['occupancy'].iloc[-1]:.1f}% by 2030")
print(f"  - Construction pipeline normalizes to ~{forecast_2026_2030['construction_pct'].iloc[-1]:.1f}%")

print(f"\nBULL CASE:  {bull_avg:.1f}% average annual rent growth")
print(f"  - Assumptions: Strong California migration, rapid demand recovery")
print(f"  - Faster construction absorption, tech job growth")
print(f"  - Occupancy reaches 92%+ by 2027")

print(f"\nBEAR CASE:  {bear_avg:.1f}% average annual rent growth")
print(f"  - Assumptions: Prolonged oversupply, slower demographic growth")
print(f"  - Construction continues at elevated levels longer")
print(f"  - Occupancy recovery delayed to 2029+")

# SUPPLY/DEMAND BALANCE ASSESSMENT
print("\n=== SUPPLY/DEMAND BALANCE ASSESSMENT ===")

print(f"\nCURRENT IMBALANCE (Q3 2025):")
print(f"  Supply Surplus: {supply_surplus_pct:.1f}%")
print(f"  Under Construction: {latest['Under Constr Units']:.0f} units ({latest['Under Constr % of Inventory']:.1f}%)")
print(f"  Occupancy: {100 - latest['Vacancy Rate']:.1f}% (vs. equilibrium {equilibrium_occupancy:.1f}%)")

print(f"\nPROJECTED REBALANCING:")
print(f"  2026: Construction starts declining, absorption stabilizes")
print(f"  2027-2028: Gradual occupancy recovery toward 90%+")
print(f"  2029-2030: Market reaches equilibrium, rent growth normalizes to {equilibrium_rent_growth:.1f}%")

print(f"\nKEY DRIVERS:")
print(f"  1. Construction Pipeline Decline: {construction_start:.0f} → {forecast_2026_2030['construction_units'].iloc[-1]:.0f} units")
print(f"  2. Occupancy Recovery: {current_occupancy:.1f}% → {forecast_2026_2030['occupancy'].iloc[-1]:.1f}%")
print(f"  3. Rent Growth Mean Reversion: {latest['Annual Rent Growth']:.1f}% → {equilibrium_rent_growth:.1f}%")

# Save detailed forecast
forecast_df.to_csv('phoenix_rent_growth_forecast_2026_2030.csv', index=False)
print(f"\nDetailed forecast saved to: phoenix_rent_growth_forecast_2026_2030.csv")

# Summary statistics
summary_stats = {
    'market': 'Phoenix MSA',
    'cycle_phase': 'Hypersupply (Early-Mid Stage)',
    'current_rent': latest['Market Asking Rent/Unit'],
    'current_vacancy': latest['Vacancy Rate'],
    'current_occupancy': 100 - latest['Vacancy Rate'],
    'equilibrium_occupancy': equilibrium_occupancy,
    'base_2026_2030_avg': base_avg,
    'bull_2026_2030_avg': bull_avg,
    'bear_2026_2030_avg': bear_avg,
    'construction_decline_2025_2030': construction_start - forecast_2026_2030['construction_units'].iloc[-1],
    'occupancy_recovery_2025_2030': forecast_2026_2030['occupancy'].iloc[-1] - current_occupancy
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('phoenix_forecast_summary.csv', index=False)
print(f"Summary statistics saved to: phoenix_forecast_summary.csv")

print("\n=== ANALYSIS COMPLETE ===")
