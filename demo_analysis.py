#!/usr/bin/env python3
"""
Demo Analysis Pipeline with Synthetic Data
Demonstrates the rent growth analysis capabilities without requiring API keys
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.time_series_analyzer import TimeSeriesAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(start_date='1985-01-01', end_date='2025-09-12'):
    """
    Generate realistic synthetic economic and real estate data
    """
    logger.info("Generating synthetic data for demonstration...")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n_periods = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate economic indicators with realistic patterns
    
    # Federal Funds Rate (cyclical with trends)
    cycle_length = 120  # ~10 year cycle
    fed_funds_base = 4.0
    fed_funds_cycle = 2.0 * np.sin(2 * np.pi * np.arange(n_periods) / cycle_length)
    fed_funds_trend = -0.005 * np.arange(n_periods)  # Declining trend
    fed_funds_noise = np.random.normal(0, 0.2, n_periods)
    fed_funds = fed_funds_base + fed_funds_cycle + fed_funds_trend + fed_funds_noise
    fed_funds = np.maximum(0.1, fed_funds)  # Floor at 0.1%
    
    # Unemployment Rate (inverse of economic cycle)
    unemployment_base = 5.5
    unemployment_cycle = -1.5 * np.sin(2 * np.pi * np.arange(n_periods) / cycle_length)
    unemployment_noise = np.random.normal(0, 0.3, n_periods)
    unemployment = unemployment_base + unemployment_cycle + unemployment_noise
    unemployment = np.clip(unemployment, 3.5, 10.0)
    
    # GDP Growth (quarterly, with business cycles)
    gdp_base = 2.5
    gdp_cycle = 1.5 * np.sin(2 * np.pi * np.arange(n_periods) / (cycle_length * 0.8))
    gdp_noise = np.random.normal(0, 0.5, n_periods)
    gdp_growth = gdp_base + gdp_cycle + gdp_noise
    
    # CPI (inflation with trend)
    cpi_base = 100
    cpi_growth_rate = 0.002  # ~2.4% annual inflation
    cpi_noise = np.random.normal(0, 0.1, n_periods)
    cpi = cpi_base * np.exp(cpi_growth_rate * np.arange(n_periods) + cpi_noise.cumsum() * 0.01)
    
    # 10-Year Treasury Yield (correlated with fed funds but smoother)
    treasury_10y = fed_funds + 1.5 + np.random.normal(0, 0.15, n_periods)
    treasury_10y = pd.Series(treasury_10y).rolling(3).mean().fillna(method='bfill').values
    
    # Housing Starts (leading indicator, seasonal)
    housing_starts_base = 1500  # thousands of units
    housing_starts_cycle = 300 * np.sin(2 * np.pi * np.arange(n_periods) / cycle_length)
    housing_starts_seasonal = 100 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    housing_starts_noise = np.random.normal(0, 50, n_periods)
    housing_starts = housing_starts_base + housing_starts_cycle + housing_starts_seasonal + housing_starts_noise
    housing_starts = np.maximum(800, housing_starts)
    
    # S&P 500 (growth with volatility)
    sp500_returns = np.random.normal(0.007, 0.04, n_periods)  # ~8.4% annual with 4% monthly vol
    sp500 = 1000 * np.exp(sp500_returns.cumsum())
    
    # Multifamily specific metrics
    
    # Vacancy Rate (inverse of economic conditions)
    vacancy_base = 5.0
    vacancy_cycle = -unemployment_cycle * 0.3  # Opposite of unemployment
    vacancy_noise = np.random.normal(0, 0.2, n_periods)
    vacancy_rate = vacancy_base + vacancy_cycle + vacancy_noise
    vacancy_rate = np.clip(vacancy_rate, 2.0, 10.0)
    
    # New Supply (construction deliveries)
    supply_base = 250  # thousands of units annually
    supply_cycle = 50 * np.sin(2 * np.pi * np.arange(n_periods) / (cycle_length * 1.2))
    supply_lag = 18  # months construction lag
    supply_noise = np.random.normal(0, 20, n_periods)
    new_supply = supply_base + supply_cycle + supply_noise
    new_supply = np.maximum(100, new_supply)
    
    # Absorption (demand for new units)
    absorption_base = 240  # slightly less than supply on average
    absorption_economic = gdp_growth * 15  # Tied to economic growth
    absorption_demographic = 10 * np.sin(2 * np.pi * np.arange(n_periods) / (cycle_length * 2))
    absorption_noise = np.random.normal(0, 25, n_periods)
    absorption = absorption_base + absorption_economic + absorption_demographic + absorption_noise
    absorption = np.maximum(50, absorption)
    
    # RENT GROWTH - The target variable
    # Complex function of multiple factors
    
    # Base components
    rent_growth_base = 3.0  # Base annual growth rate
    
    # Economic impact (negative correlation with interest rates, positive with GDP)
    economic_impact = (
        -0.4 * (fed_funds - fed_funds.mean()) / fed_funds.std() +
        0.3 * (gdp_growth - gdp_growth.mean()) / gdp_growth.std() +
        -0.3 * (unemployment - unemployment.mean()) / unemployment.std()
    )
    
    # Supply/Demand impact
    supply_demand_balance = (absorption - new_supply) / new_supply
    supply_impact = 5.0 * supply_demand_balance  # Strong impact from supply/demand
    
    # Vacancy impact (high vacancy = lower rent growth)
    vacancy_impact = -0.8 * (vacancy_rate - vacancy_base)
    
    # Inflation pass-through
    cpi_change = pd.Series(cpi).pct_change().fillna(0).values * 100
    inflation_impact = 0.5 * cpi_change
    
    # Combine all factors
    rent_growth = (
        rent_growth_base + 
        economic_impact + 
        supply_impact + 
        vacancy_impact + 
        inflation_impact +
        np.random.normal(0, 0.5, n_periods)  # Random noise
    )
    
    # Smooth the series slightly
    rent_growth = pd.Series(rent_growth).rolling(3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    
    # Create DataFrame
    data = pd.DataFrame({
        'rent_growth': rent_growth,
        'fed_funds': fed_funds,
        'treasury_10y': treasury_10y,
        'unemployment': unemployment,
        'gdp_growth': gdp_growth,
        'cpi': cpi,
        'sp500': sp500,
        'housing_starts': housing_starts,
        'vacancy_rate': vacancy_rate,
        'new_supply': new_supply,
        'absorption': absorption,
    }, index=dates)
    
    # Add some lagged/derived features
    data['cpi_change'] = data['cpi'].pct_change() * 100
    data['sp500_returns'] = data['sp500'].pct_change() * 100
    data['supply_demand_balance'] = (data['absorption'] - data['new_supply']) / data['new_supply'] * 100
    data['real_rates'] = data['fed_funds'] - data['cpi_change']
    
    # Fill any remaining NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data


def run_demo_analysis():
    """
    Run complete analysis with synthetic data
    """
    logger.info("="*60)
    logger.info("MULTIFAMILY RENT GROWTH ANALYSIS - DEMO MODE")
    logger.info("="*60)
    
    # Generate synthetic data
    data = generate_synthetic_data()
    logger.info(f"Generated {len(data)} months of synthetic data")
    logger.info(f"Variables: {list(data.columns)}")
    
    # Save synthetic data
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    data_file = output_dir / f"synthetic_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
    data.to_csv(data_file)
    logger.info(f"Saved synthetic data to {data_file}")
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(target_variable='rent_growth')
    
    # Select key variables for analysis
    analysis_vars = [
        'rent_growth', 'fed_funds', 'treasury_10y', 'unemployment',
        'gdp_growth', 'vacancy_rate', 'new_supply', 'absorption',
        'supply_demand_balance', 'real_rates'
    ]
    
    analysis_data = data[analysis_vars].copy()
    
    # Run complete analysis
    logger.info("\nRunning comprehensive time series analysis...")
    results = analyzer.perform_complete_analysis(analysis_data, 'rent_growth')
    
    # Print results summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS RESULTS SUMMARY")
    logger.info("="*60)
    
    # 1. Stationarity Results
    logger.info("\n1. STATIONARITY TEST RESULTS:")
    stationary_count = 0
    for var, result in results.stationarity.items():
        if result['conclusion'] == 'Stationary':
            stationary_count += 1
            logger.info(f"  ✓ {var}: Stationary")
        else:
            logger.info(f"  ✗ {var}: {result['conclusion']}")
    logger.info(f"  Summary: {stationary_count}/{len(results.stationarity)} variables are stationary")
    
    # 2. Granger Causality Results
    logger.info("\n2. GRANGER CAUSALITY (Variables that predict rent growth):")
    causal_vars = []
    for var, result in results.causality.items():
        if isinstance(result, dict) and result.get('causes_target'):
            causal_vars.append((var, result['optimal_lag'], result['min_p_value']))
    
    causal_vars.sort(key=lambda x: x[2])  # Sort by p-value
    for var, lag, p_val in causal_vars[:5]:
        logger.info(f"  • {var}: Optimal lag = {lag} months, p-value = {p_val:.4f}")
    
    # 3. Cointegration Results
    logger.info("\n3. COINTEGRATION TEST:")
    if results.cointegration.get('has_cointegration'):
        logger.info(f"  ✓ Found {results.cointegration['n_relationships']} cointegrating relationship(s)")
        logger.info("  → Long-term equilibrium relationships exist")
    else:
        logger.info("  ✗ No cointegration detected")
    
    # 4. Model Performance
    logger.info("\n4. PREDICTIVE MODEL PERFORMANCE:")
    for metric, value in results.validation_metrics.items():
        model_name = "Random Forest" if 'rf' in metric else "XGBoost"
        metric_type = "R²" if 'r2' in metric else "RMSE"
        logger.info(f"  {model_name} {metric_type}: {value:.4f}")
    
    # 5. Top Predictive Features
    logger.info("\n5. TOP PREDICTIVE FEATURES (Random Forest Importance):")
    if results.feature_importance:
        sorted_features = sorted(
            results.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    # 6. Key Insights
    logger.info("\n6. KEY INSIGHTS:")
    
    # Correlation at different lags
    if results.correlations:
        logger.info("\n  Correlation Analysis:")
        for lag_key, lag_data in results.correlations.items():
            if isinstance(lag_data, dict) and 'pearson' in lag_data:
                if isinstance(lag_data['pearson'], pd.Series):
                    top_corr = lag_data['pearson'].drop('rent_growth', errors='ignore').head(3)
                    logger.info(f"    {lag_key}:")
                    for var, corr in top_corr.items():
                        logger.info(f"      • {var}: {corr:.3f}")
    
    # Generate recommendations
    logger.info("\n7. RECOMMENDATIONS:")
    
    recommendations = [
        "Monitor supply/demand balance as a leading indicator",
        "Track Federal Reserve policy changes for interest rate impacts",
        "Consider vacancy rates for short-term rent growth predictions",
        "Use ensemble models for robust forecasting",
        "Implement VECM models if long-term relationships are needed"
    ]
    
    for rec in recommendations:
        logger.info(f"  → {rec}")
    
    # Save complete results
    results_file = output_dir / f"analysis_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    # Convert results to serializable format
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape,
        'variables_analyzed': analysis_vars,
        'validation_metrics': results.validation_metrics,
        'has_cointegration': results.cointegration.get('has_cointegration', False),
        'top_features': dict(sorted_features[:10]) if results.feature_importance else {},
        'granger_causes': {
            var: {
                'optimal_lag': info['optimal_lag'],
                'p_value': info['min_p_value'],
                'causes_target': info['causes_target']
            }
            for var, info in results.causality.items()
            if isinstance(info, dict) and 'optimal_lag' in info
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\n✓ Complete results saved to {results_file}")
    logger.info(f"✓ Synthetic data saved to {data_file}")
    
    logger.info("\n" + "="*60)
    logger.info("DEMO ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return results, data


if __name__ == "__main__":
    results, data = run_demo_analysis()