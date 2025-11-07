#!/usr/bin/env python3
"""
Phoenix MSA Multifamily Rent Growth Forecasting - Data Acquisition Script
Fetches all required data from FRED, BLS, Census, and Zillow APIs
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
import os

# API Keys
FRED_API_KEY = 'd043d26a9a4139438bb2a8d565bc01f7'
CENSUS_API_KEY = '0145eb3254e9885fa86407a72b6b0fb381e846e8'

# Output directory
OUTPUT_DIR = 'data/raw'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FRED API Functions
# ============================================================================

def fetch_fred_series(series_id, start_date='2010-01-01', api_key=FRED_API_KEY):
    """
    Fetch time series data from FRED API

    Parameters:
    -----------
    series_id : str
        FRED series identifier
    start_date : str
        Start date in YYYY-MM-DD format
    api_key : str
        FRED API key

    Returns:
    --------
    pd.DataFrame
        Time series data with date index
    """
    base_url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'value']]
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')
            df.columns = [series_id]
            print(f"✓ Fetched {series_id}: {len(df)} observations from {df.index.min()} to {df.index.max()}")
            return df
        else:
            print(f"✗ No data for {series_id}")
            return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error fetching {series_id}: {str(e)}")
        return pd.DataFrame()

def fetch_all_fred_data():
    """Fetch all required FRED series for national macro variables"""
    print("\n" + "="*80)
    print("FETCHING FRED DATA (National Macro Variables)")
    print("="*80)

    # Define FRED series to fetch
    fred_series = {
        'MORTGAGE30US': '30-Year Mortgage Rate',
        'PAYEMS': 'Total Nonfarm Employment (National)',
        'T5YIE': '5-Year Breakeven Inflation Rate',
        'DFF': 'Federal Funds Rate',
        'UNRATE': 'Unemployment Rate (National)',
        'CPIAUCSL': 'Consumer Price Index',
        'HOUST': 'Housing Starts (National)',
        'PERMIT': 'Building Permits (National)',
        'MSPUS': 'Median Sales Price of Houses Sold (National)',
        'CSUSHPISA': 'S&P/Case-Shiller U.S. National Home Price Index'
    }

    all_data = []

    for series_id, description in fred_series.items():
        print(f"\nFetching: {description}")
        df = fetch_fred_series(series_id)
        if not df.empty:
            all_data.append(df)
        time.sleep(0.5)  # Rate limiting

    # Combine all series
    if all_data:
        fred_df = pd.concat(all_data, axis=1)
        output_file = f'{OUTPUT_DIR}/fred_national_macro.csv'
        fred_df.to_csv(output_file)
        print(f"\n✓ Saved FRED data to {output_file}")
        print(f"  Shape: {fred_df.shape}, Date range: {fred_df.index.min()} to {fred_df.index.max()}")
        return fred_df
    else:
        print("\n✗ No FRED data fetched")
        return pd.DataFrame()

# ============================================================================
# BLS API Functions
# ============================================================================

def fetch_bls_series(series_id, start_year='2010', end_year='2024'):
    """
    Fetch employment/wage data from BLS API

    Parameters:
    -----------
    series_id : str
        BLS series identifier
    start_year : str
        Start year
    end_year : str
        End year

    Returns:
    --------
    pd.DataFrame
        Time series data
    """
    bls_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    payload = {
        'seriesid': [series_id],
        'startyear': start_year,
        'endyear': end_year
    }

    try:
        response = requests.post(bls_url, json=payload)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'REQUEST_SUCCEEDED':
            series_data = data['Results']['series'][0]['data']

            # Convert to DataFrame
            df = pd.DataFrame(series_data)
            df['date'] = pd.to_datetime(df['year'] + '-' + df['period'].str.replace('M', ''), format='%Y-%m')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].set_index('date').sort_index()
            df.columns = [series_id]

            print(f"✓ Fetched {series_id}: {len(df)} observations from {df.index.min()} to {df.index.max()}")
            return df
        else:
            print(f"✗ BLS API error for {series_id}: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error fetching {series_id}: {str(e)}")
        return pd.DataFrame()

def fetch_all_bls_data():
    """Fetch all required BLS series for Phoenix employment"""
    print("\n" + "="*80)
    print("FETCHING BLS DATA (Phoenix MSA Employment)")
    print("="*80)

    # Phoenix MSA series codes
    bls_series = {
        'SMU04383400000000001': 'Phoenix MSA Total Nonfarm Employment',
        'SMU04383406000000001': 'Phoenix MSA Professional & Business Services',
        'SMU04383405000000001': 'Phoenix MSA Information (Tech)',
        'SMU04383403100000001': 'Phoenix MSA Manufacturing',
        'SMU04383407000000001': 'Phoenix MSA Leisure & Hospitality'
    }

    all_data = []

    for series_id, description in bls_series.items():
        print(f"\nFetching: {description}")
        df = fetch_bls_series(series_id)
        if not df.empty:
            all_data.append(df)
        time.sleep(1)  # Rate limiting (BLS has stricter limits)

    # Combine all series
    if all_data:
        bls_df = pd.concat(all_data, axis=1)
        output_file = f'{OUTPUT_DIR}/bls_phoenix_employment.csv'
        bls_df.to_csv(output_file)
        print(f"\n✓ Saved BLS data to {output_file}")
        print(f"  Shape: {bls_df.shape}, Date range: {bls_df.index.min()} to {bls_df.index.max()}")
        return bls_df
    else:
        print("\n✗ No BLS data fetched")
        return pd.DataFrame()

# ============================================================================
# Census API Functions
# ============================================================================

def fetch_census_building_permits():
    """Fetch building permits for Phoenix MSA from Census API"""
    print("\n" + "="*80)
    print("FETCHING CENSUS DATA (Building Permits)")
    print("="*80)

    # Phoenix MSA CBSA code: 38060
    # Note: Census building permits API structure varies by year
    # Using Survey of Construction for multifamily permits

    base_url = 'https://api.census.gov/data/timeseries/eits/building'

    results = []

    # Try to fetch recent years
    for year in range(2010, 2025):
        params = {
            'get': 'cell_value,time_slot_name',
            'for': 'us:*',
            'key': CENSUS_API_KEY,
            'time': year
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Fetched permits data for {year}")
                results.append({'year': year, 'data': data})
            else:
                print(f"  No data for {year}")
        except Exception as e:
            print(f"  Error for {year}: {str(e)}")

        time.sleep(0.5)

    # Save raw results
    output_file = f'{OUTPUT_DIR}/census_building_permits_raw.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved Census permits data to {output_file}")

    return results

def fetch_census_population():
    """Fetch Phoenix MSA population estimates from Census API"""
    print("\n" + "="*80)
    print("FETCHING CENSUS DATA (Population Estimates)")
    print("="*80)

    # Population Estimates Program (PEP)
    # Phoenix MSA CBSA: 38060

    base_url = 'https://api.census.gov/data/2023/pep/population'
    params = {
        'get': 'POP,DENSITY',
        'for': 'metropolitan statistical area/micropolitan statistical area:38060',
        'key': CENSUS_API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        print(f"✓ Fetched Phoenix MSA population data")

        # Save results
        output_file = f'{OUTPUT_DIR}/census_phoenix_population.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved to {output_file}")

        return data
    except Exception as e:
        print(f"✗ Error fetching population: {str(e)}")
        return None

# ============================================================================
# Zillow API Functions
# ============================================================================

def fetch_zillow_data():
    """
    Fetch Phoenix home price data from Zillow
    Note: Zillow deprecated their public API in 2021
    Alternative: Use publicly available Zillow Research Data
    """
    print("\n" + "="*80)
    print("FETCHING ZILLOW DATA (Home Prices)")
    print("="*80)

    print("\nNote: Zillow's public API was deprecated in 2021.")
    print("Recommended alternatives:")
    print("  1. Download Zillow Home Value Index (ZHVI) from:")
    print("     https://www.zillow.com/research/data/")
    print("  2. Use FRED series for Phoenix home prices:")
    print("     - PHXRNSA: Phoenix-Mesa-Scottsdale Home Price Index")

    # Fetch Phoenix home price index from FRED as alternative
    print("\nFetching Phoenix home prices from FRED...")
    phx_hpi = fetch_fred_series('PHXRNSA', start_date='2010-01-01')

    if not phx_hpi.empty:
        output_file = f'{OUTPUT_DIR}/fred_phoenix_home_prices.csv'
        phx_hpi.to_csv(output_file)
        print(f"✓ Saved Phoenix home price data to {output_file}")
        return phx_hpi

    return pd.DataFrame()

# ============================================================================
# IRS Migration Data
# ============================================================================

def fetch_irs_migration_info():
    """
    Provide information about IRS migration data
    Note: IRS migration data is not available via API
    """
    print("\n" + "="*80)
    print("IRS MIGRATION DATA (California to Arizona)")
    print("="*80)

    print("\nIRS migration data is NOT available via API.")
    print("Manual download required from:")
    print("  https://www.irs.gov/statistics/soi-tax-stats-migration-data")
    print("\nSteps:")
    print("  1. Download county-to-county migration files")
    print("  2. Filter for:")
    print("     - Origin: California counties")
    print("     - Destination: Arizona counties (Phoenix MSA)")
    print("  3. Aggregate annual net migration flows")
    print("\nData characteristics:")
    print("  - Frequency: Annual")
    print("  - Release lag: ~18 months")
    print("  - Latest available: Typically 2 years prior")

    return None

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Execute all data fetching operations"""
    print("\n" + "="*80)
    print("PHOENIX MSA MULTIFAMILY RENT GROWTH FORECASTING")
    print("DATA ACQUISITION PIPELINE")
    print("="*80)
    print(f"\nExecution started: {datetime.now()}")
    print(f"Output directory: {OUTPUT_DIR}/")

    # Create summary of data fetched
    summary = {
        'execution_date': datetime.now().isoformat(),
        'data_sources': {}
    }

    # 1. Fetch FRED data
    fred_data = fetch_all_fred_data()
    if not fred_data.empty:
        summary['data_sources']['FRED'] = {
            'status': 'success',
            'variables': list(fred_data.columns),
            'date_range': f"{fred_data.index.min()} to {fred_data.index.max()}",
            'observations': len(fred_data)
        }
    else:
        summary['data_sources']['FRED'] = {'status': 'failed'}

    # 2. Fetch BLS data
    bls_data = fetch_all_bls_data()
    if not bls_data.empty:
        summary['data_sources']['BLS'] = {
            'status': 'success',
            'variables': list(bls_data.columns),
            'date_range': f"{bls_data.index.min()} to {bls_data.index.max()}",
            'observations': len(bls_data)
        }
    else:
        summary['data_sources']['BLS'] = {'status': 'failed'}

    # 3. Fetch Census data
    census_permits = fetch_census_building_permits()
    census_pop = fetch_census_population()
    summary['data_sources']['Census'] = {
        'status': 'partial',
        'building_permits': 'success' if census_permits else 'failed',
        'population': 'success' if census_pop else 'failed'
    }

    # 4. Fetch Zillow/FRED home prices
    zillow_data = fetch_zillow_data()
    if not zillow_data.empty:
        summary['data_sources']['Zillow_FRED_Alternative'] = {
            'status': 'success',
            'date_range': f"{zillow_data.index.min()} to {zillow_data.index.max()}",
            'observations': len(zillow_data)
        }
    else:
        summary['data_sources']['Zillow_FRED_Alternative'] = {'status': 'failed'}

    # 5. IRS Migration info
    fetch_irs_migration_info()
    summary['data_sources']['IRS_Migration'] = {
        'status': 'manual_download_required',
        'note': 'No API available'
    }

    # Save summary
    summary_file = f'{OUTPUT_DIR}/data_fetch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("DATA ACQUISITION COMPLETE")
    print("="*80)
    print(f"\nSummary saved to: {summary_file}")
    print(f"Execution completed: {datetime.now()}")

    # Print summary
    print("\n" + "="*80)
    print("DATA FETCH SUMMARY")
    print("="*80)
    for source, info in summary['data_sources'].items():
        print(f"\n{source}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Download IRS migration data manually")
    print("2. Download Zillow ZHVI data from zillow.com/research/data")
    print("3. Consider CoStar subscription for:")
    print("   - Supply pipeline data (units under construction)")
    print("   - Rent and occupancy data")
    print("   - Absorption rates")
    print("   - Concession data")
    print("4. Run data cleaning and feature engineering pipeline")

if __name__ == '__main__':
    main()
