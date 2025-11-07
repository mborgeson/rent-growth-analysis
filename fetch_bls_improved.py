#!/usr/bin/env python3
"""
Improved BLS data fetching using publicly available data
"""

import pandas as pd
import requests
import json
import time

def fetch_bls_v1(series_id):
    """Fetch BLS data using public v1 API"""
    base_url = f'https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}'

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'REQUEST_SUCCEEDED':
            series_data = data['Results']['series'][0]['data']

            # Convert to DataFrame
            records = []
            for item in series_data:
                if item['period'].startswith('M'):
                    month = item['period'].replace('M', '')
                    year = item['year']
                    date_str = f"{year}-{month.zfill(2)}-01"
                    records.append({
                        'date': date_str,
                        'value': float(item['value'])
                    })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df.columns = [series_id]

            print(f"✓ Fetched {series_id}: {len(df)} observations")
            return df
        else:
            print(f"✗ Failed to fetch {series_id}")
            return pd.DataFrame()

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return pd.DataFrame()

# Phoenix MSA Employment Series
phoenix_series = {
    'SMU04383400000000001': 'Phoenix MSA Total Nonfarm',
    'SMU04383406000000001': 'Phoenix Professional & Business Services',
    'SMU04383405000000001': 'Phoenix Information Sector',
    'SMU04383403100000001': 'Phoenix Manufacturing'
}

print("Fetching Phoenix MSA Employment Data from BLS...")
print("="*60)

all_data = []

for series_id, description in phoenix_series.items():
    print(f"\n{description}")
    df = fetch_bls_v1(series_id)
    if not df.empty:
        all_data.append(df)
    time.sleep(1)  # Rate limiting

if all_data:
    combined = pd.concat(all_data, axis=1)
    output_file = 'data/raw/bls_phoenix_employment_v1.csv'
    combined.to_csv(output_file)
    print(f"\n✓ Saved to {output_file}")
    print(f"  Shape: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
else:
    print("\n✗ No data fetched")
