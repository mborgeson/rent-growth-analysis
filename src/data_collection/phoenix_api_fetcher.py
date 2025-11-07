"""
Phoenix MSA Economic Data Fetcher
Retrieves variables from FRED and BLS APIs for multifamily rent growth prediction

Author: Claude Code
Date: 2025-01-07
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json


class PhoenixDataFetcher:
    """Fetch Phoenix MSA economic data from FRED and BLS APIs"""

    # API Configuration
    FRED_API_KEY = 'd043d26a9a4139438bb2a8d565bc01f7'
    FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'
    BLS_API_URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    # Phoenix MSA identifiers
    PHOENIX_MSA_CODE = '38060'
    ARIZONA_STATE_CODE = '04'

    # Series definitions
    FRED_SERIES = {
        # Phoenix-Specific
        'phoenix_home_price_index': 'PHXRNSA',

        # National variables used in Phoenix model
        'mortgage_rate_30yr': 'MORTGAGE30US',
        'fed_funds_rate': 'DFF',
        'inflation_expectations_5yr': 'T5YIE',
        'national_employment': 'PAYEMS',
        'national_unemployment': 'UNRATE',
        'consumer_price_index': 'CPIAUCSL',
        'real_gdp': 'GDPC1',
    }

    BLS_SERIES = {
        # Phoenix MSA Employment Series
        'phoenix_total_nonfarm': 'SMU04383400000000001',
        'phoenix_prof_business_services': 'SMU04380608000000001',
        'phoenix_construction': 'SMU04382000000000001',
        'phoenix_manufacturing': 'SMU04383000000000001',
        'phoenix_information': 'SMU04385000000000001',
        'phoenix_financial_activities': 'SMU04385500000000001',
        'phoenix_leisure_hospitality': 'SMU04387000000000001',
        'phoenix_education_health': 'SMU04386500000000001',
    }

    def __init__(self, bls_api_key: Optional[str] = None):
        """
        Initialize fetcher

        Parameters:
        -----------
        bls_api_key : str, optional
            BLS API key for higher rate limits (500 requests/day vs 25)
            Register at: https://data.bls.gov/registrationEngine/
        """
        self.bls_api_key = bls_api_key
        self.data_cache = {}

    def fetch_fred_series(self,
                         series_id: str,
                         start_date: str = '2010-01-01',
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch single series from FRED API

        Parameters:
        -----------
        series_id : str
            FRED series identifier
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)

        Returns:
        --------
        pd.DataFrame
            Time series data with date index
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        params = {
            'series_id': series_id,
            'api_key': self.FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        try:
            response = requests.get(self.FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'observations' not in data:
                print(f"Warning: No observations found for {series_id}")
                return pd.DataFrame()

            # Parse observations
            observations = data['observations']
            df = pd.DataFrame(observations)

            # Clean and format
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].set_index('date')
            df.columns = [series_id]

            # Remove missing values (marked as '.')
            df = df[df[series_id].notna()]

            print(f"✓ Fetched {len(df)} observations for {series_id}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching FRED series {series_id}: {e}")
            return pd.DataFrame()

    def fetch_all_fred_series(self,
                              start_date: str = '2010-01-01') -> pd.DataFrame:
        """
        Fetch all FRED series defined in FRED_SERIES

        Parameters:
        -----------
        start_date : str
            Start date for all series

        Returns:
        --------
        pd.DataFrame
            Combined dataframe with all FRED series
        """
        print("=" * 60)
        print("FETCHING FRED DATA")
        print("=" * 60)

        all_data = []

        for name, series_id in self.FRED_SERIES.items():
            print(f"\nFetching {name} ({series_id})...")
            df = self.fetch_fred_series(series_id, start_date)

            if not df.empty:
                df.columns = [name]
                all_data.append(df)
                self.data_cache[name] = df

            # Rate limiting (FRED allows 120 requests/minute)
            time.sleep(0.5)

        if all_data:
            combined = pd.concat(all_data, axis=1)
            print(f"\n✓ Successfully fetched {len(all_data)} FRED series")
            print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
            print(f"  Total observations: {len(combined)}")
            return combined
        else:
            print("✗ No FRED data fetched")
            return pd.DataFrame()

    def fetch_bls_series(self,
                        series_id: str,
                        start_year: int = 2015,
                        end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch single series from BLS API

        Parameters:
        -----------
        series_id : str
            BLS series identifier
        start_year : int
            Start year (BLS limits to 20 years max)
        end_year : int, optional
            End year (default: current year)

        Returns:
        --------
        pd.DataFrame
            Time series data with date index
        """
        if end_year is None:
            end_year = datetime.now().year

        # BLS API limits to 20 years per request
        if end_year - start_year > 20:
            start_year = end_year - 20
            print(f"Warning: Limiting to 20 years ({start_year}-{end_year})")

        payload = {
            'seriesid': [series_id],
            'startyear': str(start_year),
            'endyear': str(end_year)
        }

        # Add API key if available
        if self.bls_api_key:
            payload['registrationkey'] = self.bls_api_key

        try:
            response = requests.post(self.BLS_API_URL, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'REQUEST_SUCCEEDED':
                print(f"Warning: BLS request failed for {series_id}")
                if 'message' in data:
                    print(f"  Message: {data['message']}")
                return pd.DataFrame()

            # Parse series data
            series_data = data['Results']['series'][0]['data']

            # Convert to DataFrame
            records = []
            for item in series_data:
                # BLS format: year, period (M01-M12), value
                year = int(item['year'])
                period = item['period']

                # Handle monthly data (M01-M12)
                if period.startswith('M'):
                    month = int(period[1:])
                    date = pd.Timestamp(year=year, month=month, day=1)
                    value = float(item['value'])
                    records.append({'date': date, 'value': value})

            if not records:
                print(f"Warning: No data records for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.set_index('date').sort_index()
            df.columns = [series_id]

            print(f"✓ Fetched {len(df)} observations for {series_id}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching BLS series {series_id}: {e}")
            return pd.DataFrame()
        except (KeyError, ValueError) as e:
            print(f"✗ Error parsing BLS data for {series_id}: {e}")
            return pd.DataFrame()

    def fetch_all_bls_series(self, start_year: int = 2015) -> pd.DataFrame:
        """
        Fetch all BLS series defined in BLS_SERIES

        Parameters:
        -----------
        start_year : int
            Start year for all series

        Returns:
        --------
        pd.DataFrame
            Combined dataframe with all BLS series
        """
        print("\n" + "=" * 60)
        print("FETCHING BLS DATA")
        print("=" * 60)

        if not self.bls_api_key:
            print("\nNote: No BLS API key provided")
            print("  - Rate limited to 25 requests/day")
            print("  - Register at: https://data.bls.gov/registrationEngine/")
            print("  - With key: 500 requests/day\n")

        all_data = []

        for name, series_id in self.BLS_SERIES.items():
            print(f"\nFetching {name} ({series_id})...")
            df = self.fetch_bls_series(series_id, start_year)

            if not df.empty:
                df.columns = [name]
                all_data.append(df)
                self.data_cache[name] = df

            # Rate limiting
            # Without key: 25 requests/day, 10/minute
            # With key: 500 requests/day, 120/minute
            if self.bls_api_key:
                time.sleep(0.5)  # 120 per minute
            else:
                time.sleep(6)  # 10 per minute

        if all_data:
            combined = pd.concat(all_data, axis=1)
            print(f"\n✓ Successfully fetched {len(all_data)} BLS series")
            print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
            print(f"  Total observations: {len(combined)}")
            return combined
        else:
            print("✗ No BLS data fetched")
            return pd.DataFrame()

    def fetch_all_data(self,
                      start_date: str = '2010-01-01',
                      start_year: int = 2015) -> pd.DataFrame:
        """
        Fetch all FRED and BLS data and combine into single DataFrame

        Parameters:
        -----------
        start_date : str
            Start date for FRED data (YYYY-MM-DD)
        start_year : int
            Start year for BLS data

        Returns:
        --------
        pd.DataFrame
            Combined dataset with all variables
        """
        # Fetch FRED data
        fred_data = self.fetch_all_fred_series(start_date)

        # Fetch BLS data
        bls_data = self.fetch_all_bls_series(start_year)

        # Combine datasets
        if not fred_data.empty and not bls_data.empty:
            combined = pd.concat([fred_data, bls_data], axis=1)
            print("\n" + "=" * 60)
            print("COMBINED DATASET SUMMARY")
            print("=" * 60)
            print(f"Total variables: {len(combined.columns)}")
            print(f"Date range: {combined.index.min()} to {combined.index.max()}")
            print(f"Total rows: {len(combined)}")
            print(f"\nVariables:")
            for col in combined.columns:
                non_null = combined[col].notna().sum()
                print(f"  - {col}: {non_null} observations")
            return combined
        elif not fred_data.empty:
            return fred_data
        elif not bls_data.empty:
            return bls_data
        else:
            print("✗ No data fetched from either source")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save data to CSV file"""
        try:
            df.to_csv(filepath)
            print(f"\n✓ Data saved to: {filepath}")
        except Exception as e:
            print(f"✗ Error saving data: {e}")

    def get_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for fetched data"""
        summary = pd.DataFrame({
            'Variable': df.columns,
            'Observations': [df[col].notna().sum() for col in df.columns],
            'Missing': [df[col].isna().sum() for col in df.columns],
            'Start Date': [df[col].first_valid_index() for col in df.columns],
            'End Date': [df[col].last_valid_index() for col in df.columns],
            'Mean': [df[col].mean() for col in df.columns],
            'Std Dev': [df[col].std() for col in df.columns],
        })
        return summary


def main():
    """Example usage"""
    print("Phoenix MSA Economic Data Fetcher")
    print("=" * 60)

    # Initialize fetcher
    # To use BLS API key, pass it here: PhoenixDataFetcher(bls_api_key='YOUR_KEY')
    fetcher = PhoenixDataFetcher()

    # Fetch all data
    data = fetcher.fetch_all_data(
        start_date='2015-01-01',
        start_year=2015
    )

    # Generate summary
    if not data.empty:
        summary = fetcher.get_data_summary(data)
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(summary.to_string(index=False))

        # Save data
        output_file = 'outputs/phoenix_economic_data.csv'
        fetcher.save_data(data, output_file)

        # Save summary
        summary_file = 'outputs/phoenix_data_summary.csv'
        summary.to_csv(summary_file, index=False)
        print(f"✓ Summary saved to: {summary_file}")

    return data


if __name__ == '__main__':
    data = main()
