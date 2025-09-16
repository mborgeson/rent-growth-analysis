"""
FRED API Data Collector for Multifamily Rent Growth Analysis
Handles data collection from Federal Reserve Economic Data (FRED)
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FREDCollector:
    """
    Collector for Federal Reserve Economic Data (FRED) API
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Key economic indicators for rent growth analysis
    CORE_SERIES = {
        'economic': {
            'FEDFUNDS': 'Federal Funds Effective Rate',
            'DGS2': '2-Year Treasury Constant Maturity Rate',
            'DGS5': '5-Year Treasury Constant Maturity Rate', 
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average',
            'GDPC1': 'Real Gross Domestic Product',
            'GDPPOT': 'Real Potential Gross Domestic Product',
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'CPILFESL': 'Consumer Price Index: All Items Less Food & Energy',
            'PPIACO': 'Producer Price Index by Commodity',
        },
        'financial': {
            'SP500': 'S&P 500',
            'DEXUSEU': 'US Dollar to Euro Exchange Rate',
            'BAMLH0A0HYM2': 'ICE BofA US High Yield Index Option-Adjusted Spread',
            'BAMLC0A0CM': 'ICE BofA US Corporate Index Option-Adjusted Spread',
            'TEDRATE': 'TED Spread',
        },
        'housing': {
            'HOUST': 'Housing Starts: Total',
            'PERMIT': 'New Private Housing Units Authorized by Building Permits',
            'CSUSHPISA': 'S&P/Case-Shiller U.S. National Home Price Index',
            'MSPUS': 'Median Sales Price of Houses Sold',
            'RHVACBW': 'Rental Vacancy Rate',
            'RHORUSQ156N': 'Homeownership Rate',
        },
        'demographic': {
            'POPTHM': 'Population',
            'CIVPART': 'Labor Force Participation Rate',
            'MEHOINUSA672N': 'Real Median Household Income',
        }
    }
    
    # MSA codes for target markets
    MSA_CODES = {
        'phoenix': '38060',
        'austin': '12420',
        'dallas': '19100',
        'denver': '19740',
        'salt_lake_city': '41620',
        'nashville': '34980',
        'miami': '33100'
    }
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./data/cache"):
        """
        Initialize FRED API collector
        
        Args:
            api_key: FRED API key (if None, will try to load from environment)
            cache_dir: Directory for caching API responses
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            logger.warning("No FRED API key provided. Some functionality may be limited.")
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup session with retry strategy
        self.session = self._setup_session()
        
    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _get_cache_key(self, series_id: str, **params) -> str:
        """Generate cache key for request"""
        params_str = json.dumps(params, sort_keys=True)
        key_str = f"{series_id}_{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and fresh"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=max_age_hours):
                try:
                    return pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        try:
            data.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_series(self, 
                   series_id: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   frequency: Optional[str] = None,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch a single series from FRED
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            frequency: Data frequency (d, w, bw, m, q, sa, a)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with date index and series values
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        if frequency:
            params['frequency'] = frequency
            
        # Check cache
        cache_params = {k: v for k, v in params.items() if k != 'series_id'}
        cache_key = self._get_cache_key(series_id, **cache_params)
        if use_cache:
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {series_id}")
                return cached_data
        
        # Fetch from API
        url = f"{self.BASE_URL}/series/observations"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' not in data:
                logger.error(f"No observations found for {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date')[['value']]
            df.columns = [series_id]
            
            # Save to cache
            if use_cache:
                self._save_to_cache(df, cache_key)
            
            logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.DataFrame()
    
    def get_multiple_series(self,
                           series_ids: List[str],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           frequency: Optional[str] = 'm') -> pd.DataFrame:
        """
        Fetch multiple series and combine into single DataFrame
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date for all series
            end_date: End date for all series
            frequency: Data frequency
            
        Returns:
            DataFrame with all series as columns
        """
        all_data = []
        
        for series_id in series_ids:
            logger.info(f"Fetching {series_id}...")
            df = self.get_series(series_id, start_date, end_date, frequency)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.5)  # Rate limiting
        
        if all_data:
            combined = pd.concat(all_data, axis=1)
            # Forward fill missing values (common in economic data)
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            return combined
        
        return pd.DataFrame()
    
    def get_msa_series(self,
                      base_series: str,
                      msa_name: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch MSA-specific series
        
        Args:
            base_series: Base series name
            msa_name: MSA name (must be in MSA_CODES)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with MSA-specific data
        """
        if msa_name.lower() not in self.MSA_CODES:
            logger.error(f"Unknown MSA: {msa_name}")
            return pd.DataFrame()
        
        # Construct MSA-specific series ID
        # This varies by series type - would need specific mapping
        msa_code = self.MSA_CODES[msa_name.lower()]
        
        # Example for unemployment rate: 
        # National: UNRATE, Phoenix MSA: PHOE004UR
        # This would need proper mapping for each series type
        
        return self.get_series(base_series, start_date, end_date)
    
    def get_all_core_series(self,
                            start_date: str = "1985-01-01",
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all core series organized by category
        
        Args:
            start_date: Start date for data collection
            end_date: End date (defaults to current date)
            
        Returns:
            Dictionary with categories as keys and DataFrames as values
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        results = {}
        
        for category, series_dict in self.CORE_SERIES.items():
            logger.info(f"Fetching {category} indicators...")
            series_ids = list(series_dict.keys())
            df = self.get_multiple_series(series_ids, start_date, end_date, 'm')
            if not df.empty:
                results[category] = df
        
        return results
    
    def search_series(self, 
                     search_text: str,
                     limit: int = 100) -> pd.DataFrame:
        """
        Search for FRED series by text
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            DataFrame with series information
        """
        params = {
            'search_text': search_text,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit
        }
        
        url = f"{self.BASE_URL}/series/search"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'seriess' in data:
                return pd.DataFrame(data['seriess'])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {e}")
        
        return pd.DataFrame()


def main():
    """Example usage of FRED collector"""
    
    # Initialize collector
    collector = FREDCollector()
    
    # Fetch key economic indicators
    print("Fetching economic indicators...")
    economic_data = collector.get_multiple_series(
        ['FEDFUNDS', 'DGS10', 'UNRATE', 'CPIAUCSL'],
        start_date='2020-01-01',
        frequency='m'
    )
    
    if not economic_data.empty:
        print("\nEconomic Indicators Summary:")
        print(economic_data.describe())
        print("\nLatest Values:")
        print(economic_data.tail())
    
    # Fetch all core series
    print("\nFetching all core series...")
    all_data = collector.get_all_core_series(start_date='2020-01-01')
    
    for category, df in all_data.items():
        print(f"\n{category.upper()} - Shape: {df.shape}")
        print(f"Series: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()