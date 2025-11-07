"""
FRED Series Discovery Tool
Search for Phoenix MSA-specific series in FRED database

Uses FRED API search endpoint to discover available data series
"""

import requests
import pandas as pd
from typing import List, Dict
import time


class FREDSeriesDiscovery:
    """Discover available FRED series for Phoenix MSA"""

    FRED_API_KEY = 'd043d26a9a4139438bb2a8d565bc01f7'
    SEARCH_URL = 'https://api.stlouisfed.org/fred/series/search'
    SERIES_INFO_URL = 'https://api.stlouisfed.org/fred/series'
    TAGS_URL = 'https://api.stlouisfed.org/fred/series/tags'

    # Phoenix-related search terms
    PHOENIX_SEARCH_TERMS = [
        'Phoenix',
        'Phoenix-Mesa-Scottsdale',
        'Phoenix MSA',
        'Arizona Phoenix',
        'PHX',
        'Maricopa County',
    ]

    def __init__(self):
        """Initialize discovery tool"""
        self.discovered_series = {}

    def search_fred(self, search_text: str, limit: int = 1000) -> List[Dict]:
        """
        Search FRED database for series matching text

        Parameters:
        -----------
        search_text : str
            Search query text
        limit : int
            Maximum number of results to return

        Returns:
        --------
        List[Dict]
            List of series metadata
        """
        params = {
            'search_text': search_text,
            'api_key': self.FRED_API_KEY,
            'file_type': 'json',
            'limit': limit,
            'search_type': 'full_text',  # Search in title and notes
        }

        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'seriess' in data:
                series_list = data['seriess']
                print(f"✓ Found {len(series_list)} series for '{search_text}'")
                return series_list
            else:
                print(f"✗ No series found for '{search_text}'")
                return []

        except requests.exceptions.RequestException as e:
            print(f"✗ Error searching FRED: {e}")
            return []

    def get_series_info(self, series_id: str) -> Dict:
        """
        Get detailed information about a specific series

        Parameters:
        -----------
        series_id : str
            FRED series identifier

        Returns:
        --------
        Dict
            Series metadata including title, units, frequency, etc.
        """
        params = {
            'series_id': series_id,
            'api_key': self.FRED_API_KEY,
            'file_type': 'json'
        }

        try:
            response = requests.get(self.SERIES_INFO_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'seriess' in data and len(data['seriess']) > 0:
                return data['seriess'][0]
            else:
                return {}

        except requests.exceptions.RequestException as e:
            print(f"✗ Error getting series info for {series_id}: {e}")
            return {}

    def discover_phoenix_series(self) -> pd.DataFrame:
        """
        Discover all Phoenix MSA-related series in FRED

        Returns:
        --------
        pd.DataFrame
            DataFrame with discovered series metadata
        """
        print("=" * 80)
        print("DISCOVERING PHOENIX MSA SERIES IN FRED")
        print("=" * 80)

        all_series = []

        for search_term in self.PHOENIX_SEARCH_TERMS:
            print(f"\nSearching for: '{search_term}'...")
            results = self.search_fred(search_term)

            for series in results:
                # Avoid duplicates
                if series['id'] not in self.discovered_series:
                    self.discovered_series[series['id']] = series
                    all_series.append(series)

            # Rate limiting (FRED allows 120 requests/minute)
            time.sleep(0.5)

        if all_series:
            df = pd.DataFrame(all_series)

            # Extract relevant columns
            columns_to_keep = [
                'id', 'title', 'frequency', 'units',
                'seasonal_adjustment', 'observation_start',
                'observation_end', 'popularity', 'notes'
            ]
            df = df[[col for col in columns_to_keep if col in df.columns]]

            # Sort by popularity
            if 'popularity' in df.columns:
                df = df.sort_values('popularity', ascending=False)

            print(f"\n✓ Discovered {len(df)} unique Phoenix series")
            return df
        else:
            print("✗ No Phoenix series found")
            return pd.DataFrame()

    def filter_series_by_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Filter series by category keywords

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame of discovered series
        category : str
            Category to filter (e.g., 'employment', 'price', 'housing')

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        category_keywords = {
            'employment': ['employ', 'job', 'labor', 'workforce'],
            'housing': ['house', 'housing', 'home', 'residential'],
            'price': ['price', 'index', 'cpi', 'inflation'],
            'income': ['income', 'wage', 'earnings', 'compensation'],
            'construction': ['construction', 'building', 'permit'],
            'population': ['population', 'demographic', 'migration'],
            'gdp': ['gdp', 'gross domestic product', 'economic output'],
        }

        keywords = category_keywords.get(category.lower(), [category])

        # Filter based on title and notes
        mask = df['title'].str.lower().str.contains('|'.join(keywords), na=False)

        if 'notes' in df.columns:
            notes_mask = df['notes'].str.lower().str.contains('|'.join(keywords), na=False)
            mask = mask | notes_mask

        filtered = df[mask].copy()
        print(f"✓ Found {len(filtered)} series in category '{category}'")
        return filtered

    def get_recommended_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to most relevant series for multifamily rent prediction

        Parameters:
        -----------
        df : pd.DataFrame
            All discovered series

        Returns:
        --------
        pd.DataFrame
            Recommended series for rent growth modeling
        """
        print("\n" + "=" * 80)
        print("FILTERING TO RECOMMENDED SERIES FOR RENT GROWTH PREDICTION")
        print("=" * 80)

        # Priority categories for rent growth prediction
        priority_categories = {
            'employment': 0.30,
            'housing': 0.25,
            'income': 0.20,
            'price': 0.15,
            'construction': 0.10,
        }

        recommended = []

        for category, weight in priority_categories.items():
            print(f"\nCategory: {category.upper()} (Weight: {weight:.0%})")
            cat_series = self.filter_series_by_category(df, category)

            if not cat_series.empty:
                # Select top 3 by popularity
                top_series = cat_series.head(3)
                for _, series in top_series.iterrows():
                    print(f"  • {series['id']}: {series['title']}")
                    recommended.append(series['id'])

        return df[df['id'].isin(recommended)]

    def export_discovery_results(self, df: pd.DataFrame, filepath: str):
        """Export discovery results to CSV"""
        try:
            df.to_csv(filepath, index=False)
            print(f"\n✓ Discovery results saved to: {filepath}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")


def main():
    """Main discovery workflow"""
    print("FRED Series Discovery for Phoenix MSA")
    print("=" * 80)

    # Initialize discovery tool
    discovery = FREDSeriesDiscovery()

    # Discover all Phoenix series
    all_series = discovery.discover_phoenix_series()

    if not all_series.empty:
        # Export full results
        discovery.export_discovery_results(
            all_series,
            'outputs/fred_phoenix_all_series.csv'
        )

        # Get recommended series
        recommended = discovery.get_recommended_series(all_series)

        if not recommended.empty:
            discovery.export_discovery_results(
                recommended,
                'outputs/fred_phoenix_recommended_series.csv'
            )

        # Display summary by category
        print("\n" + "=" * 80)
        print("SUMMARY BY CATEGORY")
        print("=" * 80)

        categories = ['employment', 'housing', 'price', 'income', 'construction', 'population']

        summary_data = []
        for category in categories:
            cat_series = discovery.filter_series_by_category(all_series, category)
            summary_data.append({
                'Category': category.capitalize(),
                'Series Count': len(cat_series),
                'Top Series': cat_series.iloc[0]['id'] if len(cat_series) > 0 else 'None'
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Display top 10 most popular series
        print("\n" + "=" * 80)
        print("TOP 10 MOST POPULAR PHOENIX SERIES")
        print("=" * 80)

        if 'popularity' in all_series.columns:
            top_10 = all_series.nlargest(10, 'popularity')[['id', 'title', 'frequency', 'popularity']]
            print(top_10.to_string(index=False))

    return all_series


if __name__ == '__main__':
    results = main()
