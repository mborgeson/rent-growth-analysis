"""
Test script for Phoenix Data Fetcher
Quick test to verify API connections work
"""

from phoenix_api_fetcher import PhoenixDataFetcher
import pandas as pd

def quick_test():
    """Run quick test of fetcher"""
    print("Testing Phoenix Data Fetcher")
    print("=" * 60)

    fetcher = PhoenixDataFetcher()

    # Test single FRED series
    print("\n1. Testing FRED API (Phoenix Home Price Index)...")
    phoenix_hpi = fetcher.fetch_fred_series('PHXRNSA', start_date='2020-01-01')
    if not phoenix_hpi.empty:
        print(f"   ✓ Success! Latest value: {phoenix_hpi.iloc[-1].values[0]:.2f}")
        print(f"   ✓ Latest date: {phoenix_hpi.index[-1]}")

    # Test single BLS series
    print("\n2. Testing BLS API (Phoenix Prof/Business Services Employment)...")
    phoenix_emp = fetcher.fetch_bls_series('SMU04380608000000001', start_year=2020)
    if not phoenix_emp.empty:
        print(f"   ✓ Success! Latest value: {phoenix_emp.iloc[-1].values[0]:.1f}")
        print(f"   ✓ Latest date: {phoenix_emp.index[-1]}")

    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("\nTo fetch all data, run: python phoenix_api_fetcher.py")

if __name__ == '__main__':
    quick_test()
