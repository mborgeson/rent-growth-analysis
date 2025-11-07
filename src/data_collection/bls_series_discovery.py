"""
BLS Series Discovery Tool for Phoenix MSA
Construct and validate BLS series IDs for Phoenix industries

BLS doesn't have a search API, so this tool:
1. Explains BLS series ID structure
2. Generates candidate series IDs for Phoenix industries
3. Validates which series exist via API calls
"""

import requests
import pandas as pd
from typing import List, Dict, Tuple
import time
from itertools import product


class BLSSeriesDiscovery:
    """
    Discover available BLS employment series for Phoenix MSA

    BLS Series ID Structure (State & Metro Area Employment):
    SMU + State + Area + Supersector + Industry + Data Type
    SMU  + 04   + 38060 + SS        + IIIIII  + DDDDDDDD

    Example: SMU04380608000000001
    - SMU: Series prefix (State & Metro)
    - 04: Arizona
    - 38060: Phoenix-Mesa-Scottsdale MSA
    - 08: Supersector (Professional & Business Services)
    - 000000: No industry detail
    - 01: Employment level
    """

    BLS_API_URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    # Phoenix MSA identifiers
    PHOENIX_MSA_CODE = '38060'
    ARIZONA_STATE_CODE = '04'

    # Supersector codes
    SUPERSECTORS = {
        '00': 'Total Nonfarm',
        '05': 'Total Private',
        '06': 'Goods Producing',
        '07': 'Service Providing',
        '08': 'Private Service Providing',
        '10': 'Mining and Logging',
        '20': 'Construction',
        '30': 'Manufacturing',
        '31': 'Durable Goods',
        '32': 'Nondurable Goods',
        '40': 'Trade, Transportation, and Utilities',
        '41': 'Wholesale Trade',
        '42': 'Retail Trade',
        '43': 'Transportation and Warehousing',
        '44': 'Utilities',
        '50': 'Information',
        '55': 'Financial Activities',
        '60': 'Professional and Business Services',
        '65': 'Education and Health Services',
        '70': 'Leisure and Hospitality',
        '80': 'Other Services',
        '90': 'Government',
    }

    # NAICS industry codes (3-digit) for detailed industries
    NAICS_3_DIGIT = {
        # Construction (20)
        '236': 'Construction of Buildings',
        '237': 'Heavy and Civil Engineering Construction',
        '238': 'Specialty Trade Contractors',

        # Manufacturing (30)
        '334': 'Computer and Electronic Product Manufacturing',
        '335': 'Electrical Equipment Manufacturing',
        '336': 'Transportation Equipment Manufacturing',

        # Wholesale Trade (41)
        '423': 'Merchant Wholesalers, Durable Goods',
        '424': 'Merchant Wholesalers, Nondurable Goods',

        # Retail Trade (42)
        '441': 'Motor Vehicle and Parts Dealers',
        '445': 'Food and Beverage Stores',
        '452': 'General Merchandise Stores',

        # Information (50)
        '511': 'Publishing Industries',
        '517': 'Telecommunications',
        '518': 'Data Processing, Hosting, and Related Services',
        '519': 'Other Information Services',

        # Financial Activities (55)
        '522': 'Credit Intermediation and Related Activities',
        '523': 'Securities, Commodity Contracts, Investments',
        '524': 'Insurance Carriers and Related Activities',
        '525': 'Funds, Trusts, and Other Financial Vehicles',
        '531': 'Real Estate',

        # Professional & Business Services (60)
        '541': 'Professional, Scientific, and Technical Services',
        '561': 'Administrative and Support Services',
        '562': 'Waste Management and Remediation Services',

        # Education & Health Services (65)
        '611': 'Educational Services',
        '621': 'Ambulatory Health Care Services',
        '622': 'Hospitals',
        '623': 'Nursing and Residential Care Facilities',
        '624': 'Social Assistance',

        # Leisure & Hospitality (70)
        '713': 'Amusement, Gambling, and Recreation',
        '721': 'Accommodation',
        '722': 'Food Services and Drinking Places',
    }

    # Data type codes
    DATA_TYPES = {
        '01': 'All Employees (Employment Level)',
        '02': 'All Employees, 3-Month Average',
        '03': 'Female Employees',
        '04': 'Production Employees',
        '06': 'Hours, All Employees',
        '07': 'Hours, Production Employees',
        '11': 'Earnings, All Employees',
    }

    def __init__(self, bls_api_key: str = None):
        """Initialize BLS discovery tool"""
        self.bls_api_key = bls_api_key
        self.validated_series = {}

    def construct_series_id(self,
                           state_code: str = '04',
                           msa_code: str = '38060',
                           supersector: str = '00',
                           industry: str = '000000',
                           data_type: str = '01') -> str:
        """
        Construct BLS series ID

        Parameters:
        -----------
        state_code : str
            State FIPS code (04 = Arizona)
        msa_code : str
            MSA code (38060 = Phoenix)
        supersector : str
            2-digit supersector code
        industry : str
            6-digit industry code (000000 = total)
        data_type : str
            Data type code (01 = employment)

        Returns:
        --------
        str
            Complete BLS series ID
        """
        return f"SMU{state_code}{msa_code}{supersector}{industry}{data_type}"

    def validate_series(self, series_id: str) -> Tuple[bool, Dict]:
        """
        Validate if BLS series exists by fetching data

        Parameters:
        -----------
        series_id : str
            BLS series ID to validate

        Returns:
        --------
        Tuple[bool, Dict]
            (exists, metadata) where exists is True if series has data
        """
        payload = {
            'seriesid': [series_id],
            'startyear': '2023',
            'endyear': '2024'
        }

        if self.bls_api_key:
            payload['registrationkey'] = self.bls_api_key

        try:
            response = requests.post(self.BLS_API_URL, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] == 'REQUEST_SUCCEEDED':
                series_data = data['Results']['series'][0]

                if 'data' in series_data and len(series_data['data']) > 0:
                    return True, {
                        'series_id': series_id,
                        'exists': True,
                        'observations': len(series_data['data']),
                        'latest_value': series_data['data'][0]['value'],
                        'latest_period': f"{series_data['data'][0]['year']}-{series_data['data'][0]['period']}"
                    }

            return False, {'series_id': series_id, 'exists': False}

        except Exception as e:
            return False, {'series_id': series_id, 'exists': False, 'error': str(e)}

    def generate_supersector_series(self) -> List[Dict]:
        """
        Generate all supersector series IDs for Phoenix MSA

        Returns:
        --------
        List[Dict]
            List of series metadata
        """
        print("=" * 80)
        print("GENERATING PHOENIX MSA SUPERSECTOR EMPLOYMENT SERIES")
        print("=" * 80)

        series_list = []

        for ss_code, ss_name in self.SUPERSECTORS.items():
            series_id = self.construct_series_id(
                supersector=ss_code,
                industry='000000',
                data_type='01'  # Employment level
            )

            series_list.append({
                'series_id': series_id,
                'supersector_code': ss_code,
                'supersector_name': ss_name,
                'industry_code': '000000',
                'industry_name': 'Total',
                'data_type': '01',
                'data_type_name': 'Employment Level'
            })

        print(f"✓ Generated {len(series_list)} supersector series")
        return series_list

    def generate_industry_series(self) -> List[Dict]:
        """
        Generate detailed industry series IDs for Phoenix MSA

        Returns:
        --------
        List[Dict]
            List of series metadata
        """
        print("\n" + "=" * 80)
        print("GENERATING PHOENIX MSA DETAILED INDUSTRY EMPLOYMENT SERIES")
        print("=" * 80)

        series_list = []

        for naics_code, naics_name in self.NAICS_3_DIGIT.items():
            # Determine supersector from NAICS code
            naics_first_digit = naics_code[0]

            supersector_mapping = {
                '2': '20',  # Construction
                '3': '30',  # Manufacturing
                '4': {'1': '41', '2': '42', '3': '43', '4': '44', '5': '44'},  # Trade/Transport/Utilities
                '5': {'1': '50', '2': '55', '3': '55'},  # Information/Financial
                '6': '65',  # Education/Health
                '7': '70',  # Leisure/Hospitality
            }

            if naics_first_digit == '4':
                supersector = supersector_mapping['4'].get(naics_code[1], '40')
            elif naics_first_digit == '5':
                supersector = supersector_mapping['5'].get(naics_code[1], '50')
            else:
                supersector = supersector_mapping.get(naics_first_digit, '00')

            # Construct industry code (NAICS + 000)
            industry_code = f"{naics_code}000"

            series_id = self.construct_series_id(
                supersector=supersector,
                industry=industry_code,
                data_type='01'
            )

            series_list.append({
                'series_id': series_id,
                'supersector_code': supersector,
                'supersector_name': self.SUPERSECTORS.get(supersector, 'Unknown'),
                'industry_code': industry_code,
                'industry_name': naics_name,
                'naics_code': naics_code,
                'data_type': '01',
                'data_type_name': 'Employment Level'
            })

        print(f"✓ Generated {len(series_list)} industry series")
        return series_list

    def validate_series_batch(self,
                             series_list: List[Dict],
                             max_validate: int = 50) -> pd.DataFrame:
        """
        Validate batch of series IDs

        Parameters:
        -----------
        series_list : List[Dict]
            List of series metadata
        max_validate : int
            Maximum number to validate (BLS rate limits)

        Returns:
        --------
        pd.DataFrame
            Validated series with existence status
        """
        print(f"\nValidating series (max {max_validate})...")

        validated = []

        for i, series in enumerate(series_list[:max_validate]):
            if i > 0 and i % 10 == 0:
                print(f"  Validated {i}/{min(len(series_list), max_validate)}...")

            series_id = series['series_id']
            exists, metadata = self.validate_series(series_id)

            series['exists'] = exists
            if exists:
                series['observations'] = metadata.get('observations', 0)
                series['latest_value'] = metadata.get('latest_value', None)

            validated.append(series)

            # Rate limiting
            if self.bls_api_key:
                time.sleep(0.5)  # 120/min with key
            else:
                time.sleep(6)  # 10/min without key

        df = pd.DataFrame(validated)
        existing_count = df['exists'].sum()

        print(f"✓ Validation complete: {existing_count}/{len(validated)} series exist")

        return df

    def get_recommended_phoenix_series(self) -> List[str]:
        """
        Get recommended high-value series for Phoenix rent growth model

        Returns:
        --------
        List[str]
            List of recommended series IDs
        """
        # Based on multifamily rent growth framework
        recommendations = {
            # Tier 1: Core Predictors
            'phoenix_total_nonfarm': 'SMU04383400000000001',
            'phoenix_prof_business_services': 'SMU04380608000000001',
            'phoenix_construction': 'SMU04382000000000001',

            # Tier 2: Phoenix-Specific Industries
            'phoenix_semiconductor_mfg': 'SMU04383033440000001',  # NAICS 3344
            'phoenix_data_processing': 'SMU04385051800000001',  # NAICS 518
            'phoenix_back_office': 'SMU04380656100000001',  # NAICS 561

            # Tier 3: Supporting Industries
            'phoenix_real_estate': 'SMU04385553100000001',  # NAICS 531
            'phoenix_leisure_hospitality': 'SMU04387000000000001',
            'phoenix_education_health': 'SMU04386500000000001',
            'phoenix_financial_activities': 'SMU04385500000000001',
            'phoenix_information': 'SMU04385000000000001',
            'phoenix_manufacturing': 'SMU04383000000000001',
        }

        return recommendations


def main():
    """Main discovery workflow"""
    print("BLS SERIES DISCOVERY FOR PHOENIX MSA")
    print("=" * 80)
    print("\nBLS Series ID Structure:")
    print("  SMU + State + MSA + Supersector + Industry + DataType")
    print("  SMU + 04    + 38060 + SS        + IIIIII  + DDDDDDDD")
    print("\nExample: SMU04380608000000001")
    print("  - SMU: State & Metro series")
    print("  - 04: Arizona")
    print("  - 38060: Phoenix MSA")
    print("  - 08: Professional & Business Services")
    print("  - 000000: Total (no industry detail)")
    print("  - 01: Employment level\n")

    discovery = BLSSeriesDiscovery()

    # Generate supersector series
    supersector_series = discovery.generate_supersector_series()

    # Generate industry series
    industry_series = discovery.generate_industry_series()

    # Combine all generated series
    all_series = supersector_series + industry_series

    print(f"\n" + "=" * 80)
    print(f"TOTAL GENERATED SERIES: {len(all_series)}")
    print("=" * 80)

    # Export all generated series
    all_df = pd.DataFrame(all_series)
    all_df.to_csv('/home/mattb/Rent Growth Analysis/outputs/bls_phoenix_generated_series.csv', index=False)
    print(f"\n✓ All generated series saved to: outputs/bls_phoenix_generated_series.csv")

    # Get recommended series
    print("\n" + "=" * 80)
    print("RECOMMENDED HIGH-VALUE SERIES FOR RENT GROWTH MODEL")
    print("=" * 80)

    recommended = discovery.get_recommended_phoenix_series()

    rec_df = pd.DataFrame([
        {'name': name, 'series_id': series_id}
        for name, series_id in recommended.items()
    ])

    print(rec_df.to_string(index=False))

    rec_df.to_csv('/home/mattb/Rent Growth Analysis/outputs/bls_phoenix_recommended_series.csv', index=False)
    print(f"\n✓ Recommended series saved to: outputs/bls_phoenix_recommended_series.csv")

    # Validate supersector series (quick check)
    print("\n" + "=" * 80)
    print("VALIDATING SUPERSECTOR SERIES (Sample)")
    print("=" * 80)

    # Validate first 10 supersector series
    validated_df = discovery.validate_series_batch(supersector_series, max_validate=10)

    if not validated_df.empty:
        print("\nValidated Series:")
        print(validated_df[['series_id', 'supersector_name', 'exists']].to_string(index=False))

        validated_df.to_csv('/home/mattb/Rent Growth Analysis/outputs/bls_phoenix_validated_sample.csv', index=False)
        print(f"\n✓ Validation results saved to: outputs/bls_phoenix_validated_sample.csv")

    return all_series, recommended


if __name__ == '__main__':
    all_series, recommended = main()
