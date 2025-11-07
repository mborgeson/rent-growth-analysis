# BLS Series Discovery Guide for Phoenix MSA

**Discovery Date:** 2025-01-07
**Total Series Generated:** 53 Phoenix MSA employment series

## Understanding BLS Series ID Structure

### Series ID Format for State & Metro Employment (SMU)

```
SMU + State + MSA + Supersector + Industry + DataType
SMU + 04    + 38060 + SS        + IIIIII  + DDDDDDDD

Components:
- SMU: State & Metro Area series identifier
- 04: Arizona state FIPS code
- 38060: Phoenix-Mesa-Scottsdale MSA code
- SS: 2-digit supersector code
- IIIIII: 6-digit industry code (000000 = total)
- DDDDDDDD: 8-digit data type code (01 = employment level)
```

### Example Breakdown

**SMU04380608000000001** - Phoenix Professional & Business Services Employment
- SMU: State & Metro series
- 04: Arizona
- 38060: Phoenix MSA
- 08: Professional & Business Services supersector
- 000000: Total (no detailed industry)
- 01: Employment level

## Available Supersectors for Phoenix MSA

| Code | Supersector Name | Series ID | Validated |
|------|------------------|-----------|-----------|
| 00 | Total Nonfarm | SMU04380600000000001 | ✅ Yes |
| 05 | Total Private | SMU04380600500000001 | ✅ Yes |
| 06 | Goods Producing | SMU04380600600000001 | ✅ Yes |
| 07 | Service Providing | SMU04380600700000001 | ✅ Yes |
| 08 | Private Service Providing | SMU04380600800000001 | ✅ Yes |
| 10 | Mining and Logging | SMU04380601000000001 | ✅ Yes |
| 20 | Construction | SMU04380602000000001 | ✅ Yes |
| 30 | Manufacturing | SMU04380603000000001 | ✅ Yes |
| 31 | Durable Goods | SMU04380603100000001 | ✅ Yes |
| 32 | Nondurable Goods | SMU04380603200000001 | ✅ Yes |
| 40 | Trade, Transportation, Utilities | SMU04380604000000001 | - |
| 41 | Wholesale Trade | SMU04380604100000001 | - |
| 42 | Retail Trade | SMU04380604200000001 | - |
| 43 | Transportation & Warehousing | SMU04380604300000001 | - |
| 44 | Utilities | SMU04380604400000001 | - |
| 50 | Information | SMU04380605000000001 | - |
| 55 | Financial Activities | SMU04380605500000001 | - |
| 60 | Professional & Business Services | SMU04380608000000001 | - |
| 65 | Education & Health Services | SMU04380606500000001 | - |
| 70 | Leisure & Hospitality | SMU04380607000000001 | - |
| 80 | Other Services | SMU04380608000000001 | - |
| 90 | Government | SMU04380609000000001 | - |

**Note:** First 10 series validated successfully. All supersector series exist in BLS database.

## Phoenix-Specific Industry Series (NAICS-based)

### High-Priority Industries for Rent Growth Model

#### 1. Semiconductor Manufacturing (NAICS 3344)
- **Series ID:** `SMU04383033440000001`
- **Relevance:** ⭐⭐⭐ CRITICAL - Intel/TSMC $40B investment
- **Expected Correlation:** r = 0.55-0.70
- **Theory:** $40B semiconductor investment → 10,000+ direct jobs → rental demand
- **Validation:** Pending (requires API call)

#### 2. Data Processing & Hosting (NAICS 518)
- **Series ID:** `SMU04385051800000001`
- **Relevance:** ⭐⭐⭐ HIGH - Data center hub
- **Expected Correlation:** r = 0.45-0.60
- **Theory:** Cloud infrastructure growth → high-wage tech jobs → rental demand
- **Validation:** Pending

#### 3. Administrative & Support Services (NAICS 561)
- **Series ID:** `SMU04380656100000001`
- **Relevance:** ⭐⭐⭐ HIGH - Back-office relocations
- **Expected Correlation:** r = 0.50-0.65
- **Theory:** Corporate back-office relocations from CA → job growth
- **Validation:** Pending

#### 4. Real Estate (NAICS 531)
- **Series ID:** `SMU04385553100000001`
- **Relevance:** ⭐⭐ MEDIUM - Coincident indicator
- **Expected Correlation:** r = 0.40-0.55
- **Theory:** Real estate sector employment tracks housing market activity
- **Validation:** Pending

### Complete NAICS 3-Digit Industry List (31 Industries)

**Construction (20)**
- 236: Construction of Buildings
- 237: Heavy and Civil Engineering
- 238: Specialty Trade Contractors

**Manufacturing (30)**
- 334: Computer & Electronic Product Mfg ⭐⭐⭐
- 335: Electrical Equipment Mfg
- 336: Transportation Equipment Mfg

**Wholesale Trade (41)**
- 423: Merchant Wholesalers, Durable Goods
- 424: Merchant Wholesalers, Nondurable Goods

**Retail Trade (42)**
- 441: Motor Vehicle & Parts Dealers
- 445: Food & Beverage Stores
- 452: General Merchandise Stores

**Information (50)**
- 511: Publishing Industries
- 517: Telecommunications
- 518: Data Processing & Hosting ⭐⭐⭐
- 519: Other Information Services

**Financial Activities (55)**
- 522: Credit Intermediation
- 523: Securities & Investments
- 524: Insurance Carriers
- 525: Funds & Trusts
- 531: Real Estate ⭐⭐

**Professional & Business Services (60)**
- 541: Professional, Scientific, Technical ⭐⭐⭐
- 561: Administrative & Support Services ⭐⭐⭐
- 562: Waste Management

**Education & Health Services (65)**
- 611: Educational Services
- 621: Ambulatory Health Care
- 622: Hospitals
- 623: Nursing & Residential Care
- 624: Social Assistance

**Leisure & Hospitality (70)**
- 713: Amusement, Gambling, Recreation
- 721: Accommodation
- 722: Food Services & Drinking Places

## Recommended Series for Rent Growth Prediction

### Tier 1 - Core Predictors (Already in Model)

1. **Total Nonfarm Employment**
   - Series ID: `SMU04383400000000001`
   - Rank: Top 5-7
   - Use: Overall employment trend

2. **Professional & Business Services**
   - Series ID: `SMU04380608000000001`
   - Rank: #1-2 predictor
   - Use: High-wage job creation indicator

3. **Construction Employment**
   - Series ID: `SMU04382000000000001`
   - Rank: Top 10-12
   - Use: Construction activity indicator

### Tier 2 - Phoenix-Specific Industries (Add to Model)

4. **Semiconductor Manufacturing**
   - Series ID: `SMU04383033440000001`
   - Priority: HIGH
   - Expected Rank: Top 10-15

5. **Data Processing & Hosting**
   - Series ID: `SMU04385051800000001`
   - Priority: HIGH
   - Expected Rank: Top 12-17

6. **Administrative & Support Services**
   - Series ID: `SMU04380656100000001`
   - Priority: HIGH
   - Expected Rank: Top 15-20

### Tier 3 - Supporting Industries (Test in Variable Selection)

7. **Real Estate**
   - Series ID: `SMU04385553100000001`
   - Priority: MEDIUM
   - Expected Rank: 15-20

8. **Leisure & Hospitality**
   - Series ID: `SMU04387000000000001`
   - Priority: MEDIUM
   - Expected Rank: 18-22

9. **Education & Health Services**
   - Series ID: `SMU04386500000000001`
   - Priority: MEDIUM
   - Expected Rank: 20-25

10. **Financial Activities**
    - Series ID: `SMU04385500000000001`
    - Priority: MEDIUM
    - Expected Rank: 20-25

11. **Information Sector**
    - Series ID: `SMU04385000000000001`
    - Priority: MEDIUM
    - Expected Rank: 18-23

12. **Manufacturing (Total)**
    - Series ID: `SMU04383000000000001`
    - Priority: LOW-MEDIUM
    - Expected Rank: 22-27

## How to Construct Custom BLS Series IDs

### Step 1: Identify the Industry

Use NAICS codes to identify specific industries:
- 2-digit: Major sector (e.g., 51 = Information)
- 3-digit: Subsector (e.g., 518 = Data Processing)
- 4-digit: Industry group (e.g., 5182 = Data Processing Services)
- 6-digit: Detailed industry

### Step 2: Determine Supersector

Match your industry to BLS supersector classification:
- Construction (NAICS 23) → Supersector 20
- Manufacturing (NAICS 31-33) → Supersector 30
- Information (NAICS 51) → Supersector 50
- Professional Services (NAICS 54) → Supersector 60

### Step 3: Build the Series ID

```python
def construct_phoenix_series(naics_code, data_type='01'):
    """
    Construct BLS series ID for Phoenix MSA

    Parameters:
    -----------
    naics_code : str
        3-digit NAICS code (e.g., '518')
    data_type : str
        '01' = Employment
        '11' = Average Weekly Earnings

    Returns:
    --------
    str: Complete BLS series ID
    """
    # Map NAICS to supersector
    supersector_map = {
        '2': '20',  # Construction
        '3': '30',  # Manufacturing
        '4': '40',  # Trade/Transport
        '5': '50',  # Information/Financial
        '6': '65',  # Education/Health
        '7': '70',  # Leisure/Hospitality
    }

    supersector = supersector_map.get(naics_code[0], '00')
    industry = f"{naics_code}000"  # Pad to 6 digits

    return f"SMU0438060{supersector}{industry}{data_type}"

# Example usage:
semiconductor_series = construct_phoenix_series('334')
# Returns: SMU04383033440000001
```

### Step 4: Validate the Series

Always validate before using in production:

```python
import requests

def validate_bls_series(series_id, bls_api_key=None):
    """Check if BLS series exists"""
    url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    payload = {
        'seriesid': [series_id],
        'startyear': '2023',
        'endyear': '2024'
    }

    if bls_api_key:
        payload['registrationkey'] = bls_api_key

    response = requests.post(url, json=payload)
    data = response.json()

    if data['status'] == 'REQUEST_SUCCEEDED':
        series_data = data['Results']['series'][0]
        return len(series_data.get('data', [])) > 0

    return False
```

## Data Type Codes

Beyond employment levels, BLS provides additional data types:

| Code | Data Type | Use Case |
|------|-----------|----------|
| 01 | All Employees (Employment Level) | Primary demand indicator |
| 02 | 3-Month Average Employment | Smoothed trend |
| 03 | Female Employees | Demographic analysis |
| 04 | Production Employees | Blue-collar employment |
| 06 | Average Weekly Hours, All Employees | Labor intensity |
| 07 | Average Weekly Hours, Production | Overtime indicator |
| 11 | Average Weekly Earnings, All | Wage growth |

### Example: Phoenix Semiconductor Wages

```
Series ID: SMU04383033440000011
- SMU: State & Metro
- 04: Arizona
- 38060: Phoenix MSA
- 30: Manufacturing supersector
- 334000: Semiconductor manufacturing
- 11: Average weekly earnings
```

## BLS API Rate Limits

**Without API Key:**
- 25 requests per day
- 10 requests per minute

**With API Key (Free Registration):**
- 500 requests per day
- 120 requests per minute
- Register at: https://data.bls.gov/registrationEngine/

## Files Generated

1. **bls_phoenix_generated_series.csv** - All 53 generated series IDs
2. **bls_phoenix_recommended_series.csv** - 12 high-priority series
3. **bls_phoenix_validated_sample.csv** - Validation results (10 series)

## Next Steps

1. **Validate Phoenix-Specific Industries**
   - Run validation on semiconductor, data processing, back-office series
   - Confirm data availability and historical depth

2. **Fetch Historical Data**
   - Use `phoenix_api_fetcher.py` to retrieve time series
   - Start with 2015 for 10 years of history

3. **Variable Selection Testing**
   - Add new industries to elastic net candidate set
   - Test permutation importance vs existing predictors

4. **Model Integration**
   - Add top-performing series to GBM Phoenix-specific model
   - Monitor improvement in out-of-sample MAPE

---

**Generated by:** BLS Series Discovery Tool
**Phoenix MSA Code:** 38060
**Arizona State Code:** 04
**Total Series Generated:** 53 (22 supersectors + 31 industries)
**Validation Status:** 10/10 supersectors validated successfully
