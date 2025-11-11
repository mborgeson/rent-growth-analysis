#!/usr/bin/env python3
"""
Data Quality Fix Script
- Removes duplicate 2025-12-31 entry
- Adds 'period' column (e.g., "2010Q1")
- Adds 'market_name' column (all rows = "Phoenix")
"""

import pandas as pd
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent
data_file = project_root / "data" / "processed" / "phoenix_modeling_dataset.csv"
backup_file = project_root / "data" / "processed" / "phoenix_modeling_dataset_backup.csv"

print("=" * 80)
print("DATA QUALITY FIX SCRIPT")
print("=" * 80)

# Backup original file
print(f"\n1. Creating backup: {backup_file.name}")
import shutil
shutil.copy(data_file, backup_file)
print("   ✅ Backup created")

# Load data with date parsing
print(f"\n2. Loading data from: {data_file.name}")
df = pd.read_csv(data_file, parse_dates=['date'])
print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Check for duplicates
print("\n3. Checking for duplicate dates...")
duplicate_dates = df[df.duplicated(subset=['date'], keep=False)]['date'].unique()
if len(duplicate_dates) > 0:
    print(f"   ⚠️  Found {len(duplicate_dates)} duplicate dates:")
    for date in duplicate_dates:
        count = len(df[df['date'] == date])
        print(f"      - {date}: {count} occurrences")
        rows = df[df['date'] == date]
        for idx, row in rows.iterrows():
            print(f"        Row {idx}: rent_growth_yoy={row['rent_growth_yoy']}, "
                  f"asking_rent={row['asking_rent']}, "
                  f"units_under_construction={row['units_under_construction']}")
else:
    print("   ✅ No duplicate dates found")

# Remove duplicates - keep first occurrence
print("\n4. Removing duplicate dates (keeping first occurrence)...")
original_count = len(df)
df = df.drop_duplicates(subset=['date'], keep='first')
removed_count = original_count - len(df)
print(f"   ✅ Removed {removed_count} duplicate row(s)")
print(f"   ✅ New row count: {len(df)}")

# Add 'period' column (e.g., "2010Q1", "2010Q2", etc.)
print("\n5. Adding 'period' column...")
df['period'] = df['date'].dt.to_period('Q').astype(str)
print(f"   ✅ Period column added (sample: {df['period'].iloc[0]})")

# Add 'market_name' column (all rows = "Phoenix")
print("\n6. Adding 'market_name' column...")
df['market_name'] = 'Phoenix'
print("   ✅ Market name column added (all rows = 'Phoenix')")

# Reorder columns to put new columns after 'date'
print("\n7. Reordering columns...")
cols = df.columns.tolist()
# Move 'period' and 'market_name' to positions 1 and 2 (after 'date')
cols.remove('period')
cols.remove('market_name')
cols.insert(1, 'period')
cols.insert(2, 'market_name')
df = df[cols]
print("   ✅ Columns reordered (period and market_name after date)")

# Save cleaned data
print(f"\n8. Saving cleaned data to: {data_file.name}")
df.to_csv(data_file, index=False)
print(f"   ✅ Data saved: {len(df)} rows, {len(df.columns)} columns")

# Summary
print("\n" + "=" * 80)
print("DATA QUALITY FIX COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   - Original rows: {original_count}")
print(f"   - Duplicates removed: {removed_count}")
print(f"   - Final rows: {len(df)}")
print(f"   - New columns added: period, market_name")
print(f"   - Backup saved: {backup_file}")
print(f"\n✅ Data quality issues fixed!")
