# Automated Monthly Forecast Update Guide

**Script**: `scripts/automated_monthly_forecast_update.py`
**Purpose**: Automatically update Phoenix rent growth forecasts on a monthly basis
**Author**: Generated from Root Cause Analysis (Nov 2025)
**Version**: 1.0

---

## Quick Start

### Basic Usage

```bash
# Navigate to project directory
cd "/home/mattb/Rent Growth Analysis"

# Run monthly update
python3 scripts/automated_monthly_forecast_update.py
```

### Common Options

```bash
# Force retrain models (e.g., after data quality improvements)
python3 scripts/automated_monthly_forecast_update.py --force-retrain

# Skip comparison to previous forecasts
python3 scripts/automated_monthly_forecast_update.py --skip-comparison

# With email alerts (requires email configuration)
python3 scripts/automated_monthly_forecast_update.py --alert-email analyst@company.com

# Custom output directory
python3 scripts/automated_monthly_forecast_update.py --output-dir /custom/path
```

---

## Automated Scheduling

### Cron Setup (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add entry to run on 1st of each month at 8am
0 8 1 * * cd "/home/mattb/Rent Growth Analysis" && python3 scripts/automated_monthly_forecast_update.py

# Add entry to run weekly on Mondays at 9am
0 9 * * 1 cd "/home/mattb/Rent Growth Analysis" && python3 scripts/automated_monthly_forecast_update.py
```

### Windows Task Scheduler

1. Open **Task Scheduler**
2. Create **New Task**
3. Set **Trigger**: Monthly, 1st day, 8:00 AM
4. Set **Action**: Start a program
   - Program: `python3`
   - Arguments: `scripts/automated_monthly_forecast_update.py`
   - Start in: `/home/mattb/Rent Growth Analysis`

---

## How It Works

### 1. Data Loading
- Loads latest `phoenix_modeling_dataset.csv`
- Detects new quarters added since last run
- Validates data quality (missing values, duplicates, extremes)

### 2. Model Decision
**Retrains models if**:
- `--force-retrain` flag used
- New data detected since last run
- Model files don't exist

**Uses existing models if**:
- No new data
- Models exist
- No force retrain flag

### 3. Model Training (if needed)
- **LightGBM**: Early stopping (50 rounds), 31 features
- **SARIMA**: Production config (1,1,2)(0,0,1,4)
- **Ridge Meta-Learner**: Alpha range [0.1, 1.0, 10.0, 100.0, 1000.0]

### 4. Validation
**Checks**:
- SARIMA stability (<10% predictions)
- Component correlation (>-0.5 threshold)
- Ridge alpha (≥1.0 threshold)
- Test/train RMSE ratio (<2.0)

**Alerts generated for**:
- SARIMA explosive predictions
- Poor component correlation
- Weak Ridge regularization

### 5. Forecast Generation
- Forecasts 2026-2030 (20 quarters)
- Last actual data: 2025-12-31
- Generates LightGBM, SARIMA, and ensemble predictions

### 6. Comparison
- Loads previous month's forecast
- Calculates revisions (percentage point changes)
- Alerts on large revisions (>1.0pp)

### 7. Accuracy Tracking
- Tracks forecast vs. actual over time
- Calculates MAE and RMSE for completed forecasts
- Updates rolling accuracy metrics

### 8. Output Generation
**Files Created**:
- `phoenix_forecast_2026_2028_YYYYMMDD.csv` - Forecast data
- `phoenix_forecast_2026_2028_metadata_YYYYMMDD.json` - Model metadata
- `phoenix_forecast_update_YYYYMMDD.png` - 4-panel visualization
- `PHOENIX_FORECAST_EXECUTIVE_SUMMARY_LATEST.md` - Updated summary
- `forecast_comparison_YYYYMMDD.csv` - Comparison to previous (if available)
- `forecast_accuracy_tracking.csv` - Rolling accuracy metrics
- `forecast_update_YYYYMMDD_HHMMSS.log` - Execution log

---

## Output Files Explained

### 1. Forecast CSV
**File**: `phoenix_forecast_2026_2028_YYYYMMDD.csv`

| Column | Description |
|--------|-------------|
| `date` | Quarter end date (YYYY-MM-DD) |
| `lightgbm_prediction` | LightGBM component prediction (%) |
| `sarima_prediction` | SARIMA component prediction (%) |
| `ensemble_prediction` | Final ensemble prediction (%) |

**Example**:
```csv
date,lightgbm_prediction,sarima_prediction,ensemble_prediction
2026-03-31,2.14,0.06,0.52
2026-06-30,2.14,-0.02,0.44
```

### 2. Metadata JSON
**File**: `phoenix_forecast_2026_2028_metadata_YYYYMMDD.json`

**Contains**:
- Forecast date and period count
- Date range (start/end)
- Ensemble prediction statistics (mean, min, max, std)
- Model configuration (SARIMA, LightGBM, Ridge)
- Validation thresholds

### 3. Visualization
**File**: `phoenix_forecast_update_YYYYMMDD.png`

**4-Panel Chart**:
1. **Historical + Forecast**: Actual data + forecast time series
2. **Component Predictions**: LightGBM, SARIMA, and ensemble
3. **Forecast Revisions**: Changes from previous month (if available)
4. **Accuracy Tracking**: Forecast vs. actual scatter plot (if completed forecasts exist)

### 4. Executive Summary
**File**: `PHOENIX_FORECAST_EXECUTIVE_SUMMARY_LATEST.md`

**Sections**:
- 2026-2028 outlook (average, range)
- Annual breakdown by year
- Quarterly forecast detail
- Changes from previous forecast (if available)
- Model configuration reference

### 5. Comparison CSV
**File**: `forecast_comparison_YYYYMMDD.csv` (if previous forecast exists)

| Column | Description |
|--------|-------------|
| `date` | Quarter end date |
| `ensemble_prediction_current` | Current forecast |
| `ensemble_prediction_previous` | Previous forecast |
| `revision` | Change in percentage points |

### 6. Accuracy Tracking CSV
**File**: `forecast_accuracy_tracking.csv`

| Column | Description |
|--------|-------------|
| `forecast_date` | When forecast was made |
| `target_date` | Quarter being forecasted |
| `forecast_value` | Predicted rent growth (%) |
| `actual_value` | Actual rent growth (%) |
| `error` | Forecast error (actual - forecast) |

---

## Workflow Examples

### Example 1: Standard Monthly Update

**Scenario**: First business day of the month

```bash
# Run automated update
python3 scripts/automated_monthly_forecast_update.py
```

**Expected Behavior**:
1. Loads latest data
2. No new data detected → Uses existing models
3. Generates updated forecast
4. Compares to last month
5. Updates accuracy tracking
6. Creates outputs

**Review**:
- Check log file for alerts
- Review `PHOENIX_FORECAST_EXECUTIVE_SUMMARY_LATEST.md`
- Compare revisions in visualization

### Example 2: After New Data Added

**Scenario**: Q4 2025 actual data just added to `phoenix_modeling_dataset.csv`

```bash
# Run automated update (auto-detects new data)
python3 scripts/automated_monthly_forecast_update.py
```

**Expected Behavior**:
1. Detects new quarter of data
2. **Retrains models** with expanded training set
3. Generates updated forecast
4. Updates accuracy tracking (Q4 2025 forecast vs actual)
5. Compares to previous forecast

**Review**:
- Check accuracy tracking for Q4 2025
- Review model metrics (did performance improve?)
- Validate forecast revisions

### Example 3: After Data Quality Fixes

**Scenario**: Fixed missing values or data errors in `phoenix_modeling_dataset.csv`

```bash
# Force retrain with cleaned data
python3 scripts/automated_monthly_forecast_update.py --force-retrain
```

**Expected Behavior**:
1. Loads cleaned data
2. Data quality validation passes (no alerts)
3. Retrains models with better data
4. Generates improved forecast

**Review**:
- Confirm data quality alerts resolved
- Compare forecast to previous (should be more stable)
- Check component weights (Ridge alpha)

### Example 4: Scheduled Automation

**Scenario**: Cron job runs 1st of month at 8am

```bash
# Crontab entry
0 8 1 * * cd "/home/mattb/Rent Growth Analysis" && python3 scripts/automated_monthly_forecast_update.py --alert-email analyst@company.com
```

**Expected Behavior**:
1. Runs automatically without intervention
2. Logs to timestamped file
3. Sends email alerts if issues detected
4. Updates all outputs

**Review**:
- Check email for alerts
- Review log file: `logs/forecast_update_YYYYMMDD_HHMMSS.log`
- Validate outputs created successfully

---

## Validation Alerts

### Data Quality Alerts

**Missing Values**:
```
[WARNING] Data quality issue: Missing values in 23 features
```
**Action**: Review data source, impute missing values if possible

**Duplicate Dates**:
```
[WARNING] Data quality issue: Duplicate dates found
```
**Action**: Remove duplicates from dataset

**Extreme Values**:
```
[WARNING] Data quality issue: Extreme rent growth values
```
**Action**: Investigate outliers, validate data accuracy

### Model Validation Alerts

**SARIMA Explosive Predictions**:
```
[CRITICAL] SARIMA predictions explosive! Max: 15.20%
```
**Action**: Review SARIMA configuration, may need order adjustment

**Poor Component Correlation**:
```
[WARNING] Component correlation: -0.8 (below -0.5 threshold)
```
**Action**: Investigate regime shift, models predicting opposite directions

**Weak Ridge Regularization**:
```
[WARNING] Ridge alpha (0.5) below threshold (1.0)
```
**Action**: May indicate overfitting, monitor test performance

### Forecast Alerts

**Large Revisions**:
```
[WARNING] Large forecast revisions detected (5 periods >±1.0pp)
```
**Action**: Review what changed (new data, model retrain, data fixes)

**No Future Periods**:
```
[WARNING] No future periods to forecast
```
**Action**: Expected if dataset doesn't extend beyond actuals

---

## Troubleshooting

### Issue: "Data file not found"
**Cause**: Missing or moved data file
**Fix**:
```bash
# Verify file exists
ls -la data/processed/phoenix_modeling_dataset.csv

# Check path in script (line 116)
# DATA_FILE = DATA_DIR / 'phoenix_modeling_dataset.csv'
```

### Issue: "Failed to load models"
**Cause**: Missing model files or corrupted files
**Fix**:
```bash
# Force retrain to recreate models
python3 scripts/automated_monthly_forecast_update.py --force-retrain
```

### Issue: "SARIMA forecast failed during validation"
**Cause**: SARIMA model issues or data mismatch
**Fix**:
1. Check log file for specific error
2. Verify training period matches expectations (2010-2022)
3. Force retrain if model seems corrupted

### Issue: "Forecast generation failed"
**Cause**: Usually data or model compatibility issues
**Fix**:
1. Check log file for traceback
2. Verify feature list matches dataset columns
3. Force retrain models

### Issue: "No new data detected"
**Cause**: Dataset hasn't been updated since last run
**Fix**:
- This is expected behavior if no new data added
- Script will use existing models and generate updated forecast
- To force retrain anyway, use `--force-retrain`

---

## Integration with Main Forecast Script

### Relationship

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `phoenix_rent_growth_forecast_2026_2028.py` | **One-time forecast generation** | Initial forecast, special scenarios |
| `automated_monthly_forecast_update.py` | **Automated updates** | Regular updates, cron jobs |

### Workflow

```
1. Initial Setup:
   └── Run phoenix_rent_growth_forecast_2026_2028.py
       └── Creates initial forecast
       └── Saves models

2. Ongoing Updates:
   └── Run automated_monthly_forecast_update.py (monthly)
       ├── Loads latest data
       ├── Retrains if new data
       ├── Updates forecast
       ├── Tracks accuracy
       └── Compares to previous
```

### Data Flow

```
phoenix_modeling_dataset.csv
  ↓
[Manual Forecast Script] → Initial forecast + models
  ↓
[Automated Update Script] → Monthly updates
  ├── Load models (or retrain)
  ├── Generate forecast
  ├── Compare to previous
  ├── Track accuracy
  └── Update outputs
```

---

## Best Practices

### 1. Data Management
✅ **DO**:
- Update `phoenix_modeling_dataset.csv` with actual data monthly
- Keep dataset clean (no duplicates, validated values)
- Document data source and update dates

❌ **DON'T**:
- Manually edit forecast columns in dataset
- Mix actual and forecasted data without clear delineation
- Skip data quality validation

### 2. Model Management
✅ **DO**:
- Review model metrics after retraining
- Monitor Ridge alpha selection
- Track component weights over time

❌ **DON'T**:
- Force retrain unnecessarily (wastes resources)
- Ignore validation alerts
- Delete model files without backup

### 3. Forecast Review
✅ **DO**:
- Review executive summary monthly
- Compare to previous forecasts
- Track accuracy metrics
- Document significant changes

❌ **DON'T**:
- Blindly trust automated forecasts
- Ignore large revisions
- Skip accuracy validation

### 4. Automation
✅ **DO**:
- Set up cron job for consistency
- Configure email alerts
- Monitor logs regularly
- Backup outputs periodically

❌ **DON'T**:
- Run multiple instances simultaneously
- Skip log review
- Ignore email alerts
- Delete old forecasts without archiving

---

## Advanced Configuration

### Email Alerts

**Setup SMTP** (example with Gmail):

```python
# Add to script after line 173
import smtplib
from email.mime.text import MIMEText

def send_email_alert(alerts, to_email):
    """Send email alerts via SMTP"""
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    from_email = 'forecast@company.com'
    password = 'your_app_password'

    # Create message
    msg = MIMEText('\n'.join([f"[{a['severity']}] {a['message']}" for a in alerts]))
    msg['Subject'] = f'Phoenix Forecast Alert - {len(alerts)} issues'
    msg['From'] = from_email
    msg['To'] = to_email

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
```

### Custom Validation Thresholds

**Edit lines 48-54**:
```python
'validation_thresholds': {
    'sarima_max_prediction': 10.0,  # Increase if stable >10% growth expected
    'component_correlation_min': -0.5,  # Adjust based on regime shifts
    'test_train_rmse_ratio': 2.0,  # Decrease for stricter overfitting detection
    'ridge_alpha_min': 1.0,  # Increase for stronger regularization
    'forecast_revision_alert': 1.0  # Decrease for more sensitive revision alerts
}
```

### Output Customization

**Change output directory**:
```bash
python3 scripts/automated_monthly_forecast_update.py --output-dir /custom/path
```

**Custom datestamp format** (edit line 583):
```python
datestamp = datetime.now().strftime('%Y%m%d')  # Current: 20251108
datestamp = datetime.now().strftime('%Y-%m-%d')  # Alternative: 2025-11-08
```

---

## Maintenance

### Weekly
- [ ] Check log files for errors
- [ ] Review email alerts (if configured)
- [ ] Validate output files created

### Monthly
- [ ] Review executive summary
- [ ] Compare forecast revisions
- [ ] Update accuracy tracking analysis
- [ ] Archive old outputs

### Quarterly
- [ ] Review model performance metrics
- [ ] Analyze accuracy trends
- [ ] Update validation thresholds if needed
- [ ] Document significant forecast changes

### Annually
- [ ] Comprehensive model review
- [ ] Evaluate against alternative approaches
- [ ] Update documentation
- [ ] Archive full year of outputs

---

## Support and Contact

**For Questions**:
- Review comprehensive analysis: `COMPLETE_ROOT_CAUSE_ANALYSIS.md`
- Review usage guide: `COMPREHENSIVE_ANALYSIS_USAGE_GUIDE.txt`
- Check technical specs: `phoenix_rent_growth_forecast_2026_2028.py`

**For Issues**:
1. Check log files in `logs/forecast_update_*.log`
2. Review validation alerts
3. Verify data quality
4. Try `--force-retrain` if models seem corrupted

---

**Version**: 1.0
**Last Updated**: November 8, 2025
**Script Location**: `scripts/automated_monthly_forecast_update.py`
**Documentation**: This guide + inline code documentation
