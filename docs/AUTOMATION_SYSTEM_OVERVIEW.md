# Automation System Overview

**Created**: November 10, 2025
**Status**: ✅ Operational
**Purpose**: Automated monthly Phoenix rent growth forecast updates

---

## System Architecture

### Components

1. **Automated Monthly Update Script** (`scripts/automated_monthly_forecast_update.py`)
   - 1,158 lines of code
   - 11-stage workflow execution
   - Automatic data detection and model retraining

2. **Email Alert System** (`scripts/email_alerts.py`)
   - 441 lines of code
   - HTML-formatted professional reports
   - SMTP integration with TLS/SSL support

3. **Enhanced Visualization Module** (`scripts/visualizations.py`)
   - 625 lines of code
   - 4 chart types (confidence intervals, scenarios, components, interactive)
   - Matplotlib and optional Plotly support

4. **Data Pipeline** (`scripts/data_pipeline.py`)
   - 521 lines of code
   - 6 comprehensive quality checks
   - Automatic data validation

5. **Configuration Loader** (`scripts/config_loader.py`)
   - 298 lines of code
   - Centralized configuration management
   - Graceful fallback to defaults

### Total System Size
- **3,043 lines of Python code** across 5 modules
- **4 configuration files** (validation, email, visualization, data pipeline)
- **2 comprehensive guides** (monthly update guide, visualization guide)

---

## Configuration Files

All configurations located in `config/` directory:

1. **validation_thresholds.json** ✅ Active
   - SARIMA stability checks
   - Component correlation thresholds
   - Ridge regularization validation
   - Data quality thresholds

2. **email_config.json** ✅ Active (Disabled by default)
   - SMTP server configuration
   - Alert recipients
   - Email formatting preferences
   - **Status**: Currently disabled for testing

3. **visualization_config.json** ✅ Active
   - Color schemes
   - Chart dimensions and DPI
   - Confidence interval levels
   - Interactive dashboard settings

4. **data_pipeline_config.json** ✅ Active
   - Data source configuration
   - Quality check thresholds
   - Update detection settings
   - Data transformation rules

---

## Workflow

### Monthly Automation Process

```
1. Data Loading
   ↓ Load phoenix_modeling_dataset.csv
   ↓ Parse dates and validate structure
   ↓
2. Quality Checks
   ↓ Missing value detection
   ↓ Duplicate date detection
   ↓ Extreme value detection
   ↓
3. Update Detection
   ↓ Check for new quarterly data
   ↓ Compare to last successful run
   ↓
4. Model Decision
   ↓ Retrain if new data OR force flag
   ↓ Otherwise use existing models
   ↓
5. Forecast Generation
   ↓ LightGBM predictions
   ↓ SARIMA predictions
   ↓ Ridge ensemble combination
   ↓
6. Validation
   ↓ SARIMA stability check
   ↓ Component correlation check
   ↓ Ridge alpha validation
   ↓
7. Output Generation
   ↓ Forecast CSV with datestamp
   ↓ Metadata JSON
   ↓ Visualizations (basic + enhanced)
   ↓ Executive summary update
   ↓
8. Alert Distribution (Optional)
   ↓ Compile alerts
   ↓ Format HTML email
   ↓ Send via SMTP
```

---

## Test Results (2025-11-10)

### ✅ Initial Test: SUCCESSFUL

**Execution Time**: 4 seconds
**Status**: Completed successfully

**Outputs Created**:
- ✅ `phoenix_forecast_2026_2028_20251110.csv` (1.4 KB)
- ✅ `phoenix_forecast_update_20251110.png` (208 KB)
- ✅ 3 enhanced visualizations (confidence intervals, scenarios, components)
- ✅ Executive summary updated
- ✅ Detailed log file (7.1 KB)

**Forecast Results**:
- Average 2026-2028 rent growth: **2.04%**
- Range: 1.90% to 2.58%
- 20 quarters forecasted (2026-Q1 to 2030-Q4)

**Warnings** (Non-blocking):
- Missing columns 'period', 'market_name' (pipeline config, not used by models)
- Duplicate date 2025-12-31 (data file issue)
- Missing values in 23 features (expected, handled by models)

---

## Usage

### Manual Execution

```bash
# Navigate to project directory
cd "/home/mattb/Rent Growth Analysis"

# Standard monthly update
python3 scripts/automated_monthly_forecast_update.py

# Force retrain models
python3 scripts/automated_monthly_forecast_update.py --force-retrain

# Skip comparison to previous forecasts
python3 scripts/automated_monthly_forecast_update.py --skip-comparison

# With email alerts (requires email config)
python3 scripts/automated_monthly_forecast_update.py --alert-email analyst@company.com
```

### Automated Execution

**Cron Job** (Monthly on 1st at 9am):
```bash
0 9 1 * * cd "/home/mattb/Rent Growth Analysis" && python3 scripts/automated_monthly_forecast_update.py
```

**Setup Script**:
```bash
chmod +x scripts/setup_cron.sh
./scripts/setup_cron.sh
```

---

## Monitoring

### Log Files
- Location: `logs/forecast_update_YYYYMMDD_HHMMSS.log`
- Retention: Manual cleanup recommended quarterly
- Contents: Detailed execution trace with timestamps

### Alerts
Current configuration generates alerts for:
- Data quality issues (missing values, duplicates)
- SARIMA explosive predictions (>10%)
- Poor component correlation (<-0.5)
- Weak Ridge regularization (<1.0)
- Large forecast revisions (>1.0pp)

---

## Next Steps

1. **Enable Email Alerts** (Optional)
   - Update `config/email_config.json` with SMTP credentials
   - Set `enabled: true`
   - Test with: `python3 scripts/email_alerts.py --test`

2. **Set Up Cron Job** (Recommended)
   - Run `scripts/setup_cron.sh`
   - Verify with: `crontab -l`

3. **Enhance Visualizations** (Optional)
   - Install plotly: `pip install plotly`
   - Generates interactive HTML dashboards

4. **Data Quality Improvements** (Recommended)
   - Fix duplicate date 2025-12-31
   - Add missing columns 'period', 'market_name' if needed by pipeline

---

## Maintenance

### Weekly
- Review log files for errors
- Check email alerts (if enabled)
- Validate output files created

### Monthly
- Review executive summary
- Compare forecast revisions
- Update accuracy tracking analysis

### Quarterly
- Review model performance metrics
- Analyze accuracy trends
- Archive old outputs

---

## Documentation

**Local Files**:
- `docs/AUTOMATION_SYSTEM_OVERVIEW.md` (this file)
- `docs/VISUALIZATION_GUIDE.md` (visualization documentation)
- `reports/AUTOMATED_MONTHLY_UPDATE_GUIDE.md` (detailed usage guide)
- `COMPREHENSIVE_SYSTEM_ENHANCEMENTS_ANALYSIS_20251110.txt` (complete analysis)

**Dart Documentation** (To be created):
- Workspace: `Rent Growth Analysis`
- Docs Folder: `Rent Growth Analysis/Docs`
- Files to sync: All local markdown documentation

---

## Support

For issues or questions:
1. Check log files in `logs/`
2. Review validation alerts in output
3. Verify data quality in `phoenix_modeling_dataset.csv`
4. Try `--force-retrain` if models seem corrupted

**Version**: 1.0
**Last Updated**: 2025-11-10
**Status**: Production Ready ✅
