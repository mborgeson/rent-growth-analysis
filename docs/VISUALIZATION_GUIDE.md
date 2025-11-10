# Enhanced Visualization Guide

## Overview

The enhanced visualization module provides advanced chart types and interactive reports for Phoenix Rent Growth forecasts, including:

- **Confidence Interval Charts** - Show forecast uncertainty with multiple confidence bands
- **Scenario Comparison Charts** - Compare best/worst/base case scenarios
- **Component Analysis Charts** - Visualize model component contributions
- **Interactive HTML Dashboards** - Multi-panel interactive reports (requires plotly)

## Quick Start

### 1. Test Configuration

```bash
python3 scripts/visualizations.py --test
```

This displays:
- Configuration file status
- Color scheme settings
- Chart settings
- Plotly availability

### 2. Generate Demo Charts

```bash
python3 scripts/visualizations.py --demo
```

This creates sample visualizations in `outputs/visualizations/` to verify the module works.

## Configuration

### Setup

1. Copy the example configuration:
```bash
cp config/visualization_config.example.json config/visualization_config.json
```

2. Edit `config/visualization_config.json` to customize:
   - Color scheme (hex color codes)
   - Chart dimensions and DPI
   - Confidence interval levels
   - Interactive chart settings

### Configuration Options

**Color Scheme**:
```json
{
  "color_scheme": {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#06A77D",
    "warning": "#F18F01",
    "danger": "#C73E1D",
    "info": "#6C757D"
  }
}
```

**Chart Settings**:
```json
{
  "chart_settings": {
    "figsize": [14, 10],
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid",
    "font_size": 10
  }
}
```

Available styles:
- `seaborn-v0_8-darkgrid` (default)
- `seaborn-v0_8-whitegrid`
- `ggplot`
- `bmh`
- `fivethirtyeight`

**Confidence Intervals**:
```json
{
  "confidence_intervals": {
    "enabled": true,
    "levels": [50, 80, 95],
    "alpha": [0.4, 0.3, 0.2]
  }
}
```

## Usage in Automated Forecast Script

The enhanced visualizations are automatically generated when running the monthly forecast update:

```bash
python3 scripts/automated_monthly_forecast_update.py
```

### Output Files

When enhanced visualizations are enabled, the script generates:

**Basic (always created)**:
- `phoenix_forecast_update_YYYYMMDD.png` - 4-panel summary chart

**Enhanced (if module available)**:
- `report_YYYYMMDD_HHMMSS/confidence_intervals.png` - Forecast with uncertainty bands
- `report_YYYYMMDD_HHMMSS/scenario_comparison.png` - Best/worst/base case comparison
- `report_YYYYMMDD_HHMMSS/component_analysis.png` - Model component breakdown
- `report_YYYYMMDD_HHMMSS/interactive_dashboard.html` - Interactive HTML (if plotly installed)

## Programmatic Usage

### Basic Example

```python
from visualizations import ForecastVisualizer
import pandas as pd

# Initialize visualizer
viz = ForecastVisualizer()

# Prepare data
historical_data = pd.DataFrame({
    'Date': [...],
    'Rent_Growth': [...]
})

forecast_data = pd.DataFrame({
    'Date': [...],
    'Forecast': [...]
})

# Generate confidence interval chart
confidence_intervals = {
    95: (lower_95, upper_95),
    80: (lower_80, upper_80),
    50: (lower_50, upper_50)
}

viz.create_confidence_interval_chart(
    historical_data,
    forecast_data,
    confidence_intervals
)
```

### Scenario Comparison

```python
scenarios = {
    'Best Case': pd.DataFrame({
        'Date': [...],
        'Forecast': [...]
    }),
    'Base Case': pd.DataFrame({
        'Date': [...],
        'Forecast': [...]
    }),
    'Worst Case': pd.DataFrame({
        'Date': [...],
        'Forecast': [...]
    })
}

viz.create_scenario_comparison(scenarios)
```

### Comprehensive Report

```python
# Generate all visualizations at once
generated_files = viz.generate_comprehensive_report(
    historical_data,
    forecast_data,
    components={'LightGBM': [...], 'SARIMA': [...]},
    confidence_intervals=confidence_intervals,
    scenarios=scenarios,
    metrics={'RMSE': 2.5}
)

# Returns dict mapping chart types to file paths
print(generated_files['confidence_intervals'])
print(generated_files['scenario_comparison'])
```

## Interactive Dashboards (Optional)

For interactive HTML dashboards with hover tooltips and zoom, install plotly:

```bash
pip install plotly
```

Once installed, the module will automatically generate interactive dashboards:

```python
viz.create_interactive_dashboard(
    historical_data,
    forecast_data,
    components={'LightGBM': [...], 'SARIMA': [...]},
    metrics={'RMSE': 2.5}
)
```

## Chart Types

### 1. Confidence Interval Chart

Shows forecast uncertainty with multiple confidence bands (50%, 80%, 95%).

**Features**:
- Historical data overlay
- Multiple confidence levels
- Color-coded bands
- Automatic legend

**Use case**: Communicate forecast uncertainty to stakeholders

### 2. Scenario Comparison Chart

Compare multiple forecast scenarios (best/worst/base case).

**Features**:
- Multiple scenario lines
- Color-coded by scenario type
- Best case in green, worst case in red
- Baseline reference at 0%

**Use case**: Scenario planning and risk assessment

### 3. Component Analysis Chart

Stacked area chart showing how each model component contributes to the final forecast.

**Features**:
- Model component breakdown
- Contribution over time
- Color-coded components
- Cumulative view

**Use case**: Model explainability and debugging

### 4. Interactive Dashboard (Plotly)

Multi-panel interactive HTML report with:
- Forecast with history
- Model components bar chart
- Performance metrics gauge
- Forecast distribution histogram

**Features**:
- Hover tooltips
- Zoom and pan
- Export to PNG
- Responsive design

**Use case**: Executive presentations and exploration

## Customization

### Color Schemes

Modify colors in `visualization_config.json`:

**Corporate Branding**:
```json
{
  "color_scheme": {
    "primary": "#003D5B",
    "secondary": "#00798C",
    "success": "#30C5B0",
    "warning": "#FFB81C",
    "danger": "#D62828",
    "info": "#6C757D"
  }
}
```

**High Contrast**:
```json
{
  "color_scheme": {
    "primary": "#000000",
    "secondary": "#0066CC",
    "success": "#00AA00",
    "warning": "#FF8800",
    "danger": "#CC0000",
    "info": "#666666"
  }
}
```

### Chart Dimensions

Adjust size and resolution:

```json
{
  "chart_settings": {
    "figsize": [16, 12],  // Larger charts
    "dpi": 150,           // Higher resolution
    "style": "ggplot",
    "font_size": 12
  }
}
```

### Confidence Levels

Customize confidence interval levels:

```json
{
  "confidence_intervals": {
    "enabled": true,
    "levels": [50, 75, 90, 99],  // Custom levels
    "alpha": [0.5, 0.4, 0.3, 0.2]  // Transparency per level
  }
}
```

## Troubleshooting

### Issue: "plotly not available"

**Solution**: Install plotly for interactive charts
```bash
pip install plotly
```

Interactive dashboards are optional - basic charts still work without plotly.

### Issue: Charts not generated

**Check**:
1. `outputs/visualizations/` directory exists and is writable
2. matplotlib is installed: `pip install matplotlib seaborn`
3. Data has required columns ('Date', 'Rent_Growth', 'Forecast')

### Issue: Font rendering issues

**Solution**: Set matplotlib backend
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Issue: Colors not applied

**Check**:
1. Configuration file exists: `config/visualization_config.json`
2. Color codes are valid hex format: `#RRGGBB`
3. Configuration file is valid JSON (use `python3 -m json.tool config/visualization_config.json`)

## Advanced Usage

### Custom Color Palettes

```python
viz = ForecastVisualizer()

# Override colors programmatically
viz.colors = {
    'primary': '#FF5733',
    'secondary': '#C70039',
    'success': '#28B463',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#5DADE2'
}

viz.create_confidence_interval_chart(...)
```

### Batch Generation

```python
# Generate multiple reports with different configurations
for scenario_name, scenario_data in scenarios.items():
    output_dir = f"outputs/visualizations/{scenario_name}"
    viz.generate_comprehensive_report(
        historical_data,
        scenario_data,
        output_dir=Path(output_dir)
    )
```

### Custom Chart Styling

```python
import matplotlib.pyplot as plt

# Apply custom style before visualization
plt.style.use('seaborn-v0_8-poster')  # Large fonts for presentations
plt.rcParams['figure.dpi'] = 200       # High resolution

viz = ForecastVisualizer()
viz.create_confidence_interval_chart(...)
```

## Integration with Forecast Pipeline

The enhanced visualizations integrate seamlessly with the automated forecast update:

1. **Automatic Generation**: When `automated_monthly_forecast_update.py` runs, enhanced visualizations are automatically created (if module available)

2. **Graceful Degradation**: If visualization module fails, basic 4-panel chart still created

3. **Email Alerts**: Enhanced visualizations can be attached to email alerts (configure in `email_config.json`)

4. **Report Archiving**: All visualizations saved to timestamped report directories for historical tracking

## Best Practices

1. **Use confidence intervals** for stakeholder communication to convey uncertainty
2. **Generate scenario comparisons** for planning meetings and risk assessment
3. **Review component analysis** when debugging model performance issues
4. **Create interactive dashboards** for executive presentations (install plotly)
5. **Customize colors** to match organizational branding
6. **Archive reports** for historical tracking and comparison

## File Locations

- **Module**: `scripts/visualizations.py`
- **Configuration**: `config/visualization_config.json` (copy from `.example.json`)
- **Output**: `outputs/visualizations/`
- **Documentation**: `docs/VISUALIZATION_GUIDE.md` (this file)

## Support

For issues or questions:
1. Check configuration: `python3 scripts/visualizations.py --test`
2. Generate demo: `python3 scripts/visualizations.py --demo`
3. Review logs in automated forecast output
4. Verify matplotlib and seaborn are installed
