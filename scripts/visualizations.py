#!/usr/bin/env python3
"""
Enhanced Visualization Module for Phoenix Rent Growth Forecasting

Purpose: Provide advanced visualization capabilities including confidence intervals,
         scenario comparisons, heat maps, and interactive HTML reports

Usage:
    from visualizations import ForecastVisualizer

    viz = ForecastVisualizer()
    viz.create_confidence_interval_chart(data, predictions)
    viz.create_scenario_comparison(scenarios)
    viz.generate_interactive_report(results)

Author: Generated for automated forecast system
Version: 1.0
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'config'
OUTPUTS_DIR = BASE_DIR / 'outputs'
VISUALIZATIONS_DIR = OUTPUTS_DIR / 'visualizations'

# Ensure visualizations directory exists
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / 'visualization_config.json'
EXAMPLE_CONFIG_FILE = CONFIG_DIR / 'visualization_config.example.json'

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_VIZ_CONFIG = {
    'color_scheme': {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#06A77D',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'info': '#6C757D'
    },
    'chart_settings': {
        'figsize': (14, 10),
        'dpi': 100,
        'style': 'seaborn-v0_8-darkgrid',
        'font_size': 10
    },
    'confidence_intervals': {
        'enabled': True,
        'levels': [50, 80, 95],
        'alpha': [0.4, 0.3, 0.2]
    },
    'interactive_charts': {
        'enabled': True,
        'template': 'plotly_white',
        'export_html': True,
        'export_png': False
    }
}

# ============================================================================
# FORECAST VISUALIZER CLASS
# ============================================================================

class ForecastVisualizer:
    """
    Enhanced visualization capabilities for forecast analysis
    """

    def __init__(self, config_file=None):
        """
        Initialize visualizer with configuration

        Args:
            config_file: Path to visualization configuration JSON
        """
        self.config_file = config_file or CONFIG_FILE
        self.config = self._load_config()
        self.colors = self.config.get('color_scheme', DEFAULT_VIZ_CONFIG['color_scheme'])

        # Set matplotlib style
        chart_settings = self.config.get('chart_settings', DEFAULT_VIZ_CONFIG['chart_settings'])
        plt.style.use(chart_settings.get('style', 'seaborn-v0_8-darkgrid'))
        plt.rcParams['figure.figsize'] = chart_settings.get('figsize', (14, 10))
        plt.rcParams['figure.dpi'] = chart_settings.get('dpi', 100)
        plt.rcParams['font.size'] = chart_settings.get('font_size', 10)

    def _load_config(self) -> Dict[str, Any]:
        """Load visualization configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Error loading visualization config: {e}")

        return DEFAULT_VIZ_CONFIG.copy()

    # ========================================================================
    # CONFIDENCE INTERVAL CHARTS
    # ========================================================================

    def create_confidence_interval_chart(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        confidence_intervals: Dict[int, Tuple[np.ndarray, np.ndarray]],
        output_file: Optional[Path] = None,
        title: str = "Rent Growth Forecast with Confidence Intervals"
    ) -> Path:
        """
        Create forecast chart with confidence interval bands

        Args:
            historical_data: Historical data with 'Date' and 'Rent_Growth' columns
            forecast_data: Forecast data with 'Date' and 'Forecast' columns
            confidence_intervals: Dict mapping confidence levels to (lower, upper) bounds
            output_file: Output file path (auto-generated if None)
            title: Chart title

        Returns:
            Path to saved chart
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = VISUALIZATIONS_DIR / f'confidence_intervals_{timestamp}.png'

        fig, ax = plt.subplots(figsize=self.config['chart_settings']['figsize'])

        # Plot historical data
        ax.plot(historical_data['Date'], historical_data['Rent_Growth'],
               label='Historical', color=self.colors['primary'], linewidth=2)

        # Plot forecast
        ax.plot(forecast_data['Date'], forecast_data['Forecast'],
               label='Forecast', color=self.colors['secondary'],
               linewidth=2, linestyle='--')

        # Plot confidence intervals (from widest to narrowest)
        ci_config = self.config.get('confidence_intervals', DEFAULT_VIZ_CONFIG['confidence_intervals'])
        levels = sorted(confidence_intervals.keys(), reverse=True)
        alphas = ci_config.get('alpha', [0.2, 0.3, 0.4])

        for i, level in enumerate(levels):
            lower, upper = confidence_intervals[level]
            alpha = alphas[i] if i < len(alphas) else 0.2

            ax.fill_between(
                forecast_data['Date'],
                lower,
                upper,
                alpha=alpha,
                color=self.colors['info'],
                label=f'{level}% Confidence'
            )

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rent Growth (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    # ========================================================================
    # SCENARIO COMPARISON CHARTS
    # ========================================================================

    def create_scenario_comparison(
        self,
        scenarios: Dict[str, pd.DataFrame],
        output_file: Optional[Path] = None,
        title: str = "Forecast Scenarios Comparison"
    ) -> Path:
        """
        Create comparison chart for multiple forecast scenarios

        Args:
            scenarios: Dict mapping scenario names to DataFrames with 'Date' and 'Forecast'
            output_file: Output file path (auto-generated if None)
            title: Chart title

        Returns:
            Path to saved chart
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = VISUALIZATIONS_DIR / f'scenario_comparison_{timestamp}.png'

        fig, ax = plt.subplots(figsize=self.config['chart_settings']['figsize'])

        # Define colors for scenarios
        scenario_colors = {
            'Best Case': self.colors['success'],
            'Base Case': self.colors['primary'],
            'Worst Case': self.colors['danger'],
            'Conservative': self.colors['warning'],
            'Optimistic': self.colors['info']
        }

        # Plot each scenario
        for scenario_name, data in scenarios.items():
            color = scenario_colors.get(scenario_name, self.colors['secondary'])
            linestyle = '--' if 'Case' in scenario_name else '-'

            ax.plot(data['Date'], data['Forecast'],
                   label=scenario_name, color=color,
                   linewidth=2, linestyle=linestyle)

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rent Growth (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    # ========================================================================
    # COMPONENT CONTRIBUTION ANALYSIS
    # ========================================================================

    def create_component_analysis(
        self,
        components: Dict[str, np.ndarray],
        dates: pd.DatetimeIndex,
        output_file: Optional[Path] = None,
        title: str = "Model Component Contributions"
    ) -> Path:
        """
        Create stacked area chart showing component contributions

        Args:
            components: Dict mapping component names to arrays of values
            dates: Date index for x-axis
            output_file: Output file path (auto-generated if None)
            title: Chart title

        Returns:
            Path to saved chart
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = VISUALIZATIONS_DIR / f'component_analysis_{timestamp}.png'

        fig, ax = plt.subplots(figsize=self.config['chart_settings']['figsize'])

        # Prepare data for stacked area chart
        component_names = list(components.keys())
        component_values = np.array([components[name] for name in component_names])

        # Create stacked area chart
        ax.stackplot(dates, component_values,
                    labels=component_names,
                    alpha=0.7,
                    colors=[self.colors[c] for c in ['primary', 'secondary', 'success', 'warning', 'info']])

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Contribution to Forecast', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    # ========================================================================
    # HEAT MAP VISUALIZATIONS
    # ========================================================================

    def create_forecast_heatmap(
        self,
        forecast_history: pd.DataFrame,
        output_file: Optional[Path] = None,
        title: str = "Forecast Evolution Heat Map"
    ) -> Path:
        """
        Create heat map showing how forecasts have evolved over time

        Args:
            forecast_history: DataFrame with forecast dates as columns, target periods as rows
            output_file: Output file path (auto-generated if None)
            title: Chart title

        Returns:
            Path to saved chart
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = VISUALIZATIONS_DIR / f'forecast_heatmap_{timestamp}.png'

        fig, ax = plt.subplots(figsize=self.config['chart_settings']['figsize'])

        # Create heat map
        sns.heatmap(forecast_history,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': 'Rent Growth (%)'},
                   ax=ax)

        # Formatting
        ax.set_xlabel('Forecast Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Period', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    # ========================================================================
    # INTERACTIVE HTML REPORTS (PLOTLY)
    # ========================================================================

    def create_interactive_dashboard(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        components: Optional[Dict[str, np.ndarray]] = None,
        metrics: Optional[Dict[str, float]] = None,
        output_file: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Create interactive HTML dashboard with multiple charts

        Args:
            historical_data: Historical data
            forecast_data: Forecast data
            components: Model components (optional)
            metrics: Performance metrics (optional)
            output_file: Output file path (auto-generated if None)

        Returns:
            Path to saved HTML file, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive dashboard creation.")
            return None

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = VISUALIZATIONS_DIR / f'interactive_dashboard_{timestamp}.html'

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Forecast with History', 'Model Components',
                          'Performance Metrics', 'Forecast Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'indicator'}, {'type': 'histogram'}]]
        )

        # 1. Forecast with history
        fig.add_trace(
            go.Scatter(x=historical_data['Date'], y=historical_data['Rent_Growth'],
                      name='Historical', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=forecast_data['Date'], y=forecast_data['Forecast'],
                      name='Forecast', line=dict(color=self.colors['secondary'], dash='dash')),
            row=1, col=1
        )

        # 2. Model components (if available)
        if components:
            component_names = list(components.keys())
            component_means = [np.mean(components[name]) for name in component_names]

            fig.add_trace(
                go.Bar(x=component_names, y=component_means,
                      marker_color=self.colors['success']),
                row=1, col=2
            )

        # 3. Performance metrics (if available)
        if metrics:
            metric_value = metrics.get('RMSE', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=metric_value,
                    title={'text': "Test RMSE"},
                    delta={'reference': 5.0},
                    gauge={'axis': {'range': [None, 10]},
                          'bar': {'color': self.colors['primary']},
                          'steps': [
                              {'range': [0, 3], 'color': self.colors['success']},
                              {'range': [3, 7], 'color': self.colors['warning']},
                              {'range': [7, 10], 'color': self.colors['danger']}
                          ]}
                ),
                row=2, col=1
            )

        # 4. Forecast distribution
        fig.add_trace(
            go.Histogram(x=forecast_data['Forecast'],
                        marker_color=self.colors['info'],
                        nbinsx=20),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Phoenix Rent Growth Forecast Dashboard",
            showlegend=True,
            height=800,
            template=self.config.get('interactive_charts', {}).get('template', 'plotly_white')
        )

        # Save HTML
        fig.write_html(str(output_file))
        print(f"✅ Interactive dashboard saved: {output_file}")

        return output_file

    # ========================================================================
    # MULTI-PANEL REPORT
    # ========================================================================

    def generate_comprehensive_report(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        components: Optional[Dict[str, np.ndarray]] = None,
        confidence_intervals: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        scenarios: Optional[Dict[str, pd.DataFrame]] = None,
        metrics: Optional[Dict[str, float]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate comprehensive visualization report with all chart types

        Args:
            historical_data: Historical data
            forecast_data: Forecast data
            components: Model components (optional)
            confidence_intervals: Confidence intervals (optional)
            scenarios: Forecast scenarios (optional)
            metrics: Performance metrics (optional)
            output_dir: Output directory (auto-generated if None)

        Returns:
            Dict mapping chart types to file paths
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = VISUALIZATIONS_DIR / f'report_{timestamp}'

        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # 1. Confidence interval chart (if available)
        if confidence_intervals:
            ci_file = output_dir / 'confidence_intervals.png'
            self.create_confidence_interval_chart(
                historical_data, forecast_data, confidence_intervals, ci_file
            )
            generated_files['confidence_intervals'] = ci_file

        # 2. Scenario comparison (if available)
        if scenarios:
            scenario_file = output_dir / 'scenario_comparison.png'
            self.create_scenario_comparison(scenarios, scenario_file)
            generated_files['scenario_comparison'] = scenario_file

        # 3. Component analysis (if available)
        if components:
            component_file = output_dir / 'component_analysis.png'
            self.create_component_analysis(
                components, forecast_data['Date'], component_file
            )
            generated_files['component_analysis'] = component_file

        # 4. Interactive dashboard (if plotly available)
        if PLOTLY_AVAILABLE:
            dashboard_file = output_dir / 'interactive_dashboard.html'
            dashboard_path = self.create_interactive_dashboard(
                historical_data, forecast_data, components, metrics, dashboard_file
            )
            if dashboard_path:
                generated_files['interactive_dashboard'] = dashboard_path

        print(f"\n📊 Comprehensive Report Generated:")
        for chart_type, file_path in generated_files.items():
            print(f"   ✅ {chart_type}: {file_path.name}")

        return generated_files

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced Visualization Module for Phoenix Rent Growth Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test visualization configuration
  python3 visualizations.py --test

  # Generate sample charts
  python3 visualizations.py --demo
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test visualization configuration')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo charts with sample data')

    args = parser.parse_args()

    viz = ForecastVisualizer()

    if args.test:
        print("=" * 80)
        print("VISUALIZATION CONFIGURATION TEST")
        print("=" * 80)
        print(f"\n📁 Config File: {viz.config_file}")
        print(f"   Exists: {viz.config_file.exists()}")
        print(f"\n🎨 Color Scheme: {viz.colors}")
        print(f"\n📊 Chart Settings: {viz.config.get('chart_settings')}")
        print(f"\n📈 Plotly Available: {PLOTLY_AVAILABLE}")
        print("=" * 80)

    elif args.demo:
        print("Generating demo charts...")

        # Create sample data
        dates = pd.date_range('2020-01-01', periods=50, freq='Q')
        historical_data = pd.DataFrame({
            'Date': dates[:40],
            'Rent_Growth': np.random.randn(40).cumsum() + 3
        })
        forecast_data = pd.DataFrame({
            'Date': dates[40:],
            'Forecast': np.random.randn(10).cumsum() + 3 + historical_data['Rent_Growth'].iloc[-1]
        })

        # Generate confidence intervals
        confidence_intervals = {
            95: (forecast_data['Forecast'] - 2, forecast_data['Forecast'] + 2),
            80: (forecast_data['Forecast'] - 1.5, forecast_data['Forecast'] + 1.5),
            50: (forecast_data['Forecast'] - 1, forecast_data['Forecast'] + 1)
        }

        # Generate scenarios
        scenarios = {
            'Best Case': pd.DataFrame({
                'Date': dates[40:],
                'Forecast': forecast_data['Forecast'] + 1
            }),
            'Base Case': forecast_data.copy(),
            'Worst Case': pd.DataFrame({
                'Date': dates[40:],
                'Forecast': forecast_data['Forecast'] - 1
            })
        }

        # Generate comprehensive report
        viz.generate_comprehensive_report(
            historical_data,
            forecast_data,
            confidence_intervals=confidence_intervals,
            scenarios=scenarios
        )

        print("\n✅ Demo charts generated in outputs/visualizations/")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
