#!/usr/bin/env python3
"""
Main Analysis Pipeline for Multifamily Rent Growth Analysis
Orchestrates data collection, processing, and analysis
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
from data_acquisition.fred_collector import FREDCollector
from analysis.time_series_analyzer import TimeSeriesAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RentGrowthAnalysisPipeline:
    """
    Main orchestration pipeline for rent growth analysis
    """
    
    def __init__(self, config_path: str = "config/analysis_config.yaml"):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path("data")
        self.output_dir = Path("outputs")
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.fred_collector = FREDCollector()
        self.analyzer = TimeSeriesAnalyzer(target_variable='rent_growth')
        
        # Data containers
        self.raw_data = {}
        self.processed_data = None
        self.analysis_results = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(self.config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'data': {
                    'start_date': '1985-01-01',
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'frequency': 'm',
                    'target_markets': ['phoenix', 'austin', 'dallas', 'denver', 
                                      'salt_lake_city', 'nashville', 'miami']
                },
                'analysis': {
                    'lags': [0, 1, 3, 6, 12],
                    'rolling_windows': [3, 6, 12, 24],
                    'max_granger_lag': 12,
                    'cv_splits': 5,
                    'test_months': 24
                },
                'output': {
                    'save_plots': True,
                    'save_csv': True,
                    'save_json': True
                }
            }
    
    def collect_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data from various sources
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of collected data by category
        """
        logger.info("Starting data collection...")
        
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        frequency = self.config['data']['frequency']
        
        # Collect FRED data
        logger.info("Collecting FRED economic indicators...")
        fred_data = self.fred_collector.get_all_core_series(start_date, end_date)
        
        # Collect additional data sources (placeholder for expansion)
        # This would include:
        # - Real estate specific data (CoStar, RealPage, etc.)
        # - Census demographic data
        # - Local MSA data
        
        # For now, generate synthetic rent growth data for demonstration
        if 'economic' in fred_data:
            dates = fred_data['economic'].index
            np.random.seed(42)
            
            # Generate synthetic rent growth correlated with economic indicators
            base_growth = 3.0  # Base annual rent growth %
            
            # Create synthetic rent growth influenced by economic factors
            fed_funds_impact = -0.3 * fred_data['economic'].get('FEDFUNDS', 0).fillna(0)
            unemployment_impact = -0.5 * fred_data['economic'].get('UNRATE', 0).fillna(0)
            gdp_impact = 0.2 * fred_data['economic'].get('GDPC1', 0).pct_change().fillna(0) * 100
            
            # Combine factors with some noise
            rent_growth = (base_growth + 
                          fed_funds_impact + 
                          unemployment_impact + 
                          gdp_impact +
                          np.random.normal(0, 1, len(dates)))
            
            # Smooth the series
            rent_growth = pd.Series(rent_growth, index=dates).rolling(3, center=True).mean()
            
            fred_data['real_estate'] = pd.DataFrame({
                'rent_growth': rent_growth,
                'vacancy_rate': 5 + np.random.normal(0, 1, len(dates)),
                'new_supply': 2 + np.random.normal(0, 0.5, len(dates))
            }, index=dates)
        
        self.raw_data = fred_data
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for category, df in fred_data.items():
            if not df.empty:
                output_path = self.data_dir / f"raw_{category}_{timestamp}.parquet"
                df.to_parquet(output_path)
                logger.info(f"Saved {category} data to {output_path}")
        
        return fred_data
    
    def process_data(self) -> pd.DataFrame:
        """
        Process and combine data for analysis
        
        Returns:
            Processed DataFrame ready for analysis
        """
        logger.info("Processing data...")
        
        # Combine all data categories
        all_data = []
        
        for category, df in self.raw_data.items():
            if not df.empty:
                # Add category prefix to column names
                df.columns = [f"{category}_{col}" for col in df.columns]
                all_data.append(df)
        
        if all_data:
            # Combine all DataFrames
            combined = pd.concat(all_data, axis=1)
            
            # Handle missing values
            # Forward fill for most economic indicators
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            
            # Remove columns with too many missing values
            threshold = len(combined) * 0.3
            combined = combined.dropna(thresh=threshold, axis=1)
            
            # Ensure we have the target variable
            if 'real_estate_rent_growth' not in combined.columns:
                logger.warning("Target variable 'rent_growth' not found in data")
            
            self.processed_data = combined
            
            # Save processed data
            output_path = self.output_dir / f"processed_data_{datetime.now():%Y%m%d}.parquet"
            combined.to_parquet(output_path)
            logger.info(f"Saved processed data to {output_path}")
            
            return combined
        
        logger.error("No data to process")
        return pd.DataFrame()
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive time series analysis
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running time series analysis...")
        
        if self.processed_data is None or self.processed_data.empty:
            logger.error("No processed data available for analysis")
            return {}
        
        # Identify target column
        target_col = 'real_estate_rent_growth'
        
        if target_col not in self.processed_data.columns:
            # Use first column with 'rent' in name or first column
            rent_cols = [col for col in self.processed_data.columns if 'rent' in col.lower()]
            target_col = rent_cols[0] if rent_cols else self.processed_data.columns[0]
            logger.warning(f"Using {target_col} as target variable")
        
        # Select features for analysis (top correlated variables)
        initial_corr = self.processed_data.corr()[target_col].abs().sort_values(ascending=False)
        top_features = initial_corr.head(15).index.tolist()
        
        analysis_data = self.processed_data[top_features]
        
        # Run complete analysis
        self.analysis_results = self.analyzer.perform_complete_analysis(
            analysis_data, 
            target_col
        )
        
        # Generate summary report
        summary = self._generate_summary_report()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report from analysis results"""
        
        if not self.analysis_results:
            return {}
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': {
                'start': str(self.processed_data.index[0]),
                'end': str(self.processed_data.index[-1]),
                'observations': len(self.processed_data)
            },
            'key_findings': {},
            'model_performance': self.analysis_results.validation_metrics,
            'recommendations': []
        }
        
        # Extract key findings
        
        # 1. Top predictive variables from Granger causality
        if self.analysis_results.causality:
            significant_causes = {
                var: info for var, info in self.analysis_results.causality.items()
                if isinstance(info, dict) and info.get('causes_target', False)
            }
            summary['key_findings']['granger_causes'] = significant_causes
        
        # 2. Cointegration results
        if self.analysis_results.cointegration:
            summary['key_findings']['cointegration'] = {
                'has_long_term_relationship': self.analysis_results.cointegration.get('has_cointegration', False),
                'n_relationships': self.analysis_results.cointegration.get('n_relationships', 0)
            }
        
        # 3. Top features from ML models
        if self.analysis_results.feature_importance:
            sorted_features = sorted(
                self.analysis_results.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            summary['key_findings']['top_predictive_features'] = dict(sorted_features)
        
        # 4. Stationarity results
        stationary_vars = []
        non_stationary_vars = []
        for var, result in self.analysis_results.stationarity.items():
            if result['conclusion'] == 'Stationary':
                stationary_vars.append(var)
            elif result['conclusion'] == 'Non-stationary':
                non_stationary_vars.append(var)
        
        summary['key_findings']['stationarity'] = {
            'stationary_variables': stationary_vars,
            'non_stationary_variables': non_stationary_vars
        }
        
        # Generate recommendations
        recommendations = []
        
        # Model performance recommendation
        if self.analysis_results.validation_metrics:
            best_model = max(
                self.analysis_results.validation_metrics.items(),
                key=lambda x: x[1] if 'r2' in x[0] else -x[1]
            )
            recommendations.append(
                f"Best performing model: {best_model[0]} with score {best_model[1]:.4f}"
            )
        
        # Feature recommendation
        if self.analysis_results.feature_importance:
            top_3_features = list(sorted_features[:3])
            recommendations.append(
                f"Focus on monitoring: {', '.join([f[0] for f in top_3_features])}"
            )
        
        # Cointegration recommendation
        if self.analysis_results.cointegration and self.analysis_results.cointegration.get('has_cointegration'):
            recommendations.append(
                "Consider VECM models for long-term forecasting due to cointegration"
            )
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save analysis results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary as JSON
        if self.config['output'].get('save_json', True):
            json_path = self.output_dir / f"analysis_summary_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved summary to {json_path}")
        
        # Save detailed results
        if self.analysis_results:
            # Save feature importance as CSV
            if self.config['output'].get('save_csv', True) and self.analysis_results.feature_importance:
                importance_df = pd.DataFrame.from_dict(
                    self.analysis_results.feature_importance,
                    orient='index',
                    columns=['importance']
                ).sort_values('importance', ascending=False)
                
                csv_path = self.output_dir / f"feature_importance_{timestamp}.csv"
                importance_df.to_csv(csv_path)
                logger.info(f"Saved feature importance to {csv_path}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline
        
        Returns:
            Summary of analysis results
        """
        logger.info("="*60)
        logger.info("Starting Multifamily Rent Growth Analysis Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Collect data
            logger.info("\nStep 1: Data Collection")
            self.collect_data()
            
            # Step 2: Process data
            logger.info("\nStep 2: Data Processing")
            self.process_data()
            
            # Step 3: Run analysis
            logger.info("\nStep 3: Time Series Analysis")
            summary = self.run_analysis()
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("ANALYSIS COMPLETE")
            logger.info("="*60)
            
            if summary:
                logger.info(f"\nData Period: {summary['data_period']['start']} to {summary['data_period']['end']}")
                logger.info(f"Observations: {summary['data_period']['observations']}")
                
                if 'model_performance' in summary:
                    logger.info("\nModel Performance:")
                    for metric, value in summary['model_performance'].items():
                        logger.info(f"  {metric}: {value:.4f}")
                
                if 'recommendations' in summary:
                    logger.info("\nKey Recommendations:")
                    for rec in summary['recommendations']:
                        logger.info(f"  • {rec}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Multifamily Rent Growth Analysis')
    parser.add_argument('--config', type=str, default='config/analysis_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = RentGrowthAnalysisPipeline(config_path=args.config)
    
    try:
        results = pipeline.run_complete_pipeline()
        
        # Save final results
        with open('outputs/latest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\nAnalysis complete! Results saved to outputs/")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()