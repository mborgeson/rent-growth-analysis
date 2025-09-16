"""
Time Series Analysis Module for Multifamily Rent Growth Analysis
Implements comprehensive statistical and ML-based analysis methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling imports
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for analysis results"""
    correlations: Dict[str, Any]
    causality: Dict[str, Any]
    stationarity: Dict[str, Any]
    cointegration: Optional[Dict[str, Any]]
    forecasts: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    validation_metrics: Dict[str, float]


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for rent growth relationships
    """
    
    def __init__(self, target_variable: str = 'rent_growth'):
        """
        Initialize analyzer
        
        Args:
            target_variable: Name of the target variable to analyze
        """
        self.target_variable = target_variable
        self.results = {}
        
    def test_stationarity(self, series: pd.Series, name: str = "") -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests
        
        Args:
            series: Time series to test
            name: Name of the series
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        # ADF Test
        adf_result = adfuller(series_clean, autolag='AIC')
        
        # KPSS Test
        kpss_result = kpss(series_clean, regression='c', nlags='auto')
        
        results = {
            'series_name': name,
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
        
        # Interpretation
        if results['adf']['is_stationary'] and results['kpss']['is_stationary']:
            results['conclusion'] = 'Stationary'
        elif not results['adf']['is_stationary'] and not results['kpss']['is_stationary']:
            results['conclusion'] = 'Non-stationary'
        else:
            results['conclusion'] = 'Uncertain (conflicting tests)'
        
        logger.info(f"Stationarity test for {name}: {results['conclusion']}")
        return results
    
    def calculate_correlations(self, 
                              data: pd.DataFrame,
                              target_col: str,
                              lags: List[int] = [0, 1, 3, 6, 12]) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlation matrices at various lags
        
        Args:
            data: DataFrame with time series
            target_col: Target column name
            lags: List of lag periods to test
            
        Returns:
            Dictionary of correlation matrices by lag
        """
        correlations = {}
        
        for lag in lags:
            if lag == 0:
                # Contemporaneous correlations
                corr_pearson = data.corr(method='pearson')
                corr_spearman = data.corr(method='spearman')
                corr_kendall = data.corr(method='kendall')
                
                correlations[f'lag_{lag}'] = {
                    'pearson': corr_pearson[target_col].sort_values(ascending=False),
                    'spearman': corr_spearman[target_col].sort_values(ascending=False),
                    'kendall': corr_kendall[target_col].sort_values(ascending=False)
                }
            else:
                # Lagged correlations
                lagged_corr = {}
                for col in data.columns:
                    if col != target_col:
                        # Create lagged series
                        lagged = data[col].shift(lag)
                        valid_idx = ~(lagged.isna() | data[target_col].isna())
                        
                        if valid_idx.sum() > 30:  # Minimum sample size
                            pearson_corr, _ = pearsonr(lagged[valid_idx], data[target_col][valid_idx])
                            spearman_corr, _ = spearmanr(lagged[valid_idx], data[target_col][valid_idx])
                            kendall_corr, _ = kendalltau(lagged[valid_idx], data[target_col][valid_idx])
                            
                            lagged_corr[col] = {
                                'pearson': pearson_corr,
                                'spearman': spearman_corr,
                                'kendall': kendall_corr
                            }
                
                correlations[f'lag_{lag}'] = lagged_corr
        
        return correlations
    
    def test_granger_causality(self,
                              data: pd.DataFrame,
                              target_col: str,
                              max_lag: int = 12,
                              significance: float = 0.05) -> Dict[str, Any]:
        """
        Test Granger causality between variables
        
        Args:
            data: DataFrame with time series
            target_col: Target variable name
            max_lag: Maximum lag to test
            significance: Significance level
            
        Returns:
            Dictionary with causality test results
        """
        results = {}
        
        for col in data.columns:
            if col != target_col:
                try:
                    # Prepare data for test
                    test_data = data[[target_col, col]].dropna()
                    
                    if len(test_data) > max_lag * 2:
                        # Test if col Granger-causes target
                        gc_result = grangercausalitytests(test_data, max_lag, verbose=False)
                        
                        # Extract p-values for each lag
                        p_values = {}
                        optimal_lag = None
                        min_p_value = 1.0
                        
                        for lag in range(1, max_lag + 1):
                            # Get p-value from F-test
                            p_val = gc_result[lag][0]['ssr_ftest'][1]
                            p_values[lag] = p_val
                            
                            if p_val < min_p_value:
                                min_p_value = p_val
                                optimal_lag = lag
                        
                        results[col] = {
                            'optimal_lag': optimal_lag,
                            'min_p_value': min_p_value,
                            'causes_target': min_p_value < significance,
                            'p_values_by_lag': p_values
                        }
                        
                except Exception as e:
                    logger.warning(f"Granger causality test failed for {col}: {e}")
                    results[col] = {'error': str(e)}
        
        return results
    
    def test_cointegration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test for cointegration using Johansen test
        
        Args:
            data: DataFrame with time series
            
        Returns:
            Dictionary with cointegration test results
        """
        try:
            # Remove NaN values
            data_clean = data.dropna()
            
            # Johansen test
            result = coint_johansen(data_clean, det_order=0, k_ar_diff=1)
            
            # Extract results
            trace_stats = result.lr1
            critical_values_trace = result.cvt
            eigenvalues = result.eig
            
            # Determine number of cointegrating relationships
            n_coint = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > critical_values_trace[i, 1]:  # 5% significance
                    n_coint += 1
                else:
                    break
            
            return {
                'n_relationships': n_coint,
                'trace_statistics': trace_stats.tolist(),
                'critical_values_5pct': critical_values_trace[:, 1].tolist(),
                'eigenvalues': eigenvalues.tolist(),
                'has_cointegration': n_coint > 0
            }
            
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return {'error': str(e)}
    
    def fit_var_model(self,
                     data: pd.DataFrame,
                     max_lag: int = 12) -> Tuple[Optional[VAR], Dict[str, Any]]:
        """
        Fit Vector Autoregression model
        
        Args:
            data: DataFrame with time series
            max_lag: Maximum lag order to consider
            
        Returns:
            Fitted VAR model and diagnostics
        """
        try:
            # Clean data
            data_clean = data.dropna()
            
            # Create VAR model
            model = VAR(data_clean)
            
            # Select optimal lag order
            lag_selection = model.select_order(max_lag)
            optimal_lag = lag_selection.aic
            
            # Fit model
            fitted_model = model.fit(optimal_lag)
            
            # Generate diagnostics
            diagnostics = {
                'optimal_lag': optimal_lag,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'coefficients': fitted_model.params.to_dict(),
                'residual_correlation': fitted_model.resid_corr.tolist()
            }
            
            # Test residuals for autocorrelation
            for col in data_clean.columns:
                lb_test = acorr_ljungbox(fitted_model.resid[col], lags=10, return_df=True)
                diagnostics[f'ljungbox_{col}'] = {
                    'p_values': lb_test['lb_pvalue'].tolist(),
                    'no_autocorrelation': (lb_test['lb_pvalue'] > 0.05).all()
                }
            
            return fitted_model, diagnostics
            
        except Exception as e:
            logger.error(f"VAR model fitting failed: {e}")
            return None, {'error': str(e)}
    
    def create_lagged_features(self,
                              data: pd.DataFrame,
                              target_col: str,
                              lags: List[int] = [1, 3, 6, 12],
                              rolling_windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create lagged and rolling features for ML models
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            # Original feature
            features[col] = data[col]
            
            # Lagged features
            for lag in lags:
                features[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # Rolling statistics
            for window in rolling_windows:
                features[f'{col}_roll_mean_{window}'] = data[col].rolling(window).mean()
                features[f'{col}_roll_std_{window}'] = data[col].rolling(window).std()
                
                # Rolling correlation with target
                if col != target_col:
                    rolling_corr = data[col].rolling(window).corr(data[target_col])
                    features[f'{col}_roll_corr_{window}'] = rolling_corr
        
        # Add time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['year'] = data.index.year
            features['days_in_month'] = data.index.days_in_month
        
        return features
    
    def train_ml_models(self,
                       features: pd.DataFrame,
                       target: pd.Series,
                       test_size: int = 24,
                       n_splits: int = 5) -> Dict[str, Any]:
        """
        Train and evaluate machine learning models
        
        Args:
            features: Feature DataFrame
            target: Target series
            test_size: Size of test set (months)
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with model results
        """
        # Remove NaN values and infinite values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        # Replace infinite values with NaN and then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN after replacement
        valid_idx2 = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx2]
        y = y[valid_idx2]
        
        # Split data
        split_point = len(X) - test_size
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        results['random_forest'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'feature_importance': dict(zip(X.columns, rf_model.feature_importances_)),
            'predictions': rf_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        results['xgboost'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': mean_absolute_error(y_test, xgb_pred),
            'r2': r2_score(y_test, xgb_pred),
            'feature_importance': dict(zip(X.columns, xgb_model.feature_importances_)),
            'predictions': xgb_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores_rf = []
        cv_scores_xgb = []
        
        for train_idx, val_idx in tscv.split(X_train):
            # RF CV
            X_cv_train = X_train_scaled[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train_scaled[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            rf_model_cv = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf_model_cv.fit(X_cv_train, y_cv_train)
            cv_pred = rf_model_cv.predict(X_cv_val)
            cv_scores_rf.append(r2_score(y_cv_val, cv_pred))
            
            # XGB CV
            xgb_model_cv = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
            xgb_model_cv.fit(X_cv_train, y_cv_train)
            cv_pred = xgb_model_cv.predict(X_cv_val)
            cv_scores_xgb.append(r2_score(y_cv_val, cv_pred))
        
        results['cross_validation'] = {
            'rf_cv_scores': cv_scores_rf,
            'rf_cv_mean': np.mean(cv_scores_rf),
            'xgb_cv_scores': cv_scores_xgb,
            'xgb_cv_mean': np.mean(cv_scores_xgb)
        }
        
        return results
    
    def perform_complete_analysis(self,
                                 data: pd.DataFrame,
                                 target_col: str = None) -> AnalysisResults:
        """
        Perform complete time series analysis
        
        Args:
            data: DataFrame with time series data
            target_col: Name of target column
            
        Returns:
            AnalysisResults object with all results
        """
        if target_col is None:
            target_col = self.target_variable
        
        logger.info("Starting comprehensive time series analysis...")
        
        # 1. Test stationarity
        logger.info("Testing stationarity...")
        stationarity_results = {}
        for col in data.columns:
            stationarity_results[col] = self.test_stationarity(data[col], col)
        
        # 2. Calculate correlations
        logger.info("Calculating correlations...")
        correlations = self.calculate_correlations(data, target_col)
        
        # 3. Test Granger causality
        logger.info("Testing Granger causality...")
        causality = self.test_granger_causality(data, target_col)
        
        # 4. Test cointegration
        logger.info("Testing cointegration...")
        cointegration = self.test_cointegration(data)
        
        # 5. Fit VAR model
        logger.info("Fitting VAR model...")
        var_model, var_diagnostics = self.fit_var_model(data)
        
        # 6. Create features and train ML models
        logger.info("Training ML models...")
        features = self.create_lagged_features(data, target_col)
        ml_results = self.train_ml_models(features, data[target_col])
        
        # Compile results
        results = AnalysisResults(
            correlations=correlations,
            causality=causality,
            stationarity=stationarity_results,
            cointegration=cointegration,
            forecasts={
                'var': var_diagnostics,
                'ml': ml_results
            },
            feature_importance=ml_results['random_forest']['feature_importance'],
            validation_metrics={
                'rf_r2': ml_results['random_forest']['r2'],
                'xgb_r2': ml_results['xgboost']['r2'],
                'rf_rmse': ml_results['random_forest']['rmse'],
                'xgb_rmse': ml_results['xgboost']['rmse']
            }
        )
        
        logger.info("Analysis complete!")
        return results


def main():
    """Example usage of time series analyzer"""
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    
    # Generate synthetic data
    data = pd.DataFrame({
        'rent_growth': np.random.randn(60).cumsum() + 2,
        'fed_funds': np.random.randn(60).cumsum() + 3,
        'unemployment': np.random.randn(60).cumsum() + 5,
        'cpi': np.random.randn(60).cumsum() + 100,
        'gdp': np.random.randn(60).cumsum() + 1000
    }, index=dates)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(target_variable='rent_growth')
    
    # Perform analysis
    results = analyzer.perform_complete_analysis(data, 'rent_growth')
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"\nValidation Metrics:")
    for metric, value in results.validation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nCointegration: {results.cointegration.get('has_cointegration', False)}")
    
    print(f"\nTop 5 Features by Importance:")
    if results.feature_importance:
        sorted_features = sorted(results.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()