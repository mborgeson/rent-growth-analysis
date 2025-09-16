"""
Specialized Analysis Agents for Multifamily Rent Growth Analysis
Each agent approaches the relationship analysis from a different perspective
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RelationshipFinding:
    """Container for relationship findings from an agent"""
    agent_id: str
    variable_pair: Tuple[str, str]
    relationship_type: str  # 'strong', 'moderate', 'weak', 'none'
    confidence: float  # 0-1 confidence in finding
    evidence: Dict[str, Any]
    lag_structure: Optional[int]
    direction: Optional[str]  # 'bidirectional', 'unidirectional', 'none'
    interpretation: str


class AnalysisAgent(ABC):
    """Base class for analysis agents"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.findings = []
        
    @abstractmethod
    def analyze_relationships(self, data: pd.DataFrame, target: str) -> List[RelationshipFinding]:
        """Analyze relationships between variables"""
        pass
    
    def classify_relationship_strength(self, metric: float, thresholds: Dict[str, float]) -> str:
        """Classify relationship strength based on metric and thresholds"""
        if abs(metric) >= thresholds['strong']:
            return 'strong'
        elif abs(metric) >= thresholds['moderate']:
            return 'moderate'
        elif abs(metric) >= thresholds['weak']:
            return 'weak'
        else:
            return 'none'


class EconometricAgent(AnalysisAgent):
    """
    Agent specializing in econometric analysis
    Focuses on causality, cointegration, and structural relationships
    """
    
    def __init__(self):
        super().__init__("an-econ-001", "Econometrician")
        self.methodology = "Time series econometrics with focus on causality"
        
    def analyze_relationships(self, data: pd.DataFrame, target: str) -> List[RelationshipFinding]:
        """Perform econometric analysis of relationships"""
        findings = []
        
        for col in data.columns:
            if col != target:
                # 1. Granger Causality Test
                causality_result = self._test_granger_causality(data, target, col)
                
                # 2. Cointegration Test
                coint_result = self._test_cointegration(data[[target, col]])
                
                # 3. VAR Impulse Response
                impulse_result = self._analyze_impulse_response(data[[target, col]])
                
                # Synthesize findings
                relationship_type = self._determine_relationship_type(
                    causality_result, coint_result, impulse_result
                )
                
                confidence = self._calculate_confidence(
                    causality_result, coint_result, impulse_result
                )
                
                finding = RelationshipFinding(
                    agent_id=self.agent_id,
                    variable_pair=(col, target),
                    relationship_type=relationship_type,
                    confidence=confidence,
                    evidence={
                        'granger_causality': causality_result,
                        'cointegration': coint_result,
                        'impulse_response': impulse_result
                    },
                    lag_structure=causality_result.get('optimal_lag'),
                    direction=self._determine_direction(causality_result),
                    interpretation=self._generate_interpretation(
                        col, target, relationship_type, causality_result, coint_result
                    )
                )
                
                findings.append(finding)
                
        return findings
    
    def _test_granger_causality(self, data: pd.DataFrame, target: str, predictor: str) -> Dict:
        """Test Granger causality between variables"""
        try:
            test_data = data[[target, predictor]].dropna()
            if len(test_data) < 50:
                return {'error': 'Insufficient data'}
            
            max_lag = min(12, len(test_data) // 10)
            gc_result = grangercausalitytests(test_data, max_lag, verbose=False)
            
            # Find optimal lag with minimum p-value
            min_p = 1.0
            optimal_lag = 1
            
            for lag in range(1, max_lag + 1):
                p_val = gc_result[lag][0]['ssr_ftest'][1]
                if p_val < min_p:
                    min_p = p_val
                    optimal_lag = lag
            
            return {
                'causes_target': min_p < 0.05,
                'p_value': min_p,
                'optimal_lag': optimal_lag,
                'f_statistic': gc_result[optimal_lag][0]['ssr_ftest'][0]
            }
            
        except Exception as e:
            logger.warning(f"Granger causality test failed for {predictor}: {e}")
            return {'error': str(e)}
    
    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        """Test for cointegration between variables"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < 50:
                return {'error': 'Insufficient data'}
            
            # Johansen test
            result = coint_johansen(clean_data, det_order=0, k_ar_diff=1)
            
            # Check if cointegrated at 5% level
            trace_stat = result.lr1[0]
            critical_value = result.cvt[0, 1]  # 5% critical value
            
            return {
                'is_cointegrated': trace_stat > critical_value,
                'trace_statistic': float(trace_stat),
                'critical_value_5pct': float(critical_value),
                'eigenvalue': float(result.eig[0])
            }
            
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return {'error': str(e)}
    
    def _analyze_impulse_response(self, data: pd.DataFrame) -> Dict:
        """Analyze impulse response functions"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < 50:
                return {'error': 'Insufficient data'}
            
            # Fit VAR model
            model = VAR(clean_data)
            lag_order = model.select_order(maxlags=8)
            var_result = model.fit(lag_order.aic)
            
            # Calculate impulse response
            irf = var_result.irf(10)
            
            # Get response of target to shock in predictor
            response = irf.irfs[:, 0, 1]  # Target response to predictor shock
            
            # Calculate cumulative impact
            cumulative_impact = np.sum(response)
            peak_impact = np.max(np.abs(response))
            peak_period = np.argmax(np.abs(response))
            
            return {
                'cumulative_impact': float(cumulative_impact),
                'peak_impact': float(peak_impact),
                'peak_period': int(peak_period),
                'persistence': float(np.mean(np.abs(response[5:])))  # Late period average
            }
            
        except Exception as e:
            logger.warning(f"Impulse response analysis failed: {e}")
            return {'error': str(e)}
    
    def _determine_relationship_type(self, causality: Dict, coint: Dict, impulse: Dict) -> str:
        """Determine overall relationship type based on tests"""
        
        # Strong relationship: Granger causes, cointegrated, strong impulse
        if (causality.get('causes_target') and 
            coint.get('is_cointegrated') and 
            abs(impulse.get('peak_impact', 0)) > 0.5):
            return 'strong'
        
        # Moderate: Some evidence of relationship
        elif (causality.get('causes_target') or 
              coint.get('is_cointegrated') or 
              abs(impulse.get('peak_impact', 0)) > 0.3):
            return 'moderate'
        
        # Weak: Limited evidence
        elif (causality.get('p_value', 1.0) < 0.1 or 
              abs(impulse.get('peak_impact', 0)) > 0.1):
            return 'weak'
        
        else:
            return 'none'
    
    def _calculate_confidence(self, causality: Dict, coint: Dict, impulse: Dict) -> float:
        """Calculate confidence in findings"""
        confidence = 0.0
        
        # Weight different tests
        if not causality.get('error'):
            if causality.get('causes_target'):
                confidence += 0.4 * (1 - causality.get('p_value', 1.0))
            else:
                confidence += 0.1
        
        if not coint.get('error'):
            if coint.get('is_cointegrated'):
                confidence += 0.3
            else:
                confidence += 0.1
        
        if not impulse.get('error'):
            impact_strength = min(abs(impulse.get('peak_impact', 0)), 1.0)
            confidence += 0.3 * impact_strength
        
        return min(confidence, 1.0)
    
    def _determine_direction(self, causality: Dict) -> str:
        """Determine causal direction"""
        if causality.get('causes_target'):
            return 'unidirectional'
        else:
            return 'none'
    
    def _generate_interpretation(self, predictor: str, target: str, 
                                relationship_type: str, causality: Dict, coint: Dict) -> str:
        """Generate human-readable interpretation"""
        
        if relationship_type == 'strong':
            interp = f"{predictor} has a strong relationship with {target}. "
            if causality.get('causes_target'):
                interp += f"Changes in {predictor} significantly predict {target} with {causality.get('optimal_lag')} period lag. "
            if coint.get('is_cointegrated'):
                interp += f"Variables share a long-term equilibrium relationship."
                
        elif relationship_type == 'moderate':
            interp = f"{predictor} shows moderate relationship with {target}. "
            if causality.get('causes_target'):
                interp += f"Some predictive power detected at {causality.get('optimal_lag')} period lag."
            else:
                interp += "Limited causal evidence but structural connection exists."
                
        elif relationship_type == 'weak':
            interp = f"{predictor} has weak relationship with {target}. "
            interp += "Statistical evidence is limited but not absent."
            
        else:
            interp = f"{predictor} appears independent of {target}. "
            interp += "No significant statistical relationship detected."
        
        return interp


class MachineLearningAgent(AnalysisAgent):
    """
    Agent specializing in machine learning approaches
    Focuses on non-linear relationships and feature importance
    """
    
    def __init__(self):
        super().__init__("an-ml-001", "ML Specialist")
        self.methodology = "Machine learning with focus on non-linear patterns"
        
    def analyze_relationships(self, data: pd.DataFrame, target: str) -> List[RelationshipFinding]:
        """Perform ML-based relationship analysis"""
        findings = []
        
        # Prepare features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # 1. Feature Importance from Random Forest
        importance_scores = self._calculate_feature_importance(X_clean, y_clean)
        
        # 2. Mutual Information
        mi_scores = self._calculate_mutual_information(X_clean, y_clean)
        
        # 3. Non-linear correlation (using polynomial features)
        nonlinear_scores = self._test_nonlinear_relationships(X_clean, y_clean)
        
        # Generate findings for each variable
        for col in X.columns:
            # Combine evidence
            rf_importance = importance_scores.get(col, 0)
            mi_score = mi_scores.get(col, 0)
            nonlinear_score = nonlinear_scores.get(col, 0)
            
            # Determine relationship type
            relationship_type = self._determine_ml_relationship(
                rf_importance, mi_score, nonlinear_score
            )
            
            # Calculate confidence
            confidence = self._calculate_ml_confidence(
                rf_importance, mi_score, nonlinear_score
            )
            
            finding = RelationshipFinding(
                agent_id=self.agent_id,
                variable_pair=(col, target),
                relationship_type=relationship_type,
                confidence=confidence,
                evidence={
                    'random_forest_importance': rf_importance,
                    'mutual_information': mi_score,
                    'nonlinear_correlation': nonlinear_score
                },
                lag_structure=None,  # ML doesn't explicitly model lags
                direction='bidirectional' if relationship_type != 'none' else 'none',
                interpretation=self._generate_ml_interpretation(
                    col, target, relationship_type, rf_importance, mi_score, nonlinear_score
                )
            )
            
            findings.append(finding)
        
        return findings
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using Random Forest"""
        try:
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            importance_dict = dict(zip(X.columns, rf.feature_importances_))
            
            # Normalize to 0-1 scale
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {k: v/max_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate mutual information between features and target"""
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_dict = dict(zip(X.columns, mi_scores))
            
            # Normalize to 0-1 scale
            max_mi = max(mi_dict.values()) if mi_dict else 1
            if max_mi > 0:
                mi_dict = {k: v/max_mi for k, v in mi_dict.items()}
            
            return mi_dict
            
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            return {}
    
    def _test_nonlinear_relationships(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Test for non-linear relationships using polynomial features"""
        nonlinear_scores = {}
        
        for col in X.columns:
            try:
                # Test linear correlation
                linear_corr = np.corrcoef(X[col], y)[0, 1]
                
                # Test quadratic relationship
                X_poly = np.column_stack([X[col], X[col]**2])
                
                # Remove any infinite values
                valid_idx = ~(np.isinf(X_poly).any(axis=1))
                if valid_idx.sum() < 10:
                    nonlinear_scores[col] = abs(linear_corr)
                    continue
                
                X_poly_clean = X_poly[valid_idx]
                y_clean = y[valid_idx]
                
                # Fit polynomial model and calculate R²
                from sklearn.linear_model import LinearRegression
                poly_model = LinearRegression()
                poly_model.fit(X_poly_clean, y_clean)
                poly_r2 = poly_model.score(X_poly_clean, y_clean)
                
                # Fit linear model for comparison
                linear_model = LinearRegression()
                linear_model.fit(X_poly_clean[:, [0]], y_clean)
                linear_r2 = linear_model.score(X_poly_clean[:, [0]], y_clean)
                
                # Non-linearity score is improvement from polynomial
                nonlinear_improvement = max(0, poly_r2 - linear_r2)
                
                # Combine linear and nonlinear effects
                total_relationship = abs(linear_corr) + nonlinear_improvement
                nonlinear_scores[col] = min(total_relationship, 1.0)
                
            except Exception as e:
                logger.warning(f"Non-linear test failed for {col}: {e}")
                nonlinear_scores[col] = 0.0
        
        return nonlinear_scores
    
    def _determine_ml_relationship(self, rf_importance: float, mi_score: float, 
                                  nonlinear_score: float) -> str:
        """Determine relationship type from ML metrics"""
        
        # Average the three metrics
        avg_score = (rf_importance + mi_score + nonlinear_score) / 3
        
        if avg_score >= 0.6:
            return 'strong'
        elif avg_score >= 0.3:
            return 'moderate'
        elif avg_score >= 0.1:
            return 'weak'
        else:
            return 'none'
    
    def _calculate_ml_confidence(self, rf_importance: float, mi_score: float,
                                nonlinear_score: float) -> float:
        """Calculate confidence in ML findings"""
        
        # Weighted average with higher weight on consistent signals
        scores = [rf_importance, mi_score, nonlinear_score]
        
        # Base confidence is average
        base_confidence = np.mean(scores)
        
        # Boost confidence if all methods agree
        consistency_bonus = 0.2 * (1 - np.std(scores))
        
        return min(base_confidence + consistency_bonus, 1.0)
    
    def _generate_ml_interpretation(self, predictor: str, target: str,
                                   relationship_type: str, rf_imp: float,
                                   mi_score: float, nonlinear: float) -> str:
        """Generate ML-based interpretation"""
        
        if relationship_type == 'strong':
            interp = f"{predictor} shows strong predictive power for {target}. "
            if nonlinear > 0.5:
                interp += "Non-linear patterns detected, suggesting complex relationship. "
            interp += f"Feature importance: {rf_imp:.2f}, Information content: {mi_score:.2f}."
            
        elif relationship_type == 'moderate':
            interp = f"{predictor} has moderate predictive value for {target}. "
            if nonlinear > rf_imp:
                interp += "Non-linear effects present. "
            interp += f"Contributing {rf_imp:.1%} to model predictions."
            
        elif relationship_type == 'weak':
            interp = f"{predictor} shows limited predictive power for {target}. "
            interp += "May contribute in combination with other variables."
            
        else:
            interp = f"{predictor} appears uninformative for predicting {target}. "
            interp += "No significant patterns detected by ML algorithms."
        
        return interp


class StatisticalAgent(AnalysisAgent):
    """
    Agent specializing in classical statistical analysis
    Focuses on correlations, distributions, and hypothesis testing
    """
    
    def __init__(self):
        super().__init__("an-stat-001", "Statistician")
        self.methodology = "Classical statistics with rigorous hypothesis testing"
        
    def analyze_relationships(self, data: pd.DataFrame, target: str) -> List[RelationshipFinding]:
        """Perform statistical analysis of relationships"""
        findings = []
        
        for col in data.columns:
            if col != target:
                # 1. Correlation analysis (Pearson, Spearman, Kendall)
                correlations = self._calculate_correlations(data[col], data[target])
                
                # 2. Distribution similarity tests
                distribution_test = self._test_distribution_similarity(data[col], data[target])
                
                # 3. Lead-lag correlation analysis
                lead_lag = self._analyze_lead_lag_correlation(data[col], data[target])
                
                # 4. Stability over time
                stability = self._test_relationship_stability(data[col], data[target])
                
                # Determine relationship type
                relationship_type = self._determine_stat_relationship(
                    correlations, distribution_test, lead_lag, stability
                )
                
                # Calculate confidence
                confidence = self._calculate_stat_confidence(
                    correlations, distribution_test, lead_lag, stability
                )
                
                finding = RelationshipFinding(
                    agent_id=self.agent_id,
                    variable_pair=(col, target),
                    relationship_type=relationship_type,
                    confidence=confidence,
                    evidence={
                        'correlations': correlations,
                        'distribution_test': distribution_test,
                        'lead_lag': lead_lag,
                        'stability': stability
                    },
                    lag_structure=lead_lag.get('optimal_lag'),
                    direction='bidirectional' if correlations['pearson']['significant'] else 'none',
                    interpretation=self._generate_stat_interpretation(
                        col, target, relationship_type, correlations, lead_lag, stability
                    )
                )
                
                findings.append(finding)
        
        return findings
    
    def _calculate_correlations(self, x: pd.Series, y: pd.Series) -> Dict:
        """Calculate multiple correlation measures"""
        try:
            # Remove NaN values
            valid_idx = ~(x.isna() | y.isna())
            x_clean = x[valid_idx]
            y_clean = y[valid_idx]
            
            if len(x_clean) < 30:
                return {'error': 'Insufficient data'}
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
            
            # Kendall correlation
            kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
            
            return {
                'pearson': {
                    'coefficient': pearson_r,
                    'p_value': pearson_p,
                    'significant': pearson_p < 0.05
                },
                'spearman': {
                    'coefficient': spearman_r,
                    'p_value': spearman_p,
                    'significant': spearman_p < 0.05
                },
                'kendall': {
                    'coefficient': kendall_tau,
                    'p_value': kendall_p,
                    'significant': kendall_p < 0.05
                }
            }
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return {'error': str(e)}
    
    def _test_distribution_similarity(self, x: pd.Series, y: pd.Series) -> Dict:
        """Test if variables follow similar distributions"""
        try:
            # Normalize both series
            x_norm = (x - x.mean()) / x.std()
            y_norm = (y - y.mean()) / y.std()
            
            # Remove NaN values
            x_clean = x_norm.dropna()
            y_clean = y_norm.dropna()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(x_clean, y_clean)
            
            # Calculate distribution moments
            x_skew = stats.skew(x_clean)
            y_skew = stats.skew(y_clean)
            x_kurt = stats.kurtosis(x_clean)
            y_kurt = stats.kurtosis(y_clean)
            
            return {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'similar_distribution': ks_p > 0.05,
                'skew_difference': abs(x_skew - y_skew),
                'kurtosis_difference': abs(x_kurt - y_kurt)
            }
            
        except Exception as e:
            logger.warning(f"Distribution test failed: {e}")
            return {'error': str(e)}
    
    def _analyze_lead_lag_correlation(self, x: pd.Series, y: pd.Series, 
                                     max_lag: int = 12) -> Dict:
        """Analyze correlation at different lags"""
        try:
            correlations = {}
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # X leads Y
                    x_shifted = x.shift(-lag)
                    valid_idx = ~(x_shifted.isna() | y.isna())
                    if valid_idx.sum() > 30:
                        corr = np.corrcoef(x_shifted[valid_idx], y[valid_idx])[0, 1]
                        correlations[lag] = corr
                elif lag > 0:
                    # Y leads X
                    y_shifted = y.shift(lag)
                    valid_idx = ~(x.isna() | y_shifted.isna())
                    if valid_idx.sum() > 30:
                        corr = np.corrcoef(x[valid_idx], y_shifted[valid_idx])[0, 1]
                        correlations[lag] = corr
                else:
                    # Contemporaneous
                    valid_idx = ~(x.isna() | y.isna())
                    if valid_idx.sum() > 30:
                        corr = np.corrcoef(x[valid_idx], y[valid_idx])[0, 1]
                        correlations[lag] = corr
            
            # Find optimal lag
            if correlations:
                optimal_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]))
                max_correlation = correlations[optimal_lag]
            else:
                optimal_lag = 0
                max_correlation = 0
            
            return {
                'correlations_by_lag': correlations,
                'optimal_lag': optimal_lag,
                'max_correlation': max_correlation,
                'lead_lag_pattern': 'x_leads' if optimal_lag < 0 else 'y_leads' if optimal_lag > 0 else 'contemporaneous'
            }
            
        except Exception as e:
            logger.warning(f"Lead-lag analysis failed: {e}")
            return {'error': str(e)}
    
    def _test_relationship_stability(self, x: pd.Series, y: pd.Series, 
                                    window_size: int = 60) -> Dict:
        """Test if relationship is stable over time"""
        try:
            # Calculate rolling correlation
            rolling_corr = x.rolling(window_size).corr(y)
            
            # Remove NaN values
            rolling_corr_clean = rolling_corr.dropna()
            
            if len(rolling_corr_clean) < 10:
                return {'error': 'Insufficient data for stability test'}
            
            # Calculate stability metrics
            corr_mean = rolling_corr_clean.mean()
            corr_std = rolling_corr_clean.std()
            corr_min = rolling_corr_clean.min()
            corr_max = rolling_corr_clean.max()
            
            # Test for structural breaks using Chow test approximation
            mid_point = len(rolling_corr_clean) // 2
            first_half = rolling_corr_clean[:mid_point]
            second_half = rolling_corr_clean[mid_point:]
            
            # Test if means are different
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            
            return {
                'mean_correlation': corr_mean,
                'correlation_volatility': corr_std,
                'correlation_range': corr_max - corr_min,
                'is_stable': corr_std < 0.2 and p_value > 0.05,
                'structural_break_p_value': p_value,
                'coefficient_of_variation': corr_std / abs(corr_mean) if corr_mean != 0 else np.inf
            }
            
        except Exception as e:
            logger.warning(f"Stability test failed: {e}")
            return {'error': str(e)}
    
    def _determine_stat_relationship(self, correlations: Dict, distribution: Dict,
                                    lead_lag: Dict, stability: Dict) -> str:
        """Determine relationship type from statistical tests"""
        
        if 'error' in correlations:
            return 'none'
        
        # Get maximum correlation (absolute value)
        max_corr = max(
            abs(correlations.get('pearson', {}).get('coefficient', 0)),
            abs(correlations.get('spearman', {}).get('coefficient', 0)),
            abs(lead_lag.get('max_correlation', 0))
        )
        
        # Check if relationship is stable
        is_stable = stability.get('is_stable', False)
        
        # Strong: High correlation, significant, and stable
        if max_corr >= 0.7 and is_stable:
            return 'strong'
        
        # Moderate: Moderate correlation or unstable strong correlation
        elif max_corr >= 0.4 or (max_corr >= 0.6 and not is_stable):
            return 'moderate'
        
        # Weak: Low correlation but significant
        elif max_corr >= 0.2 and correlations.get('pearson', {}).get('significant', False):
            return 'weak'
        
        else:
            return 'none'
    
    def _calculate_stat_confidence(self, correlations: Dict, distribution: Dict,
                                  lead_lag: Dict, stability: Dict) -> float:
        """Calculate confidence in statistical findings"""
        
        confidence = 0.0
        
        # Correlation significance (40% weight)
        if not correlations.get('error'):
            if correlations.get('pearson', {}).get('significant'):
                confidence += 0.2
            if correlations.get('spearman', {}).get('significant'):
                confidence += 0.2
        
        # Stability (30% weight)
        if not stability.get('error'):
            if stability.get('is_stable'):
                confidence += 0.3
            else:
                confidence += 0.1
        
        # Lead-lag clarity (30% weight)
        if not lead_lag.get('error'):
            max_corr = abs(lead_lag.get('max_correlation', 0))
            confidence += 0.3 * min(max_corr / 0.7, 1.0)
        
        return min(confidence, 1.0)
    
    def _generate_stat_interpretation(self, predictor: str, target: str,
                                     relationship_type: str, correlations: Dict,
                                     lead_lag: Dict, stability: Dict) -> str:
        """Generate statistical interpretation"""
        
        if relationship_type == 'strong':
            interp = f"{predictor} shows strong statistical relationship with {target}. "
            
            if correlations.get('pearson', {}).get('coefficient', 0) > 0:
                interp += "Positive correlation detected. "
            else:
                interp += "Negative correlation detected. "
            
            if lead_lag.get('optimal_lag', 0) != 0:
                interp += f"Optimal lag: {abs(lead_lag.get('optimal_lag', 0))} periods. "
            
            if stability.get('is_stable'):
                interp += "Relationship is stable over time."
            else:
                interp += "Relationship varies over time."
                
        elif relationship_type == 'moderate':
            interp = f"{predictor} has moderate statistical association with {target}. "
            
            corr_val = correlations.get('pearson', {}).get('coefficient', 0)
            interp += f"Correlation coefficient: {corr_val:.2f}. "
            
            if not stability.get('is_stable'):
                interp += "Relationship strength varies across periods."
                
        elif relationship_type == 'weak':
            interp = f"{predictor} shows weak statistical relationship with {target}. "
            interp += "Signal present but not strong enough for reliable predictions."
            
        else:
            interp = f"{predictor} is statistically independent of {target}. "
            interp += "No significant correlation or pattern detected."
        
        return interp


class ConsensusBuilder:
    """
    Builds consensus from multiple agent findings
    """
    
    def __init__(self):
        self.agent_weights = {
            'an-econ-001': 0.35,  # Econometrician
            'an-ml-001': 0.35,    # ML Specialist
            'an-stat-001': 0.30   # Statistician
        }
        
    def build_consensus(self, all_findings: List[List[RelationshipFinding]]) -> Dict[str, Any]:
        """
        Build consensus from multiple agent findings
        
        Args:
            all_findings: List of findings from each agent
            
        Returns:
            Consensus report with aggregated findings
        """
        
        # Flatten all findings
        flat_findings = [f for agent_findings in all_findings for f in agent_findings]
        
        # Group by variable pair
        findings_by_pair = {}
        for finding in flat_findings:
            pair = finding.variable_pair
            if pair not in findings_by_pair:
                findings_by_pair[pair] = []
            findings_by_pair[pair].append(finding)
        
        # Build consensus for each variable pair
        consensus_results = {}
        
        for pair, findings in findings_by_pair.items():
            consensus = self._build_pair_consensus(findings)
            consensus_results[f"{pair[0]}_to_{pair[1]}"] = consensus
        
        # Generate overall summary
        summary = self._generate_summary(consensus_results)
        
        return {
            'consensus_results': consensus_results,
            'summary': summary,
            'methodology': self._describe_methodology(),
            'confidence_metrics': self._calculate_overall_confidence(consensus_results)
        }
    
    def _build_pair_consensus(self, findings: List[RelationshipFinding]) -> Dict:
        """Build consensus for a specific variable pair"""
        
        # Collect relationship types and confidences
        relationships = {}
        interpretations = []
        
        for finding in findings:
            agent_id = finding.agent_id
            weight = self.agent_weights.get(agent_id, 0.33)
            
            relationships[agent_id] = {
                'type': finding.relationship_type,
                'confidence': finding.confidence,
                'weight': weight,
                'interpretation': finding.interpretation
            }
            
            interpretations.append(finding.interpretation)
        
        # Calculate weighted consensus on relationship type
        type_scores = {'strong': 0, 'moderate': 0, 'weak': 0, 'none': 0}
        
        for agent_id, info in relationships.items():
            rel_type = info['type']
            confidence = info['confidence']
            weight = info['weight']
            
            type_scores[rel_type] += confidence * weight
        
        # Determine consensus relationship type
        consensus_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        consensus_confidence = type_scores[consensus_type] / sum(self.agent_weights.values())
        
        # Check for agreement
        agent_types = [info['type'] for info in relationships.values()]
        agreement_level = self._calculate_agreement(agent_types)
        
        return {
            'consensus_relationship': consensus_type,
            'consensus_confidence': consensus_confidence,
            'agreement_level': agreement_level,
            'agent_findings': relationships,
            'interpretations': interpretations,
            'recommendation': self._generate_recommendation(consensus_type, consensus_confidence, agreement_level)
        }
    
    def _calculate_agreement(self, agent_types: List[str]) -> str:
        """Calculate level of agreement between agents"""
        
        if len(set(agent_types)) == 1:
            return 'unanimous'
        elif len(set(agent_types)) == 2:
            return 'high'
        else:
            return 'moderate'
    
    def _generate_recommendation(self, consensus_type: str, confidence: float, agreement: str) -> str:
        """Generate actionable recommendation"""
        
        if consensus_type == 'strong' and agreement in ['unanimous', 'high']:
            return "HIGH CONFIDENCE: Include this variable as a primary predictor. Strong, consistent relationship detected across multiple methodologies."
        
        elif consensus_type == 'strong' and agreement == 'moderate':
            return "MODERATE-HIGH CONFIDENCE: Include as predictor but note methodological disagreement. Further investigation recommended."
        
        elif consensus_type == 'moderate':
            return "MODERATE CONFIDENCE: Include as secondary predictor. Relationship exists but with limited strength or consistency."
        
        elif consensus_type == 'weak':
            return "LOW CONFIDENCE: Consider for ensemble models only. Weak signal that may contribute in combination with others."
        
        else:
            return "EXCLUDE: No meaningful relationship detected. Variable appears independent of target."
    
    def _generate_summary(self, consensus_results: Dict) -> Dict:
        """Generate summary of all consensus findings"""
        
        # Categorize variables
        primary_predictors = []
        secondary_predictors = []
        weak_predictors = []
        independent_variables = []
        
        for var_pair, consensus in consensus_results.items():
            rel_type = consensus['consensus_relationship']
            confidence = consensus['consensus_confidence']
            
            if rel_type == 'strong' and confidence > 0.7:
                primary_predictors.append(var_pair)
            elif rel_type in ['strong', 'moderate'] and confidence > 0.5:
                secondary_predictors.append(var_pair)
            elif rel_type == 'weak':
                weak_predictors.append(var_pair)
            else:
                independent_variables.append(var_pair)
        
        return {
            'primary_predictors': primary_predictors,
            'secondary_predictors': secondary_predictors,
            'weak_predictors': weak_predictors,
            'independent_variables': independent_variables,
            'total_variables_analyzed': len(consensus_results),
            'predictive_variables_count': len(primary_predictors) + len(secondary_predictors)
        }
    
    def _describe_methodology(self) -> str:
        """Describe the consensus methodology"""
        
        return (
            "This consensus report synthesizes findings from three specialized agents:\n"
            "1. Econometric Agent (35% weight): Focuses on causality, cointegration, and structural relationships\n"
            "2. Machine Learning Agent (35% weight): Identifies non-linear patterns and feature importance\n"
            "3. Statistical Agent (30% weight): Performs classical correlation and distribution analysis\n\n"
            "Each agent independently analyzes relationships, then findings are aggregated using weighted voting. "
            "Agreement levels indicate consistency across methodologies, with unanimous agreement providing highest confidence."
        )
    
    def _calculate_overall_confidence(self, consensus_results: Dict) -> Dict:
        """Calculate overall confidence metrics"""
        
        confidences = [c['consensus_confidence'] for c in consensus_results.values()]
        agreements = [c['agreement_level'] for c in consensus_results.values()]
        
        return {
            'mean_confidence': np.mean(confidences) if confidences else 0,
            'median_confidence': np.median(confidences) if confidences else 0,
            'unanimous_agreement_rate': agreements.count('unanimous') / len(agreements) if agreements else 0,
            'high_agreement_rate': (agreements.count('unanimous') + agreements.count('high')) / len(agreements) if agreements else 0
        }