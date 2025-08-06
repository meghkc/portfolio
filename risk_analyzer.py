"""
Risk analysis and portfolio metrics calculation
Implements advanced risk metrics including VaR, portfolio volatility, and correlation analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Comprehensive portfolio risk analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        self.trading_days = config['data']['trading_days_per_year']
        self.risk_free_rate = config['data']['risk_free_rate']
    
    def calculate_portfolio_volatility(self, returns_data: Dict[str, pd.Series], 
                                     weights: Dict[str, float]) -> Tuple[float, pd.DataFrame]:
        """
        Calculate portfolio volatility using covariance matrix
        Formula: σ_portfolio = √(w^T * Σ * w)
        Returns volatility and correlation matrix
        """
        logger.info("Calculating portfolio volatility")
        
        try:
            # Align returns data with weights
            aligned_returns = []
            aligned_symbols = []
            weight_vector = []
            
            for symbol, weight in weights.items():
                if symbol in returns_data and not returns_data[symbol].empty:
                    aligned_returns.append(returns_data[symbol])
                    aligned_symbols.append(symbol)
                    weight_vector.append(weight)
            
            if not aligned_returns:
                logger.warning("No valid returns data for volatility calculation")
                return 0.0, pd.DataFrame()
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(dict(zip(aligned_symbols, aligned_returns)))
            returns_df = returns_df.dropna()
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov() * self.trading_days
            
            # Convert weights to numpy array
            weights_array = np.array(weight_vector)
            weights_array = weights_array / np.sum(weights_array)  # Normalize weights
            
            # Calculate portfolio variance: w^T * Σ * w
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix.values, weights_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate correlation matrix for additional insights
            correlation_matrix = returns_df.corr()
            
            logger.info(f"Portfolio volatility: {portfolio_volatility:.1%}")
            return portfolio_volatility, correlation_matrix
            
        except Exception as e:
            logger.error(f"Portfolio volatility calculation failed: {e}")
            return 0.0, pd.DataFrame()
    
    def calculate_var_cvar(self, portfolio_returns: pd.Series, confidence_level: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        """
        try:
            # Sort returns
            sorted_returns = portfolio_returns.sort_values()
            
            # Calculate VaR (percentile)
            var_index = int(confidence_level * len(sorted_returns))
            var = sorted_returns.iloc[var_index]
            
            # Calculate CVaR (expected shortfall)
            cvar = sorted_returns.iloc[:var_index].mean()
            
            logger.debug(f"VaR ({confidence_level:.1%}): {var:.1%}, CVaR: {cvar:.1%}")
            return var, cvar
            
        except Exception as e:
            logger.error(f"VaR/CVaR calculation failed: {e}")
            return 0.0, 0.0
    
    def calculate_sharpe_ratio(self, portfolio_return: float, portfolio_volatility: float) -> float:
        """Calculate Sharpe ratio: (Return - Risk-free rate) / Volatility"""
        try:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            logger.debug(f"Sharpe ratio: {sharpe:.2f}")
            return sharpe
        except:
            return 0.0
    
    def calculate_sortino_ratio(self, portfolio_returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio using downside deviation
        Only considers negative volatility (downside risk)
        """
        try:
            # Calculate downside returns (below target)
            downside_returns = portfolio_returns[portfolio_returns < target_return]
            
            if len(downside_returns) == 0:
                return float('inf')  # No downside
            
            # Calculate downside deviation
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.trading_days)
            
            # Calculate mean return
            mean_return = portfolio_returns.mean() * self.trading_days
            
            sortino = (mean_return - target_return) / downside_deviation
            logger.debug(f"Sortino ratio: {sortino:.2f}")
            return sortino
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            return 0.0
    
    def calculate_maximum_drawdown(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        try:
            # Calculate running maximum
            peak = portfolio_values.expanding().max()
            
            # Calculate drawdown series
            drawdown = (portfolio_values - peak) / peak
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown duration
            drawdown_duration = 0
            current_dd = 0
            max_dd_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_dd += 1
                    max_dd_duration = max(max_dd_duration, current_dd)
                else:
                    current_dd = 0
            
            logger.debug(f"Max drawdown: {max_drawdown:.1%}, Duration: {max_dd_duration} days")
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_dd_duration,
                'current_drawdown': drawdown.iloc[-1],
                'recovery_factor': portfolio_values.iloc[-1] / peak.iloc[-1] - 1
            }
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return {}
    
    def calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark (e.g., SPY)"""
        try:
            # Align the data
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data points
                return 1.0
            
            # Calculate beta using linear regression
            covariance = aligned_data['asset'].cov(aligned_data['benchmark'])
            benchmark_variance = aligned_data['benchmark'].var()
            
            beta = covariance / benchmark_variance
            logger.debug(f"Beta: {beta:.2f}")
            return beta
            
        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    def assess_concentration_risk(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze portfolio concentration using Herfindahl-Hirschman Index and other metrics
        """
        try:
            weight_values = np.array(list(weights.values()))
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = np.sum(weight_values ** 2)
            
            # Effective number of holdings
            effective_holdings = 1 / hhi if hhi > 0 else 0
            
            # Largest position weight
            max_weight = np.max(weight_values)
            
            # Top 3 concentration
            top3_weight = np.sum(np.sort(weight_values)[-3:])
            
            # Gini coefficient (inequality measure)
            gini = self._calculate_gini_coefficient(weight_values)
            
            concentration_metrics = {
                'hhi': hhi,
                'effective_holdings': effective_holdings,
                'max_position': max_weight,
                'top3_concentration': top3_weight,
                'gini_coefficient': gini
            }
            
            logger.debug(f"Concentration - HHI: {hhi:.3f}, Effective holdings: {effective_holdings:.1f}")
            return concentration_metrics
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return {}
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for portfolio concentration"""
        try:
            # Sort weights
            sorted_weights = np.sort(weights)
            n = len(sorted_weights)
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_weights)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights))) / (n * np.sum(sorted_weights)) - (n + 1) / n
            
            return gini
        except:
            return 0.0
    
    def generate_risk_report(self, portfolio_data: Dict) -> Dict:
        """
        Generate comprehensive risk analysis report
        """
        logger.info("Generating comprehensive risk report")
        
        risk_report = {
            'timestamp': datetime.now(),
            'portfolio_metrics': {},
            'individual_risks': {},
            'correlation_analysis': {},
            'risk_warnings': []
        }
        
        try:
            # Extract data
            weights = portfolio_data.get('weights', {})
            returns_data = portfolio_data.get('returns_data', {})
            historical_data = portfolio_data.get('historical_data', {})
            
            # Portfolio-level risk metrics
            portfolio_vol, corr_matrix = self.calculate_portfolio_volatility(returns_data, weights)
            risk_report['portfolio_metrics']['volatility'] = portfolio_vol
            risk_report['correlation_analysis']['matrix'] = corr_matrix
            
            # Concentration risk
            concentration = self.assess_concentration_risk(weights)
            risk_report['portfolio_metrics']['concentration'] = concentration
            
            # Individual asset risk assessment
            for symbol, weight in weights.items():
                if symbol in historical_data and not historical_data[symbol].empty:
                    asset_data = historical_data[symbol]
                    asset_returns = asset_data['Close'].pct_change().dropna()
                    
                    # Individual volatility
                    asset_vol = asset_returns.std() * np.sqrt(self.trading_days)
                    
                    # Maximum drawdown
                    asset_dd = self.calculate_maximum_drawdown(asset_data['Close'])
                    
                    risk_report['individual_risks'][symbol] = {
                        'weight': weight,
                        'volatility': asset_vol,
                        'max_drawdown': asset_dd.get('max_drawdown', 0),
                        'current_drawdown': asset_dd.get('current_drawdown', 0)
                    }
            
            # Generate risk warnings
            risk_report['risk_warnings'] = self._generate_risk_warnings(risk_report)
            
            logger.info("Risk report generated successfully")
            
        except Exception as e:
            logger.error(f"Risk report generation failed: {e}")
            
        return risk_report
    
    def _generate_risk_warnings(self, risk_report: Dict) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []
        
        try:
            # Portfolio volatility warning
            portfolio_vol = risk_report['portfolio_metrics'].get('volatility', 0)
            if portfolio_vol > 0.35:  # >35% volatility
                warnings.append(f"High portfolio volatility: {portfolio_vol:.1%}")
            
            # Concentration warnings
            concentration = risk_report['portfolio_metrics'].get('concentration', {})
            max_position = concentration.get('max_position', 0)
            if max_position > 0.15:  # >15% in single position
                warnings.append(f"High concentration risk: {max_position:.1%} in largest position")
            
            # Individual asset warnings
            for symbol, metrics in risk_report['individual_risks'].items():
                if metrics['volatility'] > 0.60:  # >60% individual volatility
                    warnings.append(f"High volatility in {symbol}: {metrics['volatility']:.1%}")
                
                if metrics['current_drawdown'] < -0.30:  # >30% current drawdown
                    warnings.append(f"Large drawdown in {symbol}: {metrics['current_drawdown']:.1%}")
            
        except Exception as e:
            logger.error(f"Risk warning generation failed: {e}")
            
        return warnings
