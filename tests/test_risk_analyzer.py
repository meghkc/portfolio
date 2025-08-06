"""
Unit tests for risk analyzer
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_analyzer import RiskAnalyzer
from tests.conftest import TEST_PORTFOLIO_CONFIG

class TestRiskAnalyzer(unittest.TestCase):
    """Test cases for RiskAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TEST_PORTFOLIO_CONFIG.copy()
        self.risk_analyzer = RiskAnalyzer(self.config)
        
        # Create sample returns data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        
        self.sample_returns = {
            'AAPL': pd.Series(np.random.normal(0.001, 0.02, 252), index=dates),
            'GOOGL': pd.Series(np.random.normal(0.0008, 0.025, 252), index=dates),
            'SPY': pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
        }
        
        self.sample_weights = {
            'AAPL': 0.4,
            'GOOGL': 0.3,
            'SPY': 0.3
        }
    
    def test_risk_analyzer_initialization(self):
        """Test risk analyzer initialization"""
        self.assertIsNotNone(self.risk_analyzer)
        self.assertEqual(self.risk_analyzer.trading_days, 252)
        self.assertEqual(self.risk_analyzer.risk_free_rate, 0.045)
    
    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation"""
        volatility, correlation_matrix = self.risk_analyzer.calculate_portfolio_volatility(
            self.sample_returns, self.sample_weights
        )
        
        # Assertions
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)
        self.assertLess(volatility, 2)  # Should be less than 200% annualized
        
        # Check correlation matrix
        self.assertIsInstance(correlation_matrix, pd.DataFrame)
        self.assertEqual(len(correlation_matrix), 3)  # 3x3 matrix
        
        # Diagonal should be 1 (perfect self-correlation)
        for symbol in correlation_matrix.columns:
            self.assertAlmostEqual(correlation_matrix.loc[symbol, symbol], 1.0, places=10)
    
    def test_calculate_var_cvar(self):
        """Test VaR and CVaR calculations"""
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        var, cvar = self.risk_analyzer.calculate_var_cvar(portfolio_returns, confidence_level=0.05)
        
        # Assertions
        self.assertIsInstance(var, float)
        self.assertIsInstance(cvar, float)
        self.assertLess(var, 0)  # VaR should be negative (loss)
        self.assertLess(cvar, var)  # CVaR should be more negative than VaR
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        portfolio_return = 0.12  # 12% annual return
        portfolio_volatility = 0.20  # 20% annual volatility
        
        sharpe = self.risk_analyzer.calculate_sharpe_ratio(portfolio_return, portfolio_volatility)
        
        # Expected: (0.12 - 0.045) / 0.20 = 0.375
        expected_sharpe = (portfolio_return - self.risk_analyzer.risk_free_rate) / portfolio_volatility
        self.assertAlmostEqual(sharpe, expected_sharpe, places=3)
        self.assertIsInstance(sharpe, float)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        # Create returns with some negative values
        returns = pd.Series([0.02, -0.01, 0.03, -0.015, 0.025, -0.005, 0.01])
        
        sortino = self.risk_analyzer.calculate_sortino_ratio(returns, target_return=0.0)
        
        self.assertIsInstance(sortino, float)
        self.assertNotEqual(sortino, 0)  # Should calculate a meaningful ratio
    
    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create price series with a clear drawdown
        prices = pd.Series([100, 105, 110, 95, 85, 90, 108, 115])
        
        drawdown_metrics = self.risk_analyzer.calculate_maximum_drawdown(prices)
        
        # Assertions
        self.assertIn('max_drawdown', drawdown_metrics)
        self.assertIn('max_drawdown_duration', drawdown_metrics)
        self.assertIn('current_drawdown', drawdown_metrics)
        
        # Max drawdown should be negative
        self.assertLess(drawdown_metrics['max_drawdown'], 0)
        
        # Duration should be positive integer
        self.assertIsInstance(drawdown_metrics['max_drawdown_duration'], int)
        self.assertGreaterEqual(drawdown_metrics['max_drawdown_duration'], 0)
    
    def test_calculate_beta(self):
        """Test beta calculation"""
        # Create correlated returns for asset and benchmark
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        asset_returns = benchmark_returns * 1.2 + pd.Series(np.random.normal(0, 0.01, 252))
        
        beta = self.risk_analyzer.calculate_beta(asset_returns, benchmark_returns)
        
        self.assertIsInstance(beta, float)
        self.assertGreater(beta, 0.5)  # Should be positive and reasonable
        self.assertLess(beta, 3.0)
    
    def test_assess_concentration_risk(self):
        """Test concentration risk assessment"""
        # Test with concentrated portfolio
        concentrated_weights = {'AAPL': 0.7, 'GOOGL': 0.2, 'SPY': 0.1}
        
        concentration = self.risk_analyzer.assess_concentration_risk(concentrated_weights)
        
        # Assertions
        self.assertIn('hhi', concentration)
        self.assertIn('effective_holdings', concentration)
        self.assertIn('max_position', concentration)
        self.assertIn('top3_concentration', concentration)
        
        # HHI should be high for concentrated portfolio
        self.assertGreater(concentration['hhi'], 0.3)
        
        # Max position should be 70%
        self.assertAlmostEqual(concentration['max_position'], 0.7, places=2)
        
        # Effective holdings should be low
        self.assertLess(concentration['effective_holdings'], 2.0)
    
    def test_assess_concentration_risk_diversified(self):
        """Test concentration risk with diversified portfolio"""
        # Test with diversified portfolio
        diversified_weights = {f'STOCK_{i}': 0.1 for i in range(10)}
        
        concentration = self.risk_analyzer.assess_concentration_risk(diversified_weights)
        
        # HHI should be low for diversified portfolio
        self.assertLess(concentration['hhi'], 0.2)
        
        # Effective holdings should be close to actual holdings count
        self.assertGreater(concentration['effective_holdings'], 8.0)
        
        # Max position should be 10%
        self.assertAlmostEqual(concentration['max_position'], 0.1, places=2)
    
    def test_generate_risk_report(self):
        """Test comprehensive risk report generation"""
        # Prepare portfolio data
        portfolio_data = {
            'weights': self.sample_weights,
            'returns_data': self.sample_returns,
            'historical_data': {
                symbol: pd.DataFrame({'Close': np.cumprod(1 + returns)})
                for symbol, returns in self.sample_returns.items()
            }
        }
        
        risk_report = self.risk_analyzer.generate_risk_report(portfolio_data)
        
        # Assertions
        self.assertIsInstance(risk_report, dict)
        self.assertIn('timestamp', risk_report)
        self.assertIn('portfolio_metrics', risk_report)
        self.assertIn('individual_risks', risk_report)
        self.assertIn('correlation_analysis', risk_report)
        self.assertIn('risk_warnings', risk_report)
        
        # Check timestamp
        self.assertIsInstance(risk_report['timestamp'], datetime)
        
        # Check portfolio metrics
        portfolio_metrics = risk_report['portfolio_metrics']
        self.assertIn('volatility', portfolio_metrics)
        self.assertIn('concentration', portfolio_metrics)
        
        # Check individual risks
        individual_risks = risk_report['individual_risks']
        for symbol in self.sample_weights:
            if symbol in individual_risks:
                self.assertIn('weight', individual_risks[symbol])
                self.assertIn('volatility', individual_risks[symbol])
    
    def test_generate_risk_warnings(self):
        """Test risk warning generation"""
        # Create a risk report with high volatility
        risk_report = {
            'portfolio_metrics': {
                'volatility': 0.45,  # High portfolio volatility
                'concentration': {
                    'max_position': 0.20  # High concentration
                }
            },
            'individual_risks': {
                'AAPL': {
                    'volatility': 0.70,  # High individual volatility
                    'current_drawdown': -0.35  # Large drawdown
                }
            }
        }
        
        warnings = self.risk_analyzer._generate_risk_warnings(risk_report)
        
        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0)  # Should generate warnings
        
        # Check for specific warning types
        warning_text = ' '.join(warnings)
        self.assertIn('volatility', warning_text.lower())

if __name__ == '__main__':
    unittest.main()
