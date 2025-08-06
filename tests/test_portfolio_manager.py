"""
Unit tests for portfolio manager
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_manager import PortfolioManager
from tests.conftest import TEST_PORTFOLIO_CONFIG, MockDataFetcher, MockEmailSender

class TestPortfolioManager(unittest.TestCase):
    """Test cases for PortfolioManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TEST_PORTFOLIO_CONFIG.copy()
        
        # Mock the dependencies
        with patch('portfolio_manager.DataFetcher', MockDataFetcher), \
             patch('portfolio_manager.EmailSender', MockEmailSender), \
             patch('portfolio_manager.RiskAnalyzer') as mock_risk, \
             patch('portfolio_manager.NewsAnalyzer') as mock_news:
            
            self.portfolio = PortfolioManager(self.config)
    
    def test_portfolio_initialization(self):
        """Test portfolio manager initialization"""
        self.assertIsNotNone(self.portfolio)
        self.assertEqual(len(self.portfolio.holdings), 6)
        self.assertEqual(len(self.portfolio.watchlist), 2)
        self.assertIn('AAPL', self.portfolio.holdings)
        self.assertIn('TSLA', self.portfolio.watchlist)
    
    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Mock price data
        current_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'SPY': 400.0,
            'BTC-USD': 45000.0,
            'NVDA': 800.0
        }
        
        # Mock historical data
        historical_data = {}
        for symbol in current_prices:
            dates = pd.date_range(end='2025-01-01', periods=252, freq='D')
            prices = np.random.uniform(0.8, 1.2, 252) * current_prices[symbol]
            historical_data[symbol] = pd.DataFrame({
                'Close': prices
            }, index=dates)
        
        metrics = self.portfolio._calculate_portfolio_metrics(current_prices, historical_data)
        
        # Assertions
        self.assertIn('total_value', metrics)
        self.assertIn('current_allocations', metrics)
        self.assertIn('target_allocations', metrics)
        self.assertIn('allocation_drift', metrics)
        self.assertGreater(metrics['total_value'], 0)
        
        # Check that allocations sum to approximately 1
        total_current = sum(metrics['current_allocations'].values())
        self.assertAlmostEqual(total_current, 1.0, places=2)
    
    def test_analyze_performance(self):
        """Test performance analysis"""
        # Create mock historical data
        historical_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        for symbol in symbols:
            dates = pd.date_range(end='2025-01-01', periods=252, freq='D')
            returns = np.random.normal(0.001, 0.02, 252)
            prices = [100]  # Starting price
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            historical_data[symbol] = pd.DataFrame({
                'Close': prices
            }, index=dates)
        
        performance = self.portfolio._analyze_performance(historical_data)
        
        # Assertions
        self.assertIn('best_performers', performance)
        self.assertIn('worst_performers', performance)
        self.assertIn('high_volatility_assets', performance)
        self.assertIn('sector_performance', performance)
        
        # Check that we have performance data
        self.assertIsInstance(performance['best_performers'], list)
        self.assertIsInstance(performance['worst_performers'], list)
    
    @patch('portfolio_manager.update_portfolio_prices')
    def test_generate_weekly_report(self, mock_update_prices):
        """Test weekly report generation"""
        # Mock the update_portfolio_prices function
        mock_update_prices.return_value = self.portfolio.holdings
        
        # Mock the data fetcher methods
        with patch.object(self.portfolio.data_fetcher, 'fetch_current_prices') as mock_prices, \
             patch.object(self.portfolio.data_fetcher, 'fetch_historical_data') as mock_hist, \
             patch.object(self.portfolio.data_fetcher, 'fetch_news_for_symbols') as mock_news, \
             patch.object(self.portfolio.risk_analyzer, 'generate_risk_report') as mock_risk, \
             patch.object(self.portfolio.news_analyzer, 'analyze_news_sentiment') as mock_sentiment:
            
            # Set up mock returns
            mock_prices.return_value = {'AAPL': 150.0, 'GOOGL': 2500.0}
            mock_hist.return_value = {'AAPL': pd.DataFrame({'Close': [140, 145, 150]})}
            mock_news.return_value = {'AAPL': []}
            mock_risk.return_value = {'portfolio_metrics': {}, 'individual_risks': {}}
            mock_sentiment.return_value = {'symbol_sentiments': {}}
            
            report = self.portfolio.generate_weekly_report()
            
            # Assertions
            self.assertIsInstance(report, dict)
            self.assertIn('timestamp', report)
            self.assertIn('portfolio_summary', report)
            self.assertIn('performance_metrics', report)
            self.assertIsInstance(report['timestamp'], datetime)
    
    def test_generate_rebalancing_recommendations(self):
        """Test rebalancing recommendations"""
        current_prices = {'AAPL': 150.0, 'GOOGL': 2500.0}
        
        portfolio_metrics = {
            'current_allocations': {'AAPL': 0.25, 'GOOGL': 0.10},  # AAPL over-allocated
            'allocation_drift': {'AAPL': 0.05, 'GOOGL': -0.05}     # AAPL +5%, GOOGL -5%
        }
        
        risk_analysis = {
            'individual_risks': {
                'AAPL': {'volatility': 0.30, 'current_drawdown': -0.10},
                'GOOGL': {'volatility': 0.80, 'current_drawdown': -0.40}  # High vol, large drawdown
            }
        }
        
        recommendations = self.portfolio._generate_rebalancing_recommendations(
            current_prices, portfolio_metrics, risk_analysis
        )
        
        # Assertions
        self.assertIn('trim_positions', recommendations)
        self.assertIn('add_positions', recommendations)
        self.assertIn('risk_adjustments', recommendations)
        
        # Should recommend trimming AAPL (over-allocated)
        trim_symbols = [pos['symbol'] for pos in recommendations['trim_positions']]
        self.assertIn('AAPL', trim_symbols)
        
        # Should recommend risk adjustment for GOOGL (high volatility)
        risk_symbols = [adj['symbol'] for adj in recommendations['risk_adjustments']]
        self.assertIn('GOOGL', risk_symbols)
    
    def test_analyze_watchlist_opportunities(self):
        """Test watchlist analysis"""
        current_prices = {
            'TSLA': 180.0,  # Below entry target of 200
            'AMD': 120.0    # Above entry target of 100
        }
        
        opportunities = self.portfolio._analyze_watchlist_opportunities(current_prices)
        
        # Assertions
        self.assertIn('ready_to_buy', opportunities)
        self.assertIn('approaching_targets', opportunities)
        self.assertIn('overpriced', opportunities)
        
        # TSLA should be ready to buy (below target)
        ready_symbols = [opp['symbol'] for opp in opportunities['ready_to_buy']]
        self.assertIn('TSLA', ready_symbols)
        
        # AMD should be overpriced (above target)
        overpriced_symbols = [opp['symbol'] for opp in opportunities['overpriced']]
        self.assertIn('AMD', overpriced_symbols)
    
    def test_interpret_fear_greed(self):
        """Test Fear & Greed Index interpretation"""
        self.assertEqual(self.portfolio._interpret_fear_greed(10), "Extreme Fear")
        self.assertEqual(self.portfolio._interpret_fear_greed(30), "Fear")
        self.assertEqual(self.portfolio._interpret_fear_greed(50), "Neutral")
        self.assertEqual(self.portfolio._interpret_fear_greed(70), "Greed")
        self.assertEqual(self.portfolio._interpret_fear_greed(90), "Extreme Greed")
    
    def test_send_email_report(self):
        """Test email report sending"""
        mock_report = {
            'timestamp': datetime.now(),
            'portfolio_summary': {'total_value': 100000},
            'performance_metrics': {'best_performers': []},
            'risk_analysis': {'risk_warnings': []},
            'rebalancing_recommendations': {'priority_actions': []},
            'news_highlights': {},
            'watchlist_analysis': {},
            'market_outlook': {},
            'data_warnings': []
        }
        
        result = self.portfolio.send_email_report(mock_report)
        
        # Should return True for successful send (mocked)
        self.assertTrue(result)
        
        # Check that email was "sent"
        self.assertEqual(len(self.portfolio.email_sender.sent_emails), 1)

if __name__ == '__main__':
    unittest.main()
