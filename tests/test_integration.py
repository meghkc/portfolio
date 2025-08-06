"""
Integration tests for the complete portfolio automation system
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_manager import PortfolioManager
from data_fetcher import DataFetcher
from risk_analyzer import RiskAnalyzer
from news_scraper import NewsAnalyzer
from email_sender import EmailSender
from tests.conftest import (
    TEST_PORTFOLIO_CONFIG, 
    generate_mock_price_data, 
    generate_mock_news_data,
    assert_portfolio_structure,
    assert_risk_metrics_valid
)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.config = TEST_PORTFOLIO_CONFIG.copy()
        
        # Create realistic mock data
        self.symbols = list(self.config['holdings'].keys())
        self.mock_price_data = generate_mock_price_data(self.symbols)
        self.mock_news_data = generate_mock_news_data(self.symbols)
        self.mock_current_prices = {
            symbol: data['Close'].iloc[-1] for symbol, data in self.mock_price_data.items()
        }
    
    @patch('yfinance.download')
    @patch('yfinance.Ticker')
    def test_data_fetcher_integration(self, mock_ticker, mock_download):
        """Test data fetcher with realistic data"""
        # Mock yfinance responses
        mock_download.return_value = self.mock_price_data['AAPL']
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.mock_price_data['AAPL']
        mock_ticker_instance.news = [
            {
                'title': 'AAPL reports strong earnings',
                'link': 'https://example.com/news/1',
                'providerPublishTime': 1640995200  # Unix timestamp
            }
        ]
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data fetcher
        data_fetcher = DataFetcher(self.config)
        
        # Test price fetching
        prices = data_fetcher.fetch_current_prices(['AAPL'])
        self.assertIn('AAPL', prices)
        self.assertIsInstance(prices['AAPL'], (int, float))
        
        # Test historical data
        historical_data = data_fetcher.fetch_historical_data(['AAPL'])
        self.assertIn('AAPL', historical_data)
        self.assertIsInstance(historical_data['AAPL'], pd.DataFrame)
    
    def test_risk_analyzer_integration(self):
        """Test risk analyzer with realistic portfolio data"""
        risk_analyzer = RiskAnalyzer(self.config)
        
        # Create portfolio data
        returns_data = {
            symbol: data['Close'].pct_change().dropna() 
            for symbol, data in self.mock_price_data.items()
        }
        
        weights = {symbol: info['target_weight'] for symbol, info in self.config['holdings'].items()}
        
        portfolio_data = {
            'weights': weights,
            'returns_data': returns_data,
            'historical_data': self.mock_price_data
        }
        
        # Generate risk report
        risk_report = risk_analyzer.generate_risk_report(portfolio_data)
        
        # Validate structure
        self.assertIn('portfolio_metrics', risk_report)
        self.assertIn('individual_risks', risk_report)
        self.assertIn('risk_warnings', risk_report)
        
        # Validate metrics
        portfolio_metrics = risk_report['portfolio_metrics']
        if 'volatility' in portfolio_metrics:
            assert_risk_metrics_valid(portfolio_metrics)
    
    def test_news_analyzer_integration(self):
        """Test news analyzer with mock news data"""
        news_analyzer = NewsAnalyzer(self.config)
        
        # Analyze mock news
        analysis = news_analyzer.analyze_news_sentiment(self.mock_news_data)
        
        # Validate structure
        self.assertIn('symbol_sentiments', analysis)
        self.assertIn('market_themes', analysis)
        self.assertIn('catalyst_alerts', analysis)
        
        # Check symbol-specific analysis
        for symbol in self.symbols:
            if symbol in analysis['symbol_sentiments']:
                symbol_analysis = analysis['symbol_sentiments'][symbol]
                self.assertIn('average_sentiment', symbol_analysis)
                self.assertIn('sentiment_trend', symbol_analysis)
                self.assertIn('articles', symbol_analysis)
    
    def test_email_sender_integration(self):
        """Test email sender without actually sending emails"""
        email_sender = EmailSender(self.config['email'])
        
        # Test email content formatting
        mock_report_content = """
        # Portfolio Report
        ## Summary
        Total Value: $100,000
        
        ## Performance
        Best Performer: AAPL (+5.2%)
        """
        
        # Mock SMTP to avoid actual email sending
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Test sending report
            result = email_sender.send_portfolio_report(mock_report_content)
            self.assertTrue(result)
            
            # Verify SMTP was called
            mock_smtp.assert_called_once()
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
    
    @patch('portfolio_manager.update_portfolio_prices')
    def test_full_system_integration(self, mock_update_prices):
        """Test the complete portfolio management workflow"""
        # Mock external dependencies
        mock_update_prices.return_value = self.config['holdings']
        
        with patch('data_fetcher.yf.download') as mock_download, \
             patch('data_fetcher.yf.Ticker') as mock_ticker, \
             patch('smtplib.SMTP') as mock_smtp:
            
            # Set up mocks
            mock_download.return_value = self.mock_price_data
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = self.mock_price_data['AAPL']
            mock_ticker_instance.info = {'regularMarketPrice': 150.0}
            mock_ticker_instance.news = []
            mock_ticker.return_value = mock_ticker_instance
            
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Initialize portfolio manager
            portfolio = PortfolioManager(self.config)
            
            # Generate weekly report
            report = portfolio.generate_weekly_report()
            
            # Validate report structure
            assert_portfolio_structure(report)
            
            # Test specific report sections
            self.assertIn('total_value', report['portfolio_summary'])
            self.assertGreater(report['portfolio_summary']['total_value'], 0)
            
            # Send email report
            email_result = portfolio.send_email_report(report)
            self.assertTrue(email_result)
    
    def test_portfolio_rebalancing_workflow(self):
        """Test the portfolio rebalancing workflow"""
        portfolio = PortfolioManager(self.config)
        
        # Mock scenario: AAPL is over-allocated
        current_prices = self.mock_current_prices
        portfolio_metrics = {
            'current_allocations': {
                'AAPL': 0.30,  # Target is 0.20, so over-allocated
                'GOOGL': 0.10,  # Target is 0.15, so under-allocated
                'MSFT': 0.15,
                'SPY': 0.30,
                'BTC-USD': 0.10,
                'NVDA': 0.05
            },
            'allocation_drift': {
                'AAPL': 0.10,   # +10% drift (over)
                'GOOGL': -0.05,  # -5% drift (under)
                'MSFT': 0.0,
                'SPY': 0.0,
                'BTC-USD': 0.0,
                'NVDA': -0.05
            }
        }
        
        risk_analysis = {
            'individual_risks': {
                'AAPL': {'volatility': 0.25, 'current_drawdown': -0.05},
                'GOOGL': {'volatility': 0.30, 'current_drawdown': -0.10},
                'BTC-USD': {'volatility': 0.80, 'current_drawdown': -0.20}  # High volatility
            }
        }
        
        # Test rebalancing recommendations
        recommendations = portfolio._generate_rebalancing_recommendations(
            current_prices, portfolio_metrics, risk_analysis
        )
        
        # Validate recommendations
        self.assertIn('trim_positions', recommendations)
        self.assertIn('add_positions', recommendations)
        self.assertIn('risk_adjustments', recommendations)
        
        # Should recommend trimming AAPL
        trim_symbols = [pos['symbol'] for pos in recommendations['trim_positions']]
        self.assertIn('AAPL', trim_symbols)
        
        # Should recommend adding GOOGL
        add_symbols = [pos['symbol'] for pos in recommendations['add_positions']]
        self.assertIn('GOOGL', add_symbols)
        
        # Should flag BTC-USD for high volatility
        risk_symbols = [adj['symbol'] for adj in recommendations['risk_adjustments']]
        self.assertIn('BTC-USD', risk_symbols)
    
    def test_watchlist_monitoring_workflow(self):
        """Test watchlist monitoring and opportunity identification"""
        portfolio = PortfolioManager(self.config)
        
        # Mock prices: TSLA below target, AMD above target
        current_prices = {
            'TSLA': 180.0,  # Target: 200, so good buy opportunity
            'AMD': 120.0    # Target: 100, so overpriced
        }
        
        opportunities = portfolio._analyze_watchlist_opportunities(current_prices)
        
        # Validate structure
        self.assertIn('ready_to_buy', opportunities)
        self.assertIn('approaching_targets', opportunities)
        self.assertIn('overpriced', opportunities)
        
        # TSLA should be ready to buy
        ready_symbols = [opp['symbol'] for opp in opportunities['ready_to_buy']]
        self.assertIn('TSLA', ready_symbols)
        
        # Find TSLA opportunity details
        tsla_opp = next((opp for opp in opportunities['ready_to_buy'] if opp['symbol'] == 'TSLA'), None)
        self.assertIsNotNone(tsla_opp)
        self.assertEqual(tsla_opp['current_price'], 180.0)
        self.assertEqual(tsla_opp['entry_target'], 200.0)
    
    def test_risk_visualization_integration(self):
        """Test risk visualization component"""
        with patch('risk_visualization.plt') as mock_plt:
            from risk_visualization import plot_portfolio_risk
            
            # Create mock portfolio returns
            portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            
            # Test chart generation
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                chart_path = plot_portfolio_risk(portfolio_returns, tmp_file.name)
                
                # Cleanup
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            
            # Verify matplotlib was called
            mock_plt.figure.assert_called_once()
            mock_plt.hist.assert_called_once()
            mock_plt.axvline.assert_called_once()
    
    def test_error_handling_integration(self):
        """Test system behavior under error conditions"""
        portfolio = PortfolioManager(self.config)
        
        # Test with invalid data
        with patch.object(portfolio.data_fetcher, 'fetch_current_prices') as mock_prices:
            # Mock API failure
            mock_prices.side_effect = Exception("API Error")
            
            # Should not crash, should handle gracefully
            try:
                report = portfolio.generate_weekly_report()
                # Report should still be generated with warnings
                self.assertIn('data_warnings', report)
            except Exception as e:
                self.fail(f"System should handle API errors gracefully: {e}")

if __name__ == '__main__':
    unittest.main()
