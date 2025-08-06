"""
Test configuration for portfolio automation system
"""

import os
import sys
from unittest.mock import Mock
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration data
TEST_PORTFOLIO_CONFIG = {
    'holdings': {
        'AAPL': {'shares': 100, 'target_weight': 0.20, 'category': 'tech', 'current_price': None},
        'GOOGL': {'shares': 50, 'target_weight': 0.15, 'category': 'tech', 'current_price': None},
        'MSFT': {'shares': 75, 'target_weight': 0.15, 'category': 'tech', 'current_price': None},
        'SPY': {'shares': 200, 'target_weight': 0.30, 'category': 'etf', 'current_price': None},
        'BTC-USD': {'shares': 1, 'target_weight': 0.10, 'category': 'crypto', 'current_price': None},
        'NVDA': {'shares': 25, 'target_weight': 0.10, 'category': 'semiconductors', 'current_price': None},
    },
    'watchlist': {
        'TSLA': {'entry_target': 200, 'target_weight': 0.05, 'category': 'EV'},
        'AMD': {'entry_target': 100, 'target_weight': 0.03, 'category': 'semiconductors'},
    },
    'rebalancing': {
        'max_position_size': 0.25,
        'min_rebalance_threshold': 0.05,
        'max_crypto_allocation': 0.15,
        'cash_buffer_target': 0.05,
        'volatility_threshold': 0.60,
    },
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'from_email': 'test@example.com',
        'to_email': 'test@example.com',
        'username': 'test@example.com',
        'password': 'test_password',
    },
    'data': {
        'yfinance_enabled': True,
        'news_sources': ['finviz', 'yahoo_finance'],
        'risk_free_rate': 0.045,
        'trading_days_per_year': 252,
        'price_history_period': '1y',
        'price_update_frequency': 'daily',
    }
}

# Mock data generators
def generate_mock_price_data(symbols, days=252):
    """Generate mock historical price data for testing"""
    dates = pd.date_range(end='2025-01-01', periods=days, freq='D')
    data = {}
    
    for symbol in symbols:
        # Generate realistic price movements
        base_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data[symbol] = pd.DataFrame({
            'Open': prices,
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
    
    return data

def generate_mock_news_data(symbols):
    """Generate mock news data for testing"""
    news_data = {}
    
    for symbol in symbols:
        news_data[symbol] = [
            {
                'title': f'{symbol} reports strong quarterly earnings',
                'url': f'https://example.com/news/{symbol}/1',
                'published': pd.Timestamp.now() - pd.Timedelta(days=1),
                'source': 'Yahoo Finance'
            },
            {
                'title': f'Analysts upgrade {symbol} price target',
                'url': f'https://example.com/news/{symbol}/2',
                'published': pd.Timestamp.now() - pd.Timedelta(days=2),
                'source': 'Finviz'
            },
            {
                'title': f'{symbol} announces new product launch',
                'url': f'https://example.com/news/{symbol}/3',
                'published': pd.Timestamp.now() - pd.Timedelta(days=3),
                'source': 'Yahoo Finance'
            }
        ]
    
    return news_data

# Mock classes for testing
class MockDataFetcher:
    """Mock data fetcher for testing"""
    
    def __init__(self, config):
        self.config = config
    
    def fetch_current_prices(self, symbols):
        return {symbol: np.random.uniform(50, 500) for symbol in symbols}
    
    def fetch_historical_data(self, symbols, period="1y"):
        return generate_mock_price_data(symbols)
    
    def fetch_news_for_symbols(self, symbols):
        return generate_mock_news_data(symbols)
    
    def fetch_crypto_fear_greed_index(self):
        return np.random.randint(10, 90)

class MockEmailSender:
    """Mock email sender for testing"""
    
    def __init__(self, config):
        self.config = config
        self.sent_emails = []
    
    def send_portfolio_report(self, content, risk_chart_path=None, csv_path=None):
        self.sent_emails.append({
            'content': content,
            'risk_chart_path': risk_chart_path,
            'csv_path': csv_path,
            'timestamp': pd.Timestamp.now()
        })
        return True
    
    def send_alert_email(self, subject, message, priority='normal'):
        self.sent_emails.append({
            'subject': subject,
            'message': message,
            'priority': priority,
            'timestamp': pd.Timestamp.now()
        })
        return True
    
    def test_email_connection(self):
        return True

# Test utilities
def assert_portfolio_structure(portfolio_data):
    """Assert that portfolio data has the expected structure"""
    required_keys = ['timestamp', 'portfolio_summary', 'performance_metrics', 
                    'risk_analysis', 'rebalancing_recommendations']
    
    for key in required_keys:
        assert key in portfolio_data, f"Missing required key: {key}"

def assert_risk_metrics_valid(risk_metrics):
    """Assert that risk metrics are valid"""
    if 'volatility' in risk_metrics:
        assert 0 <= risk_metrics['volatility'] <= 2, "Volatility should be between 0 and 200%"
    
    if 'sharpe_ratio' in risk_metrics:
        assert -5 <= risk_metrics['sharpe_ratio'] <= 5, "Sharpe ratio should be reasonable"

# Environment setup for tests
def setup_test_environment():
    """Set up test environment variables"""
    os.environ['SMTP_SERVER'] = 'smtp.gmail.com'
    os.environ['SMTP_PORT'] = '587'
    os.environ['FROM_EMAIL'] = 'test@example.com'
    os.environ['TO_EMAIL'] = 'test@example.com'
    os.environ['EMAIL_USERNAME'] = 'test@example.com'
    os.environ['EMAIL_PASSWORD'] = 'test_password'

def cleanup_test_environment():
    """Clean up test environment"""
    test_env_vars = ['SMTP_SERVER', 'SMTP_PORT', 'FROM_EMAIL', 'TO_EMAIL', 
                    'EMAIL_USERNAME', 'EMAIL_PASSWORD']
    
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]
