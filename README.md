# 📈 Automated Portfolio Management System

An intelligent, automated portfolio management system that provides comprehensive analysis, risk assessment, and rebalancing recommendations for your investment portfolio. The system runs scheduled weekly reports and includes advanced features like news sentiment analysis, risk visualization, and email notifications.

## 🌟 Features

### Core Functionality
- **Automated Portfolio Analysis**: Weekly comprehensive portfolio reviews with performance metrics
- **Risk Management**: Advanced risk analysis including VaR, Sharpe ratio, correlation analysis, and drawdown calculations
- **Rebalancing Recommendations**: Intelligent suggestions based on target allocations and risk thresholds
- **News Sentiment Analysis**: Real-time news monitoring and sentiment scoring for portfolio holdings
- **Email Reporting**: Automated HTML email reports with charts and actionable insights
- **Watchlist Management**: Monitor potential investment opportunities with entry targets

### Advanced Analytics
- **Portfolio Volatility Calculation**: Using covariance matrix analysis
- **Value at Risk (VaR)** and **Conditional VaR (CVaR)** calculations
- **Sharpe and Sortino Ratio** computations
- **Maximum Drawdown Analysis**
- **Beta Calculation** vs benchmark (SPY)
- **Concentration Risk Assessment** with Herfindahl-Hirschman Index
- **Risk Visualization** with interactive charts

### Data Sources
- **Market Data**: Yahoo Finance (yfinance) for real-time and historical prices
- **News Sources**: Finviz, Yahoo Finance news aggregation
- **Crypto Sentiment**: Fear & Greed Index integration
- **Multiple Asset Classes**: Stocks, ETFs, and cryptocurrencies

## 🏗️ System Architecture

```
├── main.py                 # Entry point and scheduler
├── portfolio_manager.py    # Core portfolio orchestration
├── data_fetcher.py        # Market data and news collection
├── risk_analyzer.py       # Risk metrics and calculations
├── news_scraper.py        # News sentiment analysis
├── email_sender.py        # Email reporting system
├── risk_visualization.py  # Chart generation
└── config.py             # Configuration and portfolio holdings
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Email account with app password (Gmail recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/portfolio-automation.git
   cd portfolio-automation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your email credentials
   ```

4. **Update portfolio holdings**
   ```python
   # Edit config.py with your actual holdings
   PORTFOLIO_HOLDINGS = {
       'AAPL': {'shares': 100, 'target_weight': 0.15, 'category': 'tech'},
       # Add your holdings...
   }
   ```

5. **Test the system**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Portfolio Holdings (`config.py`)
Define your current holdings with shares, target weights, and categories:

```python
PORTFOLIO_HOLDINGS = {
    'AAPL': {
        'shares': 100,
        'target_weight': 0.15,  # 15% target allocation
        'category': 'consumer_tech',
        'current_price': None   # Auto-updated
    },
    # Add more holdings...
}
```

### Email Setup (`.env`)
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
FROM_EMAIL=your_email@gmail.com
TO_EMAIL=recipient@gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### Rebalancing Rules
```python
REBALANCING_CONFIG = {
    'max_position_size': 0.10,           # Max 10% per position
    'min_rebalance_threshold': 0.02,     # Rebalance if >2% drift
    'max_crypto_allocation': 0.10,       # Max 10% in crypto
    'volatility_threshold': 0.60,        # Flag if >60% volatility
}
```

## 📊 Sample Report Output

The system generates comprehensive weekly reports including:

### Portfolio Summary
- Total portfolio value and allocation breakdown
- Performance metrics (YTD returns, volatility)
- Top/worst performers analysis

### Risk Analysis
- Portfolio volatility and correlation matrix
- Value at Risk (5% VaR) and Conditional VaR
- Sharpe and Sortino ratios
- Maximum drawdown analysis

### Rebalancing Recommendations
- **Trim Positions**: Over-allocated holdings
- **Add Positions**: Under-allocated holdings  
- **New Entries**: Watchlist opportunities
- **Risk Adjustments**: High volatility warnings

### News Sentiment
- Symbol-specific sentiment scores
- Market themes and catalyst alerts
- Price-moving news identification

## 🔧 Advanced Usage

### Custom Risk Metrics
```python
# Add custom risk calculations in risk_analyzer.py
def calculate_custom_metric(self, returns_data):
    # Your custom risk logic
    return custom_metric
```

### Additional Data Sources
```python
# Extend data_fetcher.py for new sources
def fetch_alternative_data(self, symbol):
    # Integration with other APIs
    return data
```

### Scheduling Options
```python
# Modify scheduling in main.py
schedule.every().monday.at("07:30").do(run_weekly_portfolio_review)
schedule.every().day.at("09:00").do(run_daily_check)  # Daily checks
```

## 🛡️ Risk Management Features

### Automated Alerts
- High volatility warnings (>60%)
- Large drawdown alerts (>30%)
- Concentration risk notifications
- News-driven catalyst alerts

### Portfolio Protection
- Maximum position size limits
- Crypto allocation caps
- Volatility threshold monitoring
- Correlation risk assessment

## 🧪 Testing

Run individual components for testing:

```bash
# Test email configuration
python -c "from email_sender import EmailSender; from config import EMAIL_CONFIG; EmailSender(EMAIL_CONFIG).test_email_connection()"

# Test data fetching
python -c "from data_fetcher import DataFetcher; from config import PORTFOLIO_CONFIG; print(DataFetcher(PORTFOLIO_CONFIG).fetch_current_prices(['AAPL']))"

# Test news analysis
python test_newsintegration.py
```

## 📝 Dependencies

Core packages required:
- `yfinance>=0.2.25` - Market data
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computations
- `requests>=2.31.0` - Web requests
- `beautifulsoup4>=4.12.0` - Web scraping
- `schedule>=1.2.0` - Task scheduling
- `textblob>=0.17.1` - Sentiment analysis
- `scipy>=1.11.0` - Statistical functions
- `matplotlib>=3.5.0` - Visualization

## 🚨 Important Notes

### Security
- **Never commit `.env` file** with real credentials
- Use app passwords for Gmail (not your regular password)
- Consider using environment variables in production

### Data Accuracy
- Market data depends on yfinance API availability
- News sentiment is indicative, not financial advice
- Always verify recommendations before trading

### Disclaimer
This system is for educational and informational purposes only. Not financial advice. Always consult with qualified financial professionals before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the test files for usage examples

## 🔮 Roadmap

Future enhancements planned:
- [ ] Integration with additional brokers (TD Ameritrade, E*TRADE)
- [ ] Machine learning price prediction models
- [ ] Options analysis and strategies
- [ ] Real-time alerts via Slack/Discord
- [ ] Web dashboard interface
- [ ] Backtesting framework
- [ ] Tax-loss harvesting automation

---

**⚡ Built with Python | 📈 Powered by yfinance | 🤖 Automated Intelligence**
