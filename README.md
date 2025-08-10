# 📈 Automated Portfolio Management System

An intelligent, automated portfolio management system that provides comprehensive analysis, risk assessment, and rebalancing recommendations for your investment portfolio. The system runs scheduled weekly reports and includes advanced features like news sentiment analysis, risk visualization, and email notifications.

<p align="center">
    <!-- Badges: replace placeholder links when public -->
    <img src="https://img.shields.io/badge/status-beta-blue" alt="Project Status" />
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
    <img src="https://img.shields.io/badge/ci-github_actions-lightgrey" alt="CI" />
    <img src="https://img.shields.io/badge/python-3.11%7C3.12%7C3.13-yellow" alt="Python Versions" />
</p>

## 🌟 Features

### Core Functionality
- **Automated Portfolio Analysis**: Weekly comprehensive portfolio reviews with performance metrics
- **Risk Management**: Advanced risk analysis including VaR, Sharpe ratio, correlation analysis, and drawdown calculations
- **Rebalancing Recommendations**: Intelligent suggestions based on target allocations and risk thresholds
- **News Sentiment Analysis**: Real-time news monitoring and sentiment scoring for portfolio holdings
- **Email Reporting**: Automated HTML email reports with charts and actionable insights
- **Watchlist Management**: Monitor potential investment opportunities with entry targets

### Advanced Analytics (Extended)
- **Historical & Parametric VaR (Gaussian + Cornish-Fisher)** with skew/kurtosis adjustment
- **Monte Carlo VaR & CVaR** (multivariate normal simulations)
- **Risk Contributions** (marginal & % total) & concentration metrics (HHI, Gini, effective holdings)
- **Portfolio Volatility (covariance matrix)** & correlation diagnostics
- **Sharpe / Sortino / Drawdown** analytics
- **Rolling Metrics** (20d rolling volatility & return per asset)
- **Risk Parity Indicative Weights** (iterative approximation)
- **Stress Scenarios** (configurable category/default shocks)
- **Multi-Factor Betas** (ETF proxy factors with alpha & R²)
- **Performance Attribution** (category-level contribution)
- **Sentiment-Driven Tilts** and negative sentiment risk flag
- **Composite Watchlist Scoring** (distance-to-target, momentum, valuation band, sentiment)
- **Trend Visualization** (composite score history chart base64-embedded)

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
    git clone https://github.com/meghkc/portfolio.git
    cd portfolio
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt --constraint constraints.txt
    ```
    (Using the provided `constraints.txt` improves reproducibility. Update pins periodically after security review.)

    To refresh dependency pins (after installing dev extras which include pip-tools):
    ```bash
    pip install .[dev]
    pip-compile --generate-hashes --output-file=constraints.txt pyproject.toml
    ```

    PEP 621 metadata is defined in `pyproject.toml`; legacy `setup.py` has been removed (PEP 517 build).

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
- Historical / Parametric (Gaussian & Cornish-Fisher) / Monte Carlo VaR & CVaR
- Sharpe and Sortino ratios
- Maximum drawdown analysis
- Risk contributions & concentration diagnostics (HHI, Gini, effective holdings)
- Rolling volatility / rolling returns
- Stress scenario impacts with portfolio % effect and P/L
- Multi-factor betas (factor exposures & alpha, R²)
- Indicative risk parity weights

### Rebalancing Recommendations
- **Trim Positions**: Over-allocated holdings
- **Add Positions**: Under-allocated holdings  
- **New Entries**: Watchlist opportunities
- **Risk Adjustments**: High volatility warnings

### News Sentiment & Watchlist Intelligence
- Symbol-specific sentiment scores & article counts
- Portfolio-weighted sentiment score & NEGATIVE_SENTIMENT_RISK flag
- Market themes and catalyst alerts
- Price-moving news identification
- Sentiment-driven rebalancing tilt adjustments (budgeted & bounded)
- Composite watchlist scoring (0–100) + CSV persistence (`watchlist_scores_history.csv`)
- Embedded watchlist composite trend chart

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

## 🛡️ Risk Management & Intelligence Features

### Automated Alerts
- High volatility warnings (>60%)
- Large drawdown alerts (>30%)
- Concentration risk notifications
- News-driven catalyst alerts

### Portfolio Protection & Optimization
- Maximum position size limits
- Crypto allocation caps
- Volatility threshold monitoring
- Correlation risk assessment
- Sentiment tilt guardrails & budget constraints
- Indicative risk parity targets for optional allocation guidance

## 🧪 Testing

Run individual components for testing:

```bash
# Test email configuration
python -c "from email_sender import EmailSender; from config import EMAIL_CONFIG; EmailSender(EMAIL_CONFIG).test_email_connection()"

# Test data fetching
python -c "from data_fetcher import DataFetcher; from config import PORTFOLIO_CONFIG; print(DataFetcher(PORTFOLIO_CONFIG).fetch_current_prices(['AAPL']))"

# Test news analysis (integration tests live under tests/)
pytest -k news --maxfail=1 -q
```

### Continuous Integration
GitHub Actions workflow (`.github/workflows/ci.yml`) executes on pushes & PRs:
- Constraints drift check (pip-compile vs committed `constraints.txt`)
- Install & test (pytest)
- Bandit static analysis
- Safety vulnerability scan
- Artifact upload (logs, trend chart)

Manual pin refresh workflow available via dispatch. Local Windows helper:
```powershell
./scripts/update_pins.ps1
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
- **Never commit `.env`** or credentials
- Use app passwords (Gmail) / token-based auth
- Environment variables or secrets manager in production
- Pin dependencies via `constraints.txt` (refresh with `pip-compile`)
- Run security scanners: `bandit -r .`, `safety check --full-report`
- Optional SBOM generation: `cyclonedx-py --format json -o sbom.json`
- Minimal logging of sensitive values (avoid raw holdings in debug logs)
- Validate external data (empty frames, NaNs) before risk computations
- Graceful fallbacks for chart generation and network failures

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
- [ ] Enhanced factor library (macro & style expansion)
- [ ] Historical regime stress replay & scenario generator
- [ ] Optimization engine (risk budgeting / convex allocation)
- [ ] Explainable AI sentiment classifier (beyond polarity)
- [ ] Advanced attribution (Brinson-Fachler multi-level)

---

**⚡ Built with Python | 📈 Powered by yfinance | 🤖 Automated Intelligence**
