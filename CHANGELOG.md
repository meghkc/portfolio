# Changelog

All notable changes to the Portfolio Automation System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- Web dashboard interface
- Real-time alerts via Slack/Discord
- Options analysis and strategies
- Backtesting framework
- Tax-loss harvesting automation
- Machine learning price predictions

## [1.0.0] - 2025-01-XX

### Added
- **Core Portfolio Management**
  - Automated weekly portfolio analysis and reporting
  - Support for stocks, ETFs, and cryptocurrencies
  - Target allocation tracking and drift analysis
  - Performance metrics calculation (YTD returns, volatility)

- **Advanced Risk Analysis**
  - Portfolio volatility calculation using covariance matrix
  - Value at Risk (VaR) and Conditional VaR (CVaR) calculations
  - Sharpe and Sortino ratio computations
  - Maximum drawdown analysis with recovery factors
  - Beta calculation vs benchmark (SPY)
  - Concentration risk assessment using Herfindahl-Hirschman Index

- **Rebalancing Intelligence**
  - Automated rebalancing recommendations
  - Position size limit enforcement
  - Allocation drift monitoring
  - Risk-based adjustment suggestions
  - Watchlist opportunity identification

- **News Sentiment Analysis**
  - Real-time news monitoring for portfolio holdings
  - Sentiment scoring using TextBlob
  - Market theme identification
  - Catalyst alert system
  - Multiple news source integration (Finviz, Yahoo Finance)

- **Email Reporting System**
  - HTML-formatted weekly reports
  - Risk visualization charts
  - Attachment support for charts and CSV exports
  - Error notification system
  - Email configuration testing

- **Data Sources and APIs**
  - Yahoo Finance integration via yfinance
  - Crypto Fear & Greed Index
  - Multi-source news aggregation
  - Historical data analysis (1-year lookback)
  - Real-time price updates

- **Risk Visualization**
  - Portfolio returns distribution histograms
  - Value at Risk visualization
  - Risk chart generation and email attachment
  - Matplotlib-based chart creation

- **Configuration Management**
  - Environment variable support via python-dotenv
  - Flexible portfolio holdings configuration
  - Customizable rebalancing rules
  - Email and data source configuration
  - Example configuration templates

- **Automation and Scheduling**
  - Weekly automated reports (Monday 7:30 AM)
  - Schedule library integration
  - Background process support
  - Comprehensive logging system

- **Security Features**
  - Environment variable protection for credentials
  - .gitignore for sensitive files
  - Input validation and error handling
  - Secure email authentication

### Technical Implementation
- **Architecture**: Modular design with separate components
- **Data Handling**: Pandas for data manipulation and analysis
- **Numerical Computing**: NumPy and SciPy for risk calculations
- **Web Scraping**: BeautifulSoup for news data extraction
- **Email**: SMTP with TLS encryption support
- **Logging**: Comprehensive logging with file and console output

### Documentation
- Comprehensive README with setup instructions
- Contributing guidelines
- Security policy
- MIT License
- Environment configuration examples
- Code documentation with docstrings

### Dependencies
- yfinance>=0.2.25 (Market data)
- pandas>=2.0.0 (Data manipulation)
- numpy>=1.24.0 (Numerical computations)
- requests>=2.31.0 (HTTP requests)
- beautifulsoup4>=4.12.0 (Web scraping)
- schedule>=1.2.0 (Task scheduling)
- textblob>=0.17.1 (Sentiment analysis)
- scipy>=1.11.0 (Statistical functions)
- matplotlib>=3.5.0 (Visualization)
- python-dotenv>=1.0.0 (Environment variables)

### Configuration Examples
- Portfolio holdings template with 16 stocks and 2 cryptocurrencies
- Watchlist with 7 potential investments
- Rebalancing rules with risk thresholds
- Email configuration for Gmail SMTP
- Risk analysis parameters

---

## Version History

### Pre-release Development
- Initial proof of concept
- Core data fetching implementation
- Basic portfolio calculations
- Email notification prototype
- Risk analysis framework development

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this changelog and the project.

## Security

See [SECURITY.md](SECURITY.md) for information about reporting security vulnerabilities.

---

*This changelog follows the [Keep a Changelog](https://keepachangelog.com/) format for clear communication of changes.*
