# Portfolio Automation System - Test Suite

This directory contains comprehensive tests for the Portfolio Automation System.

## Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_portfolio_manager.py # Core portfolio management tests
├── test_risk_analyzer.py     # Risk analysis tests  
├── test_integration.py       # Integration tests
└── README.md               # This file
```

## Running Tests

### Prerequisites
Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Run All Tests
```bash
# From project root directory
python -m pytest tests/

# With coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Files
```bash
# Portfolio manager tests
python -m pytest tests/test_portfolio_manager.py -v

# Risk analyzer tests  
python -m pytest tests/test_risk_analyzer.py -v

# Integration tests
python -m pytest tests/test_integration.py -v
```

### Run Individual Tests
```bash
# Specific test method
python -m pytest tests/test_portfolio_manager.py::TestPortfolioManager::test_calculate_portfolio_metrics -v
```

## Test Categories

### Unit Tests

#### Portfolio Manager Tests (`test_portfolio_manager.py`)
- Portfolio initialization and configuration
- Portfolio metrics calculation (total value, allocations, drift)
- Performance analysis (best/worst performers, volatility)
- Rebalancing recommendations logic
- Watchlist opportunity identification
- Email report generation and sending
- Fear & Greed Index interpretation

#### Risk Analyzer Tests (`test_risk_analyzer.py`)
- Portfolio volatility calculation using covariance matrix
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Sharpe and Sortino ratio computations
- Maximum drawdown analysis
- Beta calculation vs benchmark
- Concentration risk assessment (HHI, effective holdings)
- Risk warning generation

### Integration Tests (`test_integration.py`)
- Complete portfolio analysis workflow
- Data fetcher integration with yfinance mocking
- News analyzer sentiment analysis pipeline
- Email sender with SMTP mocking
- Portfolio rebalancing workflow
- Watchlist monitoring workflow
- Risk visualization integration
- Error handling and graceful degradation

## Test Data and Mocking

### Mock Data Generation
- **Price Data**: Realistic stock price movements with volatility
- **News Data**: Sample financial news articles with varied sentiment
- **Portfolio Configuration**: Test portfolio with 6 holdings and 2 watchlist items

### External API Mocking
- **yfinance**: Mock stock price and historical data
- **News Sources**: Mock Finviz and Yahoo Finance news
- **SMTP**: Mock email sending without actual transmission
- **Crypto APIs**: Mock Fear & Greed Index data

## Test Configuration

### Test Portfolio (`conftest.py`)
```python
TEST_PORTFOLIO_CONFIG = {
    'holdings': {
        'AAPL': {'shares': 100, 'target_weight': 0.20, 'category': 'tech'},
        'GOOGL': {'shares': 50, 'target_weight': 0.15, 'category': 'tech'},
        # ... more holdings
    },
    'watchlist': {
        'TSLA': {'entry_target': 200, 'target_weight': 0.05},
        # ... more watchlist items
    }
}
```

### Mock Classes
- `MockDataFetcher`: Simulates market data API responses
- `MockEmailSender`: Tracks email sending without actual transmission
- `generate_mock_price_data()`: Creates realistic historical price data
- `generate_mock_news_data()`: Creates sample news articles

## Test Utilities

### Assertion Helpers
- `assert_portfolio_structure()`: Validates report structure
- `assert_risk_metrics_valid()`: Validates risk metric ranges
- `setup_test_environment()`: Sets up test environment variables

### Data Validation
Tests ensure:
- Portfolio allocations sum to 1.0
- Risk metrics are within reasonable ranges
- Volatility calculations are positive
- Correlation matrices are properly formed
- Email content is properly formatted

## Coverage Goals

Target test coverage by component:
- Portfolio Manager: >90%
- Risk Analyzer: >90%  
- Data Fetcher: >80%
- News Analyzer: >80%
- Email Sender: >85%
- Overall System: >85%

## Performance Tests

### Load Testing
- Portfolio analysis with 100+ holdings
- Risk calculations with 5+ years of data
- News analysis with 50+ articles per symbol

### Memory Testing
- Large DataFrame handling
- Historical data processing
- Correlation matrix calculations

## Security Tests

### Data Validation
- Input sanitization for external data
- Email content safety
- Configuration validation

### Credential Handling
- Environment variable protection
- Mock credential testing
- Secure configuration validation

## Continuous Integration

### GitHub Actions (Future)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
```

## Adding New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality_description>`

### Test Structure Template
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_specific_functionality(self):
        """Test description"""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        self.assertEqual(expected, result)
    
    def tearDown(self):
        """Clean up after tests"""
        pass
```

## Debugging Tests

### Verbose Output
```bash
# Show detailed test output
python -m pytest tests/ -v -s

# Show print statements
python -m pytest tests/ -s --capture=no
```

### Test Debugging
```python
import pdb; pdb.set_trace()  # Add breakpoint in test
```

### Mock Debugging
```python
# Check mock calls
mock_function.assert_called_with(expected_args)
print(mock_function.call_args_list)
```

## Known Test Limitations

1. **External Dependencies**: Tests use mocked data, may not catch API changes
2. **Timing**: Some financial calculations depend on market timing
3. **Randomness**: Some tests use random data generation with fixed seeds
4. **Network**: No actual network calls in tests (all mocked)

## Best Practices

1. **Isolation**: Each test is independent and can run alone
2. **Repeatability**: Tests produce consistent results
3. **Fast Execution**: Tests complete quickly (<30 seconds total)
4. **Clear Assertions**: Test failures provide clear error messages
5. **Realistic Data**: Mock data resembles real market conditions

---

**For questions about testing, see the main project documentation or open an issue.**
