# Contributing to Portfolio Automation System

Thank you for your interest in contributing to the Portfolio Automation System! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of financial markets and portfolio management

### Development Setup

1. **Fork and clone the repository**
    ```bash
    git clone https://github.com/meghkc/portfolio.git
    cd portfolio
    ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt --constraint constraints.txt
    pip install .[dev]  # Development & tooling extras
    ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with test email credentials
   ```

## 📝 Code Style and Standards

### Python Style Guidelines
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for all classes and functions
- Maximum line length: 100 characters

### Code Structure
```python
"""
Module description
Brief explanation of the module's purpose
"""

import standard_library
import third_party_packages
import local_modules

class ExampleClass:
    """
    Class description
    
    Args:
        param: Parameter description
    """
    
    def __init__(self, param):
        self.param = param
    
    def example_method(self) -> ReturnType:
        """
        Method description
        
        Returns:
            Description of return value
        """
        pass
```

### Testing
- Write unit tests for new features
- Ensure existing tests pass
- Include integration tests for complex workflows
- Test with different portfolio configurations

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional broker integrations (TD Ameritrade, E*TRADE)
- [ ] Machine learning price prediction models
- [ ] Real-time data streaming
- [ ] Web dashboard interface
- [ ] Options analysis capabilities

### Medium Priority
- [ ] Alternative data sources (Alpha Vantage, Finnhub)
- [ ] Cryptocurrency exchange integrations
- [ ] Tax-loss harvesting algorithms
- [ ] Performance benchmarking tools
- [ ] Risk scenario modeling

### Documentation
- [ ] API documentation improvements
- [ ] Tutorial videos
- [ ] Example portfolio configurations
- [ ] Deployment guides
- [ ] FAQ section

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions (`pip freeze`)

2. **Bug Description**
   - Clear description of the issue
   - Expected vs actual behavior
   - Steps to reproduce

3. **Error Logs**
   - Full error traceback
   - Relevant log entries
   - Portfolio configuration (sanitized)

4. **Additional Context**
   - Screenshots if applicable
   - Data samples (anonymized)

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and business value
3. **Provide examples** of how it would work
4. **Consider implementation complexity**
5. **Discuss with maintainers** before large changes

## 🔄 Pull Request Process

### Before Submitting
1. Create a feature branch from `main`
2. Write tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Follow the code style guidelines

### Pull Request Checklist
- [ ] Descriptive title and summary
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or clearly marked)
- [ ] Follows existing code patterns
- [ ] Includes example usage if applicable
 - [ ] constraints.txt updated (only if dependency changes) via pip-compile

### Review Process
1. Automated tests must pass
2. Code review by maintainers
3. Documentation review
4. Final approval and merge

## 🧪 Testing Guidelines

### Unit Tests
```python
import unittest
from portfolio_manager import PortfolioManager

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'holdings': {'AAPL': {'shares': 100, 'target_weight': 0.1}},
            # ... other config
        }
        self.portfolio = PortfolioManager(self.config)
    
    def test_portfolio_initialization(self):
        self.assertIsNotNone(self.portfolio)
        self.assertEqual(len(self.portfolio.holdings), 1)
```

### Integration Tests
```python
def test_full_portfolio_analysis():
    """Test complete portfolio analysis workflow"""
    portfolio = PortfolioManager(test_config)
    report = portfolio.generate_weekly_report()
    
    assert 'portfolio_summary' in report
    assert 'risk_analysis' in report
    assert report['portfolio_summary']['total_value'] > 0
```

## 🔐 Security Guidelines

### Data Handling
- Never log sensitive data (passwords, API keys)
- Sanitize portfolio data in examples
- Use environment variables for credentials
 - Keep `.env` out of version control (already gitignored)

### Repository Hygiene
- Do not commit generated artifacts (logs, charts, CSV histories) unless explicitly required for docs.
- Run `pytest -q` before pushing.
- Regenerate pins: `pip-compile --output-file constraints.txt pyproject.toml` in a dedicated chore PR.
- Security scans: `bandit -r .` and `safety check --full-report` locally prior to release tagging.
- Validate all external data inputs

### API Security
- Implement rate limiting for external APIs
- Handle API failures gracefully
- Use secure connections (HTTPS) for all requests
- Validate API response data

## 📚 Documentation Standards

### Code Documentation
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculate the Sharpe ratio for a returns series.
    
    The Sharpe ratio measures risk-adjusted returns by comparing the excess
    return of an investment to its volatility.
    
    Args:
        returns: Time series of investment returns
        risk_free_rate: Annual risk-free rate (e.g., 0.03 for 3%)
    
    Returns:
        Sharpe ratio as a float. Higher values indicate better risk-adjusted returns.
    
    Raises:
        ValueError: If returns series is empty or risk_free_rate is negative
    
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns, 0.03)
        >>> print(f"Sharpe ratio: {sharpe:.2f}")
    """
```

### README Updates
- Keep installation instructions current
- Update feature lists for new capabilities
- Include new configuration options
- Add examples for new functionality

## 🤝 Community Guidelines

### Communication
- Be respectful and professional
- Provide constructive feedback
- Help other contributors
- Ask questions when unclear

### Issue Discussion
- Stay on topic
- Provide helpful information
- Consider different use cases
- Be patient with responses

## 📈 Performance Considerations

### Code Optimization
- Profile code for bottlenecks
- Use vectorized operations with pandas/numpy
- Implement caching for expensive calculations
- Consider async operations for I/O

### Memory Usage
- Clean up large DataFrames when done
- Use generators for large datasets
- Monitor memory usage in long-running processes
- Optimize data structures

## 🔧 Development Tools

### Recommended Tools
- **IDE**: VS Code, PyCharm
- **Linting**: flake8, pylint
- **Formatting**: black
- **Testing**: pytest
- **Type Checking**: mypy

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## 📞 Getting Help

- **Documentation**: Check the README and docs folder
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## 🏆 Recognition

Contributors will be recognized in:
- README contributors section
- Release notes for significant contributions
- GitHub contributor graphs
- Project documentation

Thank you for contributing to the Portfolio Automation System! Your efforts help make portfolio management more accessible and intelligent for everyone.

---

**Happy coding! 🐍📈**
