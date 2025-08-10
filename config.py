import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()



"""
Portfolio configuration and settings
Update this file with your actual portfolio holdings and preferences
"""

# Default portfolio holdings (sanitized placeholder sample)
# Replace by providing portfolio_config.json (see README) or editing these examples.
PORTFOLIO_HOLDINGS = {
    'AAPL': {'shares': 50, 'target_weight': 0.15, 'category': 'consumer_tech', 'current_price': None},
    'MSFT': {'shares': 30, 'target_weight': 0.15, 'category': 'tech', 'current_price': None},
    'SPY': {'shares': 40, 'target_weight': 0.25, 'category': 'etf', 'current_price': None},
    'BTC-USD': {'shares': 0.5, 'target_weight': 0.05, 'category': 'crypto', 'current_price': None},
    'NVDA': {'shares': 10, 'target_weight': 0.10, 'category': 'semiconductors', 'current_price': None},
}

# Default watchlist (sanitized examples)
WATCHLIST = {
    'TSLA': {'entry_target': 200, 'target_weight': 0.05, 'category': 'EV'},
    'AMD': {'entry_target': 95, 'target_weight': 0.04, 'category': 'semiconductors'},
    'GOOGL': {'entry_target': 150, 'target_weight': 0.05, 'category': 'tech'},
}

# Rebalancing thresholds and rules
REBALANCING_CONFIG = {
    'max_position_size': 0.10,  # Maximum 10% in any single position
    'min_rebalance_threshold': 0.02,  # Rebalance if allocation drifts >2%
    'max_crypto_allocation': 0.10,  # Maximum 10% total in crypto
    'cash_buffer_target': 0.05,  # Maintain 5% cash buffer
    'volatility_threshold': 0.60,  # Flag high volatility positions >60%
}

def _parse_email_list(raw: str):
    if not raw:
        return []
    # Allow comma or semicolon separated
    parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
    return parts

_to_raw = os.getenv('TO_EMAILS') or os.getenv('TO_EMAIL')
_cc_raw = os.getenv('CC_EMAILS') or os.getenv('CC_EMAIL')
_bcc_raw = os.getenv('BCC_EMAILS') or os.getenv('BCC_EMAIL')

EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'from_email': os.getenv('FROM_EMAIL'),
    'to_email': os.getenv('TO_EMAIL'),  # backward compat single
    'to_emails': _parse_email_list(_to_raw),
    'cc_emails': _parse_email_list(_cc_raw),
    'bcc_emails': _parse_email_list(_bcc_raw),
    'username': os.getenv('EMAIL_USERNAME'),
    'password': os.getenv('EMAIL_PASSWORD'),
}

# Data source configurations
DATA_CONFIG = {
    'yfinance_enabled': True,
    'news_sources': ['finviz', 'yahoo_finance', 'google_news'],
    'risk_free_rate': 0.045,  # Current risk-free rate for Sharpe ratio calculations
    'trading_days_per_year': 252,
    'price_history_period': '1y',  # Period for risk calculations
    'price_update_frequency': 'daily',  # How often to update prices
    'sentiment': {
        'global_positive_tilt_budget': 0.02,   # Max total +2% weight redistributed on strong positive sentiment
        'global_negative_tilt_budget': -0.02,  # Max total -2% on strong negative sentiment
        'symbol_positive_threshold': 0.25,     # Score above to consider increasing
        'symbol_negative_threshold': -0.25,    # Score below to consider decreasing
        'max_symbol_delta': 0.01,              # Cap absolute adjustment per symbol (1%)
        'min_articles_for_action': 3,          # Require minimum news flow
        'volatility_guardrail': 0.80,          # Skip positive tilt if asset vol above this
        'drawdown_guardrail': -0.30,           # Skip positive tilt if deep drawdown
        'neutral_band': 0.15,                  # |score| under this considered neutral
        'decay_exponent': 1.2                  # Nonlinear scaling exponent for extreme scores
    },
    'stress_scenarios': [
        {
            'name': 'Equity -15% / Semi -20% Shock',
            'category_shocks': {
                'semiconductors': -0.20,
                'tech': -0.16,
                'consumer_tech': -0.16,
            },
            'default_shock': -0.15
        },
        {
            'name': 'Crypto Crash -40%',
            'category_shocks': {
                'crypto': -0.40
            },
            'default_shock': -0.05
        },
        {
            'name': 'AI Software Setback',
            'category_shocks': {
                'ai_software': -0.25,
                'ad_tech': -0.18
            },
            'default_shock': -0.08
        },
        {
            'name': 'Flight Demand Slump',
            'category_shocks': {
                'airlines': -0.22
            },
            'default_shock': -0.04
        }
    ],
    'factor_proxies': {
        # Basic multi-factor approximations using ETFs (can be expanded)
        'MKT': 'SPY',      # Market
        'SIZE': 'IWM',     # Small-cap proxy
        'VALUE': 'IVE',    # Value proxy
        'MOM': 'MTUM'      # Momentum proxy
    }
}

def _load_external_portfolio_overrides():
    """Allow users who clone the repo to supply their own holdings/watchlist.

    Looks for a JSON file specified by environment variable PORTFOLIO_CONFIG_FILE
    (defaults to 'portfolio_config.json' in the project root). If present, it may
    contain keys: holdings, watchlist, rebalancing. Only provided keys override
    the in-repo defaults. This keeps the repo usable out-of-the-box while letting
    each user keep their private position sizes untracked (the file should be
    git-ignored by default when named portfolio_config.json).
    """
    config_path = os.getenv('PORTFOLIO_CONFIG_FILE', 'portfolio_config.json')
    p = Path(config_path)
    if not p.exists():
        return
    try:
        with p.open('r', encoding='utf-8') as f:
            external = json.load(f)
        # Basic validation and override
        if isinstance(external, dict):
            holdings = external.get('holdings')
            if isinstance(holdings, dict) and holdings:
                PORTFOLIO_HOLDINGS.clear()
                PORTFOLIO_HOLDINGS.update(holdings)
            watch = external.get('watchlist')
            if isinstance(watch, dict) and watch:
                WATCHLIST.clear()
                WATCHLIST.update(watch)
            reb = external.get('rebalancing')
            if isinstance(reb, dict) and reb:
                REBALANCING_CONFIG.update(reb)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Warning: Failed to load external portfolio config '{p}': {e}")


_load_external_portfolio_overrides()

# Portfolio configuration container (after any overrides)
PORTFOLIO_CONFIG = {
    'holdings': PORTFOLIO_HOLDINGS,
    'watchlist': WATCHLIST,
    'rebalancing': REBALANCING_CONFIG,
    'email': EMAIL_CONFIG,
    'data': DATA_CONFIG,
}

# Add a function to update prices dynamically
def update_portfolio_prices():
    """
    Updates current prices for all holdings using yfinance
    This should be called before generating risk visualizations
    """
    import yfinance as yf
    
    tickers = list(PORTFOLIO_HOLDINGS.keys())
    data = yf.download(tickers, period="1d", interval="1d")
    
    if len(tickers) == 1:
        # Handle single ticker case
        current_price = data['Close'].iloc[-1]
        PORTFOLIO_HOLDINGS[tickers[0]]['current_price'] = current_price
    else:
        # Handle multiple tickers
        for ticker in tickers:
            try:
                current_price = data['Close'][ticker].iloc[-1]
                PORTFOLIO_HOLDINGS[ticker]['current_price'] = current_price
            except (KeyError, IndexError):
                print(f"Warning: Could not fetch price for {ticker}")
                PORTFOLIO_HOLDINGS[ticker]['current_price'] = None
    
    return PORTFOLIO_HOLDINGS

