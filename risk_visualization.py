import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_portfolio_risk(portfolio_returns: pd.Series, output_path: str = 'risk_chart.png') -> str:
    """
    Generate and save a risk visualization (histogram of returns + VaR line).
    Returns the path to the saved chart.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        print("No returns data for risk visualization.")
        return None

    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(portfolio_returns, bins=30, color='#3498db', alpha=0.7, label='Daily Returns')
    var_95 = portfolio_returns.quantile(0.05)
    plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'5% VaR: {var_95:.2%}')
    plt.title('Portfolio Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return os.path.abspath(output_path)
