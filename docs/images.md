# Project Images

Runtime-generated charts now default to `docs/images/`:

| Chart | Default Path | Description |
|-------|--------------|-------------|
| Risk distribution | `docs/images/risk_chart.png` | Histogram of daily returns with 5% VaR line |
| Composite trends | `docs/images/composite_trends.png` | Time series of composite watchlist scores (auto top symbols) |

These are ignored by Git via the `*.png` pattern. They will appear locally after running a weekly report.

If you need static examples for documentation, generate them and save under distinct names (e.g. `risk_chart_example.png`) so you can selectively commit those (temporarily remove or narrow the ignore rule for that filename if necessary).

Quick regeneration snippet:

```python
from risk_visualization import plot_portfolio_risk, plot_composite_trends
# returns: pandas Series of portfolio returns; CSV must exist for composite trends
plot_portfolio_risk(returns)
plot_composite_trends()
```

Avoid committing images containing sensitive holdings, dates, or proprietary strategy annotations.
