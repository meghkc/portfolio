"""
Core portfolio management engine
Orchestrates data collection, analysis, rebalancing, and reporting
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data_fetcher import DataFetcher
from risk_analyzer import RiskAnalyzer
from news_scraper import NewsAnalyzer
from email_sender import EmailSender

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Main portfolio management class that orchestrates all components
    Based on your previous portfolio review requirements
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.holdings = config['holdings']
        self.watchlist = config['watchlist']
        self.rebalancing_rules = config['rebalancing']
        
        # Initialize components
        self.data_fetcher = DataFetcher(config)
        self.risk_analyzer = RiskAnalyzer(config)
        self.news_analyzer = NewsAnalyzer(config)
        self.email_sender = EmailSender(config['email'])
        
        logger.info("Portfolio Manager initialized")
    
    def generate_weekly_report(self) -> Dict:
        """
        Generate comprehensive weekly portfolio report
        This is the main function called every Monday at 7:30 AM
        """
        logger.info("Starting weekly portfolio analysis")
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_summary': {},
            'performance_metrics': {},
            'risk_analysis': {},
            'rebalancing_recommendations': {},
            'news_highlights': {},
            'watchlist_analysis': {},
            'market_outlook': {},
            'data_warnings': []  # For user-facing data issues
        }
        
        try:
            # 1. Collect current data

            all_symbols = list(self.holdings.keys()) + list(self.watchlist.keys())
            current_prices = self.data_fetcher.fetch_current_prices(all_symbols)
            # Log fetched prices for debugging
            logger.info('Fetched current prices:')
            for symbol in self.holdings:
                logger.info(f"  {symbol}: {current_prices.get(symbol)}")
            historical_data = self.data_fetcher.fetch_historical_data(all_symbols, period="1y")

            # --- Data Validation: Holdings ---
            invalid_holdings = []
            for symbol, holding_info in self.holdings.items():
                shares = holding_info.get('shares', 0)
                price = current_prices.get(symbol, None)
                if shares is None or shares <= 0 or price is None or np.isnan(price) or price <= 0:
                    invalid_holdings.append(symbol)
            if invalid_holdings:
                warning = f"The following holdings have missing or invalid shares/price and were excluded from calculations: {', '.join(invalid_holdings)}."
                logger.warning(warning)
                report['data_warnings'].append(warning)

            # --- Data Validation: Historical Data ---
            missing_hist = []
            for symbol in self.holdings:
                data = historical_data.get(symbol)
                if data is None or data.empty or 'Close' not in data or data['Close'].isnull().all():
                    missing_hist.append(symbol)
            if missing_hist:
                warning = f"No valid historical data for: {', '.join(missing_hist)}. Performance and risk metrics may be incomplete."
                logger.warning(warning)
                report['data_warnings'].append(warning)

            # 2. Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(current_prices, historical_data)
            report['portfolio_summary'] = portfolio_metrics

            # 3. Performance analysis
            performance_metrics = self._analyze_performance(historical_data)
            report['performance_metrics'] = performance_metrics

            # 4. Risk analysis
            risk_data = {
                'weights': {symbol: info['target_weight'] for symbol, info in self.holdings.items()},
                'returns_data': {symbol: data['Close'].pct_change().dropna() 
                               for symbol, data in historical_data.items() if not data.empty},
                'historical_data': historical_data
            }
            risk_analysis = self.risk_analyzer.generate_risk_report(risk_data)
            report['risk_analysis'] = risk_analysis

            # --- Advanced Analytics ---
            # Portfolio returns for risk metrics and visualization


            def _get_portfolio_returns():
                returns_data = {symbol: data['Close'].pct_change().dropna() for symbol, data in historical_data.items() if not data.empty and 'Close' in data and not data['Close'].isnull().all()}
                if not returns_data:
                    logger.warning("No valid returns data for any holdings. Portfolio returns will be empty.")
                    report['data_warnings'].append("No valid returns data for any holdings. Risk visualization and advanced analytics will be skipped.")
                    return pd.Series(dtype=float)
                returns_df = pd.DataFrame(returns_data).dropna()
                # Only use symbols present in both returns_df and holdings
                valid_symbols = [s for s in returns_df.columns if s in self.holdings]
                if not valid_symbols:
                    logger.warning("No valid symbols with both returns and holdings. Portfolio returns will be empty.")
                    report['data_warnings'].append("No valid symbols with both returns and holdings. Risk visualization and advanced analytics will be skipped.")
                    return pd.Series(dtype=float)
                returns_df = returns_df[valid_symbols]
                weights = {s: self.holdings[s]['target_weight'] for s in valid_symbols}
                weights_vec = pd.Series(weights)
                # Align weights vector with returns_df columns
                weights_vec = weights_vec.reindex(returns_df.columns).fillna(0)
                if returns_df.empty or weights_vec.empty:
                    logger.warning("Returns DataFrame or weights vector is empty. Portfolio returns will be empty.")
                    report['data_warnings'].append("Returns DataFrame or weights vector is empty. Risk visualization and advanced analytics will be skipped.")
                    return pd.Series(dtype=float)
                return returns_df.dot(weights_vec)

            portfolio_returns = _get_portfolio_returns()
            from risk_visualization import plot_portfolio_risk
            risk_chart_path = plot_portfolio_risk(portfolio_returns)
            report['risk_chart_path'] = risk_chart_path

            # 5. Rebalancing recommendations
            rebalancing_recs = self._generate_rebalancing_recommendations(
                current_prices, portfolio_metrics, risk_analysis
            )
            report['rebalancing_recommendations'] = rebalancing_recs

            # 6. News analysis
            portfolio_symbols = list(self.holdings.keys())
            news_data = self.data_fetcher.fetch_news_for_symbols(portfolio_symbols)
            news_analysis = self.news_analyzer.analyze_news_sentiment(news_data)
            report['news_highlights'] = news_analysis

            # 7. Watchlist analysis
            watchlist_analysis = self._analyze_watchlist_opportunities(current_prices)
            report['watchlist_analysis'] = watchlist_analysis

            # 8. Market outlook
            market_outlook = self._generate_market_outlook()
            report['market_outlook'] = market_outlook

            logger.info("Weekly portfolio report generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")

        return report
    
    def _calculate_portfolio_metrics(self, current_prices: Dict[str, float], 
                                   historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate current portfolio metrics and allocations"""
        logger.info("Calculating portfolio metrics")
        
        metrics = {
            'total_value': 0,
            'current_allocations': {},
            'target_allocations': {},
            'allocation_drift': {},
            'ytd_performance': {},
            'position_values': {}
        }
        
        try:
            total_value = 0
            # Only include valid holdings (shares > 0 and price > 0)
            for symbol, holding_info in self.holdings.items():
                shares = holding_info.get('shares', 0)
                current_price = current_prices.get(symbol, 0)
                if shares is None or shares <= 0 or current_price is None or np.isnan(current_price) or current_price <= 0:
                    logger.warning(f"Skipping holding {symbol} due to invalid shares or price.")
                    continue
                position_value = shares * current_price
                metrics['position_values'][symbol] = position_value
                total_value += position_value
            metrics['total_value'] = total_value
            # Calculate current vs target allocations
            for symbol, holding_info in self.holdings.items():
                position_value = metrics['position_values'].get(symbol, 0)
                current_weight = position_value / total_value if total_value > 0 else 0
                target_weight = holding_info['target_weight']
                metrics['current_allocations'][symbol] = current_weight
                metrics['target_allocations'][symbol] = target_weight
                metrics['allocation_drift'][symbol] = current_weight - target_weight
                # YTD performance calculation
                data = historical_data.get(symbol)
                if data is not None and not data.empty and 'Close' in data and not data['Close'].isnull().all():
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    if start_price > 0:
                        ytd_return = (end_price / start_price) - 1
                        metrics['ytd_performance'][symbol] = ytd_return
            logger.info(f"Portfolio total value: ${total_value:,.2f}")
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
        return metrics
    
    def _analyze_performance(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze portfolio performance metrics"""
        logger.info("Analyzing performance metrics")
        
        performance = {
            'best_performers': [],
            'worst_performers': [],
            'high_volatility_assets': [],
            'correlation_insights': [],
            'sector_performance': {}
        }
        
        try:
            ytd_returns = {}
            volatilities = {}

            # Calculate returns and volatilities with validation
            for symbol, data in historical_data.items():
                if data is None or data.empty or 'Close' not in data or data['Close'].isnull().all():
                    logger.warning(f"No valid price data for {symbol}, skipping performance calculation.")
                    continue
                returns = data['Close'].pct_change().dropna()
                if returns.empty:
                    logger.warning(f"No returns data for {symbol}, skipping.")
                    continue
                ytd_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
                volatility = returns.std() * np.sqrt(252)
                ytd_returns[symbol] = ytd_return
                volatilities[symbol] = volatility

            # Identify best and worst performers
            if ytd_returns:
                sorted_returns = sorted(ytd_returns.items(), key=lambda x: x[1], reverse=True)
                performance['best_performers'] = sorted_returns[:3]
                performance['worst_performers'] = sorted_returns[-3:]
            else:
                performance['best_performers'] = []
                performance['worst_performers'] = []

            # High volatility assets (>60%)
            high_vol = [(symbol, vol) for symbol, vol in volatilities.items() if vol > 0.60]
            performance['high_volatility_assets'] = high_vol

            # Sector performance aggregation
            sector_performance = {}
            for symbol, holding_info in self.holdings.items():
                category = holding_info.get('category', 'other')
                ytd_return = ytd_returns.get(symbol, None)
                if ytd_return is None:
                    continue
                if category not in sector_performance:
                    sector_performance[category] = []
                sector_performance[category].append(ytd_return)

            # Average sector returns
            for sector, returns in sector_performance.items():
                if returns and len(returns) > 1:
                    avg = np.mean(returns)
                    cov = np.cov(returns)
                    performance['sector_performance'][sector] = avg
                elif returns:
                    logger.warning(f"Insufficient data for sector {sector} to calculate covariance; only one data point.")
                    performance['sector_performance'][sector] = np.mean(returns)
                else:
                    logger.warning(f"No returns data for sector {sector}.")
                    performance['sector_performance'][sector] = 0

            logger.info("Performance analysis completed")

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")

        return performance
    
    def _generate_rebalancing_recommendations(self, current_prices: Dict[str, float],
                                            portfolio_metrics: Dict, risk_analysis: Dict) -> Dict:
        """
        Generate specific rebalancing recommendations based on your strategy
        Implements the logic from your previous portfolio review
        """
        logger.info("Generating rebalancing recommendations")
        
        recommendations = {
            'trim_positions': [],
            'add_positions': [],
            'maintain_positions': [],
            'new_entries': [],
            'risk_adjustments': [],
            'priority_actions': []
        }
        
        try:
            current_allocations = portfolio_metrics.get('current_allocations', {})
            allocation_drift = portfolio_metrics.get('allocation_drift', {})
            
            # Apply your specific rebalancing rules
            
            # 1. Trim over-allocated positions
            for symbol, drift in allocation_drift.items():
                current_weight = current_allocations.get(symbol, 0)
                target_weight = self.holdings[symbol]['target_weight']
                
                if drift > self.rebalancing_rules['min_rebalance_threshold']:
                    trim_amount = drift
                    recommendations['trim_positions'].append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'trim_amount': trim_amount,
                        'reason': f"Over-allocated by {drift:.1%}"
                    })
                
                elif drift < -self.rebalancing_rules['min_rebalance_threshold']:
                    add_amount = abs(drift)
                    recommendations['add_positions'].append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'add_amount': add_amount,
                        'reason': f"Under-allocated by {abs(drift):.1%}"
                    })
                
                else:
                    recommendations['maintain_positions'].append(symbol)
            
            # 2. Check watchlist for entry opportunities
            for symbol, watchlist_info in self.watchlist.items():
                current_price = current_prices.get(symbol, float('inf'))
                entry_target = watchlist_info['entry_target']
                
                if entry_target == 0 or current_price <= entry_target:
                    recommendations['new_entries'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': entry_target,
                        'target_weight': watchlist_info['target_weight'],
                        'category': watchlist_info['category'],
                        'reason': "Entry target reached" if entry_target > 0 else "Add regardless of price"
                    })
            
            # 3. Risk-based adjustments
            individual_risks = risk_analysis.get('individual_risks', {})
            for symbol, risk_metrics in individual_risks.items():
                if risk_metrics['volatility'] > 0.70:  # >70% volatility
                    recommendations['risk_adjustments'].append({
                        'symbol': symbol,
                        'action': 'reduce_position',
                        'current_volatility': risk_metrics['volatility'],
                        'reason': f"Excessive volatility: {risk_metrics['volatility']:.1%}"
                    })
                
                if risk_metrics['current_drawdown'] < -0.35:  # >35% drawdown
                    recommendations['risk_adjustments'].append({
                        'symbol': symbol,
                        'action': 'review_position',
                        'current_drawdown': risk_metrics['current_drawdown'],
                        'reason': f"Large drawdown: {risk_metrics['current_drawdown']:.1%}"
                    })
            
            # 4. Generate priority actions (your specific strategy)
            priority_actions = []
            
            # DOGE trim (from your previous recommendations)
            doge_weight = current_allocations.get('DOGE-USD', 0)
            if doge_weight > 0.01:
                priority_actions.append({
                    'action': 'TRIM',
                    'symbol': 'DOGE-USD',
                    'from_weight': doge_weight,
                    'to_weight': 0.01,
                    'reason': 'Reduce speculative crypto exposure'
                })
            
            # MRVL increase (from your previous recommendations)
            mrvl_weight = current_allocations.get('MRVL', 0)
            if mrvl_weight < 0.07:
                priority_actions.append({
                    'action': 'ADD',
                    'symbol': 'MRVL',
                    'from_weight': mrvl_weight,
                    'to_weight': 0.07,
                    'reason': 'AI infrastructure play with lower regulatory risk'
                })
            
            # SOUN profit taking (from your previous recommendations)
            soun_weight = current_allocations.get('SOUN', 0)
            if soun_weight > 0.08:
                priority_actions.append({
                    'action': 'TRIM',
                    'symbol': 'SOUN',
                    'from_weight': soun_weight,
                    'to_weight': 0.08,
                    'reason': 'Take profits after strong YTD gains'
                })
            
            recommendations['priority_actions'] = priority_actions
            
            logger.info(f"Generated {len(priority_actions)} priority rebalancing actions")
            
        except Exception as e:
            logger.error(f"Rebalancing recommendations failed: {e}")
            
        return recommendations
    
    def _analyze_watchlist_opportunities(self, current_prices: Dict[str, float]) -> Dict:
        """Analyze watchlist stocks for entry opportunities"""
        logger.info("Analyzing watchlist opportunities")
        
        opportunities = {
            'ready_to_buy': [],
            'approaching_targets': [],
            'overpriced': []
        }
        
        try:
            for symbol, watchlist_info in self.watchlist.items():
                current_price = current_prices.get(symbol, 0)
                entry_target = watchlist_info['entry_target']
                
                if entry_target == 0:  # Add regardless of price
                    opportunities['ready_to_buy'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'reason': 'Strategic addition',
                        'category': watchlist_info['category']
                    })
                elif current_price <= entry_target:
                    opportunities['ready_to_buy'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': entry_target,
                        'discount': (entry_target - current_price) / entry_target,
                        'category': watchlist_info['category']
                    })
                elif current_price <= entry_target * 1.05:  # Within 5% of target
                    opportunities['approaching_targets'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': entry_target,
                        'premium': (current_price - entry_target) / entry_target,
                        'category': watchlist_info['category']
                    })
                else:
                    opportunities['overpriced'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': entry_target,
                        'premium': (current_price - entry_target) / entry_target,
                        'category': watchlist_info['category']
                    })
            
            logger.info(f"Found {len(opportunities['ready_to_buy'])} immediate opportunities")
            
        except Exception as e:
            logger.error(f"Watchlist analysis failed: {e}")
            
        return opportunities
    
    def _generate_market_outlook(self) -> Dict:
        """Generate market outlook and macro analysis"""
        logger.info("Generating market outlook")
        
        outlook = {
            'crypto_sentiment': {},
            'sector_trends': {},
            'macro_factors': [],
            'key_catalysts': []
        }
        
        try:
            # Crypto Fear & Greed Index
            fear_greed = self.data_fetcher.fetch_crypto_fear_greed_index()
            if fear_greed:
                outlook['crypto_sentiment'] = {
                    'fear_greed_index': fear_greed,
                    'sentiment': self._interpret_fear_greed(fear_greed)
                }
            
            # Macro factors based on your previous analysis
            outlook['macro_factors'] = [
                "Fed maintains cautious stance with core PCE at 3.2%",
                "AI capex trending toward $330B in 2025 (+33% Y/Y)",
                "Travel demand remains strong with load factors >89%",
                "China chip export restrictions continue to impact semis"
            ]
            
            # Key catalysts to watch
            outlook['key_catalysts'] = [
                "DOJ/SAMR antitrust actions on mega-cap tech",
                "NVDA Blackwell shipment timeline (H2 2025)",
                "Fed rate decision (November cut probability)",
                "AI datacenter buildout acceleration",
                "Boeing 737-MAX10 delivery timeline"
            ]
            
        except Exception as e:
            logger.error(f"Market outlook generation failed: {e}")
            
        return outlook
    
    def _interpret_fear_greed(self, index_value: int) -> str:
        """Interpret Fear & Greed Index value"""
        if index_value <= 25:
            return "Extreme Fear"
        elif index_value <= 45:
            return "Fear"
        elif index_value <= 55:
            return "Neutral"
        elif index_value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def send_email_report(self, report: Dict) -> bool:
        """Send formatted email report, with risk chart attachment if available."""
        try:
            email_content = self._format_email_report(report)
            risk_chart_path = report.get('risk_chart_path')
            success = self.email_sender.send_portfolio_report(email_content, risk_chart_path=risk_chart_path)

            if success:
                logger.info("Weekly portfolio report emailed successfully")
            else:
                logger.error("Failed to send email report")

            return success

        except Exception as e:
            logger.error(f"Email report sending failed: {e}")
            return False
    
    def _format_email_report(self, report: Dict) -> str:
        """Format the portfolio report as a styled HTML email with detailed news sentiment and catalysts."""
        from html import escape
        timestamp = report['timestamp'].strftime('%B %d, %Y at %I:%M %p')

        def html_link(text, url):
            if url:
                return f'<a href="{escape(url)}" target="_blank" style="color:#1a73e8;text-decoration:none;">{escape(text)}</a>'
            return escape(text)

        email_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
    body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.5; margin: 20px; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 5px; }}
    h2 {{ color: #34495e; border-bottom: 2px solid #2980b9; padding-bottom: 3px; }}
    h3 {{ color: #4a6274; margin-top: 15px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; text-align: left; }}
    ul {{ padding-left: 15px; }}
    li {{ margin-bottom: 8px; }}
    .positive {{ color: #27ae60; font-weight: bold; }}
    .negative {{ color: #e74c3c; font-weight: bold; }}
    .neutral {{ color: #7f8c8d; font-weight: normal; }}
    .warn {{ color: #f39c12; font-weight: bold; }}
    .datamissing {{ color: #e67e22; font-weight: bold; }}
</style>
</head>
<body>
<h1>Weekly Portfolio Review – {escape(timestamp)}</h1>

<!-- Data Warnings Section -->
{self._format_data_warnings(report.get('data_warnings', []))}

<h2>Portfolio Summary</h2>
<p><strong>Total Value:</strong> ${report['portfolio_summary'].get('total_value', 0):,.2f}</p>

<h2>Performance Highlights</h2>
"""
        # Top Performers Table
        performance = report.get('performance_metrics', {})
        best_performers = performance.get('best_performers', [])[:3]
        worst_performers = performance.get('worst_performers', [])[:3]

        if best_performers:
            email_html += "<h3>Top Performers</h3><ul>"
            for symbol, ret in best_performers:
                email_html += f"<li>{escape(symbol)}: <span class='positive'>{ret:.1%}</span></li>"
            email_html += "</ul>"
        if worst_performers:
            email_html += "<h3>Underperformers</h3><ul>"
            for symbol, ret in worst_performers:
                email_html += f"<li>{escape(symbol)}: <span class='negative'>{ret:.1%}</span></li>"
            email_html += "</ul>"

        # Advanced Analytics: Risk Metrics and Scenario Analysis
        risk_metrics = report.get('risk_metrics', {})
        scenario_analysis = report.get('scenario_analysis', {})
        if risk_metrics:
            email_html += "<h2>Risk Metrics</h2><ul>"
            if 'sharpe_ratio' in risk_metrics:
                email_html += f"<li>Sharpe Ratio: <strong>{risk_metrics['sharpe_ratio']:.2f}</strong></li>"
            if 'sortino_ratio' in risk_metrics:
                email_html += f"<li>Sortino Ratio: <strong>{risk_metrics['sortino_ratio']:.2f}</strong></li>"
            if 'value_at_risk' in risk_metrics:
                email_html += f"<li>Value at Risk (5% VaR): <strong>{risk_metrics['value_at_risk']:.2%}</strong></li>"
            if 'conditional_var' in risk_metrics:
                email_html += f"<li>Conditional VaR (CVaR): <strong>{risk_metrics['conditional_var']:.2%}</strong></li>"
            if 'beta' in risk_metrics and risk_metrics['beta'] is not None:
                email_html += f"<li>Beta (vs. SPY): <strong>{risk_metrics['beta']:.2f}</strong></li>"
            email_html += "</ul>"
        if scenario_analysis:
            email_html += "<h2>Scenario Analysis: -10% Market Drop</h2><ul>"
            for symbol, new_price in scenario_analysis.items():
                email_html += f"<li>{escape(symbol)}: <strong>${new_price:.2f}</strong> (if -10% drop)</li>"
            email_html += "</ul>"

        # Priority Rebalancing Actions
        rebalancing = report.get('rebalancing_recommendations', {})
        priority_actions = rebalancing.get('priority_actions', [])

        if priority_actions:
            email_html += "<h2>Priority Rebalancing Actions</h2><ul>"
            for action in priority_actions:
                act = escape(action['action'])
                sym = escape(action['symbol'])
                reason = escape(action['reason'])
                email_html += f"<li><strong>{act}</strong> {sym}: {reason}</li>"
            email_html += "</ul>"

        # Watchlist Opportunities
        watchlist = report.get('watchlist_analysis', {})
        ready_to_buy = watchlist.get('ready_to_buy', [])

        if ready_to_buy:
            email_html += "<h2>Watchlist Opportunities</h2><ul>"
            for opp in ready_to_buy:
                sym = escape(opp['symbol'])
                price = opp['current_price']
                reason = escape(opp.get('reason', 'Entry target reached'))
                email_html += f"<li>{sym} at <strong>${price:.2f}</strong> - {reason}</li>"
            email_html += "</ul>"

        # Risk Alerts
        risk_analysis = report.get('risk_analysis', {})
        risk_warnings = risk_analysis.get('risk_warnings', [])

        if risk_warnings:
            email_html += "<h2>Risk Alerts</h2><ul>"
            for warning in risk_warnings:
                email_html += f"<li class='warn'>⚠️ {escape(warning)}</li>"
            email_html += "</ul>"

        # Detailed News Highlights and Sentiment
        news_highlights = report.get('news_highlights', {})
        if news_highlights:
            email_html += "<h2>News Highlights and Sentiment</h2>"
            symbol_sentiments = news_highlights.get('symbol_sentiments', {})

            for symbol, sentiment_info in symbol_sentiments.items():
                avg_sent = sentiment_info.get('average_sentiment', 0.0)
                sent_label = sentiment_info.get('sentiment_trend', 'neutral')
                article_count = sentiment_info.get('article_count', 0)

                sentiment_class = {
                    'positive': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral'
                }.get(sent_label.lower(), 'neutral')

                email_html += f"""
<h3>{escape(symbol)} — Average Sentiment: <span class="{sentiment_class}">{avg_sent:.2f} ({sent_label.capitalize()})</span></h3>
<p>Total Articles Analyzed: {article_count}</p>
<ul>
"""
                for article in sentiment_info.get('articles', [])[:3]:
                    title = article.get('title', 'N/A')
                    sentiment = article.get('sentiment_label', 'neutral').capitalize()
                    url = article.get('url', '')
                    sentiment_css = {
                        'Positive': 'positive',
                        'Negative': 'negative',
                        'Neutral': 'neutral'
                    }.get(sentiment, 'neutral')

                    email_html += f"<li>{html_link(title, url)} [<span class='{sentiment_css}'>{sentiment}</span>]</li>"

                email_html += "</ul>"

                catalysts = sentiment_info.get('catalysts', [])
                if catalysts:
                    email_html += f"<p><strong>Catalysts:</strong> {', '.join(escape(c) for c in catalysts)}</p>"

            # Market-wide catalyst alerts
            catalysts_overall = news_highlights.get('catalyst_alerts', [])
            if catalysts_overall:
                email_html += "<h2>Market-Wide Catalyst Alerts</h2><ul>"
                for cat in catalysts_overall[:5]:
                    symbol = cat.get('symbol', 'N/A')
                    title = cat.get('title', 'No title')
                    impact = cat.get('impact', 'medium').capitalize()
                    url = cat.get('url', '')
                    email_html += f"<li>[{escape(symbol)}] {html_link(title, url)} (Impact: {escape(impact)})</li>"
                email_html += "</ul>"

        # Market sentiment overview
        market_outlook = report.get('market_outlook', {})
        crypto_sentiment = market_outlook.get('crypto_sentiment', {})
        if crypto_sentiment:
            fg_index = crypto_sentiment.get('fear_greed_index', 'N/A')
            sentiment = crypto_sentiment.get('sentiment', 'N/A')
            email_html += f"""
<h2>Market Sentiment</h2>
<p>Crypto Fear &amp; Greed Index: <strong>{fg_index}</strong> ({escape(sentiment)})</p>
"""

        email_html += """
<hr>
<p style="font-size: small; color: #888;">Automated Portfolio Management System</p>
</body>
</html>
"""
        return email_html

    def _format_data_warnings(self, warnings: list) -> str:
        """Format user-facing data warnings for the email report."""
        if not warnings:
            return ""
        html = '<div class="datamissing"><h2>⚠️ Data Issues Detected</h2><ul>'
        for w in warnings:
            html += f'<li>{w}</li>'
        html += '</ul></div>'
        return html


