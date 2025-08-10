"""
Core portfolio management engine
Orchestrates data collection, analysis, rebalancing, and reporting
"""

import pandas as pd
import numpy as np
import logging
import os
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data_fetcher import DataFetcher
from risk_analyzer import RiskAnalyzer
from news_scraper import NewsAnalyzer
from email_sender import EmailSender
from config import update_portfolio_prices  # re-export for backward compatibility with tests

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
            # --- Configuration validation & normalization ---
            config_warnings = self._validate_and_normalize_config()
            if config_warnings:
                report['data_warnings'].extend(config_warnings)
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

            # 4. Risk analysis (sentiment will be attached later once news is processed)
            risk_data = {
                'weights': {symbol: info['target_weight'] for symbol, info in self.holdings.items()},
                'returns_data': {symbol: data['Close'].pct_change().dropna() 
                               for symbol, data in historical_data.items() if not data.empty},
                'historical_data': historical_data
            }
            # placeholder; sentiment attribute may be set after news analysis and risk report can be regenerated if desired
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
                current_prices, portfolio_metrics, risk_analysis, historical_data=historical_data
            )
            report['rebalancing_recommendations'] = rebalancing_recs

            # 6. News analysis
            portfolio_symbols = list(self.holdings.keys())
            news_data = self.data_fetcher.fetch_news_for_symbols(portfolio_symbols)
            news_analysis = self.news_analyzer.analyze_news_sentiment(news_data)
            report['news_highlights'] = news_analysis
            # Pass summarized symbol sentiments into risk analyzer and augment existing risk report
            try:
                symbol_sentiments = news_analysis.get('symbol_sentiments', {})
                simplified = {sym: {'average_sentiment': data.get('average_sentiment', 0.0), 'article_count': data.get('article_count', 0)} for sym, data in symbol_sentiments.items()}
                self.risk_analyzer.latest_sentiment = simplified
                # Regenerate only sentiment portion without recomputing heavy metrics? Simpler to regenerate full risk report
                updated_risk = self.risk_analyzer.generate_risk_report(risk_data)
                # Retain previously computed chart path etc.
                if 'risk_chart_path' in report:
                    updated_risk['risk_chart_path'] = report['risk_chart_path']
                report['risk_analysis'] = updated_risk
                # Inject sentiment-driven weight tilt recommendations (non-binding)
                try:
                    sent_cfg = self.config['data'].get('sentiment', {})
                    sent = updated_risk.get('sentiment', {})
                    by_symbol = sent.get('by_symbol', {})
                    if not by_symbol:
                        raise ValueError('No symbol sentiment data')
                    tilts = {}
                    pos_budget = sent_cfg.get('global_positive_tilt_budget', 0.0)
                    neg_budget = sent_cfg.get('global_negative_tilt_budget', 0.0)
                    pos_thresh = sent_cfg.get('symbol_positive_threshold', 0.25)
                    neg_thresh = sent_cfg.get('symbol_negative_threshold', -0.25)
                    max_delta = sent_cfg.get('max_symbol_delta', 0.01)
                    min_articles = sent_cfg.get('min_articles_for_action', 3)
                    vol_guard = sent_cfg.get('volatility_guardrail', 0.80)
                    dd_guard = sent_cfg.get('drawdown_guardrail', -0.30)
                    neutral_band = sent_cfg.get('neutral_band', 0.15)
                    exponent = sent_cfg.get('decay_exponent', 1.0)

                    indiv_risks = updated_risk.get('individual_risks', {})

                    # Build candidate lists
                    positive_candidates = []
                    negative_candidates = []
                    for sym, data in by_symbol.items():
                        score = data.get('score', 0.0)
                        arts = data.get('articles', 0)
                        if arts < min_articles or abs(score) < neutral_band:
                            continue
                        risks = indiv_risks.get(sym, {})
                        vol = risks.get('volatility', 0)
                        dd = risks.get('current_drawdown', 0)
                        if score >= pos_thresh and vol < vol_guard and dd > dd_guard:
                            positive_candidates.append((sym, score))
                        elif score <= neg_thresh:
                            negative_candidates.append((sym, score))

                    # Allocate positive budget proportionally to powered scores
                    def allocate_budget(candidates, budget, positive=True):
                        if not candidates or budget == 0:
                            return {}
                        # Transform scores
                        transformed = []
                        for sym, sc in candidates:
                            mag = abs(sc) ** exponent
                            transformed.append((sym, mag))
                        total = sum(m for _, m in transformed) or 1.0
                        allocs = {}
                        for sym, mag in transformed:
                            raw = (mag / total) * abs(budget)
                            raw = min(raw, max_delta)
                            allocs[sym] = raw if positive else -raw
                        return allocs

                    if pos_budget > 0:
                        tilts.update(allocate_budget(positive_candidates, pos_budget, positive=True))
                    if neg_budget < 0:
                        tilts.update(allocate_budget(negative_candidates, neg_budget, positive=False))

                    if tilts:
                        # Normalize if absolute sum exceeds combined budgets due to caps
                        total_pos = sum(v for v in tilts.values() if v > 0)
                        total_neg = sum(-v for v in tilts.values() if v < 0)
                        scale_pos = min(1.0, (pos_budget or 1e-9) / total_pos) if pos_budget > 0 and total_pos > pos_budget else 1.0
                        scale_neg = min(1.0, (abs(neg_budget) or 1e-9) / total_neg) if neg_budget < 0 and total_neg > abs(neg_budget) else 1.0
                        report.setdefault('rebalancing_recommendations', {}).setdefault('sentiment_tilts', [])
                        for sym, delta in tilts.items():
                            scaled_delta = delta * (scale_pos if delta > 0 else scale_neg)
                            cur_w = self.holdings.get(sym, {}).get('target_weight', 0)
                            new_w = max(cur_w + scaled_delta, 0)
                            report['rebalancing_recommendations']['sentiment_tilts'].append({
                                'symbol': sym,
                                'current_target_weight': cur_w,
                                'suggested_target_weight': new_w,
                                'delta': scaled_delta,
                                'reason': f"Sentiment tilt (score {by_symbol.get(sym, {}).get('score',0):.2f})"
                            })
                except Exception as e:
                    logger.warning(f"Failed to compute sentiment tilts: {e}")
            except Exception as e:
                logger.warning(f"Failed to integrate sentiment into risk report: {e}")

            # 7. Watchlist analysis
            watchlist_analysis = self._analyze_watchlist_opportunities(current_prices)
            report['watchlist_analysis'] = watchlist_analysis
            # Persist composite scores for trend analysis
            try:
                self._persist_watchlist_scores(watchlist_analysis.get('composite_scores', []), report['timestamp'])
            except Exception as e:
                logger.warning(f"Failed to persist watchlist composite scores: {e}")
            # Generate composite trend chart
            try:
                from risk_visualization import plot_composite_trends
                comp_chart = plot_composite_trends()
                if comp_chart:
                    report['composite_trends_chart_path'] = comp_chart
            except Exception as e:
                logger.warning(f"Composite trend plotting failed: {e}")

            # 8. Market outlook
            market_outlook = self._generate_market_outlook()
            report['market_outlook'] = market_outlook

            logger.info("Weekly portfolio report generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")

        return report

    def _validate_and_normalize_config(self) -> List[str]:
        """Validate holdings configuration and normalize target weights if needed.

        Returns list of warning strings for inclusion in the report.
        """
        warnings = []
        try:
            weights = {sym: info.get('target_weight', 0) for sym, info in self.holdings.items()}
            negative = [s for s, w in weights.items() if w is None or w < 0]
            if negative:
                warnings.append(f"Removed/zeroed negative target weights for: {', '.join(negative)}")
                for s in negative:
                    self.holdings[s]['target_weight'] = 0
            total_weight = sum(w for w in weights.values() if w and w > 0)
            if total_weight <= 0:
                warnings.append("Total target weight is zero or negative; cannot normalize weights.")
                return warnings
            # If weights don't sum approximately to 1, normalize
            if abs(total_weight - 1.0) > 0.02:  # >2% drift considered mis-specified
                for s, info in self.holdings.items():
                    w = info.get('target_weight', 0)
                    info['target_weight'] = (w / total_weight) if total_weight > 0 else 0
                warnings.append(f"Target weights normalized (original sum {total_weight:.3f}).")
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
        return warnings

    def _persist_watchlist_scores(self, composite_scores: List[Dict], timestamp: datetime):
        """Append composite watchlist scores to a CSV for longitudinal analysis.

        File: watchlist_scores_history.csv with columns:
        timestamp,symbol,composite_score,distance_to_target,momentum_20d,valuation_score,sentiment_score,entry_target,adjusted_entry_target,category
        """
        if not composite_scores:
            return
        filename = 'watchlist_scores_history.csv'
        file_exists = os.path.isfile(filename)
        fieldnames = ['timestamp','symbol','composite_score','distance_to_target','momentum_20d','valuation_score','sentiment_score','entry_target','adjusted_entry_target','category']
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in composite_scores:
                writer.writerow({
                    'timestamp': timestamp.isoformat(),
                    'symbol': row.get('symbol'),
                    'composite_score': f"{row.get('composite_score',0):.6f}",
                    'distance_to_target': f"{row.get('distance_to_target',0):.6f}",
                    'momentum_20d': f"{row.get('momentum_20d',0):.6f}",
                    'valuation_score': f"{row.get('valuation_score',0):.6f}",
                    'sentiment_score': f"{row.get('sentiment_score',0):.6f}",
                    'entry_target': row.get('entry_target'),
                    'adjusted_entry_target': row.get('adjusted_entry_target'),
                    'category': row.get('category')
                })
    
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
            'sector_performance': {},
            'attribution': {}
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
            sector_weights = {}
            # Capture initial weights for simple attribution baseline (first day value proxies)
            initial_prices = {}
            for symbol, data in historical_data.items():
                if data is not None and not data.empty and 'Close' in data:
                    initial_prices[symbol] = data['Close'].iloc[0]
            for symbol, holding_info in self.holdings.items():
                category = holding_info.get('category', 'other')
                ytd_return = ytd_returns.get(symbol, None)
                if ytd_return is None:
                    continue
                if category not in sector_performance:
                    sector_performance[category] = []
                    sector_weights[category] = []
                sector_performance[category].append(ytd_return)
                # Approximate weight using shares*last price (re-derive each loop for simplicity)
                shares = holding_info.get('shares', 0)
                last_price = historical_data.get(symbol, pd.DataFrame())['Close'].iloc[-1] if symbol in historical_data and not historical_data[symbol].empty else 0
                value = shares * last_price
                sector_weights[category].append(value)

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

            # --- Simple Brinson-style attribution (allocation vs selection) by category ---
            try:
                total_port_value = 0
                asset_values_end = {}
                for symbol, info in self.holdings.items():
                    data = historical_data.get(symbol)
                    if data is None or data.empty or 'Close' not in data:
                        continue
                    end_price = data['Close'].iloc[-1]
                    shares = info.get('shares', 0)
                    val = shares * end_price
                    asset_values_end[symbol] = val
                    total_port_value += val
                category_end_weights = {}
                category_returns = {}
                for symbol, val in asset_values_end.items():
                    category = self.holdings[symbol].get('category', 'other')
                    ret = ytd_returns.get(symbol, 0)
                    category_returns.setdefault(category, []).append(ret)
                    category_end_weights[category] = category_end_weights.get(category, 0) + val
                for c in category_end_weights:
                    category_end_weights[c] /= total_port_value if total_port_value > 0 else 1
                # Benchmark: equal weight across categories as naive benchmark
                categories = list(category_end_weights.keys())
                if categories:
                    bench_weight = 1 / len(categories)
                    attribution = {}
                    for c in categories:
                        actual_w = category_end_weights[c]
                        cat_ret = np.mean(category_returns.get(c, [0]))
                        # Allocation effect: (actual_w - bench_w) * bench_return(=avg all categories)
                        bench_return = np.mean([np.mean(category_returns[k]) for k in categories if category_returns.get(k)]) if categories else 0
                        allocation_effect = (actual_w - bench_weight) * bench_return
                        selection_effect = bench_weight * (cat_ret - bench_return)
                        interaction_effect = (actual_w - bench_weight) * (cat_ret - bench_return)
                        attribution[c] = {
                            'weight': actual_w,
                            'return': cat_ret,
                            'allocation_effect': allocation_effect,
                            'selection_effect': selection_effect,
                            'interaction_effect': interaction_effect
                        }
                    performance['attribution'] = attribution
            except Exception as e:
                logger.warning(f"Attribution calculation failed: {e}")

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")

        return performance
    
    def _generate_rebalancing_recommendations(self, current_prices: Dict[str, float],
                                            portfolio_metrics: Dict, risk_analysis: Dict,
                                            historical_data: Dict[str, pd.DataFrame] = None) -> Dict:
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
                
                # Treat >= threshold as actionable to align with tests expecting action at exact value
                if drift >= self.rebalancing_rules['min_rebalance_threshold']:
                    trim_amount = drift
                    recommendations['trim_positions'].append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'trim_amount': trim_amount,
                        'reason': f"Over-allocated by {drift:.1%}"
                    })
                
                elif drift <= -self.rebalancing_rules['min_rebalance_threshold']:
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
            
            # Enrich with technical indicators if historical data provided
            if historical_data:
                for pa in priority_actions:
                    sym = pa.get('symbol')
                    df = historical_data.get(sym)
                    if df is not None and not df.empty and 'Close' in df:
                        try:
                            tech = self._compute_technical_indicators(df['Close'])
                            pa['technicals'] = tech
                        except Exception as e:
                            logger.warning(f"Failed technicals for {sym}: {e}")
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
            'overpriced': [],
            'composite_scores': []  # ranked list with composite watchlist intelligence
        }
        
        try:
            # To compute momentum/valuation we may need historical data; attempt fetch for watchlist if not already
            try:
                historical_watch = self.data_fetcher.fetch_historical_data(list(self.watchlist.keys()), period="6mo")
            except Exception:
                historical_watch = {}

            # Factor exposures for beta adjustments if available (market beta via risk_analysis stored somewhere?)
            market_beta_shift = 0.0
            try:
                # Approximate using portfolio factor exposures if computed
                # Placeholder: could integrate recent change in market volatility or factor trend; set static 0 for now
                market_beta_shift = 0.0
            except Exception:
                market_beta_shift = 0.0

            composite_entries = []
            for symbol, watchlist_info in self.watchlist.items():
                current_price = current_prices.get(symbol, 0)
                entry_target = watchlist_info['entry_target']
                category = watchlist_info.get('category', 'other')

                # Auto-adjust entry target based on beta shift (placeholder: if positive shift >0 raise targets modestly)
                adjusted_entry = entry_target
                if entry_target > 0 and market_beta_shift != 0:
                    adjusted_entry = entry_target * (1 + 0.2 * market_beta_shift)  # dampened adjustment

                # Momentum: 20d return (or shorter if limited data)
                mom_score = 0.0
                valuation_proxy = 0.0  # simplified: relative position vs 6m high/low
                hist_df = historical_watch.get(symbol)
                if hist_df is not None and not hist_df.empty and 'Close' in hist_df:
                    closes = hist_df['Close'].dropna()
                    if len(closes) >= 5:
                        recent = closes.tail(21) if len(closes) >= 21 else closes
                        mom_score = (recent.iloc[-1] / recent.iloc[0]) - 1
                        six_high = closes.max()
                        six_low = closes.min()
                        if six_high > six_low:
                            valuation_proxy = (recent.iloc[-1] - six_low) / (six_high - six_low)  # 0=cheap,1=expensive
                            valuation_proxy = 1 - valuation_proxy  # invert so higher is cheaper / more attractive
                # Distance to target (positive if below target, else negative)
                distance = 0.0
                if entry_target > 0 and current_price > 0:
                    distance = (entry_target - current_price) / entry_target
                # Sentiment placeholder: if we have risk_analyzer sentiment attach symbol score
                sentiment_score = 0.0
                try:
                    sentiment = getattr(self.risk_analyzer, 'latest_sentiment', {})
                    if symbol in sentiment:
                        sentiment_score = sentiment[symbol].get('average_sentiment', 0.0)
                except Exception:
                    pass

                # Normalize components to z-like scale / simple bounds
                # Clamp extreme values
                def clamp(v, lo, hi):
                    return max(lo, min(hi, v))
                mom_n = clamp(mom_score, -0.5, 0.5) / 0.5  # -1 to 1
                val_n = clamp(valuation_proxy, 0, 1) * 2 - 1  # 0..1 -> -1..1
                dist_n = clamp(distance, -0.5, 0.5) / 0.5  # -1..1
                sent_n = clamp(sentiment_score, -0.5, 0.5) / 0.5  # -1..1
                composite = 0.35 * dist_n + 0.25 * mom_n + 0.25 * val_n + 0.15 * sent_n

                composite_entries.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'entry_target': entry_target,
                    'adjusted_entry_target': adjusted_entry,
                    'category': category,
                    'distance_to_target': distance,
                    'momentum_20d': mom_score,
                    'valuation_score': valuation_proxy,
                    'sentiment_score': sentiment_score,
                    'composite_score': composite
                })

                # Traditional buckets using adjusted entry
                bucket_target = adjusted_entry
                if entry_target == 0:  # unconditional add
                    opportunities['ready_to_buy'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'reason': 'Strategic addition',
                        'category': category
                    })
                elif current_price <= bucket_target:
                    opportunities['ready_to_buy'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': bucket_target,
                        'discount': (bucket_target - current_price) / bucket_target if bucket_target else 0,
                        'category': category
                    })
                elif current_price <= bucket_target * 1.05:
                    opportunities['approaching_targets'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': bucket_target,
                        'premium': (current_price - bucket_target) / bucket_target if bucket_target else 0,
                        'category': category
                    })
                else:
                    opportunities['overpriced'].append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'entry_target': bucket_target,
                        'premium': (current_price - bucket_target) / bucket_target if bucket_target else 0,
                        'category': category
                    })

            # Rank composites descending
            opportunities['composite_scores'] = sorted(composite_entries, key=lambda x: x['composite_score'], reverse=True)
            
            logger.info(f"Found {len(opportunities['ready_to_buy'])} immediate opportunities")
            
        except Exception as e:
            logger.error(f"Watchlist analysis failed: {e}")
            
        return opportunities

    def _compute_technical_indicators(self, close: pd.Series) -> Dict[str, float]:
        """Compute selected technical indicators for enrichment.

        Returns dict with:
          ema_20 / ema_50 / ema_200
          rsi_14 (Wilder)
          macd (12-26 EMA diff)
          macd_signal (9-period EMA of macd)
          macd_hist (macd - signal)
          interpretations (human readable brief meanings)
        """
        if close is None or close.empty:
            return {}
        s = close.dropna()
        if s.empty:
            return {}
        ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1] if len(s) >= 20 else float('nan')
        ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1] if len(s) >= 50 else float('nan')
        ema200 = s.ewm(span=200, adjust=False).mean().iloc[-1] if len(s) >= 200 else float('nan')
        # RSI 14
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        roll_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        roll_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_gain / roll_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] if not rsi.empty else float('nan')
        # MACD
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        macd_val = macd.iloc[-1] if not macd.empty else float('nan')
        signal_val = signal.iloc[-1] if not signal.empty else float('nan')
        hist_val = hist.iloc[-1] if not hist.empty else float('nan')
        latest_price = s.iloc[-1]
        interp = []
        if not pd.isna(ema20) and latest_price > ema20:
            interp.append('Price > EMA20 (short-term strength)')
        if not pd.isna(ema50) and latest_price > ema50:
            interp.append('Price > EMA50 (medium trend up)')
        if not pd.isna(ema200) and latest_price > ema200:
            interp.append('Price > EMA200 (long-term uptrend)')
        if not pd.isna(rsi_val):
            if rsi_val > 70:
                interp.append('RSI overbought')
            elif rsi_val < 30:
                interp.append('RSI oversold')
        if not (pd.isna(macd_val) or pd.isna(signal_val)):
            if macd_val > signal_val:
                interp.append('MACD bullish')
            else:
                interp.append('MACD bearish')
        return {
            'ema_20': float(ema20),
            'ema_50': float(ema50),
            'ema_200': float(ema200),
            'rsi_14': float(rsi_val),
            'macd': float(macd_val),
            'macd_signal': float(signal_val),
            'macd_hist': float(hist_val),
            'interpretations': interp
        }
    
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
            # Prefer sending risk chart; composite trends could be optionally embedded later
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
            email_html += "<h2>Priority Rebalancing Actions</h2><table><tr><th>Action</th><th>Symbol</th><th>Reason</th><th>EMA20</th><th>EMA50</th><th>EMA200</th><th>RSI14</th><th>MACD</th><th>Signal</th><th>Hist</th></tr>"
            for action in priority_actions:
                act = escape(action['action'])
                sym = escape(action['symbol'])
                reason = escape(action['reason'])
                tech = action.get('technicals', {})
                ema20 = tech.get('ema_20', float('nan'))
                ema50 = tech.get('ema_50', float('nan'))
                ema200 = tech.get('ema_200', float('nan'))
                rsi14 = tech.get('rsi_14', float('nan'))
                macd_v = tech.get('macd', float('nan'))
                macd_sig = tech.get('macd_signal', float('nan'))
                macd_hist = tech.get('macd_hist', float('nan'))
                email_html += (
                    f"<tr><td>{act}</td><td>{sym}</td><td>{reason}</td>"
                    f"<td>{ema20:.2f}</td><td>{ema50:.2f}</td><td>{ema200:.2f}</td>"
                    f"<td>{rsi14:.1f}</td><td>{macd_v:.3f}</td><td>{macd_sig:.3f}</td><td>{macd_hist:.3f}</td></tr>"
                )
            email_html += "</table>"
            # Per-symbol interpretations
            sym_interps = [pa for pa in priority_actions if pa.get('technicals', {}).get('interpretations')]
            if sym_interps:
                email_html += "<h3>Technical Interpretations (Per Symbol)</h3>"
                for pa in sym_interps:
                    sym = escape(pa['symbol'])
                    interps = pa['technicals']['interpretations']
                    email_html += f"<p><strong>{sym}</strong></p><ul>"
                    for it in interps:
                        email_html += f"<li>{escape(it)}</li>"
                    email_html += "</ul>"
            # General legend
            email_html += "<h3>Indicator Legend</h3><ul>"
            email_html += "<li><strong>EMA20/50/200:</strong> Short/medium/long trend baselines; price above indicates uptrend for that horizon.</li>"
            email_html += "<li><strong>RSI14:</strong> >70 overbought, <30 oversold (momentum/mean-reversion signal).</li>"
            email_html += "<li><strong>MACD / Signal / Hist:</strong> MACD above Signal = bullish momentum; Histogram measures momentum of crossover.</li>"
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
        # Composite Watchlist Intelligence
        composite_scores = watchlist.get('composite_scores', [])
        if composite_scores:
            email_html += "<h2>Watchlist Composite Scores</h2><table><tr><th>Symbol</th><th>Composite</th><th>Dist</th><th>Mom 20d</th><th>Val</th><th>Sent</th><th>Adj Target</th></tr>"
            for entry in composite_scores[:10]:  # top 10
                email_html += (
                    f"<tr><td>{escape(entry['symbol'])}</td>"
                    f"<td>{entry['composite_score']:.2f}</td>"
                    f"<td>{entry['distance_to_target']:.1%}</td>"
                    f"<td>{entry['momentum_20d']:.1%}</td>"
                    f"<td>{entry['valuation_score']:.2f}</td>"
                    f"<td>{entry['sentiment_score']:.2f}</td>"
                    f"<td>{entry['adjusted_entry_target']:.2f}</td></tr>"
                )
            email_html += "</table>"
            # Inline composite trend chart (if generated)
            comp_chart_path = report.get('composite_trends_chart_path')
            if comp_chart_path and os.path.isfile(comp_chart_path):
                try:
                    import base64
                    with open(comp_chart_path, 'rb') as img_f:
                        encoded = base64.b64encode(img_f.read()).decode('utf-8')
                    email_html += "<h3>Composite Score Trends</h3>"
                    email_html += f"<img src='data:image/png;base64,{encoded}' alt='Composite Score Trends' style='max-width:100%;height:auto;border:1px solid #ccc;padding:4px;'/>"
                except Exception as e:
                    logger.warning(f"Failed to embed composite trends chart: {e}")

        # Risk Alerts
        risk_analysis = report.get('risk_analysis', {})
        risk_warnings = risk_analysis.get('risk_warnings', [])

        # Inject newly added parametric VaR & risk contributions if present
        parametric_var = risk_analysis.get('portfolio_metrics', {}).get('parametric_var') if risk_analysis else None
        risk_contrib = risk_analysis.get('portfolio_metrics', {}).get('risk_contributions') if risk_analysis else None
        if parametric_var:
            email_html += "<h2>Parametric VaR (95%)</h2><ul>"
            email_html += f"<li>Method: {escape(parametric_var.get('method',''))}</li>"
            email_html += f"<li>Daily VaR: <strong>{parametric_var.get('daily_var',0):.2%}</strong></li>"
            email_html += f"<li>Annualized VaR: <strong>{parametric_var.get('annual_var',0):.2%}</strong></li>"
            email_html += f"<li>Daily Mean: {parametric_var.get('daily_mean',0):.3%}</li>"
            email_html += f"<li>Daily Volatility: {parametric_var.get('daily_vol',0):.2%}</li>"
            email_html += "</ul>"
        if risk_contrib:
            email_html += "<h2>Risk Contributions</h2><table><tr><th>Symbol</th><th>Weight</th><th>% Risk</th></tr>"
            # Show top contributors first
            sorted_rc = sorted(risk_contrib.items(), key=lambda x: x[1]['pct_risk_contribution'], reverse=True)
            for symbol, metrics in sorted_rc[:10]:
                email_html += f"<tr><td>{escape(symbol)}</td><td>{metrics['weight']:.1%}</td><td>{metrics['pct_risk_contribution']:.1%}</td></tr>"
            email_html += "</table>"
        monte_carlo_var = risk_analysis.get('portfolio_metrics', {}).get('monte_carlo_var') if risk_analysis else None
        if monte_carlo_var:
            email_html += "<h2>Monte Carlo VaR (95%)</h2><ul>"
            email_html += f"<li>Daily Mean Simulated: {monte_carlo_var.get('mean',0):.3%}</li>"
            email_html += f"<li>Daily Std Simulated: {monte_carlo_var.get('std',0):.2%}</li>"
            email_html += f"<li>Simulated VaR: <strong>{monte_carlo_var.get('mc_var',0):.2%}</strong></li>"
            email_html += f"<li>Simulated CVaR: <strong>{monte_carlo_var.get('mc_cvar',0):.2%}</strong></li>"
            email_html += f"<li>Simulations: {monte_carlo_var.get('simulations')}</li>"
            email_html += "</ul>"
        rp_weights = risk_analysis.get('portfolio_metrics', {}).get('risk_parity_weights') if risk_analysis else None
        if rp_weights:
            email_html += "<h2>Risk Parity Suggested Weights</h2><table><tr><th>Symbol</th><th>Suggested Weight</th></tr>"
            for sym, w in sorted(rp_weights.items(), key=lambda x: x[1], reverse=True):
                email_html += f"<tr><td>{escape(sym)}</td><td>{w:.1%}</td></tr>"
            email_html += "</table>"
        rolling_metrics = risk_analysis.get('portfolio_metrics', {}).get('rolling_metrics') if risk_analysis else None
        if rolling_metrics:
            email_html += "<h2>Rolling 20d Metrics</h2><table><tr><th>Symbol</th><th>20d Vol</th><th>20d Return</th></tr>"
            for sym, vals in rolling_metrics.items():
                email_html += f"<tr><td>{escape(sym)}</td><td>{vals.get('rolling_vol_20d',0):.1%}</td><td>{vals.get('rolling_return_20d',0):.1%}</td></tr>"
            email_html += "</table>"
        # Sentiment summary (if integrated into risk report)
        sentiment_section = risk_analysis.get('sentiment') if risk_analysis else None
        if sentiment_section:
            email_html += "<h2>News Sentiment</h2><ul>"
            email_html += f"<li>Weighted Score: {sentiment_section.get('portfolio_weighted_score',0):.3f}</li>"
            email_html += f"<li>Flag: {escape(sentiment_section.get('flag',''))}</li>"
            email_html += "</ul><table><tr><th>Symbol</th><th>Score</th><th>Articles</th></tr>"
            for sym, sdata in sentiment_section.get('by_symbol', {}).items():
                email_html += f"<tr><td>{escape(sym)}</td><td>{sdata.get('score',0):.2f}</td><td>{sdata.get('articles',0)}</td></tr>"
            email_html += "</table>"
        # Stress tests
        stress_tests = risk_analysis.get('stress_tests') if risk_analysis else None
        if stress_tests:
            email_html += "<h2>Stress Scenarios</h2><table><tr><th>Scenario</th><th>Pct Impact</th><th>P/L</th></tr>"
            for st in stress_tests:
                email_html += f"<tr><td>{escape(st.get('scenario',''))}</td><td>{st.get('portfolio_pct_impact',0):.1%}</td><td>{st.get('pnl',0):,.0f}</td></tr>"
            email_html += "</table>"
        factor_exp = risk_analysis.get('factor_exposures') if risk_analysis else None
        if factor_exp:
            # Show top few assets by absolute market beta
            email_html += "<h2>Factor Exposures (Sample)</h2><table><tr><th>Symbol</th><th>Alpha</th><th>R2</th>"
            # Determine factor columns dynamically
            first_entry = next(iter(factor_exp.values()))
            factor_cols = [c for c in first_entry.keys() if c not in ('alpha','r2','n')]
            for fc in factor_cols:
                email_html += f"<th>{escape(fc)}</th>"
            email_html += "</tr>"
            # Sort by |MKT| if exists else r2
            def sort_key(item):
                sym, vals = item
                if 'MKT' in vals:
                    return abs(vals['MKT'])
                return vals.get('r2', 0)
            for sym, vals in sorted(factor_exp.items(), key=sort_key, reverse=True)[:8]:
                email_html += f"<tr><td>{escape(sym)}</td><td>{vals.get('alpha',0):.4f}</td><td>{vals.get('r2',0):.2f}</td>"
                for fc in factor_cols:
                    email_html += f"<td>{vals.get(fc,0):.2f}</td>"
                email_html += "</tr>"
            email_html += "</table>"

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


