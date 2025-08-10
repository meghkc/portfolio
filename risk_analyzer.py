"""
Risk analysis and portfolio metrics calculation
Implements advanced risk metrics including VaR, portfolio volatility, and correlation analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Comprehensive portfolio risk analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        self.trading_days = config['data']['trading_days_per_year']
        self.risk_free_rate = config['data']['risk_free_rate']
    
    def calculate_portfolio_volatility(self, returns_data: Dict[str, pd.Series], 
                                     weights: Dict[str, float]) -> Tuple[float, pd.DataFrame]:
        """
        Calculate portfolio volatility using covariance matrix
        Formula: σ_portfolio = √(w^T * Σ * w)
        Returns volatility and correlation matrix
        """
        logger.info("Calculating portfolio volatility")
        
        try:
            # Align returns data with weights
            aligned_returns = []
            aligned_symbols = []
            weight_vector = []
            
            for symbol, weight in weights.items():
                if symbol in returns_data and not returns_data[symbol].empty:
                    aligned_returns.append(returns_data[symbol])
                    aligned_symbols.append(symbol)
                    weight_vector.append(weight)
            
            if not aligned_returns:
                logger.warning("No valid returns data for volatility calculation")
                return 0.0, pd.DataFrame()
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(dict(zip(aligned_symbols, aligned_returns)))
            returns_df = returns_df.dropna()
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov() * self.trading_days
            
            # Convert weights to numpy array
            weights_array = np.array(weight_vector)
            weights_array = weights_array / np.sum(weights_array)  # Normalize weights
            
            # Calculate portfolio variance: w^T * Σ * w
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix.values, weights_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate correlation matrix for additional insights
            correlation_matrix = returns_df.corr()
            
            logger.info(f"Portfolio volatility: {portfolio_volatility:.1%}")
            return portfolio_volatility, correlation_matrix
            
        except Exception as e:
            logger.error(f"Portfolio volatility calculation failed: {e}")
            return 0.0, pd.DataFrame()
    
    def calculate_var_cvar(self, portfolio_returns: pd.Series, confidence_level: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        """
        try:
            # Sort returns
            sorted_returns = portfolio_returns.sort_values()
            
            # Calculate VaR (percentile)
            var_index = int(confidence_level * len(sorted_returns))
            var = sorted_returns.iloc[var_index]
            
            # Calculate CVaR (expected shortfall)
            cvar = sorted_returns.iloc[:var_index].mean()
            
            logger.debug(f"VaR ({confidence_level:.1%}): {var:.1%}, CVaR: {cvar:.1%}")
            return var, cvar
            
        except Exception as e:
            logger.error(f"VaR/CVaR calculation failed: {e}")
            return 0.0, 0.0

    def calculate_parametric_var(self, weights: Dict[str, float], returns_data: Dict[str, pd.Series],
                                 confidence_level: float = 0.05, method: str = 'gaussian') -> Dict[str, float]:
        """Parametric VaR using covariance matrix.

        method:
            'gaussian' - standard normal
            'cornish-fisher' - adjust for skew/kurtosis (if enough data)
        Returns dict with daily_var, annual_var, z_score and method used.
        """
        try:
            # Align returns
            aligned = {s: r for s, r in returns_data.items() if s in weights and not r.empty}
            if not aligned:
                return {}
            returns_df = pd.DataFrame(aligned).dropna()
            if returns_df.empty:
                return {}
            w_vec = pd.Series({s: weights[s] for s in returns_df.columns})
            w_vec = w_vec / w_vec.sum()
            mean_vec = returns_df.mean()
            cov = returns_df.cov()
            # Portfolio stats
            mu_p = float(np.dot(w_vec.values, mean_vec.values))
            sigma_p = float(np.sqrt(np.dot(w_vec.values.T, np.dot(cov.values, w_vec.values))))
            # Z-score for lower tail
            z = stats.norm.ppf(confidence_level)
            var_gaussian = mu_p + z * sigma_p
            if method == 'cornish-fisher':
                # Adjust z for skew/kurtosis of portfolio returns
                port_ret_series = returns_df.dot(w_vec)
                skew = stats.skew(port_ret_series)
                kurt = stats.kurtosis(port_ret_series, fisher=True)  # excess kurtosis
                z_cf = (z + (z**2 - 1) * skew / 6 +
                        (z**3 - 3 * z) * kurt / 24 -
                        (2 * z**3 - 5 * z) * (skew**2) / 36)
                var_cf = mu_p + z_cf * sigma_p
                daily_var = min(var_gaussian, var_cf)  # conservative: take more negative
                method_used = 'cornish-fisher'
            else:
                daily_var = var_gaussian
                method_used = 'gaussian'
            annual_var = daily_var * np.sqrt(self.trading_days)
            return {
                'daily_var': daily_var,
                'annual_var': annual_var,
                'z_score': z,
                'method': method_used,
                'daily_mean': mu_p,
                'daily_vol': sigma_p
            }
        except Exception as e:
            logger.error(f"Parametric VaR calculation failed: {e}")
            return {}

    def calculate_risk_contributions(self, weights: Dict[str, float], returns_data: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Compute marginal and percentage risk contributions for each asset.

        RC_i = w_i * (Σ w)_i / σ_p
        Returns dict mapping symbol to {'weight','marginal_contribution','pct_risk_contribution'}.
        """
        try:
            aligned = {s: r for s, r in returns_data.items() if s in weights and not r.empty}
            if not aligned:
                return {}
            df = pd.DataFrame(aligned).dropna()
            if df.empty:
                return {}
            w = pd.Series({s: weights[s] for s in df.columns}, dtype=float)
            w = w / w.sum()
            cov = df.cov() * self.trading_days  # annualize
            # Portfolio variance
            port_var = float(np.dot(w.values.T, np.dot(cov.values, w.values)))
            if port_var <= 0:
                return {}
            port_vol = np.sqrt(port_var)
            # Marginal contribution vector (Σ w)
            marginal = cov.dot(w)
            # Risk contribution
            rc = w * marginal / port_vol
            total_rc = rc.sum()
            results = {}
            for s in df.columns:
                # Use explicit label-based access (.loc) to avoid any future positional ambiguity warnings
                results[s] = {
                    'weight': float(w.loc[s]),
                    'marginal_contribution': float(marginal.loc[s]),
                    'pct_risk_contribution': float((rc.loc[s] / total_rc)) if total_rc != 0 else 0.0
                }
            return results
        except Exception as e:
            logger.error(f"Risk contribution calculation failed: {e}")
            return {}

    def calculate_monte_carlo_var(self, weights: Dict[str, float], returns_data: Dict[str, pd.Series],
                                   confidence_level: float = 0.05, n_sims: int = 5000, seed: int = 42) -> Dict[str, float]:
        """Monte Carlo VaR & CVaR using multivariate normal approximation of returns.

        Returns dict with mc_var (VaR), mc_cvar, distribution stats. Using daily horizon.
        """
        try:
            aligned = {s: r for s, r in returns_data.items() if s in weights and not r.empty}
            if not aligned:
                return {}
            df = pd.DataFrame(aligned).dropna()
            if df.empty or len(df.columns) < 2:
                return {}
            w = pd.Series({s: weights[s] for s in df.columns}); w = w / w.sum()
            mean_vec = df.mean().values
            cov = df.cov().values
            rng = np.random.default_rng(seed)
            sims = rng.multivariate_normal(mean_vec, cov, size=n_sims)
            port_rets = sims.dot(w.values)
            sorted_rets = np.sort(port_rets)
            idx = int(confidence_level * len(sorted_rets))
            mc_var = sorted_rets[idx]
            mc_cvar = sorted_rets[:idx].mean() if idx > 0 else mc_var
            return {
                'mc_var': mc_var,
                'mc_cvar': mc_cvar,
                'mean': float(port_rets.mean()),
                'std': float(port_rets.std()),
                'confidence_level': confidence_level,
                'simulations': n_sims
            }
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return {}

    def compute_risk_parity_weights(self, returns_data: Dict[str, pd.Series], max_iter: int = 500, tol: float = 1e-4) -> Dict[str, float]:
        """Approximate risk parity weights using iterative proportional adjustment.

        Returns dict of symbol->weight. Empty if insufficient data.
        """
        try:
            aligned = {s: r for s, r in returns_data.items() if not r.empty}
            if len(aligned) < 2:
                return {}
            df = pd.DataFrame(aligned).dropna()
            if df.empty:
                return {}
            cov = df.cov() * self.trading_days
            assets = list(df.columns)
            n = len(assets)
            w = np.array([1.0 / n] * n, dtype=float)
            for _ in range(max_iter):
                # Marginal contribution Σ w using numpy arrays to avoid Series positional ambiguity
                marginal_vec = cov.values @ w
                port_vol = np.sqrt(w.T @ cov.values @ w)
                rc = w * marginal_vec / port_vol
                target = port_vol / n
                diff = rc - target
                if np.max(np.abs(diff)) < tol:
                    break
                # Adjust weights inversely to deviation
                w = w * (target / (rc + 1e-12))
                w = np.maximum(w, 1e-8)
                w = w / w.sum()
            # Use explicit indexing to avoid FutureWarning on Series positional access
            # w is numpy array so positional access is explicit and pandas FutureWarning avoided
            return {assets[idx]: float(w[idx]) for idx in range(n)}
        except Exception as e:
            logger.error(f"Risk parity weight computation failed: {e}")
            return {}

    def compute_rolling_metrics(self, returns_data: Dict[str, pd.Series], window: int = 20) -> Dict[str, Dict[str, float]]:
        """Compute rolling volatility (annualized) and last rolling return for each asset."""
        metrics = {}
        try:
            for symbol, series in returns_data.items():
                if series is None or series.empty:
                    continue
                roll_vol = series.tail(window).std() * np.sqrt(self.trading_days) if len(series) >= window else series.std() * np.sqrt(self.trading_days)
                roll_ret = (1 + series.tail(window)).prod() - 1 if len(series) >= window else (1 + series).prod() - 1
                metrics[symbol] = {
                    'rolling_vol_20d': float(roll_vol) if roll_vol == roll_vol else 0.0,
                    'rolling_return_20d': float(roll_ret)
                }
        except Exception as e:
            logger.error(f"Rolling metrics computation failed: {e}")
        return metrics
    
    def calculate_sharpe_ratio(self, portfolio_return: float, portfolio_volatility: float) -> float:
        """Calculate Sharpe ratio: (Return - Risk-free rate) / Volatility"""
        try:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            logger.debug(f"Sharpe ratio: {sharpe:.2f}")
            return sharpe
        except:
            return 0.0
    
    def calculate_sortino_ratio(self, portfolio_returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio using downside deviation
        Only considers negative volatility (downside risk)
        """
        try:
            # Calculate downside returns (below target)
            downside_returns = portfolio_returns[portfolio_returns < target_return]
            
            if len(downside_returns) == 0:
                return float('inf')  # No downside
            
            # Calculate downside deviation
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.trading_days)
            
            # Calculate mean return
            mean_return = portfolio_returns.mean() * self.trading_days
            
            sortino = (mean_return - target_return) / downside_deviation
            logger.debug(f"Sortino ratio: {sortino:.2f}")
            return sortino
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            return 0.0
    
    def calculate_maximum_drawdown(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        try:
            # Calculate running maximum
            peak = portfolio_values.expanding().max()
            
            # Calculate drawdown series
            drawdown = (portfolio_values - peak) / peak
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown duration
            drawdown_duration = 0
            current_dd = 0
            max_dd_duration = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_dd += 1
                    max_dd_duration = max(max_dd_duration, current_dd)
                else:
                    current_dd = 0
            
            logger.debug(f"Max drawdown: {max_drawdown:.1%}, Duration: {max_dd_duration} days")
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_dd_duration,
                'current_drawdown': drawdown.iloc[-1],
                'recovery_factor': portfolio_values.iloc[-1] / peak.iloc[-1] - 1
            }
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return {}
    
    def calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark (e.g., SPY)"""
        try:
            # Align the data
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data points
                return 1.0
            
            # Calculate beta using linear regression
            covariance = aligned_data['asset'].cov(aligned_data['benchmark'])
            benchmark_variance = aligned_data['benchmark'].var()
            
            beta = covariance / benchmark_variance
            logger.debug(f"Beta: {beta:.2f}")
            return beta
            
        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    def assess_concentration_risk(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze portfolio concentration using Herfindahl-Hirschman Index and other metrics
        """
        try:
            weight_values = np.array(list(weights.values()))
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = np.sum(weight_values ** 2)
            
            # Effective number of holdings
            effective_holdings = 1 / hhi if hhi > 0 else 0
            
            # Largest position weight
            max_weight = np.max(weight_values)
            
            # Top 3 concentration
            top3_weight = np.sum(np.sort(weight_values)[-3:])
            
            # Gini coefficient (inequality measure)
            gini = self._calculate_gini_coefficient(weight_values)
            
            concentration_metrics = {
                'hhi': hhi,
                'effective_holdings': effective_holdings,
                'max_position': max_weight,
                'top3_concentration': top3_weight,
                'gini_coefficient': gini
            }
            
            logger.debug(f"Concentration - HHI: {hhi:.3f}, Effective holdings: {effective_holdings:.1f}")
            return concentration_metrics
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return {}
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for portfolio concentration"""
        try:
            # Sort weights
            sorted_weights = np.sort(weights)
            n = len(sorted_weights)
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_weights)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights))) / (n * np.sum(sorted_weights)) - (n + 1) / n
            
            return gini
        except:
            return 0.0
    
    def generate_risk_report(self, portfolio_data: Dict) -> Dict:
        """Generate comprehensive risk analysis report.

        Expects portfolio_data with keys:
          - weights: Dict[symbol->weight]
          - returns_data: Dict[symbol->pd.Series of returns]
          - historical_data: Dict[symbol->pd.DataFrame with 'Close']

        Optionally, an attribute self.latest_sentiment may be pre-populated externally
        with structure {symbol: { 'average_sentiment': float, 'article_count': int }} which
        will be integrated to produce sentiment-adjusted flags.
        """
        logger.info("Generating comprehensive risk report")

        risk_report = {
            'timestamp': datetime.now(),
            'portfolio_metrics': {},
            'individual_risks': {},
            'correlation_analysis': {},
            'risk_warnings': []
        }

        try:
            weights = portfolio_data.get('weights', {})
            returns_data = portfolio_data.get('returns_data', {})
            historical_data = portfolio_data.get('historical_data', {})

            # Portfolio-level risk metrics
            portfolio_vol, corr_matrix = self.calculate_portfolio_volatility(returns_data, weights)
            risk_report['portfolio_metrics']['volatility'] = portfolio_vol
            risk_report['correlation_analysis']['matrix'] = corr_matrix

            parametric = self.calculate_parametric_var(weights, returns_data, confidence_level=0.05, method='cornish-fisher')
            if parametric:
                risk_report['portfolio_metrics']['parametric_var'] = parametric

            risk_contrib = self.calculate_risk_contributions(weights, returns_data)
            if risk_contrib:
                risk_report['portfolio_metrics']['risk_contributions'] = risk_contrib

            mc = self.calculate_monte_carlo_var(weights, returns_data, confidence_level=0.05, n_sims=3000)
            if mc:
                risk_report['portfolio_metrics']['monte_carlo_var'] = mc

            rp_weights = self.compute_risk_parity_weights(returns_data)
            if rp_weights:
                risk_report['portfolio_metrics']['risk_parity_weights'] = rp_weights

            rolling = self.compute_rolling_metrics(returns_data)
            if rolling:
                risk_report['portfolio_metrics']['rolling_metrics'] = rolling

            concentration = self.assess_concentration_risk(weights)
            risk_report['portfolio_metrics']['concentration'] = concentration

            # Stress scenario analysis if config provides scenarios
            scenarios = self.config.get('data', {}).get('stress_scenarios', [])
            if scenarios:
                try:
                    risk_report['stress_tests'] = self._run_stress_scenarios(weights, historical_data, scenarios)
                except Exception as e:
                    logger.warning(f"Stress scenario computation failed: {e}")

            # Individual asset risk assessment
            for symbol, weight in weights.items():
                if symbol in historical_data and not historical_data[symbol].empty:
                    asset_data = historical_data[symbol]
                    if 'Close' not in asset_data:
                        continue
                    asset_returns = asset_data['Close'].pct_change().dropna()
                    if asset_returns.empty:
                        continue
                    asset_vol = asset_returns.std() * np.sqrt(self.trading_days)
                    asset_dd = self.calculate_maximum_drawdown(asset_data['Close'])
                    risk_report['individual_risks'][symbol] = {
                        'weight': weight,
                        'volatility': asset_vol,
                        'max_drawdown': asset_dd.get('max_drawdown', 0),
                        'current_drawdown': asset_dd.get('current_drawdown', 0)
                    }

            # Multi-factor beta estimation (OLS) if factor proxies available
            try:
                factor_map = self.config.get('data', {}).get('factor_proxies', {})
                if factor_map:
                    factor_results = self._compute_multi_factor_betas(historical_data, factor_map)
                    if factor_results:
                        risk_report['factor_exposures'] = factor_results
            except Exception as e:
                logger.warning(f"Factor model computation failed: {e}")

            # Integrate sentiment if available
            sentiment_source = getattr(self, 'latest_sentiment', None)
            if sentiment_source and isinstance(sentiment_source, dict):
                by_symbol = {}
                weighted_score = 0.0
                total_w = 0.0
                for sym, w in weights.items():
                    s_info = sentiment_source.get(sym, {})
                    score = s_info.get('average_sentiment', 0.0)
                    articles = s_info.get('article_count', s_info.get('articles', 0))
                    by_symbol[sym] = {'score': score, 'articles': articles, 'weighted_score': score * w}
                    weighted_score += score * w
                    total_w += w
                portfolio_sent = weighted_score / total_w if total_w > 0 else 0.0
                flag = 'NEUTRAL'
                if portfolio_sent <= -0.25:
                    flag = 'NEGATIVE_SENTIMENT_RISK'
                elif portfolio_sent >= 0.25:
                    flag = 'POSITIVE_SENTIMENT'
                try:
                    from datetime import UTC as _UTC
                    asof_ts = datetime.now(_UTC).isoformat()
                except Exception:
                    # Fallback if UTC not available in older Python (though 3.11+ should have it)
                    asof_ts = datetime.utcnow().isoformat() + 'Z'
                risk_report['sentiment'] = {
                    'by_symbol': by_symbol,
                    'portfolio_weighted_score': portfolio_sent,
                    'flag': flag,
                    'asof': asof_ts
                }

            # Risk warnings last (may use sentiment flag)
            risk_report['risk_warnings'] = self._generate_risk_warnings(risk_report)
            if 'sentiment' in risk_report and risk_report['sentiment']['flag'] == 'NEGATIVE_SENTIMENT_RISK':
                risk_report['risk_warnings'].append('Portfolio-level negative news sentiment detected.')

            logger.info("Risk report generated successfully")
        except Exception as e:
            logger.error(f"Risk report generation failed: {e}")

        return risk_report

    def _compute_multi_factor_betas(self, historical_data: Dict[str, pd.DataFrame], factor_map: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Compute multi-factor betas using simple OLS vs ETF proxies.

        Steps:
          1. Build factor returns DataFrame from provided proxy symbols if data available.
          2. For each asset with sufficient overlap (>=60 data points), regress asset excess returns on factor returns.
          3. Return dict: symbol -> { 'alpha': float, factor_name: beta, 'r2': float, 'n': int }.
        """
        # Collect factor returns
        factor_returns = {}
        for f_name, proxy in factor_map.items():
            df = historical_data.get(proxy)
            if df is None or df.empty or 'Close' not in df:
                continue
            ret = df['Close'].pct_change().dropna()
            if not ret.empty:
                factor_returns[f_name] = ret
        if len(factor_returns) < 1:
            return {}
        factors_df = pd.DataFrame(factor_returns).dropna()
        if factors_df.empty:
            return {}
        results = {}
        # Prepare matrix with intercept
        X = factors_df
        X_mean = X.mean()
        X_centered = X - X_mean
        # Precompute (X'X)^-1 for speed if assets align; but each asset may have different overlap window
        for sym, df in historical_data.items():
            if sym in factor_map.values():
                continue  # skip factor proxies themselves to reduce clutter
            if df is None or df.empty or 'Close' not in df:
                continue
            y = df['Close'].pct_change().dropna()
            merged = pd.concat([y.rename('asset'), factors_df], axis=1).dropna()
            if len(merged) < 60:
                continue
            y_vec = merged['asset'] - merged['asset'].mean()
            X_m = merged[factors_df.columns]
            X_c = X_m - X_m.mean()
            try:
                # Solve betas = (X'X)^-1 X'y (no intercept due to centering)
                XtX = X_c.T.dot(X_c)
                # Regularization safeguard: add tiny ridge if ill-conditioned
                if np.linalg.cond(XtX.values) > 1e8:
                    XtX += np.eye(XtX.shape[0]) * 1e-6
                betas = np.linalg.solve(XtX.values, X_c.T.dot(y_vec).values)
                y_hat = X_c.values.dot(betas)
                ss_tot = np.sum((y_vec.values) ** 2)
                ss_res = np.sum((y_vec.values - y_hat) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                # Alpha approximation (original mean difference)
                alpha = merged['asset'].mean() - np.sum(betas * X_m.mean().values)
                res_entry = {'alpha': float(alpha), 'r2': float(r2), 'n': int(len(merged))}
                for i, f_name in enumerate(factors_df.columns):
                    res_entry[f_name] = float(betas[i])
                results[sym] = res_entry
            except Exception as e:
                logger.debug(f"Beta regression failed for {sym}: {e}")
        return results

    def _run_stress_scenarios(self, weights: Dict[str, float], historical_data: Dict[str, pd.DataFrame], scenarios: List[Dict]) -> List[Dict]:
        """Apply predefined category and default shocks to estimate portfolio P/L impact.

        Approach: For each scenario, compute shocked price = last_close * (1 + shock) per asset, shock derived from
        category_shocks mapping else default_shock. Estimate position value change and aggregate.
        """
        results = []
        # Derive last prices & categories
        last_prices = {}
        categories = {}
        for sym, data in historical_data.items():
            if data is not None and not data.empty and 'Close' in data:
                last_prices[sym] = float(data['Close'].iloc[-1])
        for sym, info in self.config.get('holdings', {}).items():
            categories[sym] = info.get('category', 'other')
        total_portfolio_value = sum(last_prices.get(sym, 0) * info.get('shares', 0) for sym, info in self.config.get('holdings', {}).items())
        if total_portfolio_value <= 0:
            return results
        for scenario in scenarios:
            name = scenario.get('name', 'Unnamed')
            category_shocks = scenario.get('category_shocks', {})
            default_shock = scenario.get('default_shock', -0.10)
            pl_total = 0.0
            per_asset = {}
            for sym, info in self.config.get('holdings', {}).items():
                shares = info.get('shares', 0)
                if shares <= 0:
                    continue
                price = last_prices.get(sym)
                if price is None:
                    continue
                cat = categories.get(sym, 'other')
                shock = category_shocks.get(cat, default_shock)
                shocked_price = price * (1 + shock)
                delta_value = (shocked_price - price) * shares
                pl_total += delta_value
                per_asset[sym] = {
                    'category': cat,
                    'shock': shock,
                    'price': price,
                    'shocked_price': shocked_price,
                    'pnl': delta_value
                }
            results.append({
                'scenario': name,
                'portfolio_pct_impact': pl_total / total_portfolio_value,
                'pnl': pl_total,
                'details': per_asset
            })
        return results
    
    def _generate_risk_warnings(self, risk_report: Dict) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []
        
        try:
            # Portfolio volatility warning
            portfolio_vol = risk_report['portfolio_metrics'].get('volatility', 0)
            if portfolio_vol > 0.35:  # >35% volatility
                warnings.append(f"High portfolio volatility: {portfolio_vol:.1%}")
            
            # Concentration warnings
            concentration = risk_report['portfolio_metrics'].get('concentration', {})
            max_position = concentration.get('max_position', 0)
            if max_position > 0.15:  # >15% in single position
                warnings.append(f"High concentration risk: {max_position:.1%} in largest position")
            
            # Individual asset warnings
            for symbol, metrics in risk_report['individual_risks'].items():
                if metrics['volatility'] > 0.60:  # >60% individual volatility
                    warnings.append(f"High volatility in {symbol}: {metrics['volatility']:.1%}")
                
                if metrics['current_drawdown'] < -0.30:  # >30% current drawdown
                    warnings.append(f"Large drawdown in {symbol}: {metrics['current_drawdown']:.1%}")
            
        except Exception as e:
            logger.error(f"Risk warning generation failed: {e}")
            
        return warnings
