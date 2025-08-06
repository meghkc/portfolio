"""
Data fetching module for stocks, crypto, and news data
Uses yfinance for market data and web scraping for news
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles all external data collection for portfolio analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for all symbols using configured data sources
        Returns dict mapping symbol to current price
        """
        logger.info(f"Fetching current prices for {len(symbols)} symbols")
        prices = {}
        source = self.config.get('data', {}).get('price_source', 'yfinance')

        if source == 'yfinance':
            try:
                tickers = yf.download(symbols, period="1d", interval="1d", 
                                    group_by='ticker', auto_adjust=True, 
                                    prepost=True, threads=True)
                for symbol in symbols:
                    price = None
                    try:
                        if len(symbols) == 1:
                            price = tickers['Close'].iloc[-1]
                        else:
                            price = tickers[symbol]['Close'].iloc[-1]
                        if pd.isna(price) or price == 0:
                            raise ValueError("NaN or zero price from batch fetch")
                        prices[symbol] = float(price)
                        logger.debug(f"Fetched {symbol}: ${price:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not fetch price for {symbol} from batch: {e}")
                        # Fallback to individual ticker fetch (history)
                        try:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(period="1d")
                            if not hist.empty and not pd.isna(hist['Close'].iloc[-1]) and hist['Close'].iloc[-1] != 0:
                                prices[symbol] = float(hist['Close'].iloc[-1])
                                logger.debug(f"Fetched {symbol} from history: ${hist['Close'].iloc[-1]:.2f}")
                            else:
                                # Fallback to info['regularMarketPrice']
                                info = ticker.info
                                reg_price = info.get('regularMarketPrice')
                                if reg_price is not None and not pd.isna(reg_price) and reg_price != 0:
                                    prices[symbol] = float(reg_price)
                                    logger.debug(f"Fetched {symbol} from info['regularMarketPrice']: ${reg_price:.2f}")
                                else:
                                    logger.error(f"Failed to fetch {symbol} price from all sources")
                                    prices[symbol] = float('nan')
                        except Exception as e2:
                            logger.error(f"Failed to fetch {symbol} price from all sources: {e2}")
                            prices[symbol] = float('nan')
            except Exception as e:
                logger.error(f"Batch price fetch failed: {e}")
        elif source == 'alphavantage':
            # Placeholder for Alpha Vantage integration
            logger.warning("Alpha Vantage price source not yet implemented.")
        elif source == 'finnhub':
            # Placeholder for Finnhub integration
            logger.warning("Finnhub price source not yet implemented.")
        else:
            logger.error(f"Unknown price source: {source}")
        return prices
    
    def fetch_historical_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for portfolio analysis
        Returns dict mapping symbol to DataFrame with OHLCV data
        """
        logger.info(f"Fetching {period} historical data for {len(symbols)} symbols")
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    historical_data[symbol] = hist
                    logger.debug(f"Fetched {len(hist)} days of data for {symbol}")
                else:
                    logger.warning(f"No historical data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {symbol}: {e}")
                
        return historical_data
    
    def calculate_returns_and_volatility(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Calculate returns, volatility, and other metrics from historical data
        Returns dict with metrics for each symbol
        """
        logger.info("Calculating returns and volatility metrics")
        metrics = {}
        
        for symbol, data in historical_data.items():
            try:
                # Calculate daily returns
                daily_returns = data['Close'].pct_change().dropna()
                
                # Annualized metrics
                annual_return = (1 + daily_returns.mean()) ** 252 - 1
                annual_volatility = daily_returns.std() * np.sqrt(252)
                
                # Recent performance
                ytd_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
                five_day_return = (data['Close'].iloc[-1] / data['Close'].iloc[-6]) - 1
                
                # Risk metrics
                sharpe_ratio = (annual_return - self.config['data']['risk_free_rate']) / annual_volatility
                max_drawdown = self._calculate_max_drawdown(data['Close'])
                
                metrics[symbol] = {
                    'current_price': float(data['Close'].iloc[-1]),
                    'ytd_return': ytd_return,
                    'five_day_return': five_day_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'last_updated': datetime.now()
                }
                
                logger.debug(f"Calculated metrics for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to calculate metrics for {symbol}: {e}")
                
        return metrics
    
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        peak = price_series.expanding().max()
        drawdown = (price_series - peak) / peak
        return drawdown.min()
    
    def fetch_news_for_symbols(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        Fetch recent news for portfolio symbols using configured sources
        Returns dict mapping symbol to list of news articles
        """
        logger.info(f"Fetching news for {len(symbols)} symbols")
        news_data = {}
        sources = self.config.get('data', {}).get('news_sources', ['finviz', 'yahoo_finance'])

        for symbol in symbols:
            try:
                symbol_news = []
                if 'yahoo_finance' in sources:
                    yahoo_news = self._fetch_yahoo_news(symbol)
                    symbol_news.extend(yahoo_news)
                if 'finviz' in sources:
                    finviz_news = self._fetch_finviz_news(symbol)
                    symbol_news.extend(finviz_news)
                # Placeholder for additional news sources
                if 'finnhub' in sources:
                    logger.warning("Finnhub news source not yet implemented.")
                # Remove duplicates and sort by date
                symbol_news = self._deduplicate_news(symbol_news)
                news_data[symbol] = symbol_news[:5]  # Keep top 5 most recent
                logger.debug(f"Fetched {len(symbol_news)} news items for {symbol}")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {e}")
                news_data[symbol] = []
        return news_data
    
    def _fetch_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        news_items = []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:3]:  # Top 3 from Yahoo
                news_items.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'source': 'Yahoo Finance'
                })
                
        except Exception as e:
            logger.debug(f"Yahoo news fetch failed for {symbol}: {e}")
            
        return news_items
    
    def _fetch_finviz_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Finviz"""
        news_items = []
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_table = soup.find('table', {'id': 'news-table'})
            if news_table:
                for row in news_table.find_all('tr')[:3]:  # Top 3 from Finviz
                    try:
                        link_cell = row.find('a')
                        if link_cell:
                            title = link_cell.text.strip()
                            url = link_cell.get('href', '')
                            
                            # Parse date from Finviz format
                            date_cell = row.find('td')
                            published = self._parse_finviz_date(date_cell.text if date_cell else '')
                            
                            news_items.append({
                                'title': title,
                                'url': url,
                                'published': published,
                                'source': 'Finviz'
                            })
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Finviz news fetch failed for {symbol}: {e}")
            
        return news_items
    
    def _parse_finviz_date(self, date_str: str) -> datetime:
        """Parse Finviz date format"""
        try:
            # Finviz uses formats like "Jul-26-24 07:30PM"
            if len(date_str.split()) >= 2:
                date_part = date_str.split()[0]
                return datetime.strptime(date_part, "%b-%d-%y")
            else:
                return datetime.now() - timedelta(days=1)
        except:
            return datetime.now() - timedelta(days=1)
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news items and sort by date"""
        seen_titles = set()
        unique_news = []
        
        for item in news_list:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_news.append(item)
        
        # Sort by publication date (newest first)
        unique_news.sort(key=lambda x: x['published'], reverse=True)
        return unique_news

    def fetch_crypto_fear_greed_index(self) -> Optional[int]:
        """Fetch Fear & Greed Index for crypto market sentiment"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['data']:
                return int(data['data'][0]['value'])
                
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            
        return None
