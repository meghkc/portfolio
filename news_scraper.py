"""
news_scraper.py

Module to fetch and analyze news sentiment for portfolio holdings.
Uses multiple news sources and performs sentiment analysis with TextBlob.

Dependencies:
- requests
- beautifulsoup4
- textblob (pip install textblob)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textblob import TextBlob
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Analyzes news sentiment and extracts key themes for portfolio holdings."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })

    def analyze_news_sentiment(self, news_data: Dict[str, List[Dict]]) -> Dict:
        """
        Analyzes sentiment for provided news articles grouped by symbols.
        Filters out irrelevant news (not containing financial/market keywords).
        Returns structured analysis including sentiment scores, catalysts, and themes.
        """
        logger.info(f"Starting news sentiment analysis for {len(news_data)} symbols.")

        sentiment_report = {
            'symbol_sentiments': {},
            'market_themes': [],
            'catalyst_alerts': [],
            'sentiment_summary': {}
        }

        try:
            all_sentiment_scores = []
            for symbol, articles in news_data.items():
                if not articles:
                    continue
                # Filter out irrelevant news
                filtered_articles = [a for a in articles if self._extract_keywords(a.get('title', ''))]
                if not filtered_articles:
                    logger.info(f"No relevant news for {symbol} after filtering.")
                    continue
                symbol_result = self._analyze_symbol_news(symbol, filtered_articles)
                sentiment_report['symbol_sentiments'][symbol] = symbol_result
                all_sentiment_scores.extend([article['sentiment'] for article in symbol_result['articles']])

            if all_sentiment_scores:
                avg_sentiment = sum(all_sentiment_scores) / len(all_sentiment_scores)
                sentiment_report['sentiment_summary'] = {
                    'average_sentiment': avg_sentiment,
                    'sentiment_label': self._sentiment_label(avg_sentiment),
                    'total_articles': len(all_sentiment_scores),
                    'positive_ratio': len([s for s in all_sentiment_scores if s > 0.1]) / len(all_sentiment_scores)
                }

            sentiment_report['market_themes'] = self._extract_market_themes(news_data)
            sentiment_report['catalyst_alerts'] = self._identify_catalysts(news_data)

            logger.info("Completed news sentiment analysis.")
        except Exception as e:
            logger.error(f"Error during news sentiment analysis: {e}")

        return sentiment_report

    def _analyze_symbol_news(self, symbol: str, articles: List[Dict]) -> Dict:
        """Analyze sentiment for each article of a specific symbol."""
        result = {
            'symbol': symbol,
            'article_count': len(articles),
            'articles': [],
            'average_sentiment': 0.0,
            'sentiment_trend': 'neutral',
            'key_themes': [],
            'catalysts': []
        }

        try:
            sentiments = []
            all_keywords = []

            for article in articles:
                title = article.get('title', '')
                blob = TextBlob(title)
                sentiment_score = blob.sentiment.polarity  # Range: -1 to 1

                sentiment_label = self._sentiment_label(sentiment_score)
                keywords = self._extract_keywords(title)

                article_data = {
                    'title': title,
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published': article.get('published', ''),
                    'sentiment': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'keywords': keywords
                }

                result['articles'].append(article_data)
                sentiments.append(sentiment_score)
                all_keywords.extend(keywords)

            if sentiments:
                avg_sent = sum(sentiments) / len(sentiments)
                result['average_sentiment'] = avg_sent
                result['sentiment_trend'] = self._sentiment_label(avg_sent)

            result['key_themes'] = self._get_top_keywords(all_keywords)
            result['catalysts'] = self._identify_symbol_catalysts(symbol, articles)

        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {e}")

        return result

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract financial/market-related keywords from text."""
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'upgrade', 'downgrade', 'target', 'buy', 'sell', 'hold',
            'acquisition', 'merger', 'partnership', 'contract', 'deal',
            'lawsuit', 'investigation', 'regulatory', 'approval', 'fda',
            'breakthrough', 'innovation', 'product', 'launch', 'expansion',
            'layoffs', 'hiring', 'ceo', 'management', 'dividend', 'buyback',
            'ipo', 'spinoff', 'bankruptcy', 'debt', 'credit', 'rating'
        ]
        text_lower = text.lower()
        return [kw for kw in financial_keywords if kw in text_lower]

    def _get_top_keywords(self, keywords: List[str], top_n: int = 3) -> List[str]:
        """Return most frequent keywords."""
        freq = {}
        for kw in keywords:
            freq[kw] = freq.get(kw, 0) + 1
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:top_n]]

    def _sentiment_label(self, score: float) -> str:
        """Translate sentiment score to label."""
        if score > 0.2:
            return 'positive'
        elif score < -0.2:
            return 'negative'
        else:
            return 'neutral'

    def _extract_market_themes(self, news_data: Dict[str, List[Dict]]) -> List[str]:
        """Identify prevalent market themes across all news."""
        themes = []
        try:
            all_titles = [article['title'] for articles in news_data.values() for article in articles]

            theme_keywords = {
                'AI/Technology': ['ai', 'artificial intelligence', 'machine learning', 'tech', 'chip', 'semiconductor'],
                'Regulatory': ['regulatory', 'fda', 'sec', 'investigation', 'antitrust', 'lawsuit'],
                'Earnings': ['earnings', 'revenue', 'profit', 'guidance', 'quarter'],
                'M&A': ['merger', 'acquisition', 'deal', 'buyout', 'takeover'],
                'Market Sentiment': ['rally', 'sell-off', 'volatility', 'uncertainty', 'optimism']
            }

            for theme, keywords in theme_keywords.items():
                count = sum(any(k in title.lower() for k in keywords) for title in all_titles)
                if count >= 2:
                    themes.append(f"{theme} ({count} mentions)")

        except Exception as e:
            logger.error(f"Error extracting market themes: {e}")

        return themes

    def _identify_catalysts(self, news_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Identify potential price-moving catalysts from news."""
        catalysts = []
        try:
            catalyst_keywords = {
                'earnings': ['earnings', 'quarterly results', 'guidance'],
                'regulatory': ['fda approval', 'regulatory approval', 'investigation'],
                'corporate_action': ['merger', 'acquisition', 'dividend', 'buyback', 'spinoff'],
                'product': ['product launch', 'breakthrough', 'innovation', 'patent'],
                'management': ['ceo', 'management change', 'leadership']
            }

            for symbol, articles in news_data.items():
                for article in articles:
                    title_lower = article.get('title', '').lower()
                    for cat_type, keywords in catalyst_keywords.items():
                        if any(k in title_lower for k in keywords):
                            catalysts.append({
                                'symbol': symbol,
                                'type': cat_type,
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'published': article.get('published', ''),
                                'impact': self._assess_catalyst_impact(cat_type, title_lower)
                            })
                            break
        except Exception as e:
            logger.error(f"Error identifying catalysts: {e}")
        return catalysts

    def _identify_symbol_catalysts(self, symbol: str, articles: List[Dict]) -> List[str]:
        """Identify key catalysts specific to a given symbol."""
        catalysts = []
        try:
            for article in articles:
                title = article.get('title', '').lower()
                if any(x in title for x in ['upgrade', 'buy', 'outperform', 'bullish']):
                    catalysts.append('Analyst upgrade')
                elif any(x in title for x in ['earnings beat', 'revenue beat', 'strong results']):
                    catalysts.append('Strong earnings')
                elif any(x in title for x in ['partnership', 'contract', 'deal']):
                    catalysts.append('Business development')
                elif any(x in title for x in ['downgrade', 'sell', 'underperform', 'bearish']):
                    catalysts.append('Analyst downgrade')
                elif any(x in title for x in ['investigation', 'lawsuit', 'regulatory']):
                    catalysts.append('Regulatory risk')
                elif any(x in title for x in ['earnings miss', 'revenue miss', 'weak results']):
                    catalysts.append('Weak earnings')
        except Exception as e:
            logger.error(f"Error identifying catalysts for symbol {symbol}: {e}")

        return list(set(catalysts))  # Remove duplicates

    def _assess_catalyst_impact(self, cat_type: str, title_lower: str) -> str:
        """Assign impact level to catalyst based on keywords."""
        if any(word in title_lower for word in ['major', 'significant', 'breakthrough', 'blockbuster']):
            return 'high'
        if any(word in title_lower for word in ['minor', 'small', 'routine']):
            return 'low'
        return 'medium'
