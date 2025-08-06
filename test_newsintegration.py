# # Test email configuration
# from email_sender import EmailSender
# from config import EMAIL_CONFIG
# sender = EmailSender(EMAIL_CONFIG)
# sender.test_email_connection() 


# # Test data fetching
# from data_fetcher import DataFetcher
# from config import PORTFOLIO_CONFIG
# fetcher = DataFetcher(PORTFOLIO_CONFIG)
# prices = fetcher.fetch_current_prices(['AAPL', 'NVDA'])
# print(prices)


# # Test news analysis
# from data_fetcher import DataFetcher
# from news_scraper import NewsAnalyzer
# from config import PORTFOLIO_CONFIG

# fetcher = DataFetcher(PORTFOLIO_CONFIG)
# news_analyzer = NewsAnalyzer(PORTFOLIO_CONFIG)

# symbols = list(PORTFOLIO_CONFIG['holdings'].keys())
# news = fetcher.fetch_news_for_symbols(symbols)
# analysis = news_analyzer.analyze_news_sentiment(news)

# print(analysis)  # Inspect output for sanity

# import yfinance as yf
# print(yf.Ticker("AAPL").info["regularMarketPrice"])


