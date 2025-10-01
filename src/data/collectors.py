"""Data collectors for AI Trading Advisor."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.config import settings

# Try to import yfinance, fallback to dummy data if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available, using dummy data")

# Try to import newsapi
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logging.warning("newsapi-python not available, using dummy data")

# Try to import heavy dependencies, fall back to lighter alternatives
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects market data from various sources."""

    def __init__(self):
        """Initialize market data collector."""
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds

    def collect_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Collect real-time market data for a symbol.

        Args:
            symbol: Stock/ETF symbol to collect data for

        Returns:
            Dictionary containing market data or None if failed
        """
        logger.info(f"Collecting real-time data for {symbol}")
        
        if not settings.use_real_data:
            return self._generate_dummy_data(symbol)
        
        return self._collect_yfinance_data(symbol)
    
    def _collect_yfinance_data(self, symbol: str) -> Optional[Dict]:
        """Collect data using Yahoo Finance API (direct or yfinance fallback)."""
        for attempt in range(self.retry_attempts):
            try:
                # Try direct Yahoo Finance API first (more reliable)
                result = self._collect_yahoo_direct(symbol)
                if result:
                    return result
                
                # Fallback to yfinance library
                ticker = yf.Ticker(symbol)
                
                # Get 15-minute data for the last 5 days
                hist = ticker.history(period="5d", interval="15m")
                
                if hist.empty:
                    logger.warning(f"No data returned for {symbol}")
                    # If all attempts fail, return dummy data
                    if attempt == self.retry_attempts - 1:
                        logger.info(f"Falling back to dummy data for {symbol}")
                        return self._generate_dummy_data(symbol)
                    continue
                
                # Get latest data point
                latest = hist.iloc[-1]
                
                # Try to get current info (may fail due to rate limiting)
                try:
                    info = ticker.info
                    current_price = info.get('currentPrice', latest['Close'])
                    market_cap = info.get('marketCap')
                    pe_ratio = info.get('forwardPE')
                except:
                    current_price = latest['Close']
                    market_cap = None
                    pe_ratio = None
                
                return {
                    "symbol": symbol,
                    "status": "success",
                    "price": round(float(current_price), 2),
                    "volume": int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                    "open": round(float(latest['Open']), 2),
                    "high": round(float(latest['High']), 2),
                    "low": round(float(latest['Low']), 2),
                    "close": round(float(latest['Close']), 2),
                    "timestamp": datetime.now().isoformat(),
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "data_source": "yfinance"
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to collect data for {symbol} after {self.retry_attempts} attempts")
                    return self._generate_dummy_data(symbol)
        
        return None
    
    def _collect_yahoo_direct(self, symbol: str) -> Optional[Dict]:
        """Collect data directly from Yahoo Finance API (bypassing yfinance)."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract market data
            result = data['chart']['result'][0]
            meta = result['meta']
            
            # Get the latest quote
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            if timestamps and quotes['close']:
                # Get latest data point
                latest_idx = -1
                latest_time = timestamps[latest_idx]
                
                # Handle None values
                close_price = quotes['close'][latest_idx]
                volume = quotes['volume'][latest_idx] if quotes['volume'] and quotes['volume'][latest_idx] else 0
                open_price = quotes['open'][latest_idx] if quotes['open'] else close_price
                high_price = quotes['high'][latest_idx] if quotes['high'] else close_price
                low_price = quotes['low'][latest_idx] if quotes['low'] else close_price
                
                return {
                    "symbol": symbol,
                    "status": "success",
                    "price": round(float(close_price), 2),
                    "volume": int(volume),
                    "open": round(float(open_price), 2),
                    "high": round(float(high_price), 2),
                    "low": round(float(low_price), 2),
                    "close": round(float(close_price), 2),
                    "timestamp": datetime.fromtimestamp(latest_time).isoformat(),
                    "market_cap": meta.get('marketCap'),
                    "pe_ratio": None,  # Not available in chart API
                    "data_source": "yahoo_direct"
                }
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Yahoo direct API failed for {symbol}: {e}")
            return None
    
    def _generate_dummy_data(self, symbol: str) -> Dict:
        """Generate dummy data for development/testing."""
        import random
        
        base_prices = {
            "SPY": 450.0, "QQQ": 380.0, "AAPL": 180.0, "MSFT": 340.0,
            "TSLA": 240.0, "GOOGL": 140.0, "AMZN": 150.0, "META": 320.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        price_variance = random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + price_variance)
        
        return {
            "symbol": symbol,
            "status": "success",
            "price": round(current_price, 2),
            "volume": random.randint(500000, 5000000),
            "open": round(current_price * 0.99, 2),
            "high": round(current_price * 1.02, 2),
            "low": round(current_price * 0.98, 2),
            "close": round(current_price, 2),
            "timestamp": datetime.now().isoformat(),
            "market_cap": random.randint(1000000000, 3000000000000),
            "pe_ratio": round(random.uniform(10, 35), 2),
            "data_source": "dummy"
        }

    def collect_historical_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Collect historical data for analysis."""
        if not settings.use_real_data:
            return self._generate_dummy_historical_data(symbol, period)
            
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="15m")
            return hist if not hist.empty else None
        except Exception as e:
            logger.error(f"Failed to collect historical data for {symbol}: {e}")
            return None
    
    def _generate_dummy_historical_data(self, symbol: str, period: str = "1mo"):
        """Generate dummy historical data for development/testing."""
        import random
        from datetime import datetime, timedelta
        
        # Use global pd for now - the MockDataFrame should handle this
        real_pd = pd
        
        # Determine number of data points based on period
        period_map = {
            "1d": 96,    # 15-min intervals in 1 day
            "5d": 480,   # 15-min intervals in 5 days
            "1mo": 672,  # 15-min intervals in ~1 month (28 days)
            "3mo": 2016, # 15-min intervals in ~3 months
            "6mo": 4032, # 15-min intervals in ~6 months
            "1y": 8064   # 15-min intervals in ~1 year
        }
        
        num_points = period_map.get(period, 672)  # Default to 1 month
        
        # Base prices for different symbols
        base_prices = {
            "SPY": 450.0, "QQQ": 380.0, "AAPL": 180.0, "MSFT": 340.0,
            "TSLA": 240.0, "GOOGL": 140.0, "AMZN": 150.0, "META": 320.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate time series
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15 * num_points)
        timestamps = real_pd.date_range(start=start_time, end=end_time, freq='15min')
        
        # Generate price data with some trend and volatility
        prices = []
        current_price = base_price
        
        for i in range(len(timestamps)):
            # Add some random walk with slight upward bias
            change = random.uniform(-0.02, 0.025)  # Slight positive bias
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Create OHLC data
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC from the base price
            volatility = random.uniform(0.005, 0.02)  # 0.5% to 2% volatility
            
            open_price = price * (1 + random.uniform(-volatility/2, volatility/2))
            close_price = price * (1 + random.uniform(-volatility/2, volatility/2))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))
            volume = random.randint(100000, 2000000)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        return real_pd.DataFrame(data, index=timestamps[:len(data)])


class NewsCollector:
    """Collects and analyzes news sentiment."""

    def __init__(self):
        """Initialize news collector."""
        self.newsapi_client = None
        self.sentiment_analyzer = None
        
        if settings.use_real_data and settings.newsapi_key and NEWSAPI_AVAILABLE:
            try:
                self.newsapi_client = NewsApiClient(api_key=settings.newsapi_key)
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI client: {e}")
                self.newsapi_client = None
            
        # Initialize sentiment analyzer (prefer transformers, fall back to textblob)
        self.sentiment_analyzer = None
        self.sentiment_method = "dummy"
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
                self.sentiment_method = "finbert"
                logger.info("Using FinBERT for sentiment analysis")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}, falling back to TextBlob")
        
        if not self.sentiment_analyzer and TEXTBLOB_AVAILABLE:
            self.sentiment_method = "textblob"
            logger.info("Using TextBlob for sentiment analysis")

    def collect_financial_news(self) -> List[Dict]:
        """Collect financial news articles.

        Returns:
            List of news articles
        """
        logger.info("Collecting financial news")
        
        if not settings.use_real_data or not self.newsapi_client:
            return self._generate_dummy_news()
        
        return self._collect_newsapi_data()
    
    def _collect_newsapi_data(self) -> List[Dict]:
        """Collect news using NewsAPI."""
        try:
            # Financial keywords for better relevance
            query = "stocks OR market OR trading OR finance OR economy OR investment"
            
            # Get news from the last 24 hours
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            articles = self.newsapi_client.get_everything(
                q=query,
                language='en',
                sort_by='popularity',
                from_param=from_date,
                page_size=min(settings.max_news_articles, 50)
            )
            
            news_data = []
            for article in articles['articles'][:settings.max_news_articles]:
                if article.get('title') and article.get('description'):
                    news_data.append({
                        "title": article['title'],
                        "content": article.get('description', ''),
                        "full_content": article.get('content', ''),
                        "url": article.get('url'),
                        "source": article.get('source', {}).get('name', 'Unknown'),
                        "published_at": article.get('publishedAt'),
                        "timestamp": datetime.now().isoformat(),
                        "data_source": "newsapi"
                    })
            
            logger.info(f"Collected {len(news_data)} articles from NewsAPI")
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to collect news from NewsAPI: {e}")
            return self._generate_dummy_news()
    
    def _generate_dummy_news(self) -> List[Dict]:
        """Generate dummy news for development/testing."""
        import random
        
        sample_news = [
            {
                "title": "Tech Stocks Rally on AI Optimism",
                "content": "Technology stocks surged today as investors showed renewed optimism about artificial intelligence developments and quarterly earnings.",
                "source": "Financial Times",
                "timestamp": datetime.now().isoformat(),
                "data_source": "dummy"
            },
            {
                "title": "Federal Reserve Holds Rates Steady",
                "content": "The Federal Reserve announced it will maintain current interest rates amid economic uncertainty and inflation concerns.",
                "source": "Reuters",
                "timestamp": datetime.now().isoformat(),
                "data_source": "dummy"
            },
            {
                "title": "Energy Sector Shows Strong Performance",
                "content": "Oil and gas companies posted solid earnings this quarter, boosting the energy sector performance significantly.",
                "source": "Bloomberg",
                "timestamp": datetime.now().isoformat(),
                "data_source": "dummy"
            },
            {
                "title": "Market Volatility Expected to Continue",
                "content": "Analysts predict continued market volatility as economic indicators remain mixed and geopolitical tensions persist.",
                "source": "Wall Street Journal",
                "timestamp": datetime.now().isoformat(),
                "data_source": "dummy"
            }
        ]
        
        num_articles = random.randint(3, len(sample_news))
        return random.sample(sample_news, num_articles)

    def analyze_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment of news articles.

        Args:
            news_data: List of news articles

        Returns:
            Sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(news_data)} articles")
        
        if not settings.use_real_data:
            return self._generate_dummy_sentiment(news_data)
        
        if self.sentiment_method == "finbert":
            return self._analyze_finbert_sentiment(news_data)
        elif self.sentiment_method == "textblob":
            return self._analyze_textblob_sentiment(news_data)
        else:
            return self._generate_dummy_sentiment(news_data)
    
    def _analyze_finbert_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using FinBERT model."""
        sentiments = []
        processed_articles = []
        
        for article in news_data:
            try:
                text = f"{article.get('title', '')} {article.get('content', '')}"
                if len(text.strip()) < 10:
                    continue
                
                # Truncate text to model's max length
                text = text[:512]
                
                result = self.sentiment_analyzer(text)
                
                # FinBERT returns positive, negative, neutral scores
                sentiment_scores = {item['label']: item['score'] for item in result[0]}
                
                # Convert to numerical sentiment (-1 to 1)
                sentiment_score = (
                    sentiment_scores.get('positive', 0) - 
                    sentiment_scores.get('negative', 0)
                )
                
                sentiments.append(sentiment_score)
                processed_articles.append({
                    **article,
                    "sentiment_score": round(sentiment_score, 3),
                    "sentiment_scores": sentiment_scores
                })
                
            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for article: {e}")
                # Use neutral sentiment as fallback
                sentiments.append(0.0)
                processed_articles.append({
                    **article,
                    "sentiment_score": 0.0
                })
        
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "status": "success",
            "average": round(average_sentiment, 3),
            "articles_analyzed": len(processed_articles),
            "sentiment_range": [min(sentiments), max(sentiments)] if sentiments else [0, 0],
            "articles": processed_articles,
            "data_source": "finbert"
        }
    
    def _analyze_textblob_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment using TextBlob (lighter alternative)."""
        sentiments = []
        processed_articles = []
        
        for article in news_data:
            try:
                text = f"{article.get('title', '')} {article.get('content', '')}"
                if len(text.strip()) < 10:
                    continue
                
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # Range: -1 to 1
                
                sentiments.append(sentiment_score)
                processed_articles.append({
                    **article,
                    "sentiment_score": round(sentiment_score, 3),
                    "sentiment_polarity": sentiment_score,
                    "sentiment_subjectivity": round(blob.sentiment.subjectivity, 3)
                })
                
            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for article: {e}")
                sentiments.append(0.0)
                processed_articles.append({
                    **article,
                    "sentiment_score": 0.0
                })
        
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "status": "success",
            "average": round(average_sentiment, 3),
            "articles_analyzed": len(processed_articles),
            "sentiment_range": [min(sentiments), max(sentiments)] if sentiments else [0, 0],
            "articles": processed_articles,
            "data_source": "textblob"
        }
    
    def _generate_dummy_sentiment(self, news_data: List[Dict]) -> Dict:
        """Generate dummy sentiment analysis for development/testing."""
        import random
        
        sentiments = []
        processed_articles = []
        
        for article in news_data:
            title = article.get("title", "")
            # Simple keyword-based sentiment
            if any(word in title.lower() for word in ["rally", "surge", "strong", "optimism", "gain"]):
                sentiment = random.uniform(0.2, 0.7)
            elif any(word in title.lower() for word in ["volatility", "uncertainty", "decline", "drop"]):
                sentiment = random.uniform(-0.5, -0.1)
            else:
                sentiment = random.uniform(-0.2, 0.2)
            
            sentiments.append(sentiment)
            processed_articles.append({
                **article,
                "sentiment_score": round(sentiment, 3)
            })
        
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "status": "success",
            "average": round(average_sentiment, 3),
            "articles_analyzed": len(processed_articles),
            "sentiment_range": [min(sentiments), max(sentiments)] if sentiments else [0, 0],
            "articles": processed_articles,
            "data_source": "dummy"
        }


class FundamentalDataCollector:
    """Collects fundamental financial data weekly."""

    def __init__(self):
        """Initialize fundamental data collector."""
        self.retry_attempts = 3
        self.retry_delay = 2

    def collect_weekly_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Collect weekly fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol to collect data for
            
        Returns:
            Dictionary containing fundamental data or None if failed
        """
        logger.info(f"Collecting fundamental data for {symbol}")
        
        if not settings.use_real_data:
            return self._generate_dummy_fundamentals(symbol)
        
        return self._collect_yfinance_fundamentals(symbol)
    
    def _collect_yfinance_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Collect fundamental data using yfinance."""
        if not YFINANCE_AVAILABLE:
            logger.error(f"yfinance not available, returning dummy fundamental data for {symbol}. Install yfinance: pip install yfinance")
            dummy_data = self._generate_dummy_fundamentals(symbol)
            dummy_data["data_source"] = "dummy_fallback"
            return dummy_data
            
        for attempt in range(self.retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get company info and financials
                info = ticker.info
                
                # Get quarterly and annual financials
                quarterly_financials = ticker.quarterly_financials
                annual_financials = ticker.financials
                
                # Get balance sheet
                quarterly_balance_sheet = ticker.quarterly_balance_sheet
                
                # Get cashflow
                quarterly_cashflow = ticker.quarterly_cashflow
                
                # Extract key fundamental metrics
                fundamental_data = {
                    "symbol": symbol,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "data_source": "yfinance",
                    
                    # Valuation metrics
                    "market_cap": info.get('marketCap'),
                    "enterprise_value": info.get('enterpriseValue'),
                    "pe_ratio": info.get('forwardPE'),
                    "pb_ratio": info.get('priceToBook'),
                    "ps_ratio": info.get('priceToSalesTrailing12Months'),
                    "peg_ratio": info.get('pegRatio'),
                    
                    # Profitability metrics
                    "profit_margins": info.get('profitMargins'),
                    "operating_margins": info.get('operatingMargins'),
                    "return_on_assets": info.get('returnOnAssets'),
                    "return_on_equity": info.get('returnOnEquity'),
                    
                    # Growth metrics
                    "revenue_growth": info.get('revenueGrowth'),
                    "earnings_growth": info.get('earningsGrowth'),
                    "earnings_quarterly_growth": info.get('earningsQuarterlyGrowth'),
                    
                    # Financial health
                    "total_cash": info.get('totalCash'),
                    "total_debt": info.get('totalDebt'),
                    "debt_to_equity": info.get('debtToEquity'),
                    "current_ratio": info.get('currentRatio'),
                    "quick_ratio": info.get('quickRatio'),
                    
                    # Dividend info
                    "dividend_yield": info.get('dividendYield'),
                    "dividend_rate": info.get('dividendRate'),
                    "payout_ratio": info.get('payoutRatio'),
                    
                    # Trading metrics
                    "beta": info.get('beta'),
                    "shares_outstanding": info.get('sharesOutstanding'),
                    "float_shares": info.get('floatShares'),
                    
                    # Recent financials summary
                    "latest_revenue": self._get_latest_financial_item(quarterly_financials, 'Total Revenue') if not quarterly_financials.empty else None,
                    "latest_earnings": self._get_latest_financial_item(quarterly_financials, 'Net Income') if not quarterly_financials.empty else None,
                }
                
                return fundamental_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} fundamentals: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to collect fundamentals for {symbol} after {self.retry_attempts} attempts")
                    return self._generate_dummy_fundamentals(symbol)
        
        return None
    
    def _get_latest_financial_item(self, financials, item_name: str):
        """Extract latest financial item from DataFrame."""
        try:
            if item_name in financials.index:
                return float(financials.loc[item_name].iloc[0])
        except (KeyError, IndexError, ValueError):
            pass
        return None
    
    def _generate_dummy_fundamentals(self, symbol: str) -> Dict:
        """Generate dummy fundamental data for development/testing."""
        import random
        
        # Base values for different symbol types
        base_values = {
            "SPY": {"market_cap": 400000000000, "pe_ratio": 22.5, "dividend_yield": 0.016},
            "AAPL": {"market_cap": 3500000000000, "pe_ratio": 28.5, "dividend_yield": 0.004},
            "MSFT": {"market_cap": 2800000000000, "pe_ratio": 31.2, "dividend_yield": 0.007},
            "TSLA": {"market_cap": 800000000000, "pe_ratio": 65.8, "dividend_yield": 0.0},
        }
        
        base = base_values.get(symbol, base_values["AAPL"])
        
        return {
            "symbol": symbol,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_source": "dummy",
            
            # Valuation metrics with some randomness
            "market_cap": int(base["market_cap"] * random.uniform(0.95, 1.05)),
            "pe_ratio": round(base["pe_ratio"] * random.uniform(0.9, 1.1), 2),
            "pb_ratio": round(random.uniform(2.5, 8.0), 2),
            "ps_ratio": round(random.uniform(4.0, 12.0), 2),
            "peg_ratio": round(random.uniform(1.0, 2.5), 2),
            
            # Profitability metrics
            "profit_margins": round(random.uniform(0.08, 0.25), 4),
            "operating_margins": round(random.uniform(0.12, 0.30), 4),
            "return_on_assets": round(random.uniform(0.05, 0.15), 4),
            "return_on_equity": round(random.uniform(0.15, 0.35), 4),
            
            # Growth metrics
            "revenue_growth": round(random.uniform(0.02, 0.15), 4),
            "earnings_growth": round(random.uniform(-0.05, 0.25), 4),
            "earnings_quarterly_growth": round(random.uniform(-0.10, 0.30), 4),
            
            # Financial health
            "debt_to_equity": round(random.uniform(0.2, 1.8), 2),
            "current_ratio": round(random.uniform(1.1, 2.5), 2),
            "quick_ratio": round(random.uniform(0.8, 2.0), 2),
            
            # Dividend info
            "dividend_yield": base["dividend_yield"],
            "dividend_rate": round(base["dividend_yield"] * base["market_cap"] / 1000000000, 2) if base["dividend_yield"] else 0,
            "payout_ratio": round(random.uniform(0.15, 0.60), 4) if base["dividend_yield"] else 0,
            
            # Trading metrics
            "beta": round(random.uniform(0.8, 1.8), 3),
        }


class VolatilityMonitor:
    """Monitors market volatility and triggers emergency analysis."""

    def __init__(self):
        """Initialize volatility monitor."""
        self.retry_attempts = 2
        self.retry_delay = 1
        
        # Volatility thresholds
        self.vix_high_threshold = 30.0  # VIX above 30 indicates high fear
        self.vix_extreme_threshold = 40.0  # VIX above 40 indicates extreme fear
        self.volume_spike_threshold = 2.0  # 2x normal volume
        self.price_movement_threshold = 0.05  # 5% price movement

    def check_market_volatility(self) -> Dict:
        """Check current market volatility conditions.
        
        Returns:
            Dictionary containing volatility status and metrics
        """
        logger.info("Checking market volatility conditions")
        
        if not settings.use_real_data:
            return self._generate_dummy_volatility()
        
        return self._check_real_volatility()
    
    def _check_real_volatility(self) -> Dict:
        """Check real market volatility using multiple indicators."""
        try:
            volatility_data = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "data_source": "yfinance",
                "alerts": [],
                "triggers": []
            }
            
            # Check VIX (volatility index)
            vix_data = self._get_vix_data()
            if vix_data:
                volatility_data.update(vix_data)
                
                # Check VIX thresholds
                current_vix = vix_data.get("vix_current", 0)
                if current_vix > self.vix_extreme_threshold:
                    volatility_data["alerts"].append("EXTREME_VIX")
                    volatility_data["triggers"].append("emergency_analysis")
                elif current_vix > self.vix_high_threshold:
                    volatility_data["alerts"].append("HIGH_VIX")
                    volatility_data["triggers"].append("enhanced_monitoring")
            
            # Check major indices for volume spikes and price movements
            indices = ["SPY", "QQQ", "IWM"]
            for index in indices:
                index_data = self._check_index_volatility(index)
                if index_data:
                    volatility_data[f"{index.lower()}_metrics"] = index_data
                    
                    # Check for alerts
                    if index_data.get("volume_spike", False):
                        volatility_data["alerts"].append(f"{index}_VOLUME_SPIKE")
                    
                    if index_data.get("price_movement_alert", False):
                        volatility_data["alerts"].append(f"{index}_PRICE_MOVEMENT")
                        volatility_data["triggers"].append("market_stress_analysis")
            
            # Determine overall volatility level
            volatility_data["volatility_level"] = self._determine_volatility_level(volatility_data)
            
            return volatility_data
            
        except Exception as e:
            logger.error(f"Failed to check market volatility: {e}")
            return self._generate_dummy_volatility()
    
    def _get_vix_data(self) -> Optional[Dict]:
        """Get VIX (volatility index) data."""
        if not YFINANCE_AVAILABLE:
            # Return dummy VIX data when yfinance is not available
            logger.error("yfinance not available - cannot collect real VIX data. Install yfinance: pip install yfinance")
            return {
                "vix_current": 20.5,
                "vix_5day_avg": 19.8,
                "vix_change": 0.7,
                "vix_change_pct": 3.5,
                "data_source": "dummy_fallback"
            }
            
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="5d", interval="1h")
            
            if not vix_hist.empty:
                current_vix = float(vix_hist['Close'].iloc[-1])
                vix_5day_avg = float(vix_hist['Close'].mean())
                vix_change = current_vix - float(vix_hist['Close'].iloc[-2]) if len(vix_hist) > 1 else 0
                
                return {
                    "vix_current": round(current_vix, 2),
                    "vix_5day_avg": round(vix_5day_avg, 2),
                    "vix_change": round(vix_change, 2),
                    "vix_change_pct": round((vix_change / float(vix_hist['Close'].iloc[-2])) * 100, 2) if len(vix_hist) > 1 else 0
                }
        except Exception as e:
            logger.warning(f"Failed to get VIX data: {e}")
        
        return None
    
    def _check_index_volatility(self, symbol: str) -> Optional[Dict]:
        """Check volatility metrics for a specific index."""
        if not YFINANCE_AVAILABLE:
            # Return dummy volatility data when yfinance is not available
            logger.error(f"yfinance not available, returning dummy volatility data for {symbol}. Install yfinance: pip install yfinance")
            return {
                "symbol": symbol,
                "latest_volume": 45000000,
                "avg_volume_5d": 42000000,
                "volume_ratio": 1.07,
                "volume_spike": False,
                "price_change_pct": 1.2,
                "price_movement_alert": False,
                "intraday_range_pct": 2.1,
                "high_intraday_volatility": False,
                "data_source": "dummy_fallback"
            }
            
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10d", interval="1h")
            
            if len(hist) < 20:  # Need sufficient data
                return None
            
            # Calculate metrics
            latest_volume = int(hist['Volume'].iloc[-1])
            avg_volume_5d = int(hist['Volume'].tail(120).mean())  # 5 days * 24 hours
            volume_ratio = latest_volume / avg_volume_5d if avg_volume_5d > 0 else 1
            
            latest_price = float(hist['Close'].iloc[-1])
            price_24h_ago = float(hist['Close'].iloc[-24]) if len(hist) >= 24 else latest_price
            price_change_pct = abs((latest_price - price_24h_ago) / price_24h_ago) if price_24h_ago > 0 else 0
            
            # Calculate intraday volatility (high-low range)
            latest_high = float(hist['High'].iloc[-1])
            latest_low = float(hist['Low'].iloc[-1])
            intraday_range = (latest_high - latest_low) / latest_price if latest_price > 0 else 0
            
            return {
                "symbol": symbol,
                "latest_volume": latest_volume,
                "avg_volume_5d": avg_volume_5d,
                "volume_ratio": round(volume_ratio, 2),
                "volume_spike": volume_ratio > self.volume_spike_threshold,
                "price_change_pct": round(price_change_pct * 100, 2),
                "price_movement_alert": price_change_pct > self.price_movement_threshold,
                "intraday_range_pct": round(intraday_range * 100, 2),
                "latest_price": latest_price
            }
            
        except Exception as e:
            logger.warning(f"Failed to check volatility for {symbol}: {e}")
        
        return None
    
    def _determine_volatility_level(self, volatility_data: Dict) -> str:
        """Determine overall market volatility level."""
        alerts = volatility_data.get("alerts", [])
        
        if "EXTREME_VIX" in alerts:
            return "EXTREME"
        elif "HIGH_VIX" in alerts and len(alerts) >= 3:
            return "HIGH" 
        elif len(alerts) >= 2:
            return "ELEVATED"
        elif len(alerts) >= 1:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_dummy_volatility(self) -> Dict:
        """Generate dummy volatility data for development/testing."""
        import random
        
        # Generate realistic but dummy volatility data
        current_vix = random.uniform(15.0, 45.0)
        alerts = []
        triggers = []
        
        if current_vix > self.vix_extreme_threshold:
            alerts.extend(["EXTREME_VIX", "SPY_VOLUME_SPIKE"])
            triggers.append("emergency_analysis")
        elif current_vix > self.vix_high_threshold:
            alerts.append("HIGH_VIX")
            triggers.append("enhanced_monitoring")
        
        # Random additional alerts
        if random.random() < 0.3:  # 30% chance
            alerts.append("QQQ_PRICE_MOVEMENT")
            triggers.append("market_stress_analysis")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success", 
            "data_source": "dummy",
            "vix_current": round(current_vix, 2),
            "vix_5day_avg": round(current_vix * random.uniform(0.9, 1.1), 2),
            "vix_change": round(random.uniform(-3.0, 3.0), 2),
            "vix_change_pct": round(random.uniform(-10.0, 10.0), 2),
            "alerts": alerts,
            "triggers": triggers,
            "volatility_level": self._determine_volatility_level({"alerts": alerts})
        }
