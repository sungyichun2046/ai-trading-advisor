"""Data collectors for AI Trading Advisor."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
import requests

from src.config import settings

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
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="15m")
            return hist if not hist.empty else None
        except Exception as e:
            logger.error(f"Failed to collect historical data for {symbol}: {e}")
            return None


class NewsCollector:
    """Collects and analyzes news sentiment."""

    def __init__(self):
        """Initialize news collector."""
        self.newsapi_client = None
        self.sentiment_analyzer = None
        
        if settings.use_real_data and settings.newsapi_key:
            self.newsapi_client = NewsApiClient(api_key=settings.newsapi_key)
            
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
