"""
Data Manager Module
Consolidated data collection, processing, and storage functionality.

Configuration:
    USE_REAL_DATA (bool): **REQUIRED** Flag to toggle between real API data and dummy data
        - True: Use real APIs (yfinance, NewsAPI, etc.) with fallback to dummy data
        - False: Always use dummy data for testing/development
        - Must be explicitly set in settings.py or environment variables
        
    Environment variables for API keys:
        - NEWSAPI_KEY: Required for real news data collection when USE_REAL_DATA=True
        - POSTGRES_*: Database connection parameters
"""

import logging
import os
import time
import random
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager

import pandas as pd
import numpy as np
import requests

# Configuration
from ..config import settings

# Optional imports with fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    YFINANCE_AVAILABLE = False
    logging.error(f"Failed to import yfinance: {e}. Market data will use dummy data only.")

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError as e:
    NEWSAPI_AVAILABLE = False
    logging.error(f"Failed to import newsapi-python: {e}. News data will use dummy data only.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logging.warning(f"Failed to import transformers: {e}. Will use TextBlob or dummy sentiment analysis.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError as e:
    TEXTBLOB_AVAILABLE = False
    logging.warning(f"Failed to import textblob: {e}. Will use dummy sentiment analysis only.")

logger = logging.getLogger(__name__)


class DataManager:
    """
    Consolidated data management system for all data collection, processing, and storage.
    
    Combines functionality from:
    - src/data/collectors.py
    - src/data/processors.py  
    - src/data/technical_collectors.py
    - src/data/database.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: Optional configuration dictionary
            
        Raises:
            ValueError: If USE_REAL_DATA flag is not properly configured
        """
        self.config = config or {}
        self.retry_attempts = 3
        self.retry_delay = 1
        
        # Validate required configuration
        if not hasattr(settings, 'use_real_data'):
            raise ValueError(
                "USE_REAL_DATA flag is required in settings. "
                "Set USE_REAL_DATA=True for production or USE_REAL_DATA=False for development/testing."
            )
        
        logger.info(f"DataManager initialized with USE_REAL_DATA={settings.use_real_data}")
        
        # Database connection parameters
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        self.sentiment_method = "dummy"
        self._setup_sentiment_analyzer()
        
        # Initialize news client
        self.newsapi_client = None
        if settings.use_real_data and hasattr(settings, 'newsapi_key') and settings.newsapi_key and NEWSAPI_AVAILABLE:
            try:
                self.newsapi_client = NewsApiClient(api_key=settings.newsapi_key)
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI client: {e}")
    
    def _setup_sentiment_analyzer(self) -> None:
        """Setup sentiment analyzer with fallbacks."""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert"
                )
                self.sentiment_method = "finbert"
                logger.info("Using FinBERT for sentiment analysis")
                return
            except Exception as e:
                logger.debug(f"FinBERT not available: {e}")
        
        if TEXTBLOB_AVAILABLE:
            self.sentiment_method = "textblob"
            logger.info("Using TextBlob for sentiment analysis")
        else:
            logger.warning("No sentiment analysis libraries available, using dummy method")
    
    # Market Data Collection Methods
    def collect_market_data(self, symbols: List[str], timeframe: str = "1d", period: str = "1mo") -> Dict[str, Any]:
        """
        Collect market data for specified symbols.
        
        Args:
            symbols: List of stock symbols to collect
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            
        Returns:
            Dictionary containing market data results
        """
        logger.info(f"Collecting market data for {len(symbols)} symbols")
        
        collected_data = {}
        errors = []
        
        for symbol in symbols:
            try:
                if not settings.use_real_data:
                    data = self._generate_dummy_market_data(symbol, period)
                else:
                    data = self._collect_yfinance_data(symbol, period, timeframe)
                
                if data is not None:
                    collected_data[symbol] = data
                else:
                    errors.append(f"Failed to collect data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                errors.append(f"Error collecting data for {symbol}: {str(e)}")
        
        return {
            "status": "success" if collected_data else "failed",
            "data": collected_data,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "symbols_collected": len(collected_data),
            "total_symbols": len(symbols)
        }
    
    def _collect_yfinance_data(self, symbol: str, period: str = "1mo", interval: str = "15m") -> Optional[Dict]:
        """Collect data using Yahoo Finance API with fallbacks."""
        for attempt in range(self.retry_attempts):
            try:
                # Try direct Yahoo Finance API first
                result = self._collect_yahoo_direct(symbol)
                if result:
                    return result
                
                # Fallback to yfinance library for historical data
                if YFINANCE_AVAILABLE:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        return {
                            "symbol": symbol,
                            "status": "success",
                            "price": round(float(latest['Close']), 2),
                            "volume": int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                            "open": round(float(latest['Open']), 2),
                            "high": round(float(latest['High']), 2),
                            "low": round(float(latest['Low']), 2),
                            "close": round(float(latest['Close']), 2),
                            "timestamp": datetime.now().isoformat(),
                            "data_source": "yfinance"
                        }
                
                # If all attempts fail, return dummy data
                if attempt == self.retry_attempts - 1:
                    return self._generate_dummy_market_data(symbol)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    # On final attempt failure, return dummy data
                    return self._generate_dummy_market_data(symbol)
        
        return self._generate_dummy_market_data(symbol)
    
    def _collect_yahoo_direct(self, symbol: str) -> Optional[Dict]:
        """Collect data directly from Yahoo Finance API."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result = data['chart']['result'][0]
            meta = result['meta']
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            if timestamps and quotes['close']:
                latest_idx = -1
                close_price = quotes['close'][latest_idx]
                volume = quotes['volume'][latest_idx] if quotes['volume'] and quotes['volume'][latest_idx] else 0
                
                return {
                    "symbol": symbol,
                    "status": "success",
                    "price": round(float(close_price), 2),
                    "volume": int(volume),
                    "open": round(float(quotes['open'][latest_idx] or close_price), 2),
                    "high": round(float(quotes['high'][latest_idx] or close_price), 2),
                    "low": round(float(quotes['low'][latest_idx] or close_price), 2),
                    "close": round(float(close_price), 2),
                    "timestamp": datetime.fromtimestamp(timestamps[latest_idx]).isoformat(),
                    "market_cap": meta.get('marketCap'),
                    "data_source": "yahoo_direct"
                }
        except Exception as e:
            logger.debug(f"Yahoo direct API failed for {symbol}: {e}")
        
        return None
    
    def _generate_dummy_market_data(self, symbol: str, period: str = "1mo") -> Dict:
        """Generate dummy market data for development/testing."""
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
            "volume": random.randint(100000, 2000000),
            "open": round(current_price * random.uniform(0.99, 1.01), 2),
            "high": round(current_price * random.uniform(1.00, 1.02), 2),
            "low": round(current_price * random.uniform(0.98, 1.00), 2),
            "close": round(current_price, 2),
            "timestamp": datetime.now().isoformat(),
            "market_cap": random.randint(50000000000, 3000000000000),
            "pe_ratio": round(random.uniform(15.0, 35.0), 2),
            "data_source": "dummy"
        }
    
    def collect_fundamental_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect fundamental data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary containing fundamental data
        """
        logger.info(f"Collecting fundamental data for {len(symbols)} symbols")
        
        collected_data = []
        errors = []
        
        for symbol in symbols:
            try:
                fundamental_data = self._collect_weekly_fundamentals(symbol)
                if fundamental_data and fundamental_data.get("status") == "success":
                    collected_data.append(fundamental_data)
                else:
                    errors.append(f"Failed to collect fundamental data for {symbol}")
            except Exception as e:
                logger.error(f"Error collecting fundamental data for {symbol}: {e}")
                errors.append(f"Error for {symbol}: {str(e)}")
        
        return {
            "status": "success" if collected_data else "failed",
            "data": collected_data,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "symbols_collected": len(collected_data),
            "total_symbols": len(symbols)
        }
    
    def _collect_weekly_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Collect weekly fundamental data for a symbol."""
        if not settings.use_real_data:
            return self._generate_dummy_fundamental_data(symbol)
        
        # Try to collect real fundamental data using yfinance
        try:
            if YFINANCE_AVAILABLE:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    return {
                        "status": "success",
                        "symbol": symbol,
                        "pe_ratio": info.get('forwardPE', 20.0),
                        "pb_ratio": info.get('priceToBook', 3.0),
                        "ps_ratio": info.get('priceToSalesTrailing12Months', 2.0),
                        "debt_to_equity": info.get('debtToEquity', 0.5),
                        "profit_margins": info.get('profitMargins', 0.15),
                        "return_on_equity": info.get('returnOnEquity', 0.18),
                        "revenue_growth": info.get('revenueGrowth', 0.12),
                        "earnings_growth": info.get('earningsGrowth', 0.10),
                        "current_ratio": info.get('currentRatio', 1.5),
                        "quick_ratio": info.get('quickRatio', 1.2),
                        "timestamp": datetime.now().isoformat(),
                        "data_source": "yfinance"
                    }
        except Exception as e:
            logger.warning(f"Failed to collect real fundamental data for {symbol}: {e}")
        
        return self._generate_dummy_fundamental_data(symbol)
    
    def _generate_dummy_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Generate dummy fundamental data."""
        return {
            "status": "success",
            "symbol": symbol,
            "pe_ratio": round(random.uniform(15.0, 35.0), 2),
            "pb_ratio": round(random.uniform(1.5, 5.0), 2),
            "ps_ratio": round(random.uniform(1.0, 4.0), 2),
            "debt_to_equity": round(random.uniform(0.2, 1.5), 2),
            "profit_margins": round(random.uniform(0.05, 0.25), 3),
            "return_on_equity": round(random.uniform(0.08, 0.30), 3),
            "revenue_growth": round(random.uniform(-0.05, 0.20), 3),
            "earnings_growth": round(random.uniform(-0.10, 0.25), 3),
            "current_ratio": round(random.uniform(1.0, 3.0), 2),
            "quick_ratio": round(random.uniform(0.8, 2.5), 2),
            "timestamp": datetime.now().isoformat(),
            "data_source": "dummy"
        }
    
    def collect_sentiment_data(self, symbols: Optional[List[str]] = None, max_articles: int = 50) -> Dict[str, Any]:
        """
        Collect news data and sentiment analysis.
        
        Args:
            symbols: Optional list of symbols to filter news
            max_articles: Maximum number of articles to collect
            
        Returns:
            Dictionary containing news and sentiment data
        """
        logger.info(f"Collecting news sentiment data (max {max_articles} articles)")
        
        if not settings.use_real_data or not self.newsapi_client:
            return self._generate_dummy_news_sentiment(max_articles)
        
        try:
            articles = self._collect_newsapi_data(max_articles)
            processed_articles = []
            
            for article in articles:
                sentiment = self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                
                processed_article = {
                    "title": article.get('title', ''),
                    "content": article.get('description', ''),
                    "url": article.get('url', ''),
                    "source": article.get('source', {}).get('name', ''),
                    "published_at": article.get('publishedAt', ''),
                    "sentiment_score": sentiment['score'],
                    "sentiment_label": sentiment['label'],
                    "timestamp": datetime.now().isoformat()
                }
                processed_articles.append(processed_article)
            
            return {
                "status": "success",
                "articles": processed_articles,
                "article_count": len(processed_articles),
                "sentiment_method": self.sentiment_method,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect news sentiment: {e}")
            return self._generate_dummy_news_sentiment(max_articles)
    
    def _collect_newsapi_data(self, max_articles: int = 50) -> List[Dict]:
        """Collect news using NewsAPI."""
        keywords = ["stock market", "earnings", "economy", "federal reserve", "inflation"]
        
        all_articles = []
        for keyword in keywords:
            try:
                response = self.newsapi_client.get_everything(
                    q=keyword,
                    language='en',
                    sort_by='publishedAt',
                    page_size=max_articles // len(keywords)
                )
                
                if response.get('status') == 'ok':
                    all_articles.extend(response.get('articles', []))
                    
            except Exception as e:
                logger.warning(f"Failed to collect news for keyword '{keyword}': {e}")
        
        return all_articles[:max_articles]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not text or not text.strip():
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
        
        if self.sentiment_method == "finbert" and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {
                    "score": result['score'] if result['label'] == 'positive' else -result['score'],
                    "label": result['label'],
                    "confidence": result['score']
                }
            except Exception as e:
                logger.debug(f"FinBERT analysis failed: {e}")
        
        if self.sentiment_method == "textblob" and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
                return {
                    "score": polarity,
                    "label": label,
                    "confidence": abs(polarity)
                }
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
        
        # Dummy sentiment
        sentiment_score = random.uniform(-1.0, 1.0)
        label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        return {
            "score": sentiment_score,
            "label": label,
            "confidence": abs(sentiment_score)
        }
    
    def _generate_dummy_news_sentiment(self, max_articles: int) -> Dict[str, Any]:
        """Generate dummy news sentiment data."""
        dummy_articles = [
            {"title": "Market Outlook Remains Positive", "content": "Analysts are optimistic about market conditions"},
            {"title": "Fed Maintains Interest Rates", "content": "Federal Reserve keeps rates steady amid economic uncertainty"},
            {"title": "Tech Stocks Rally", "content": "Technology sector shows strong performance"},
            {"title": "Economic Growth Slows", "content": "GDP growth rate shows signs of deceleration"},
            {"title": "Inflation Concerns Rise", "content": "Consumer prices continue to increase"}
        ]
        
        articles = []
        for i in range(min(max_articles, len(dummy_articles) * 3)):
            base_article = dummy_articles[i % len(dummy_articles)]
            sentiment = self._analyze_sentiment(base_article['title'] + ' ' + base_article['content'])
            
            articles.append({
                "title": f"{base_article['title']} {i+1}",
                "content": base_article['content'],
                "url": f"https://example.com/article/{i+1}",
                "source": "Dummy News",
                "published_at": (datetime.now() - timedelta(hours=i)).isoformat(),
                "sentiment_score": sentiment['score'],
                "sentiment_label": sentiment['label'],
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "status": "success",
            "articles": articles,
            "article_count": len(articles),
            "sentiment_method": "dummy",
            "timestamp": datetime.now().isoformat()
        }
    
    # Database Methods
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except psycopg2.OperationalError as e:
            if self.connection_params['host'] == 'postgres':
                try:
                    localhost_params = self.connection_params.copy()
                    localhost_params['host'] = 'localhost'
                    if conn:
                        conn.close()
                    conn = psycopg2.connect(**localhost_params)
                    yield conn
                except Exception as localhost_error:
                    logger.error(f"Failed to connect to database: {e}, {localhost_error}")
                    if conn:
                        conn.rollback()
                    raise
            else:
                logger.error(f"Failed to connect to database: {e}")
                if conn:
                    conn.rollback()
                raise
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def store_market_data(self, market_data: Dict, execution_date: datetime) -> bool:
        """Store market data in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO market_data (
                        symbol, price, volume, open_price, high_price, low_price, close_price,
                        market_cap, pe_ratio, data_source, timestamp, execution_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    market_data['symbol'],
                    market_data['price'],
                    market_data.get('volume', 0),
                    market_data.get('open'),
                    market_data.get('high'),
                    market_data.get('low'),
                    market_data.get('close'),
                    market_data.get('market_cap'),
                    market_data.get('pe_ratio'),
                    market_data.get('data_source', 'unknown'),
                    market_data.get('timestamp', datetime.now()),
                    execution_date.date()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            return False
    
    def store_news_data(self, news_data: Dict, execution_date: datetime) -> bool:
        """Store news and sentiment data."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO news_data (
                        title, content, url, source, published_at, 
                        sentiment_score, sentiment_label, execution_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    news_data['title'],
                    news_data.get('content'),
                    news_data.get('url'),
                    news_data.get('source'),
                    news_data.get('published_at'),
                    news_data.get('sentiment_score'),
                    news_data.get('sentiment_label'),
                    execution_date.date()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store news data: {e}")
            return False
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Market data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        price DECIMAL(10,2) NOT NULL,
                        volume BIGINT,
                        open_price DECIMAL(10,2),
                        high_price DECIMAL(10,2),
                        low_price DECIMAL(10,2),
                        close_price DECIMAL(10,2),
                        market_cap BIGINT,
                        pe_ratio DECIMAL(6,2),
                        data_source VARCHAR(20),
                        timestamp TIMESTAMP NOT NULL,
                        execution_date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # News data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_data (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT,
                        url TEXT,
                        source VARCHAR(100),
                        published_at TIMESTAMP,
                        sentiment_score DECIMAL(4,3),
                        sentiment_label VARCHAR(20),
                        execution_date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                conn.commit()
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on data systems."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Database health
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        # Data collection health
        health_status["components"]["yfinance"] = "available" if YFINANCE_AVAILABLE else "unavailable"
        health_status["components"]["newsapi"] = "available" if self.newsapi_client else "unavailable"
        health_status["components"]["sentiment"] = self.sentiment_method
        
        return health_status


# Utility Functions
def get_data_manager(config: Optional[Dict[str, Any]] = None) -> DataManager:
    """Factory function to create DataManager instance."""
    return DataManager(config)


def validate_symbols(symbols: List[str]) -> List[str]:
    """Validate and clean symbol list."""
    return [symbol.upper().strip() for symbol in symbols if symbol.strip()]