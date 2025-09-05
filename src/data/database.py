"""Database operations for AI Trading Advisor."""

import logging
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Base database manager with connection handling."""

    def __init__(self):
        """Initialize database manager."""
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def create_tables(self):
        """Create database tables if they don't exist."""
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
                    full_content TEXT,
                    url TEXT,
                    source VARCHAR(100),
                    published_at TIMESTAMP,
                    sentiment_score DECIMAL(4,3),
                    sentiment_scores JSONB,
                    data_source VARCHAR(20),
                    timestamp TIMESTAMP NOT NULL,
                    execution_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10),
                    analysis_type VARCHAR(50) NOT NULL,
                    results JSONB NOT NULL,
                    confidence DECIMAL(4,3),
                    timestamp TIMESTAMP NOT NULL,
                    execution_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    action VARCHAR(20) NOT NULL,
                    confidence DECIMAL(4,3) NOT NULL,
                    position_size DECIMAL(12,2),
                    risk_level VARCHAR(20),
                    reasoning TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    execution_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, execution_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_data_date ON news_data(execution_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol_type ON analysis_results(symbol, analysis_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON recommendations(symbol, execution_date);")
            
            conn.commit()
            logger.info("Database tables created successfully")


class MarketDataStorage(DatabaseManager):
    """Storage operations for market data."""

    def store_market_data(self, market_data: Dict, execution_date: datetime) -> Dict:
        """Store market data in database.

        Args:
            market_data: Market data to store
            execution_date: Pipeline execution date

        Returns:
            Storage operation results
        """
        logger.info(f"Storing market data for {execution_date}")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                count = 0
                
                for symbol, data in market_data.items():
                    if data.get("status") == "success":
                        cursor.execute("""
                            INSERT INTO market_data (
                                symbol, price, volume, open_price, high_price, 
                                low_price, close_price, market_cap, pe_ratio, 
                                data_source, timestamp, execution_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            symbol,
                            data.get("price", 0),
                            data.get("volume", 0),
                            data.get("open", 0),
                            data.get("high", 0),
                            data.get("low", 0),
                            data.get("close", 0),
                            data.get("market_cap"),
                            data.get("pe_ratio"),
                            data.get("data_source", "unknown"),
                            data.get("timestamp", execution_date.isoformat()),
                            execution_date.date()
                        ))
                        count += 1
                
                conn.commit()
                logger.info(f"Successfully stored {count} market data records")
                return {"count": count, "timestamp": execution_date.isoformat()}
                
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            return {"count": 0, "error": str(e)}

    def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Retrieve latest market data for a symbol."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (symbol, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve market data for {symbol}: {e}")
            return []


class NewsStorage(DatabaseManager):
    """Storage operations for news data."""

    def store_news_data(self, news_data: Dict, execution_date: datetime) -> Dict:
        """Store news data in database.

        Args:
            news_data: News data to store
            execution_date: Pipeline execution date

        Returns:
            Storage operation results
        """
        logger.info(f"Storing news data for {execution_date}")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                count = 0
                
                if news_data.get("status") == "success":
                    articles = news_data.get("articles", [])
                    
                    for article in articles:
                        cursor.execute("""
                            INSERT INTO news_data (
                                title, content, full_content, url, source, 
                                published_at, sentiment_score, sentiment_scores, 
                                data_source, timestamp, execution_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            article.get("title", ""),
                            article.get("content", ""),
                            article.get("full_content", ""),
                            article.get("url"),
                            article.get("source"),
                            article.get("published_at"),
                            article.get("sentiment_score"),
                            psycopg2.extras.Json(article.get("sentiment_scores", {})),
                            article.get("data_source", "unknown"),
                            article.get("timestamp", execution_date.isoformat()),
                            execution_date.date()
                        ))
                        count += 1
                
                conn.commit()
                logger.info(f"Successfully stored {count} news records")
                return {"count": count, "timestamp": execution_date.isoformat()}
                
        except Exception as e:
            logger.error(f"Failed to store news data: {e}")
            return {"count": 0, "error": str(e)}

    def get_latest_news(self, limit: int = 50) -> List[Dict]:
        """Retrieve latest news articles."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute("""
                    SELECT * FROM news_data 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve news data: {e}")
            return []


# Legacy compatibility - can be removed in future versions  
class AnalysisDataManager(DatabaseManager):
    """Legacy analysis data manager - use DatabaseManager directly."""
    pass


class RecommendationDataManager(DatabaseManager):
    """Legacy recommendation data manager - use DatabaseManager directly."""
    pass
