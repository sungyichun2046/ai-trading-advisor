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
            
            # Fundamental data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    market_cap BIGINT,
                    enterprise_value BIGINT,
                    pe_ratio DECIMAL(8,3),
                    pb_ratio DECIMAL(8,3),
                    ps_ratio DECIMAL(8,3),
                    peg_ratio DECIMAL(8,3),
                    profit_margins DECIMAL(8,6),
                    operating_margins DECIMAL(8,6),
                    return_on_assets DECIMAL(8,6),
                    return_on_equity DECIMAL(8,6),
                    revenue_growth DECIMAL(8,6),
                    earnings_growth DECIMAL(8,6),
                    earnings_quarterly_growth DECIMAL(8,6),
                    total_cash BIGINT,
                    total_debt BIGINT,
                    debt_to_equity DECIMAL(8,3),
                    current_ratio DECIMAL(8,3),
                    quick_ratio DECIMAL(8,3),
                    dividend_yield DECIMAL(8,6),
                    dividend_rate DECIMAL(10,4),
                    payout_ratio DECIMAL(8,6),
                    beta DECIMAL(8,4),
                    shares_outstanding BIGINT,
                    float_shares BIGINT,
                    latest_revenue BIGINT,
                    latest_earnings BIGINT,
                    data_source VARCHAR(20),
                    timestamp TIMESTAMP NOT NULL,
                    execution_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Volatility events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_events (
                    id SERIAL PRIMARY KEY,
                    volatility_level VARCHAR(20) NOT NULL,
                    vix_current DECIMAL(6,2),
                    vix_5day_avg DECIMAL(6,2),
                    vix_change DECIMAL(6,2),
                    vix_change_pct DECIMAL(6,2),
                    alerts TEXT[],
                    triggers TEXT[],
                    spy_metrics JSONB,
                    qqq_metrics JSONB,
                    iwm_metrics JSONB,
                    data_source VARCHAR(20),
                    timestamp TIMESTAMP NOT NULL,
                    execution_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10),
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB,
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fundamental_data_symbol_date ON fundamental_data(symbol, execution_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_volatility_events_level_date ON volatility_events(volatility_level, execution_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type_severity ON alerts(alert_type, severity);")
            
            conn.commit()
            logger.info("Database tables created successfully")

    def store_fundamental_data(self, fundamental_data: List[Dict]) -> int:
        """Store fundamental data in database.
        
        Args:
            fundamental_data: List of fundamental data records
            
        Returns:
            Number of records stored
        """
        if not fundamental_data:
            return 0
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                count = 0
                
                for data in fundamental_data:
                    if data.get("status") == "success":
                        cursor.execute("""
                            INSERT INTO fundamental_data (
                                symbol, market_cap, enterprise_value, pe_ratio, pb_ratio, ps_ratio, peg_ratio,
                                profit_margins, operating_margins, return_on_assets, return_on_equity,
                                revenue_growth, earnings_growth, earnings_quarterly_growth,
                                total_cash, total_debt, debt_to_equity, current_ratio, quick_ratio,
                                dividend_yield, dividend_rate, payout_ratio, beta,
                                shares_outstanding, float_shares, latest_revenue, latest_earnings,
                                data_source, timestamp, execution_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            data.get("symbol"),
                            data.get("market_cap"),
                            data.get("enterprise_value"),
                            data.get("pe_ratio"),
                            data.get("pb_ratio"),
                            data.get("ps_ratio"),
                            data.get("peg_ratio"),
                            data.get("profit_margins"),
                            data.get("operating_margins"),
                            data.get("return_on_assets"),
                            data.get("return_on_equity"),
                            data.get("revenue_growth"),
                            data.get("earnings_growth"),
                            data.get("earnings_quarterly_growth"),
                            data.get("total_cash"),
                            data.get("total_debt"),
                            data.get("debt_to_equity"),
                            data.get("current_ratio"),
                            data.get("quick_ratio"),
                            data.get("dividend_yield"),
                            data.get("dividend_rate"),
                            data.get("payout_ratio"),
                            data.get("beta"),
                            data.get("shares_outstanding"),
                            data.get("float_shares"),
                            data.get("latest_revenue"),
                            data.get("latest_earnings"),
                            data.get("data_source"),
                            data.get("timestamp"),
                            data.get("execution_date")
                        ))
                        count += 1
                
                conn.commit()
                logger.info(f"Successfully stored {count} fundamental data records")
                return count
                
        except Exception as e:
            logger.error(f"Failed to store fundamental data: {e}")
            return 0

    def get_fundamental_data(self, start_date, end_date, symbol: Optional[str] = None) -> List[Dict]:
        """Retrieve fundamental data within date range.
        
        Args:
            start_date: Start date for query
            end_date: End date for query
            symbol: Optional symbol filter
            
        Returns:
            List of fundamental data records
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                if symbol:
                    cursor.execute("""
                        SELECT * FROM fundamental_data 
                        WHERE symbol = %s AND execution_date BETWEEN %s AND %s
                        ORDER BY execution_date DESC, symbol
                    """, (symbol, start_date, end_date))
                else:
                    cursor.execute("""
                        SELECT * FROM fundamental_data 
                        WHERE execution_date BETWEEN %s AND %s
                        ORDER BY execution_date DESC, symbol
                    """, (start_date, end_date))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve fundamental data: {e}")
            return []

    def store_volatility_event(self, volatility_event: Dict) -> bool:
        """Store volatility event in database.
        
        Args:
            volatility_event: Volatility event data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO volatility_events (
                        volatility_level, vix_current, vix_5day_avg, vix_change, vix_change_pct,
                        alerts, triggers, spy_metrics, qqq_metrics, iwm_metrics,
                        data_source, timestamp, execution_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    volatility_event.get("volatility_level"),
                    volatility_event.get("vix_current"),
                    volatility_event.get("vix_5day_avg"),
                    volatility_event.get("vix_change"),
                    volatility_event.get("vix_change_pct"),
                    volatility_event.get("alerts", []),
                    volatility_event.get("triggers", []),
                    psycopg2.extras.Json(volatility_event.get("spy_metrics", {})),
                    psycopg2.extras.Json(volatility_event.get("qqq_metrics", {})),
                    psycopg2.extras.Json(volatility_event.get("iwm_metrics", {})),
                    volatility_event.get("data_source"),
                    volatility_event.get("timestamp"),
                    volatility_event.get("execution_date")
                ))
                
                conn.commit()
                logger.info("Volatility event stored successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store volatility event: {e}")
            return False

    def store_alerts(self, alerts: List[Dict], alert_type: str) -> int:
        """Store alerts in database.
        
        Args:
            alerts: List of alert records
            alert_type: Type of alert (e.g., 'fundamental_alerts')
            
        Returns:
            Number of alerts stored
        """
        if not alerts:
            return 0
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                count = 0
                
                for alert in alerts:
                    cursor.execute("""
                        INSERT INTO alerts (
                            symbol, alert_type, severity, message, details, timestamp, execution_date
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        alert.get("symbol"),
                        alert_type,
                        alert.get("severity", "medium"),
                        str(alert.get("alerts", [])),
                        psycopg2.extras.Json(alert),
                        alert.get("timestamp"),
                        datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat())).date()
                    ))
                    count += 1
                
                conn.commit()
                logger.info(f"Successfully stored {count} alerts")
                return count
                
        except Exception as e:
            logger.error(f"Failed to store alerts: {e}")
            return 0

    def store_analysis_results(self, analysis_results: List[Dict], analysis_type: str) -> int:
        """Store analysis results in database.
        
        Args:
            analysis_results: List of analysis results
            analysis_type: Type of analysis
            
        Returns:
            Number of records stored
        """
        if not analysis_results:
            return 0
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                count = 0
                
                for result in analysis_results:
                    cursor.execute("""
                        INSERT INTO analysis_results (
                            symbol, analysis_type, results, confidence, timestamp, execution_date
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        result.get("symbol"),
                        analysis_type,
                        psycopg2.extras.Json(result),
                        result.get("confidence"),
                        result.get("timestamp"),
                        result.get("execution_date")
                    ))
                    count += 1
                
                conn.commit()
                logger.info(f"Successfully stored {count} analysis results")
                return count
                
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
            return 0

    def cleanup_fundamental_data(self, cutoff_date) -> int:
        """Clean up fundamental data older than cutoff date.
        
        Args:
            cutoff_date: Date to delete records before
            
        Returns:
            Number of records deleted
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM fundamental_data WHERE execution_date < %s
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old fundamental data records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup fundamental data: {e}")
            return 0


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
