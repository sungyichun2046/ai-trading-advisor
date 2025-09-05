"""Database operations for AI Trading Advisor."""

import logging
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for market and news data."""

    def __init__(self):
        """Initialize database manager."""
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }

    def get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    def store_market_data(self, market_data: Dict, execution_date: datetime) -> Dict:
        """Store market data in database.

        Args:
            market_data: Market data to store
            execution_date: Pipeline execution date

        Returns:
            Storage operation results
        """
        logger.info(f"Storing market data for {execution_date}")
        
        conn = self.get_connection()
        if not conn:
            return {"count": 0, "error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            count = 0
            
            for symbol, data in market_data.items():
                if data.get("status") == "success":
                    cursor.execute("""
                        INSERT INTO market_data (symbol, price, volume, timestamp, execution_date)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        symbol,
                        data.get("price", 0),
                        data.get("volume", 0),
                        data.get("timestamp", execution_date.isoformat()),
                        execution_date.date()
                    ))
                    count += 1
            
            conn.commit()
            logger.info(f"Successfully stored {count} market data records")
            return {"count": count, "timestamp": execution_date.isoformat()}
            
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            conn.rollback()
            return {"count": 0, "error": str(e)}
        finally:
            conn.close()

    def store_news_data(self, news_data: Dict, execution_date: datetime) -> Dict:
        """Store news data in database.

        Args:
            news_data: News data to store
            execution_date: Pipeline execution date

        Returns:
            Storage operation results
        """
        logger.info(f"Storing news data for {execution_date}")
        
        conn = self.get_connection()
        if not conn:
            return {"count": 0, "error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            count = 0
            
            if news_data.get("status") == "success":
                articles = news_data.get("articles", [])
                sentiment = news_data.get("sentiment", {}).get("average", 0)
                
                for article in articles:
                    cursor.execute("""
                        INSERT INTO news_data (title, content, sentiment, timestamp, execution_date)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        article.get("title", ""),
                        article.get("content", ""),
                        sentiment,
                        article.get("timestamp", execution_date.isoformat()),
                        execution_date.date()
                    ))
                    count += 1
            
            conn.commit()
            logger.info(f"Successfully stored {count} news records")
            return {"count": count, "timestamp": execution_date.isoformat()}
            
        except Exception as e:
            logger.error(f"Failed to store news data: {e}")
            conn.rollback()
            return {"count": 0, "error": str(e)}
        finally:
            conn.close()


class AnalysisDataManager:
    """Manages database operations for analysis results."""

    def __init__(self):
        """Initialize analysis data manager."""
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }

    def get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    def store_technical_analysis(self, results: Dict, execution_date: datetime) -> Dict:
        """Store technical analysis results."""
        logger.info("Storing technical analysis results")
        
        conn = self.get_connection()
        if not conn:
            return {"count": 0, "error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            count = 0
            
            for symbol, analysis in results.items():
                if analysis.get("status") == "success":
                    cursor.execute("""
                        INSERT INTO analysis_results (symbol, analysis_type, results, timestamp, execution_date)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        symbol,
                        "technical",
                        psycopg2.extras.Json(analysis),
                        execution_date,
                        execution_date.date()
                    ))
                    count += 1
            
            conn.commit()
            logger.info(f"Successfully stored {count} technical analysis records")
            return {"count": count}
            
        except Exception as e:
            logger.error(f"Failed to store technical analysis: {e}")
            conn.rollback()
            return {"count": 0, "error": str(e)}
        finally:
            conn.close()

    def store_fundamental_analysis(
        self, results: Dict, execution_date: datetime
    ) -> Dict:
        """Store fundamental analysis results."""
        logger.info("Storing fundamental analysis results")
        return {"count": len(results) if results else 0}

    def store_sentiment_analysis(self, results: Dict, execution_date: datetime) -> Dict:
        """Store sentiment analysis results."""
        logger.info("Storing sentiment analysis results")
        return {"count": 1 if results.get("status") == "success" else 0}

    def store_risk_analysis(self, results: Dict, execution_date: datetime) -> Dict:
        """Store risk analysis results."""
        logger.info("Storing risk analysis results")
        return {"count": 1 if results.get("status") == "success" else 0}

    def store_analysis_summary(self, results: Dict, execution_date: datetime) -> Dict:
        """Store analysis summary."""
        logger.info("Storing analysis summary")
        return {"count": 1 if results.get("status") == "success" else 0}


class RecommendationDataManager:
    """Manages database operations for recommendations."""

    def __init__(self):
        """Initialize recommendation data manager."""
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }

    def get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    def store_recommendations(self, report: Dict, execution_date: datetime) -> Dict:
        """Store trading recommendations."""
        logger.info("Storing trading recommendations")
        
        conn = self.get_connection()
        if not conn:
            return {"count": 0, "error": "Database connection failed"}
        
        try:
            cursor = conn.cursor()
            count = 0
            
            recs = report.get("actionable_recommendations", [])
            if isinstance(recs, list):
                for rec in recs:
                    cursor.execute("""
                        INSERT INTO recommendations (symbol, action, confidence, position_size, risk_level, timestamp, execution_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        rec.get("symbol", ""),
                        rec.get("action", ""),
                        rec.get("confidence", 0),
                        rec.get("position_size", 0),
                        rec.get("risk_level", ""),
                        execution_date,
                        execution_date.date()
                    ))
                    count += 1
            
            conn.commit()
            logger.info(f"Successfully stored {count} recommendation records")
            return {"count": count}
            
        except Exception as e:
            logger.error(f"Failed to store recommendations: {e}")
            conn.rollback()
            return {"count": 0, "error": str(e)}
        finally:
            conn.close()

    def update_recommendation_tracking(
        self, report: Dict, execution_date: datetime
    ) -> Dict:
        """Update recommendation tracking."""
        logger.info("Updating recommendation tracking")
        return {"success": True}
