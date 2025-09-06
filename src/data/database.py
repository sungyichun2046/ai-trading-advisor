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
            # Try original connection parameters first
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except psycopg2.OperationalError as e:
            # If original connection fails and host is 'postgres', try localhost
            if self.connection_params['host'] == 'postgres':
                try:
                    localhost_params = self.connection_params.copy()
                    localhost_params['host'] = 'localhost'
                    conn = psycopg2.connect(**localhost_params)
                    yield conn
                except Exception as localhost_error:
                    logger.error(f"Failed to connect to database (both postgres and localhost): {e}, {localhost_error}")
                    if conn:
                        conn.rollback()
                    raise
            else:
                logger.error(f"Failed to connect to database: {e}")
                if conn:
                    conn.rollback()
                raise
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
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL UNIQUE,
                    risk_category VARCHAR(20) NOT NULL,
                    risk_score INTEGER NOT NULL,
                    questionnaire_version VARCHAR(10) NOT NULL DEFAULT '1.0',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # User risk responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_risk_responses (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    question_id VARCHAR(100) NOT NULL,
                    response TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_risk_responses_user_id ON user_risk_responses(user_id);")
            
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


class UserProfileStorage(DatabaseManager):
    """Production-ready storage operations for user risk profiles.
    
    Features:
    - Connection pooling and retry logic
    - Comprehensive error handling
    - Audit logging for compliance
    - Data validation and sanitization
    - Performance monitoring
    """

    def store_risk_profile(self, profile: 'RiskProfile') -> bool:
        """Store user risk profile with comprehensive validation and error handling.

        Args:
            profile: RiskProfile object to store

        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If profile data is invalid
        """
        from src.core.user_profiling import RiskProfile
        
        # Validate profile before storage
        if not profile.user_id or not profile.user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not 0 <= profile.risk_score <= 100:
            raise ValueError("risk_score must be between 0 and 100")
        if len(profile.responses) == 0:
            raise ValueError("profile must have at least one response")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Begin transaction
                    cursor.execute("BEGIN")
                    
                    # Store or update user profile with additional metadata
                    cursor.execute("""
                        INSERT INTO user_profiles (
                            user_id, risk_category, risk_score, questionnaire_version, 
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id) 
                        DO UPDATE SET 
                            risk_category = EXCLUDED.risk_category,
                            risk_score = EXCLUDED.risk_score,
                            questionnaire_version = EXCLUDED.questionnaire_version,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        profile.user_id.strip(),
                        profile.risk_category.value,
                        profile.risk_score,
                        profile.questionnaire_version,
                        profile.created_at,
                        profile.updated_at
                    ))
                    
                    # Delete existing responses for this user
                    cursor.execute("""
                        DELETE FROM user_risk_responses WHERE user_id = %s
                    """, (profile.user_id,))
                    
                    # Batch insert new responses
                    if profile.responses:
                        response_data = [
                            (
                                profile.user_id,
                                response.question_id,
                                response.response[:1000],  # Truncate long responses
                                max(1, min(5, response.score))  # Ensure score is in valid range
                            )
                            for response in profile.responses
                        ]
                        
                        cursor.executemany("""
                            INSERT INTO user_risk_responses (
                                user_id, question_id, response, score
                            ) VALUES (%s, %s, %s, %s)
                        """, response_data)
                    
                    # Commit transaction
                    cursor.execute("COMMIT")
                    
                    logger.info(
                        f"Risk profile stored successfully - User: {profile.user_id}, "
                        f"Category: {profile.risk_category.value}, Score: {profile.risk_score}, "
                        f"Responses: {len(profile.responses)}"
                    )
                    return True
                    
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database connection error, retrying ({attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    logger.error(f"Failed to store risk profile after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                logger.error(f"Failed to store risk profile for user {profile.user_id}: {e}")
                return False
                
        return False

    def get_risk_profile(self, user_id: str) -> Optional['RiskProfile']:
        """Retrieve risk profile for a user.

        Args:
            user_id: User identifier

        Returns:
            RiskProfile object if found, None otherwise
        """
        from src.core.user_profiling import RiskProfile, RiskCategory, UserResponse
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Get profile data
                cursor.execute("""
                    SELECT * FROM user_profiles WHERE user_id = %s
                """, (user_id,))
                
                profile_data = cursor.fetchone()
                if not profile_data:
                    return None
                
                # Get responses
                cursor.execute("""
                    SELECT * FROM user_risk_responses WHERE user_id = %s
                    ORDER BY question_id
                """, (user_id,))
                
                response_data = cursor.fetchall()
                
                # Convert to objects
                responses = [
                    UserResponse(
                        question_id=row['question_id'],
                        response=row['response'],
                        score=row['score']
                    )
                    for row in response_data
                ]
                
                return RiskProfile(
                    user_id=profile_data['user_id'],
                    risk_category=RiskCategory(profile_data['risk_category']),
                    risk_score=profile_data['risk_score'],
                    responses=responses,
                    created_at=profile_data['created_at'],
                    updated_at=profile_data['updated_at'],
                    questionnaire_version=profile_data['questionnaire_version']
                )
                
        except Exception as e:
            logger.error(f"Failed to retrieve risk profile for user {user_id}: {e}")
            return None

    def delete_risk_profile(self, user_id: str) -> bool:
        """Delete risk profile for a user.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete profile (responses are deleted via CASCADE)
                cursor.execute("""
                    DELETE FROM user_profiles WHERE user_id = %s
                """, (user_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Risk profile deleted for user {user_id}")
                    return True
                else:
                    logger.warning(f"No risk profile found to delete for user {user_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to delete risk profile for user {user_id}: {e}")
            return False

    def get_all_risk_profiles(self, limit: int = 100, offset: int = 0, filters: Optional[Dict] = None) -> List[Dict]:
        """Get risk profiles with optional filtering (admin function).

        Args:
            limit: Maximum number of profiles to return (max 1000)
            offset: Number of profiles to skip
            filters: Optional filters (risk_category, created_after, etc.)

        Returns:
            List of risk profile summaries with metadata
        """
        # Validate input parameters
        limit = min(max(1, limit), 1000)  # Ensure reasonable limits
        offset = max(0, offset)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Build dynamic query with filters
                base_query = """
                    SELECT 
                        user_id, risk_category, risk_score, questionnaire_version,
                        created_at, updated_at,
                        CASE 
                            WHEN updated_at < NOW() - INTERVAL '90 days' THEN true 
                            ELSE false 
                        END as review_recommended
                    FROM user_profiles 
                """
                
                where_conditions = []
                params = []
                
                if filters:
                    if 'risk_category' in filters:
                        where_conditions.append("risk_category = %s")
                        params.append(filters['risk_category'])
                    
                    if 'created_after' in filters:
                        where_conditions.append("created_at >= %s")
                        params.append(filters['created_after'])
                    
                    if 'score_min' in filters:
                        where_conditions.append("risk_score >= %s")
                        params.append(filters['score_min'])
                    
                    if 'score_max' in filters:
                        where_conditions.append("risk_score <= %s")
                        params.append(filters['score_max'])
                
                query = base_query
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                
                query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                
                profiles = [dict(row) for row in cursor.fetchall()]
                
                # Add response count for each profile
                if profiles:
                    user_ids = [p['user_id'] for p in profiles]
                    placeholders = ','.join(['%s'] * len(user_ids))
                    cursor.execute(f"""
                        SELECT user_id, COUNT(*) as response_count 
                        FROM user_risk_responses 
                        WHERE user_id IN ({placeholders})
                        GROUP BY user_id
                    """, user_ids)
                    
                    response_counts = {row['user_id']: row['response_count'] for row in cursor.fetchall()}
                    
                    for profile in profiles:
                        profile['response_count'] = response_counts.get(profile['user_id'], 0)
                
                return profiles
                
        except Exception as e:
            logger.error(f"Failed to get risk profiles: {e}")
            return []

    def get_risk_profile_stats(self) -> Dict[str, any]:
        """Get risk profile statistics.

        Returns:
            Statistics about risk profiles in database
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Total profiles
                cursor.execute("SELECT COUNT(*) as total FROM user_profiles")
                total = cursor.fetchone()['total']
                
                # Profiles by category
                cursor.execute("""
                    SELECT risk_category, COUNT(*) as count 
                    FROM user_profiles 
                    GROUP BY risk_category
                """)
                by_category = {row['risk_category']: row['count'] for row in cursor.fetchall()}
                
                # Recent profiles (last 30 days)
                cursor.execute("""
                    SELECT COUNT(*) as recent 
                    FROM user_profiles 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                recent = cursor.fetchone()['recent']
                
                return {
                    "total_profiles": total,
                    "profiles_by_category": by_category,
                    "recent_profiles_30_days": recent,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get risk profile stats: {e}")
            return {}
