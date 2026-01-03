"""
Trading Database Manager for Multi-Profile Trading Simulator
===========================================================

This module provides a high-level interface for managing trading profiles,
portfolios, trades, and performance data in the Supabase PostgreSQL database.

Features:
- User profile management (swing vs long-term trading styles)
- Portfolio creation and management
- Trade execution tracking
- Performance metrics calculation
- Market data storage and retrieval
- Notification management
- Data validation and integrity checks
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from enum import Enum

from config.supabase_config import get_supabase_manager, SupabaseManager
from src.models.validation_models import (
    UserProfile, ActiveSymbol, MarketData, TechnicalAnalysis, 
    SentimentAnalysis, TradingDecision, DagRun
)

# Configure logging
logger = logging.getLogger(__name__)

# Enums for database fields
class TradingStyle(Enum):
    SWING = "swing"
    LONG_TERM = "long_term"

class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class InvestmentHorizon(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PortfolioType(Enum):
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"

# All data models are now imported from validation_models.py

class TradingDatabaseManager:
    """
    High-level database manager for the trading simulator.
    
    Provides methods for managing users, portfolios, trades, and performance
    data with proper validation and error handling.
    """
    
    def __init__(self, supabase_manager: Optional[SupabaseManager] = None):
        """
        Initialize the trading database manager.
        
        Args:
            supabase_manager: Optional SupabaseManager instance. If None, uses global manager.
        """
        self.db_manager = supabase_manager or get_supabase_manager()
        
    # =====================================================================
    # USER PROFILE MANAGEMENT
    # =====================================================================
    
    def create_user_profile(self, profile: UserProfile) -> str:
        """
        Create a new user profile.
        
        Args:
            profile: UserProfile object with user information
            
        Returns:
            str: ID of the created user profile
            
        Raises:
            ValueError: If profile data is invalid
            Exception: If database operation fails
        """
        # Validate profile data
        self._validate_user_profile(profile)
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_profiles (
                        user_id, budget, risk_tolerance, trading_style, interested_symbols
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    profile.user_id, profile.budget, profile.risk_tolerance,
                    profile.trading_style, profile.interested_symbols
                ))
                
                conn.commit()
                logger.info(f"Created user profile: {profile.user_id}")
                return profile.user_id
                
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile by ID.
        
        Args:
            user_id: ID of the user
            
        Returns:
            UserProfile object or None if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT user_id, budget, risk_tolerance, trading_style, interested_symbols, created_at
                    FROM user_profiles
                    WHERE user_id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return UserProfile(
                        user_id=row[0], budget=row[1], risk_tolerance=row[2],
                        trading_style=row[3], interested_symbols=row[4] or [],
                        created_at=row[5]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a user profile with the given changes.
        
        Args:
            user_id: UUID of the user to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Validate updates
            valid_fields = {
                'display_name', 'trading_style', 'risk_tolerance', 'investment_horizon',
                'preferred_sectors', 'max_position_size', 'auto_rebalance',
                'notification_preferences', 'subscription_tier', 'last_login'
            }
            
            update_fields = {k: v for k, v in updates.items() if k in valid_fields}
            
            if not update_fields:
                logger.warning("No valid fields to update")
                return False
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                set_clauses = []
                values = []
                
                for field, value in update_fields.items():
                    if field in ['trading_style', 'risk_tolerance', 'investment_horizon']:
                        # Handle enum fields
                        if isinstance(value, Enum):
                            value = value.value
                    
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
                
                values.append(user_id)
                
                query = f"""
                    UPDATE user_profiles 
                    SET {', '.join(set_clauses)}, updated_at = NOW()
                    WHERE id = %s AND is_active = true
                """
                
                cursor.execute(query, values)
                updated_rows = cursor.rowcount
                conn.commit()
                
                if updated_rows > 0:
                    logger.info(f"Updated user profile {user_id}")
                    return True
                else:
                    logger.warning(f"No user profile found for ID {user_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {e}")
            return False
    
    # =====================================================================
    # ACTIVE SYMBOLS MANAGEMENT
    # =====================================================================
    
    def add_active_symbol(self, symbol: str, user_id: str) -> bool:
        """
        Add a symbol to active_symbols or update existing one.
        
        Args:
            symbol: Stock symbol to add
            user_id: User ID who requested the symbol
            
        Returns:
            bool: True if successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if symbol exists
                cursor.execute("""
                    SELECT added_by_users FROM active_symbols WHERE symbol = %s
                """, (symbol,))
                
                row = cursor.fetchone()
                if row:
                    # Update existing symbol
                    added_by_users = row[0] or []
                    if user_id not in added_by_users:
                        added_by_users.append(user_id)
                    
                    cursor.execute("""
                        UPDATE active_symbols 
                        SET added_by_users = %s, last_updated = NOW()
                        WHERE symbol = %s
                    """, (added_by_users, symbol))
                else:
                    # Insert new symbol
                    cursor.execute("""
                        INSERT INTO active_symbols (symbol, added_by_users, last_updated, is_active)
                        VALUES (%s, %s, NOW(), true)
                    """, (symbol, [user_id]))
                
                conn.commit()
                logger.info(f"Added active symbol: {symbol} for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add active symbol {symbol}: {e}")
            return False
    
    def get_active_symbols(self) -> List[str]:
        """Get all active symbols."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT symbol FROM active_symbols WHERE is_active = true
                """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get active symbols: {e}")
            return []
    
    # =====================================================================
    # MARKET DATA MANAGEMENT
    # =====================================================================
    
    def store_market_data(self, market_data: MarketData) -> int:
        """Store market data for a symbol."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO market_data (
                        run_timestamp, symbol, price, volume, open_price, high_price,
                        low_price, close_price, market_cap, pe_ratio, data_source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    market_data.run_timestamp or datetime.utcnow(),
                    market_data.symbol, market_data.price, market_data.volume,
                    market_data.open_price, market_data.high_price, market_data.low_price,
                    market_data.close_price, market_data.market_cap, market_data.pe_ratio,
                    market_data.data_source
                ))
                
                market_id = cursor.fetchone()[0]
                conn.commit()
                logger.debug(f"Stored market data for {market_data.symbol}")
                return market_id
                
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            raise
    
    def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for a symbol."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, run_timestamp, symbol, price, volume, open_price, high_price,
                           low_price, close_price, market_cap, pe_ratio, data_source, created_at
                    FROM market_data
                    WHERE symbol = %s
                    ORDER BY run_timestamp DESC
                    LIMIT 1
                """, (symbol,))
                
                row = cursor.fetchone()
                if row:
                    return MarketData(
                        id=row[0], run_timestamp=row[1], symbol=row[2], price=row[3],
                        volume=row[4], open_price=row[5], high_price=row[6], low_price=row[7],
                        close_price=row[8], market_cap=row[9], pe_ratio=row[10],
                        data_source=row[11], created_at=row[12]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    # =====================================================================
    # TECHNICAL ANALYSIS MANAGEMENT
    # =====================================================================
    
    def store_technical_analysis(self, analysis: TechnicalAnalysis) -> int:
        """Store technical analysis results."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO technical_analysis (
                        run_timestamp, symbol, rsi, macd_value, macd_signal, macd_histogram,
                        bb_upper, bb_middle, bb_lower, signal, confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    analysis.run_timestamp or datetime.utcnow(), analysis.symbol, analysis.rsi,
                    analysis.macd_value, analysis.macd_signal, analysis.macd_histogram,
                    analysis.bb_upper, analysis.bb_middle, analysis.bb_lower,
                    analysis.signal, analysis.confidence
                ))
                
                analysis_id = cursor.fetchone()[0]
                conn.commit()
                logger.debug(f"Stored technical analysis for {analysis.symbol}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"Failed to store technical analysis: {e}")
            raise
    
    # =====================================================================
    # SENTIMENT ANALYSIS MANAGEMENT
    # =====================================================================
    
    def store_sentiment_analysis(self, sentiment: SentimentAnalysis) -> int:
        """Store sentiment analysis results."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO sentiment_analysis (
                        run_timestamp, symbol, sentiment_score, sentiment_label,
                        confidence, article_count, data_source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    sentiment.run_timestamp or datetime.utcnow(), sentiment.symbol,
                    sentiment.sentiment_score, sentiment.sentiment_label,
                    sentiment.confidence, sentiment.article_count, sentiment.data_source
                ))
                
                sentiment_id = cursor.fetchone()[0]
                conn.commit()
                logger.debug(f"Stored sentiment analysis for {sentiment.symbol}")
                return sentiment_id
                
        except Exception as e:
            logger.error(f"Failed to store sentiment analysis: {e}")
            raise
    
    # =====================================================================
    # TRADING DECISIONS MANAGEMENT
    # =====================================================================
    
    def store_trading_decision(self, decision: TradingDecision) -> int:
        """Store trading decision."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_decisions (
                        run_timestamp, user_id, symbol, action, recommended_quantity,
                        recommended_price, confidence, reasoning, budget_allocated, executed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    decision.run_timestamp or datetime.utcnow(), decision.user_id, decision.symbol,
                    decision.action, decision.recommended_quantity, decision.recommended_price,
                    decision.confidence, decision.reasoning, decision.budget_allocated, decision.executed
                ))
                
                decision_id = cursor.fetchone()[0]
                conn.commit()
                logger.debug(f"Stored trading decision for {decision.user_id}: {decision.symbol}")
                return decision_id
                
        except Exception as e:
            logger.error(f"Failed to store trading decision: {e}")
            raise
    
    def get_user_trading_decisions(self, user_id: str, limit: int = 50) -> List[TradingDecision]:
        """Get trading decisions for a user."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, run_timestamp, user_id, symbol, action, recommended_quantity,
                           recommended_price, confidence, reasoning, budget_allocated,
                           executed, created_at
                    FROM trading_decisions
                    WHERE user_id = %s
                    ORDER BY run_timestamp DESC
                    LIMIT %s
                """, (user_id, limit))
                
                decisions = []
                for row in cursor.fetchall():
                    decisions.append(TradingDecision(
                        id=row[0], run_timestamp=row[1], user_id=row[2], symbol=row[3],
                        action=row[4], recommended_quantity=row[5], recommended_price=row[6],
                        confidence=row[7], reasoning=row[8], budget_allocated=row[9],
                        executed=row[10], created_at=row[11]
                    ))
                
                return decisions
                
        except Exception as e:
            logger.error(f"Failed to get trading decisions for user {user_id}: {e}")
            return []
    
    # =====================================================================
    # DAG RUN MANAGEMENT
    # =====================================================================
    
    def record_dag_run(self, dag_run: DagRun) -> bool:
        """Record DAG run execution."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO dag_runs (
                        run_timestamp, dag_status, symbols_processed, users_served, execution_time_ms
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    dag_run.run_timestamp or datetime.utcnow(), dag_run.dag_status,
                    dag_run.symbols_processed, dag_run.users_served, dag_run.execution_time_ms
                ))
                
                conn.commit()
                logger.info(f"Recorded DAG run: {dag_run.dag_status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to record DAG run: {e}")
            return False
    
    # =====================================================================
    # DATA VALIDATION
    # =====================================================================
    
    def _validate_user_profile(self, profile: UserProfile) -> None:
        """Validate user profile data using Pydantic validation."""
        # Pydantic models now handle validation automatically
        try:
            # This will trigger pydantic validation
            profile.model_validate(profile.model_dump())
        except Exception as e:
            raise ValueError(f"Invalid user profile: {e}")
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def get_database_status(self) -> Dict[str, Any]:
        """
        Get comprehensive database status and health information.
        
        Returns:
            Dict containing database health, table sizes, and usage statistics
        """
        try:
            health = self.db_manager.health_check()
            table_sizes = self.db_manager.get_table_sizes()
            
            # Get record counts for new schema tables
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                record_counts = {}
                tables = ['user_profiles', 'active_symbols', 'market_data', 
                         'technical_analysis', 'sentiment_analysis', 'trading_decisions', 'dag_runs']
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        record_counts[table] = count
                    except Exception:
                        record_counts[table] = 0
            
            return {
                'health': health,
                'table_sizes': table_sizes,
                'record_counts': record_counts,
                'status': health.get('status', 'unknown'),
                'storage_usage_pct': health.get('storage_usage', {}).get('usage_pct', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get database status: {e}")
            return {'error': str(e), 'status': 'error'}

# Factory function for easy access
def get_trading_db_manager() -> TradingDatabaseManager:
    """
    Get a TradingDatabaseManager instance.
    
    Returns:
        TradingDatabaseManager: Configured database manager
    """
    return TradingDatabaseManager()

# Example usage and testing
if __name__ == "__main__":
    
    # Test the trading database manager
    try:
        db_manager = get_trading_db_manager()
        
        print("🔍 Trading Database Manager Test")
        print("=" * 50)
        
        # Test database status
        status = db_manager.get_database_status()
        print(f"Database status: {status.get('status')}")
        print(f"Storage usage: {status.get('storage_usage_pct', 0)}%")
        
        print("\n📊 Record Counts:")
        for table, count in status.get('record_counts', {}).items():
            print(f"  {table}: {count:,} records")
        
        print(f"\n✅ Trading database manager is ready!")
        
    except Exception as e:
        print(f"❌ Trading database manager test failed: {e}")
        import traceback
        traceback.print_exc()