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
from dataclasses import dataclass, asdict
from enum import Enum

from config.supabase_config import get_supabase_manager, SupabaseManager

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

# Data classes for type safety
@dataclass
class UserProfile:
    """User profile data structure."""
    id: Optional[UUID] = None
    email: str = ""
    display_name: str = ""
    trading_style: TradingStyle = TradingStyle.LONG_TERM
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    investment_horizon: InvestmentHorizon = InvestmentHorizon.MEDIUM
    preferred_sectors: List[str] = None
    max_position_size: Decimal = Decimal('10.00')
    auto_rebalance: bool = False
    notification_preferences: Dict[str, Any] = None
    is_active: bool = True
    subscription_tier: str = "free"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    def __post_init__(self):
        if self.preferred_sectors is None:
            self.preferred_sectors = []
        if self.notification_preferences is None:
            self.notification_preferences = {}

@dataclass
class Portfolio:
    """Portfolio data structure."""
    id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    portfolio_name: str = ""
    portfolio_type: PortfolioType = PortfolioType.SIMULATION
    initial_balance: Decimal = Decimal('100000.00')
    current_cash: Decimal = Decimal('100000.00')
    total_value: Decimal = Decimal('100000.00')
    unrealized_pnl: Decimal = Decimal('0.00')
    realized_pnl: Decimal = Decimal('0.00')
    performance_benchmark: str = "SPY"
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_rebalance: Optional[datetime] = None

@dataclass
class Trade:
    """Trade data structure."""
    id: Optional[UUID] = None
    portfolio_id: Optional[UUID] = None
    symbol: str = ""
    trade_type: TradeType = TradeType.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal('0')
    price: Decimal = Decimal('0')
    total_amount: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')
    strategy_used: Optional[str] = None
    confidence_score: Optional[Decimal] = None
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    holding_period: Optional[timedelta] = None
    position_size_pct: Optional[Decimal] = None
    executed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

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
    
    def create_user_profile(self, profile: UserProfile) -> UUID:
        """
        Create a new user profile.
        
        Args:
            profile: UserProfile object with user information
            
        Returns:
            UUID: ID of the created user profile
            
        Raises:
            ValueError: If profile data is invalid
            Exception: If database operation fails
        """
        # Validate profile data
        self._validate_user_profile(profile)
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Generate new ID if not provided
                profile_id = profile.id or uuid4()
                
                cursor.execute("""
                    INSERT INTO user_profiles (
                        id, email, display_name, trading_style, risk_tolerance, 
                        investment_horizon, preferred_sectors, max_position_size,
                        auto_rebalance, notification_preferences, is_active, subscription_tier
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    profile_id, profile.email, profile.display_name,
                    profile.trading_style.value, profile.risk_tolerance.value,
                    profile.investment_horizon.value, profile.preferred_sectors,
                    profile.max_position_size, profile.auto_rebalance,
                    profile.notification_preferences, profile.is_active, profile.subscription_tier
                ))
                
                conn.commit()
                logger.info(f"Created user profile: {profile.email} (ID: {profile_id})")
                return profile_id
                
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    def get_user_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """
        Retrieve a user profile by ID.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            UserProfile object or None if not found
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, email, display_name, trading_style, risk_tolerance,
                           investment_horizon, preferred_sectors, max_position_size,
                           auto_rebalance, notification_preferences, is_active,
                           subscription_tier, created_at, updated_at, last_login
                    FROM user_profiles
                    WHERE id = %s AND is_active = true
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return UserProfile(
                        id=row[0], email=row[1], display_name=row[2],
                        trading_style=TradingStyle(row[3]),
                        risk_tolerance=RiskTolerance(row[4]),
                        investment_horizon=InvestmentHorizon(row[5]),
                        preferred_sectors=row[6] or [],
                        max_position_size=row[7],
                        auto_rebalance=row[8],
                        notification_preferences=row[9] or {},
                        is_active=row[10], subscription_tier=row[11],
                        created_at=row[12], updated_at=row[13], last_login=row[14]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    def update_user_profile(self, user_id: UUID, updates: Dict[str, Any]) -> bool:
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
    # PORTFOLIO MANAGEMENT
    # =====================================================================
    
    def create_portfolio(self, portfolio: Portfolio) -> UUID:
        """
        Create a new portfolio for a user.
        
        Args:
            portfolio: Portfolio object with portfolio information
            
        Returns:
            UUID: ID of the created portfolio
        """
        self._validate_portfolio(portfolio)
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                portfolio_id = portfolio.id or uuid4()
                
                cursor.execute("""
                    INSERT INTO portfolios (
                        id, user_id, portfolio_name, portfolio_type,
                        initial_balance, current_cash, total_value,
                        unrealized_pnl, realized_pnl, performance_benchmark, is_active
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    portfolio_id, portfolio.user_id, portfolio.portfolio_name,
                    portfolio.portfolio_type.value, portfolio.initial_balance,
                    portfolio.current_cash, portfolio.total_value,
                    portfolio.unrealized_pnl, portfolio.realized_pnl,
                    portfolio.performance_benchmark, portfolio.is_active
                ))
                
                conn.commit()
                logger.info(f"Created portfolio: {portfolio.portfolio_name} (ID: {portfolio_id})")
                return portfolio_id
                
        except Exception as e:
            logger.error(f"Failed to create portfolio: {e}")
            raise
    
    def get_user_portfolios(self, user_id: UUID, active_only: bool = True) -> List[Portfolio]:
        """
        Get all portfolios for a user.
        
        Args:
            user_id: UUID of the user
            active_only: Whether to return only active portfolios
            
        Returns:
            List of Portfolio objects
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, user_id, portfolio_name, portfolio_type,
                           initial_balance, current_cash, total_value,
                           unrealized_pnl, realized_pnl, performance_benchmark,
                           is_active, created_at, updated_at, last_rebalance
                    FROM portfolios
                    WHERE user_id = %s
                """
                
                params = [user_id]
                if active_only:
                    query += " AND is_active = true"
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                
                portfolios = []
                for row in cursor.fetchall():
                    portfolios.append(Portfolio(
                        id=row[0], user_id=row[1], portfolio_name=row[2],
                        portfolio_type=PortfolioType(row[3]),
                        initial_balance=row[4], current_cash=row[5], total_value=row[6],
                        unrealized_pnl=row[7], realized_pnl=row[8],
                        performance_benchmark=row[9], is_active=row[10],
                        created_at=row[11], updated_at=row[12], last_rebalance=row[13]
                    ))
                
                return portfolios
                
        except Exception as e:
            logger.error(f"Failed to get portfolios for user {user_id}: {e}")
            return []
    
    def update_portfolio_balance(self, portfolio_id: UUID, 
                               current_cash: Decimal, total_value: Decimal,
                               unrealized_pnl: Decimal, realized_pnl: Decimal) -> bool:
        """
        Update portfolio balance and P&L information.
        
        Args:
            portfolio_id: UUID of the portfolio
            current_cash: Current cash balance
            total_value: Total portfolio value
            unrealized_pnl: Unrealized profit/loss
            realized_pnl: Realized profit/loss
            
        Returns:
            bool: True if update was successful
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE portfolios
                    SET current_cash = %s, total_value = %s,
                        unrealized_pnl = %s, realized_pnl = %s, updated_at = NOW()
                    WHERE id = %s AND is_active = true
                """, (current_cash, total_value, unrealized_pnl, realized_pnl, portfolio_id))
                
                updated_rows = cursor.rowcount
                conn.commit()
                
                if updated_rows > 0:
                    logger.debug(f"Updated portfolio {portfolio_id} balance")
                    return True
                else:
                    logger.warning(f"No portfolio found for ID {portfolio_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update portfolio balance {portfolio_id}: {e}")
            return False
    
    # =====================================================================
    # TRADE MANAGEMENT
    # =====================================================================
    
    def record_trade(self, trade: Trade) -> UUID:
        """
        Record a new trade execution.
        
        Args:
            trade: Trade object with trade information
            
        Returns:
            UUID: ID of the recorded trade
        """
        self._validate_trade(trade)
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                trade_id = trade.id or uuid4()
                executed_at = trade.executed_at or datetime.utcnow()
                
                cursor.execute("""
                    INSERT INTO trades (
                        id, portfolio_id, symbol, trade_type, order_type,
                        quantity, price, total_amount, commission,
                        strategy_used, confidence_score, entry_reason,
                        exit_reason, holding_period, position_size_pct, executed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    trade_id, trade.portfolio_id, trade.symbol,
                    trade.trade_type.value, trade.order_type.value,
                    trade.quantity, trade.price, trade.total_amount, trade.commission,
                    trade.strategy_used, trade.confidence_score, trade.entry_reason,
                    trade.exit_reason, trade.holding_period, trade.position_size_pct,
                    executed_at
                ))
                
                conn.commit()
                logger.info(f"Recorded trade: {trade.symbol} {trade.trade_type.value} (ID: {trade_id})")
                return trade_id
                
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            raise
    
    def get_portfolio_trades(self, portfolio_id: UUID, 
                           symbol: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[Trade]:
        """
        Get trades for a portfolio with optional filtering.
        
        Args:
            portfolio_id: UUID of the portfolio
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of trades to return
            
        Returns:
            List of Trade objects
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, portfolio_id, symbol, trade_type, order_type,
                           quantity, price, total_amount, commission,
                           strategy_used, confidence_score, entry_reason,
                           exit_reason, holding_period, position_size_pct,
                           executed_at, created_at
                    FROM trades
                    WHERE portfolio_id = %s
                """
                
                params = [portfolio_id]
                
                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)
                
                if start_date:
                    query += " AND executed_at >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND executed_at <= %s"
                    params.append(end_date)
                
                query += " ORDER BY executed_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(query, params)
                
                trades = []
                for row in cursor.fetchall():
                    trades.append(Trade(
                        id=row[0], portfolio_id=row[1], symbol=row[2],
                        trade_type=TradeType(row[3]), order_type=OrderType(row[4]),
                        quantity=row[5], price=row[6], total_amount=row[7], commission=row[8],
                        strategy_used=row[9], confidence_score=row[10],
                        entry_reason=row[11], exit_reason=row[12], holding_period=row[13],
                        position_size_pct=row[14], executed_at=row[15], created_at=row[16]
                    ))
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to get trades for portfolio {portfolio_id}: {e}")
            return []
    
    # =====================================================================
    # PERFORMANCE METRICS
    # =====================================================================
    
    def calculate_portfolio_performance(self, portfolio_id: UUID) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a portfolio.
        
        Args:
            portfolio_id: UUID of the portfolio
            
        Returns:
            Dict containing performance metrics
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get portfolio info
                cursor.execute("""
                    SELECT initial_balance, total_value, unrealized_pnl, realized_pnl,
                           created_at
                    FROM portfolios
                    WHERE id = %s
                """, (portfolio_id,))
                
                portfolio_row = cursor.fetchone()
                if not portfolio_row:
                    return {'error': 'Portfolio not found'}
                
                initial_balance = portfolio_row[0]
                total_value = portfolio_row[1]
                unrealized_pnl = portfolio_row[2]
                realized_pnl = portfolio_row[3]
                created_at = portfolio_row[4]
                
                # Get trade statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN trade_type = 'BUY' THEN 1 END) as buy_trades,
                        COUNT(CASE WHEN trade_type = 'SELL' THEN 1 END) as sell_trades,
                        SUM(total_amount * CASE WHEN trade_type = 'BUY' THEN -1 ELSE 1 END) as net_trading_amount,
                        AVG(confidence_score) as avg_confidence
                    FROM trades
                    WHERE portfolio_id = %s
                """, (portfolio_id,))
                
                trade_stats = cursor.fetchone()
                
                # Calculate metrics
                total_return = total_value - initial_balance
                total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0
                
                days_since_creation = (datetime.utcnow() - created_at).days
                annualized_return = (total_return_pct / days_since_creation * 365) if days_since_creation > 0 else 0
                
                return {
                    'portfolio_id': portfolio_id,
                    'initial_balance': float(initial_balance),
                    'current_value': float(total_value),
                    'total_return': float(total_return),
                    'total_return_pct': float(total_return_pct),
                    'annualized_return_pct': float(annualized_return),
                    'unrealized_pnl': float(unrealized_pnl),
                    'realized_pnl': float(realized_pnl),
                    'total_trades': trade_stats[0] if trade_stats else 0,
                    'buy_trades': trade_stats[1] if trade_stats else 0,
                    'sell_trades': trade_stats[2] if trade_stats else 0,
                    'avg_confidence': float(trade_stats[4]) if trade_stats and trade_stats[4] else 0,
                    'days_active': days_since_creation,
                    'calculated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate portfolio performance {portfolio_id}: {e}")
            return {'error': str(e)}
    
    # =====================================================================
    # DATA VALIDATION
    # =====================================================================
    
    def _validate_user_profile(self, profile: UserProfile) -> None:
        """Validate user profile data."""
        if not profile.email or '@' not in profile.email:
            raise ValueError("Valid email address is required")
        
        if not profile.display_name or len(profile.display_name.strip()) < 2:
            raise ValueError("Display name must be at least 2 characters")
        
        if profile.max_position_size <= 0 or profile.max_position_size > 100:
            raise ValueError("Max position size must be between 0 and 100 percent")
    
    def _validate_portfolio(self, portfolio: Portfolio) -> None:
        """Validate portfolio data."""
        if not portfolio.user_id:
            raise ValueError("User ID is required")
        
        if not portfolio.portfolio_name or len(portfolio.portfolio_name.strip()) < 2:
            raise ValueError("Portfolio name must be at least 2 characters")
        
        if portfolio.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
    
    def _validate_trade(self, trade: Trade) -> None:
        """Validate trade data."""
        if not trade.portfolio_id:
            raise ValueError("Portfolio ID is required")
        
        if not trade.symbol or len(trade.symbol.strip()) < 1:
            raise ValueError("Symbol is required")
        
        if trade.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if trade.price <= 0:
            raise ValueError("Price must be positive")
        
        if trade.total_amount <= 0:
            raise ValueError("Total amount must be positive")
    
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
            
            # Get record counts
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                record_counts = {}
                tables = ['user_profiles', 'portfolios', 'trades', 'market_data', 
                         'performance_metrics', 'notification_settings']
                
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
    import json
    from decimal import Decimal
    
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