"""
Simplified Trading Utilities for DAG Skip Conditions

Replaces the complex dependency_manager.py with simple business logic functions
that can be used directly in Airflow BranchPythonOperators.
"""

import logging
from datetime import datetime, time
from typing import Dict, Any

logger = logging.getLogger(__name__)


def is_market_open() -> bool:
    """
    Check if the stock market is currently open.
    Uses the more complete implementation from environment_utils.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        from .environment_utils import is_market_open as env_is_market_open
        return env_is_market_open()
    except ImportError:
        logger.warning("Could not import environment_utils, using fallback logic")
        # Fallback to simple logic if environment_utils not available
        now = datetime.now().time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        is_open = market_open <= now <= market_close
        is_weekday = datetime.now().weekday() < 5
        return is_open and is_weekday


def get_current_volatility() -> float:
    """
    Get current market volatility as a simple metric.
    
    In production, this would connect to real market data APIs.
    For now, returns a simulated volatility value.
    
    Returns:
        float: Volatility percentage (0.0 to 1.0)
    """
    try:
        # Simulate volatility based on time of day
        # Higher volatility at market open/close
        hour = datetime.now().hour
        
        if 9 <= hour <= 10 or 15 <= hour <= 16:
            # Higher volatility during opening/closing hours
            volatility = 0.3
        elif 11 <= hour <= 14:
            # Lower volatility during mid-day
            volatility = 0.15
        else:
            # After hours - assume low volatility
            volatility = 0.1
            
        logger.info(f"Current volatility: {volatility}")
        return volatility
        
    except Exception as e:
        logger.warning(f"Error calculating volatility: {e}, using default")
        return 0.2  # Default moderate volatility


def safe_to_trade() -> bool:
    """
    Determine if it's safe to execute trades based on market conditions.
    
    Combines market hours and volatility checks for trading decisions.
    
    Returns:
        bool: True if safe to trade, False if should skip
    """
    try:
        # Check if market is open
        if not is_market_open():
            logger.info("Trading skipped: Market is closed")
            return False
        
        # Check volatility threshold
        volatility = get_current_volatility()
        volatility_threshold = 0.4  # Skip trading if volatility > 40%
        
        if volatility > volatility_threshold:
            logger.info(f"Trading skipped: High volatility ({volatility:.2f} > {volatility_threshold})")
            return False
        
        logger.info("Trading conditions are safe")
        return True
        
    except Exception as e:
        logger.warning(f"Error checking trading safety: {e}, defaulting to safe")
        return True  # Default to allowing trades if check fails


def should_run_analysis() -> bool:
    """
    Determine if analysis should run based on market state.
    
    Analysis can run even when market is closed, but may skip on weekends
    or during maintenance periods.
    
    Returns:
        bool: True if analysis should run, False if should skip
    """
    try:
        # Analysis can run on weekdays even when market is closed
        is_weekday = datetime.now().weekday() < 5
        
        if not is_weekday:
            logger.info("Analysis skipped: Weekend")
            return False
        
        # Skip during early morning hours (maintenance window)
        hour = datetime.now().hour
        if 1 <= hour <= 5:
            logger.info("Analysis skipped: Maintenance window (1-5 AM)")
            return False
        
        logger.info("Analysis conditions are good")
        return True
        
    except Exception as e:
        logger.warning(f"Error checking analysis conditions: {e}, defaulting to run")
        return True  # Default to running analysis if check fails


def get_data_quality_score() -> float:
    """
    Get a simple data quality score for skip conditions.
    
    In production, this would check actual data completeness,
    freshness, and accuracy metrics.
    
    Returns:
        float: Quality score (0.0 to 1.0)
    """
    try:
        # Simulate data quality based on time since market close
        now = datetime.now()
        hour = now.hour
        
        # Data quality is highest during market hours
        if 9 <= hour <= 16 and now.weekday() < 5:
            quality = 0.9
        # Good quality shortly after market close
        elif 16 <= hour <= 20 and now.weekday() < 5:
            quality = 0.8
        # Lower quality overnight and weekends
        else:
            quality = 0.6
            
        logger.info(f"Data quality score: {quality}")
        return quality
        
    except Exception as e:
        logger.warning(f"Error calculating data quality: {e}, using default")
        return 0.7  # Default moderate quality


def should_collect_data() -> bool:
    """
    Determine if data collection should run.
    
    Data collection should run regularly during weekdays,
    but may skip during maintenance windows.
    
    Returns:
        bool: True if data collection should run, False if should skip
    """
    try:
        # Check if it's a weekday
        is_weekday = datetime.now().weekday() < 5
        
        if not is_weekday:
            logger.info("Data collection skipped: Weekend")
            return False
        
        # Skip during maintenance window (1-3 AM)
        hour = datetime.now().hour
        if 1 <= hour <= 3:
            logger.info("Data collection skipped: Maintenance window (1-3 AM)")
            return False
        
        logger.info("Data collection conditions are good")
        return True
        
    except Exception as e:
        logger.warning(f"Error checking data collection conditions: {e}, defaulting to run")
        return True  # Default to running data collection if check fails


# Branch operator helper functions for Airflow
def data_collection_branch_function(**context) -> str:
    """
    Branch function for data collection DAG skip logic.
    Used with BranchPythonOperator.
    
    Returns:
        str: Next task ID to execute
    """
    if should_collect_data():
        return 'proceed_with_data_collection'
    else:
        return 'skip_data_collection'


def analysis_branch_function(**context) -> str:
    """
    Branch function for analysis DAG skip logic.
    Used with BranchPythonOperator.
    
    Returns:
        str: Next task ID to execute
    """
    if should_run_analysis():
        return 'proceed_with_analysis'
    else:
        return 'skip_analysis'


def trading_branch_function(**context) -> str:
    """
    Branch function for trading DAG skip logic.
    Used with BranchPythonOperator.
    
    Returns:
        str: Next task ID to execute
    """
    if safe_to_trade():
        return 'proceed_with_trading'
    else:
        return 'skip_trading'