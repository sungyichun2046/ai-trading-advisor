"""
Environment detection and trading session management utilities.
"""

import logging
import os
import pytz
from typing import Dict, Any, Optional
from datetime import datetime, time, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


class MarketSession(Enum):
    """Market session states."""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class EnvironmentManager:
    """Environment detection and trading session management."""
    
    def __init__(self):
        self.market_timezone = pytz.timezone('US/Eastern')
        self.market_holidays = self._get_market_holidays()
    
    def get_environment(self) -> str:
        """
        Detect current environment.
        
        Returns:
            Environment name (dev/staging/prod)
        """
        try:
            # Check environment variables in order of precedence
            env_vars = ['ENVIRONMENT', 'ENV', 'DEPLOYMENT_ENV', 'STAGE']
            
            for var in env_vars:
                env_value = os.environ.get(var, '').lower()
                if env_value:
                    if env_value in ['prod', 'production']:
                        return Environment.PRODUCTION.value
                    elif env_value in ['staging', 'stage']:
                        return Environment.STAGING.value
                    elif env_value in ['dev', 'development']:
                        return Environment.DEVELOPMENT.value
            
            # Check for production indicators
            prod_indicators = [
                'KUBERNETES_SERVICE_HOST',  # Running in k8s
                'AWS_EXECUTION_ENV',        # Running in AWS
                'HEROKU_APP_NAME',          # Running on Heroku
                'VERCEL_ENV'                # Running on Vercel
            ]
            
            if any(os.environ.get(indicator) for indicator in prod_indicators):
                return Environment.PRODUCTION.value
            
            # Check for development indicators
            dev_indicators = [
                'VIRTUAL_ENV',              # Python virtual environment
                'CONDA_DEFAULT_ENV',        # Conda environment
                'DEBUG'                     # Debug mode
            ]
            
            if any(os.environ.get(indicator) for indicator in dev_indicators):
                return Environment.DEVELOPMENT.value
            
            # Default to development
            return Environment.DEVELOPMENT.value
            
        except Exception as e:
            logger.error(f"Error detecting environment: {e}")
            return Environment.DEVELOPMENT.value
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            check_time: Time to check (default: current time)
            
        Returns:
            True if market is open
        """
        try:
            if check_time is None:
                check_time = datetime.now(self.market_timezone)
            elif check_time.tzinfo is None:
                check_time = self.market_timezone.localize(check_time)
            else:
                check_time = check_time.astimezone(self.market_timezone)
            
            # Check if it's a weekday
            if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's a holiday
            date_str = check_time.strftime('%Y-%m-%d')
            if date_str in self.market_holidays:
                return False
            
            # Check market hours (9:30 AM - 4:00 PM ET)
            market_open = time(9, 30)
            market_close = time(16, 0)
            current_time = check_time.time()
            
            return market_open <= current_time <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market open status: {e}")
            return False
    
    def get_market_session(self, check_time: Optional[datetime] = None) -> str:
        """
        Get current market session state.
        
        Args:
            check_time: Time to check (default: current time)
            
        Returns:
            Market session state
        """
        try:
            if check_time is None:
                check_time = datetime.now(self.market_timezone)
            elif check_time.tzinfo is None:
                check_time = self.market_timezone.localize(check_time)
            else:
                check_time = check_time.astimezone(self.market_timezone)
            
            # Check if it's a weekend
            if check_time.weekday() >= 5:
                return MarketSession.WEEKEND.value
            
            # Check if it's a holiday
            date_str = check_time.strftime('%Y-%m-%d')
            if date_str in self.market_holidays:
                return MarketSession.HOLIDAY.value
            
            current_time = check_time.time()
            
            # Market sessions (Eastern Time)
            pre_market_start = time(4, 0)   # 4:00 AM
            market_open = time(9, 30)       # 9:30 AM
            market_close = time(16, 0)      # 4:00 PM
            after_hours_end = time(20, 0)   # 8:00 PM
            
            if pre_market_start <= current_time < market_open:
                return MarketSession.PRE_MARKET.value
            elif market_open <= current_time <= market_close:
                return MarketSession.OPEN.value
            elif market_close < current_time <= after_hours_end:
                return MarketSession.AFTER_HOURS.value
            else:
                return MarketSession.CLOSED.value
                
        except Exception as e:
            logger.error(f"Error getting market session: {e}")
            return MarketSession.CLOSED.value
    
    def get_next_market_open(self, from_time: Optional[datetime] = None) -> datetime:
        """
        Get next market open time.
        
        Args:
            from_time: Time to calculate from (default: current time)
            
        Returns:
            Next market open datetime
        """
        try:
            if from_time is None:
                from_time = datetime.now(self.market_timezone)
            elif from_time.tzinfo is None:
                from_time = self.market_timezone.localize(from_time)
            else:
                from_time = from_time.astimezone(self.market_timezone)
            
            current_date = from_time.date()
            market_open_time = time(9, 30)
            
            # Start checking from current date
            check_date = current_date
            
            for _ in range(10):  # Look up to 10 days ahead
                # If it's a weekday and not a holiday
                if check_date.weekday() < 5:  # Monday = 0, Friday = 4
                    date_str = check_date.strftime('%Y-%m-%d')
                    if date_str not in self.market_holidays:
                        next_open = datetime.combine(check_date, market_open_time)
                        next_open = self.market_timezone.localize(next_open)
                        
                        # If it's today, check if market hasn't opened yet
                        if check_date == current_date and from_time.time() >= market_open_time:
                            # Market already opened today, check next day
                            check_date += timedelta(days=1)
                            continue
                        
                        return next_open
                
                check_date += timedelta(days=1)
            
            # Fallback: return Monday 9:30 AM of next week
            days_ahead = 7 - current_date.weekday()
            next_monday = current_date + timedelta(days=days_ahead)
            return self.market_timezone.localize(datetime.combine(next_monday, market_open_time))
            
        except Exception as e:
            logger.error(f"Error calculating next market open: {e}")
            # Fallback to tomorrow 9:30 AM
            tomorrow = datetime.now(self.market_timezone) + timedelta(days=1)
            return tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
    
    def get_trading_session_info(self) -> Dict[str, Any]:
        """Get comprehensive trading session information."""
        try:
            current_time = datetime.now(self.market_timezone)
            
            return {
                'current_time': current_time.isoformat(),
                'timezone': str(self.market_timezone),
                'is_market_open': self.is_market_open(current_time),
                'market_session': self.get_market_session(current_time),
                'next_market_open': self.get_next_market_open(current_time).isoformat(),
                'is_trading_day': current_time.weekday() < 5 and 
                                current_time.strftime('%Y-%m-%d') not in self.market_holidays,
                'weekday': current_time.strftime('%A'),
                'environment': self.get_environment()
            }
            
        except Exception as e:
            logger.error(f"Error getting trading session info: {e}")
            return {
                'error': str(e),
                'environment': self.get_environment(),
                'current_time': datetime.now().isoformat()
            }
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == Environment.DEVELOPMENT.value
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == Environment.PRODUCTION.value
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration flags."""
        env = self.get_environment()
        
        config = {
            'environment': env,
            'debug': env == Environment.DEVELOPMENT.value,
            'enable_logging': True,
            'log_level': 'DEBUG' if env == Environment.DEVELOPMENT.value else 'INFO',
            'enable_caching': env != Environment.DEVELOPMENT.value,
            'enable_real_trading': env == Environment.PRODUCTION.value,
            'paper_trading_only': env != Environment.PRODUCTION.value,
            'enable_email_alerts': env == Environment.PRODUCTION.value,
            'websocket_enabled': True,
            'rate_limit_enabled': env == Environment.PRODUCTION.value
        }
        
        # Override with environment variables
        overrides = {
            'DEBUG': 'debug',
            'ENABLE_REAL_TRADING': 'enable_real_trading', 
            'PAPER_TRADING_ONLY': 'paper_trading_only',
            'ENABLE_EMAIL_ALERTS': 'enable_email_alerts',
            'WEBSOCKET_ENABLED': 'websocket_enabled'
        }
        
        for env_var, config_key in overrides.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                config[config_key] = env_value.lower() in ['true', '1', 'yes', 'on']
        
        return config
    
    def _get_market_holidays(self) -> set:
        """Get set of market holidays for current and nearby years."""
        try:
            current_year = datetime.now().year
            holidays = []
            
            # Generate holidays for a range of years (current year +/- 2)
            for year in range(current_year - 2, current_year + 3):
                year_holidays = [
                    f'{year}-01-01',  # New Year's Day
                    f'{year}-01-15',  # MLK Day (approximate)
                    f'{year}-02-19',  # Presidents Day (approximate) 
                    f'{year}-04-07',  # Good Friday (approximate)
                    f'{year}-05-27',  # Memorial Day (approximate)
                    f'{year}-07-04',  # Independence Day
                    f'{year}-09-02',  # Labor Day (approximate)
                    f'{year}-11-28',  # Thanksgiving (approximate)
                    f'{year}-12-25',  # Christmas Day
                ]
                holidays.extend(year_holidays)
            
            return set(holidays)
            
        except Exception as e:
            logger.error(f"Error getting market holidays: {e}")
            return set()


# Global environment manager instance
environment_manager = EnvironmentManager()

# Convenience functions
def get_environment() -> str:
    """Get current environment."""
    return environment_manager.get_environment()

def is_market_open(check_time: Optional[datetime] = None) -> bool:
    """Check if market is open."""
    return environment_manager.is_market_open(check_time)

def get_market_session(check_time: Optional[datetime] = None) -> str:
    """Get market session state."""
    return environment_manager.get_market_session(check_time)

def get_trading_session_info() -> Dict[str, Any]:
    """Get trading session information."""
    return environment_manager.get_trading_session_info()

def is_development() -> bool:
    """Check if in development environment."""
    return environment_manager.is_development()

def is_production() -> bool:
    """Check if in production environment."""
    return environment_manager.is_production()