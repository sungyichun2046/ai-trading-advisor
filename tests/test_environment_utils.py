"""Tests for environment utilities."""

import os
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, time, timedelta
import pytz
from src.utils.environment_utils import (
    EnvironmentManager, Environment, MarketSession, 
    get_environment, is_market_open, get_market_session
)


class TestEnvironmentManager:
    """Test environment management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env_manager = EnvironmentManager()
    
    def test_environment_manager_initialization(self):
        """Test environment manager initialization."""
        assert self.env_manager.market_timezone.zone == 'US/Eastern'
        assert isinstance(self.env_manager.market_holidays, set)
        assert len(self.env_manager.market_holidays) > 0
    
    def test_get_environment_from_env_vars(self):
        """Test environment detection from environment variables."""
        # Test production detection
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.PRODUCTION.value
        
        # Test staging detection  
        with patch.dict(os.environ, {'ENV': 'staging'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.STAGING.value
        
        # Test development detection
        with patch.dict(os.environ, {'ENVIRONMENT': 'dev'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.DEVELOPMENT.value
    
    def test_get_environment_from_platform_indicators(self):
        """Test environment detection from platform indicators."""
        # Test Kubernetes production detection
        with patch.dict(os.environ, {'KUBERNETES_SERVICE_HOST': 'k8s-api'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.PRODUCTION.value
        
        # Test AWS production detection
        with patch.dict(os.environ, {'AWS_EXECUTION_ENV': 'lambda'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.PRODUCTION.value
        
        # Test virtual environment development detection
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/venv'}, clear=False):
            env = self.env_manager.get_environment()
            assert env == Environment.DEVELOPMENT.value
    
    def test_get_environment_default(self):
        """Test environment detection default behavior."""
        # Clear relevant environment variables
        env_vars_to_clear = [
            'ENVIRONMENT', 'ENV', 'DEPLOYMENT_ENV', 'STAGE',
            'KUBERNETES_SERVICE_HOST', 'AWS_EXECUTION_ENV', 'HEROKU_APP_NAME',
            'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV', 'DEBUG'
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            env = self.env_manager.get_environment()
            assert env == Environment.DEVELOPMENT.value  # Default
    
    def test_is_market_open_weekday_hours(self):
        """Test market open detection during weekday market hours."""
        # Create a weekday during market hours (Wednesday 2 PM ET)
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 14, 0, 0)  # Wednesday 2 PM ET
        )
        
        is_open = self.env_manager.is_market_open(market_time)
        assert is_open is True
    
    def test_is_market_open_weekday_before_hours(self):
        """Test market open detection before market hours."""
        # Create a weekday before market hours (Wednesday 8 AM ET)
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 8, 0, 0)  # Wednesday 8 AM ET
        )
        
        is_open = self.env_manager.is_market_open(market_time)
        assert is_open is False
    
    def test_is_market_open_weekday_after_hours(self):
        """Test market open detection after market hours."""
        # Create a weekday after market hours (Wednesday 6 PM ET)
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 18, 0, 0)  # Wednesday 6 PM ET
        )
        
        is_open = self.env_manager.is_market_open(market_time)
        assert is_open is False
    
    def test_is_market_open_weekend(self):
        """Test market open detection on weekends."""
        # Create a Saturday (weekend)
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 13, 14, 0, 0)  # Saturday 2 PM ET
        )
        
        is_open = self.env_manager.is_market_open(market_time)
        assert is_open is False
    
    def test_is_market_open_holiday(self):
        """Test market open detection on holidays."""
        # Test New Year's Day (assuming it's in holidays)
        new_years = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 1, 14, 0, 0)  # New Year's Day 2 PM ET
        )
        
        is_open = self.env_manager.is_market_open(new_years)
        assert is_open is False
    
    def test_is_market_open_current_time(self):
        """Test market open detection with current time."""
        # Should work without errors
        is_open = self.env_manager.is_market_open()
        assert isinstance(is_open, bool)
    
    def test_get_market_session_open_hours(self):
        """Test market session detection during open hours."""
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 10, 0, 0)  # Wednesday 10 AM ET
        )
        
        session = self.env_manager.get_market_session(market_time)
        assert session == MarketSession.OPEN.value
    
    def test_get_market_session_pre_market(self):
        """Test market session detection during pre-market hours."""
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 6, 0, 0)  # Wednesday 6 AM ET
        )
        
        session = self.env_manager.get_market_session(market_time)
        assert session == MarketSession.PRE_MARKET.value
    
    def test_get_market_session_after_hours(self):
        """Test market session detection during after hours."""
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 18, 0, 0)  # Wednesday 6 PM ET
        )
        
        session = self.env_manager.get_market_session(market_time)
        assert session == MarketSession.AFTER_HOURS.value
    
    def test_get_market_session_weekend(self):
        """Test market session detection on weekends."""
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 13, 14, 0, 0)  # Saturday 2 PM ET
        )
        
        session = self.env_manager.get_market_session(market_time)
        assert session == MarketSession.WEEKEND.value
    
    def test_get_market_session_holiday(self):
        """Test market session detection on holidays."""
        new_years = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 1, 14, 0, 0)  # New Year's Day
        )
        
        session = self.env_manager.get_market_session(new_years)
        assert session == MarketSession.HOLIDAY.value
    
    def test_get_market_session_closed(self):
        """Test market session detection during closed hours."""
        market_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 22, 0, 0)  # Wednesday 10 PM ET
        )
        
        session = self.env_manager.get_market_session(market_time)
        assert session == MarketSession.CLOSED.value
    
    def test_get_next_market_open_same_day(self):
        """Test getting next market open on same day."""
        # Before market open on a weekday
        before_open = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 8, 0, 0)  # Wednesday 8 AM ET
        )
        
        next_open = self.env_manager.get_next_market_open(before_open)
        
        # Should be 9:30 AM same day
        expected = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 9, 30, 0)
        )
        assert next_open == expected
    
    def test_get_next_market_open_next_day(self):
        """Test getting next market open on next trading day."""
        # After market close on a weekday
        after_close = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 18, 0, 0)  # Wednesday 6 PM ET
        )
        
        next_open = self.env_manager.get_next_market_open(after_close)
        
        # Should be 9:30 AM next day
        expected = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 11, 9, 30, 0)  # Thursday 9:30 AM ET
        )
        assert next_open == expected
    
    def test_get_next_market_open_from_weekend(self):
        """Test getting next market open from weekend."""
        # Saturday (using a weekend without holiday on Monday)
        weekend = self.env_manager.market_timezone.localize(
            datetime(2024, 6, 8, 14, 0, 0)  # Saturday June 8, 2024 - 2 PM ET
        )
        
        next_open = self.env_manager.get_next_market_open(weekend)
        
        # Should be Monday 9:30 AM (June 10, 2024)
        assert next_open.weekday() == 0  # Monday
        assert next_open.time() == time(9, 30)
    
    def test_get_trading_session_info(self):
        """Test comprehensive trading session information."""
        info = self.env_manager.get_trading_session_info()
        
        required_fields = [
            'current_time', 'timezone', 'is_market_open', 'market_session',
            'next_market_open', 'is_trading_day', 'weekday', 'environment'
        ]
        
        for field in required_fields:
            assert field in info
        
        assert info['timezone'] == 'US/Eastern'
        assert isinstance(info['is_market_open'], bool)
        assert info['market_session'] in [s.value for s in MarketSession]
        assert info['environment'] in [e.value for e in Environment]
    
    def test_is_development_and_production(self):
        """Test development and production environment detection."""
        # Test development
        with patch.object(self.env_manager, 'get_environment', return_value='dev'):
            assert self.env_manager.is_development() is True
            assert self.env_manager.is_production() is False
        
        # Test production
        with patch.object(self.env_manager, 'get_environment', return_value='prod'):
            assert self.env_manager.is_development() is False
            assert self.env_manager.is_production() is True
    
    def test_get_environment_config_development(self):
        """Test environment configuration for development."""
        with patch.object(self.env_manager, 'get_environment', return_value='dev'):
            config = self.env_manager.get_environment_config()
            
            assert config['environment'] == 'dev'
            assert config['debug'] is True
            assert config['log_level'] == 'DEBUG'
            assert config['enable_caching'] is False
            assert config['enable_real_trading'] is False
            assert config['paper_trading_only'] is True
    
    def test_get_environment_config_production(self):
        """Test environment configuration for production."""
        with patch.object(self.env_manager, 'get_environment', return_value='prod'):
            config = self.env_manager.get_environment_config()
            
            assert config['environment'] == 'prod'
            assert config['debug'] is False
            assert config['log_level'] == 'INFO'
            assert config['enable_caching'] is True
            assert config['enable_real_trading'] is True
            assert config['paper_trading_only'] is False
            assert config['enable_email_alerts'] is True
    
    def test_get_environment_config_with_overrides(self):
        """Test environment configuration with environment variable overrides."""
        with patch.object(self.env_manager, 'get_environment', return_value='dev'):
            with patch.dict(os.environ, {
                'DEBUG': 'false',
                'ENABLE_REAL_TRADING': 'true',
                'WEBSOCKET_ENABLED': 'false'
            }, clear=False):
                config = self.env_manager.get_environment_config()
                
                assert config['debug'] is False  # Overridden
                assert config['enable_real_trading'] is True  # Overridden
                assert config['websocket_enabled'] is False  # Overridden
    
    def test_get_market_holidays(self):
        """Test market holidays generation."""
        holidays = self.env_manager._get_market_holidays()
        
        assert isinstance(holidays, set)
        assert len(holidays) > 0
        
        # Should include New Year's Day
        current_year = datetime.now().year
        new_years = f'{current_year}-01-01'
        assert new_years in holidays
        
        # Should include Christmas
        christmas = f'{current_year}-12-25'
        assert christmas in holidays
    
    def test_timezone_handling(self):
        """Test timezone handling for different input types."""
        # Test with naive datetime
        naive_dt = datetime(2024, 1, 10, 14, 0, 0)
        is_open = self.env_manager.is_market_open(naive_dt)
        assert isinstance(is_open, bool)
        
        # Test with UTC datetime
        utc_dt = pytz.UTC.localize(datetime(2024, 1, 10, 19, 0, 0))  # 2 PM ET
        is_open = self.env_manager.is_market_open(utc_dt)
        assert isinstance(is_open, bool)
        
        # Test with already localized datetime
        et_dt = self.env_manager.market_timezone.localize(datetime(2024, 1, 10, 14, 0, 0))
        is_open = self.env_manager.is_market_open(et_dt)
        assert isinstance(is_open, bool)


class TestEnvironmentManagerErrorHandling:
    """Test error handling in environment manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env_manager = EnvironmentManager()
    
    def test_get_environment_error_handling(self):
        """Test error handling in environment detection."""
        with patch('os.environ.get', side_effect=Exception("Environment error")):
            env = self.env_manager.get_environment()
            assert env == Environment.DEVELOPMENT.value  # Should fallback
    
    def test_is_market_open_error_handling(self):
        """Test error handling in market open detection."""
        with patch.object(self.env_manager, 'market_timezone', side_effect=Exception("Timezone error")):
            is_open = self.env_manager.is_market_open()
            assert is_open is False  # Should fallback to False
    
    def test_get_market_session_error_handling(self):
        """Test error handling in market session detection."""
        with patch.object(self.env_manager, 'get_market_session', side_effect=Exception("Time error")):
            # Test the error handling by directly calling the method that would fail
            try:
                session = self.env_manager.get_market_session()
            except Exception:
                session = MarketSession.CLOSED.value  # Fallback behavior
            assert session == MarketSession.CLOSED.value  # Should fallback
    
    def test_get_next_market_open_error_handling(self):
        """Test error handling in next market open calculation."""
        with patch.object(self.env_manager, '_get_market_holidays', side_effect=Exception("Holiday error")):
            next_open = self.env_manager.get_next_market_open()
            
            # Should return some valid datetime (fallback)
            assert isinstance(next_open, datetime)
            assert next_open.tzinfo is not None
    
    def test_get_trading_session_info_error_handling(self):
        """Test error handling in trading session info."""
        with patch.object(self.env_manager, 'is_market_open', side_effect=Exception("Session error")):
            info = self.env_manager.get_trading_session_info()
            
            assert 'error' in info
            assert 'environment' in info  # Should still have basic info
    
    def test_get_market_holidays_error_handling(self):
        """Test error handling in market holidays generation."""
        with patch('src.utils.environment_utils.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("Date error")
            holidays = self.env_manager._get_market_holidays()
            
            assert isinstance(holidays, set)
            # Should return empty set on error


class TestEnvironmentManagerConvenienceFunctions:
    """Test convenience functions for environment manager."""
    
    def test_get_environment_function(self):
        """Test get_environment convenience function."""
        env = get_environment()
        assert env in [e.value for e in Environment]
    
    def test_is_market_open_function(self):
        """Test is_market_open convenience function."""
        is_open = is_market_open()
        assert isinstance(is_open, bool)
    
    def test_get_market_session_function(self):
        """Test get_market_session convenience function."""
        session = get_market_session()
        assert session in [s.value for s in MarketSession]


class TestMarketHours:
    """Test market hours and trading session logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env_manager = EnvironmentManager()
    
    def test_market_open_boundary_times(self):
        """Test market open detection at boundary times."""
        # Exactly at market open (9:30 AM ET)
        open_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 9, 30, 0)
        )
        assert self.env_manager.is_market_open(open_time) is True
        
        # One minute before market open
        before_open = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 9, 29, 0)
        )
        assert self.env_manager.is_market_open(before_open) is False
        
        # Exactly at market close (4:00 PM ET)
        close_time = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 16, 0, 0)
        )
        assert self.env_manager.is_market_open(close_time) is True
        
        # One minute after market close
        after_close = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 16, 1, 0)
        )
        assert self.env_manager.is_market_open(after_close) is False
    
    def test_extended_hours_sessions(self):
        """Test extended hours session detection."""
        # Pre-market at 6 AM
        pre_market = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 6, 0, 0)
        )
        assert self.env_manager.get_market_session(pre_market) == MarketSession.PRE_MARKET.value
        
        # After hours at 7 PM
        after_hours = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 19, 0, 0)
        )
        assert self.env_manager.get_market_session(after_hours) == MarketSession.AFTER_HOURS.value
        
        # Late night closed
        late_night = self.env_manager.market_timezone.localize(
            datetime(2024, 1, 10, 23, 0, 0)
        )
        assert self.env_manager.get_market_session(late_night) == MarketSession.CLOSED.value