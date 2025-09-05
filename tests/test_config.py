"""Tests for configuration module."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import (
    APIConfig,
    DatabaseConfig,
    RedisConfig,
    RiskConfig,
    Settings,
    settings,
)


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        assert settings.max_risk_per_trade == 0.02
        assert settings.max_portfolio_risk == 0.20
        assert settings.daily_loss_limit == 0.06
        assert settings.min_trade_size == 100.0
        assert settings.max_trade_size == 50000.0

    def test_invalid_risk_per_trade(self):
        """Test validation of risk per trade."""
        with pytest.raises(ValidationError):
            Settings(max_risk_per_trade=0.10)  # 10% is too high

        with pytest.raises(ValidationError):
            Settings(max_risk_per_trade=0.0001)  # 0.01% is too low

    def test_invalid_daily_loss_limit(self):
        """Test validation of daily loss limit."""
        with pytest.raises(ValidationError):
            Settings(daily_loss_limit=0.25)  # 25% is too high

        with pytest.raises(ValidationError):
            Settings(daily_loss_limit=0.005)  # 0.5% is too low


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_get_database_url(self):
        """Test getting database URL."""
        url = DatabaseConfig.get_database_url()
        assert "postgresql://" in url

    def test_get_async_database_url(self):
        """Test getting async database URL."""
        url = DatabaseConfig.get_async_database_url()
        assert "postgresql+asyncpg://" in url


class TestRedisConfig:
    """Test RedisConfig class."""

    def test_get_redis_url(self):
        """Test getting Redis URL."""
        url = RedisConfig.get_redis_url()
        assert "redis://" in url


class TestAPIConfig:
    """Test APIConfig class."""

    def test_validate_api_keys_no_keys(self):
        """Test API key validation with no keys."""
        with patch.object(settings, "alpha_vantage_api_key", None):
            with patch.object(settings, "polygon_api_key", None):
                assert not APIConfig.validate_api_keys()

    def test_validate_api_keys_with_alpha_vantage(self):
        """Test API key validation with Alpha Vantage key."""
        with patch.object(settings, "alpha_vantage_api_key", "test_key"):
            assert APIConfig.validate_api_keys()

    def test_validate_api_keys_with_polygon(self):
        """Test API key validation with Polygon key."""
        with patch.object(settings, "polygon_api_key", "test_key"):
            assert APIConfig.validate_api_keys()


class TestRiskConfig:
    """Test RiskConfig class."""

    def test_validate_trade_size_valid(self):
        """Test valid trade size validation."""
        assert RiskConfig.validate_trade_size(1000.0)
        assert RiskConfig.validate_trade_size(10000.0)

    def test_validate_trade_size_too_small(self):
        """Test trade size too small."""
        assert not RiskConfig.validate_trade_size(50.0)

    def test_validate_trade_size_too_large(self):
        """Test trade size too large."""
        assert not RiskConfig.validate_trade_size(100000.0)

    def test_calculate_position_size_normal(self):
        """Test normal position size calculation."""
        account_balance = 10000.0
        risk_percentage = 0.02  # 2%
        stop_loss_distance = 0.05  # 5%

        position_size = RiskConfig.calculate_position_size(
            account_balance, risk_percentage, stop_loss_distance
        )

        # Expected size is limited by max_position_size (10% of account)
        max_position = account_balance * settings.max_position_size  # $1000
        expected_size = min(
            10000.0 * 0.02 / 0.05, max_position
        )  # min($4000, $1000) = $1000
        assert position_size == expected_size

    def test_calculate_position_size_exceeds_risk(self):
        """Test position size calculation with excessive risk."""
        account_balance = 10000.0
        risk_percentage = 0.10  # 10% - exceeds max
        stop_loss_distance = 0.05

        with pytest.raises(ValueError, match="Risk exceeds maximum allowed"):
            RiskConfig.calculate_position_size(
                account_balance, risk_percentage, stop_loss_distance
            )

    def test_calculate_position_size_position_limit(self):
        """Test position size limited by max position size."""
        account_balance = 100000.0
        risk_percentage = 0.02
        stop_loss_distance = 0.01  # 1% - would create large position

        position_size = RiskConfig.calculate_position_size(
            account_balance, risk_percentage, stop_loss_distance
        )

        # Should be limited by max position size (10% of account)
        max_position = account_balance * settings.max_position_size
        assert position_size <= max_position

    def test_calculate_position_size_trade_limit(self):
        """Test position size limited by max trade size."""
        account_balance = 1000000.0  # $1M account
        risk_percentage = 0.02
        stop_loss_distance = 0.01

        position_size = RiskConfig.calculate_position_size(
            account_balance, risk_percentage, stop_loss_distance
        )

        # Should be limited by max trade size
        assert position_size <= settings.max_trade_size
