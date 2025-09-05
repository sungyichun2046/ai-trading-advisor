"""Configuration management for AI Trading Advisor."""

from typing import Optional

from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Postgres db
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int

    # Airflow
    airflow__core__executor: str
    airflow__database__sql_alchemy_conn: str
    airflow__core__fernet_key: str
    airflow__core__dags_folder: str
    airflow__core__load_examples: bool
    # Database Configuration
    database_url: str = (
        "postgresql://trader:trader_password@localhost:5432/trading_advisor"
    )

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"

    # Security Configuration
    secret_key: str = "your_secret_key_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 240

    # API Keys
    alpha_vantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None

    # Application Settings
    debug: bool = True
    log_level: str = "INFO"

    # Risk Management Settings
    max_risk_per_trade: float = 0.02  # 2%
    max_portfolio_risk: float = 0.20  # 20%
    daily_loss_limit: float = 0.06  # 6%
    max_position_size: float = 0.10  # 10%

    # Trading Settings
    min_trade_size: float = 100.0  # $100 USD
    max_trade_size: float = 50000.0  # $50,000 USD
    daily_trade_limit: int = 50  # 50 trades per day

    # Data Settings
    data_staleness_minutes: int = 5  # 5 minutes for day trading
    price_movement_threshold: float = 0.10  # 10% price movement flag

    @validator("max_risk_per_trade")
    def validate_risk_per_trade(cls, v):
        """Validate risk per trade is within acceptable bounds."""
        if not 0.001 <= v <= 0.05:  # 0.1% to 5%
            raise ValueError("Risk per trade must be between 0.1% and 5%")
        return v

    @validator("daily_loss_limit")
    def validate_daily_loss_limit(cls, v):
        """Validate daily loss limit is reasonable."""
        if not 0.01 <= v <= 0.20:  # 1% to 20%
            raise ValueError("Daily loss limit must be between 1% and 20%")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


class DatabaseConfig:
    """Database configuration helper."""

    @staticmethod
    def get_database_url() -> str:
        """Get database URL for SQLAlchemy."""
        return settings.database_url

    @staticmethod
    def get_async_database_url() -> str:
        """Get async database URL for SQLAlchemy."""
        return settings.database_url.replace("postgresql://", "postgresql+asyncpg://")


class RedisConfig:
    """Redis configuration helper."""

    @staticmethod
    def get_redis_url() -> str:
        """Get Redis URL."""
        return settings.redis_url


class APIConfig:
    """API configuration helper."""

    @staticmethod
    def get_alpha_vantage_key() -> Optional[str]:
        """Get Alpha Vantage API key."""
        return settings.alpha_vantage_api_key

    @staticmethod
    def get_polygon_key() -> Optional[str]:
        """Get Polygon API key."""
        return settings.polygon_api_key

    @staticmethod
    def validate_api_keys() -> bool:
        """Validate that required API keys are present."""
        return bool(settings.alpha_vantage_api_key or settings.polygon_api_key)


class RiskConfig:
    """Risk management configuration."""

    @staticmethod
    def get_max_risk_per_trade() -> float:
        """Get maximum risk per trade."""
        return settings.max_risk_per_trade

    @staticmethod
    def get_max_portfolio_risk() -> float:
        """Get maximum portfolio risk."""
        return settings.max_portfolio_risk

    @staticmethod
    def get_daily_loss_limit() -> float:
        """Get daily loss limit."""
        return settings.daily_loss_limit

    @staticmethod
    def validate_trade_size(size: float) -> bool:
        """Validate trade size is within limits."""
        return settings.min_trade_size <= size <= settings.max_trade_size

    @staticmethod
    def calculate_position_size(
        account_balance: float, risk_percentage: float, stop_loss_distance: float
    ) -> float:
        """Calculate position size based on risk management rules.

        Args:
            account_balance: Total account balance in USD
            risk_percentage: Risk per trade as percentage (0.01 = 1%)
            stop_loss_distance: Distance to stop loss as percentage

        Returns:
            Position size in USD

        Raises:
            ValueError: If risk percentage exceeds maximum allowed
        """
        if risk_percentage > settings.max_risk_per_trade:
            raise ValueError(
                f"Risk exceeds maximum allowed: {settings.max_risk_per_trade}"
            )

        risk_amount = account_balance * risk_percentage
        position_size = risk_amount / stop_loss_distance

        # Apply position size limits
        max_position = account_balance * settings.max_position_size
        return min(position_size, max_position, settings.max_trade_size)


# Export commonly used configs
__all__ = ["settings", "DatabaseConfig", "RedisConfig", "APIConfig", "RiskConfig"]
