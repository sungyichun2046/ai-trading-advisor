from typing import List, Optional, Dict
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------
# API Validation
# ------------------------------------------------------------

# NewsAPI article
class NewsArticle(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    url: str = Field(..., pattern=r'^https?://.+')
    publishedAt: datetime
    source: Dict[str, str]  # e.g., {"id": "cnn", "name": "CNN"}

# Yahoo Finance API Response
class YahooFinanceResponse(BaseModel):
    symbol: str = Field(..., max_length=10)
    price: Decimal = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    open_price: Decimal = Field(..., gt=0)
    high_price: Decimal = Field(..., gt=0)
    low_price: Decimal = Field(..., gt=0)
    close_price: Decimal = Field(..., gt=0)
    market_cap: Optional[int] = Field(None, ge=0)
    pe_ratio: Optional[Decimal] = Field(None, gt=0)

# NewsAPI Response
class NewsAPIResponse(BaseModel):
    articles: List[NewsArticle]
    totalResults: int = Field(..., ge=0)
    status: str = Field(..., pattern="^(ok|error)$")

# FinBERT Sentiment Response
class FinBERTResponse(BaseModel):
    label: str = Field(..., pattern="^(positive|negative|neutral)$")
    score: Decimal = Field(..., ge=0, le=1)


# ------------------------------------------------------------
# DB Models
# ------------------------------------------------------------

class ActiveSymbolDB(BaseModel):
    symbol: str = Field(..., max_length=10, pattern="^[A-Z]{1,5}$")
    added_by_users: List[str] = Field(..., min_length=1)
    last_updated: datetime
    is_active: bool = True


class UserProfileDB(BaseModel):
    user_id: str = Field(..., max_length=50, pattern="^user[0-9]+$")
    budget: Decimal = Field(..., gt=0, le=Decimal('10000000'))
    risk_tolerance: str = Field(..., pattern="^(conservative|moderate|aggressive)$")
    trading_style: str = Field(..., pattern="^(swing|long_term|day_trading)$")
    interested_symbols: List[str] = Field(..., min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketDataDB(BaseModel):
    run_timestamp: datetime
    symbol: str = Field(..., max_length=10)
    price: Decimal = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    open_price: Decimal = Field(..., gt=0)
    high_price: Decimal = Field(..., gt=0)
    low_price: Decimal = Field(..., gt=0)
    close_price: Decimal = Field(..., gt=0)
    market_cap: Optional[int] = Field(None, ge=0)
    pe_ratio: Optional[Decimal] = Field(None, gt=0)
    data_source: str = Field(..., pattern="^(yahoo_finance|alpha_vantage|polygon)$")


class SentimentDataDB(BaseModel):
    run_timestamp: datetime
    symbol: Optional[str] = Field(None, max_length=10)
    sentiment_score: Decimal = Field(..., ge=-1, le=1)
    sentiment_label: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: Decimal = Field(..., ge=0, le=1)
    article_count: int = Field(..., ge=1, le=1000)
    data_source: str = Field(..., pattern="^(newsapi|finbert|textblob)$")


class TechnicalAnalysisDB(BaseModel):
    run_timestamp: datetime
    symbol: str = Field(..., max_length=10)
    rsi: Optional[Decimal] = Field(None, ge=0, le=100)
    macd_value: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bb_upper: Optional[Decimal] = Field(None, gt=0)
    bb_middle: Optional[Decimal] = Field(None, gt=0)
    bb_lower: Optional[Decimal] = Field(None, gt=0)
    signal: str = Field(..., pattern="^(buy|sell|hold)$")
    confidence: Decimal = Field(..., ge=0, le=1)


class TradingDecisionDB(BaseModel):
    run_timestamp: datetime
    user_id: str = Field(..., max_length=50)
    symbol: str = Field(..., max_length=10)
    action: str = Field(..., pattern="^(buy|sell|hold)$")
    recommended_quantity: int = Field(..., ge=0, le=100000)
    recommended_price: Decimal = Field(..., gt=0)
    confidence: Decimal = Field(..., ge=0, le=1)
    reasoning: str = Field(..., min_length=10, max_length=1000)
    budget_allocated: Decimal = Field(..., ge=0)
    executed: bool = Field(default=False)


# ------------------------------------------------------------
# Trading Database Manager Models (Converted from @dataclass)
# ------------------------------------------------------------

class UserProfile(BaseModel):
    """User profile data structure matching the schema."""
    user_id: str = Field(..., min_length=2, max_length=50)
    budget: Decimal = Field(default=Decimal('100000.00'), gt=0, le=Decimal('10000000'))
    risk_tolerance: str = Field(default="moderate", pattern="^(conservative|moderate|aggressive)$")
    trading_style: str = Field(default="long_term", pattern="^(swing|long_term)$")
    interested_symbols: List[str] = Field(default_factory=list, max_length=100)
    created_at: Optional[datetime] = None
    
    @field_validator('interested_symbols', mode='before')
    @classmethod
    def validate_symbols(cls, v):
        if v is None:
            return []
        return v


class ActiveSymbol(BaseModel):
    """Active symbol data structure."""
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]{1,10}$")
    added_by_users: List[str] = Field(default_factory=list, min_length=1)
    last_updated: Optional[datetime] = None
    is_active: bool = True
    
    @field_validator('added_by_users', mode='before')
    @classmethod
    def validate_users(cls, v):
        if v is None:
            return []
        return v


class MarketData(BaseModel):
    """Market data structure."""
    id: Optional[int] = None
    run_timestamp: Optional[datetime] = None
    symbol: str = Field(..., min_length=1, max_length=10)
    price: Decimal = Field(..., gt=0)
    volume: Optional[int] = Field(None, ge=0)
    open_price: Optional[Decimal] = Field(None, gt=0)
    high_price: Optional[Decimal] = Field(None, gt=0)
    low_price: Optional[Decimal] = Field(None, gt=0)
    close_price: Optional[Decimal] = Field(None, gt=0)
    market_cap: Optional[int] = Field(None, ge=0)
    pe_ratio: Optional[Decimal] = Field(None, gt=0)
    data_source: Optional[str] = Field(None, pattern="^(yahoo_finance|alpha_vantage|polygon|test)$")
    created_at: Optional[datetime] = None


class TechnicalAnalysis(BaseModel):
    """Technical analysis data structure."""
    id: Optional[int] = None
    run_timestamp: Optional[datetime] = None
    symbol: str = Field(..., min_length=1, max_length=10)
    rsi: Optional[Decimal] = Field(None, ge=0, le=100)
    macd_value: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bb_upper: Optional[Decimal] = Field(None, gt=0)
    bb_middle: Optional[Decimal] = Field(None, gt=0)
    bb_lower: Optional[Decimal] = Field(None, gt=0)
    signal: Optional[str] = Field(None, pattern="^(buy|sell|hold)$")
    confidence: Optional[Decimal] = Field(None, ge=0, le=1)
    created_at: Optional[datetime] = None


class SentimentAnalysis(BaseModel):
    """Sentiment analysis data structure."""
    id: Optional[int] = None
    run_timestamp: Optional[datetime] = None
    symbol: Optional[str] = Field(None, max_length=10)
    sentiment_score: Optional[Decimal] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str] = Field(None, pattern="^(positive|negative|neutral)$")
    confidence: Optional[Decimal] = Field(None, ge=0, le=1)
    article_count: Optional[int] = Field(None, ge=1, le=1000)
    data_source: Optional[str] = Field(None, pattern="^(newsapi|finbert|textblob)$")
    created_at: Optional[datetime] = None


class TradingDecision(BaseModel):
    """Trading decision data structure."""
    id: Optional[int] = None
    run_timestamp: Optional[datetime] = None
    user_id: str = Field(..., min_length=2, max_length=50)
    symbol: str = Field(..., min_length=1, max_length=10)
    action: Optional[str] = Field(None, pattern="^(buy|sell|hold)$")
    recommended_quantity: Optional[int] = Field(None, ge=0, le=100000)
    recommended_price: Optional[Decimal] = Field(None, gt=0)
    confidence: Optional[Decimal] = Field(None, ge=0, le=1)
    reasoning: Optional[str] = Field(None, min_length=10, max_length=1000)
    budget_allocated: Optional[Decimal] = Field(None, ge=0)
    executed: bool = False
    created_at: Optional[datetime] = None


class DagRun(BaseModel):
    """DAG run data structure."""
    run_timestamp: Optional[datetime] = None
    dag_status: Optional[str] = Field(None, pattern="^(success|running|failed)$")
    symbols_processed: Optional[int] = Field(None, ge=0)
    users_served: Optional[int] = Field(None, ge=0)
    execution_time_ms: Optional[int] = Field(None, ge=0)
    created_at: Optional[datetime] = None
