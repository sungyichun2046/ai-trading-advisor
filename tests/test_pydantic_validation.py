"""
Comprehensive Pydantic Validation Tests for Trading Database Models
=================================================================

This module tests all Pydantic validation rules for trading database models
to ensure data integrity and proper validation handling.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from pydantic import ValidationError

from src.models.validation_models import (
    ActiveSymbol, UserProfile, MarketData, TechnicalAnalysis,
    SentimentAnalysis, TradingDecision, DagRun
)


class TestActiveSymbolsValidation:
    """Test validation rules for ActiveSymbol model."""
    
    def test_valid_symbol_with_users(self):
        """Test valid symbol creation with user list."""
        symbol = ActiveSymbol(
            symbol="AAPL",
            added_by_users=["user123", "user456"]
        )
        assert symbol.symbol == "AAPL"
        assert len(symbol.added_by_users) == 2
        assert symbol.is_active is True
    
    def test_invalid_symbol_format(self):
        """Test invalid symbol format (should fail regex)."""
        with pytest.raises(ValidationError) as exc_info:
            ActiveSymbol(
                symbol="invalid123",
                added_by_users=["user123"]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_empty_user_list(self):
        """Test empty user list (should fail min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ActiveSymbol(
                symbol="AAPL",
                added_by_users=[]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'too_short' for error in errors)
    
    def test_duplicate_users_in_list(self):
        """Test duplicate users in list (should pass - duplicates allowed)."""
        symbol = ActiveSymbol(
            symbol="MSFT",
            added_by_users=["user123", "user123", "user456"]
        )
        assert len(symbol.added_by_users) == 3
        assert symbol.added_by_users.count("user123") == 2
    
    def test_symbol_length_limits(self):
        """Test symbol length validation."""
        # Valid short symbol
        ActiveSymbol(symbol="A", added_by_users=["user1"])
        
        # Valid long symbol (max 10 chars)
        ActiveSymbol(symbol="BERKSHIREA", added_by_users=["user1"])
        
        # Invalid - too long
        with pytest.raises(ValidationError):
            ActiveSymbol(symbol="TOOLONGSYMBOL", added_by_users=["user1"])


class TestUserProfilesValidation:
    """Test validation rules for UserProfile model."""
    
    def test_valid_user_profile(self):
        """Test valid user profile creation."""
        profile = UserProfile(
            user_id="user123",
            budget=Decimal("10000"),
            risk_tolerance="moderate",
            trading_style="swing",
            interested_symbols=["AAPL", "MSFT"]
        )
        assert profile.user_id == "user123"
        assert profile.budget == Decimal("10000")
        assert profile.risk_tolerance == "moderate"
        assert profile.trading_style == "swing"
    
    def test_user_id_length_validation(self):
        """Test user_id length validation (current implementation is flexible)."""
        # The current UserProfile model only validates length, not format
        # Test minimum length
        with pytest.raises(ValidationError):
            UserProfile(
                user_id="u",  # Too short (min_length=2)
                budget=Decimal("10000"),
                risk_tolerance="moderate",
                trading_style="swing",
                interested_symbols=["AAPL"]
            )
        
        # Test maximum length
        with pytest.raises(ValidationError):
            UserProfile(
                user_id="u" * 51,  # Too long (max_length=50)
                budget=Decimal("10000"),
                risk_tolerance="moderate",
                trading_style="swing",
                interested_symbols=["AAPL"]
            )
    
    def test_invalid_budget_negative(self):
        """Test negative budget (should fail gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="user123",
                budget=Decimal("-1000"),
                risk_tolerance="moderate",
                trading_style="swing",
                interested_symbols=["AAPL"]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than' for error in errors)
    
    def test_invalid_budget_too_large(self):
        """Test budget exceeding maximum (should fail le constraint)."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="user123",
                budget=Decimal("20000000"),
                risk_tolerance="moderate", 
                trading_style="swing",
                interested_symbols=["AAPL"]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)
    
    def test_invalid_risk_tolerance(self):
        """Test invalid risk tolerance (should fail enum)."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="user123",
                budget=Decimal("10000"),
                risk_tolerance="risky",
                trading_style="swing",
                interested_symbols=["AAPL"]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_invalid_trading_style(self):
        """Test invalid trading style (should fail enum)."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="user123",
                budget=Decimal("10000"),
                risk_tolerance="moderate",
                trading_style="day_trading",
                interested_symbols=["AAPL"]
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_empty_symbols_list_allowed(self):
        """Test empty symbols list (should pass - defaults to empty)."""
        profile = UserProfile(
            user_id="user123",
            budget=Decimal("10000"),
            risk_tolerance="moderate",
            trading_style="swing"
        )
        assert profile.interested_symbols == []
    
    def test_too_many_symbols(self):
        """Test exceeding maximum symbols limit."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="user123",
                budget=Decimal("10000"),
                risk_tolerance="moderate",
                trading_style="swing",
                interested_symbols=["SYM" + str(i) for i in range(101)]  # 101 symbols, max is 100
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'too_long' for error in errors)


class TestMarketDataValidation:
    """Test validation rules for MarketData model."""
    
    def test_valid_market_data(self):
        """Test valid market data with all OHLC prices > 0."""
        data = MarketData(
            symbol="AAPL",
            price=Decimal("150.00"),
            volume=1000000,
            open_price=Decimal("149.50"),
            high_price=Decimal("151.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("150.00"),
            data_source="yahoo_finance"
        )
        assert data.price > 0
        assert data.volume >= 0
        assert data.open_price > 0
        assert data.high_price > 0
        assert data.low_price > 0
        assert data.close_price > 0
    
    def test_invalid_price_zero(self):
        """Test zero price (should fail gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="AAPL",
                price=Decimal("0"),
                volume=1000000,
                data_source="yahoo_finance"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than' for error in errors)
    
    def test_invalid_price_negative(self):
        """Test negative price (should fail gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="AAPL",
                price=Decimal("-10.00"),
                volume=1000000,
                data_source="yahoo_finance"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than' for error in errors)
    
    def test_invalid_data_source(self):
        """Test invalid data source (should fail enum)."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="AAPL",
                price=Decimal("150.00"),
                volume=1000000,
                data_source="unknown_api"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_missing_required_fields(self):
        """Test missing required fields (should fail)."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData()  # Missing symbol and price
        errors = exc_info.value.errors()
        # Should have errors for missing symbol and price
        assert len(errors) >= 2
    
    def test_negative_volume(self):
        """Test negative volume (should fail ge=0)."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="AAPL",
                price=Decimal("150.00"),
                volume=-1000,
                data_source="yahoo_finance"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_valid_optional_fields_none(self):
        """Test that optional fields can be None."""
        data = MarketData(
            symbol="AAPL",
            price=Decimal("150.00"),
            volume=None,
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            market_cap=None,
            pe_ratio=None,
            data_source=None
        )
        assert data.symbol == "AAPL"
        assert data.price == Decimal("150.00")
        assert data.volume is None


class TestSentimentDataValidation:
    """Test validation rules for SentimentAnalysis model."""
    
    def test_valid_sentiment_data(self):
        """Test valid sentiment data with score in range."""
        sentiment = SentimentAnalysis(
            symbol="AAPL",
            sentiment_score=Decimal("-0.5"),
            sentiment_label="negative",
            confidence=Decimal("0.8"),
            article_count=50,
            data_source="newsapi"
        )
        assert sentiment.sentiment_score == Decimal("-0.5")
        assert sentiment.confidence == Decimal("0.8")
        assert sentiment.article_count == 50
    
    def test_invalid_sentiment_score_too_low(self):
        """Test sentiment score below -1.0 (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentAnalysis(
                symbol="AAPL",
                sentiment_score=Decimal("-2.0"),
                sentiment_label="negative",
                confidence=Decimal("0.8"),
                article_count=50,
                data_source="newsapi"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_invalid_sentiment_score_too_high(self):
        """Test sentiment score above 1.0 (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentAnalysis(
                symbol="AAPL",
                sentiment_score=Decimal("1.5"),
                sentiment_label="positive",
                confidence=Decimal("0.8"),
                article_count=50,
                data_source="newsapi"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)
    
    def test_invalid_sentiment_label(self):
        """Test invalid sentiment label (should fail enum)."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentAnalysis(
                symbol="AAPL",
                sentiment_score=Decimal("0.5"),
                sentiment_label="very_positive",
                confidence=Decimal("0.8"),
                article_count=50,
                data_source="newsapi"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_invalid_article_count_zero(self):
        """Test article count of 0 (should fail ge=1)."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentAnalysis(
                symbol="AAPL",
                sentiment_score=Decimal("0.5"),
                sentiment_label="positive",
                confidence=Decimal("0.8"),
                article_count=0,
                data_source="newsapi"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_invalid_article_count_too_high(self):
        """Test article count above 1000 (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            SentimentAnalysis(
                symbol="AAPL",
                sentiment_score=Decimal("0.5"),
                sentiment_label="positive",
                confidence=Decimal("0.8"),
                article_count=1001,
                data_source="newsapi"
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)
    
    def test_valid_confidence_range(self):
        """Test confidence values at boundaries."""
        # Minimum confidence
        SentimentAnalysis(
            symbol="AAPL",
            sentiment_score=Decimal("0.5"),
            sentiment_label="positive",
            confidence=Decimal("0.0"),
            article_count=10,
            data_source="newsapi"
        )
        
        # Maximum confidence
        SentimentAnalysis(
            symbol="AAPL",
            sentiment_score=Decimal("0.5"),
            sentiment_label="positive",
            confidence=Decimal("1.0"),
            article_count=10,
            data_source="newsapi"
        )


class TestTechnicalAnalysisValidation:
    """Test validation rules for TechnicalAnalysis model."""
    
    def test_valid_rsi_range(self):
        """Test valid RSI in 0-100 range."""
        analysis = TechnicalAnalysis(
            symbol="AAPL",
            rsi=Decimal("70.5"),
            signal="buy",
            confidence=Decimal("0.8")
        )
        assert analysis.rsi == Decimal("70.5")
        
        # Test boundary values
        TechnicalAnalysis(symbol="AAPL", rsi=Decimal("0"), signal="sell", confidence=Decimal("0.1"))
        TechnicalAnalysis(symbol="AAPL", rsi=Decimal("100"), signal="buy", confidence=Decimal("0.9"))
    
    def test_invalid_rsi_negative(self):
        """Test RSI below 0 (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            TechnicalAnalysis(
                symbol="AAPL",
                rsi=Decimal("-10"),
                signal="buy",
                confidence=Decimal("0.8")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_invalid_rsi_too_high(self):
        """Test RSI above 100 (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            TechnicalAnalysis(
                symbol="AAPL",
                rsi=Decimal("150"),
                signal="buy",
                confidence=Decimal("0.8")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)
    
    def test_invalid_signal(self):
        """Test invalid signal value (should fail enum)."""
        with pytest.raises(ValidationError) as exc_info:
            TechnicalAnalysis(
                symbol="AAPL",
                rsi=Decimal("70"),
                signal="maybe_buy",
                confidence=Decimal("0.8")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_valid_optional_fields_none(self):
        """Test that optional fields can be None."""
        analysis = TechnicalAnalysis(
            symbol="AAPL",
            rsi=None,
            macd_value=None,
            macd_signal=None,
            macd_histogram=None,
            bb_upper=None,
            bb_middle=None,
            bb_lower=None,
            signal=None,
            confidence=None
        )
        assert analysis.symbol == "AAPL"
        assert analysis.rsi is None
    
    def test_bollinger_bands_positive_values(self):
        """Test Bollinger Bands must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TechnicalAnalysis(
                symbol="AAPL",
                bb_upper=Decimal("-10"),  # Should be positive
                signal="buy",
                confidence=Decimal("0.8")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than' for error in errors)


class TestTradingDecisionsValidation:
    """Test validation rules for TradingDecision model."""
    
    def test_valid_trading_decision(self):
        """Test valid trading decision creation."""
        decision = TradingDecision(
            user_id="user123",
            symbol="AAPL",
            action="buy",
            recommended_quantity=100,
            recommended_price=Decimal("150.00"),
            confidence=Decimal("0.8"),
            reasoning="Strong technical indicators suggest upward momentum",
            budget_allocated=Decimal("15000.00")
        )
        assert decision.action == "buy"
        assert decision.recommended_quantity == 100
        assert decision.confidence == Decimal("0.8")
        assert len(decision.reasoning) >= 10
    
    def test_invalid_quantity_negative(self):
        """Test negative quantity (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="buy",
                recommended_quantity=-50,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("0.8"),
                reasoning="This should fail validation",
                budget_allocated=Decimal("15000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_invalid_quantity_too_large(self):
        """Test quantity above maximum (should fail range)."""
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="buy",
                recommended_quantity=200000,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("0.8"),
                reasoning="This quantity is too large",
                budget_allocated=Decimal("15000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)
    
    def test_invalid_reasoning_too_short(self):
        """Test reasoning too short (should fail min_length=10)."""
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="buy",
                recommended_quantity=100,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("0.8"),
                reasoning="buy",  # Too short
                budget_allocated=Decimal("15000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_too_short' for error in errors)
    
    def test_invalid_action(self):
        """Test invalid action value."""
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="maybe",  # Invalid action
                recommended_quantity=100,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("0.8"),
                reasoning="This action is invalid",
                budget_allocated=Decimal("15000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_budget_allocation_validation(self):
        """Test budget allocation must be non-negative."""
        # Valid zero allocation
        TradingDecision(
            user_id="user123",
            symbol="AAPL",
            action="hold",
            recommended_quantity=0,
            recommended_price=Decimal("150.00"),
            confidence=Decimal("0.5"),
            reasoning="Hold position due to market uncertainty",
            budget_allocated=Decimal("0.00")
        )
        
        # Invalid negative allocation
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="buy",
                recommended_quantity=100,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("0.8"),
                reasoning="This should fail due to negative budget",
                budget_allocated=Decimal("-1000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_confidence_range_validation(self):
        """Test confidence must be between 0 and 1."""
        # Valid boundary values
        TradingDecision(
            user_id="user123",
            symbol="AAPL",
            action="buy",
            recommended_quantity=100,
            recommended_price=Decimal("150.00"),
            confidence=Decimal("0.0"),
            reasoning="Minimum confidence decision",
            budget_allocated=Decimal("15000.00")
        )
        
        TradingDecision(
            user_id="user123",
            symbol="AAPL",
            action="buy",
            recommended_quantity=100,
            recommended_price=Decimal("150.00"),
            confidence=Decimal("1.0"),
            reasoning="Maximum confidence decision",
            budget_allocated=Decimal("15000.00")
        )
        
        # Invalid confidence > 1
        with pytest.raises(ValidationError) as exc_info:
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                action="buy",
                recommended_quantity=100,
                recommended_price=Decimal("150.00"),
                confidence=Decimal("1.5"),
                reasoning="Over-confident decision",
                budget_allocated=Decimal("15000.00")
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'less_than_equal' for error in errors)


class TestDagRunValidation:
    """Test validation rules for DagRun model."""
    
    def test_valid_dag_run(self):
        """Test valid DAG run creation."""
        dag_run = DagRun(
            dag_status="success",
            symbols_processed=10,
            users_served=5,
            execution_time_ms=30000
        )
        assert dag_run.dag_status == "success"
        assert dag_run.symbols_processed == 10
        assert dag_run.users_served == 5
        assert dag_run.execution_time_ms == 30000
    
    def test_invalid_dag_status(self):
        """Test invalid DAG status."""
        with pytest.raises(ValidationError) as exc_info:
            DagRun(
                dag_status="unknown",  # Invalid status
                symbols_processed=10,
                users_served=5,
                execution_time_ms=30000
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'string_pattern_mismatch' for error in errors)
    
    def test_negative_counts(self):
        """Test negative values for counts."""
        with pytest.raises(ValidationError) as exc_info:
            DagRun(
                dag_status="success",
                symbols_processed=-1,  # Should be >= 0
                users_served=5,
                execution_time_ms=30000
            )
        errors = exc_info.value.errors()
        assert any(error['type'] == 'greater_than_equal' for error in errors)
    
    def test_valid_optional_none_values(self):
        """Test that all fields can be None (optional)."""
        dag_run = DagRun()
        assert dag_run.dag_status is None
        assert dag_run.symbols_processed is None
        assert dag_run.users_served is None
        assert dag_run.execution_time_ms is None


class TestComplexValidationScenarios:
    """Test complex validation scenarios and edge cases."""
    
    def test_decimal_precision_handling(self):
        """Test that Decimal fields handle precision correctly."""
        # Test market data with high precision
        data = MarketData(
            symbol="AAPL",
            price=Decimal("150.123456789"),
            volume=1000000
        )
        assert data.price == Decimal("150.123456789")
    
    def test_datetime_field_handling(self):
        """Test datetime fields with various inputs."""
        now = datetime.utcnow()
        
        # Test with explicit datetime
        profile = UserProfile(
            user_id="user123",
            budget=Decimal("10000"),
            created_at=now
        )
        assert profile.created_at == now
        
        # Test with None (should be allowed)
        profile2 = UserProfile(
            user_id="user456",
            budget=Decimal("10000"),
            created_at=None
        )
        assert profile2.created_at is None
    
    def test_string_length_validation(self):
        """Test string length constraints across models."""
        # Test maximum symbol length
        with pytest.raises(ValidationError):
            MarketData(
                symbol="VERYLONGSYMBOL",  # Too long
                price=Decimal("100.00")
            )
        
        # Test maximum reasoning length
        with pytest.raises(ValidationError):
            TradingDecision(
                user_id="user123",
                symbol="AAPL",
                reasoning="x" * 1001,  # Too long (max 1000)
                action="buy"
            )
    
    def test_enum_case_sensitivity(self):
        """Test that enum validation is case-sensitive."""
        # This should fail - wrong case
        with pytest.raises(ValidationError):
            UserProfile(
                user_id="user123",
                risk_tolerance="MODERATE",  # Should be "moderate"
                budget=Decimal("10000")
            )
    
    def test_model_defaults(self):
        """Test that model defaults work correctly."""
        # UserProfile defaults
        profile = UserProfile(user_id="user123")
        assert profile.budget == Decimal("100000.00")
        assert profile.risk_tolerance == "moderate"
        assert profile.trading_style == "long_term"
        assert profile.interested_symbols == []
        
        # TradingDecision defaults
        decision = TradingDecision(
            user_id="user123",
            symbol="AAPL"
        )
        assert decision.executed is False
    
    def test_cross_field_logical_validation(self):
        """Test logical relationships between fields."""
        # For market data, high should be >= low (when both present)
        # Note: The current model doesn't enforce this, but it's a logical constraint
        data = MarketData(
            symbol="AAPL",
            price=Decimal("150.00"),
            high_price=Decimal("155.00"),
            low_price=Decimal("145.00")
        )
        # This should pass basic validation
        assert data.high_price > data.low_price
    
    def test_none_vs_empty_validation(self):
        """Test difference between None and empty values."""
        # Empty list should be allowed for interested_symbols
        profile = UserProfile(
            user_id="user123",
            interested_symbols=[]
        )
        assert profile.interested_symbols == []
        
        # None should be converted to empty list
        profile2 = UserProfile(
            user_id="user123",
            interested_symbols=None
        )
        assert profile2.interested_symbols == []