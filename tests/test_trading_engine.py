"""
Tests for Trading Engine Module.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.trading_engine import (
    TradingEngine, RiskManager, StrategyEngine, UserProfileManager, PortfolioOptimizer,
    TradeRecommendation, TradingParameters, RiskProfile, Strategy,
    RiskLevel, RiskCategory, TradeAction, StrategyType, MarketRegime, SizingMethod,
    validate_trade_parameters, calculate_risk_adjusted_return, format_recommendation
)


class TestRiskManager:
    """Test risk management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager()
        self.user_params = TradingParameters(
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.20,
            max_position_size=0.10,
            daily_loss_limit=0.06,
            leverage_limit=1.5
        )
    
    def test_risk_manager_initialization(self):
        """Test RiskManager initialization."""
        assert self.risk_manager.max_risk_per_trade == 0.02
        assert self.risk_manager.max_portfolio_risk == 0.20
        assert self.risk_manager.max_position_size == 0.10
    
    def test_calculate_position_size_fixed_percent(self):
        """Test position size calculation with fixed percent method."""
        result = self.risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.02,
            entry_price=100.0,
            sizing_method=SizingMethod.FIXED_PERCENT
        )
        
        assert result["position_size"] > 0
        assert result["position_value"] > 0
        assert result["actual_risk_percentage"] <= self.risk_manager.max_position_size
        assert "error" not in result
    
    def test_calculate_position_size_volatility_adjusted(self):
        """Test position size calculation with volatility adjustment."""
        result = self.risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.02,
            entry_price=100.0,
            stop_loss_price=95.0,
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED
        )
        
        assert result["position_size"] > 0
        assert result["stop_loss"] == 95.0
        assert "error" not in result
    
    def test_calculate_position_size_invalid_entry_price(self):
        """Test position size calculation with invalid entry price."""
        result = self.risk_manager.calculate_position_size(
            account_balance=100000,
            risk_per_trade=0.02,
            entry_price=0.0
        )
        
        assert result["position_size"] == 0
        assert "error" in result
    
    def test_validate_trade_risk_approved(self):
        """Test trade risk validation - approved trade."""
        trade = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 50,
            "price": 150.0
        }
        portfolio = {
            "total_value": 100000,
            "daily_pnl": 0.0
        }
        
        result = self.risk_manager.validate_trade_risk(trade, portfolio, self.user_params)
        
        assert result["approved"] is True
        assert isinstance(result["risk_score"], float)
        assert 0 <= result["risk_score"] <= 1
        assert result["position_percentage"] < self.user_params.max_position_size
    
    def test_validate_trade_risk_rejected_position_size(self):
        """Test trade risk validation - rejected due to position size."""
        trade = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 1000,  # Large position
            "price": 150.0
        }
        portfolio = {
            "total_value": 100000,
            "daily_pnl": 0.0
        }
        
        result = self.risk_manager.validate_trade_risk(trade, portfolio, self.user_params)
        
        assert len(result["warnings"]) > 0
        assert "suggested_quantity" in result["adjustments"]
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        portfolio = {
            "total_value": 100000,
            "positions": {
                "AAPL": {"value": 20000},
                "MSFT": {"value": 15000},
                "GOOGL": {"value": 10000}
            }
        }
        
        result = self.risk_manager.calculate_portfolio_risk(portfolio)
        
        assert "total_risk" in result
        assert "concentration_risk" in result
        assert "diversification_score" in result
        assert result["position_count"] == 3
        assert 0 <= result["diversification_score"] <= 1
    
    def test_check_daily_loss_limit(self):
        """Test daily loss limit checking."""
        portfolio = {
            "total_value": 100000,
            "daily_pnl": -3000  # 3% loss
        }
        
        result = self.risk_manager.check_daily_loss_limit(portfolio, 0.06)
        
        assert result["current_loss"] == 0.03
        assert result["limit_reached"] is False
        assert result["remaining_capacity"] == 0.03
    
    def test_calculate_stop_loss(self):
        """Test stop loss calculation."""
        stop_loss = self.risk_manager.calculate_stop_loss(100.0, 0.05)
        assert stop_loss == 95.0
    
    def test_calculate_take_profit(self):
        """Test take profit calculation."""
        take_profit = self.risk_manager.calculate_take_profit(100.0, 2.0)
        assert take_profit == 104.0  # 100 * (1 + 2.0 * 0.02)


class TestStrategyEngine:
    """Test strategy selection and recommendation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy_engine = StrategyEngine()
        self.user_params = TradingParameters(
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.20,
            max_position_size=0.10,
            daily_loss_limit=0.06,
            leverage_limit=1.5
        )
    
    def test_strategy_engine_initialization(self):
        """Test StrategyEngine initialization."""
        assert len(self.strategy_engine.strategies) > 0
        assert hasattr(self.strategy_engine, 'strategy_performance')
    
    def test_select_strategy_conservative(self):
        """Test strategy selection for conservative user."""
        market_conditions = {
            "volatility": 0.15,
            "trend": "bullish",
            "sentiment": "positive"
        }
        user_profile = {
            "risk_category": "conservative",
            "risk_tolerance": 0.3
        }
        
        result = self.strategy_engine.select_strategy(market_conditions, user_profile)
        
        assert "strategy" in result
        assert "confidence" in result
        assert "parameters" in result
        assert "reasoning" in result
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
    
    def test_select_strategy_aggressive(self):
        """Test strategy selection for aggressive user."""
        market_conditions = {
            "volatility": 0.25,
            "trend": "bullish",
            "sentiment": "positive"
        }
        user_profile = {
            "risk_category": "aggressive",
            "risk_tolerance": 0.8
        }
        
        result = self.strategy_engine.select_strategy(market_conditions, user_profile)
        
        assert result["strategy"] is not None
        assert result["confidence"] > 0
    
    def test_determine_market_regime(self):
        """Test market regime determination."""
        # High volatility
        high_vol_conditions = {"volatility": 0.35, "trend": "sideways", "sentiment": "neutral"}
        regime = self.strategy_engine._determine_market_regime(high_vol_conditions)
        assert regime == MarketRegime.HIGH_VOLATILITY
        
        # Bull market
        bull_conditions = {"volatility": 0.2, "trend": "bullish", "sentiment": "positive"}
        regime = self.strategy_engine._determine_market_regime(bull_conditions)
        assert regime == MarketRegime.BULL_MARKET
        
        # Low volatility
        low_vol_conditions = {"volatility": 0.1, "trend": "sideways", "sentiment": "neutral"}
        regime = self.strategy_engine._determine_market_regime(low_vol_conditions)
        assert regime == MarketRegime.LOW_VOLATILITY
    
    def test_generate_recommendations(self):
        """Test trade recommendation generation."""
        analysis_results = {
            "symbol": "AAPL",
            "signals": {
                "overall_signal": "buy",
                "signal_strength": "strong",
                "component_signals": {
                    "technical": "buy",
                    "fundamental": "hold"
                }
            },
            "analysis_components": {
                "technical": {
                    "latest_price": 150.0
                }
            }
        }
        
        strategy = {
            "strategy": "moderate_momentum",
            "confidence": 0.75,
            "parameters": {"max_position_size": 0.1}
        }
        
        recommendations = self.strategy_engine.generate_recommendations(
            analysis_results, strategy, self.user_params
        )
        
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.symbol == "AAPL"
        assert rec.action == TradeAction.BUY
        assert rec.quantity > 0
        assert rec.price == 150.0
        assert rec.confidence > 0
    
    def test_generate_recommendations_hold_signal(self):
        """Test recommendation generation with hold signal."""
        analysis_results = {
            "symbol": "AAPL",
            "signals": {
                "overall_signal": "hold",
                "signal_strength": "weak",
                "component_signals": {}
            }
        }
        
        strategy = {"strategy": "conservative_growth"}
        
        recommendations = self.strategy_engine.generate_recommendations(
            analysis_results, strategy, self.user_params
        )
        
        assert len(recommendations) == 0


class TestUserProfileManager:
    """Test user profile management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.user_manager = UserProfileManager()
    
    def test_user_profile_manager_initialization(self):
        """Test UserProfileManager initialization."""
        assert hasattr(self.user_manager, 'config')
    
    def test_assess_risk_profile(self):
        """Test risk profile assessment."""
        questionnaire = {
            "investment_experience": "moderate",
            "risk_tolerance": "medium",
            "investment_horizon": "long_term"
        }
        
        result = self.user_manager.assess_risk_profile(questionnaire)
        
        assert "risk_category" in result
        assert "risk_score" in result
        assert "trading_parameters" in result
        assert "confidence_score" in result
        assert isinstance(result["trading_parameters"], TradingParameters)
    
    def test_get_trading_parameters(self):
        """Test getting trading parameters for user."""
        params = self.user_manager.get_trading_parameters("test_user")
        
        assert isinstance(params, TradingParameters)
        assert params.max_risk_per_trade > 0
        assert params.max_portfolio_risk > 0
        assert params.max_position_size > 0
    
    def test_validate_trade_against_profile(self):
        """Test trade validation against user profile."""
        trade = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 150.0
        }
        
        result = self.user_manager.validate_trade_against_profile("test_user", trade)
        
        assert "approved" in result
        assert "message" in result
        assert isinstance(result["approved"], bool)


class TestPortfolioOptimizer:
    """Test portfolio optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PortfolioOptimizer()
    
    def test_portfolio_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization."""
        assert hasattr(self.optimizer, 'config')
    
    def test_optimize_portfolio(self):
        """Test portfolio optimization."""
        assets = ["AAPL", "MSFT", "GOOGL"]
        expected_returns = pd.Series([0.10, 0.12, 0.08], index=assets)
        cov_matrix = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)
        
        result = self.optimizer.optimize_portfolio(assets, expected_returns, cov_matrix, 0.6)
        
        assert "weights" in result
        assert "expected_return" in result
        assert "expected_risk" in result
        assert "sharpe_ratio" in result
    
    def test_rebalance_portfolio(self):
        """Test portfolio rebalancing."""
        current_portfolio = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        target_weights = {"AAPL": 0.35, "MSFT": 0.35, "GOOGL": 0.3}
        
        result = self.optimizer.rebalance_portfolio(current_portfolio, target_weights)
        
        assert "rebalance_needed" in result
        assert "trades" in result
        assert "cost_estimate" in result


class TestTradingEngine:
    """Test integrated trading engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trading_engine = TradingEngine()
        self.sample_analysis = {
            "symbol": "AAPL",
            "signals": {
                "overall_signal": "buy",
                "signal_strength": "moderate",
                "component_signals": {
                    "technical": "buy",
                    "fundamental": "hold"
                }
            },
            "analysis_components": {
                "technical": {"latest_price": 150.0},
                "sentiment": {"overall_sentiment": "positive"}
            }
        }
        self.sample_portfolio = {
            "total_value": 100000,
            "daily_pnl": 500,
            "positions": {
                "MSFT": {"value": 20000},
                "GOOGL": {"value": 15000}
            }
        }
    
    def test_trading_engine_initialization(self):
        """Test TradingEngine initialization."""
        assert hasattr(self.trading_engine, 'risk_manager')
        assert hasattr(self.trading_engine, 'user_manager')
        assert hasattr(self.trading_engine, 'strategy_engine')
        assert hasattr(self.trading_engine, 'portfolio_optimizer')
    
    def test_process_trading_decision(self):
        """Test complete trading decision processing."""
        result = self.trading_engine.process_trading_decision(
            "test_user", self.sample_analysis, self.sample_portfolio
        )
        
        assert "user_id" in result
        assert "timestamp" in result
        assert "recommendations" in result
        assert "risk_assessment" in result
        assert "strategy_selection" in result
        assert "portfolio_allocation" in result
        assert "validation_summary" in result
        
        assert result["user_id"] == "test_user"
        assert isinstance(result["recommendations"], list)
    
    def test_execute_trade_validation(self):
        """Test trade validation before execution."""
        trade_rec = TradeRecommendation(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price=150.0,
            confidence=0.8,
            reasoning="Strong buy signal",
            risk_level=RiskLevel.MODERATE
        )
        
        result = self.trading_engine.execute_trade_validation(trade_rec, "test_user")
        
        assert "approved" in result
        assert "risk_score" in result
        assert isinstance(result["approved"], bool)
    
    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        result = self.trading_engine.calculate_portfolio_metrics(self.sample_portfolio)
        
        assert "total_value" in result
        assert "daily_pnl" in result
        assert "total_return" in result
        assert "risk_metrics" in result
        assert "diversification_score" in result


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_trade_parameters(self):
        """Test trade parameter validation."""
        valid_trade = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 150.0
        }
        invalid_trade = {
            "symbol": "AAPL",
            "action": "buy"
            # Missing quantity and price
        }
        
        assert validate_trade_parameters(valid_trade) is True
        assert validate_trade_parameters(invalid_trade) is False
    
    def test_calculate_risk_adjusted_return(self):
        """Test risk-adjusted return calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        benchmark = pd.Series([0.008, 0.015, -0.005, 0.025, 0.008])
        
        result = calculate_risk_adjusted_return(returns, benchmark)
        
        assert "alpha" in result
        assert "beta" in result
        assert "sharpe_ratio" in result
        assert "information_ratio" in result
        assert all(isinstance(v, float) for v in result.values())
    
    def test_format_recommendation(self):
        """Test recommendation formatting."""
        trade_rec = TradeRecommendation(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price=150.0,
            confidence=0.8,
            reasoning="Strong buy signal",
            risk_level=RiskLevel.MODERATE
        )
        
        formatted = format_recommendation(trade_rec)
        
        assert formatted["symbol"] == "AAPL"
        assert formatted["action"] == "buy"
        assert formatted["quantity"] == 100
        assert formatted["price"] == 150.0
        assert formatted["confidence"] == 0.8
        assert formatted["reasoning"] == "Strong buy signal"


class TestDataClasses:
    """Test data class structures."""
    
    def test_trading_parameters_creation(self):
        """Test TradingParameters creation."""
        params = TradingParameters(
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.20,
            max_position_size=0.10,
            daily_loss_limit=0.06,
            leverage_limit=1.5
        )
        
        assert params.max_risk_per_trade == 0.02
        assert params.max_portfolio_risk == 0.20
        assert params.max_position_size == 0.10
        assert params.daily_loss_limit == 0.06
        assert params.leverage_limit == 1.5
    
    def test_trade_recommendation_creation(self):
        """Test TradeRecommendation creation."""
        rec = TradeRecommendation(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price=150.0,
            confidence=0.8,
            reasoning="Strong buy signal",
            risk_level=RiskLevel.MODERATE
        )
        
        assert rec.symbol == "AAPL"
        assert rec.action == TradeAction.BUY
        assert rec.quantity == 100
        assert rec.price == 150.0
        assert rec.confidence == 0.8
        assert rec.risk_level == RiskLevel.MODERATE
    
    def test_strategy_creation(self):
        """Test Strategy creation."""
        strategy = Strategy(
            id="test_strategy",
            name="Test Strategy",
            strategy_type=StrategyType.MOMENTUM,
            description="Test strategy description",
            risk_level=0.5,
            min_investment_horizon_days=30,
            max_position_size=0.10,
            volatility_tolerance=0.25,
            market_conditions=[MarketRegime.BULL_MARKET],
            suitable_risk_categories=[RiskCategory.MODERATE],
            expected_return=0.12,
            expected_volatility=0.18,
            min_confidence_threshold=0.6
        )
        
        assert strategy.id == "test_strategy"
        assert strategy.strategy_type == StrategyType.MOMENTUM
        assert strategy.risk_level == 0.5
        assert RiskCategory.MODERATE in strategy.suitable_risk_categories


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trading_engine = TradingEngine()
    
    def test_trading_decision_with_invalid_data(self):
        """Test trading decision with invalid data."""
        invalid_analysis = {}
        invalid_portfolio = {}
        
        result = self.trading_engine.process_trading_decision(
            "test_user", invalid_analysis, invalid_portfolio
        )
        
        # Should handle gracefully without crashing
        assert "user_id" in result
        assert result["user_id"] == "test_user"
    
    def test_risk_manager_error_handling(self):
        """Test RiskManager error handling."""
        risk_manager = RiskManager()
        
        # Test with invalid trade data
        invalid_trade = {}
        portfolio = {"total_value": 100000}
        user_params = TradingParameters(0.02, 0.20, 0.10, 0.06, 1.5)
        
        result = risk_manager.validate_trade_risk(invalid_trade, portfolio, user_params)
        
        assert result["approved"] is False
        assert len(result["warnings"]) > 0
    
    def test_strategy_engine_error_handling(self):
        """Test StrategyEngine error handling."""
        strategy_engine = StrategyEngine()
        
        # Test with invalid market conditions
        result = strategy_engine.select_strategy({}, {})
        
        # Should return fallback strategy
        assert "strategy" in result
        assert result["confidence"] >= 0


class TestEnumValues:
    """Test enumeration values."""
    
    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.AGGRESSIVE.value == "aggressive"
        assert RiskLevel.SPECULATIVE.value == "speculative"
    
    def test_trade_action_enum(self):
        """Test TradeAction enum values."""
        assert TradeAction.BUY.value == "buy"
        assert TradeAction.SELL.value == "sell"
        assert TradeAction.HOLD.value == "hold"
        assert TradeAction.REDUCE.value == "reduce"
    
    def test_strategy_type_enum(self):
        """Test StrategyType enum values."""
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.BUY_AND_HOLD.value == "buy_and_hold"
    
    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.BULL_MARKET.value == "bull_market"
        assert MarketRegime.BEAR_MARKET.value == "bear_market"
        assert MarketRegime.SIDEWAYS_MARKET.value == "sideways_market"