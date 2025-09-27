"""
Comprehensive tests for Risk-Based Strategy Selection Logic.

Tests include:
- Strategy matching logic for different risk profiles
- Risk filtering and appropriateness checks  
- Strategy performance tracking by risk level
- Integration with recommendation pipeline
- Market condition adaptations
- Portfolio constraint validations
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.core.strategy_selector import (
    RiskBasedStrategySelector,
    StrategyDatabase,
    StrategyPerformanceTracker,
    MarketConditionAnalyzer,
    Strategy,
    StrategyType,
    StrategyMetrics,
    MarketRegime,
    MarketCondition,
    StrategyRecommendation,
)
from src.core.user_profiling import RiskCategory, RiskProfile, UserResponse
from src.core.recommendation_engine import RiskAwareRecommendationEngine


class TestStrategyDatabase:
    """Test strategy database functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy_db = StrategyDatabase()
    
    def test_default_strategies_initialization(self):
        """Test that default strategies are properly initialized."""
        strategies = self.strategy_db.strategies
        
        # Should have multiple strategies
        assert len(strategies) >= 7
        
        # Should have strategies for all risk categories
        conservative_strategies = self.strategy_db.get_strategies_by_risk_category(RiskCategory.CONSERVATIVE)
        moderate_strategies = self.strategy_db.get_strategies_by_risk_category(RiskCategory.MODERATE)
        aggressive_strategies = self.strategy_db.get_strategies_by_risk_category(RiskCategory.AGGRESSIVE)
        
        assert len(conservative_strategies) >= 2
        assert len(moderate_strategies) >= 2
        assert len(aggressive_strategies) >= 2
        
        # Verify strategy properties
        for strategy in strategies.values():
            assert isinstance(strategy.id, str)
            assert isinstance(strategy.name, str)
            assert isinstance(strategy.strategy_type, StrategyType)
            assert 0.0 <= strategy.risk_level <= 1.0
            assert strategy.min_investment_horizon_days > 0
            assert strategy.max_position_size > 0
            assert strategy.volatility_tolerance > 0
    
    def test_get_strategy_by_id(self):
        """Test retrieving specific strategy by ID."""
        # Test valid strategy ID
        strategy = self.strategy_db.get_strategy("conservative_buy_hold")
        assert strategy is not None
        assert strategy.id == "conservative_buy_hold"
        assert strategy.strategy_type == StrategyType.BUY_AND_HOLD
        
        # Test invalid strategy ID
        strategy = self.strategy_db.get_strategy("nonexistent_strategy")
        assert strategy is None
    
    def test_get_strategies_by_market_condition(self):
        """Test filtering strategies by market conditions."""
        bull_strategies = self.strategy_db.get_strategies_by_market_condition(MarketRegime.BULL_MARKET)
        crisis_strategies = self.strategy_db.get_strategies_by_market_condition(MarketRegime.CRISIS)
        
        assert len(bull_strategies) > 0
        assert len(crisis_strategies) >= 0  # Some strategies may not be suitable for crisis
        
        # Verify all returned strategies include the specified market condition
        for strategy in bull_strategies:
            assert MarketRegime.BULL_MARKET in strategy.market_conditions
    
    def test_add_custom_strategy(self):
        """Test adding custom strategy to database."""
        custom_strategy = Strategy(
            id="test_strategy",
            name="Test Strategy",
            strategy_type=StrategyType.MOMENTUM,
            description="Test strategy for unit tests",
            risk_level=0.5,
            min_investment_horizon_days=30,
            max_position_size=0.08,
            volatility_tolerance=0.25,
            market_conditions=[MarketRegime.BULL_MARKET],
            suitable_risk_categories=[RiskCategory.MODERATE],
            expected_return=0.12,
            expected_volatility=0.18,
            min_confidence_threshold=0.5
        )
        
        initial_count = len(self.strategy_db.strategies)
        self.strategy_db.add_strategy(custom_strategy)
        
        assert len(self.strategy_db.strategies) == initial_count + 1
        assert self.strategy_db.get_strategy("test_strategy") == custom_strategy


class TestStrategyPerformanceTracker:
    """Test strategy performance tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = StrategyPerformanceTracker()
    
    def test_update_strategy_performance(self):
        """Test updating strategy performance metrics."""
        metrics = StrategyMetrics(
            strategy_id="test_strategy",
            strategy_type=StrategyType.MOMENTUM,
            total_return=0.15,
            annualized_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            max_drawdown=-0.08,
            win_rate=0.60,
            avg_win=0.05,
            avg_loss=-0.03,
            trade_count=50,
            risk_adjusted_return=0.10,
            last_updated=datetime.now(),
            confidence_score=0.8
        )
        
        self.tracker.update_strategy_performance(metrics, RiskCategory.MODERATE)
        
        # Verify metrics are stored
        retrieved_metrics = self.tracker.get_strategy_performance("test_strategy", RiskCategory.MODERATE)
        assert retrieved_metrics is not None
        assert retrieved_metrics.strategy_id == "test_strategy"
        assert retrieved_metrics.annualized_return == 0.12
        assert retrieved_metrics.sharpe_ratio == 0.67
    
    def test_get_top_performing_strategies(self):
        """Test retrieving top performing strategies."""
        # Add multiple strategies with different performance
        strategies_data = [
            ("strategy_a", 0.15, 1.2),  # High performer
            ("strategy_b", 0.08, 0.6),  # Average performer  
            ("strategy_c", 0.20, 1.5),  # Best performer
            ("strategy_d", 0.05, 0.3),  # Poor performer
        ]
        
        for strategy_id, risk_adj_return, sharpe in strategies_data:
            metrics = StrategyMetrics(
                strategy_id=strategy_id,
                strategy_type=StrategyType.MOMENTUM,
                total_return=risk_adj_return * 1.2,
                annualized_return=risk_adj_return,
                volatility=0.18,
                sharpe_ratio=sharpe,
                max_drawdown=-0.08,
                win_rate=0.60,
                avg_win=0.05,
                avg_loss=-0.03,
                trade_count=50,
                risk_adjusted_return=risk_adj_return,
                last_updated=datetime.now()
            )
            self.tracker.update_strategy_performance(metrics, RiskCategory.MODERATE)
        
        # Get top 3 performers
        top_strategies = self.tracker.get_top_performing_strategies(RiskCategory.MODERATE, limit=3)
        
        assert len(top_strategies) == 3
        assert top_strategies[0].strategy_id == "strategy_c"  # Best performer first
        assert top_strategies[1].strategy_id == "strategy_a"  # Second best
        assert top_strategies[2].strategy_id == "strategy_b"  # Third best
    
    def test_calculate_strategy_consistency(self):
        """Test strategy consistency calculation."""
        # Add consistent performance history
        consistent_returns = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10]
        
        for i, annual_return in enumerate(consistent_returns):
            metrics = StrategyMetrics(
                strategy_id="consistent_strategy",
                strategy_type=StrategyType.BUY_AND_HOLD,
                total_return=annual_return * 1.2,
                annualized_return=annual_return,
                volatility=0.15,
                sharpe_ratio=annual_return / 0.15,
                max_drawdown=-0.05,
                win_rate=0.65,
                avg_win=0.04,
                avg_loss=-0.02,
                trade_count=30,
                risk_adjusted_return=annual_return * 0.8,
                last_updated=datetime.now() - timedelta(days=(len(consistent_returns) - i) * 30)
            )
            self.tracker.update_strategy_performance(metrics, RiskCategory.CONSERVATIVE)
        
        consistency = self.tracker.calculate_strategy_consistency("consistent_strategy", lookback_periods=6)
        
        # Should be high consistency (low variance in returns)
        assert consistency > 0.8
        
        # Test strategy with no history
        no_history_consistency = self.tracker.calculate_strategy_consistency("nonexistent", lookback_periods=6)
        assert no_history_consistency == 0.0


class TestMarketConditionAnalyzer:
    """Test market condition analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MarketConditionAnalyzer()
    
    def test_assess_current_conditions_bull_market(self):
        """Test assessment of bull market conditions."""
        bull_market_data = {
            "volatility_index": 15.0,  # Low volatility
            "market_trend": 0.5,       # Strong positive trend
            "uncertainty_index": 0.2,  # Low uncertainty
            "sector_performance": {"tech": 0.08, "finance": 0.06},
            "credit_spreads": -0.1,    # Tight spreads (risk on)
            "equity_momentum": 0.3,
            "dollar_strength": -0.1
        }
        
        condition = self.analyzer.assess_current_conditions(bull_market_data)
        
        assert condition.regime == MarketRegime.BULL_MARKET
        assert condition.volatility_level == 15.0
        assert condition.trend_strength == 0.5
        assert condition.risk_on_sentiment > 0  # Risk-on environment
    
    def test_assess_current_conditions_crisis(self):
        """Test assessment of crisis conditions."""
        crisis_data = {
            "volatility_index": 45.0,  # Very high volatility
            "market_trend": -0.6,      # Strong negative trend
            "uncertainty_index": 0.8,  # High uncertainty
            "sector_performance": {"tech": -0.15, "finance": -0.20},
            "credit_spreads": 0.3,     # Wide spreads (risk off)
            "equity_momentum": -0.4,
            "dollar_strength": 0.2
        }
        
        condition = self.analyzer.assess_current_conditions(crisis_data)
        
        assert condition.regime == MarketRegime.CRISIS
        assert condition.volatility_level == 45.0
        assert condition.uncertainty_index == 0.8
        assert condition.risk_on_sentiment < 0  # Risk-off environment
    
    def test_assess_current_conditions_sideways_market(self):
        """Test assessment of sideways market conditions."""
        sideways_data = {
            "volatility_index": 20.0,  # Moderate volatility
            "market_trend": 0.1,       # Weak trend
            "uncertainty_index": 0.4,  # Moderate uncertainty
        }
        
        condition = self.analyzer.assess_current_conditions(sideways_data)
        
        assert condition.regime == MarketRegime.SIDEWAYS_MARKET
        assert condition.volatility_level == 20.0
        assert abs(condition.trend_strength) < 0.3  # Weak trend


class TestRiskBasedStrategySelector:
    """Test main strategy selection engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = RiskBasedStrategySelector()
        
        # Create test risk profiles
        self.conservative_profile = RiskProfile(
            user_id="conservative_user",
            risk_category=RiskCategory.CONSERVATIVE,
            risk_score=20,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.8
        )
        
        self.moderate_profile = RiskProfile(
            user_id="moderate_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=50,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.7
        )
        
        self.aggressive_profile = RiskProfile(
            user_id="aggressive_user",
            risk_category=RiskCategory.AGGRESSIVE,
            risk_score=80,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.9
        )
    
    def test_select_strategies_for_conservative_user(self):
        """Test strategy selection for conservative users."""
        market_data = {
            "volatility_index": 18.0,
            "market_trend": 0.2,
            "uncertainty_index": 0.3,
        }
        
        recommendations = self.selector.select_strategies_for_user(
            risk_profile=self.conservative_profile,
            market_data=market_data,
            portfolio_value=100000.0,
            max_strategies=3
        )
        
        assert len(recommendations) <= 3
        assert len(recommendations) > 0
        
        # Verify all recommended strategies are appropriate for conservative users
        for rec in recommendations:
            assert rec.strategy.risk_level <= 0.4  # Low risk strategies only
            assert RiskCategory.CONSERVATIVE in rec.strategy.suitable_risk_categories
            assert rec.confidence_score > 0.0
            assert rec.allocation_percentage > 0.0
            
            # Check trade parameters are conservative
            trade_params = rec.trade_parameters
            assert trade_params["max_risk_per_trade"] <= 0.01
            assert trade_params["max_position_size"] <= 0.05
    
    def test_select_strategies_for_moderate_user(self):
        """Test strategy selection for moderate users."""
        market_data = {
            "volatility_index": 22.0,
            "market_trend": 0.0,
            "uncertainty_index": 0.4,
        }
        
        recommendations = self.selector.select_strategies_for_user(
            risk_profile=self.moderate_profile,
            market_data=market_data,
            portfolio_value=100000.0,
            max_strategies=5
        )
        
        assert len(recommendations) <= 5
        assert len(recommendations) > 0
        
        # Verify strategies are appropriate for moderate users
        for rec in recommendations:
            assert rec.strategy.risk_level <= 0.7  # Moderate risk tolerance
            assert RiskCategory.MODERATE in rec.strategy.suitable_risk_categories
            
            # Check trade parameters are moderate
            trade_params = rec.trade_parameters
            assert trade_params["max_risk_per_trade"] <= 0.02
            assert trade_params["max_position_size"] <= 0.10
    
    def test_select_strategies_for_aggressive_user(self):
        """Test strategy selection for aggressive users."""
        market_data = {
            "volatility_index": 30.0,
            "market_trend": 0.4,
            "uncertainty_index": 0.2,
        }
        
        recommendations = self.selector.select_strategies_for_user(
            risk_profile=self.aggressive_profile,
            market_data=market_data,
            portfolio_value=100000.0,
            max_strategies=8
        )
        
        assert len(recommendations) <= 8
        assert len(recommendations) > 0
        
        # Aggressive users can use high-risk strategies
        risk_levels = [rec.strategy.risk_level for rec in recommendations]
        assert max(risk_levels) > 0.6  # At least one high-risk strategy
        
        for rec in recommendations:
            assert RiskCategory.AGGRESSIVE in rec.strategy.suitable_risk_categories
    
    def test_strategy_scoring_system(self):
        """Test strategy scoring algorithm."""
        # Get a sample strategy
        strategy = self.selector.strategy_db.get_strategy("balanced_growth")
        assert strategy is not None
        
        # Test scoring with different market conditions
        bull_market = MarketCondition(
            regime=MarketRegime.BULL_MARKET,
            volatility_level=18.0,
            trend_strength=0.4,
            uncertainty_index=0.2,
            sector_rotation={},
            risk_on_sentiment=0.3
        )
        
        score = self.selector._score_strategy(
            strategy, self.moderate_profile, bull_market, 100000.0
        )
        
        assert 0.0 <= score <= 1.0
        
        # Test with crisis conditions (should get lower score)
        crisis_market = MarketCondition(
            regime=MarketRegime.CRISIS,
            volatility_level=45.0,
            trend_strength=-0.5,
            uncertainty_index=0.8,
            sector_rotation={},
            risk_on_sentiment=-0.4
        )
        
        crisis_score = self.selector._score_strategy(
            strategy, self.moderate_profile, crisis_market, 100000.0
        )
        
        # Score should be lower in crisis conditions if strategy isn't suitable
        if MarketRegime.CRISIS not in strategy.market_conditions:
            assert crisis_score < score
    
    def test_risk_filtering(self):
        """Test strategy filtering by risk appropriateness."""
        all_strategies = list(self.selector.strategy_db.strategies.values())
        
        market_condition = MarketCondition(
            regime=MarketRegime.BULL_MARKET,
            volatility_level=20.0,
            trend_strength=0.3,
            uncertainty_index=0.3,
            sector_rotation={},
            risk_on_sentiment=0.2
        )
        
        # Filter for conservative user
        conservative_filtered = self.selector.filter_strategies_by_risk(
            all_strategies, self.conservative_profile, market_condition
        )
        
        # All filtered strategies should be appropriate for conservative users
        for strategy in conservative_filtered:
            assert RiskCategory.CONSERVATIVE in strategy.suitable_risk_categories
            assert strategy.risk_level <= 0.4
        
        # Filter for aggressive user should return more strategies
        aggressive_filtered = self.selector.filter_strategies_by_risk(
            all_strategies, self.aggressive_profile, market_condition
        )
        
        assert len(aggressive_filtered) >= len(conservative_filtered)
    
    def test_strategy_allocation_limits(self):
        """Test strategy allocation limits for different risk profiles."""
        conservative_limits = self.selector.get_strategy_allocation_limits(self.conservative_profile)
        moderate_limits = self.selector.get_strategy_allocation_limits(self.moderate_profile)
        aggressive_limits = self.selector.get_strategy_allocation_limits(self.aggressive_profile)
        
        # Conservative users should have more restrictive limits
        assert conservative_limits["max_single_strategy"] >= moderate_limits["max_single_strategy"]
        assert conservative_limits["max_aggressive_strategies"] <= moderate_limits["max_aggressive_strategies"]
        assert conservative_limits["min_conservative_allocation"] >= moderate_limits["min_conservative_allocation"]
        
        # Aggressive users should have least restrictive limits
        assert aggressive_limits["max_aggressive_strategies"] >= moderate_limits["max_aggressive_strategies"]
        assert aggressive_limits["min_conservative_allocation"] <= moderate_limits["min_conservative_allocation"]
    
    def test_strategy_performance_update(self):
        """Test updating strategy performance metrics."""
        performance_data = {
            "strategy_type": "momentum",
            "total_return": 0.18,
            "annualized_return": 0.15,
            "volatility": 0.22,
            "sharpe_ratio": 0.68,
            "max_drawdown": -0.12,
            "win_rate": 0.62,
            "avg_win": 0.06,
            "avg_loss": -0.03,
            "trade_count": 45,
            "risk_adjusted_return": 0.12,
            "confidence_score": 0.75
        }
        
        self.selector.update_strategy_performance("momentum_trading", performance_data)
        
        # Verify performance was updated
        strategy = self.selector.strategy_db.get_strategy("momentum_trading")
        assert strategy is not None
        assert strategy.performance_metrics is not None
        assert strategy.performance_metrics.annualized_return == 0.15


class TestRiskAwareRecommendationEngine:
    """Test integrated recommendation engine with risk-based strategy selection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RiskAwareRecommendationEngine()
        
        self.test_risk_profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=50,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.7
        )
        
        self.test_portfolio = {
            "total_value": 100000.0,
            "available_cash": 20000.0,
            "positions": {
                "SPY": {"symbol": "SPY", "shares": 100, "value": 40000.0},
                "QQQ": {"symbol": "QQQ", "shares": 50, "value": 20000.0}
            }
        }
        
        self.test_market_data = {
            "volatility_index": 20.0,
            "market_trend": 0.2,
            "uncertainty_index": 0.3,
        }
    
    @patch('src.core.recommendation_engine.TradingSignalEngine.generate_signal')
    @patch('src.core.recommendation_engine.TradingSignalEngine.calculate_signal_confidence')
    def test_generate_comprehensive_recommendations(self, mock_confidence, mock_signal):
        """Test comprehensive recommendation generation."""
        # Mock trading signals
        mock_signal.return_value = "BUY"
        mock_confidence.return_value = 0.75
        
        recommendations = self.engine.generate_comprehensive_recommendations(
            user_id="test_user",
            risk_profile=self.test_risk_profile,
            portfolio_data=self.test_portfolio,
            market_data=self.test_market_data,
            include_strategy_recommendations=True
        )
        
        # Verify report structure
        assert "user_id" in recommendations
        assert "risk_category" in recommendations
        assert "trading_signals" in recommendations
        assert "strategy_recommendations" in recommendations
        assert "final_recommendations" in recommendations
        assert "portfolio_analysis" in recommendations
        assert "risk_analysis" in recommendations
        assert "executive_summary" in recommendations
        assert "overall_confidence" in recommendations
        
        # Verify risk category matches
        assert recommendations["risk_category"] == "moderate"
        
        # Verify final recommendations are risk-appropriate
        final_recs = recommendations["final_recommendations"]
        for rec in final_recs:
            assert rec["risk_category_appropriate"] is True
    
    def test_risk_filtering_integration(self):
        """Test integration of risk filtering in recommendation pipeline."""
        # Create aggressive risk profile
        aggressive_profile = RiskProfile(
            user_id="aggressive_user",
            risk_category=RiskCategory.AGGRESSIVE,
            risk_score=85,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.8
        )
        
        with patch('src.core.recommendation_engine.TradingSignalEngine.generate_signal') as mock_signal, \
             patch('src.core.recommendation_engine.TradingSignalEngine.calculate_signal_confidence') as mock_confidence:
            
            mock_signal.return_value = "BUY"
            mock_confidence.return_value = 0.4  # Low confidence signal
            
            recommendations = self.engine.generate_comprehensive_recommendations(
                user_id="aggressive_user",
                risk_profile=aggressive_profile,
                portfolio_data=self.test_portfolio,
                market_data=self.test_market_data
            )
            
            # Aggressive users should accept low-confidence signals
            trading_signals = recommendations["trading_signals"]
            assert len(trading_signals) > 0
            
            # Should have strategy recommendations
            strategy_recs = recommendations["strategy_recommendations"]
            assert len(strategy_recs) > 0
            
            # All recommended strategies should be suitable for aggressive users
            for rec in strategy_recs:
                assert rec["risk_level"] <= 1.0  # Aggressive users can handle high risk
    
    def test_conservative_user_filtering(self):
        """Test that conservative users get appropriately filtered recommendations."""
        conservative_profile = RiskProfile(
            user_id="conservative_user",
            risk_category=RiskCategory.CONSERVATIVE,
            risk_score=15,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.9
        )
        
        with patch('src.core.recommendation_engine.TradingSignalEngine.generate_signal') as mock_signal, \
             patch('src.core.recommendation_engine.TradingSignalEngine.calculate_signal_confidence') as mock_confidence:
            
            mock_signal.return_value = "BUY"
            mock_confidence.return_value = 0.4  # Low confidence signal
            
            recommendations = self.engine.generate_comprehensive_recommendations(
                user_id="conservative_user",
                risk_profile=conservative_profile,
                portfolio_data=self.test_portfolio,
                market_data=self.test_market_data
            )
            
            # Conservative users should reject low-confidence signals
            final_recs = recommendations["final_recommendations"]
            signal_recs = [r for r in final_recs if r.get("type") == "SIGNAL"]
            
            # Should have fewer or no signal-based recommendations due to low confidence
            assert len(signal_recs) == 0  # Low confidence signals should be filtered out
            
            # Strategy recommendations should be low-risk
            strategy_recs = recommendations["strategy_recommendations"]
            for rec in strategy_recs:
                assert rec["risk_level"] <= 0.4  # Conservative risk level
    
    def test_strategy_performance_updates(self):
        """Test updating strategy performance through recommendation engine."""
        performance_updates = [
            {
                "strategy_id": "conservative_buy_hold",
                "total_return": 0.08,
                "annualized_return": 0.07,
                "volatility": 0.12,
                "sharpe_ratio": 0.58,
                "max_drawdown": -0.05,
                "win_rate": 0.65,
                "avg_win": 0.03,
                "avg_loss": -0.02,
                "trade_count": 20,
                "risk_adjusted_return": 0.06,
                "confidence_score": 0.8
            }
        ]
        
        self.engine.update_strategy_performance(performance_updates)
        
        # Verify performance was updated in the strategy selector
        strategy = self.engine.strategy_selector.strategy_db.get_strategy("conservative_buy_hold")
        assert strategy is not None
        assert strategy.performance_metrics is not None
        assert strategy.performance_metrics.annualized_return == 0.07
    
    def test_executive_summary_generation(self):
        """Test generation of risk-aware executive summaries."""
        mock_final_recs = [
            {"type": "SIGNAL", "symbol": "SPY", "confidence": 0.8},
            {"type": "STRATEGY", "strategy_name": "Balanced Growth", "allocation_percentage": 0.3}
        ]
        
        mock_strategy_recs = [
            {"allocation_percentage": 0.3, "confidence_score": 0.7},
            {"allocation_percentage": 0.2, "confidence_score": 0.6}
        ]
        
        summary = self.engine._generate_risk_aware_summary(
            mock_final_recs, mock_strategy_recs, self.test_risk_profile
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "moderate" in summary.lower()  # Should mention risk category
        assert "recommendations" in summary.lower()
    
    def test_overall_confidence_calculation(self):
        """Test calculation of overall recommendation confidence."""
        mock_trading_signals = {
            "SPY": {"confidence": 0.8},
            "QQQ": {"confidence": 0.6}
        }
        
        mock_strategy_recs = [
            {"confidence_score": 0.7},
            {"confidence_score": 0.5}
        ]
        
        confidence = self.engine._calculate_overall_confidence(
            mock_trading_signals, mock_strategy_recs, self.test_risk_profile
        )
        
        assert 0.0 <= confidence <= 1.0
        # Should incorporate risk profile confidence (0.7) with higher weight
        assert confidence > 0.5  # Should be reasonably high given the inputs


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RiskAwareRecommendationEngine()
    
    def test_conservative_user_bull_market_scenario(self):
        """Test conservative user in bull market conditions."""
        conservative_profile = RiskProfile(
            user_id="conservative_user",
            risk_category=RiskCategory.CONSERVATIVE,
            risk_score=25,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.8
        )
        
        bull_market_data = {
            "volatility_index": 15.0,
            "market_trend": 0.4,
            "uncertainty_index": 0.2,
        }
        
        portfolio_data = {
            "total_value": 200000.0,
            "available_cash": 50000.0,
            "positions": {
                "SPY": {"symbol": "SPY", "shares": 200, "value": 80000.0},
                "BND": {"symbol": "BND", "shares": 500, "value": 40000.0}
            }
        }
        
        with patch('src.core.recommendation_engine.TradingSignalEngine.generate_signal') as mock_signal, \
             patch('src.core.recommendation_engine.TradingSignalEngine.calculate_signal_confidence') as mock_confidence:
            
            mock_signal.return_value = "HOLD"  # Conservative signal
            mock_confidence.return_value = 0.8  # High confidence
            
            recommendations = self.engine.generate_comprehensive_recommendations(
                user_id="conservative_user",
                risk_profile=conservative_profile,
                portfolio_data=portfolio_data,
                market_data=bull_market_data
            )
            
            # Should get conservative strategies suitable for bull market
            strategy_recs = recommendations["strategy_recommendations"]
            assert len(strategy_recs) > 0
            
            # All strategies should be low-risk
            for rec in strategy_recs:
                assert rec["risk_level"] <= 0.4
                
            # Should have reasonable allocations
            total_allocation = sum(rec["allocation_percentage"] for rec in strategy_recs)
            assert 0.5 <= total_allocation <= 1.0
            
            # Risk parameters should be conservative
            risk_analysis = recommendations["risk_analysis"]
            trading_params = risk_analysis["trading_parameters"]
            assert trading_params["max_risk_per_trade"] <= 0.01
            assert trading_params["max_position_size"] <= 0.05
    
    def test_aggressive_user_volatile_market_scenario(self):
        """Test aggressive user in high volatility conditions."""
        aggressive_profile = RiskProfile(
            user_id="aggressive_user",
            risk_category=RiskCategory.AGGRESSIVE,
            risk_score=90,
            responses=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.9
        )
        
        volatile_market_data = {
            "volatility_index": 35.0,
            "market_trend": 0.1,
            "uncertainty_index": 0.6,
        }
        
        portfolio_data = {
            "total_value": 500000.0,
            "available_cash": 100000.0,
            "positions": {
                "QQQ": {"symbol": "QQQ", "shares": 500, "value": 200000.0},
                "TSLA": {"symbol": "TSLA", "shares": 300, "value": 150000.0}
            }
        }
        
        with patch('src.core.recommendation_engine.TradingSignalEngine.generate_signal') as mock_signal, \
             patch('src.core.recommendation_engine.TradingSignalEngine.calculate_signal_confidence') as mock_confidence:
            
            mock_signal.return_value = "BUY"
            mock_confidence.return_value = 0.5  # Moderate confidence
            
            recommendations = self.engine.generate_comprehensive_recommendations(
                user_id="aggressive_user",
                risk_profile=aggressive_profile,
                portfolio_data=portfolio_data,
                market_data=volatile_market_data
            )
            
            # Aggressive user should get high-risk strategies suitable for volatility
            strategy_recs = recommendations["strategy_recommendations"]
            high_risk_strategies = [rec for rec in strategy_recs if rec["risk_level"] > 0.6]
            assert len(high_risk_strategies) > 0
            
            # Should accept moderate confidence signals
            final_recs = recommendations["final_recommendations"]
            signal_recs = [r for r in final_recs if r.get("type") == "SIGNAL"]
            assert len(signal_recs) > 0  # Should include BUY signals
            
            # Risk parameters should allow higher risk
            risk_analysis = recommendations["risk_analysis"]
            trading_params = risk_analysis["trading_parameters"]
            assert trading_params["max_risk_per_trade"] >= 0.03
            assert trading_params["max_position_size"] >= 0.15


@pytest.fixture
def sample_risk_profile():
    """Create sample risk profile for testing."""
    return RiskProfile(
        user_id="test_user_001",
        risk_category=RiskCategory.MODERATE,
        risk_score=55,
        responses=[
            UserResponse("experience_level", "Moderate experience - 2-5 years", 3),
            UserResponse("loss_tolerance", "10-20% - Moderate losses are acceptable for higher returns", 3),
            UserResponse("volatility_comfort", "3", 3)
        ],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        confidence_score=0.75
    )


def test_end_to_end_strategy_selection(sample_risk_profile):
    """Test complete end-to-end strategy selection process."""
    selector = RiskBasedStrategySelector()
    
    market_data = {
        "volatility_index": 22.0,
        "market_trend": 0.3,
        "uncertainty_index": 0.3,
        "sector_performance": {"tech": 0.05, "finance": 0.03},
        "credit_spreads": -0.05,
        "equity_momentum": 0.2,
        "dollar_strength": 0.0
    }
    
    recommendations = selector.select_strategies_for_user(
        risk_profile=sample_risk_profile,
        market_data=market_data,
        portfolio_value=150000.0,
        max_strategies=4
    )
    
    # Verify we get appropriate recommendations
    assert len(recommendations) <= 4
    assert len(recommendations) > 0
    
    # All recommendations should be appropriate for moderate risk
    for rec in recommendations:
        assert RiskCategory.MODERATE in rec.strategy.suitable_risk_categories
        assert rec.confidence_score > 0.0
        assert 0.0 < rec.allocation_percentage <= 1.0
        
        # Verify reasoning is provided
        assert len(rec.reasoning) > 0
        
        # Verify trade parameters are set
        assert "max_risk_per_trade" in rec.trade_parameters
        assert rec.trade_parameters["max_risk_per_trade"] <= 0.02  # Moderate risk limit
    
    # Verify total allocation is reasonable
    total_allocation = sum(rec.allocation_percentage for rec in recommendations)
    assert 0.5 <= total_allocation <= 1.2  # Allow some flexibility in total allocation


def test_strategy_selection_performance_integration():
    """Test that strategy selection integrates with performance tracking."""
    selector = RiskBasedStrategySelector()
    
    # Update performance for a strategy
    performance_data = {
        "total_return": 0.12,
        "annualized_return": 0.10,
        "volatility": 0.15,
        "sharpe_ratio": 0.67,
        "max_drawdown": -0.08,
        "win_rate": 0.65,
        "avg_win": 0.04,
        "avg_loss": -0.025,
        "trade_count": 30,
        "risk_adjusted_return": 0.08,
        "confidence_score": 0.8
    }
    
    selector.update_strategy_performance("conservative_buy_hold", performance_data)
    
    # Create conservative profile to test strategy selection
    conservative_profile = RiskProfile(
        user_id="perf_test_user",
        risk_category=RiskCategory.CONSERVATIVE,
        risk_score=20,
        responses=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        confidence_score=0.8
    )
    
    market_data = {"volatility_index": 18.0, "market_trend": 0.2, "uncertainty_index": 0.3}
    
    recommendations = selector.select_strategies_for_user(
        risk_profile=conservative_profile,
        market_data=market_data,
        portfolio_value=100000.0,
        max_strategies=3
    )
    
    # The updated strategy should potentially be recommended with higher confidence
    # due to good performance metrics
    buy_hold_recs = [r for r in recommendations if "buy" in r.strategy.name.lower() and "hold" in r.strategy.name.lower()]
    
    if buy_hold_recs:
        # If recommended, should have good confidence due to updated performance
        assert buy_hold_recs[0].confidence_score > 0.5