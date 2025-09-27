"""Risk-Based Strategy Selection Logic for AI Trading Advisor.

This module provides comprehensive strategy selection based on:
- User risk profiles (Conservative/Moderate/Aggressive)
- Market conditions and volatility regimes
- Strategy performance tracking by risk level
- Dynamic strategy filtering and allocation
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from .user_profiling import RiskCategory, RiskProfile, UserProfilingEngine

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Trading strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    BUY_AND_HOLD = "buy_and_hold"
    VALUE_INVESTING = "value_investing"
    GROWTH_INVESTING = "growth_investing"
    DIVIDEND_INVESTING = "dividend_investing"
    PAIRS_TRADING = "pairs_trading"
    VOLATILITY_TRADING = "volatility_trading"
    TREND_FOLLOWING = "trend_following"


class MarketRegime(str, Enum):
    """Market condition regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class StrategyMetrics:
    """Performance metrics for a trading strategy."""
    strategy_id: str
    strategy_type: StrategyType
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trade_count: int
    risk_adjusted_return: float
    last_updated: datetime
    confidence_score: float = field(default=0.5)


@dataclass
class Strategy:
    """Trading strategy definition."""
    id: str
    name: str
    strategy_type: StrategyType
    description: str
    risk_level: float  # 0.0-1.0 scale
    min_investment_horizon_days: int
    max_position_size: float
    volatility_tolerance: float
    market_conditions: List[MarketRegime]
    suitable_risk_categories: List[RiskCategory]
    expected_return: float
    expected_volatility: float
    min_confidence_threshold: float
    performance_metrics: Optional[StrategyMetrics] = field(default=None)
    is_active: bool = field(default=True)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyRecommendation:
    """Strategy recommendation for a user."""
    strategy: Strategy
    allocation_percentage: float
    confidence_score: float
    reasoning: str
    expected_return: float
    expected_risk: float
    market_context: Dict[str, Any]
    trade_parameters: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MarketCondition:
    """Current market conditions assessment."""
    regime: MarketRegime
    volatility_level: float  # VIX-like measure
    trend_strength: float  # -1.0 (strong bear) to 1.0 (strong bull)
    uncertainty_index: float  # 0.0-1.0
    sector_rotation: Dict[str, float]  # Sector performance
    risk_on_sentiment: float  # -1.0 (risk off) to 1.0 (risk on)
    last_updated: datetime = field(default_factory=datetime.now)


class StrategyPerformanceTracker:
    """Tracks strategy performance by risk level and market conditions."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[StrategyMetrics]] = {}
        self.risk_level_performance: Dict[RiskCategory, Dict[str, StrategyMetrics]] = {
            RiskCategory.CONSERVATIVE: {},
            RiskCategory.MODERATE: {},
            RiskCategory.AGGRESSIVE: {}
        }
    
    def update_strategy_performance(self, metrics: StrategyMetrics, risk_category: RiskCategory):
        """Update strategy performance metrics."""
        strategy_id = metrics.strategy_id
        
        # Update overall performance history
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        
        self.performance_history[strategy_id].append(metrics)
        
        # Update risk-level performance
        self.risk_level_performance[risk_category][strategy_id] = metrics
        
        logger.info(f"Updated performance for {strategy_id} in {risk_category.value} category")
    
    def get_strategy_performance(self, strategy_id: str, risk_category: RiskCategory) -> Optional[StrategyMetrics]:
        """Get latest performance metrics for a strategy at specific risk level."""
        return self.risk_level_performance[risk_category].get(strategy_id)
    
    def get_top_performing_strategies(self, risk_category: RiskCategory, limit: int = 5) -> List[StrategyMetrics]:
        """Get top performing strategies for a risk category."""
        strategies = list(self.risk_level_performance[risk_category].values())
        
        # Sort by risk-adjusted return
        strategies.sort(key=lambda x: x.risk_adjusted_return, reverse=True)
        
        return strategies[:limit]
    
    def calculate_strategy_consistency(self, strategy_id: str, lookback_periods: int = 12) -> float:
        """Calculate strategy consistency over time."""
        if strategy_id not in self.performance_history:
            return 0.0
        
        history = self.performance_history[strategy_id][-lookback_periods:]
        if len(history) < 2:
            return 0.5
        
        returns = [h.annualized_return for h in history]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Consistency score based on standard deviation of returns
        consistency = max(0.0, min(1.0, 1.0 - (std_return / (abs(mean_return) + 0.01))))
        
        return consistency


class MarketConditionAnalyzer:
    """Analyzes current market conditions for strategy selection."""
    
    def __init__(self):
        self.volatility_thresholds = {
            "low": 15.0,
            "moderate": 25.0,
            "high": 40.0
        }
    
    def assess_current_conditions(self, market_data: Dict[str, Any]) -> MarketCondition:
        """Assess current market conditions."""
        # Extract key indicators from market data
        vix_level = market_data.get("volatility_index", 20.0)
        market_trend = market_data.get("market_trend", 0.0)
        uncertainty = market_data.get("uncertainty_index", 0.3)
        
        # Determine market regime
        regime = self._determine_market_regime(vix_level, market_trend, uncertainty)
        
        # Calculate risk sentiment
        risk_sentiment = self._calculate_risk_sentiment(market_data)
        
        return MarketCondition(
            regime=regime,
            volatility_level=vix_level,
            trend_strength=market_trend,
            uncertainty_index=uncertainty,
            sector_rotation=market_data.get("sector_performance", {}),
            risk_on_sentiment=risk_sentiment
        )
    
    def _determine_market_regime(self, volatility: float, trend: float, uncertainty: float) -> MarketRegime:
        """Determine market regime based on indicators."""
        if volatility > 35.0 or uncertainty > 0.7:
            return MarketRegime.CRISIS
        elif volatility > 25.0:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 15.0:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.3:
            return MarketRegime.BULL_MARKET
        elif trend < -0.3:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _calculate_risk_sentiment(self, market_data: Dict[str, Any]) -> float:
        """Calculate risk-on/risk-off sentiment."""
        # Combine multiple factors for risk sentiment
        factors = [
            market_data.get("credit_spreads", 0.0) * -1.0,  # Lower spreads = risk on
            market_data.get("equity_momentum", 0.0),
            market_data.get("dollar_strength", 0.0) * -0.5,  # Strong dollar can be risk off
        ]
        
        # Average the factors and normalize to -1 to 1
        sentiment = np.mean([f for f in factors if f != 0.0]) if any(factors) else 0.0
        return max(-1.0, min(1.0, sentiment))


class StrategyDatabase:
    """Database of available trading strategies."""
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default set of trading strategies."""
        strategies = [
            # Conservative strategies
            Strategy(
                id="conservative_buy_hold",
                name="Conservative Buy & Hold",
                strategy_type=StrategyType.BUY_AND_HOLD,
                description="Long-term diversified portfolio with blue-chip stocks and bonds",
                risk_level=0.2,
                min_investment_horizon_days=365,
                max_position_size=0.05,
                volatility_tolerance=0.15,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS_MARKET, MarketRegime.LOW_VOLATILITY],
                suitable_risk_categories=[RiskCategory.CONSERVATIVE],
                expected_return=0.08,
                expected_volatility=0.12,
                min_confidence_threshold=0.7
            ),
            Strategy(
                id="dividend_income",
                name="Dividend Income Strategy",
                strategy_type=StrategyType.DIVIDEND_INVESTING,
                description="Focus on high-quality dividend-paying stocks",
                risk_level=0.3,
                min_investment_horizon_days=180,
                max_position_size=0.08,
                volatility_tolerance=0.20,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS_MARKET, MarketRegime.BEAR_MARKET],
                suitable_risk_categories=[RiskCategory.CONSERVATIVE, RiskCategory.MODERATE],
                expected_return=0.09,
                expected_volatility=0.15,
                min_confidence_threshold=0.6
            ),
            
            # Moderate strategies
            Strategy(
                id="balanced_growth",
                name="Balanced Growth Strategy",
                strategy_type=StrategyType.GROWTH_INVESTING,
                description="Balanced approach mixing growth and value stocks",
                risk_level=0.5,
                min_investment_horizon_days=90,
                max_position_size=0.10,
                volatility_tolerance=0.25,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS_MARKET],
                suitable_risk_categories=[RiskCategory.MODERATE],
                expected_return=0.12,
                expected_volatility=0.18,
                min_confidence_threshold=0.5
            ),
            Strategy(
                id="mean_reversion",
                name="Mean Reversion Strategy",
                strategy_type=StrategyType.MEAN_REVERSION,
                description="Capitalize on price reversions to historical means",
                risk_level=0.4,
                min_investment_horizon_days=30,
                max_position_size=0.08,
                volatility_tolerance=0.30,
                market_conditions=[MarketRegime.SIDEWAYS_MARKET, MarketRegime.HIGH_VOLATILITY],
                suitable_risk_categories=[RiskCategory.MODERATE],
                expected_return=0.15,
                expected_volatility=0.22,
                min_confidence_threshold=0.6
            ),
            
            # Aggressive strategies
            Strategy(
                id="momentum_trading",
                name="Momentum Trading Strategy",
                strategy_type=StrategyType.MOMENTUM,
                description="Follow strong price trends and momentum signals",
                risk_level=0.7,
                min_investment_horizon_days=14,
                max_position_size=0.15,
                volatility_tolerance=0.35,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET, MarketRegime.HIGH_VOLATILITY],
                suitable_risk_categories=[RiskCategory.AGGRESSIVE],
                expected_return=0.20,
                expected_volatility=0.30,
                min_confidence_threshold=0.4
            ),
            Strategy(
                id="breakout_trading",
                name="Breakout Trading Strategy",
                strategy_type=StrategyType.BREAKOUT,
                description="Trade breakouts from key technical levels",
                risk_level=0.8,
                min_investment_horizon_days=7,
                max_position_size=0.12,
                volatility_tolerance=0.40,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.HIGH_VOLATILITY],
                suitable_risk_categories=[RiskCategory.AGGRESSIVE],
                expected_return=0.25,
                expected_volatility=0.35,
                min_confidence_threshold=0.3
            ),
            Strategy(
                id="volatility_trading",
                name="Volatility Trading Strategy",
                strategy_type=StrategyType.VOLATILITY_TRADING,
                description="Profit from volatility expansion and contraction",
                risk_level=0.9,
                min_investment_horizon_days=3,
                max_position_size=0.10,
                volatility_tolerance=0.50,
                market_conditions=[MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS],
                suitable_risk_categories=[RiskCategory.AGGRESSIVE],
                expected_return=0.18,
                expected_volatility=0.45,
                min_confidence_threshold=0.5
            ),
        ]
        
        for strategy in strategies:
            self.strategies[strategy.id] = strategy
            
        logger.info(f"Initialized {len(strategies)} default strategies")
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def get_strategies_by_risk_category(self, risk_category: RiskCategory) -> List[Strategy]:
        """Get all strategies suitable for a risk category."""
        return [
            strategy for strategy in self.strategies.values()
            if risk_category in strategy.suitable_risk_categories and strategy.is_active
        ]
    
    def get_strategies_by_market_condition(self, market_condition: MarketRegime) -> List[Strategy]:
        """Get strategies suitable for current market conditions."""
        return [
            strategy for strategy in self.strategies.values()
            if market_condition in strategy.market_conditions and strategy.is_active
        ]
    
    def add_strategy(self, strategy: Strategy):
        """Add a new strategy to the database."""
        self.strategies[strategy.id] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def update_strategy_performance(self, strategy_id: str, metrics: StrategyMetrics):
        """Update strategy performance metrics."""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].performance_metrics = metrics


class RiskBasedStrategySelector:
    """Main engine for risk-based strategy selection."""
    
    def __init__(self):
        self.strategy_db = StrategyDatabase()
        self.performance_tracker = StrategyPerformanceTracker()
        self.market_analyzer = MarketConditionAnalyzer()
        self.user_profiling_engine = UserProfilingEngine()
    
    def select_strategies_for_user(
        self,
        risk_profile: RiskProfile,
        market_data: Dict[str, Any],
        portfolio_value: float,
        max_strategies: int = 5
    ) -> List[StrategyRecommendation]:
        """Select optimal strategies for a user based on risk profile and market conditions."""
        
        # Assess current market conditions
        market_condition = self.market_analyzer.assess_current_conditions(market_data)
        
        # Get candidate strategies
        risk_suitable = self.strategy_db.get_strategies_by_risk_category(risk_profile.risk_category)
        market_suitable = self.strategy_db.get_strategies_by_market_condition(market_condition.regime)
        
        # Find intersection of suitable strategies
        candidate_strategies = [s for s in risk_suitable if s in market_suitable]
        
        if not candidate_strategies:
            logger.warning(f"No suitable strategies found for {risk_profile.risk_category.value} in {market_condition.regime.value}")
            # Fallback to risk-appropriate strategies regardless of market condition
            candidate_strategies = risk_suitable
        
        # Score and rank strategies
        strategy_scores = []
        for strategy in candidate_strategies:
            score = self._score_strategy(strategy, risk_profile, market_condition, portfolio_value)
            strategy_scores.append((strategy, score))
        
        # Sort by score and select top strategies
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        selected_strategies = strategy_scores[:max_strategies]
        
        # Generate recommendations with allocations
        recommendations = []
        total_allocation = 0.0
        
        for i, (strategy, score) in enumerate(selected_strategies):
            # Allocate more to higher-scoring strategies
            base_allocation = 1.0 / len(selected_strategies)
            score_adjustment = (score - 0.5) * 0.3  # Reduced adjustment to prevent extreme allocations
            allocation = max(0.05, min(0.4, base_allocation + score_adjustment))
            
            # Ensure we don't over-allocate
            remaining_allocation = 1.0 - total_allocation
            if remaining_allocation <= 0.01:  # Very little remaining
                allocation = max(0.01, remaining_allocation)
            else:
                allocation = min(allocation, remaining_allocation)
            
            total_allocation += allocation
            
            recommendation = self._create_strategy_recommendation(
                strategy, allocation, score, risk_profile, market_condition, portfolio_value
            )
            recommendations.append(recommendation)
        
        # Normalize allocations to ensure they sum to reasonable total
        if total_allocation > 0 and total_allocation < 0.8:  # If total is too low, normalize to 0.8
            target_allocation = min(1.0, 0.8)
            scaling_factor = target_allocation / total_allocation
            for rec in recommendations:
                rec.allocation_percentage = rec.allocation_percentage * scaling_factor
        elif total_allocation > 1.0:  # If over-allocated, normalize to 1.0
            for rec in recommendations:
                rec.allocation_percentage = rec.allocation_percentage / total_allocation
        
        logger.info(f"Selected {len(recommendations)} strategies for {risk_profile.risk_category.value} user")
        return recommendations
    
    def _score_strategy(
        self,
        strategy: Strategy,
        risk_profile: RiskProfile,
        market_condition: MarketCondition,
        portfolio_value: float
    ) -> float:
        """Score a strategy based on various factors."""
        score = 0.0
        
        # Risk alignment score (30% weight)
        risk_alignment = self._calculate_risk_alignment_score(strategy, risk_profile)
        score += risk_alignment * 0.30
        
        # Market condition suitability (25% weight)
        market_suitability = self._calculate_market_suitability_score(strategy, market_condition)
        score += market_suitability * 0.25
        
        # Performance score (25% weight)
        performance_score = self._calculate_performance_score(strategy, risk_profile.risk_category)
        score += performance_score * 0.25
        
        # Portfolio fit score (20% weight)
        portfolio_fit = self._calculate_portfolio_fit_score(strategy, portfolio_value)
        score += portfolio_fit * 0.20
        
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_alignment_score(self, strategy: Strategy, risk_profile: RiskProfile) -> float:
        """Calculate how well strategy aligns with user risk profile."""
        # Get trading parameters for user's risk category
        trading_params = self.user_profiling_engine.get_trading_parameters(
            risk_profile.risk_category, risk_profile.confidence_score
        )
        
        # Check alignment with key risk parameters
        alignment_scores = []
        
        # Position size alignment
        max_position = trading_params["max_position_size"]
        if strategy.max_position_size <= max_position:
            alignment_scores.append(1.0)
        else:
            alignment_scores.append(max_position / strategy.max_position_size)
        
        # Volatility tolerance alignment
        volatility_threshold = trading_params["volatility_threshold"]
        if strategy.volatility_tolerance <= volatility_threshold:
            alignment_scores.append(1.0)
        else:
            alignment_scores.append(volatility_threshold / strategy.volatility_tolerance)
        
        # Investment horizon alignment
        if risk_profile.risk_category == RiskCategory.CONSERVATIVE:
            if strategy.min_investment_horizon_days >= 90:
                alignment_scores.append(1.0)
            else:
                alignment_scores.append(strategy.min_investment_horizon_days / 90)
        elif risk_profile.risk_category == RiskCategory.MODERATE:
            if 30 <= strategy.min_investment_horizon_days <= 180:
                alignment_scores.append(1.0)
            else:
                alignment_scores.append(0.7)
        else:  # AGGRESSIVE
            alignment_scores.append(1.0)  # Aggressive users are flexible on horizon
        
        return np.mean(alignment_scores)
    
    def _calculate_market_suitability_score(self, strategy: Strategy, market_condition: MarketCondition) -> float:
        """Calculate strategy suitability for current market conditions."""
        base_score = 0.8 if market_condition.regime in strategy.market_conditions else 0.3
        
        # Adjust based on volatility environment
        volatility_adjustment = 0.0
        if strategy.volatility_tolerance >= market_condition.volatility_level / 100:
            volatility_adjustment = 0.2
        elif strategy.volatility_tolerance < market_condition.volatility_level / 200:
            volatility_adjustment = -0.2
        
        # Adjust based on trend strength for trend-sensitive strategies
        trend_adjustment = 0.0
        if strategy.strategy_type in [StrategyType.MOMENTUM, StrategyType.TREND_FOLLOWING]:
            trend_adjustment = abs(market_condition.trend_strength) * 0.1
        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            trend_adjustment = (1.0 - abs(market_condition.trend_strength)) * 0.1
        
        return max(0.0, min(1.0, base_score + volatility_adjustment + trend_adjustment))
    
    def _calculate_performance_score(self, strategy: Strategy, risk_category: RiskCategory) -> float:
        """Calculate performance score based on historical metrics."""
        metrics = self.performance_tracker.get_strategy_performance(strategy.id, risk_category)
        
        if not metrics:
            return 0.5  # Neutral score for strategies without performance history
        
        # Combine multiple performance factors
        performance_factors = []
        
        # Risk-adjusted return
        if metrics.risk_adjusted_return > 0:
            performance_factors.append(min(1.0, metrics.risk_adjusted_return / 0.5))  # Cap at 50% risk-adjusted return
        
        # Sharpe ratio
        if metrics.sharpe_ratio > 0:
            performance_factors.append(min(1.0, metrics.sharpe_ratio / 2.0))  # Cap at Sharpe ratio of 2.0
        
        # Win rate
        performance_factors.append(metrics.win_rate)
        
        # Drawdown penalty
        max_acceptable_drawdown = 0.5 if risk_category == RiskCategory.AGGRESSIVE else (0.3 if risk_category == RiskCategory.MODERATE else 0.15)
        drawdown_score = max(0.0, 1.0 - (abs(metrics.max_drawdown) / max_acceptable_drawdown))
        performance_factors.append(drawdown_score)
        
        # Consistency score
        consistency = self.performance_tracker.calculate_strategy_consistency(strategy.id)
        performance_factors.append(consistency)
        
        return np.mean(performance_factors) if performance_factors else 0.5
    
    def _calculate_portfolio_fit_score(self, strategy: Strategy, portfolio_value: float) -> float:
        """Calculate how well strategy fits portfolio constraints."""
        # Base score
        score = 0.7
        
        # Adjust based on minimum investment requirements
        min_investment = strategy.max_position_size * 10000  # Assume minimum $10k for max position calculation
        
        if portfolio_value >= min_investment * 2:  # Comfortable fit
            score += 0.3
        elif portfolio_value >= min_investment:  # Adequate fit
            score += 0.1
        else:  # Tight fit
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _create_strategy_recommendation(
        self,
        strategy: Strategy,
        allocation: float,
        confidence: float,
        risk_profile: RiskProfile,
        market_condition: MarketCondition,
        portfolio_value: float
    ) -> StrategyRecommendation:
        """Create a strategy recommendation with detailed information."""
        
        # Get risk-adjusted trading parameters
        trading_params = self.user_profiling_engine.get_trading_parameters(
            risk_profile.risk_category, risk_profile.confidence_score
        )
        
        # Adjust strategy parameters to user risk profile
        adjusted_params = trading_params.copy()
        adjusted_params["max_position_size"] = min(
            trading_params["max_position_size"],
            strategy.max_position_size
        )
        
        # Generate reasoning
        reasoning = self._generate_recommendation_reasoning(
            strategy, risk_profile, market_condition, confidence
        )
        
        # Calculate expected metrics
        expected_return = strategy.expected_return * (confidence ** 0.5)  # Adjust for confidence
        expected_risk = strategy.expected_volatility * (1.0 + (1.0 - confidence) * 0.5)
        
        return StrategyRecommendation(
            strategy=strategy,
            allocation_percentage=allocation,
            confidence_score=confidence,
            reasoning=reasoning,
            expected_return=expected_return,
            expected_risk=expected_risk,
            market_context={
                "regime": market_condition.regime.value,
                "volatility_level": market_condition.volatility_level,
                "trend_strength": market_condition.trend_strength,
                "risk_sentiment": market_condition.risk_on_sentiment
            },
            trade_parameters=adjusted_params
        )
    
    def _generate_recommendation_reasoning(
        self,
        strategy: Strategy,
        risk_profile: RiskProfile,
        market_condition: MarketCondition,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for strategy recommendation."""
        
        reasons = []
        
        # Risk alignment
        reasons.append(f"Aligned with {risk_profile.risk_category.value} risk profile")
        
        # Market conditions
        if market_condition.regime in strategy.market_conditions:
            reasons.append(f"Well-suited for {market_condition.regime.value.replace('_', ' ')} conditions")
        
        # Strategy characteristics
        if strategy.strategy_type == StrategyType.BUY_AND_HOLD:
            reasons.append("Long-term approach reduces transaction costs and market timing risk")
        elif strategy.strategy_type == StrategyType.DIVIDEND_INVESTING:
            reasons.append("Provides steady income with lower volatility")
        elif strategy.strategy_type == StrategyType.MOMENTUM:
            reasons.append("Can capitalize on strong market trends")
        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            reasons.append("Benefits from market volatility and price corrections")
        
        # Performance note
        if confidence > 0.7:
            reasons.append("Strong historical performance indicators")
        elif confidence < 0.4:
            reasons.append("Consider smaller allocation due to uncertainty")
        
        return "; ".join(reasons)
    
    def filter_strategies_by_risk(
        self,
        strategies: List[Strategy],
        risk_profile: RiskProfile,
        market_condition: MarketCondition
    ) -> List[Strategy]:
        """Filter strategies to only include risk-appropriate ones."""
        
        filtered = []
        trading_params = self.user_profiling_engine.get_trading_parameters(risk_profile.risk_category)
        
        for strategy in strategies:
            # Check if strategy is suitable for user's risk category
            if risk_profile.risk_category not in strategy.suitable_risk_categories:
                continue
            
            # Check position size limits
            if strategy.max_position_size > trading_params["max_position_size"]:
                continue
            
            # Check volatility tolerance
            if strategy.volatility_tolerance > trading_params["volatility_threshold"]:
                continue
            
            # Check market condition suitability
            if market_condition.regime not in strategy.market_conditions:
                # Allow some flexibility for conservative strategies in uncertain conditions
                if (risk_profile.risk_category == RiskCategory.CONSERVATIVE and 
                    strategy.risk_level < 0.3):
                    pass  # Allow conservative strategies
                else:
                    continue
            
            filtered.append(strategy)
        
        return filtered
    
    def update_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """Update strategy performance metrics."""
        
        metrics = StrategyMetrics(
            strategy_id=strategy_id,
            strategy_type=StrategyType(performance_data.get("strategy_type", "buy_and_hold")),
            total_return=performance_data.get("total_return", 0.0),
            annualized_return=performance_data.get("annualized_return", 0.0),
            volatility=performance_data.get("volatility", 0.0),
            sharpe_ratio=performance_data.get("sharpe_ratio", 0.0),
            max_drawdown=performance_data.get("max_drawdown", 0.0),
            win_rate=performance_data.get("win_rate", 0.5),
            avg_win=performance_data.get("avg_win", 0.0),
            avg_loss=performance_data.get("avg_loss", 0.0),
            trade_count=performance_data.get("trade_count", 0),
            risk_adjusted_return=performance_data.get("risk_adjusted_return", 0.0),
            last_updated=datetime.now(),
            confidence_score=performance_data.get("confidence_score", 0.5)
        )
        
        # Update performance for all risk categories where strategy is suitable
        strategy = self.strategy_db.get_strategy(strategy_id)
        if strategy:
            for risk_category in strategy.suitable_risk_categories:
                self.performance_tracker.update_strategy_performance(metrics, risk_category)
            
            # Update strategy database
            self.strategy_db.update_strategy_performance(strategy_id, metrics)
    
    def get_strategy_allocation_limits(self, risk_profile: RiskProfile) -> Dict[str, float]:
        """Get strategy allocation limits based on risk profile."""
        
        if risk_profile.risk_category == RiskCategory.CONSERVATIVE:
            return {
                "max_single_strategy": 0.40,
                "max_aggressive_strategies": 0.10,
                "min_conservative_allocation": 0.60,
                "max_alternative_strategies": 0.05
            }
        elif risk_profile.risk_category == RiskCategory.MODERATE:
            return {
                "max_single_strategy": 0.30,
                "max_aggressive_strategies": 0.25,
                "min_conservative_allocation": 0.30,
                "max_alternative_strategies": 0.15
            }
        else:  # AGGRESSIVE
            return {
                "max_single_strategy": 0.50,
                "max_aggressive_strategies": 0.70,
                "min_conservative_allocation": 0.10,
                "max_alternative_strategies": 0.30
            }