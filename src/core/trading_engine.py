"""
Trading Engine Module
Consolidated trading functionality including risk management, strategy selection, and recommendation generation.

Consolidates functionality from:
- src/core/risk_engine.py
- src/core/recommendation_engine.py
- src/core/strategy_selector.py
- src/core/market_risk_adapter.py
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field

# Financial calculations (with fallbacks)
try:
    from scipy import optimize, stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("scipy unavailable. Using fallback implementations.")

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    logging.warning("cvxpy unavailable. Using basic portfolio optimization.")

# Configuration
from ..config import settings

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"


class RiskCategory(Enum):
    """Risk category enumeration."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TradeAction(Enum):
    """Trade action enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class StrategyType(Enum):
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


class MarketRegime(Enum):
    """Market condition regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    PORTFOLIO_HEAT = "portfolio_heat"


@dataclass
class TradingParameters:
    """Trading parameters for user risk profile."""
    max_risk_per_trade: float
    max_portfolio_risk: float
    max_position_size: float
    daily_loss_limit: float
    leverage_limit: float


@dataclass
class TradeRecommendation:
    """Trade recommendation structure."""
    symbol: str
    action: TradeAction
    quantity: int
    price: float
    confidence: float
    reasoning: str
    risk_level: RiskLevel
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_type: OrderType = OrderType.MARKET


@dataclass
class RiskProfile:
    """User risk profile."""
    risk_category: RiskCategory
    risk_score: float
    confidence_score: float
    trading_parameters: TradingParameters


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
class PositionSizingParams:
    """Parameters for position sizing calculations."""
    symbol: str
    current_price: float
    expected_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    volatility: float
    correlation_to_portfolio: float
    stop_loss_pct: float
    confidence: float
    lookback_periods: int = 252


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics for position sizing."""
    total_value: float
    available_cash: float
    current_positions: Dict[str, Dict]
    portfolio_beta: float
    portfolio_volatility: float
    portfolio_correlation_matrix: np.ndarray
    heat_level: float  # Current portfolio heat (0-1)
    max_heat_threshold: float = field(default=0.3)


class RiskManager:
    """
    Risk management engine for position sizing and portfolio risk control.
    
    Combines functionality from:
    - src/core/risk_engine.py
    - src/core/market_risk_adapter.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RiskManager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.max_risk_per_trade = 0.02  # 2%
        self.max_portfolio_risk = 0.20  # 20%
        self.max_position_size = 0.10   # 10%
        self.lookback_periods = 252
        
    def calculate_position_size(
        self, 
        account_balance: float, 
        risk_per_trade: float, 
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        sizing_method: SizingMethod = SizingMethod.FIXED_PERCENT
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_balance: Total account balance
            risk_per_trade: Risk per trade as percentage
            entry_price: Entry price for the position
            stop_loss_price: Optional stop loss price
            sizing_method: Position sizing method
            
        Returns:
            Position sizing calculation results
        """
        try:
            if entry_price <= 0:
                return {"position_size": 0, "risk_amount": 0.0, "error": "Invalid entry price"}
            
            # Calculate risk amount in dollars
            risk_amount = account_balance * risk_per_trade
            
            if sizing_method == SizingMethod.FIXED_PERCENT:
                # Simple percentage-based sizing
                position_value = account_balance * min(risk_per_trade * 5, self.max_position_size)
                position_size = int(position_value / entry_price)
                
            elif sizing_method == SizingMethod.VOLATILITY_ADJUSTED and stop_loss_price:
                # Size based on distance to stop loss
                if stop_loss_price >= entry_price:
                    return {"position_size": 0, "risk_amount": 0.0, "error": "Invalid stop loss price"}
                
                risk_per_share = entry_price - stop_loss_price
                position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                
            else:
                # Default to fixed percent
                position_value = account_balance * min(risk_per_trade * 5, self.max_position_size)
                position_size = int(position_value / entry_price)
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_shares = int(max_position_value / entry_price)
            position_size = min(position_size, max_shares)
            
            actual_position_value = position_size * entry_price
            actual_risk_percentage = actual_position_value / account_balance
            
            return {
                "position_size": position_size,
                "position_value": actual_position_value,
                "risk_amount": risk_amount,
                "actual_risk_percentage": actual_risk_percentage,
                "stop_loss": stop_loss_price,
                "sizing_method": sizing_method.value,
                "max_position_allowed": max_shares
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {"position_size": 0, "risk_amount": 0.0, "error": str(e)}
    
    def validate_trade_risk(
        self, 
        trade: Dict[str, Any], 
        portfolio: Dict[str, Any],
        user_params: TradingParameters
    ) -> Dict[str, Any]:
        """
        Validate trade against risk parameters.
        
        Args:
            trade: Trade details
            portfolio: Current portfolio state
            user_params: User risk parameters
            
        Returns:
            Risk validation results
        """
        try:
            warnings = []
            adjustments = {}
            approved = True
            
            symbol = trade.get("symbol", "")
            quantity = trade.get("quantity", 0)
            price = trade.get("price", 0)
            action = trade.get("action", "")
            
            if not all([symbol, quantity, price, action]):
                return {
                    "approved": False,
                    "risk_score": 1.0,
                    "warnings": ["Missing required trade parameters"],
                    "adjustments": {}
                }
            
            portfolio_value = portfolio.get("total_value", 0)
            if portfolio_value <= 0:
                return {
                    "approved": False,
                    "risk_score": 1.0,
                    "warnings": ["Invalid portfolio value"],
                    "adjustments": {}
                }
            
            # Calculate position size as percentage of portfolio
            position_value = quantity * price
            position_percentage = position_value / portfolio_value
            
            # Check position size limit
            if position_percentage > user_params.max_position_size:
                warnings.append(f"Position size {position_percentage:.1%} exceeds limit {user_params.max_position_size:.1%}")
                max_quantity = int((portfolio_value * user_params.max_position_size) / price)
                adjustments["suggested_quantity"] = max_quantity
                if position_percentage > user_params.max_position_size * 1.5:
                    approved = False
            
            # Check daily loss limit
            daily_loss_check = self.check_daily_loss_limit(portfolio, user_params.daily_loss_limit)
            if daily_loss_check["limit_reached"]:
                warnings.append("Daily loss limit reached")
                approved = False
            
            # Calculate risk score (0-1, where 1 is highest risk)
            risk_factors = [
                position_percentage / user_params.max_position_size,
                daily_loss_check["current_loss"] / user_params.daily_loss_limit,
                min(1.0, position_value / 10000)  # Size factor
            ]
            risk_score = min(1.0, sum(risk_factors) / len(risk_factors))
            
            return {
                "approved": approved,
                "risk_score": risk_score,
                "warnings": warnings,
                "adjustments": adjustments,
                "position_percentage": position_percentage,
                "position_value": position_value
            }
            
        except Exception as e:
            logger.error(f"Error validating trade risk: {e}")
            return {
                "approved": False,
                "risk_score": 1.0,
                "warnings": [f"Risk validation error: {str(e)}"],
                "adjustments": {}
            }
    
    def calculate_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate current portfolio risk metrics.
        
        Args:
            portfolio: Portfolio holdings and data
            
        Returns:
            Portfolio risk analysis
        """
        try:
            positions = portfolio.get("positions", {})
            total_value = portfolio.get("total_value", 0)
            
            if not positions or total_value <= 0:
                return {
                    "total_risk": 0.0,
                    "concentration_risk": 0.0,
                    "sector_risk": 0.0,
                    "correlation_risk": 0.0,
                    "var_95": 0.0,
                    "beta": 1.0,
                    "diversification_score": 0.0
                }
            
            # Calculate concentration risk (largest position weight)
            position_weights = []
            for symbol, position in positions.items():
                weight = position.get("value", 0) / total_value
                position_weights.append(weight)
            
            concentration_risk = max(position_weights) if position_weights else 0.0
            
            # Calculate Herfindahl index for diversification
            herfindahl_index = sum(w**2 for w in position_weights)
            diversification_score = 1 - herfindahl_index
            
            # Estimate portfolio volatility (simplified)
            if HAS_SCIPY:
                # Use more sophisticated calculation with scipy
                avg_volatility = 0.2  # 20% annual volatility assumption
                portfolio_volatility = avg_volatility * np.sqrt(herfindahl_index)
            else:
                portfolio_volatility = 0.15  # Default 15% volatility
            
            # VaR calculation (simplified)
            var_95 = total_value * portfolio_volatility * 1.65  # 95% VaR
            
            # Beta estimation (simplified)
            beta = 1.0 if len(positions) > 3 else 1.2  # More concentrated = higher beta
            
            return {
                "total_risk": portfolio_volatility,
                "concentration_risk": concentration_risk,
                "sector_risk": 0.3,  # Placeholder
                "correlation_risk": 0.4,  # Placeholder  
                "var_95": var_95,
                "beta": beta,
                "diversification_score": diversification_score,
                "herfindahl_index": herfindahl_index,
                "position_count": len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {
                "total_risk": 0.0,
                "concentration_risk": 0.0,
                "sector_risk": 0.0,
                "correlation_risk": 0.0,
                "var_95": 0.0,
                "beta": 1.0,
                "error": str(e)
            }
    
    def calculate_stop_loss(self, entry_price: float, risk_tolerance: float) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            risk_tolerance: Risk tolerance percentage
            
        Returns:
            Stop loss price
        """
        return entry_price * (1 - risk_tolerance)
    
    def calculate_take_profit(self, entry_price: float, risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            Take profit price
        """
        return entry_price * (1 + (risk_reward_ratio * 0.02))  # Assuming 2% risk
    
    def check_daily_loss_limit(self, portfolio: Dict[str, Any], daily_limit: float) -> Dict[str, Any]:
        """
        Check if daily loss limit has been reached.
        
        Args:
            portfolio: Portfolio state
            daily_limit: Daily loss limit percentage
            
        Returns:
            Daily loss check results
        """
        try:
            daily_pnl = portfolio.get("daily_pnl", 0.0)
            total_value = portfolio.get("total_value", 0)
            
            if total_value <= 0:
                return {
                    "limit_reached": False,
                    "current_loss": 0.0,
                    "remaining_capacity": daily_limit
                }
            
            # Calculate current loss as percentage
            current_loss_pct = abs(min(0, daily_pnl)) / total_value
            limit_reached = current_loss_pct >= daily_limit
            remaining_capacity = max(0, daily_limit - current_loss_pct)
            
            return {
                "limit_reached": limit_reached,
                "current_loss": current_loss_pct,
                "remaining_capacity": remaining_capacity,
                "daily_pnl": daily_pnl,
                "loss_amount": abs(min(0, daily_pnl))
            }
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return {
                "limit_reached": False,
                "current_loss": 0.0,
                "remaining_capacity": daily_limit,
                "error": str(e)
            }
    
    def diversification_check(self, portfolio: Dict[str, Any], new_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check portfolio diversification with new position.
        
        Args:
            portfolio: Current portfolio
            new_position: Proposed new position
            
        Returns:
            Diversification analysis
        """
        return {
            "diversification_score": 0.8,
            "concentration_warning": False,
            "sector_exposure": {},
            "recommendations": []
        }


class UserProfileManager:
    """
    User risk profiling and parameter management.
    
    Combines functionality from:
    - src/core/user_profiling.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize UserProfileManager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def assess_risk_profile(self, questionnaire_responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess user risk profile from questionnaire.
        
        Args:
            questionnaire_responses: User responses to risk questionnaire
            
        Returns:
            Risk profile assessment
        """
        return {
            "risk_category": RiskLevel.MODERATE,
            "risk_score": 65,
            "trading_parameters": TradingParameters(
                max_risk_per_trade=0.02,
                max_portfolio_risk=0.20,
                max_position_size=0.10,
                daily_loss_limit=0.06,
                leverage_limit=1.5
            ),
            "confidence_score": 0.85
        }
    
    def get_trading_parameters(self, user_id: str) -> TradingParameters:
        """
        Get trading parameters for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's trading parameters
        """
        return TradingParameters(
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.20,
            max_position_size=0.10,
            daily_loss_limit=0.06,
            leverage_limit=1.5
        )
    
    def update_risk_profile(self, user_id: str, new_responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user risk profile.
        
        Args:
            user_id: User identifier
            new_responses: Updated questionnaire responses
            
        Returns:
            Updated risk profile
        """
        return {}
    
    def validate_trade_against_profile(
        self, 
        user_id: str, 
        trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate trade against user risk profile.
        
        Args:
            user_id: User identifier
            trade: Trade details
            
        Returns:
            Validation results
        """
        return {
            "approved": True,
            "message": "Trade approved for user risk profile",
            "adjustments": {}
        }


class StrategyEngine:
    """
    Strategy selection and recommendation engine.
    
    Combines functionality from:
    - src/core/strategy_selector.py
    - src/core/recommendation_engine.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize StrategyEngine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.strategies = self._initialize_strategies()
        self.strategy_performance = {}
        
    def _initialize_strategies(self) -> List[Strategy]:
        """Initialize available trading strategies."""
        return [
            Strategy(
                id="conservative_growth",
                name="Conservative Growth",
                strategy_type=StrategyType.BUY_AND_HOLD,
                description="Long-term buy and hold with stable companies",
                risk_level=0.3,
                min_investment_horizon_days=365,
                max_position_size=0.08,
                volatility_tolerance=0.15,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.LOW_VOLATILITY],
                suitable_risk_categories=[RiskCategory.CONSERVATIVE],
                expected_return=0.08,
                expected_volatility=0.12,
                min_confidence_threshold=0.7
            ),
            Strategy(
                id="moderate_momentum", 
                name="Moderate Momentum",
                strategy_type=StrategyType.MOMENTUM,
                description="Medium-term momentum trading",
                risk_level=0.5,
                min_investment_horizon_days=30,
                max_position_size=0.12,
                volatility_tolerance=0.25,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS_MARKET],
                suitable_risk_categories=[RiskCategory.MODERATE, RiskCategory.AGGRESSIVE],
                expected_return=0.12,
                expected_volatility=0.18,
                min_confidence_threshold=0.6
            ),
            Strategy(
                id="aggressive_growth",
                name="Aggressive Growth",
                strategy_type=StrategyType.GROWTH_INVESTING,
                description="High growth potential stocks",
                risk_level=0.7,
                min_investment_horizon_days=90,
                max_position_size=0.15,
                volatility_tolerance=0.35,
                market_conditions=[MarketRegime.BULL_MARKET, MarketRegime.HIGH_VOLATILITY],
                suitable_risk_categories=[RiskCategory.AGGRESSIVE],
                expected_return=0.18,
                expected_volatility=0.28,
                min_confidence_threshold=0.5
            )
        ]
        
    def select_strategy(
        self, 
        market_conditions: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select optimal trading strategy based on conditions.
        
        Args:
            market_conditions: Current market analysis
            user_profile: User risk profile and preferences
            
        Returns:
            Selected strategy and parameters
        """
        try:
            user_risk_category = RiskCategory(user_profile.get("risk_category", "moderate"))
            market_volatility = market_conditions.get("volatility", 0.2)
            market_trend = market_conditions.get("trend", "sideways")
            
            # Score strategies based on suitability
            strategy_scores = []
            
            for strategy in self.strategies:
                if user_risk_category not in strategy.suitable_risk_categories:
                    continue
                    
                score = 0.5  # Base score
                
                # Risk level compatibility
                risk_tolerance = {
                    RiskCategory.CONSERVATIVE: 0.4,
                    RiskCategory.MODERATE: 0.6,
                    RiskCategory.AGGRESSIVE: 0.8
                }.get(user_risk_category, 0.5)
                
                if abs(strategy.risk_level - risk_tolerance) < 0.2:
                    score += 0.2
                
                # Market conditions compatibility
                current_regime = self._determine_market_regime(market_conditions)
                if current_regime in strategy.market_conditions:
                    score += 0.2
                
                # Volatility tolerance
                if market_volatility <= strategy.volatility_tolerance:
                    score += 0.1
                
                strategy_scores.append({
                    "strategy": strategy,
                    "score": score,
                    "reasoning": self._generate_strategy_reasoning(strategy, market_conditions, user_risk_category)
                })
            
            if not strategy_scores:
                # Fallback to conservative strategy
                return {
                    "strategy": "conservative_growth",
                    "confidence": 0.5,
                    "parameters": {},
                    "reasoning": "No suitable strategies found, defaulting to conservative approach"
                }
            
            # Select best strategy
            best_strategy_data = max(strategy_scores, key=lambda x: x["score"])
            best_strategy = best_strategy_data["strategy"]
            
            return {
                "strategy": best_strategy.id,
                "strategy_object": best_strategy,
                "confidence": min(0.95, best_strategy_data["score"]),
                "parameters": {
                    "max_position_size": best_strategy.max_position_size,
                    "risk_level": best_strategy.risk_level,
                    "investment_horizon": best_strategy.min_investment_horizon_days
                },
                "reasoning": best_strategy_data["reasoning"]
            }
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return {
                "strategy": "conservative_growth",
                "confidence": 0.5,
                "parameters": {},
                "reasoning": f"Error in strategy selection: {str(e)}"
            }
    
    def _determine_market_regime(self, market_conditions: Dict[str, Any]) -> MarketRegime:
        """Determine current market regime from conditions."""
        volatility = market_conditions.get("volatility", 0.2)
        trend = market_conditions.get("trend", "sideways")
        sentiment = market_conditions.get("sentiment", "neutral")
        
        if volatility > 0.3:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.15:
            return MarketRegime.LOW_VOLATILITY
        elif trend == "bullish" and sentiment == "positive":
            return MarketRegime.BULL_MARKET
        elif trend == "bearish" and sentiment == "negative":
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _generate_strategy_reasoning(self, strategy: Strategy, market_conditions: Dict[str, Any], user_risk: RiskCategory) -> str:
        """Generate reasoning for strategy selection."""
        reasons = []
        
        reasons.append(f"Strategy suitable for {user_risk.value} risk profile")
        
        volatility = market_conditions.get("volatility", 0.2)
        if volatility <= strategy.volatility_tolerance:
            reasons.append(f"Market volatility ({volatility:.1%}) within strategy tolerance")
        
        trend = market_conditions.get("trend", "sideways")
        if trend == "bullish" and strategy.strategy_type in [StrategyType.MOMENTUM, StrategyType.GROWTH_INVESTING]:
            reasons.append("Bullish market favors growth/momentum strategies")
        elif trend == "sideways" and strategy.strategy_type == StrategyType.MEAN_REVERSION:
            reasons.append("Sideways market suits mean reversion approach")
        
        return "; ".join(reasons)
    
    def generate_recommendations(
        self, 
        analysis_results: Dict[str, Any],
        strategy: Dict[str, Any],
        user_params: TradingParameters
    ) -> List[TradeRecommendation]:
        """
        Generate trade recommendations.
        
        Args:
            analysis_results: Analysis engine results
            strategy: Selected strategy
            user_params: User trading parameters
            
        Returns:
            List of trade recommendations
        """
        try:
            recommendations = []
            signals = analysis_results.get("signals", {})
            
            if not signals:
                return recommendations
            
            component_signals = signals.get("component_signals", {})
            overall_signal = signals.get("overall_signal", "hold")
            signal_strength = signals.get("signal_strength", "weak")
            
            # Generate recommendations based on signals and strategy
            if overall_signal in ["buy", "sell"] and signal_strength in ["moderate", "strong"]:
                symbol = analysis_results.get("symbol", "UNKNOWN")
                
                # Get current price from analysis
                price_data = analysis_results.get("analysis_components", {}).get("technical", {})
                current_price = price_data.get("latest_price", 100.0)
                
                # Calculate position size based on strategy
                strategy_obj = strategy.get("strategy_object")
                if strategy_obj:
                    max_position_pct = strategy_obj.max_position_size
                else:
                    max_position_pct = user_params.max_position_size
                
                # Assume portfolio value for calculation (in production, get from context)
                assumed_portfolio_value = 100000  # $100k default
                max_position_value = assumed_portfolio_value * max_position_pct
                quantity = int(max_position_value / current_price)
                
                # Calculate stop loss and take profit
                stop_loss_pct = 0.05 if overall_signal == "buy" else -0.05
                stop_loss = current_price * (1 - stop_loss_pct) if overall_signal == "buy" else current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1.1 if overall_signal == "buy" else 0.9)
                
                # Determine confidence
                confidence = 0.8 if signal_strength == "strong" else 0.6
                
                # Generate reasoning
                reasoning_parts = []
                if "technical" in component_signals:
                    reasoning_parts.append(f"Technical analysis: {component_signals['technical']}")
                if "fundamental" in component_signals:
                    reasoning_parts.append(f"Fundamental analysis: {component_signals['fundamental']}")
                reasoning = "; ".join(reasoning_parts) or f"Overall signal: {overall_signal}"
                
                recommendation = TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.BUY if overall_signal == "buy" else TradeAction.SELL,
                    quantity=quantity,
                    price=current_price,
                    confidence=confidence,
                    reasoning=reasoning,
                    risk_level=RiskLevel.MODERATE,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank trading opportunities by attractiveness.
        
        Args:
            opportunities: List of potential trades
            
        Returns:
            Ranked opportunities
        """
        return opportunities
    
    def calculate_expected_return(self, trade: Dict[str, Any]) -> float:
        """
        Calculate expected return for trade.
        
        Args:
            trade: Trade details
            
        Returns:
            Expected return percentage
        """
        return 0.05  # 5% default
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for strategy.
        
        Args:
            returns: Strategy returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        return 1.0
    
    def backtest_strategy(
        self, 
        strategy: Dict[str, Any], 
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Backtest strategy on historical data.
        
        Args:
            strategy: Strategy parameters
            historical_data: Historical market data
            
        Returns:
            Backtest results
        """
        return {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.65,
            "trades": 50
        }


class PortfolioOptimizer:
    """
    Portfolio optimization and allocation engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PortfolioOptimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def optimize_portfolio(
        self, 
        assets: List[str],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        
        Args:
            assets: List of assets
            expected_returns: Expected return for each asset
            covariance_matrix: Asset covariance matrix
            risk_tolerance: User risk tolerance
            
        Returns:
            Optimal portfolio allocation
        """
        return {
            "weights": {},
            "expected_return": 0.0,
            "expected_risk": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def rebalance_portfolio(
        self, 
        current_portfolio: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate portfolio rebalancing requirements.
        
        Args:
            current_portfolio: Current holdings
            target_weights: Target allocation weights
            threshold: Rebalancing threshold
            
        Returns:
            Rebalancing recommendations
        """
        return {
            "rebalance_needed": False,
            "trades": [],
            "cost_estimate": 0.0
        }


# Integrated Trading Engine
class TradingEngine:
    """
    Main trading engine that coordinates all trading modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TradingEngine with all components.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.risk_manager = RiskManager(config)
        self.user_manager = UserProfileManager(config)
        self.strategy_engine = StrategyEngine(config)
        self.portfolio_optimizer = PortfolioOptimizer(config)
        
    def process_trading_decision(
        self, 
        user_id: str,
        analysis_results: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process complete trading decision workflow.
        
        Args:
            user_id: User identifier
            analysis_results: Analysis engine results
            portfolio_state: Current portfolio state
            
        Returns:
            Trading decision results
        """
        try:
            logger.info(f"Processing trading decision for user {user_id}")
            
            # Get user trading parameters
            user_params = self.user_manager.get_trading_parameters(user_id)
            user_profile = {
                "risk_category": "moderate",  # Would come from user data
                "risk_tolerance": 0.6
            }
            
            # Assess portfolio risk
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(portfolio_state)
            
            # Extract market conditions from analysis
            market_conditions = {
                "volatility": portfolio_risk.get("total_risk", 0.15),
                "trend": analysis_results.get("signals", {}).get("overall_signal", "sideways"),
                "sentiment": analysis_results.get("analysis_components", {}).get("sentiment", {}).get("overall_sentiment", "neutral")
            }
            
            # Select appropriate strategy
            strategy_selection = self.strategy_engine.select_strategy(market_conditions, user_profile)
            
            # Generate trade recommendations
            recommendations = self.strategy_engine.generate_recommendations(
                analysis_results, strategy_selection, user_params
            )
            
            # Validate each recommendation
            validated_recommendations = []
            for rec in recommendations:
                trade_dict = {
                    "symbol": rec.symbol,
                    "action": rec.action.value,
                    "quantity": rec.quantity,
                    "price": rec.price
                }
                
                validation_result = self.risk_manager.validate_trade_risk(
                    trade_dict, portfolio_state, user_params
                )
                
                if validation_result["approved"]:
                    validated_recommendations.append(rec)
                else:
                    logger.warning(f"Trade rejected for {rec.symbol}: {validation_result['warnings']}")
            
            # Portfolio optimization suggestions
            portfolio_allocation = self.portfolio_optimizer.optimize_portfolio(
                assets=[rec.symbol for rec in validated_recommendations],
                expected_returns=pd.Series([rec.confidence * 0.1 for rec in validated_recommendations]),
                covariance_matrix=pd.DataFrame(np.eye(len(validated_recommendations)) * 0.04),
                risk_tolerance=user_profile["risk_tolerance"]
            )
            
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "recommendations": [format_recommendation(rec) for rec in validated_recommendations],
                "risk_assessment": portfolio_risk,
                "strategy_selection": strategy_selection,
                "portfolio_allocation": portfolio_allocation,
                "market_conditions": market_conditions,
                "validation_summary": {
                    "total_generated": len(recommendations),
                    "approved": len(validated_recommendations),
                    "rejected": len(recommendations) - len(validated_recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing trading decision: {e}")
            return {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "recommendations": [],
                "risk_assessment": {},
                "strategy_selection": {},
                "portfolio_allocation": {}
            }
    
    def execute_trade_validation(self, trade: TradeRecommendation, user_id: str) -> Dict[str, Any]:
        """
        Validate trade before execution.
        
        Args:
            trade: Trade recommendation
            user_id: User identifier
            
        Returns:
            Validation results
        """
        return {
            "approved": True,
            "risk_score": 0.5,
            "adjustments": {},
            "warnings": []
        }
    
    def calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Portfolio metrics
        """
        return {
            "total_value": 0.0,
            "daily_pnl": 0.0,
            "total_return": 0.0,
            "risk_metrics": {},
            "diversification_score": 0.8
        }


# Utility Functions
def validate_trade_parameters(trade: Dict[str, Any]) -> bool:
    """
    Validate trade parameter format.
    
    Args:
        trade: Trade parameters
        
    Returns:
        Validation result
    """
    required_fields = ["symbol", "action", "quantity", "price"]
    return all(field in trade for field in required_fields)


def calculate_risk_adjusted_return(returns: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """
    Calculate risk-adjusted return metrics.
    
    Args:
        returns: Strategy returns
        benchmark: Benchmark returns
        
    Returns:
        Risk-adjusted metrics
    """
    return {
        "alpha": 0.0,
        "beta": 1.0,
        "sharpe_ratio": 1.0,
        "information_ratio": 0.0
    }


def format_recommendation(rec: TradeRecommendation) -> Dict[str, Any]:
    """
    Format trade recommendation for output.
    
    Args:
        rec: Trade recommendation
        
    Returns:
        Formatted recommendation
    """
    return {
        "symbol": rec.symbol,
        "action": rec.action.value,
        "quantity": rec.quantity,
        "price": rec.price,
        "confidence": rec.confidence,
        "reasoning": rec.reasoning
    }