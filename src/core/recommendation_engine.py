"""Recommendation engines for AI Trading Advisor."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from .user_profiling import RiskProfile, RiskCategory
from .strategy_selector import (
    RiskBasedStrategySelector, StrategyRecommendation, MarketCondition, StrategyType
)

logger = logging.getLogger(__name__)


class TradingSignalEngine:
    """Generates trading signals based on analysis."""

    def generate_signal(self, symbol: str) -> str:
        """Generate buy/sell/hold signal for a symbol."""
        logger.info(f"Generating trading signal for {symbol}")
        # Placeholder implementation
        return "HOLD"  # Conservative default

    def calculate_signal_confidence(self, symbol: str, signal: str) -> float:
        """Calculate confidence score for a signal."""
        # Placeholder: return moderate confidence
        return 0.65

    def calculate_entry_exit_points(self, symbol: str, signal: str) -> Dict:
        """Calculate optimal entry and exit points."""
        return {
            "entry_price": 100.0,
            "exit_price": 110.0,
            "stop_loss": 95.0,
            "target_price": 115.0,
            "risk_reward_ratio": 2.0,
        }


class PortfolioOptimizationEngine:
    """Generates portfolio-level recommendations."""

    def generate_allocation_recommendations(
        self, signals: Dict, position_sizes: Dict
    ) -> Dict:
        """Generate asset allocation recommendations."""
        logger.info("Generating portfolio allocation recommendations")

        allocations = {}
        total_allocation = 0

        for symbol in signals.keys():
            if symbol in position_sizes:
                allocation = 0.05  # 5% base allocation
                allocations[symbol] = allocation
                total_allocation += allocation

        return {
            "allocations": allocations,
            "total_allocation": total_allocation,
            "cash_allocation": 1.0 - total_allocation,
        }

    def calculate_portfolio_risk(self, allocations: Dict) -> Dict:
        """Calculate portfolio risk metrics."""
        return {
            "overall_score": 0.4,  # Moderate risk
            "concentration_risk": 0.3,
            "sector_diversification": 0.8,
        }

    def generate_rebalancing_recommendations(self) -> List[Dict]:
        """Generate rebalancing recommendations."""
        return [
            {
                "action": "REBALANCE",
                "symbol": "SPY",
                "current_weight": 0.6,
                "target_weight": 0.5,
                "adjustment": -0.1,
            }
        ]

    def calculate_expected_returns(self, allocations: Dict) -> Dict:
        """Calculate expected portfolio returns."""
        return {
            "expected_annual_return": 0.08,  # 8%
            "expected_volatility": 0.15,  # 15%
            "sharpe_ratio": 0.53,
        }

    def analyze_diversification(self, allocations: Dict) -> Dict:
        """Analyze portfolio diversification."""
        return {
            "diversification_score": 0.7,
            "sector_concentration": {
                "technology": 0.4,
                "healthcare": 0.2,
                "finance": 0.2,
                "other": 0.2,
            },
        }

    def calculate_portfolio_score(self, allocations: Dict) -> float:
        """Calculate overall portfolio score."""
        return 0.75  # Good score


class ReportEngine:
    """Generates comprehensive recommendation reports."""

    def generate_executive_summary(self, filtered_data: Dict) -> str:
        """Generate executive summary."""
        recs = filtered_data.get("final_recommendations", [])
        return (
            f"Generated {len(recs)} trading recommendations with moderate risk profile."
        )

    def create_signal_analysis(self, signals: Dict) -> Dict:
        """Create detailed signal analysis."""
        successful_signals = sum(
            1 for s in signals.values() if s.get("status") == "success"
        )

        return {
            "total_signals": len(signals),
            "successful_signals": successful_signals,
            "signal_quality": "GOOD"
            if successful_signals > len(signals) * 0.8
            else "FAIR",
        }

    def create_risk_summary(self, portfolio_recs: Dict, risk_filtered: Dict) -> Dict:
        """Create risk assessment summary."""
        risk_score = risk_filtered.get("risk_score", 0.5)

        return {
            "overall_risk": "MODERATE" if risk_score < 0.6 else "HIGH",
            "risk_score": risk_score,
            "risk_factors": ["Market volatility", "Position concentration"],
        }

    def create_actionable_recommendations(self, filtered_data: Dict) -> List[Dict]:
        """Create actionable recommendations."""
        return filtered_data.get("final_recommendations", [])

    def generate_performance_projections(self, portfolio_recs: Dict) -> Dict:
        """Generate performance projections."""
        if portfolio_recs.get("status") != "success":
            return {"error": "Portfolio analysis failed"}

        expected_returns = portfolio_recs.get("expected_returns", {})

        return {
            "expected_1m_return": expected_returns.get("expected_annual_return", 0.08)
            / 12,
            "expected_3m_return": expected_returns.get("expected_annual_return", 0.08)
            / 4,
            "expected_1y_return": expected_returns.get("expected_annual_return", 0.08),
            "confidence_interval": "68%",
        }

    def create_market_context(self) -> Dict:
        """Create market context information."""
        return {
            "market_regime": "NORMAL",
            "volatility_environment": "MODERATE",
            "key_themes": ["Earnings season", "Fed policy", "Economic data"],
        }

    def calculate_report_confidence(self, filtered_data: Dict) -> float:
        """Calculate overall report confidence."""
        recs = filtered_data.get("final_recommendations", [])
        if not recs:
            return 0.0

        avg_confidence = sum(r.get("confidence", 0) for r in recs) / len(recs)
        return avg_confidence


class NotificationEngine:
    """Handles recommendation notifications."""

    def send_recommendation_notifications(self, report: Dict) -> Dict:
        """Send notifications for high-priority recommendations."""
        logger.info("Processing recommendation notifications")

        actionable_recs = report.get("actionable_recommendations", [])
        high_priority = [r for r in actionable_recs if r.get("confidence", 0) > 0.8]

        # Placeholder: In real implementation, send emails/alerts
        logger.info(f"Would send {len(high_priority)} high-priority notifications")

        return {
            "count": len(high_priority),
            "high_priority_symbols": [r.get("symbol") for r in high_priority],
        }


class RiskAwareRecommendationEngine:
    """Integrated recommendation engine with risk-based strategy selection."""
    
    def __init__(self):
        self.strategy_selector = RiskBasedStrategySelector()
        self.signal_engine = TradingSignalEngine()
        self.portfolio_optimizer = PortfolioOptimizationEngine()
        self.report_engine = ReportEngine()
        self.notification_engine = NotificationEngine()
    
    def generate_comprehensive_recommendations(
        self,
        user_id: str,
        risk_profile: RiskProfile,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        include_strategy_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive trading recommendations with risk-based strategy selection."""
        
        logger.info(f"Generating comprehensive recommendations for user {user_id}")
        
        # Extract portfolio information
        portfolio_value = portfolio_data.get("total_value", 0.0)
        current_positions = portfolio_data.get("positions", {})
        available_cash = portfolio_data.get("available_cash", 0.0)
        
        # Generate trading signals for current/potential positions
        symbols = list(current_positions.keys()) if current_positions else ["SPY", "QQQ", "IWM"]
        trading_signals = {}
        
        for symbol in symbols:
            signal = self.signal_engine.generate_signal(symbol)
            confidence = self.signal_engine.calculate_signal_confidence(symbol, signal)
            entry_exit = self.signal_engine.calculate_entry_exit_points(symbol, signal)
            
            trading_signals[symbol] = {
                "signal": signal,
                "confidence": confidence,
                "entry_exit_points": entry_exit,
                "symbol": symbol,
                "status": "success"
            }
        
        # Generate strategy recommendations if requested
        strategy_recommendations = []
        if include_strategy_recommendations:
            try:
                strategy_recs = self.strategy_selector.select_strategies_for_user(
                    risk_profile=risk_profile,
                    market_data=market_data,
                    portfolio_value=portfolio_value,
                    max_strategies=5
                )
                
                for rec in strategy_recs:
                    strategy_recommendations.append({
                        "strategy_id": rec.strategy.id,
                        "strategy_name": rec.strategy.name,
                        "strategy_type": rec.strategy.strategy_type.value,
                        "allocation_percentage": rec.allocation_percentage,
                        "confidence_score": rec.confidence_score,
                        "reasoning": rec.reasoning,
                        "expected_return": rec.expected_return,
                        "expected_risk": rec.expected_risk,
                        "market_context": rec.market_context,
                        "trade_parameters": rec.trade_parameters,
                        "risk_level": rec.strategy.risk_level,
                        "min_investment_horizon_days": rec.strategy.min_investment_horizon_days
                    })
                    
            except Exception as e:
                logger.error(f"Error generating strategy recommendations: {e}")
                strategy_recommendations = []
        
        # Generate portfolio optimization recommendations
        portfolio_recs = self.portfolio_optimizer.generate_allocation_recommendations(
            trading_signals, current_positions
        )
        
        # Calculate portfolio risk
        portfolio_risk = self.portfolio_optimizer.calculate_portfolio_risk(
            portfolio_recs.get("allocations", {})
        )
        
        # Filter recommendations based on risk profile
        risk_filtered_recs = self._filter_recommendations_by_risk(
            trading_signals, strategy_recommendations, risk_profile, portfolio_value
        )
        
        # Generate final actionable recommendations
        final_recommendations = self._generate_risk_appropriate_recommendations(
            risk_filtered_recs, strategy_recommendations, risk_profile, market_data
        )
        
        # Create comprehensive report
        comprehensive_report = {
            "user_id": user_id,
            "risk_category": risk_profile.risk_category.value,
            "generated_at": datetime.now().isoformat(),
            
            # Core recommendations
            "trading_signals": trading_signals,
            "strategy_recommendations": strategy_recommendations,
            "final_recommendations": final_recommendations,
            
            # Portfolio analysis
            "portfolio_analysis": {
                "current_value": portfolio_value,
                "available_cash": available_cash,
                "positions": current_positions,
                "allocation_recs": portfolio_recs,
                "risk_metrics": portfolio_risk,
            },
            
            # Risk analysis
            "risk_analysis": {
                "risk_category": risk_profile.risk_category.value,
                "risk_score": risk_profile.risk_score,
                "confidence_score": risk_profile.confidence_score,
                "trading_parameters": self.strategy_selector.user_profiling_engine.get_trading_parameters(
                    risk_profile.risk_category, risk_profile.confidence_score
                ),
                "allocation_limits": self.strategy_selector.get_strategy_allocation_limits(risk_profile)
            },
            
            # Market context
            "market_context": self.report_engine.create_market_context(),
            
            # Performance projections
            "performance_projections": self.report_engine.generate_performance_projections(portfolio_recs),
            
            # Executive summary
            "executive_summary": self._generate_risk_aware_summary(
                final_recommendations, strategy_recommendations, risk_profile
            ),
            
            # Report confidence
            "overall_confidence": self._calculate_overall_confidence(
                trading_signals, strategy_recommendations, risk_profile
            )
        }
        
        # Send notifications for high-priority recommendations
        notification_result = self.notification_engine.send_recommendation_notifications(
            comprehensive_report
        )
        comprehensive_report["notifications_sent"] = notification_result
        
        logger.info(f"Generated comprehensive report with {len(final_recommendations)} final recommendations")
        return comprehensive_report
    
    def _filter_recommendations_by_risk(
        self,
        trading_signals: Dict[str, Any],
        strategy_recommendations: List[Dict[str, Any]],
        risk_profile: RiskProfile,
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Filter all recommendations based on user's risk profile."""
        
        # Get trading parameters for user's risk category
        trading_params = self.strategy_selector.user_profiling_engine.get_trading_parameters(
            risk_profile.risk_category, risk_profile.confidence_score
        )
        
        # Filter trading signals by confidence and risk
        filtered_signals = {}
        min_confidence = 0.7 if risk_profile.risk_category == RiskCategory.CONSERVATIVE else (
            0.5 if risk_profile.risk_category == RiskCategory.MODERATE else 0.3
        )
        
        for symbol, signal_data in trading_signals.items():
            if signal_data.get("confidence", 0) >= min_confidence:
                # Check if position size would be appropriate
                entry_price = signal_data.get("entry_exit_points", {}).get("entry_price", 100.0)
                max_position_value = portfolio_value * trading_params["max_position_size"]
                
                if entry_price > 0:
                    max_shares = int(max_position_value / entry_price)
                    signal_data["max_recommended_shares"] = max_shares
                    signal_data["max_recommended_value"] = max_shares * entry_price
                    
                    filtered_signals[symbol] = signal_data
        
        # Filter strategy recommendations by risk appropriateness
        filtered_strategy_recs = []
        for rec in strategy_recommendations:
            strategy_risk_level = rec.get("risk_level", 0.5)
            
            # Check if strategy risk level is appropriate for user
            if risk_profile.risk_category == RiskCategory.CONSERVATIVE and strategy_risk_level > 0.4:
                continue
            elif risk_profile.risk_category == RiskCategory.MODERATE and strategy_risk_level > 0.7:
                continue
            # Aggressive users can use all strategies
            
            filtered_strategy_recs.append(rec)
        
        return {
            "filtered_signals": filtered_signals,
            "filtered_strategy_recommendations": filtered_strategy_recs,
            "risk_score": min(1.0, sum(
                rec.get("risk_level", 0) * rec.get("allocation_percentage", 0) 
                for rec in filtered_strategy_recs
            )),
            "total_filtered": len(filtered_signals) + len(filtered_strategy_recs)
        }
    
    def _generate_risk_appropriate_recommendations(
        self,
        risk_filtered_data: Dict[str, Any],
        strategy_recommendations: List[Dict[str, Any]],
        risk_profile: RiskProfile,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate final risk-appropriate actionable recommendations."""
        
        final_recommendations = []
        
        # Add signal-based recommendations
        for symbol, signal_data in risk_filtered_data.get("filtered_signals", {}).items():
            if signal_data.get("signal") != "HOLD":
                recommendation = {
                    "type": "SIGNAL",
                    "symbol": symbol,
                    "action": signal_data.get("signal"),
                    "confidence": signal_data.get("confidence", 0),
                    "reasoning": f"Technical signal with {signal_data.get('confidence', 0):.1%} confidence",
                    "max_position_value": signal_data.get("max_recommended_value", 0),
                    "entry_price": signal_data.get("entry_exit_points", {}).get("entry_price"),
                    "target_price": signal_data.get("entry_exit_points", {}).get("target_price"),
                    "stop_loss": signal_data.get("entry_exit_points", {}).get("stop_loss"),
                    "risk_category_appropriate": True,
                    "priority": "HIGH" if signal_data.get("confidence", 0) > 0.8 else "MEDIUM"
                }
                final_recommendations.append(recommendation)
        
        # Add strategy-based recommendations
        for strategy_rec in risk_filtered_data.get("filtered_strategy_recommendations", []):
            recommendation = {
                "type": "STRATEGY",
                "strategy_name": strategy_rec.get("strategy_name"),
                "strategy_type": strategy_rec.get("strategy_type"),
                "action": "IMPLEMENT",
                "confidence": strategy_rec.get("confidence_score", 0),
                "reasoning": strategy_rec.get("reasoning"),
                "allocation_percentage": strategy_rec.get("allocation_percentage", 0),
                "expected_return": strategy_rec.get("expected_return", 0),
                "expected_risk": strategy_rec.get("expected_risk", 0),
                "investment_horizon_days": strategy_rec.get("min_investment_horizon_days", 0),
                "risk_category_appropriate": True,
                "priority": "HIGH" if strategy_rec.get("confidence_score", 0) > 0.7 else "MEDIUM"
            }
            final_recommendations.append(recommendation)
        
        # Sort by priority and confidence
        final_recommendations.sort(key=lambda x: (
            x["priority"] == "HIGH", x.get("confidence", 0)
        ), reverse=True)
        
        # Limit recommendations based on risk profile
        max_recs = {
            RiskCategory.CONSERVATIVE: 3,
            RiskCategory.MODERATE: 5,
            RiskCategory.AGGRESSIVE: 8
        }.get(risk_profile.risk_category, 5)
        
        return final_recommendations[:max_recs]
    
    def _generate_risk_aware_summary(
        self,
        final_recommendations: List[Dict[str, Any]],
        strategy_recommendations: List[Dict[str, Any]],
        risk_profile: RiskProfile
    ) -> str:
        """Generate executive summary with risk context."""
        
        signal_recs = [r for r in final_recommendations if r.get("type") == "SIGNAL"]
        strategy_recs = [r for r in final_recommendations if r.get("type") == "STRATEGY"]
        
        summary_parts = [
            f"Generated {len(final_recommendations)} risk-appropriate recommendations for {risk_profile.risk_category.value} investor.",
        ]
        
        if signal_recs:
            summary_parts.append(f"Identified {len(signal_recs)} trading opportunities with suitable risk levels.")
        
        if strategy_recs:
            avg_allocation = np.mean([r.get("allocation_percentage", 0) for r in strategy_recs]) if strategy_recs else 0
            summary_parts.append(f"Recommended {len(strategy_recs)} strategies with average {avg_allocation:.1%} allocation.")
        
        if risk_profile.risk_category == RiskCategory.CONSERVATIVE:
            summary_parts.append("Focus on capital preservation with steady, low-risk returns.")
        elif risk_profile.risk_category == RiskCategory.MODERATE:
            summary_parts.append("Balanced approach targeting moderate growth with measured risk.")
        else:
            summary_parts.append("Growth-focused approach with higher return potential and risk tolerance.")
        
        return " ".join(summary_parts)
    
    def _calculate_overall_confidence(
        self,
        trading_signals: Dict[str, Any],
        strategy_recommendations: List[Dict[str, Any]],
        risk_profile: RiskProfile
    ) -> float:
        """Calculate overall confidence in recommendations."""
        
        confidences = []
        
        # Add signal confidences
        for signal_data in trading_signals.values():
            confidences.append(signal_data.get("confidence", 0.5))
        
        # Add strategy confidences
        for strategy_rec in strategy_recommendations:
            confidences.append(strategy_rec.get("confidence_score", 0.5))
        
        # Add risk profile confidence if available
        if risk_profile.confidence_score:
            confidences.append(risk_profile.confidence_score)
        
        # Calculate weighted average (give more weight to risk profile confidence)
        if confidences:
            if risk_profile.confidence_score:
                # Weight risk profile confidence more heavily
                risk_weight = 0.4
                other_weight = 0.6 / (len(confidences) - 1) if len(confidences) > 1 else 0.6
                
                weighted_confidence = (risk_profile.confidence_score * risk_weight +
                                     sum(c for c in confidences if c != risk_profile.confidence_score) * other_weight)
            else:
                weighted_confidence = np.mean(confidences)
        else:
            weighted_confidence = 0.5
        
        return max(0.0, min(1.0, weighted_confidence))
    
    def update_strategy_performance(self, performance_updates: List[Dict[str, Any]]):
        """Update strategy performance metrics."""
        for update in performance_updates:
            strategy_id = update.get("strategy_id")
            if strategy_id:
                self.strategy_selector.update_strategy_performance(strategy_id, update)
                logger.info(f"Updated performance for strategy {strategy_id}")
    
    def get_user_strategy_allocations(self, risk_profile: RiskProfile) -> Dict[str, float]:
        """Get recommended strategy allocation limits for user."""
        return self.strategy_selector.get_strategy_allocation_limits(risk_profile)
