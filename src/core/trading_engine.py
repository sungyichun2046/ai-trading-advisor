"""
Extended Trading Engine Module
Enhanced trading functionality with explanation, scoring, and attribution functions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass

# Configuration
try:
    from ..config import settings
except ImportError:
    # Fallback settings for Docker environment
    class MockSettings:
        DATABASE_URL = "postgresql://airflow:airflow@test-postgres:5432/airflow"
        REDIS_URL = "redis://localhost:6379"
        USE_REAL_DATA = False
        API_TIMEOUT = 30
        MAX_RETRIES = 3
    settings = MockSettings()

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    message: str
    priority: AlertPriority
    timestamp: datetime
    category: str
    details: Dict[str, Any] = None


class TradingEngine:
    """Extended trading engine with explanation and attribution capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading engine with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def momentum_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute momentum trading strategy."""
        try:
            params = params or {"lookback_period": 20, "threshold": 0.02}
            price_data = data.get("technical", {}).get("price_data", pd.Series(dtype=float))
            
            if price_data.empty or len(price_data) < params["lookback_period"]:
                return {
                    "signal": "hold", 
                    "confidence": 0.0, 
                    "reasoning": "No price data or insufficient history",
                    "performance": {"momentum_value": 0, "signal_strength": 0}
                }
            
            returns = calculate_returns(price_data)
            lookback = params["lookback_period"] 
            momentum = returns.tail(lookback).mean()
            threshold = params["threshold"]
            
            # Generate signal
            if momentum > threshold:
                signal = "buy"
                confidence = min(0.9, 0.5 + abs(momentum) / threshold)
            elif momentum < -threshold:
                signal = "sell" 
                confidence = min(0.9, 0.5 + abs(momentum) / threshold)
            else:
                signal = "hold"
                confidence = 0.3
                
            reasoning = f"Momentum: {momentum:.4f}, threshold: {threshold}"
            
            performance = {
                "momentum_value": momentum,
                "signal_strength": confidence,
                "returns_mean": returns.mean(),
                "returns_std": returns.std()
            }
            
            log_performance("Momentum Strategy", performance)
            
            return {
                "signal": signal,
                "confidence": confidence, 
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def mean_reversion_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute mean reversion trading strategy."""
        try:
            params = params or {"lookback_period": 20, "z_threshold": 2.0}
            price_data = data.get("technical", {}).get("price_data", pd.Series(dtype=float))
            
            if price_data.empty or len(price_data) < params["lookback_period"]:
                return {
                    "signal": "hold",
                    "confidence": 0.0, 
                    "reasoning": "Insufficient data for mean reversion analysis",
                    "performance": {"z_score": 0, "mean_price": 0}
                }
            
            mean_price = price_data.mean()
            std_price = price_data.std()
            current_price = price_data.iloc[-1]
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            threshold = params["z_threshold"]
            
            if z_score < -threshold:
                signal = "buy"  # Price below mean
                confidence = min(0.9, 0.3 + abs(z_score) / threshold * 0.6)
            elif z_score > threshold:
                signal = "sell"  # Price above mean
                confidence = min(0.9, 0.3 + abs(z_score) / threshold * 0.6)
            else:
                signal = "hold"
                confidence = 0.2
                
            reasoning = f"Z-score: {z_score:.2f}, threshold: ±{threshold}"
            
            performance = {
                "z_score": z_score,
                "mean_price": mean_price,
                "current_price": current_price,
                "volatility": std_price
            }
            
            log_performance("Mean Reversion Strategy", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning, 
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def breakout_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute breakout trading strategy."""
        try:
            params = params or {"lookback_period": 20, "volume_threshold": 1.5}
            price_data = data.get("technical", {}).get("price_data", pd.Series(dtype=float))
            volume_data = data.get("technical", {}).get("volume_data", pd.Series(dtype=float))
            
            if price_data.empty or len(price_data) < params["lookback_period"]:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "Insufficient price data for breakout analysis", 
                    "performance": {"high_level": 0, "low_level": 0}
                }
            
            lookback = params["lookback_period"]
            high_level = price_data.tail(lookback).max()
            low_level = price_data.tail(lookback).min()
            current_price = price_data.iloc[-1]
            
            # Volume confirmation
            volume_confirmed = False
            if not volume_data.empty and len(volume_data) >= lookback:
                avg_volume = volume_data.tail(lookback).mean()
                recent_volume = volume_data.iloc[-1]
                volume_confirmed = recent_volume > avg_volume * params["volume_threshold"]
            
            if current_price > high_level:
                signal = "buy"  # Upward breakout
                confidence = 0.8 if volume_confirmed else 0.5
            elif current_price < low_level:
                signal = "sell"  # Downward breakout
                confidence = 0.8 if volume_confirmed else 0.5
            else:
                signal = "hold"
                confidence = 0.2
                
            reasoning = f"Price: {current_price:.2f}, Range: [{low_level:.2f}, {high_level:.2f}], Volume OK: {volume_confirmed}"
            
            performance = {
                "high_level": high_level,
                "low_level": low_level,
                "current_price": current_price,
                "price_range": high_level - low_level,
                "volume_confirmed": volume_confirmed
            }
            
            log_performance("Breakout Strategy", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def value_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute value investing strategy."""
        try:
            params = params or {"pe_threshold": 15, "pb_threshold": 2, "debt_threshold": 0.6}
            fundamentals = data.get("fundamental", {}).get("financial_metrics", {})
            
            if not fundamentals:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "No fundamental data available",
                    "performance": {"value_score": 0}
                }
            
            # Value scoring
            pe_ratio = fundamentals.get("pe_ratio", 25)
            pb_ratio = fundamentals.get("pb_ratio", 3)
            debt_ratio = fundamentals.get("debt_to_equity", 0.5)
            roe = fundamentals.get("roe", 0.1)
            current_ratio = fundamentals.get("current_ratio", 1.5)
            
            value_score = 0
            
            # PE ratio score
            if pe_ratio < 10:
                value_score += 2
            elif pe_ratio < params["pe_threshold"]:
                value_score += 1
            elif pe_ratio > 25:
                value_score -= 1
                
            # PB ratio score  
            if pb_ratio < 1:
                value_score += 2
            elif pb_ratio < params["pb_threshold"]:
                value_score += 1
            elif pb_ratio > 3:
                value_score -= 1
                
            # Debt score
            if debt_ratio < 0.3:
                value_score += 1
            elif debt_ratio > params["debt_threshold"]:
                value_score -= 1
                
            # ROE score
            if roe > 0.15:
                value_score += 1
            elif roe < 0.05:
                value_score -= 1
                
            # Liquidity score
            if current_ratio > 2:
                value_score += 1
            elif current_ratio < 1:
                value_score -= 1
            
            # Generate signal
            if value_score >= 4:
                signal = "buy"
                confidence = 0.9
            elif value_score >= 2:
                signal = "buy" 
                confidence = 0.7
            elif value_score <= -2:
                signal = "sell"
                confidence = 0.6
            else:
                signal = "hold"
                confidence = 0.3
                
            reasoning = f"Value score: {value_score}/6. "
            if pe_ratio < 15: reasoning += f"Low P/E: {pe_ratio}; "
            if pb_ratio < 2: reasoning += f"Low P/B: {pb_ratio}; "
            if debt_ratio < 0.6: reasoning += f"Low debt: {debt_ratio}; "
            if roe > 0.1: reasoning += f"Good ROE: {roe*100:.1f}%; "
            if current_ratio > 1.5: reasoning += f"Good liquidity: {current_ratio}"
            
            performance = {
                "value_score": value_score,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio, 
                "debt_ratio": debt_ratio,
                "roe": roe,
                "current_ratio": current_ratio
            }
            
            log_performance("Value Strategy", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}

    def generate_explanation(self, strategy_results: Dict[str, Any], consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation for trading decisions."""
        try:
            explanation = {
                "decision_summary": self._create_decision_summary(consensus_result),
                "strategy_breakdown": self._analyze_strategy_breakdown(strategy_results),
                "risk_analysis": self._assess_risk_factors(strategy_results, consensus_result),
                "confidence_attribution": self._attribute_confidence(strategy_results),
                "recommendations": self._generate_recommendations(consensus_result)
            }
            
            return {
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"explanation": {"error": str(e)}, "status": "error"}

    def calculate_strategy_scores(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual strategy performance scores."""
        scores = {}
        
        for strategy_name, result in strategy_results.items():
            confidence = result.get("confidence", 0.0)
            signal = result.get("signal", "hold")
            
            # Base score from confidence
            base_score = confidence * 100
            
            # Adjust based on signal strength
            if signal in ["buy", "sell"]:
                score = base_score
            else:  # hold
                score = base_score * 0.5  # Hold signals get reduced score
                
            scores[strategy_name] = round(score, 2)
            
        return scores

    def attribute_performance(self, strategy_results: Dict[str, Any], portfolio_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Attribute portfolio performance to individual strategies."""
        try:
            attribution = {}
            total_confidence = sum(r.get("confidence", 0) for r in strategy_results.values())
            
            if total_confidence == 0:
                return {"attribution": {}, "total_attribution": 0.0}
            
            portfolio_return = portfolio_performance.get("total_return", 0.0)
            
            for strategy_name, result in strategy_results.items():
                confidence = result.get("confidence", 0.0)
                weight = confidence / total_confidence
                attributed_return = portfolio_return * weight
                
                attribution[strategy_name] = {
                    "weight": round(weight, 3),
                    "attributed_return": round(attributed_return, 4),
                    "confidence": confidence,
                    "signal": result.get("signal", "hold")
                }
            
            return {
                "attribution": attribution,
                "total_attribution": portfolio_return,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {e}")
            return {"attribution": {}, "error": str(e)}

    def send_alerts(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send trading alerts and notifications."""
        try:
            alerts = []
            
            # Risk alerts
            risk_violations = alert_data.get("risk_violations", [])
            for violation in risk_violations:
                alert = Alert(
                    message=f"Risk Violation: {violation}",
                    priority=AlertPriority.HIGH,
                    timestamp=datetime.now(),
                    category="risk",
                    details={"violation": violation}
                )
                alerts.append(alert)
            
            # Signal alerts
            strong_signals = alert_data.get("strong_signals", [])
            for signal in strong_signals:
                priority = AlertPriority.HIGH if signal.get("confidence", 0) > 0.8 else AlertPriority.MEDIUM
                alert = Alert(
                    message=f"Strong {signal.get('signal', 'unknown')} signal for {signal.get('symbol', 'unknown')}",
                    priority=priority,
                    timestamp=datetime.now(),
                    category="signal",
                    details=signal
                )
                alerts.append(alert)
            
            # Performance alerts
            performance_issues = alert_data.get("performance_issues", [])
            for issue in performance_issues:
                alert = Alert(
                    message=f"Performance Alert: {issue}",
                    priority=AlertPriority.MEDIUM,
                    timestamp=datetime.now(),
                    category="performance",
                    details={"issue": issue}
                )
                alerts.append(alert)
            
            # Send alerts (simulated)
            notifications_sent = []
            for alert in alerts:
                notification = {
                    "channel": "email" if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL] else "dashboard",
                    "recipient": "trader@ai-trading-advisor.com",
                    "subject": f"Trading Alert: {alert.category.upper()}",
                    "message": alert.message,
                    "priority": alert.priority.value,
                    "timestamp": alert.timestamp.isoformat()
                }
                notifications_sent.append(notification)
            
            return {
                "alerts_generated": len(alerts),
                "notifications_sent": len(notifications_sent),
                "notifications": notifications_sent,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
            return {
                "alerts_generated": 0,
                "notifications_sent": 0,
                "error": str(e),
                "status": "error"
            }

    def _create_decision_summary(self, consensus_result: Dict[str, Any]) -> str:
        """Create human-readable decision summary."""
        signal = consensus_result.get("overall_signal", "hold")
        confidence = consensus_result.get("confidence", 0.0)
        
        confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        
        return f"Recommendation: {signal.upper()} with {confidence_text} confidence ({confidence:.1%})"

    def _analyze_strategy_breakdown(self, strategy_results: Dict[str, Any]) -> Dict[str, str]:
        """Analyze individual strategy contributions."""
        breakdown = {}
        
        for strategy_name, result in strategy_results.items():
            signal = result.get("signal", "hold")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            breakdown[strategy_name] = f"{signal.upper()} ({confidence:.1%}) - {reasoning}"
            
        return breakdown

    def _assess_risk_factors(self, strategy_results: Dict[str, Any], consensus_result: Dict[str, Any]) -> List[str]:
        """Assess risk factors in the trading decision."""
        risk_factors = []
        
        # Check for low consensus
        signal_distribution = consensus_result.get("signal_distribution", {})
        max_signals = max(signal_distribution.values()) if signal_distribution else 0
        total_signals = sum(signal_distribution.values()) if signal_distribution else 0
        
        if total_signals > 0 and max_signals / total_signals < 0.6:
            risk_factors.append("Low strategy consensus - mixed signals detected")
        
        # Check for low confidence strategies
        low_confidence_strategies = [
            name for name, result in strategy_results.items() 
            if result.get("confidence", 0) < 0.3
        ]
        
        if len(low_confidence_strategies) > len(strategy_results) / 2:
            risk_factors.append(f"Multiple low-confidence strategies: {', '.join(low_confidence_strategies)}")
        
        # Check overall confidence
        overall_confidence = consensus_result.get("confidence", 0.0)
        if overall_confidence < 0.4:
            risk_factors.append("Low overall confidence in recommendation")
            
        return risk_factors

    def _attribute_confidence(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """Attribute overall confidence to individual strategies."""
        total_confidence = sum(r.get("confidence", 0) for r in strategy_results.values())
        
        if total_confidence == 0:
            return {}
        
        attribution = {}
        for strategy_name, result in strategy_results.items():
            confidence = result.get("confidence", 0.0)
            contribution = (confidence / total_confidence) * 100
            attribution[strategy_name] = round(contribution, 1)
            
        return attribution

    def _generate_recommendations(self, consensus_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        signal = consensus_result.get("overall_signal", "hold")
        confidence = consensus_result.get("confidence", 0.0)
        
        if signal == "buy" and confidence > 0.7:
            recommendations.append("Consider entering a long position with appropriate position sizing")
        elif signal == "sell" and confidence > 0.7:
            recommendations.append("Consider exiting long positions or entering short positions")
        elif signal == "hold":
            recommendations.append("Maintain current positions and monitor for signal changes")
        
        if confidence < 0.5:
            recommendations.append("Wait for clearer signals before taking action")
            
        recommendations.append("Always apply proper risk management and position sizing rules")
        
        return recommendations


def calculate_returns(prices: pd.Series, returns_type: str = "simple") -> pd.Series:
    """Calculate returns from price series."""
    if prices.empty or len(prices) < 2:
        return pd.Series(dtype=float)
    
    if returns_type == "simple":
        returns = prices.pct_change().fillna(0)
    elif returns_type == "log":
        returns = np.log(prices / prices.shift(1)).fillna(0)
    else:
        returns = prices.pct_change().fillna(0)
    
    return returns


def log_performance(strategy_name: str, performance_data: Dict[str, Any]) -> None:
    """Log strategy performance metrics."""
    try:
        logger.info(f"Strategy Performance: {strategy_name}")
        for metric, value in performance_data.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    except Exception as e:
        logger.error(f"Error logging performance for {strategy_name}: {e}")