"""Trading Engine Module - Enhanced with paper trading and shared utilities."""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass

# Shared utilities import
try:
    from ..utils.shared import validate_data_quality, log_performance
except ImportError:
    def validate_data_quality(data, data_type="trading", min_threshold=0.8):
        return {"quality_score": 0.8, "issues": [], "data_type": data_type}
    def log_performance(operation, start_time, end_time, status="success", metrics=None):
        return {"operation": operation, "duration": 0.1, "status": status}

# Configuration
try:
    from ..config import settings
except ImportError:
    class MockSettings:
        DATABASE_URL = "postgresql://airflow:airflow@test-postgres:5432/airflow"
        REDIS_URL = "redis://localhost:6379"
        API_TIMEOUT = 30
        MAX_RETRIES = 3
    settings = MockSettings()

logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    message: str
    priority: AlertPriority
    timestamp: datetime
    category: str
    details: Dict[str, Any] = None

class TradingEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def momentum_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            start_time = datetime.now()
            end_time = start_time
            log_performance("Momentum Strategy", start_time, end_time, "success", performance)
            
            return {
                "signal": signal,
                "confidence": confidence, 
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def mean_reversion_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            start_time = datetime.now()
            end_time = start_time  
            log_performance("Mean Reversion Strategy", start_time, end_time, "success", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning, 
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def breakout_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            start_time = datetime.now()
            end_time = start_time
            log_performance("Breakout Strategy", start_time, end_time, "success", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def value_strategy(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            start_time = datetime.now()
            end_time = start_time
            log_performance("Value Strategy", start_time, end_time, "success", performance)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "performance": performance
            }
            
        except Exception as e:
            return {"signal": "hold", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}

    def generate_explanation(self, strategy_results: Dict[str, Any], consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            explanation = {
                "decision_summary": self._create_decision_summary(consensus_result),
                "strategy_breakdown": self._analyze_strategy_breakdown(strategy_results),
                "risk_analysis": self._assess_risk_factors(strategy_results, consensus_result),
                "confidence_attribution": self._attribute_confidence(strategy_results),
                "recommendations": self._generate_recommendations(consensus_result)
            }
            return {"explanation": explanation, "timestamp": datetime.now().isoformat(), "status": "success"}
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"explanation": {"error": str(e)}, "status": "error"}

    def calculate_strategy_scores(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        scores = {}
        for strategy_name, result in strategy_results.items():
            confidence = result.get("confidence", 0.0)
            signal = result.get("signal", "hold")
            base_score = confidence * 100
            score = base_score if signal in ["buy", "sell"] else base_score * 0.5
            scores[strategy_name] = round(score, 2)
        return scores

    def attribute_performance(self, strategy_results: Dict[str, Any], portfolio_performance: Dict[str, Any]) -> Dict[str, Any]:
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
                    "weight": round(weight, 3), "attributed_return": round(attributed_return, 4),
                    "confidence": confidence, "signal": result.get("signal", "hold")
                }
            return {"attribution": attribution, "total_attribution": portfolio_return, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in performance attribution: {e}")
            return {"attribution": {}, "error": str(e)}

    def send_alerts(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alerts = []
            
            # Process risk, signal, and performance alerts
            for violation in alert_data.get("risk_violations", []):
                alerts.append(Alert(f"Risk Violation: {violation}", AlertPriority.HIGH, datetime.now(), "risk"))
            
            for signal in alert_data.get("strong_signals", []):
                priority = AlertPriority.HIGH if signal.get("confidence", 0) > 0.8 else AlertPriority.MEDIUM
                msg = f"Strong {signal.get('signal', 'unknown')} signal for {signal.get('symbol', 'unknown')}"
                alerts.append(Alert(msg, priority, datetime.now(), "signal", signal))
            
            for issue in alert_data.get("performance_issues", []):
                alerts.append(Alert(f"Performance Alert: {issue}", AlertPriority.MEDIUM, datetime.now(), "performance"))
            
            # Generate notifications
            notifications = []
            for alert in alerts:
                notifications.append({
                    "channel": "email" if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL] else "dashboard",
                    "recipient": "trader@ai-trading-advisor.com", "subject": f"Trading Alert: {alert.category.upper()}",
                    "message": alert.message, "priority": alert.priority.value, "timestamp": alert.timestamp.isoformat()
                })
            
            return {"alerts_generated": len(alerts), "notifications_sent": len(notifications), 
                   "notifications": notifications, "status": "success", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
            return {"alerts_generated": 0, "notifications_sent": 0, "error": str(e), "status": "error"}

    def _create_decision_summary(self, consensus_result: Dict[str, Any]) -> str:
        signal, confidence = consensus_result.get("overall_signal", "hold"), consensus_result.get("confidence", 0.0)
        confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        return f"Recommendation: {signal.upper()} with {confidence_text} confidence ({confidence:.1%})"

    def _analyze_strategy_breakdown(self, strategy_results: Dict[str, Any]) -> Dict[str, str]:
        return {strategy_name: f"{result.get('signal', 'hold').upper()} ({result.get('confidence', 0.0):.1%}) - {result.get('reasoning', 'No reasoning provided')}" 
                for strategy_name, result in strategy_results.items()}

    def _assess_risk_factors(self, strategy_results: Dict[str, Any], consensus_result: Dict[str, Any]) -> List[str]:
        risk_factors = []
        signal_distribution = consensus_result.get("signal_distribution", {})
        max_signals = max(signal_distribution.values()) if signal_distribution else 0
        total_signals = sum(signal_distribution.values()) if signal_distribution else 0
        
        if total_signals > 0 and max_signals / total_signals < 0.6:
            risk_factors.append("Low strategy consensus - mixed signals detected")
        
        low_confidence_strategies = [name for name, result in strategy_results.items() if result.get("confidence", 0) < 0.3]
        if len(low_confidence_strategies) > len(strategy_results) / 2:
            risk_factors.append(f"Multiple low-confidence strategies: {', '.join(low_confidence_strategies)}")
        
        if consensus_result.get("confidence", 0.0) < 0.4:
            risk_factors.append("Low overall confidence in recommendation")
        return risk_factors

    def _attribute_confidence(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        total_confidence = sum(r.get("confidence", 0) for r in strategy_results.values())
        if total_confidence == 0: return {}
        return {strategy_name: round((result.get("confidence", 0.0) / total_confidence) * 100, 1) 
                for strategy_name, result in strategy_results.items()}

    def _generate_recommendations(self, consensus_result: Dict[str, Any]) -> List[str]:
        recommendations = []
        signal, confidence = consensus_result.get("overall_signal", "hold"), consensus_result.get("confidence", 0.0)
        
        if signal == "buy" and confidence > 0.7: recommendations.append("Consider entering a long position with appropriate position sizing")
        elif signal == "sell" and confidence > 0.7: recommendations.append("Consider exiting long positions or entering short positions")
        elif signal == "hold": recommendations.append("Maintain current positions and monitor for signal changes")
        
        if confidence < 0.5: recommendations.append("Wait for clearer signals before taking action")
        recommendations.append("Always apply proper risk management and position sizing rules")
        return recommendations

    def execute_paper_trade(self, signal_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            signal_quality = validate_data_quality(signal_data, "trading_signals", 0.7)
            if signal_quality["quality_score"] < 0.7:
                return {"status": "rejected", "reason": f"Low signal quality: {signal_quality['issues']}", "trade": None}
            
            symbol, signal, confidence, current_price = signal_data.get("symbol", "UNKNOWN"), signal_data.get("signal", "hold"), signal_data.get("confidence", 0.0), signal_data.get("price", 100.0)
            portfolio_value = portfolio_state.get("total_value", 100000.0)
            shares = int(portfolio_value * 0.02 / current_price)  # 2% risk rule
            
            if signal in ["buy", "sell"] and confidence > 0.5 and shares > 0:
                trade = {"symbol": symbol, "action": signal, "shares": shares, "price": current_price, 
                        "value": shares * current_price, "confidence": confidence, "timestamp": datetime.now().isoformat()}
                
                current_position = portfolio_state.get("positions", {}).get(symbol, 0)
                new_position = current_position + shares if signal == "buy" else max(0, current_position - shares)
                updated_portfolio = portfolio_state.copy()
                updated_portfolio.setdefault("positions", {})[symbol] = new_position
                
                log_performance("Paper Trade Execution", start_time, datetime.now(), "success", 
                               {"symbol": symbol, "action": signal, "shares": shares, "value": trade["value"]})
                
                return {"status": "executed", "trade": trade, "portfolio_state": updated_portfolio, "quality_check": signal_quality}
            else:
                return {"status": "no_action", "reason": f"Signal {signal} with confidence {confidence:.2f} below threshold", "trade": None}
                
        except Exception as e:
            log_performance("Paper Trade Execution", start_time, datetime.now(), "error", {"error": str(e)})
            return {"status": "error", "error": str(e), "trade": None}

    def calculate_portfolio_metrics(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            data_quality = validate_data_quality(market_data, "market_data", 0.8)
            positions = portfolio_state.get("positions", {})
            total_value, position_values = 0.0, {}
            
            for symbol, shares in positions.items():
                price = market_data.get(symbol, {}).get("price", 0)
                value = shares * price
                position_values[symbol] = {"shares": shares, "price": price, "value": value}
                total_value += value
            
            cash = portfolio_state.get("cash", 0.0)
            total_portfolio_value = total_value + cash
            position_weights = {symbol: val["value"] / total_portfolio_value for symbol, val in position_values.items() if total_portfolio_value > 0}
            concentration_risk = max(position_weights.values()) if position_weights else 0.0
            diversification_score = 1.0 - concentration_risk
            
            metrics = {"total_value": total_portfolio_value, "position_value": total_value, "cash": cash, "positions": position_values,
                      "diversification_score": round(diversification_score, 3), "concentration_risk": round(concentration_risk, 3),
                      "data_quality": data_quality, "timestamp": datetime.now().isoformat()}
            
            log_performance("Portfolio Metrics", start_time, datetime.now(), "success", {"total_value": total_portfolio_value, "positions": len(positions)})
            return {"status": "success", "metrics": metrics}
            
        except Exception as e:
            log_performance("Portfolio Metrics", start_time, datetime.now(), "error", {"error": str(e)})
            return {"status": "error", "error": str(e), "metrics": {}}

def calculate_returns(prices: pd.Series, returns_type: str = "simple") -> pd.Series:
    if prices.empty or len(prices) < 2: return pd.Series(dtype=float)
    return np.log(prices / prices.shift(1)).fillna(0) if returns_type == "log" else prices.pct_change().fillna(0)
