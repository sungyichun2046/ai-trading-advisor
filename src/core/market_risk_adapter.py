"""Market Risk Adapter - Dynamic risk adjustment based on market conditions.

This module implements VIX-based risk scaling, market regime detection, and
dynamic position size adjustments based on volatility monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import statistics

from src.config import settings
from src.data.collectors import VolatilityMonitor
from src.core.risk_engine import RiskAnalysisEngine


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"


class VolatilityLevel(Enum):
    """Volatility level classifications."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class MarketRiskAdapter:
    """Adapts risk parameters based on real-time market conditions.
    
    This class integrates with volatility monitoring to dynamically adjust
    position sizes and risk parameters based on market stress levels.
    """

    def __init__(self):
        """Initialize the market risk adapter."""
        self.volatility_monitor = VolatilityMonitor()
        self.risk_engine = RiskAnalysisEngine()
        
        # VIX-based risk scaling thresholds
        self.vix_thresholds = {
            "low": 15.0,
            "moderate": 18.0,
            "elevated": 23.0,
            "high": 30.0,
            "extreme": 40.0
        }
        
        # Risk multipliers for different volatility levels
        self.risk_multipliers = {
            VolatilityLevel.LOW: 1.2,        # Increase risk in calm markets
            VolatilityLevel.MODERATE: 1.0,   # Normal risk
            VolatilityLevel.ELEVATED: 0.8,   # Reduce risk slightly
            VolatilityLevel.HIGH: 0.6,       # Significant risk reduction
            VolatilityLevel.EXTREME: 0.3     # Maximum risk reduction
        }
        
        # Market regime risk adjustments
        self.regime_multipliers = {
            MarketRegime.BULL_LOW_VOL: 1.3,
            MarketRegime.BULL_HIGH_VOL: 0.9,
            MarketRegime.BEAR_LOW_VOL: 0.7,
            MarketRegime.BEAR_HIGH_VOL: 0.4,
            MarketRegime.SIDEWAYS_LOW_VOL: 1.1,
            MarketRegime.SIDEWAYS_HIGH_VOL: 0.8,
            MarketRegime.CRISIS: 0.2
        }
        
        # Circuit breaker thresholds
        self.circuit_breaker_vix = 50.0
        self.circuit_breaker_spike_ratio = 5.0

    def get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions from volatility monitor.
        
        Returns:
            Dict containing current market conditions and volatility data.
        """
        logger.info("Fetching current market conditions")
        
        try:
            volatility_data = self.volatility_monitor.check_market_volatility()
            
            if volatility_data["status"] != "success":
                logger.warning("Failed to get volatility data, using conservative defaults")
                return self._get_default_market_conditions()
                
            return volatility_data
            
        except Exception as e:
            logger.error(f"Error fetching market conditions: {e}")
            return self._get_default_market_conditions()

    def _get_default_market_conditions(self) -> Dict[str, Any]:
        """Get default conservative market conditions when data is unavailable."""
        return {
            "status": "success",
            "vix_current": 25.0,
            "volatility_level": "ELEVATED",
            "alerts": [],
            "triggers": [],
            "data_source": "default"
        }

    def classify_volatility_level(self, vix_value: float) -> VolatilityLevel:
        """Classify volatility level based on VIX value.
        
        Args:
            vix_value: Current VIX value
            
        Returns:
            VolatilityLevel enum
        """
        if vix_value >= self.vix_thresholds["extreme"]:
            return VolatilityLevel.EXTREME
        elif vix_value >= self.vix_thresholds["high"]:
            return VolatilityLevel.HIGH
        elif vix_value >= self.vix_thresholds["elevated"]:
            return VolatilityLevel.ELEVATED
        elif vix_value >= self.vix_thresholds["moderate"]:
            return VolatilityLevel.MODERATE
        else:
            return VolatilityLevel.LOW

    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime based on volatility and market conditions.
        
        Args:
            market_data: Market condition data from volatility monitor
            
        Returns:
            MarketRegime enum
        """
        vix_current = market_data.get("vix_current", 25.0)
        alerts = market_data.get("alerts", [])
        volatility_level = market_data.get("volatility_level", "MODERATE")
        
        # Detect crisis conditions
        if (vix_current >= self.circuit_breaker_vix or 
            "EXTREME_VIX" in alerts or 
            volatility_level == "EXTREME"):
            return MarketRegime.CRISIS
        
        # Determine market direction from alerts
        price_movement_alerts = [alert for alert in alerts if "PRICE_MOVEMENT" in alert]
        volume_spike_alerts = [alert for alert in alerts if "VOLUME_SPIKE" in alert]
        
        # Simple regime classification based on available data
        is_high_vol = volatility_level in ["HIGH", "EXTREME"]
        has_stress_signals = len(price_movement_alerts) > 0 or len(volume_spike_alerts) > 1
        
        if has_stress_signals:
            if is_high_vol:
                return MarketRegime.BEAR_HIGH_VOL
            else:
                return MarketRegime.BEAR_LOW_VOL
        else:
            if is_high_vol:
                return MarketRegime.SIDEWAYS_HIGH_VOL
            else:
                return MarketRegime.BULL_LOW_VOL

    def calculate_risk_multiplier(self, market_data: Dict[str, Any]) -> float:
        """Calculate combined risk multiplier based on market conditions.
        
        Args:
            market_data: Market condition data
            
        Returns:
            Risk multiplier (1.0 = normal risk, <1.0 = reduced risk, >1.0 = increased risk)
        """
        vix_current = market_data.get("vix_current", 25.0)
        volatility_level = self.classify_volatility_level(vix_current)
        market_regime = self.detect_market_regime(market_data)
        
        # Get base multipliers
        vix_multiplier = self.risk_multipliers[volatility_level]
        regime_multiplier = self.regime_multipliers[market_regime]
        
        # Apply circuit breaker if conditions are extreme
        if self._should_apply_circuit_breaker(market_data):
            logger.warning("Circuit breaker activated - maximum risk reduction")
            return 0.1
        
        # Combine multipliers (weighted average)
        combined_multiplier = (vix_multiplier * 0.6) + (regime_multiplier * 0.4)
        
        # Ensure multiplier stays within reasonable bounds
        combined_multiplier = max(0.1, min(2.0, combined_multiplier))
        
        logger.info(f"Risk multiplier calculated: {combined_multiplier:.3f} "
                   f"(VIX: {vix_multiplier:.2f}, Regime: {regime_multiplier:.2f})")
        
        return combined_multiplier

    def _should_apply_circuit_breaker(self, market_data: Dict[str, Any]) -> bool:
        """Check if circuit breaker should be applied.
        
        Args:
            market_data: Market condition data
            
        Returns:
            True if circuit breaker should be applied
        """
        vix_current = market_data.get("vix_current", 25.0)
        alerts = market_data.get("alerts", [])
        
        # VIX circuit breaker
        if vix_current >= self.circuit_breaker_vix:
            return True
            
        # Multiple extreme alerts
        extreme_alerts = [alert for alert in alerts if "EXTREME" in alert]
        if len(extreme_alerts) >= 2:
            return True
            
        # Emergency triggers
        triggers = market_data.get("triggers", [])
        if "emergency_analysis" in triggers:
            return True
            
        return False

    def adjust_position_size(self, base_position_size: float, 
                           account_balance: float, 
                           symbol: str = "SPY") -> Dict[str, Any]:
        """Adjust position size based on current market conditions.
        
        Args:
            base_position_size: Base position size calculation
            account_balance: Current account balance
            symbol: Trading symbol (for symbol-specific adjustments)
            
        Returns:
            Dict containing adjusted position size and risk parameters
        """
        logger.info(f"Adjusting position size for {symbol}, base size: ${base_position_size:,.2f}")
        
        # Validate inputs - negative position sizes should be set to 0
        if base_position_size < 0:
            logger.warning(f"Negative position size {base_position_size} adjusted to 0")
            base_position_size = 0.0
        
        if account_balance <= 0:
            logger.warning(f"Invalid account balance {account_balance}, returning zero position")
            return {
                "original_position_size": base_position_size,
                "adjusted_position_size": 0.0,
                "risk_multiplier": 0.0,
                "position_risk_percentage": 0.0,
                "volatility_level": "MODERATE",
                "market_regime": "SIDEWAYS_LOW_VOL",
                "vix_current": 25.0,
                "circuit_breaker_active": False,
                "market_alerts": [],
                "adjustment_reason": "Invalid account balance",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get current market conditions
        market_data = self.get_current_market_conditions()
        
        # Calculate risk multiplier
        risk_multiplier = self.calculate_risk_multiplier(market_data)
        
        # Apply multiplier to position size
        adjusted_position_size = base_position_size * risk_multiplier
        
        # Ensure position doesn't exceed account limits
        max_position_pct = settings.max_position_size * risk_multiplier
        max_position_value = account_balance * max_position_pct
        adjusted_position_size = min(adjusted_position_size, max_position_value)
        
        # Calculate risk metrics
        position_risk_pct = adjusted_position_size / account_balance
        volatility_level = self.classify_volatility_level(market_data.get("vix_current", 25.0))
        market_regime = self.detect_market_regime(market_data)
        
        result = {
            "original_position_size": base_position_size,
            "adjusted_position_size": adjusted_position_size,
            "risk_multiplier": risk_multiplier,
            "position_risk_percentage": position_risk_pct,
            "volatility_level": volatility_level.value,
            "market_regime": market_regime.value,
            "vix_current": market_data.get("vix_current", 25.0),
            "circuit_breaker_active": self._should_apply_circuit_breaker(market_data),
            "market_alerts": market_data.get("alerts", []),
            "adjustment_reason": self._get_adjustment_reason(market_data, risk_multiplier),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Position adjusted: ${base_position_size:,.2f} -> ${adjusted_position_size:,.2f} "
                   f"(multiplier: {risk_multiplier:.3f}, VIX: {result['vix_current']:.1f})")
        
        return result

    def _get_adjustment_reason(self, market_data: Dict[str, Any], risk_multiplier: float) -> str:
        """Generate human-readable reason for position adjustment.
        
        Args:
            market_data: Market condition data
            risk_multiplier: Applied risk multiplier
            
        Returns:
            String describing the adjustment reason
        """
        vix_current = market_data.get("vix_current", 25.0)
        volatility_level = market_data.get("volatility_level", "MODERATE")
        alerts = market_data.get("alerts", [])
        
        if self._should_apply_circuit_breaker(market_data):
            return "Circuit breaker activated due to extreme market conditions"
        elif risk_multiplier < 0.8:
            return f"Risk reduced due to {volatility_level.lower()} volatility (VIX: {vix_current:.1f})"
        elif risk_multiplier > 1.1:
            return f"Risk increased due to favorable market conditions (VIX: {vix_current:.1f})"
        else:
            return f"Normal risk adjustment for current market conditions (VIX: {vix_current:.1f})"

    def get_risk_parameters_for_regime(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Get risk parameters optimized for specific market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dict of risk parameters
        """
        base_params = {
            "max_risk_per_trade": settings.max_risk_per_trade,
            "max_portfolio_risk": settings.max_portfolio_risk,
            "max_position_size": settings.max_position_size,
            "daily_loss_limit": settings.daily_loss_limit
        }
        
        regime_multiplier = self.regime_multipliers[market_regime]
        
        # Adjust parameters based on regime
        adjusted_params = {}
        for param, value in base_params.items():
            if param == "daily_loss_limit":
                # More conservative loss limits in volatile markets
                adjusted_params[param] = value * max(0.5, regime_multiplier)
            else:
                adjusted_params[param] = value * regime_multiplier
                
        return adjusted_params

    def validate_trade_against_market_conditions(self, trade_size: float, 
                                               account_balance: float,
                                               symbol: str = "SPY") -> Dict[str, Any]:
        """Validate a proposed trade against current market conditions.
        
        Args:
            trade_size: Proposed trade size
            account_balance: Current account balance
            symbol: Trading symbol
            
        Returns:
            Dict with validation result and recommendations
        """
        logger.info(f"Validating trade: {symbol} ${trade_size:,.2f}")
        
        # Get market-adjusted position size
        adjustment_result = self.adjust_position_size(trade_size, account_balance, symbol)
        
        recommended_size = adjustment_result["adjusted_position_size"]
        risk_multiplier = adjustment_result["risk_multiplier"]
        
        # Determine if trade should be approved
        size_ratio = trade_size / recommended_size if recommended_size > 0 else float('inf')
        
        if size_ratio <= 1.1:  # Within 10% of recommended size
            approval_status = "approved"
            message = "Trade size is appropriate for current market conditions"
        elif size_ratio <= 1.5:  # 50% larger than recommended
            approval_status = "approved_with_warning"
            message = f"Trade size exceeds recommendation by {((size_ratio - 1) * 100):.0f}%"
        else:
            approval_status = "rejected"
            message = f"Trade size too large for current market conditions (VIX: {adjustment_result['vix_current']:.1f})"
        
        return {
            "approval_status": approval_status,
            "message": message,
            "proposed_size": trade_size,
            "recommended_size": recommended_size,
            "risk_multiplier": risk_multiplier,
            "market_conditions": adjustment_result,
            "timestamp": datetime.now().isoformat()
        }

    def get_market_stress_score(self) -> Dict[str, Any]:
        """Calculate comprehensive market stress score.
        
        Returns:
            Dict containing stress score and contributing factors
        """
        market_data = self.get_current_market_conditions()
        
        vix_current = market_data.get("vix_current", 25.0)
        alerts = market_data.get("alerts", [])
        volatility_level = market_data.get("volatility_level", "MODERATE")
        
        # Calculate stress components
        vix_stress = min(100, (vix_current / 50.0) * 100)  # VIX contribution (0-100)
        alert_stress = min(100, len(alerts) * 20)  # Alert contribution (0-100)
        
        # Volatility level stress mapping
        vol_stress_map = {
            "LOW": 10,
            "MODERATE": 25,
            "ELEVATED": 50,
            "HIGH": 75,
            "EXTREME": 100
        }
        vol_stress = vol_stress_map.get(volatility_level, 25)
        
        # Combined stress score (weighted average)
        stress_score = (vix_stress * 0.5) + (alert_stress * 0.3) + (vol_stress * 0.2)
        stress_score = min(100, max(0, stress_score))
        
        # Classify stress level
        if stress_score >= 80:
            stress_level = "EXTREME"
        elif stress_score >= 60:
            stress_level = "HIGH"
        elif stress_score >= 40:
            stress_level = "MODERATE"
        elif stress_score >= 20:
            stress_level = "LOW"
        else:
            stress_level = "MINIMAL"
        
        return {
            "stress_score": round(stress_score, 1),
            "stress_level": stress_level,
            "vix_contribution": round(vix_stress, 1),
            "alert_contribution": round(alert_stress, 1),
            "volatility_contribution": round(vol_stress, 1),
            "current_vix": vix_current,
            "active_alerts": len(alerts),
            "volatility_level": volatility_level,
            "timestamp": datetime.now().isoformat()
        }