"""Risk management engines for AI Trading Advisor."""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from src.config import RiskConfig

logger = logging.getLogger(__name__)

# Import scipy only when needed to handle missing dependency gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available, some advanced risk metrics will be simplified")
    SCIPY_AVAILABLE = False


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    PORTFOLIO_HEAT = "portfolio_heat"


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
    max_heat_threshold: float = 0.3  # Maximum allowed heat


class RiskAnalysisEngine:
    """Performs comprehensive risk analysis with advanced metrics."""

    def __init__(self):
        """Initialize risk analysis engine."""
        self.lookback_periods = 252  # 1 year of daily data
        self.confidence_levels = [0.95, 0.99]

    def calculate_market_risk(self, symbol: str, returns: List[float] = None) -> Dict:
        """Calculate comprehensive market risk metrics."""
        logger.info(f"Calculating market risk metrics for {symbol}")
        
        if returns is None:
            # Default mock data - in production, fetch from database
            returns = np.random.normal(0.001, 0.02, self.lookback_periods)
        
        returns_array = np.array(returns)
        
        # Calculate beta (correlation with market * (asset_vol / market_vol))
        market_returns = np.random.normal(0.0008, 0.015, len(returns))  # Mock S&P 500
        
        covariance = np.cov(returns_array, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        correlation = np.corrcoef(returns_array, market_returns)[0, 1]
        
        # Systematic vs idiosyncratic risk
        r_squared = correlation ** 2
        systematic_risk = r_squared
        idiosyncratic_risk = 1 - r_squared
        
        return {
            "symbol": symbol,
            "beta": float(beta),
            "correlation_to_market": float(correlation),
            "systematic_risk": float(systematic_risk),
            "idiosyncratic_risk": float(idiosyncratic_risk),
            "r_squared": float(r_squared),
            "tracking_error": float(np.std(returns_array - market_returns) * np.sqrt(252))
        }

    def calculate_volatility_metrics(self, returns: List[float] = None) -> Dict:
        """Calculate comprehensive volatility metrics."""
        if returns is None:
            returns = np.random.normal(0.001, 0.02, self.lookback_periods)
        
        returns_array = np.array(returns)
        
        # Different volatility measures
        daily_vol = np.std(returns_array)
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility for trend analysis
        window = 30
        if len(returns_array) >= window * 2:
            recent_vol = np.std(returns_array[-window:]) * np.sqrt(252)
            past_vol = np.std(returns_array[-window*2:-window]) * np.sqrt(252)
            vol_trend = "INCREASING" if recent_vol > past_vol * 1.1 else \
                       "DECREASING" if recent_vol < past_vol * 0.9 else "STABLE"
        else:
            recent_vol = annualized_vol
            vol_trend = "STABLE"
        
        # Volatility percentile (current vol vs historical distribution)
        vol_percentile = (np.sum(annualized_vol > daily_vol * np.sqrt(252)) / len(returns_array)) * 100
        
        return {
            "daily_volatility": float(daily_vol),
            "annualized_volatility": float(annualized_vol),
            "recent_volatility_30d": float(recent_vol),
            "volatility_trend": vol_trend,
            "volatility_percentile": float(vol_percentile),
            "volatility_regime": "HIGH" if annualized_vol > 0.30 else 
                                "MEDIUM" if annualized_vol > 0.15 else "LOW"
        }

    def perform_correlation_analysis(
        self, 
        portfolio_returns: Dict[str, List[float]] = None
    ) -> Dict:
        """Perform comprehensive correlation analysis."""
        if portfolio_returns is None:
            # Mock data for demonstration
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
            portfolio_returns = {
                symbol: np.random.normal(0.001, 0.02, self.lookback_periods).tolist()
                for symbol in symbols
            }
        
        # Create correlation matrix
        returns_df = pd.DataFrame(portfolio_returns)
        correlation_matrix = returns_df.corr()
        
        # Calculate key metrics
        avg_correlation = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1
        )].mean()
        
        max_correlation = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1
        )].max()
        
        min_correlation = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1
        )].min()
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        for i, symbol1 in enumerate(correlation_matrix.index):
            for j, symbol2 in enumerate(correlation_matrix.columns):
                if i < j and abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'pair': f"{symbol1}-{symbol2}",
                        'correlation': float(correlation_matrix.iloc[i, j])
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": float(avg_correlation),
            "max_correlation": float(max_correlation),
            "min_correlation": float(min_correlation),
            "high_correlation_pairs": high_corr_pairs,
            "diversification_ratio": float(1 / (1 + avg_correlation)),
            "correlation_regime": "HIGH" if avg_correlation > 0.7 else 
                                "MEDIUM" if avg_correlation > 0.3 else "LOW"
        }

    def calculate_var_metrics(
        self, 
        portfolio_value: float,
        returns: List[float] = None,
        confidence_levels: List[float] = None
    ) -> Dict:
        """Calculate comprehensive Value at Risk metrics."""
        if returns is None:
            returns = np.random.normal(0.001, 0.02, self.lookback_periods)
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
            
        returns_array = np.array(returns)
        
        # Historical VaR
        var_results = {}
        for confidence in confidence_levels:
            var_percentile = (1 - confidence) * 100
            var_return = np.percentile(returns_array, var_percentile)
            var_dollar = abs(var_return * portfolio_value)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns_array[returns_array <= var_return]
            expected_shortfall_return = np.mean(tail_returns) if len(tail_returns) > 0 else var_return
            expected_shortfall_dollar = abs(expected_shortfall_return * portfolio_value)
            
            confidence_str = f"{int(confidence*100)}"
            var_results[f"var_1d_{confidence_str}"] = float(var_return)
            var_results[f"var_1d_{confidence_str}_dollar"] = float(var_dollar)
            var_results[f"expected_shortfall_{confidence_str}"] = float(expected_shortfall_return)
            var_results[f"expected_shortfall_{confidence_str}_dollar"] = float(expected_shortfall_dollar)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        for confidence in confidence_levels:
            z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-confidence)*100))
            parametric_var = mean_return - z_score * std_return
            confidence_str = f"{int(confidence*100)}"
            var_results[f"parametric_var_1d_{confidence_str}"] = float(parametric_var)
        
        return var_results

    def assess_tail_risk(self, returns: List[float] = None) -> Dict:
        """Assess tail risk characteristics."""
        if returns is None:
            returns = np.random.normal(0.001, 0.02, self.lookback_periods)
            
        returns_array = np.array(returns)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Tail ratio (95th percentile / 5th percentile)
        percentile_95 = np.percentile(returns_array, 95)
        percentile_5 = np.percentile(returns_array, 5)
        tail_ratio = percentile_95 / abs(percentile_5) if percentile_5 != 0 else 0
        
        # Skewness and kurtosis
        if SCIPY_AVAILABLE:
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array, fisher=True)  # Excess kurtosis
        else:
            # Simplified calculations without scipy
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            skewness = np.mean(((returns_array - mean_return) / std_return) ** 3)
            kurtosis = np.mean(((returns_array - mean_return) / std_return) ** 4) - 3
        
        # Extreme value metrics
        extreme_positive = np.sum(returns_array > np.percentile(returns_array, 95))
        extreme_negative = np.sum(returns_array < np.percentile(returns_array, 5))
        
        # Tail risk score (composite metric)
        tail_risk_components = [
            abs(max_drawdown) * 2,  # Weight drawdown heavily
            abs(skewness) * 0.1 if skewness < 0 else 0,  # Negative skew is bad
            max(0, kurtosis - 3) * 0.05,  # Excess kurtosis
            (1 / tail_ratio - 1) * 0.1 if tail_ratio > 0 else 0.1
        ]
        tail_risk_score = min(1.0, sum(tail_risk_components))
        
        return {
            "max_drawdown": float(max_drawdown),
            "tail_ratio": float(tail_ratio),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "extreme_positive_days": int(extreme_positive),
            "extreme_negative_days": int(extreme_negative),
            "tail_risk_score": float(tail_risk_score),
            "tail_risk_rating": "HIGH" if tail_risk_score > 0.7 else 
                               "MEDIUM" if tail_risk_score > 0.4 else "LOW"
        }

    def calculate_overall_risk_score(self) -> float:
        """Calculate comprehensive overall risk score."""
        # Get individual risk components
        market_risk = self.calculate_market_risk('PORTFOLIO')
        vol_metrics = self.calculate_volatility_metrics()
        tail_risk = self.assess_tail_risk()
        
        # Weight different risk factors
        components = {
            'market_risk': market_risk.get('systematic_risk', 0.4) * 0.3,
            'volatility': min(vol_metrics.get('annualized_volatility', 0.15) / 0.30, 1.0) * 0.4,
            'tail_risk': tail_risk.get('tail_risk_score', 0.3) * 0.3
        }
        
        overall_score = sum(components.values())
        
        return min(1.0, overall_score)


class RealTimePositionMonitor:
    """Real-time position size monitoring and adjustment engine."""
    
    def __init__(self):
        """Initialize real-time position monitor."""
        self.sizing_engine = AdvancedPositionSizingEngine()
        self.risk_engine = RiskAnalysisEngine()
        self.diversification_analyzer = DiversificationAnalyzer()
        self.last_update_time = None
        self.position_cache = {}
        self.market_data_cache = {}
        self.portfolio_metrics_cache = {}
        self.rebalance_threshold = 0.10  # 10% difference triggers rebalance
        self.alert_thresholds = {
            "heat_warning": 0.25,  # 25% portfolio heat warning
            "heat_critical": 0.30,  # 30% portfolio heat critical
            "concentration_warning": 0.08,  # 8% single position warning
            "concentration_critical": 0.10,  # 10% single position critical
            "correlation_warning": 0.7,  # 70% average correlation warning
            "volatility_spike": 1.5  # 50% increase in volatility
        }
        self.monitoring_active = False
        
    def update_market_data(self, market_data: Dict[str, Dict]) -> None:
        """Update cached market data for real-time calculations."""
        self.market_data_cache.update(market_data)
        logger.info(f"Updated market data for {len(market_data)} symbols")
        
    def update_portfolio_positions(self, positions: Dict[str, Dict]) -> None:
        """Update cached portfolio positions."""
        self.position_cache = positions.copy()
        self.last_update_time = datetime.now()
        logger.info(f"Updated portfolio positions: {len(positions)} positions")
        
    def calculate_real_time_adjustments(
        self, 
        portfolio_value: float,
        trigger_threshold: float = None
    ) -> Dict[str, any]:
        """Calculate real-time position size adjustments.
        
        Args:
            portfolio_value: Current portfolio value
            trigger_threshold: Threshold for triggering adjustments (default: 10%)
            
        Returns:
            Dict containing adjustment recommendations
        """
        if trigger_threshold is None:
            trigger_threshold = self.rebalance_threshold
            
        logger.info("Calculating real-time position adjustments")
        
        if not self.position_cache or not self.market_data_cache:
            logger.warning("Missing position or market data for real-time calculations")
            return {"status": "insufficient_data", "adjustments": {}}
            
        adjustments = {}
        portfolio_metrics = self._create_portfolio_metrics(portfolio_value)
        
        for symbol, position in self.position_cache.items():
            if symbol in self.market_data_cache:
                market_data = self.market_data_cache[symbol]
                
                # Create sizing parameters
                params = PositionSizingParams(
                    symbol=symbol,
                    current_price=market_data.get('price', position.get('avg_cost', 100)),
                    expected_return=market_data.get('expected_return', 0.08),
                    win_rate=market_data.get('win_rate', 0.55),
                    avg_win=market_data.get('avg_win', 0.06),
                    avg_loss=market_data.get('avg_loss', -0.03),
                    volatility=market_data.get('volatility', 0.20),
                    correlation_to_portfolio=market_data.get('correlation_to_market', 0.3),
                    stop_loss_pct=0.05,
                    confidence=0.7
                )
                
                # Calculate optimal position size
                optimal_result = self.sizing_engine.calculate_optimal_position_size(
                    params, portfolio_metrics, SizingMethod.CORRELATION_ADJUSTED
                )
                
                current_value = position.get('value', 0)
                optimal_value = optimal_result['optimal_size_usd']
                difference_pct = abs(optimal_value - current_value) / current_value if current_value > 0 else 1.0
                
                # Check if adjustment is needed
                if difference_pct > trigger_threshold:
                    urgency = self._calculate_adjustment_urgency(
                        difference_pct, market_data, portfolio_metrics
                    )
                    
                    adjustments[symbol] = {
                        'current_position': position,
                        'optimal_position': optimal_result,
                        'current_value': current_value,
                        'optimal_value': optimal_value,
                        'difference_usd': optimal_value - current_value,
                        'difference_pct': difference_pct,
                        'action': 'BUY' if optimal_value > current_value else 'SELL',
                        'urgency': urgency,
                        'trigger_reason': self._get_trigger_reason(market_data, difference_pct),
                        'risk_impact': self._assess_risk_impact(optimal_result, portfolio_metrics),
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Calculate portfolio-level impact
        total_adjustment_amount = sum(
            abs(adj['difference_usd']) for adj in adjustments.values()
        )
        
        high_urgency_count = sum(
            1 for adj in adjustments.values() if adj['urgency'] == 'HIGH'
        )
        
        return {
            "status": "success",
            "adjustments": adjustments,
            "portfolio_impact": {
                "total_adjustment_amount": total_adjustment_amount,
                "high_urgency_adjustments": high_urgency_count,
                "portfolio_heat_after_adjustment": self._calculate_projected_heat(
                    adjustments, portfolio_metrics
                ),
                "adjustment_needed": len(adjustments) > 0,
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None
            }
        }
        
    def _create_portfolio_metrics(self, portfolio_value: float) -> PortfolioMetrics:
        """Create portfolio metrics from cached data."""
        return PortfolioMetrics(
            total_value=portfolio_value,
            available_cash=portfolio_value * 0.1,  # Assume 10% cash
            current_positions=self.position_cache,
            portfolio_beta=1.0,  # Simplified
            portfolio_volatility=0.15,  # Simplified
            portfolio_correlation_matrix=np.eye(len(self.position_cache)),
            heat_level=0.2  # Simplified
        )
        
    def _calculate_adjustment_urgency(
        self, 
        difference_pct: float,
        market_data: Dict,
        portfolio_metrics: PortfolioMetrics
    ) -> str:
        """Calculate urgency of position adjustment."""
        urgency_factors = []
        
        # Size difference factor
        if difference_pct > 0.3:  # >30% difference
            urgency_factors.append('large_size_deviation')
        
        # Volatility factor
        if market_data.get('volatility', 0) > 0.25:  # >25% volatility
            urgency_factors.append('high_volatility')
            
        # Portfolio heat factor
        if portfolio_metrics.heat_level > 0.25:
            urgency_factors.append('elevated_heat')
            
        # Market conditions factor (simplified)
        if market_data.get('price_change_1d', 0) > 0.05:  # >5% daily move
            urgency_factors.append('significant_price_movement')
            
        if len(urgency_factors) >= 3:
            return 'HIGH'
        elif len(urgency_factors) >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _get_trigger_reason(self, market_data: Dict, difference_pct: float) -> str:
        """Get the primary reason for triggering the adjustment."""
        reasons = []
        
        if difference_pct > 0.2:
            reasons.append('significant_size_deviation')
            
        if market_data.get('volatility', 0) > 0.3:
            reasons.append('volatility_spike')
            
        if abs(market_data.get('price_change_1d', 0)) > 0.05:
            reasons.append('price_movement')
            
        return reasons[0] if reasons else 'threshold_exceeded'
        
    def _assess_risk_impact(
        self, 
        optimal_result: Dict, 
        portfolio_metrics: PortfolioMetrics
    ) -> Dict:
        """Assess the risk impact of the position adjustment."""
        current_risk = portfolio_metrics.heat_level
        position_risk = optimal_result['risk_percentage']
        
        return {
            'position_risk_pct': position_risk,
            'portfolio_weight': optimal_result['portfolio_weight'],
            'risk_contribution': position_risk * optimal_result['portfolio_weight'],
            'risk_rating': 'HIGH' if position_risk > 0.03 else 
                          'MEDIUM' if position_risk > 0.015 else 'LOW'
        }
        
    def _calculate_projected_heat(
        self, 
        adjustments: Dict, 
        portfolio_metrics: PortfolioMetrics
    ) -> float:
        """Calculate projected portfolio heat after adjustments."""
        # Simplified calculation - in production, this would be more sophisticated
        base_heat = portfolio_metrics.heat_level
        
        adjustment_impact = sum(
            adj['risk_impact']['risk_contribution'] 
            for adj in adjustments.values()
        ) * 0.5  # Damping factor
        
        return min(1.0, base_heat + adjustment_impact)
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Start real-time monitoring with health checks."""
        self.monitoring_active = True
        
        # Perform initial health checks
        health_status = self._perform_health_checks()
        
        logger.info("Real-time position monitoring started")
        return {
            "status": "monitoring_started",
            "timestamp": datetime.now().isoformat(),
            "health_status": health_status,
            "alert_thresholds": self.alert_thresholds
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        logger.info("Real-time position monitoring stopped")
        return {
            "status": "monitoring_stopped",
            "timestamp": datetime.now().isoformat(),
            "final_portfolio_metrics": self.portfolio_metrics_cache
        }
    
    def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks on portfolio and monitoring system."""
        health_status = {
            "overall_status": "HEALTHY",
            "checks_performed": [],
            "warnings": [],
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check data availability
        if not self.position_cache:
            health_status["errors"].append("No portfolio positions cached")
            health_status["overall_status"] = "ERROR"
        else:
            health_status["checks_performed"].append("portfolio_positions_available")
            
        if not self.market_data_cache:
            health_status["errors"].append("No market data cached")
            health_status["overall_status"] = "ERROR"
        else:
            health_status["checks_performed"].append("market_data_available")
        
        # Check portfolio health if data is available
        if self.position_cache and self.market_data_cache:
            try:
                # Calculate current portfolio metrics
                total_value = sum(pos.get('value', 0) for pos in self.position_cache.values())
                
                if total_value > 0:
                    # Check concentration risk
                    max_position_weight = max(
                        pos.get('value', 0) / total_value for pos in self.position_cache.values()
                    )
                    
                    if max_position_weight > self.alert_thresholds["concentration_critical"]:
                        health_status["errors"].append(f"Critical concentration risk: {max_position_weight:.1%}")
                        health_status["overall_status"] = "ERROR"
                    elif max_position_weight > self.alert_thresholds["concentration_warning"]:
                        health_status["warnings"].append(f"High concentration risk: {max_position_weight:.1%}")
                        if health_status["overall_status"] == "HEALTHY":
                            health_status["overall_status"] = "WARNING"
                    
                    health_status["checks_performed"].append("concentration_risk_check")
                    
                    # Check portfolio heat (simplified)
                    portfolio_heat = self._calculate_simple_portfolio_heat()
                    
                    if portfolio_heat > self.alert_thresholds["heat_critical"]:
                        health_status["errors"].append(f"Critical portfolio heat: {portfolio_heat:.1%}")
                        health_status["overall_status"] = "ERROR"
                    elif portfolio_heat > self.alert_thresholds["heat_warning"]:
                        health_status["warnings"].append(f"High portfolio heat: {portfolio_heat:.1%}")
                        if health_status["overall_status"] == "HEALTHY":
                            health_status["overall_status"] = "WARNING"
                    
                    health_status["checks_performed"].append("portfolio_heat_check")
                    
                    # Check for volatility spikes
                    volatility_alerts = self._check_volatility_spikes()
                    if volatility_alerts:
                        health_status["warnings"].extend(volatility_alerts)
                        if health_status["overall_status"] == "HEALTHY":
                            health_status["overall_status"] = "WARNING"
                    
                    health_status["checks_performed"].append("volatility_spike_check")
                    
            except Exception as e:
                health_status["errors"].append(f"Health check calculation error: {str(e)}")
                health_status["overall_status"] = "ERROR"
        
        return health_status
    
    def _calculate_simple_portfolio_heat(self) -> float:
        """Calculate simplified portfolio heat from cached data."""
        if not self.position_cache or not self.market_data_cache:
            return 0.0
            
        total_value = sum(pos.get('value', 0) for pos in self.position_cache.values())
        if total_value == 0:
            return 0.0
            
        portfolio_heat = 0.0
        for symbol, position in self.position_cache.items():
            if symbol in self.market_data_cache:
                position_weight = position.get('value', 0) / total_value
                volatility = self.market_data_cache[symbol].get('volatility', 0.20)
                position_heat = position_weight * volatility
                portfolio_heat += position_heat
                
        return portfolio_heat
    
    def _check_volatility_spikes(self) -> List[str]:
        """Check for significant volatility spikes."""
        alerts = []
        
        for symbol, market_data in self.market_data_cache.items():
            current_vol = market_data.get('volatility', 0.20)
            # In a real implementation, you'd compare to historical volatility
            # For now, we'll use a simple threshold
            if current_vol > 0.35:  # 35% annualized volatility
                alerts.append(f"{symbol} high volatility: {current_vol:.1%}")
                
        return alerts
    
    def generate_real_time_alerts(self) -> Dict[str, Any]:
        """Generate real-time alerts based on current portfolio state."""
        alerts = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": "INFO",
            "alerts": [],
            "recommendations": [],
            "immediate_actions_required": []
        }
        
        if not self.monitoring_active:
            alerts["alerts"].append("Real-time monitoring is not active")
            return alerts
            
        try:
            # Perform health checks
            health_status = self._perform_health_checks()
            
            if health_status["overall_status"] == "ERROR":
                alerts["alert_level"] = "CRITICAL"
                alerts["alerts"].extend(health_status["errors"])
                alerts["immediate_actions_required"].extend([
                    "Review portfolio positions immediately",
                    "Consider reducing position sizes",
                    "Check risk management parameters"
                ])
                
            elif health_status["overall_status"] == "WARNING":
                alerts["alert_level"] = "WARNING"
                alerts["alerts"].extend(health_status["warnings"])
                alerts["recommendations"].extend([
                    "Monitor portfolio closely",
                    "Consider rebalancing if conditions persist",
                    "Review position sizing parameters"
                ])
            
            # Check for rebalancing opportunities
            if len(self.position_cache) > 0 and len(self.market_data_cache) > 0:
                rebalance_analysis = self.calculate_real_time_adjustments(
                    portfolio_value=sum(pos.get('value', 0) for pos in self.position_cache.values()),
                    trigger_threshold=self.rebalance_threshold
                )
                
                if rebalance_analysis.get("status") == "success":
                    adjustments = rebalance_analysis.get("adjustments", {})
                    high_priority_count = sum(
                        1 for adj in adjustments.values() if adj.get("urgency") == "HIGH"
                    )
                    
                    if high_priority_count > 0:
                        alerts["alert_level"] = max(alerts["alert_level"], "WARNING")
                        alerts["alerts"].append(f"{high_priority_count} positions need high-priority rebalancing")
                        alerts["recommendations"].append("Execute high-priority rebalancing actions")
                    
                    elif len(adjustments) > 0:
                        alerts["recommendations"].append(f"{len(adjustments)} positions may benefit from rebalancing")
                        
        except Exception as e:
            alerts["alert_level"] = "ERROR"
            alerts["alerts"].append(f"Error generating alerts: {str(e)}")
            alerts["immediate_actions_required"].append("Check monitoring system health")
            
        return alerts
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive real-time dashboard data."""
        if not self.monitoring_active:
            return {"status": "monitoring_inactive"}
            
        try:
            total_value = sum(pos.get('value', 0) for pos in self.position_cache.values())
            
            # Current portfolio metrics
            portfolio_metrics = {
                "total_value": total_value,
                "position_count": len(self.position_cache),
                "heat_level": self._calculate_simple_portfolio_heat(),
                "largest_position_pct": max(
                    pos.get('value', 0) / total_value for pos in self.position_cache.values()
                ) if total_value > 0 and self.position_cache else 0,
                "cash_available": total_value * 0.1,  # Simplified assumption
            }
            
            # Position analysis
            position_analysis = {}
            for symbol, position in self.position_cache.items():
                if symbol in self.market_data_cache:
                    market_data = self.market_data_cache[symbol]
                    current_price = market_data.get('price', position.get('avg_cost', 100))
                    
                    position_analysis[symbol] = {
                        "current_value": position.get('value', 0),
                        "weight": position.get('value', 0) / total_value if total_value > 0 else 0,
                        "current_price": current_price,
                        "daily_change": market_data.get('daily_change', 0),
                        "volatility": market_data.get('volatility', 0.20),
                        "unrealized_pnl": (current_price - position.get('avg_cost', current_price)) * position.get('shares', 0)
                    }
            
            # Risk metrics
            risk_metrics = {
                "portfolio_heat": portfolio_metrics["heat_level"],
                "concentration_risk": "HIGH" if portfolio_metrics["largest_position_pct"] > 0.10 else 
                                   "MEDIUM" if portfolio_metrics["largest_position_pct"] > 0.08 else "LOW",
                "volatility_status": "HIGH" if any(
                    data.get('volatility', 0) > 0.30 for data in self.market_data_cache.values()
                ) else "NORMAL"
            }
            
            # Alerts
            alerts = self.generate_real_time_alerts()
            
            return {
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "portfolio_metrics": portfolio_metrics,
                "position_analysis": position_analysis,
                "risk_metrics": risk_metrics,
                "alerts": alerts,
                "monitoring_health": self._perform_health_checks(),
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e)
            }
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status and health metrics."""
        return {
            'status': 'active' if self.position_cache and self.market_data_cache else 'inactive',
            'positions_monitored': len(self.position_cache),
            'market_data_symbols': len(self.market_data_cache),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'rebalance_threshold': self.rebalance_threshold,
            'health_check': {
                'positions_cached': len(self.position_cache) > 0,
                'market_data_cached': len(self.market_data_cache) > 0,
                'recent_update': (
                    datetime.now() - self.last_update_time
                ).total_seconds() < 900 if self.last_update_time else False  # Within 15 minutes
            }
        }
        
    def set_rebalance_threshold(self, threshold: float) -> None:
        """Set the rebalance threshold."""
        if 0.01 <= threshold <= 0.50:  # 1% to 50%
            self.rebalance_threshold = threshold
            logger.info(f"Rebalance threshold set to {threshold:.1%}")
        else:
            raise ValueError("Rebalance threshold must be between 1% and 50%")
            
    def force_position_update(self, symbol: str, new_position_data: Dict) -> bool:
        """Force update of a specific position."""
        try:
            self.position_cache[symbol] = new_position_data
            logger.info(f"Force updated position for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to force update position for {symbol}: {e}")
            return False


class DiversificationAnalyzer:
    """Analyzes portfolio diversification and concentration risk."""
    
    def __init__(self):
        """Initialize diversification analyzer."""
        self.max_single_position = 0.10  # 10% max per position
        self.max_sector_exposure = 0.30  # 30% max per sector
        self.min_positions = 8  # Minimum positions for diversification
        self.max_positions = 20  # Maximum positions before over-diversification
        
    def analyze_diversification(
        self, 
        positions: Dict[str, Dict],
        correlation_matrix: np.ndarray = None
    ) -> Dict[str, any]:
        """Analyze portfolio diversification metrics."""
        if not positions:
            return {"status": "empty_portfolio", "diversification_score": 0.0}
            
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        if total_value == 0:
            return {"status": "zero_value", "diversification_score": 0.0}
            
        # Calculate position weights
        weights = {symbol: pos.get('value', 0) / total_value for symbol, pos in positions.items()}
        
        # Concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(weights)
        
        # Sector diversification
        sector_metrics = self._analyze_sector_diversification(positions)
        
        # Correlation-based diversification
        correlation_metrics = self._analyze_correlation_diversification(
            positions, correlation_matrix
        )
        
        # Overall diversification score
        diversification_score = self._calculate_diversification_score(
            concentration_metrics, sector_metrics, correlation_metrics
        )
        
        return {
            "status": "success",
            "diversification_score": diversification_score,
            "concentration_metrics": concentration_metrics,
            "sector_metrics": sector_metrics,
            "correlation_metrics": correlation_metrics,
            "recommendations": self._generate_diversification_recommendations(
                concentration_metrics, sector_metrics, diversification_score
            )
        }
        
    def _calculate_concentration_metrics(self, weights: Dict[str, float]) -> Dict[str, any]:
        """Calculate concentration risk metrics."""
        weight_values = list(weights.values())
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(w ** 2 for w in weight_values)
        
        # Top 3 concentration
        sorted_weights = sorted(weight_values, reverse=True)
        top3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        
        # Effective number of positions (1/HHI)
        effective_positions = 1.0 / hhi if hhi > 0 else 0
        
        # Largest position weight
        max_position_weight = max(weight_values) if weight_values else 0
        
        return {
            "herfindahl_index": hhi,
            "effective_positions": effective_positions,
            "top3_concentration": top3_concentration,
            "max_position_weight": max_position_weight,
            "position_count": len(weight_values),
            "concentration_risk": "HIGH" if hhi > 0.25 else "MEDIUM" if hhi > 0.125 else "LOW"
        }
        
    def _analyze_sector_diversification(self, positions: Dict[str, Dict]) -> Dict[str, any]:
        """Analyze sector-level diversification."""
        # Simplified sector mapping - in production, use real sector data
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'NVDA': 'Technology', 'META': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer'
        }
        
        sector_exposure = {}
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        
        for symbol, position in positions.items():
            sector = sector_map.get(symbol, 'Other')
            value = position.get('value', 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            
        # Convert to percentages
        sector_weights = {
            sector: value / total_value for sector, value in sector_exposure.items()
        } if total_value > 0 else {}
        
        # Check for over-concentration
        max_sector_exposure = max(sector_weights.values()) if sector_weights else 0
        sector_count = len(sector_weights)
        
        return {
            "sector_weights": sector_weights,
            "sector_count": sector_count,
            "max_sector_exposure": max_sector_exposure,
            "sector_hhi": sum(w ** 2 for w in sector_weights.values()),
            "sector_risk": "HIGH" if max_sector_exposure > 0.5 else "MEDIUM" if max_sector_exposure > 0.3 else "LOW"
        }
        
    def _analyze_correlation_diversification(
        self, 
        positions: Dict[str, Dict],
        correlation_matrix: np.ndarray = None
    ) -> Dict[str, any]:
        """Analyze correlation-based diversification."""
        if correlation_matrix is None or correlation_matrix.size == 0:
            return {
                "average_correlation": 0.3,  # Default assumption
                "correlation_risk": "UNKNOWN",
                "effective_diversification": 0.7
            }
            
        # Calculate average pairwise correlation
        if correlation_matrix.shape[0] > 1:
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(upper_triangle)
            max_correlation = np.max(upper_triangle)
            min_correlation = np.min(upper_triangle)
        else:
            avg_correlation = max_correlation = min_correlation = 0.0
            
        # Effective diversification ratio
        effective_diversification = 1.0 - abs(avg_correlation)
        
        return {
            "average_correlation": avg_correlation,
            "max_correlation": max_correlation,
            "min_correlation": min_correlation,
            "effective_diversification": effective_diversification,
            "correlation_risk": "HIGH" if abs(avg_correlation) > 0.7 else "MEDIUM" if abs(avg_correlation) > 0.4 else "LOW"
        }
        
    def _calculate_diversification_score(
        self,
        concentration_metrics: Dict,
        sector_metrics: Dict,
        correlation_metrics: Dict
    ) -> float:
        """Calculate overall diversification score (0-1, higher is better)."""
        # Position concentration component (0-1)
        position_score = min(1.0, concentration_metrics['effective_positions'] / 10.0)
        
        # Sector diversification component (0-1)
        sector_score = min(1.0, sector_metrics['sector_count'] / 8.0)
        sector_score *= (1.0 - sector_metrics['max_sector_exposure'])  # Penalty for concentration
        
        # Correlation diversification component (0-1)
        correlation_score = correlation_metrics['effective_diversification']
        
        # Weighted average
        diversification_score = (
            position_score * 0.4 +
            sector_score * 0.3 +
            correlation_score * 0.3
        )
        
        return min(1.0, max(0.0, diversification_score))
        
    def _generate_diversification_recommendations(
        self,
        concentration_metrics: Dict,
        sector_metrics: Dict,
        diversification_score: float
    ) -> List[str]:
        """Generate diversification improvement recommendations."""
        recommendations = []
        
        # Position concentration recommendations
        if concentration_metrics['max_position_weight'] > self.max_single_position:
            recommendations.append(
                f"Reduce largest position from {concentration_metrics['max_position_weight']:.1%} "
                f"to below {self.max_single_position:.1%}"
            )
            
        if concentration_metrics['position_count'] < self.min_positions:
            recommendations.append(
                f"Increase number of positions from {concentration_metrics['position_count']} "
                f"to at least {self.min_positions} for better diversification"
            )
            
        # Sector concentration recommendations
        if sector_metrics['max_sector_exposure'] > self.max_sector_exposure:
            recommendations.append(
                f"Reduce sector concentration from {sector_metrics['max_sector_exposure']:.1%} "
                f"to below {self.max_sector_exposure:.1%}"
            )
            
        if sector_metrics['sector_count'] < 4:
            recommendations.append(
                "Consider adding positions in different sectors for better diversification"
            )
            
        # Overall score recommendations
        if diversification_score < 0.6:
            recommendations.append(
                "Portfolio shows poor diversification - consider rebalancing across sectors and positions"
            )
        elif diversification_score < 0.8:
            recommendations.append(
                "Portfolio diversification can be improved - review position and sector weights"
            )
            
        return recommendations


class AdvancedPositionSizingEngine:
    """Advanced position sizing engine with multiple algorithms."""

    def __init__(self):
        """Initialize position sizing engine."""
        self.risk_config = RiskConfig()
        self.min_position_size = 100.0  # Minimum $100 position
        self.max_position_size_pct = 0.10  # Maximum 10% of portfolio
        self.max_heat_threshold = 0.30  # Maximum portfolio heat
        self.volatility_lookback = 20  # Days for volatility calculation
        self.correlation_threshold = 0.7  # High correlation threshold
        self.diversification_analyzer = DiversificationAnalyzer()

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        logger.info("Getting portfolio status")
        # This would typically fetch from database in production
        return {
            "total_balance": 100000.0,
            "available_balance": 80000.0,
            "positions": {
                "AAPL": {"value": 10000, "shares": 100, "avg_cost": 100.0},
                "MSFT": {"value": 10000, "shares": 50, "avg_cost": 200.0},
            },
            "portfolio_beta": 1.05,
            "portfolio_volatility": 0.18,
            "heat_level": 0.15
        }

    def calculate_kelly_criterion(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> float:
        """Calculate Kelly Criterion optimal position size.
        
        Formula: f* = (bp - q) / b
        Where:
        - b = odds received on the wager (avg_win / avg_loss)
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        
        Returns:
            Optimal fraction of capital to risk (0-1)
        """
        if avg_loss >= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        b = avg_win / abs(avg_loss)  # Odds ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly fraction limits (never risk more than 25% via Kelly)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Apply fractional Kelly (typically 25-50% of full Kelly)
        return kelly_fraction * 0.25  # Conservative 25% of Kelly

    def calculate_volatility_adjusted_size(
        self,
        base_size: float,
        asset_volatility: float,
        portfolio_volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """Calculate volatility-adjusted position size.
        
        Adjusts position size based on asset volatility relative to target.
        """
        if asset_volatility <= 0:
            return 0.0
            
        volatility_adjustment = target_volatility / asset_volatility
        adjusted_size = base_size * volatility_adjustment
        
        # Limit extreme adjustments
        return max(adjusted_size * 0.25, min(adjusted_size, base_size * 2.0))

    def calculate_correlation_adjusted_size(
        self,
        base_size: float,
        correlation_to_portfolio: float,
        existing_exposure: float
    ) -> float:
        """Calculate correlation-adjusted position size.
        
        Reduces position size for assets highly correlated to existing positions.
        """
        # Correlation penalty: higher correlation = smaller position
        correlation_penalty = abs(correlation_to_portfolio) ** 2
        
        # Existing exposure penalty
        exposure_penalty = min(existing_exposure / 0.20, 1.0)  # Penalty kicks in after 20% exposure
        
        # Combined adjustment
        adjustment_factor = 1.0 - (correlation_penalty * 0.5) - (exposure_penalty * 0.3)
        adjustment_factor = max(0.1, adjustment_factor)  # Minimum 10% of base size
        
        return base_size * adjustment_factor

    def calculate_portfolio_heat(
        self, 
        portfolio_metrics: PortfolioMetrics,
        new_position_risk: float,
        correlation_matrix: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate portfolio heat after adding new position.
        
        Portfolio heat measures the total risk exposure across all positions.
        Returns detailed heat analysis including correlation effects.
        """
        current_heat = portfolio_metrics.heat_level
        
        # Calculate heat contribution of new position
        position_heat = new_position_risk / portfolio_metrics.total_value
        
        # Simple additive heat
        simple_heat = current_heat + position_heat
        
        # Correlation-adjusted heat (more sophisticated)
        if correlation_matrix is not None and correlation_matrix.size > 0:
            # Use portfolio theory to calculate diversified heat
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            diversification_factor = 1.0 - (avg_correlation * 0.5)  # Reduce heat based on diversification
            correlation_adjusted_heat = simple_heat * diversification_factor
        else:
            correlation_adjusted_heat = simple_heat * 0.8  # Default 20% diversification benefit
        
        # Risk-parity adjusted heat
        position_count = len(portfolio_metrics.current_positions) + 1  # +1 for new position
        equal_weight_heat = position_heat * position_count  # If equally weighted
        concentration_penalty = max(1.0, equal_weight_heat / simple_heat)  # Penalty for concentration
        
        risk_parity_heat = simple_heat * min(concentration_penalty, 1.5)  # Cap penalty at 50%
        
        return {
            "simple_heat": min(simple_heat, 1.0),
            "correlation_adjusted_heat": min(correlation_adjusted_heat, 1.0),
            "risk_parity_heat": min(risk_parity_heat, 1.0),
            "recommended_heat": min(correlation_adjusted_heat, 1.0),
            "heat_breakdown": {
                "current_heat": current_heat,
                "new_position_heat": position_heat,
                "diversification_benefit": simple_heat - correlation_adjusted_heat,
                "concentration_penalty": risk_parity_heat - simple_heat
            }
        }

    def calculate_optimal_position_size(
        self,
        params: PositionSizingParams,
        portfolio_metrics: PortfolioMetrics,
        method: SizingMethod = SizingMethod.CORRELATION_ADJUSTED
    ) -> Dict[str, any]:
        """Calculate optimal position size using specified method."""
        logger.info(f"Calculating optimal position size for {params.symbol} using {method.value}")
        
        # Base position size (2% of portfolio)
        base_size = portfolio_metrics.total_value * 0.02
        
        # Calculate size based on method
        if method == SizingMethod.KELLY_CRITERION:
            kelly_fraction = self.calculate_kelly_criterion(
                params.win_rate, params.avg_win, params.avg_loss
            )
            optimal_size = portfolio_metrics.total_value * kelly_fraction
            
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            optimal_size = self.calculate_volatility_adjusted_size(
                base_size, params.volatility, portfolio_metrics.portfolio_volatility
            )
            
        elif method == SizingMethod.CORRELATION_ADJUSTED:
            # Get existing exposure to similar assets
            existing_exposure = self._calculate_sector_exposure(
                params.symbol, portfolio_metrics.current_positions
            )
            
            optimal_size = self.calculate_correlation_adjusted_size(
                base_size, params.correlation_to_portfolio, existing_exposure
            )
            
        elif method == SizingMethod.PORTFOLIO_HEAT:
            # Start with volatility-adjusted size
            vol_adjusted_size = self.calculate_volatility_adjusted_size(
                base_size, params.volatility, portfolio_metrics.portfolio_volatility
            )
            
            # Calculate potential heat from this position
            potential_risk = vol_adjusted_size * params.stop_loss_pct
            heat_analysis = self.calculate_portfolio_heat(
                portfolio_metrics, potential_risk, portfolio_metrics.portfolio_correlation_matrix
            )
            
            projected_heat = heat_analysis["recommended_heat"]
            
            # Reduce size if heat is too high
            if projected_heat > portfolio_metrics.max_heat_threshold:
                heat_adjustment = portfolio_metrics.max_heat_threshold / projected_heat
                optimal_size = vol_adjusted_size * heat_adjustment
            else:
                optimal_size = vol_adjusted_size
                
        else:  # FIXED_PERCENT
            optimal_size = base_size
        
        # Apply confidence adjustment
        optimal_size *= params.confidence
        
        # Apply hard limits
        max_position = portfolio_metrics.total_value * self.max_position_size_pct
        optimal_size = max(
            self.min_position_size,
            min(optimal_size, max_position, portfolio_metrics.available_cash)
        )
        
        # Calculate shares
        shares = int(optimal_size / params.current_price)
        actual_size = shares * params.current_price
        
        # Calculate risk metrics
        position_risk = actual_size * params.stop_loss_pct
        risk_percentage = position_risk / portfolio_metrics.total_value
        
        # Calculate portfolio heat impact
        heat_analysis = self.calculate_portfolio_heat(
            portfolio_metrics, position_risk, portfolio_metrics.portfolio_correlation_matrix
        )
        
        # Analyze diversification impact
        test_positions = portfolio_metrics.current_positions.copy()
        test_positions[params.symbol] = {"value": actual_size}
        diversification_analysis = self.diversification_analyzer.analyze_diversification(
            test_positions, portfolio_metrics.portfolio_correlation_matrix
        )
        
        # Generate warnings
        warnings = []
        if risk_percentage > 0.02:  # More than 2% risk
            warnings.append(f"Position risk {risk_percentage:.1%} exceeds 2% threshold")
        if heat_analysis["recommended_heat"] > self.max_heat_threshold:
            warnings.append(f"Portfolio heat {heat_analysis['recommended_heat']:.1%} exceeds {self.max_heat_threshold:.1%} threshold")
        if actual_size / portfolio_metrics.total_value > self.max_position_size_pct:
            warnings.append(f"Position size {actual_size / portfolio_metrics.total_value:.1%} exceeds {self.max_position_size_pct:.1%} limit")
        
        return {
            "symbol": params.symbol,
            "method": method.value,
            "optimal_size_usd": actual_size,
            "shares": shares,
            "position_risk_usd": position_risk,
            "risk_percentage": risk_percentage,
            "confidence_used": params.confidence,
            "price_per_share": params.current_price,
            "stop_loss_price": params.current_price * (1 - params.stop_loss_pct),
            "portfolio_weight": actual_size / portfolio_metrics.total_value,
            "heat_analysis": heat_analysis,
            "diversification_impact": {
                "current_score": diversification_analysis.get("diversification_score", 0.5),
                "concentration_risk": diversification_analysis.get("concentration_metrics", {}).get("concentration_risk", "UNKNOWN"),
                "sector_risk": diversification_analysis.get("sector_metrics", {}).get("sector_risk", "UNKNOWN")
            },
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "recommendations": diversification_analysis.get("recommendations", [])
        }

    def _calculate_sector_exposure(
        self, 
        symbol: str, 
        positions: Dict[str, Dict]
    ) -> float:
        """Calculate existing exposure to symbol's sector (simplified)."""
        # This is a simplified version - in production, you'd use actual sector data
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        
        if symbol in tech_stocks:
            tech_exposure = sum(
                pos['value'] for sym, pos in positions.items() 
                if sym in tech_stocks
            )
            total_value = sum(pos['value'] for pos in positions.values())
            return tech_exposure / total_value if total_value > 0 else 0.0
        
        return 0.0

    def calculate_position_size(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        stop_loss: float,
        portfolio_balance: float,
        current_positions: Dict,
        method: SizingMethod = SizingMethod.CORRELATION_ADJUSTED,
        **kwargs
    ) -> float:
        """Legacy interface for position sizing."""
        # Create parameters from legacy inputs
        params = PositionSizingParams(
            symbol=symbol,
            current_price=kwargs.get('current_price', 100.0),
            expected_return=kwargs.get('expected_return', 0.1),
            win_rate=kwargs.get('win_rate', 0.55),
            avg_win=kwargs.get('avg_win', 0.08),
            avg_loss=kwargs.get('avg_loss', -0.04),
            volatility=kwargs.get('volatility', 0.20),
            correlation_to_portfolio=kwargs.get('correlation', 0.3),
            stop_loss_pct=stop_loss,
            confidence=confidence
        )
        
        # Create portfolio metrics
        portfolio_metrics = PortfolioMetrics(
            total_value=portfolio_balance,
            available_cash=portfolio_balance * 0.8,
            current_positions=current_positions,
            portfolio_beta=1.0,
            portfolio_volatility=0.15,
            portfolio_correlation_matrix=np.eye(2),
            heat_level=0.15
        )
        
        result = self.calculate_optimal_position_size(params, portfolio_metrics, method)
        return result['optimal_size_usd']

    def validate_position_size(
        self, symbol: str, position_size: float, portfolio_status: Dict
    ) -> Dict:
        """Validate position size against risk limits."""
        max_position = portfolio_status["total_balance"] * self.max_position_size_pct
        risk_amount = position_size * RiskConfig.get_max_risk_per_trade()
        
        warnings = []
        if position_size > max_position:
            warnings.append(f"Position size {position_size} exceeds maximum {max_position}")
        if position_size < self.min_position_size:
            warnings.append(f"Position size {position_size} below minimum {self.min_position_size}")
            
        return {
            "is_valid": position_size <= max_position and position_size >= self.min_position_size,
            "max_allowed": max_position,
            "max_risk_amount": risk_amount,
            "warnings": warnings,
            "position_weight": position_size / portfolio_status["total_balance"]
        }


# Legacy class for backward compatibility
class PositionSizingEngine(AdvancedPositionSizingEngine):
    """Legacy position sizing engine - use AdvancedPositionSizingEngine instead."""
    pass


class RiskFilterEngine:
    """Applies risk management filters."""

    def filter_signals_by_risk(self, signals: Dict) -> Dict:
        """Filter trading signals by risk criteria."""
        logger.info("Filtering signals by risk criteria")

        filtered = {}
        for symbol, signal_data in signals.items():
            if signal_data.get("status") == "success":
                confidence = signal_data.get("confidence", 0)
                if confidence >= 0.6:  # Minimum confidence threshold
                    filtered[symbol] = signal_data

        return filtered

    def validate_positions_risk(self, positions: Dict) -> Dict:
        """Validate positions against risk limits."""
        validated = {}
        for symbol, pos_data in positions.items():
            if pos_data.get("status") == "success":
                validation = pos_data.get("validation", {})
                if validation.get("is_valid", False):
                    validated[symbol] = pos_data

        return validated

    def validate_portfolio_risk(self, portfolio_recs: Dict) -> Dict:
        """Validate portfolio-level risk."""
        if portfolio_recs.get("status") != "success":
            return {"is_valid": False, "reason": "Portfolio analysis failed"}

        risk_score = portfolio_recs.get("portfolio_risk", {}).get("overall_score", 0)

        return {
            "is_valid": risk_score <= 0.7,  # Max 70% risk score
            "risk_score": risk_score,
            "max_allowed": 0.7,
        }

    def apply_market_condition_filters(self, signals: Dict) -> Dict:
        """Apply market condition filters."""
        # Placeholder: In volatile markets, reduce signal strength
        return signals

    def generate_final_recommendations(
        self,
        filtered_signals: Dict,
        validated_positions: Dict,
        portfolio_validation: Dict,
        market_filters: Dict,
    ) -> List[Dict]:
        """Generate final filtered recommendations."""
        recommendations = []

        for symbol in filtered_signals.keys():
            if symbol in validated_positions and portfolio_validation.get("is_valid"):
                recommendations.append(
                    {
                        "symbol": symbol,
                        "action": filtered_signals[symbol].get("signal"),
                        "confidence": filtered_signals[symbol].get("confidence"),
                        "position_size": validated_positions[symbol].get(
                            "position_size"
                        ),
                        "risk_level": "MODERATE",
                    }
                )

        return recommendations

    def update_position_sizes_real_time(
        self, 
        current_positions: Dict[str, Dict],
        market_data: Dict[str, Dict],
        portfolio_value: float
    ) -> Dict[str, Dict]:
        """Update position sizes in real-time based on market changes."""
        logger.info("Updating position sizes in real-time")
        
        updated_positions = {}
        sizing_engine = AdvancedPositionSizingEngine()
        
        for symbol, position in current_positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('price', position.get('avg_cost', 100))
                volatility = market_data[symbol].get('volatility', 0.20)
                
                # Create sizing parameters
                params = PositionSizingParams(
                    symbol=symbol,
                    current_price=current_price,
                    expected_return=0.08,  # Default expected return
                    win_rate=0.55,
                    avg_win=0.06,
                    avg_loss=-0.03,
                    volatility=volatility,
                    correlation_to_portfolio=0.3,
                    stop_loss_pct=0.05,  # 5% stop loss
                    confidence=0.7
                )
                
                # Create portfolio metrics
                portfolio_metrics = PortfolioMetrics(
                    total_value=portfolio_value,
                    available_cash=portfolio_value * 0.1,  # Assume 10% cash
                    current_positions=current_positions,
                    portfolio_beta=1.0,
                    portfolio_volatility=0.15,
                    portfolio_correlation_matrix=np.eye(len(current_positions)),
                    heat_level=0.15
                )
                
                # Calculate optimal size
                optimal_result = sizing_engine.calculate_optimal_position_size(
                    params, portfolio_metrics, SizingMethod.CORRELATION_ADJUSTED
                )
                
                # Compare with current position
                current_value = position.get('value', 0)
                optimal_value = optimal_result['optimal_size_usd']
                
                size_difference = optimal_value - current_value
                size_difference_pct = size_difference / current_value if current_value > 0 else 0
                
                updated_positions[symbol] = {
                    'current_position': position,
                    'optimal_position': optimal_result,
                    'size_difference_usd': size_difference,
                    'size_difference_pct': size_difference_pct,
                    'action_needed': abs(size_difference_pct) > 0.1,  # Rebalance if >10% difference
                    'recommended_action': 'BUY' if size_difference > 1000 else 
                                        'SELL' if size_difference < -1000 else 'HOLD',
                    'priority': 'HIGH' if abs(size_difference_pct) > 0.2 else 
                               'MEDIUM' if abs(size_difference_pct) > 0.1 else 'LOW'
                }
        
        return updated_positions

    def calculate_overall_risk_score(self, recommendations: List[Dict]) -> float:
        """Calculate overall risk score for recommendations."""
        if not recommendations:
            return 0.0

        # Multi-factor risk scoring
        avg_confidence = sum(r.get("confidence", 0) for r in recommendations) / len(recommendations)
        
        # Portfolio concentration risk
        total_position_size = sum(r.get("position_size", 0) for r in recommendations)
        concentration_penalty = min(total_position_size / 100000, 0.3)  # Up to 30% penalty
        
        # Number of positions risk (too few = concentration, too many = over-diversification)
        position_count = len(recommendations)
        diversification_penalty = 0.1 if position_count < 3 or position_count > 15 else 0.0
        
        base_risk = 1.0 - avg_confidence
        total_risk = min(1.0, base_risk + concentration_penalty + diversification_penalty)
        
        return total_risk

    def calculate_portfolio_heat_real_time(
        self, 
        current_positions: Dict[str, Dict],
        market_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate real-time portfolio heat based on current positions and market data."""
        total_value = sum(pos.get('value', 0) for pos in current_positions.values())
        
        if total_value == 0:
            return {"heat_level": 0.0, "risk_adjusted_heat": 0.0}
        
        # Calculate individual position risks
        position_risks = []
        for symbol, position in current_positions.items():
            market_info = market_data.get(symbol, {})
            volatility = market_info.get('volatility', 0.20)  # Default 20% volatility
            position_value = position.get('value', 0)
            
            # Position risk = position_weight * volatility
            position_weight = position_value / total_value
            position_risk = position_weight * volatility
            position_risks.append(position_risk)
        
        # Simple heat calculation (sum of individual risks)
        heat_level = sum(position_risks)
        
        # Risk-adjusted heat (considers correlations - simplified)
        correlation_discount = 0.8  # Assume 80% correlation discount
        risk_adjusted_heat = heat_level * correlation_discount
        
        return {
            "heat_level": heat_level,
            "risk_adjusted_heat": risk_adjusted_heat,
            "individual_risks": dict(zip(current_positions.keys(), position_risks)),
            "heat_status": "HIGH" if heat_level > 0.3 else "MEDIUM" if heat_level > 0.15 else "LOW"
        }
