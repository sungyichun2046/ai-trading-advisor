"""Market regime classification engine.

This module provides comprehensive market regime detection and classification,
identifying Bull, Bear, Sideways, and Volatile market conditions across timeframes.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRANSITIONAL = "transitional"
    UNKNOWN = "unknown"


class RegimeStrength(Enum):
    """Regime strength classifications."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNCERTAIN = "uncertain"


@dataclass
class RegimeInfo:
    """Market regime information container."""
    regime: MarketRegime
    strength: RegimeStrength
    confidence: float
    start_time: datetime
    duration_days: float
    characteristics: Dict
    volatility_percentile: float
    trend_consistency: float


class MarketRegimeClassifier:
    """Market regime detection and classification engine."""
    
    def __init__(self):
        """Initialize market regime classifier."""
        self.regime_thresholds = {
            "bull_min_return": 0.15,      # 15% annual return
            "bear_max_return": -0.10,     # -10% annual return
            "volatility_high": 0.25,      # 25% annual volatility
            "volatility_low": 0.10,       # 10% annual volatility
            "sideways_max_trend": 0.05,   # 5% annual trend
            "trend_consistency_min": 0.60  # 60% trend consistency
        }
        
    def classify_regime(self, data: pd.DataFrame, lookback_days: int = 252) -> Dict:
        """Classify current market regime based on historical data.
        
        Args:
            data: OHLCV data with datetime index
            lookback_days: Number of days to analyze for regime classification
            
        Returns:
            Dictionary with regime classification results
        """
        if data.empty or len(data) < 30:
            logger.warning("Insufficient data for regime classification")
            return self._empty_regime_result()
        
        logger.info(f"Classifying market regime with {len(data)} data points over {lookback_days} days")
        
        try:
            # Use the last N days of data for classification
            analysis_data = data.tail(min(lookback_days, len(data)))
            
            # Calculate regime indicators
            returns_analysis = self._analyze_returns(analysis_data)
            volatility_analysis = self._analyze_volatility(analysis_data)
            trend_analysis = self._analyze_trend_characteristics(analysis_data)
            momentum_analysis = self._analyze_momentum(analysis_data)
            volume_analysis = self._analyze_volume_patterns(analysis_data)
            
            # Primary regime classification
            primary_regime = self._classify_primary_regime(
                returns_analysis, volatility_analysis, trend_analysis
            )
            
            # Secondary analysis for refinement
            regime_refinement = self._refine_regime_classification(
                primary_regime, momentum_analysis, volume_analysis, data
            )
            
            # Calculate regime persistence and stability
            stability_analysis = self._analyze_regime_stability(data, primary_regime)
            
            # Generate regime transitions and signals
            transition_analysis = self._analyze_regime_transitions(data, lookback_days)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "current_regime": primary_regime.__dict__,
                "refined_analysis": regime_refinement,
                "returns_analysis": returns_analysis,
                "volatility_analysis": volatility_analysis,
                "trend_analysis": trend_analysis,
                "momentum_analysis": momentum_analysis,
                "volume_analysis": volume_analysis,
                "stability_analysis": stability_analysis,
                "transition_analysis": transition_analysis,
                "regime_indicators": self._calculate_regime_indicators(analysis_data),
                "market_conditions": self._assess_market_conditions(analysis_data)
            }
            
        except Exception as e:
            logger.error(f"Error in regime classification: {e}")
            return {"error": str(e), "lookback_days": lookback_days}
    
    def _analyze_returns(self, data: pd.DataFrame) -> Dict:
        """Analyze return characteristics."""
        try:
            close = data['Close']
            returns = close.pct_change().dropna()
            
            if returns.empty:
                return {"error": "No return data available"}
            
            # Annualized metrics
            periods_per_year = 252  # Assume daily data
            mean_return = returns.mean() * periods_per_year
            return_volatility = returns.std() * np.sqrt(periods_per_year)
            
            # Risk-adjusted metrics
            sharpe_ratio = mean_return / return_volatility if return_volatility > 0 else 0
            
            # Return distribution analysis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Recent performance vs long-term
            recent_returns = returns.tail(30)  # Last 30 periods
            recent_mean = recent_returns.mean() * periods_per_year
            
            return {
                "annualized_return": mean_return,
                "annualized_volatility": return_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "recent_return": recent_mean,
                "return_skewness": skewness,
                "return_kurtosis": kurtosis,
                "positive_return_ratio": (returns > 0).mean(),
                "return_consistency": 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing returns: {e}")
            return {"error": str(e)}
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze volatility characteristics."""
        try:
            close = data['Close']
            returns = close.pct_change().dropna()
            
            if returns.empty:
                return {"error": "No return data available"}
            
            # Historical volatility measures
            periods_per_year = 252
            volatility = returns.std() * np.sqrt(periods_per_year)
            
            # Rolling volatility analysis
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(periods_per_year)
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
            vol_percentile = (rolling_vol <= current_vol).mean() if not rolling_vol.empty else 0.5
            
            # Volatility regime classification
            if volatility > self.regime_thresholds["volatility_high"]:
                vol_regime = "high"
            elif volatility < self.regime_thresholds["volatility_low"]:
                vol_regime = "low"
            else:
                vol_regime = "medium"
            
            # Volatility clustering (GARCH-like analysis)
            vol_clustering = self._measure_volatility_clustering(returns)
            
            # Volatility trend
            vol_trend = self._analyze_volatility_trend(rolling_vol)
            
            return {
                "current_volatility": volatility,
                "current_vol_percentile": vol_percentile,
                "volatility_regime": vol_regime,
                "volatility_clustering": vol_clustering,
                "volatility_trend": vol_trend,
                "rolling_volatility_stats": {
                    "mean": rolling_vol.mean() if not rolling_vol.empty else 0,
                    "std": rolling_vol.std() if not rolling_vol.empty else 0,
                    "min": rolling_vol.min() if not rolling_vol.empty else 0,
                    "max": rolling_vol.max() if not rolling_vol.empty else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {"error": str(e)}
    
    def _analyze_trend_characteristics(self, data: pd.DataFrame) -> Dict:
        """Analyze trend characteristics for regime classification."""
        try:
            close = data['Close']
            
            # Multiple timeframe trend analysis
            trend_windows = [20, 50, 100, 200]
            trend_signals = {}
            
            for window in trend_windows:
                if len(close) >= window:
                    ma = close.rolling(window=window).mean()
                    current_price = close.iloc[-1]
                    current_ma = ma.iloc[-1]
                    
                    # Price vs MA position
                    price_above_ma = current_price > current_ma
                    
                    # MA slope (trend direction)
                    ma_slope = (ma.iloc[-1] - ma.iloc[-10]) / ma.iloc[-10] if len(ma) >= 10 else 0
                    
                    trend_signals[f"ma_{window}"] = {
                        "price_above": price_above_ma,
                        "slope": ma_slope,
                        "slope_strength": "strong" if abs(ma_slope) > 0.02 else "weak"
                    }
            
            # Overall trend consistency
            price_above_count = sum(1 for signal in trend_signals.values() if signal["price_above"])
            trend_consistency = price_above_count / len(trend_signals) if trend_signals else 0.5
            
            # Trend strength analysis
            slope_values = [signal["slope"] for signal in trend_signals.values()]
            avg_slope = sum(slope_values) / len(slope_values) if slope_values else 0
            
            # Trend classification
            if trend_consistency > 0.75 and avg_slope > 0.01:
                trend_classification = "strong_uptrend"
            elif trend_consistency < 0.25 and avg_slope < -0.01:
                trend_classification = "strong_downtrend"
            elif abs(avg_slope) < 0.005:
                trend_classification = "sideways"
            else:
                trend_classification = "mixed"
            
            return {
                "trend_consistency": trend_consistency,
                "average_slope": avg_slope,
                "trend_classification": trend_classification,
                "individual_trends": trend_signals,
                "trend_strength": "strong" if abs(avg_slope) > 0.02 else "moderate" if abs(avg_slope) > 0.01 else "weak"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend characteristics: {e}")
            return {"error": str(e)}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum characteristics."""
        try:
            close = data['Close']
            
            # Multiple timeframe momentum
            momentum_periods = [5, 10, 20, 50]
            momentum_signals = {}
            
            for period in momentum_periods:
                if len(close) >= period + 1:
                    momentum = (close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]
                    momentum_signals[f"momentum_{period}"] = momentum
            
            # Rate of change analysis
            roc_10 = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] if len(close) > 10 else 0
            roc_20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) > 20 else 0
            
            # Momentum consistency
            positive_momentum_count = sum(1 for mom in momentum_signals.values() if mom > 0)
            momentum_consistency = positive_momentum_count / len(momentum_signals) if momentum_signals else 0.5
            
            # Momentum acceleration
            momentum_values = list(momentum_signals.values())
            if len(momentum_values) >= 2:
                momentum_acceleration = momentum_values[-1] - momentum_values[-2]
            else:
                momentum_acceleration = 0
            
            return {
                "momentum_signals": momentum_signals,
                "momentum_consistency": momentum_consistency,
                "momentum_acceleration": momentum_acceleration,
                "roc_10": roc_10,
                "roc_20": roc_20,
                "momentum_strength": "strong" if abs(roc_20) > 0.05 else "moderate" if abs(roc_20) > 0.02 else "weak"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {"error": str(e)}
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns for regime identification."""
        try:
            volume = data['Volume']
            close = data['Close']
            
            if volume.empty:
                return {"error": "No volume data available"}
            
            # Volume trend analysis
            volume_ma = volume.rolling(window=20).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_ma.iloc[-1] if not volume_ma.empty else volume.mean()
            
            # Volume relative to average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume-price relationship
            returns = close.pct_change().dropna()
            volume_aligned = volume[1:].values if len(volume) > len(returns) else volume.values
            
            if len(returns) == len(volume_aligned):
                # Correlation between absolute returns and volume
                vol_price_corr = np.corrcoef(abs(returns), volume_aligned)[0, 1] if len(returns) > 1 else 0
            else:
                vol_price_corr = 0
            
            # Volume distribution analysis
            volume_volatility = volume.std() / volume.mean() if volume.mean() > 0 else 0
            
            return {
                "current_volume_ratio": volume_ratio,
                "volume_price_correlation": vol_price_corr,
                "volume_volatility": volume_volatility,
                "volume_trend": "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "stable",
                "volume_regime": "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.7 else "normal"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            return {"error": str(e)}
    
    def _classify_primary_regime(self, returns_analysis: Dict, volatility_analysis: Dict, 
                                trend_analysis: Dict) -> RegimeInfo:
        """Classify primary market regime based on analysis results."""
        try:
            # Extract key metrics
            annual_return = returns_analysis.get("annualized_return", 0)
            volatility = volatility_analysis.get("current_volatility", 0)
            trend_consistency = trend_analysis.get("trend_consistency", 0.5)
            avg_slope = trend_analysis.get("average_slope", 0)
            
            # Primary classification logic
            regime = MarketRegime.UNKNOWN
            confidence = 0.5
            characteristics = {}
            
            # High volatility check first
            if volatility > self.regime_thresholds["volatility_high"]:
                regime = MarketRegime.VOLATILE
                confidence = min(0.8, volatility / self.regime_thresholds["volatility_high"])
                characteristics["high_volatility"] = True
                characteristics["volatility_percentile"] = volatility_analysis.get("current_vol_percentile", 0.5)
            
            # Bull market conditions
            elif (annual_return > self.regime_thresholds["bull_min_return"] and 
                  trend_consistency > self.regime_thresholds["trend_consistency_min"] and 
                  avg_slope > 0):
                regime = MarketRegime.BULL
                confidence = min(0.9, (annual_return / self.regime_thresholds["bull_min_return"]) * trend_consistency)
                characteristics["strong_uptrend"] = True
                characteristics["positive_momentum"] = True
            
            # Bear market conditions  
            elif (annual_return < self.regime_thresholds["bear_max_return"] and
                  trend_consistency < (1 - self.regime_thresholds["trend_consistency_min"]) and
                  avg_slope < 0):
                regime = MarketRegime.BEAR
                confidence = min(0.9, abs(annual_return / self.regime_thresholds["bear_max_return"]) * (1 - trend_consistency))
                characteristics["strong_downtrend"] = True
                characteristics["negative_momentum"] = True
            
            # Sideways market conditions
            elif (abs(annual_return) < self.regime_thresholds["sideways_max_trend"] and
                  abs(avg_slope) < 0.005):
                regime = MarketRegime.SIDEWAYS
                confidence = 0.7
                characteristics["range_bound"] = True
                characteristics["low_trend"] = True
            
            # Transitional conditions
            elif trend_consistency < 0.4 or trend_consistency > 0.6:
                regime = MarketRegime.TRANSITIONAL
                confidence = 0.6
                characteristics["mixed_signals"] = True
            
            # Determine strength
            if confidence > 0.8:
                strength = RegimeStrength.VERY_STRONG
            elif confidence > 0.7:
                strength = RegimeStrength.STRONG
            elif confidence > 0.6:
                strength = RegimeStrength.MODERATE
            elif confidence > 0.4:
                strength = RegimeStrength.WEAK
            else:
                strength = RegimeStrength.UNCERTAIN
            
            return RegimeInfo(
                regime=regime,
                strength=strength,
                confidence=confidence,
                start_time=datetime.now(),  # Simplified - would need historical analysis
                duration_days=0.0,  # Would need historical tracking
                characteristics=characteristics,
                volatility_percentile=volatility_analysis.get("current_vol_percentile", 0.5),
                trend_consistency=trend_consistency
            )
            
        except Exception as e:
            logger.error(f"Error in primary regime classification: {e}")
            return RegimeInfo(
                regime=MarketRegime.UNKNOWN,
                strength=RegimeStrength.UNCERTAIN,
                confidence=0.0,
                start_time=datetime.now(),
                duration_days=0.0,
                characteristics={},
                volatility_percentile=0.5,
                trend_consistency=0.5
            )
    
    def _refine_regime_classification(self, primary_regime: RegimeInfo, momentum_analysis: Dict,
                                    volume_analysis: Dict, data: pd.DataFrame) -> Dict:
        """Refine regime classification with additional analysis."""
        try:
            refinements = {
                "regime_adjustments": [],
                "confidence_adjustments": [],
                "secondary_characteristics": []
            }
            
            # Volume confirmation
            volume_price_corr = volume_analysis.get("volume_price_correlation", 0)
            volume_regime = volume_analysis.get("volume_regime", "normal")
            
            if primary_regime.regime == MarketRegime.BULL:
                if volume_price_corr > 0.3 and volume_regime == "high":
                    refinements["confidence_adjustments"].append(("volume_confirmation", 0.1))
                    refinements["secondary_characteristics"].append("volume_confirmed_bull")
                elif volume_regime == "low":
                    refinements["confidence_adjustments"].append(("weak_volume", -0.1))
            
            elif primary_regime.regime == MarketRegime.BEAR:
                if volume_price_corr > 0.3:
                    refinements["confidence_adjustments"].append(("volume_confirmation", 0.1))
                    refinements["secondary_characteristics"].append("volume_confirmed_bear")
            
            # Momentum confirmation
            momentum_consistency = momentum_analysis.get("momentum_consistency", 0.5)
            momentum_strength = momentum_analysis.get("momentum_strength", "weak")
            
            if momentum_strength == "strong":
                if primary_regime.regime in [MarketRegime.BULL, MarketRegime.BEAR]:
                    if momentum_consistency > 0.7:
                        refinements["confidence_adjustments"].append(("strong_momentum", 0.15))
                    elif momentum_consistency < 0.3:
                        refinements["regime_adjustments"].append("potential_reversal")
            
            # Volatility regime refinement
            if primary_regime.regime == MarketRegime.VOLATILE:
                # Check if it's volatile bull or volatile bear
                trend_classification = momentum_analysis.get("momentum_consistency", 0.5)
                if trend_classification > 0.6:
                    refinements["secondary_characteristics"].append("volatile_bull")
                elif trend_classification < 0.4:
                    refinements["secondary_characteristics"].append("volatile_bear")
                else:
                    refinements["secondary_characteristics"].append("volatile_sideways")
            
            # Market stress indicators
            stress_indicators = self._detect_market_stress(data)
            if stress_indicators["stress_level"] > 0.7:
                refinements["regime_adjustments"].append("market_stress")
                refinements["confidence_adjustments"].append(("market_stress", -0.2))
            
            return refinements
            
        except Exception as e:
            logger.error(f"Error refining regime classification: {e}")
            return {"error": str(e)}
    
    def _analyze_regime_stability(self, data: pd.DataFrame, current_regime: RegimeInfo) -> Dict:
        """Analyze regime stability and persistence."""
        try:
            stability = {
                "persistence_score": 0.5,
                "stability_trend": "stable",
                "regime_duration_estimate": 0,
                "transition_probability": 0.5
            }
            
            if len(data) < 50:
                return stability
            
            # Analyze historical regime consistency
            window_size = 20
            regime_scores = []
            
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                
                # Quick regime classification for this window
                returns = window_data['Close'].pct_change().mean() * 252
                volatility = window_data['Close'].pct_change().std() * np.sqrt(252)
                
                if returns > 0.1 and volatility < 0.25:
                    regime_scores.append(1)  # Bull-like
                elif returns < -0.1:
                    regime_scores.append(-1)  # Bear-like
                elif volatility > 0.25:
                    regime_scores.append(2)  # Volatile
                else:
                    regime_scores.append(0)  # Sideways
            
            if regime_scores:
                # Current regime numeric representation
                current_numeric = {
                    MarketRegime.BULL: 1,
                    MarketRegime.BEAR: -1,
                    MarketRegime.VOLATILE: 2,
                    MarketRegime.SIDEWAYS: 0,
                    MarketRegime.TRANSITIONAL: 0.5
                }.get(current_regime.regime, 0)
                
                # Calculate persistence
                similar_regimes = sum(1 for score in regime_scores[-10:] 
                                    if abs(score - current_numeric) < 0.5)
                stability["persistence_score"] = similar_regimes / min(10, len(regime_scores))
                
                # Trend in regime stability
                recent_consistency = len([s for s in regime_scores[-5:] if abs(s - current_numeric) < 0.5])
                older_consistency = len([s for s in regime_scores[-10:-5] if abs(s - current_numeric) < 0.5])
                
                if recent_consistency > older_consistency:
                    stability["stability_trend"] = "strengthening"
                elif recent_consistency < older_consistency:
                    stability["stability_trend"] = "weakening"
                else:
                    stability["stability_trend"] = "stable"
            
            return stability
            
        except Exception as e:
            logger.error(f"Error analyzing regime stability: {e}")
            return {"error": str(e)}
    
    def _analyze_regime_transitions(self, data: pd.DataFrame, lookback_days: int) -> Dict:
        """Analyze potential regime transitions."""
        try:
            transitions = {
                "transition_signals": [],
                "transition_probability": 0.0,
                "potential_new_regime": MarketRegime.UNKNOWN.value,
                "transition_timeframe": "unknown"
            }
            
            if len(data) < 100:
                return transitions
            
            # Analyze recent vs historical patterns
            recent_data = data.tail(30)  # Last 30 periods
            historical_data = data.tail(lookback_days).head(-30)  # Earlier periods
            
            # Compare key metrics
            recent_returns = recent_data['Close'].pct_change().mean() * 252
            historical_returns = historical_data['Close'].pct_change().mean() * 252
            
            recent_vol = recent_data['Close'].pct_change().std() * np.sqrt(252)
            historical_vol = historical_data['Close'].pct_change().std() * np.sqrt(252)
            
            # Detect significant changes
            return_change = abs(recent_returns - historical_returns)
            vol_change = abs(recent_vol - historical_vol)
            
            if return_change > 0.1:  # 10% change in annual returns
                transitions["transition_signals"].append("significant_return_shift")
                transitions["transition_probability"] += 0.3
            
            if vol_change > 0.1:  # 10% change in volatility
                transitions["transition_signals"].append("volatility_regime_shift")
                transitions["transition_probability"] += 0.2
            
            # Trend reversal signals
            if len(data) >= 50:
                ma_short = data['Close'].tail(20).mean()
                ma_long = data['Close'].tail(50).mean()
                
                if ma_short > ma_long * 1.02:  # Short MA significantly above long MA
                    if historical_returns < 0:  # Was bearish
                        transitions["potential_new_regime"] = MarketRegime.BULL.value
                        transitions["transition_signals"].append("bullish_crossover")
                elif ma_short < ma_long * 0.98:  # Short MA significantly below long MA
                    if historical_returns > 0:  # Was bullish
                        transitions["potential_new_regime"] = MarketRegime.BEAR.value
                        transitions["transition_signals"].append("bearish_crossover")
            
            # Volatility breakout
            if recent_vol > historical_vol * 1.5:
                transitions["potential_new_regime"] = MarketRegime.VOLATILE.value
                transitions["transition_signals"].append("volatility_breakout")
            
            # Estimate transition timeframe
            if transitions["transition_probability"] > 0.6:
                transitions["transition_timeframe"] = "imminent"
            elif transitions["transition_probability"] > 0.4:
                transitions["transition_timeframe"] = "short_term"
            elif transitions["transition_probability"] > 0.2:
                transitions["transition_timeframe"] = "medium_term"
            else:
                transitions["transition_timeframe"] = "low_probability"
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            return {"error": str(e)}
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate various regime indicators."""
        try:
            indicators = {}
            
            close = data['Close']
            volume = data['Volume']
            
            # Market breadth indicators (simplified)
            returns = close.pct_change().dropna()
            positive_returns = (returns > 0).sum()
            total_returns = len(returns)
            
            indicators["market_breadth"] = positive_returns / total_returns if total_returns > 0 else 0.5
            
            # Fear/Greed indicators
            volatility = returns.std() * np.sqrt(252)
            max_vol = returns.rolling(window=50).std().max() * np.sqrt(252) if len(returns) >= 50 else volatility
            fear_index = volatility / max_vol if max_vol > 0 else 0.5
            
            indicators["fear_greed_index"] = 1 - fear_index  # Inverted so high = greed
            
            # Momentum indicators
            if len(close) >= 20:
                roc_short = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
                roc_medium = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
                
                indicators["momentum_short"] = roc_short
                indicators["momentum_medium"] = roc_medium
            
            # Trend strength
            if len(close) >= 50:
                ma_20 = close.rolling(window=20).mean()
                ma_50 = close.rolling(window=50).mean()
                
                trend_strength = (ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
                indicators["trend_strength"] = trend_strength
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating regime indicators: {e}")
            return {"error": str(e)}
    
    def _assess_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Assess overall market conditions."""
        try:
            conditions = {
                "overall_health": "neutral",
                "risk_level": "medium",
                "opportunity_score": 0.5,
                "market_phase": "mature",
                "key_factors": []
            }
            
            close = data['Close']
            volume = data['Volume']
            
            # Health assessment
            returns = close.pct_change().dropna()
            if len(returns) >= 20:
                recent_performance = (returns.tail(20) > 0).mean()
                volatility = returns.tail(20).std() * np.sqrt(252)
                
                if recent_performance > 0.6 and volatility < 0.20:
                    conditions["overall_health"] = "strong"
                    conditions["risk_level"] = "low"
                elif recent_performance < 0.4 or volatility > 0.30:
                    conditions["overall_health"] = "weak"
                    conditions["risk_level"] = "high"
                else:
                    conditions["overall_health"] = "neutral"
                    conditions["risk_level"] = "medium"
            
            # Opportunity assessment
            if len(close) >= 50:
                current_price = close.iloc[-1]
                ma_50 = close.rolling(window=50).mean().iloc[-1]
                
                # Distance from long-term average
                distance_from_ma = (current_price - ma_50) / ma_50
                
                if distance_from_ma < -0.1:  # 10% below MA
                    conditions["opportunity_score"] = 0.8
                    conditions["key_factors"].append("oversold_vs_trend")
                elif distance_from_ma > 0.1:  # 10% above MA
                    conditions["opportunity_score"] = 0.2
                    conditions["key_factors"].append("overbought_vs_trend")
                else:
                    conditions["opportunity_score"] = 0.5
            
            # Market phase
            if len(returns) >= 100:
                long_term_trend = (close.iloc[-1] - close.iloc[-101]) / close.iloc[-101]
                recent_volatility = returns.tail(30).std() * np.sqrt(252)
                
                if long_term_trend > 0.2 and recent_volatility < 0.25:
                    conditions["market_phase"] = "bull_mature"
                elif long_term_trend < -0.2:
                    conditions["market_phase"] = "bear_active"
                elif recent_volatility > 0.30:
                    conditions["market_phase"] = "volatile_uncertain"
                else:
                    conditions["market_phase"] = "consolidation"
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {"error": str(e)}
    
    def _measure_volatility_clustering(self, returns: pd.Series) -> float:
        """Measure volatility clustering (GARCH-like behavior)."""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Calculate rolling volatility
            vol = returns.rolling(window=5).std()
            vol_changes = vol.diff().dropna()
            
            # Measure autocorrelation in volatility changes
            if len(vol_changes) > 10:
                lag_1_corr = vol_changes.autocorr(lag=1)
                return abs(lag_1_corr) if not pd.isna(lag_1_corr) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error measuring volatility clustering: {e}")
            return 0.0
    
    def _analyze_volatility_trend(self, rolling_vol: pd.Series) -> str:
        """Analyze the trend in volatility."""
        try:
            if len(rolling_vol) < 10:
                return "insufficient_data"
            
            # Linear trend in recent volatility
            recent_vol = rolling_vol.tail(10)
            x = np.arange(len(recent_vol))
            slope = np.polyfit(x, recent_vol, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing volatility trend: {e}")
            return "unknown"
    
    def _detect_market_stress(self, data: pd.DataFrame) -> Dict:
        """Detect market stress indicators."""
        try:
            stress_indicators = {
                "stress_level": 0.0,
                "stress_factors": []
            }
            
            close = data['Close']
            volume = data['Volume']
            
            # High volatility stress
            returns = close.pct_change().dropna()
            if len(returns) >= 20:
                current_vol = returns.tail(20).std() * np.sqrt(252)
                if current_vol > 0.35:  # 35% annual volatility
                    stress_indicators["stress_level"] += 0.3
                    stress_indicators["stress_factors"].append("high_volatility")
            
            # Large drawdowns
            if len(close) >= 50:
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.rolling(window=50).max()
                current_drawdown = (cumulative_returns.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
                
                if current_drawdown < -0.15:  # 15% drawdown
                    stress_indicators["stress_level"] += 0.4
                    stress_indicators["stress_factors"].append("significant_drawdown")
            
            # Volume spikes
            if not volume.empty and len(volume) >= 20:
                avg_volume = volume.tail(20).mean()
                recent_volume = volume.iloc[-1]
                
                if recent_volume > avg_volume * 2:  # 2x average volume
                    stress_indicators["stress_level"] += 0.2
                    stress_indicators["stress_factors"].append("volume_spike")
            
            # Gap analysis (if available)
            if len(close) >= 2:
                gap = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
                if gap > 0.05:  # 5% gap
                    stress_indicators["stress_level"] += 0.1
                    stress_indicators["stress_factors"].append("price_gap")
            
            stress_indicators["stress_level"] = min(stress_indicators["stress_level"], 1.0)
            
            return stress_indicators
            
        except Exception as e:
            logger.error(f"Error detecting market stress: {e}")
            return {"stress_level": 0.0, "stress_factors": []}
    
    def _empty_regime_result(self) -> Dict:
        """Return empty regime result for error cases."""
        return {
            "error": "Insufficient data",
            "current_regime": {
                "regime": MarketRegime.UNKNOWN.value,
                "strength": RegimeStrength.UNCERTAIN.value,
                "confidence": 0.0
            }
        }


class MultiTimeframeRegimeAnalyzer:
    """Analyzes market regimes across multiple timeframes."""
    
    def __init__(self):
        """Initialize multi-timeframe regime analyzer."""
        self.classifier = MarketRegimeClassifier()
        self.timeframes = {
            "short": 30,    # 30 days
            "medium": 90,   # 90 days (3 months)
            "long": 252     # 252 days (1 year)
        }
    
    def analyze_regime_hierarchy(self, data: pd.DataFrame) -> Dict:
        """Analyze regimes across multiple timeframes.
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            Dictionary with multi-timeframe regime analysis
        """
        logger.info("Starting multi-timeframe regime analysis")
        
        if data.empty:
            return {"error": "No data provided"}
        
        try:
            # Analyze each timeframe
            timeframe_results = {}
            for name, days in self.timeframes.items():
                timeframe_results[name] = self.classifier.classify_regime(data, days)
            
            # Cross-timeframe analysis
            regime_consensus = self._analyze_regime_consensus(timeframe_results)
            
            # Generate regime-based signals
            regime_signals = self._generate_regime_signals(timeframe_results, regime_consensus)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "timeframe_analysis": timeframe_results,
                "regime_consensus": regime_consensus,
                "regime_signals": regime_signals,
                "regime_summary": self._create_regime_summary(timeframe_results, regime_consensus)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe regime analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_regime_consensus(self, timeframe_results: Dict) -> Dict:
        """Analyze consensus across timeframe regimes."""
        try:
            consensus = {
                "dominant_regime": MarketRegime.UNKNOWN.value,
                "consensus_strength": 0.0,
                "regime_alignment": False,
                "conflicting_timeframes": []
            }
            
            # Extract regimes from each timeframe
            regimes = {}
            confidences = {}
            
            for timeframe, result in timeframe_results.items():
                if "error" not in result and "current_regime" in result:
                    regime_info = result["current_regime"]
                    regimes[timeframe] = regime_info["regime"]
                    confidences[timeframe] = regime_info["confidence"]
            
            if not regimes:
                return consensus
            
            # Count regime occurrences
            regime_counts = {}
            for regime in regimes.values():
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Determine dominant regime
            if regime_counts:
                dominant = max(regime_counts.items(), key=lambda x: x[1])
                consensus["dominant_regime"] = dominant[0]
                consensus["consensus_strength"] = dominant[1] / len(regimes)
                
                # Check alignment
                consensus["regime_alignment"] = dominant[1] == len(regimes)
                
                # Identify conflicts
                for timeframe, regime in regimes.items():
                    if regime != dominant[0]:
                        consensus["conflicting_timeframes"].append({
                            "timeframe": timeframe,
                            "regime": regime,
                            "confidence": confidences[timeframe]
                        })
            
            # Weighted consensus (longer timeframes more important)
            weights = {"short": 1.0, "medium": 2.0, "long": 3.0}
            weighted_scores = {}
            
            for timeframe, regime in regimes.items():
                weight = weights.get(timeframe, 1.0) * confidences.get(timeframe, 0.5)
                if regime not in weighted_scores:
                    weighted_scores[regime] = 0
                weighted_scores[regime] += weight
            
            if weighted_scores:
                total_weight = sum(weighted_scores.values())
                for regime in weighted_scores:
                    weighted_scores[regime] /= total_weight
                
                weighted_dominant = max(weighted_scores.items(), key=lambda x: x[1])
                consensus["weighted_dominant"] = weighted_dominant[0]
                consensus["weighted_strength"] = weighted_dominant[1]
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error analyzing regime consensus: {e}")
            return {"error": str(e)}
    
    def _generate_regime_signals(self, timeframe_results: Dict, consensus: Dict) -> Dict:
        """Generate trading signals based on regime analysis."""
        try:
            signals = {
                "regime_signal": "HOLD",
                "signal_strength": 0.0,
                "recommended_strategy": "wait_and_see",
                "risk_level": "medium",
                "regime_factors": []
            }
            
            if "error" in consensus:
                return signals
            
            dominant_regime = consensus.get("dominant_regime", MarketRegime.UNKNOWN.value)
            consensus_strength = consensus.get("consensus_strength", 0.0)
            regime_alignment = consensus.get("regime_alignment", False)
            
            # Generate signals based on dominant regime
            if dominant_regime == MarketRegime.BULL.value:
                signals["regime_signal"] = "BUY"
                signals["recommended_strategy"] = "momentum_following"
                signals["risk_level"] = "low" if regime_alignment else "medium"
                signals["regime_factors"].append("bullish_regime_detected")
                
            elif dominant_regime == MarketRegime.BEAR.value:
                signals["regime_signal"] = "SELL"
                signals["recommended_strategy"] = "defensive_hedging"
                signals["risk_level"] = "high"
                signals["regime_factors"].append("bearish_regime_detected")
                
            elif dominant_regime == MarketRegime.VOLATILE.value:
                signals["regime_signal"] = "NEUTRAL"
                signals["recommended_strategy"] = "volatility_trading"
                signals["risk_level"] = "high"
                signals["regime_factors"].append("high_volatility_regime")
                
            elif dominant_regime == MarketRegime.SIDEWAYS.value:
                signals["regime_signal"] = "NEUTRAL"
                signals["recommended_strategy"] = "range_trading"
                signals["risk_level"] = "medium"
                signals["regime_factors"].append("sideways_regime")
                
            else:
                signals["regime_signal"] = "HOLD"
                signals["recommended_strategy"] = "wait_and_see"
                signals["risk_level"] = "medium"
                signals["regime_factors"].append("uncertain_regime")
            
            # Adjust signal strength based on consensus
            signals["signal_strength"] = consensus_strength
            
            if regime_alignment:
                signals["regime_factors"].append("strong_timeframe_alignment")
                signals["signal_strength"] *= 1.2
            
            # Check for regime transitions
            conflicting_count = len(consensus.get("conflicting_timeframes", []))
            if conflicting_count > 0:
                signals["regime_factors"].append("potential_regime_transition")
                signals["signal_strength"] *= 0.8
                signals["risk_level"] = "high"
            
            signals["signal_strength"] = min(signals["signal_strength"], 1.0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {e}")
            return {"error": str(e)}
    
    def _create_regime_summary(self, timeframe_results: Dict, consensus: Dict) -> Dict:
        """Create a comprehensive regime summary."""
        try:
            summary = {
                "overall_assessment": "Unknown market conditions",
                "key_insights": [],
                "strategic_implications": [],
                "monitoring_priorities": []
            }
            
            if "error" in consensus:
                return summary
            
            dominant_regime = consensus.get("dominant_regime", MarketRegime.UNKNOWN.value)
            consensus_strength = consensus.get("consensus_strength", 0.0)
            
            # Overall assessment
            if consensus_strength > 0.8:
                confidence_desc = "High confidence"
            elif consensus_strength > 0.6:
                confidence_desc = "Moderate confidence"
            else:
                confidence_desc = "Low confidence"
            
            summary["overall_assessment"] = f"{confidence_desc} {dominant_regime} market regime"
            
            # Key insights
            if consensus.get("regime_alignment", False):
                summary["key_insights"].append("All timeframes show regime alignment")
            else:
                summary["key_insights"].append("Mixed signals across timeframes")
            
            # Add timeframe-specific insights
            for timeframe, result in timeframe_results.items():
                if "current_regime" in result:
                    regime_info = result["current_regime"]
                    if regime_info["confidence"] > 0.8:
                        summary["key_insights"].append(f"Strong {regime_info['regime']} signal in {timeframe} timeframe")
            
            # Strategic implications
            if dominant_regime == MarketRegime.BULL.value:
                summary["strategic_implications"] = [
                    "Favor long positions and growth strategies",
                    "Consider momentum-based approaches",
                    "Monitor for overextension signals"
                ]
            elif dominant_regime == MarketRegime.BEAR.value:
                summary["strategic_implications"] = [
                    "Emphasize capital preservation",
                    "Consider defensive positioning",
                    "Look for quality at discounted prices"
                ]
            elif dominant_regime == MarketRegime.VOLATILE.value:
                summary["strategic_implications"] = [
                    "Reduce position sizes",
                    "Implement volatility-based strategies",
                    "Maintain flexible approach"
                ]
            elif dominant_regime == MarketRegime.SIDEWAYS.value:
                summary["strategic_implications"] = [
                    "Range-bound trading opportunities",
                    "Mean reversion strategies",
                    "Patience for trend development"
                ]
            
            # Monitoring priorities
            summary["monitoring_priorities"] = [
                "Watch for regime transition signals",
                "Monitor cross-timeframe confirmations",
                "Track volatility regime changes"
            ]
            
            # Add specific monitoring based on current conditions
            if len(consensus.get("conflicting_timeframes", [])) > 0:
                summary["monitoring_priorities"].append("Resolve timeframe conflicts")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating regime summary: {e}")
            return {"error": str(e)}