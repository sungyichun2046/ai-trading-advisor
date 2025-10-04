"""Multi-horizon trend detection and analysis engine.

This module provides comprehensive trend analysis across multiple timeframes,
including trend identification, strength measurement, and reversal detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Trend strength enumeration."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class TrendInfo:
    """Trend information container."""
    direction: TrendDirection
    strength: TrendStrength
    confidence: float
    start_time: datetime
    duration_hours: float
    slope: float
    r_squared: float
    support_resistance: Optional[float] = None


class TrendDetector:
    """Multi-horizon trend detection engine."""
    
    def __init__(self):
        """Initialize trend detector."""
        self.min_trend_duration = {
            "short": 1,    # 1 hour minimum
            "medium": 4,   # 4 hours minimum  
            "long": 24     # 24 hours minimum
        }
        
    def detect_trends(self, data: pd.DataFrame, horizon: str = "medium") -> Dict:
        """Detect trends for specified horizon.
        
        Args:
            data: OHLCV data with datetime index
            horizon: Trend horizon ('short', 'medium', 'long')
            
        Returns:
            Dictionary with trend analysis results
        """
        if data.empty or len(data) < 10:
            logger.warning("Insufficient data for trend detection")
            return self._empty_trend_result()
        
        logger.info(f"Detecting {horizon}-term trends with {len(data)} data points")
        
        try:
            # Determine analysis window based on horizon
            window_size = self._get_analysis_window(horizon, len(data))
            analysis_data = data.tail(window_size)
            
            # Primary trend analysis using linear regression
            primary_trend = self._analyze_linear_trend(analysis_data['Close'], horizon)
            
            # Moving average trend analysis
            ma_trend = self._analyze_ma_trend(analysis_data, horizon)
            
            # Higher highs/lower lows analysis
            structure_trend = self._analyze_structure_trend(analysis_data, horizon)
            
            # Momentum-based trend analysis
            momentum_trend = self._analyze_momentum_trend(analysis_data, horizon)
            
            # Combine multiple trend detection methods
            consensus_trend = self._combine_trend_signals(
                primary_trend, ma_trend, structure_trend, momentum_trend
            )
            
            # Trend reversal detection
            reversal_signals = self._detect_reversal_signals(data, horizon)
            
            # Calculate trend persistence and reliability
            persistence = self._calculate_trend_persistence(data, consensus_trend, horizon)
            
            return {
                "horizon": horizon,
                "timestamp": datetime.now().isoformat(),
                "primary_trend": primary_trend.__dict__,
                "moving_average_trend": ma_trend,
                "structure_trend": structure_trend,
                "momentum_trend": momentum_trend,
                "consensus_trend": consensus_trend.__dict__,
                "reversal_signals": reversal_signals,
                "persistence": persistence,
                "data_quality": {
                    "total_points": len(data),
                    "analysis_points": len(analysis_data),
                    "time_span_hours": self._calculate_time_span(analysis_data),
                    "data_gaps": self._detect_data_gaps(analysis_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return {"error": str(e), "horizon": horizon}
    
    def _get_analysis_window(self, horizon: str, total_points: int) -> int:
        """Determine optimal analysis window size."""
        # Base window sizes for different horizons
        base_windows = {
            "short": min(50, total_points),      # ~2 days at 1h intervals
            "medium": min(168, total_points),    # ~1 week at 1h intervals
            "long": min(720, total_points)       # ~1 month at 1h intervals
        }
        
        return base_windows.get(horizon, min(168, total_points))
    
    def _analyze_linear_trend(self, prices: pd.Series, horizon: str) -> TrendInfo:
        """Analyze trend using linear regression."""
        try:
            # Convert index to numeric for regression
            x = np.arange(len(prices))
            y = prices.values
            
            # Calculate linear regression
            z = np.polyfit(x, y, 1)
            slope = z[0]
            intercept = z[1]
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            price_change_pct = (slope * len(prices)) / prices.iloc[0] * 100 if prices.iloc[0] != 0 else 0
            
            if abs(price_change_pct) < 1.0:  # Less than 1% change
                direction = TrendDirection.SIDEWAYS
            elif price_change_pct > 0:
                direction = TrendDirection.BULLISH
            else:
                direction = TrendDirection.BEARISH
            
            # Determine trend strength based on slope and R-squared
            strength = self._calculate_trend_strength(abs(price_change_pct), r_squared)
            
            # Calculate confidence based on R-squared and data quality
            confidence = min(r_squared * 1.2, 1.0)  # Boost for good fits
            
            return TrendInfo(
                direction=direction,
                strength=strength,
                confidence=confidence,
                start_time=prices.index[0],
                duration_hours=self._calculate_duration_hours(prices.index[0], prices.index[-1]),
                slope=slope,
                r_squared=r_squared
            )
            
        except Exception as e:
            logger.error(f"Error in linear trend analysis: {e}")
            return TrendInfo(
                direction=TrendDirection.UNKNOWN,
                strength=TrendStrength.NONE,
                confidence=0.0,
                start_time=datetime.now(),
                duration_hours=0.0,
                slope=0.0,
                r_squared=0.0
            )
    
    def _analyze_ma_trend(self, data: pd.DataFrame, horizon: str) -> Dict:
        """Analyze trend using moving averages."""
        try:
            close = data['Close']
            
            # Different MA periods for different horizons
            ma_periods = {
                "short": [5, 10, 20],
                "medium": [10, 20, 50],
                "long": [20, 50, 100]
            }
            
            periods = ma_periods.get(horizon, [10, 20, 50])
            mas = {}
            
            # Calculate moving averages
            for period in periods:
                if len(close) >= period:
                    mas[f"ma_{period}"] = close.rolling(window=period).mean()
            
            if not mas:
                return {"direction": TrendDirection.UNKNOWN.value, "alignment": 0.0}
            
            # Check MA alignment and price position
            current_price = close.iloc[-1]
            ma_values = []
            price_above_count = 0
            
            for ma_name, ma_series in mas.items():
                if not ma_series.empty and not pd.isna(ma_series.iloc[-1]):
                    ma_val = ma_series.iloc[-1]
                    ma_values.append(ma_val)
                    if current_price > ma_val:
                        price_above_count += 1
            
            if not ma_values:
                return {"direction": TrendDirection.UNKNOWN.value, "alignment": 0.0}
            
            # Check if MAs are aligned (shorter > longer for uptrend)
            ma_values_sorted = sorted(ma_values, reverse=True)
            ma_aligned = ma_values == ma_values_sorted or ma_values == ma_values_sorted[::-1]
            
            # Determine trend direction
            price_above_ratio = price_above_count / len(ma_values)
            
            if price_above_ratio >= 0.67:  # Price above most MAs
                direction = TrendDirection.BULLISH
            elif price_above_ratio <= 0.33:  # Price below most MAs
                direction = TrendDirection.BEARISH
            else:
                direction = TrendDirection.SIDEWAYS
            
            # Calculate alignment strength
            alignment = 1.0 if ma_aligned else 0.5
            
            return {
                "direction": direction.value,
                "alignment": alignment,
                "price_above_ratio": price_above_ratio,
                "ma_values": {f"ma_{p}": v for p, v in zip(periods, ma_values)}
            }
            
        except Exception as e:
            logger.error(f"Error in MA trend analysis: {e}")
            return {"direction": TrendDirection.UNKNOWN.value, "alignment": 0.0}
    
    def _analyze_structure_trend(self, data: pd.DataFrame, horizon: str) -> Dict:
        """Analyze trend using market structure (higher highs/lower lows)."""
        try:
            high = data['High']
            low = data['Low']
            
            # Find local peaks and troughs
            peaks = self._find_local_extremes(high, 'peaks')
            troughs = self._find_local_extremes(low, 'troughs')
            
            if len(peaks) < 2 or len(troughs) < 2:
                return {"direction": TrendDirection.UNKNOWN.value, "structure_score": 0.0}
            
            # Analyze recent peaks and troughs
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs
            
            # Check for higher highs and higher lows (uptrend)
            hh_count = sum(1 for i in range(1, len(recent_peaks)) 
                          if recent_peaks[i] > recent_peaks[i-1])
            hl_count = sum(1 for i in range(1, len(recent_troughs)) 
                          if recent_troughs[i] > recent_troughs[i-1])
            
            # Check for lower lows and lower highs (downtrend)
            ll_count = sum(1 for i in range(1, len(recent_troughs)) 
                          if recent_troughs[i] < recent_troughs[i-1])
            lh_count = sum(1 for i in range(1, len(recent_peaks)) 
                          if recent_peaks[i] < recent_peaks[i-1])
            
            # Calculate structure scores
            uptrend_score = (hh_count + hl_count) / (len(recent_peaks) + len(recent_troughs) - 2)
            downtrend_score = (ll_count + lh_count) / (len(recent_peaks) + len(recent_troughs) - 2)
            
            # Determine trend direction
            if uptrend_score > 0.6:
                direction = TrendDirection.BULLISH
                structure_score = uptrend_score
            elif downtrend_score > 0.6:
                direction = TrendDirection.BEARISH
                structure_score = downtrend_score
            else:
                direction = TrendDirection.SIDEWAYS
                structure_score = max(uptrend_score, downtrend_score)
            
            return {
                "direction": direction.value,
                "structure_score": structure_score,
                "uptrend_score": uptrend_score,
                "downtrend_score": downtrend_score,
                "peaks_count": len(recent_peaks),
                "troughs_count": len(recent_troughs)
            }
            
        except Exception as e:
            logger.error(f"Error in structure trend analysis: {e}")
            return {"direction": TrendDirection.UNKNOWN.value, "structure_score": 0.0}
    
    def _analyze_momentum_trend(self, data: pd.DataFrame, horizon: str) -> Dict:
        """Analyze trend using momentum indicators."""
        try:
            close = data['Close']
            
            # Calculate momentum indicators
            periods = {"short": 5, "medium": 14, "long": 21}
            period = periods.get(horizon, 14)
            
            # Rate of Change (ROC)
            roc = ((close - close.shift(period)) / close.shift(period) * 100).dropna()
            
            # Price momentum (simple momentum)
            momentum = (close - close.shift(period)).dropna()
            
            # Moving average of momentum
            if len(momentum) >= 5:
                momentum_ma = momentum.rolling(window=5).mean()
            else:
                momentum_ma = momentum
            
            if roc.empty or momentum.empty:
                return {"direction": TrendDirection.UNKNOWN.value, "momentum_score": 0.0}
            
            # Current momentum values
            current_roc = roc.iloc[-1] if not roc.empty else 0
            current_momentum = momentum.iloc[-1] if not momentum.empty else 0
            current_momentum_ma = momentum_ma.iloc[-1] if not momentum_ma.empty else 0
            
            # Determine momentum direction
            momentum_signals = []
            if current_roc > 1.0:  # Positive ROC > 1%
                momentum_signals.append(1)
            elif current_roc < -1.0:  # Negative ROC < -1%
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            if current_momentum > 0:
                momentum_signals.append(1)
            elif current_momentum < 0:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            if current_momentum_ma > 0:
                momentum_signals.append(1)
            elif current_momentum_ma < 0:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
            
            # Aggregate momentum score
            momentum_score = sum(momentum_signals) / len(momentum_signals)
            
            # Determine direction
            if momentum_score > 0.3:
                direction = TrendDirection.BULLISH
            elif momentum_score < -0.3:
                direction = TrendDirection.BEARISH
            else:
                direction = TrendDirection.SIDEWAYS
            
            return {
                "direction": direction.value,
                "momentum_score": abs(momentum_score),
                "roc": current_roc,
                "momentum": current_momentum,
                "momentum_ma": current_momentum_ma
            }
            
        except Exception as e:
            logger.error(f"Error in momentum trend analysis: {e}")
            return {"direction": TrendDirection.UNKNOWN.value, "momentum_score": 0.0}
    
    def _combine_trend_signals(self, primary: TrendInfo, ma_trend: Dict, 
                              structure_trend: Dict, momentum_trend: Dict) -> TrendInfo:
        """Combine multiple trend signals into consensus."""
        try:
            # Weight different methods
            weights = {
                "primary": 0.4,
                "ma": 0.25,
                "structure": 0.20,
                "momentum": 0.15
            }
            
            # Convert directions to numeric scores
            direction_scores = {
                TrendDirection.BULLISH.value: 1.0,
                TrendDirection.SIDEWAYS.value: 0.0,
                TrendDirection.BEARISH.value: -1.0,
                TrendDirection.UNKNOWN.value: 0.0
            }
            
            # Helper function to get direction value
            def get_direction_value(direction):
                if hasattr(direction, 'value'):
                    return direction.value
                return direction
            
            # Calculate weighted direction score
            signals = [
                (direction_scores[get_direction_value(primary.direction)], weights["primary"]),
                (direction_scores[get_direction_value(ma_trend["direction"])], weights["ma"]),
                (direction_scores[get_direction_value(structure_trend["direction"])], weights["structure"]),
                (direction_scores[get_direction_value(momentum_trend["direction"])], weights["momentum"])
            ]
            
            weighted_score = sum(score * weight for score, weight in signals)
            
            # Determine consensus direction
            if weighted_score > 0.3:
                consensus_direction = TrendDirection.BULLISH
            elif weighted_score < -0.3:
                consensus_direction = TrendDirection.BEARISH
            else:
                consensus_direction = TrendDirection.SIDEWAYS
            
            # Calculate consensus confidence
            confidence_factors = [
                primary.confidence * weights["primary"],
                ma_trend.get("alignment", 0.5) * weights["ma"],
                structure_trend.get("structure_score", 0.5) * weights["structure"],
                momentum_trend.get("momentum_score", 0.5) * weights["momentum"]
            ]
            
            consensus_confidence = sum(confidence_factors)
            
            # Determine consensus strength
            strength_score = abs(weighted_score) + consensus_confidence
            if strength_score > 1.2:
                consensus_strength = TrendStrength.VERY_STRONG
            elif strength_score > 0.8:
                consensus_strength = TrendStrength.STRONG
            elif strength_score > 0.5:
                consensus_strength = TrendStrength.MODERATE
            elif strength_score > 0.2:
                consensus_strength = TrendStrength.WEAK
            else:
                consensus_strength = TrendStrength.NONE
            
            return TrendInfo(
                direction=consensus_direction,
                strength=consensus_strength,
                confidence=min(consensus_confidence, 1.0),
                start_time=primary.start_time,
                duration_hours=primary.duration_hours,
                slope=primary.slope,
                r_squared=primary.r_squared
            )
            
        except Exception as e:
            logger.error(f"Error combining trend signals: {e}")
            return primary  # Fallback to primary trend
    
    def _detect_reversal_signals(self, data: pd.DataFrame, horizon: str) -> Dict:
        """Detect potential trend reversal signals."""
        try:
            if len(data) < 20:
                return {"reversal_probability": 0.0, "signals": []}
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            reversal_signals = []
            
            # Price exhaustion patterns
            recent_data = data.tail(10)
            price_range = recent_data['High'].max() - recent_data['Low'].min()
            recent_range = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0] * 100
            
            if abs(recent_range) > 5.0:  # Significant move in last 10 periods
                if recent_range > 0:
                    reversal_signals.append("potential_bullish_exhaustion")
                else:
                    reversal_signals.append("potential_bearish_exhaustion")
            
            # Divergence detection (simplified)
            if len(close) >= 20:
                # Price vs momentum divergence
                price_trend = close.iloc[-1] - close.iloc[-10]
                price_momentum = close.diff().tail(10).mean()
                
                if price_trend > 0 and price_momentum < 0:
                    reversal_signals.append("bearish_divergence")
                elif price_trend < 0 and price_momentum > 0:
                    reversal_signals.append("bullish_divergence")
            
            # Volume divergence
            if len(volume) >= 10:
                recent_volume = volume.tail(5).mean()
                previous_volume = volume.tail(10).head(5).mean()
                
                if recent_volume < previous_volume * 0.7:  # Volume declining
                    reversal_signals.append("volume_divergence")
            
            # Support/Resistance levels
            sr_levels = self._identify_support_resistance(data)
            current_price = close.iloc[-1]
            
            for level in sr_levels:
                if abs(current_price - level) / level < 0.02:  # Within 2% of level
                    reversal_signals.append(f"near_sr_level_{level:.2f}")
            
            # Calculate overall reversal probability
            reversal_probability = min(len(reversal_signals) * 0.2, 1.0)
            
            return {
                "reversal_probability": reversal_probability,
                "signals": reversal_signals,
                "support_resistance_levels": sr_levels
            }
            
        except Exception as e:
            logger.error(f"Error detecting reversal signals: {e}")
            return {"reversal_probability": 0.0, "signals": []}
    
    def _calculate_trend_persistence(self, data: pd.DataFrame, trend: TrendInfo, horizon: str) -> Dict:
        """Calculate trend persistence and reliability metrics."""
        try:
            if len(data) < 20:
                return {"persistence_score": 0.5, "reliability": "low"}
            
            close = data['Close']
            
            # Look at historical trend consistency
            lookback_periods = {"short": 20, "medium": 50, "long": 100}
            lookback = min(lookback_periods.get(horizon, 50), len(data))
            
            historical_data = data.tail(lookback)
            
            # Calculate rolling trend consistency
            window_size = max(10, lookback // 5)
            trend_directions = []
            
            for i in range(window_size, len(historical_data)):
                window_data = historical_data.iloc[i-window_size:i]
                start_price = window_data['Close'].iloc[0]
                end_price = window_data['Close'].iloc[-1]
                
                if end_price > start_price * 1.01:  # > 1% gain
                    trend_directions.append(1)
                elif end_price < start_price * 0.99:  # > 1% loss
                    trend_directions.append(-1)
                else:
                    trend_directions.append(0)
            
            if not trend_directions:
                return {"persistence_score": 0.5, "reliability": "low"}
            
            # Calculate persistence score
            if trend.direction == TrendDirection.BULLISH:
                persistence_score = sum(1 for d in trend_directions if d > 0) / len(trend_directions)
            elif trend.direction == TrendDirection.BEARISH:
                persistence_score = sum(1 for d in trend_directions if d < 0) / len(trend_directions)
            else:
                persistence_score = sum(1 for d in trend_directions if d == 0) / len(trend_directions)
            
            # Determine reliability
            if persistence_score > 0.7:
                reliability = "high"
            elif persistence_score > 0.5:
                reliability = "medium"
            else:
                reliability = "low"
            
            return {
                "persistence_score": persistence_score,
                "reliability": reliability,
                "historical_consistency": len([d for d in trend_directions if d != 0]) / len(trend_directions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend persistence: {e}")
            return {"persistence_score": 0.5, "reliability": "low"}
    
    def _find_local_extremes(self, series: pd.Series, extreme_type: str) -> List[float]:
        """Find local peaks or troughs in price series."""
        try:
            extremes = []
            window = 3  # Look for extremes in 3-point windows
            
            for i in range(window, len(series) - window):
                current = series.iloc[i]
                left_window = series.iloc[i-window:i]
                right_window = series.iloc[i+1:i+window+1]
                
                if extreme_type == 'peaks':
                    if all(current >= val for val in left_window) and all(current >= val for val in right_window):
                        extremes.append(current)
                else:  # troughs
                    if all(current <= val for val in left_window) and all(current <= val for val in right_window):
                        extremes.append(current)
            
            return extremes
            
        except Exception as e:
            logger.error(f"Error finding local extremes: {e}")
            return []
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> List[float]:
        """Identify key support and resistance levels."""
        try:
            high = data['High']
            low = data['Low']
            
            # Find significant highs and lows
            peaks = self._find_local_extremes(high, 'peaks')
            troughs = self._find_local_extremes(low, 'troughs')
            
            # Combine and find clusters
            all_levels = peaks + troughs
            if not all_levels:
                return []
            
            # Group similar levels (within 1% of each other)
            clustered_levels = []
            sorted_levels = sorted(all_levels)
            
            current_cluster = [sorted_levels[0]]
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.01:
                    current_cluster.append(level)
                else:
                    # Finalize current cluster
                    clustered_levels.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            # Add the last cluster
            if current_cluster:
                clustered_levels.append(sum(current_cluster) / len(current_cluster))
            
            # Return top 5 most significant levels
            return sorted(clustered_levels)[:5]
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return []
    
    def _calculate_trend_strength(self, price_change_pct: float, r_squared: float) -> TrendStrength:
        """Calculate trend strength based on price change and R-squared."""
        # Combine price change magnitude and regression fit quality
        strength_score = (abs(price_change_pct) / 10.0) * r_squared
        
        if strength_score > 2.0:
            return TrendStrength.VERY_STRONG
        elif strength_score > 1.0:
            return TrendStrength.STRONG
        elif strength_score > 0.5:
            return TrendStrength.MODERATE
        elif strength_score > 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE
    
    def _calculate_duration_hours(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate duration in hours between two timestamps."""
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        return (end_time - start_time).total_seconds() / 3600.0
    
    def _calculate_time_span(self, data: pd.DataFrame) -> float:
        """Calculate time span of data in hours."""
        if data.empty:
            return 0.0
        
        start_time = data.index[0]
        end_time = data.index[-1]
        return self._calculate_duration_hours(start_time, end_time)
    
    def _detect_data_gaps(self, data: pd.DataFrame) -> int:
        """Detect gaps in the data."""
        if len(data) < 2:
            return 0
        
        # Calculate expected intervals
        intervals = data.index.to_series().diff().dropna()
        if intervals.empty:
            return 0
        
        typical_interval = intervals.median()
        large_gaps = intervals > typical_interval * 2
        
        return int(large_gaps.sum())
    
    def _empty_trend_result(self) -> Dict:
        """Return empty trend result for error cases."""
        return {
            "error": "Insufficient data",
            "horizon": "unknown",
            "consensus_trend": {
                "direction": TrendDirection.UNKNOWN.value,
                "strength": TrendStrength.NONE.value,
                "confidence": 0.0
            }
        }


class CrossTimeframeTrendAnalyzer:
    """Analyzes trends across multiple timeframes for confirmation."""
    
    def __init__(self):
        """Initialize cross-timeframe analyzer."""
        self.trend_detector = TrendDetector()
        self.timeframes = ["short", "medium", "long"]
    
    def analyze_multi_horizon_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze trends across all horizons and find confirmations.
        
        Args:
            data: OHLCV data with datetime index
            
        Returns:
            Dictionary with multi-horizon trend analysis
        """
        logger.info("Starting cross-timeframe trend analysis")
        
        if data.empty:
            return {"error": "No data provided"}
        
        try:
            # Analyze each timeframe
            horizon_results = {}
            for horizon in self.timeframes:
                horizon_results[horizon] = self.trend_detector.detect_trends(data, horizon)
            
            # Cross-timeframe confirmation analysis
            confirmation = self._analyze_trend_confirmation(horizon_results)
            
            # Generate trading signals based on confirmations
            signals = self._generate_trend_signals(horizon_results, confirmation)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "horizon_analysis": horizon_results,
                "cross_timeframe_confirmation": confirmation,
                "trend_signals": signals,
                "overall_assessment": self._create_overall_assessment(horizon_results, confirmation)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-horizon trend analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_trend_confirmation(self, horizon_results: Dict) -> Dict:
        """Analyze confirmation between different timeframes."""
        try:
            confirmations = {
                "bullish_confirmation": 0,
                "bearish_confirmation": 0,
                "sideways_confirmation": 0,
                "conflicting_signals": 0,
                "dominant_trend": TrendDirection.UNKNOWN.value,
                "confirmation_strength": 0.0
            }
            
            # Extract trend directions for each horizon
            directions = {}
            confidences = {}
            
            for horizon, result in horizon_results.items():
                if "error" not in result and "consensus_trend" in result:
                    trend = result["consensus_trend"]
                    direction = trend["direction"]
                    # Ensure direction is a string value
                    if hasattr(direction, 'value'):
                        direction = direction.value
                    directions[horizon] = direction
                    confidences[horizon] = trend["confidence"]
            
            if not directions:
                return confirmations
            
            # Count confirmations
            direction_counts = {}
            for direction in directions.values():
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            # Determine dominant trend
            if direction_counts:
                dominant = max(direction_counts.items(), key=lambda x: x[1])
                confirmations["dominant_trend"] = dominant[0]
                
                # Calculate confirmation strength
                total_horizons = len(directions)
                max_agreement = dominant[1]
                confirmations["confirmation_strength"] = max_agreement / total_horizons
                
                # Update confirmation counts
                for direction, count in direction_counts.items():
                    if direction == TrendDirection.BULLISH.value:
                        confirmations["bullish_confirmation"] = count
                    elif direction == TrendDirection.BEARISH.value:
                        confirmations["bearish_confirmation"] = count
                    elif direction == TrendDirection.SIDEWAYS.value:
                        confirmations["sideways_confirmation"] = count
                
                # Calculate conflicting signals
                confirmations["conflicting_signals"] = total_horizons - max_agreement
            
            # Weighted confirmation (longer timeframes have more weight)
            weights = {"short": 1.0, "medium": 2.0, "long": 3.0}
            weighted_scores = {
                TrendDirection.BULLISH.value: 0,
                TrendDirection.BEARISH.value: 0,
                TrendDirection.SIDEWAYS.value: 0
            }
            
            total_weight = 0
            for horizon, direction in directions.items():
                weight = weights.get(horizon, 1.0) * confidences.get(horizon, 0.5)
                # Ensure direction is a valid key in weighted_scores
                if direction in weighted_scores:
                    weighted_scores[direction] += weight
                total_weight += weight
            
            if total_weight > 0:
                # Normalize weighted scores
                for direction in weighted_scores:
                    weighted_scores[direction] /= total_weight
                
                confirmations["weighted_confirmation"] = weighted_scores
                
                # Update dominant trend based on weighted analysis
                weighted_dominant = max(weighted_scores.items(), key=lambda x: x[1])
                if weighted_dominant[1] > 0.4:  # At least 40% weighted confidence
                    confirmations["dominant_trend"] = weighted_dominant[0]
                    confirmations["confirmation_strength"] = weighted_dominant[1]
            
            return confirmations
            
        except Exception as e:
            logger.error(f"Error analyzing trend confirmation: {e}")
            return {"error": str(e)}
    
    def _generate_trend_signals(self, horizon_results: Dict, confirmation: Dict) -> Dict:
        """Generate trading signals based on trend analysis."""
        try:
            signals = {
                "primary_signal": "HOLD",
                "signal_strength": 0.0,
                "entry_conditions": [],
                "exit_conditions": [],
                "risk_factors": [],
                "timeframe_alignment": False
            }
            
            if "error" in confirmation:
                return signals
            
            dominant_trend = confirmation.get("dominant_trend", TrendDirection.UNKNOWN.value)
            confirmation_strength = confirmation.get("confirmation_strength", 0.0)
            
            # Determine primary signal
            if dominant_trend == TrendDirection.BULLISH.value and confirmation_strength > 0.6:
                signals["primary_signal"] = "BUY"
                signals["signal_strength"] = confirmation_strength
                signals["entry_conditions"].append("Multi-timeframe bullish confirmation")
            elif dominant_trend == TrendDirection.BEARISH.value and confirmation_strength > 0.6:
                signals["primary_signal"] = "SELL"
                signals["signal_strength"] = confirmation_strength
                signals["entry_conditions"].append("Multi-timeframe bearish confirmation")
            else:
                signals["primary_signal"] = "HOLD"
                signals["signal_strength"] = 0.5
            
            # Check timeframe alignment
            bullish_count = confirmation.get("bullish_confirmation", 0)
            bearish_count = confirmation.get("bearish_confirmation", 0)
            conflicting = confirmation.get("conflicting_signals", 0)
            
            signals["timeframe_alignment"] = conflicting <= 1  # At most 1 conflicting signal
            
            # Add specific entry/exit conditions
            if signals["timeframe_alignment"]:
                if bullish_count >= 2:
                    signals["entry_conditions"].append("Strong bullish alignment across timeframes")
                elif bearish_count >= 2:
                    signals["entry_conditions"].append("Strong bearish alignment across timeframes")
            
            # Risk factors
            if conflicting > 1:
                signals["risk_factors"].append("Conflicting signals across timeframes")
            
            if confirmation_strength < 0.5:
                signals["risk_factors"].append("Low confirmation strength")
            
            # Check for reversal risks
            for horizon, result in horizon_results.items():
                if "reversal_signals" in result:
                    reversal_prob = result["reversal_signals"].get("reversal_probability", 0)
                    if reversal_prob > 0.6:
                        signals["risk_factors"].append(f"High reversal probability in {horizon} timeframe")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trend signals: {e}")
            return {"error": str(e)}
    
    def _create_overall_assessment(self, horizon_results: Dict, confirmation: Dict) -> Dict:
        """Create overall trend assessment summary."""
        try:
            assessment = {
                "market_phase": "Unknown",
                "trend_quality": "Low",
                "conviction_level": "Low",
                "recommended_action": "Wait",
                "key_insights": []
            }
            
            if "error" in confirmation:
                return assessment
            
            dominant_trend = confirmation.get("dominant_trend", TrendDirection.UNKNOWN.value)
            confirmation_strength = confirmation.get("confirmation_strength", 0.0)
            conflicting_signals = confirmation.get("conflicting_signals", 0)
            
            # Determine market phase
            if dominant_trend == TrendDirection.BULLISH.value:
                assessment["market_phase"] = "Bullish Trend"
            elif dominant_trend == TrendDirection.BEARISH.value:
                assessment["market_phase"] = "Bearish Trend"
            elif dominant_trend == TrendDirection.SIDEWAYS.value:
                assessment["market_phase"] = "Sideways/Consolidation"
            else:
                assessment["market_phase"] = "Uncertain/Transitional"
            
            # Determine trend quality
            if confirmation_strength > 0.8 and conflicting_signals == 0:
                assessment["trend_quality"] = "Very High"
                assessment["conviction_level"] = "High"
            elif confirmation_strength > 0.6 and conflicting_signals <= 1:
                assessment["trend_quality"] = "High"
                assessment["conviction_level"] = "Medium-High"
            elif confirmation_strength > 0.4:
                assessment["trend_quality"] = "Medium"
                assessment["conviction_level"] = "Medium"
            else:
                assessment["trend_quality"] = "Low"
                assessment["conviction_level"] = "Low"
            
            # Recommended action
            if assessment["conviction_level"] in ["High", "Medium-High"]:
                if dominant_trend == TrendDirection.BULLISH.value:
                    assessment["recommended_action"] = "Consider Long Position"
                elif dominant_trend == TrendDirection.BEARISH.value:
                    assessment["recommended_action"] = "Consider Short Position"
                else:
                    assessment["recommended_action"] = "Range Trading Strategy"
            else:
                assessment["recommended_action"] = "Wait for Clearer Signals"
            
            # Key insights
            if confirmation_strength > 0.7:
                assessment["key_insights"].append("Strong cross-timeframe trend confirmation")
            
            if conflicting_signals == 0:
                assessment["key_insights"].append("Perfect timeframe alignment")
            elif conflicting_signals > 2:
                assessment["key_insights"].append("High uncertainty due to conflicting timeframes")
            
            # Add strength insights from individual horizons
            for horizon, result in horizon_results.items():
                if "consensus_trend" in result:
                    trend = result["consensus_trend"]
                    if trend["strength"] in ["very_strong", "strong"]:
                        assessment["key_insights"].append(f"Strong {horizon}-term trend detected")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error creating overall assessment: {e}")
            return {"error": str(e)}