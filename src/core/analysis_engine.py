"""Enhanced Analysis Engine Module
Advanced analysis with multi-timeframe technical indicators, pattern recognition, and trend analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum

# Import shared utilities
try:
    from ..utils.shared import calculate_returns, validate_data_quality
except ImportError:
    def calculate_returns(prices, periods=1):
        if len(prices) < periods + 1:
            return []
        return [(prices[i] - prices[i-periods]) / prices[i-periods] for i in range(periods, len(prices))]
    
    def validate_data_quality(data, data_type="general", min_threshold=0.8):
        return {'quality_score': 0.8, 'issues': [], 'data_type': data_type}

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class TechnicalAnalyzer:
    """Enhanced technical analysis with multi-timeframe indicators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.timeframes = ['1h', '1d']
        
    def calculate_indicators(self, data: pd.DataFrame, timeframe: str = '1h') -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for specific timeframe."""
        if data.empty or len(data) < 20:
            return {'timeframe': timeframe, 'indicators': {}}
        
        try:
            indicators = {
                'timeframe': timeframe,
                'data_quality': validate_data_quality({'status': 'success', 'rows': len(data)}, 'market'),
                'indicators': {}
            }
            
            if 'Close' in data.columns:
                prices = data['Close'].tolist()
                indicators['indicators'].update({
                    'rsi': self.calculate_rsi(data['Close']),
                    'macd': self.calculate_macd(data['Close']),
                    'bollinger': self.calculate_bollinger_bands(data['Close']),
                    'atr': self.calculate_atr(data),
                    'adx': self.calculate_adx(data),
                    'stochastic': self.calculate_stochastic(data),
                    'returns': calculate_returns(prices, 1),
                    'trend': self.detect_trend(data)
                })
                
            if 'Volume' in data.columns:
                indicators['indicators']['volume'] = self.analyze_volume(data)
                
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {'timeframe': timeframe, 'indicators': {}}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return {"current": 50.0, "signal": "neutral"}
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
            
            if current_rsi > 70:
                signal = "overbought"
            elif current_rsi < 30:
                signal = "oversold"
            else:
                signal = "neutral"
            
            return {"current": round(current_rsi, 2), "signal": signal}
            
        except Exception:
            return {"current": 50.0, "signal": "neutral"}
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """Calculate MACD indicator."""
        if len(prices) < max(fast, slow, signal) + 1:
            return {"crossover": "neutral", "histogram": 0.0}
        
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            if len(histogram) >= 2:
                if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                    crossover = "bullish"
                elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                    crossover = "bearish"
                else:
                    crossover = "neutral"
            else:
                crossover = "neutral"
            
            return {"crossover": crossover, "histogram": round(histogram.iloc[-1], 4)}
            
        except Exception:
            return {"crossover": "neutral", "histogram": 0.0}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"position": "neutral", "bandwidth": 0.0}
        
        try:
            sma = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper = sma + (rolling_std * std)
            lower = sma - (rolling_std * std)
            
            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1] if not upper.empty else current_price
            current_lower = lower.iloc[-1] if not lower.empty else current_price
            current_sma = sma.iloc[-1] if not sma.empty else current_price
            
            if current_price > current_upper:
                position = "above_upper"
            elif current_price < current_lower:
                position = "below_lower"
            else:
                position = "within_bands"
            
            bandwidth = ((current_upper - current_lower) / current_sma) * 100 if current_sma > 0 else 0.0
            
            return {"position": position, "bandwidth": round(bandwidth, 2)}
            
        except Exception:
            return {"position": "neutral", "bandwidth": 0.0}
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Average True Range for volatility measurement."""
        if len(data) < period or not all(col in data.columns for col in ['High', 'Low', 'Close']):
            return {"current": 0.0, "volatility_level": "unknown"}
        
        try:
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            current_atr = atr.iloc[-1] if not atr.empty else 0.0
            
            # Determine volatility level
            atr_ma = atr.rolling(20).mean().iloc[-1] if len(atr) >= 20 else current_atr
            if current_atr > atr_ma * 1.5:
                volatility_level = "high"
            elif current_atr < atr_ma * 0.7:
                volatility_level = "low"
            else:
                volatility_level = "normal"
            
            return {"current": round(current_atr, 4), "volatility_level": volatility_level}
            
        except Exception:
            return {"current": 0.0, "volatility_level": "unknown"}
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Average Directional Index for trend strength."""
        if len(data) < period + 1 or not all(col in data.columns for col in ['High', 'Low', 'Close']):
            return {"current": 0.0, "trend_strength": "weak"}
        
        try:
            high, low, close = data['High'], data['Low'], data['Close']
            
            # Calculate directional movement
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smoothed averages
            atr = true_range.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            # ADX calculation
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            current_adx = adx.iloc[-1] if not adx.empty else 0.0
            
            if current_adx > 25:
                trend_strength = "strong"
            elif current_adx > 15:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            return {"current": round(current_adx, 2), "trend_strength": trend_strength}
            
        except Exception:
            return {"current": 0.0, "trend_strength": "weak"}
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator for momentum."""
        if len(data) < k_period or not all(col in data.columns for col in ['High', 'Low', 'Close']):
            return {"k_percent": 50.0, "d_percent": 50.0, "signal": "neutral"}
        
        try:
            lowest_low = data['Low'].rolling(k_period).min()
            highest_high = data['High'].rolling(k_period).max()
            
            k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(d_period).mean()
            
            current_k = k_percent.iloc[-1] if not k_percent.empty else 50.0
            current_d = d_percent.iloc[-1] if not d_percent.empty else 50.0
            
            if current_k > 80 and current_d > 80:
                signal = "overbought"
            elif current_k < 20 and current_d < 20:
                signal = "oversold"
            elif current_k > current_d and len(k_percent) > 1 and k_percent.iloc[-2] <= d_percent.iloc[-2]:
                signal = "bullish_crossover"
            elif current_k < current_d and len(k_percent) > 1 and k_percent.iloc[-2] >= d_percent.iloc[-2]:
                signal = "bearish_crossover"
            else:
                signal = "neutral"
            
            return {"k_percent": round(current_k, 2), "d_percent": round(current_d, 2), "signal": signal}
            
        except Exception:
            return {"k_percent": 50.0, "d_percent": 50.0, "signal": "neutral"}
    
    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Multi-timeframe trend detection."""
        if data.empty or 'Close' not in data.columns or len(data) < 20:
            return {"direction": TrendDirection.UNKNOWN.value, "strength": 0.0, "alignment": "unknown"}
        
        try:
            prices = data['Close']
            
            # Multiple moving averages for trend confirmation
            ma_short = prices.rolling(10).mean()
            ma_medium = prices.rolling(20).mean()
            ma_long = prices.rolling(50).mean() if len(prices) >= 50 else ma_medium
            
            if not ma_short.empty and not ma_medium.empty:
                current_short = ma_short.iloc[-1]
                current_medium = ma_medium.iloc[-1]
                current_long = ma_long.iloc[-1] if not ma_long.empty else current_medium
                
                # Trend alignment analysis
                if current_short > current_medium > current_long:
                    direction = TrendDirection.BULLISH
                    strength = min((current_short - current_long) / current_long * 100, 100)
                    alignment = "strong_bullish"
                elif current_short < current_medium < current_long:
                    direction = TrendDirection.BEARISH
                    strength = min((current_long - current_short) / current_long * 100, 100)
                    alignment = "strong_bearish"
                elif current_short > current_medium:
                    direction = TrendDirection.BULLISH
                    strength = min((current_short - current_medium) / current_medium * 100, 50)
                    alignment = "weak_bullish"
                elif current_short < current_medium:
                    direction = TrendDirection.BEARISH
                    strength = min((current_medium - current_short) / current_medium * 100, 50)
                    alignment = "weak_bearish"
                else:
                    direction = TrendDirection.SIDEWAYS
                    strength = 0.0
                    alignment = "sideways"
            else:
                direction = TrendDirection.UNKNOWN
                strength = 0.0
                alignment = "unknown"
            
            return {
                "direction": direction.value if hasattr(direction, 'value') else str(direction), 
                "strength": round(strength, 2), 
                "alignment": alignment
            }
            
        except Exception:
            return {"direction": TrendDirection.UNKNOWN.value, "strength": 0.0, "alignment": "unknown"}
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced volume analysis with pattern detection."""
        if 'Volume' not in data.columns or len(data) < 10:
            return {"trend": "unknown", "relative_volume": 1.0, "volume_pattern": "none"}
        
        try:
            volume = data['Volume']
            volume_ma = volume.rolling(10).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_ma.iloc[-1] if not volume_ma.empty else 1
            
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend
            if relative_volume > 1.5:
                trend = "increasing"
            elif relative_volume < 0.5:
                trend = "decreasing"
            else:
                trend = "normal"
            
            # Volume pattern detection
            if len(data) >= 5:
                recent_volumes = volume.tail(5)
                if recent_volumes.is_monotonic_increasing:
                    volume_pattern = "accumulation"
                elif recent_volumes.is_monotonic_decreasing:
                    volume_pattern = "distribution"
                else:
                    volume_pattern = "mixed"
            else:
                volume_pattern = "insufficient_data"
            
            return {
                "trend": trend, 
                "relative_volume": round(relative_volume, 2),
                "volume_pattern": volume_pattern
            }
            
        except Exception:
            return {"trend": "unknown", "relative_volume": 1.0, "volume_pattern": "none"}


class PatternAnalyzer:
    """Enhanced pattern recognition with support/resistance and breakout detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def detect_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced chart patterns."""
        if data.empty or len(data) < 20:
            return {"patterns": [], "count": 0, "breakout_signals": []}
        
        try:
            patterns = []
            breakout_signals = []
            
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                # Support and resistance levels
                support_resistance = self.detect_support_resistance(data)
                
                # Triangle patterns
                triangle_pattern = self.detect_triangle_patterns(data)
                if triangle_pattern:
                    patterns.append(triangle_pattern)
                
                # Breakout detection
                breakouts = self.detect_breakouts(data, support_resistance)
                breakout_signals.extend(breakouts)
            
            return {
                "patterns": patterns, 
                "count": len(patterns),
                "breakout_signals": breakout_signals,
                "support_resistance": support_resistance if 'support_resistance' in locals() else {}
            }
            
        except Exception:
            return {"patterns": [], "count": 0, "breakout_signals": []}
    
    def detect_triangle_patterns(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        try:
            if len(data) < 10:
                return None
            
            recent_highs = data['High'].tail(10)
            recent_lows = data['Low'].tail(10)
            
            # Ascending triangle: horizontal resistance, rising support
            if (recent_highs.std() < recent_highs.mean() * 0.02 and 
                recent_lows.is_monotonic_increasing):
                return {
                    "pattern_type": "ascending_triangle",
                    "confidence": 0.7,
                    "direction": "bullish"
                }
            
            # Descending triangle: falling resistance, horizontal support
            elif (recent_lows.std() < recent_lows.mean() * 0.02 and 
                  recent_highs.is_monotonic_decreasing):
                return {
                    "pattern_type": "descending_triangle",
                    "confidence": 0.7,
                    "direction": "bearish"
                }
            
            # Symmetrical triangle: converging support and resistance
            elif (recent_highs.is_monotonic_decreasing and 
                  recent_lows.is_monotonic_increasing):
                return {
                    "pattern_type": "symmetrical_triangle",
                    "confidence": 0.6,
                    "direction": "neutral"
                }
            
            return None
            
        except Exception:
            return None
    
    def detect_support_resistance(self, data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
        """Enhanced support and resistance detection."""
        if data.empty or len(data) < window * 2:
            return {"support_levels": [], "resistance_levels": []}
        
        try:
            highs = data['High'] if 'High' in data.columns else data['Close']
            lows = data['Low'] if 'Low' in data.columns else data['Close']
            
            resistance_levels = []
            support_levels = []
            
            # Enhanced peak/trough detection
            for i in range(window, len(highs) - window):
                window_highs = highs.iloc[i-window:i+window+1]
                if highs.iloc[i] == window_highs.max():
                    resistance_levels.append(highs.iloc[i])
                
                window_lows = lows.iloc[i-window:i+window+1]
                if lows.iloc[i] == window_lows.min():
                    support_levels.append(lows.iloc[i])
            
            return {
                "support_levels": sorted(set(support_levels), reverse=True)[:3],
                "resistance_levels": sorted(set(resistance_levels))[:3]
            }
            
        except Exception:
            return {"support_levels": [], "resistance_levels": []}
    
    def detect_breakouts(self, data: pd.DataFrame, support_resistance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect breakout signals from support/resistance levels."""
        try:
            breakouts = []
            if data.empty or len(data) < 2:
                return breakouts
            
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            
            support_levels = support_resistance.get('support_levels', [])
            resistance_levels = support_resistance.get('resistance_levels', [])
            
            # Check for resistance breakouts
            for resistance in resistance_levels:
                if previous_price <= resistance and current_price > resistance:
                    breakouts.append({
                        "type": "resistance_breakout",
                        "level": resistance,
                        "direction": "bullish",
                        "strength": "strong" if current_price > resistance * 1.02 else "weak"
                    })
            
            # Check for support breakdowns
            for support in support_levels:
                if previous_price >= support and current_price < support:
                    breakouts.append({
                        "type": "support_breakdown",
                        "level": support,
                        "direction": "bearish",
                        "strength": "strong" if current_price < support * 0.98 else "weak"
                    })
            
            return breakouts
            
        except Exception:
            return []


class AnalysisEngine:
    """Main analysis engine with multi-timeframe capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical = TechnicalAnalyzer(config)
        self.pattern = PatternAnalyzer(config)
        self.timeframes = ['1h', '1d']
        
    def multi_timeframe_analysis(self, symbol: str, data_by_timeframe: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform multi-timeframe analysis."""
        try:
            analysis_results = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "timeframe_analysis": {},
                "consensus": {}
            }
            
            timeframe_signals = {}
            
            for timeframe, data in data_by_timeframe.items():
                if data.empty:
                    continue
                
                # Technical analysis for this timeframe
                technical_results = self.technical.calculate_indicators(data, timeframe)
                
                # Pattern analysis for this timeframe
                pattern_results = self.pattern.detect_chart_patterns(data)
                
                timeframe_analysis = {
                    "technical": technical_results,
                    "patterns": pattern_results,
                    "data_points": len(data)
                }
                
                analysis_results["timeframe_analysis"][timeframe] = timeframe_analysis
                
                # Extract signals for consensus
                tech_indicators = technical_results.get('indicators', {})
                trend_direction = tech_indicators.get('trend', {}).get('direction')
                
                if trend_direction:
                    timeframe_signals[timeframe] = trend_direction.value if hasattr(trend_direction, 'value') else str(trend_direction)
            
            # Calculate consensus across timeframes
            consensus = self.calculate_timeframe_consensus(timeframe_signals)
            analysis_results["consensus"] = consensus
            
            # Ensure all data is JSON serializable for Airflow XCom
            return ensure_json_serializable(analysis_results)
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return ensure_json_serializable({
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "timeframe_analysis": {},
                "consensus": {}
            })
    
    def calculate_timeframe_consensus(self, timeframe_signals: Dict[str, str]) -> Dict[str, Any]:
        """Calculate consensus across multiple timeframes."""
        try:
            if not timeframe_signals:
                return {"signal": "unknown", "strength": "weak", "agreement": 0.0}
            
            signal_counts = {}
            for signal in timeframe_signals.values():
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            # Find dominant signal
            dominant_signal = max(signal_counts, key=signal_counts.get)
            agreement_ratio = signal_counts[dominant_signal] / len(timeframe_signals)
            
            # Determine strength based on agreement
            if agreement_ratio >= 0.8:
                strength = "strong"
            elif agreement_ratio >= 0.6:
                strength = "moderate"
            else:
                strength = "weak"
            
            return {
                "signal": dominant_signal,
                "strength": strength,
                "agreement": round(agreement_ratio, 2),
                "timeframe_signals": timeframe_signals
            }
            
        except Exception:
            return {"signal": "unknown", "strength": "weak", "agreement": 0.0}


# Utility Functions
def validate_data_format(data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
    """Validate data format for analysis."""
    try:
        if isinstance(data, pd.DataFrame):
            return not data.empty and len(data.columns) > 0
        elif isinstance(data, dict):
            return len(data) > 0
        return False
    except Exception:
        return False


def ensure_json_serializable(obj: Any) -> Any:
    """Ensure all objects are JSON serializable by converting enums to strings."""
    if hasattr(obj, 'value'):  # Handle enum objects
        return obj.value
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj


def calculate_composite_score(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate weighted composite score."""
    try:
        if not scores:
            return 0.0
        
        if weights is None:
            weights = {key: 1.0 for key in scores.keys()}
        
        weighted_sum = sum(score * weights.get(key, 1.0) for key, score in scores.items())
        total_weight = sum(weights.get(key, 1.0) for key in scores.keys())
        
        return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0
        
    except Exception:
        return 0.0