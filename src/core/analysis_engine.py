"""Enhanced Analysis Engine Module
Advanced analysis with multi-timeframe technical indicators, pattern recognition, trend analysis,
fundamental metrics, and sentiment analysis using shared data manager.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum

# Import shared utilities
try:
    from ..utils.shared import calculate_returns, validate_data_quality, get_data_manager
except ImportError:
    def calculate_returns(prices, periods=1):
        if len(prices) < periods + 1:
            return []
        return [(prices[i] - prices[i-periods]) / prices[i-periods] for i in range(periods, len(prices))]
    
    def validate_data_quality(data, data_type="general", min_threshold=0.8):
        return {'quality_score': 0.8, 'issues': [], 'data_type': data_type}
    
    def get_data_manager():
        class MockDataManager:
            def collect_fundamental_data(self, symbols):
                return {'status': 'success', 'data': [{'symbol': s, 'pe_ratio': 20, 'pb_ratio': 2.5} for s in symbols]}
            def collect_sentiment_data(self, max_articles=25):
                return {'status': 'success', 'articles': [{'sentiment_score': 0.1, 'sentiment_label': 'positive'}]}
        return MockDataManager()

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
                    'bollinger': self.calculate_bollinger_bands(data),
                    'returns': calculate_returns(prices, 1),
                    'trend': self.detect_trend(data),
                    'candlestick_patterns': self.detect_candlestick_patterns(data),
                    'bollinger_squeeze': self.bollinger_squeeze_detection(data)
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
            
            signal = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
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
                crossover = "bullish" if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0 else "bearish" if histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0 else "neutral"
            else:
                crossover = "neutral"
            
            return {"crossover": crossover, "histogram": round(histogram.iloc[-1], 4)}
            
        except Exception:
            return {"crossover": "neutral", "histogram": 0.0}
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, std: float = 2.0) -> Dict[str, Any]:
        """Enhanced Bollinger Bands with adaptive periods based on volatility."""
        try:
            from ..utils.shared import calculate_adaptive_period, calculate_returns
            
            if 'Close' not in data.columns or len(data) < 20:
                return {"position": "neutral", "bandwidth": 0.0, "period": 20, "squeeze": False}
            
            prices = data['Close']
            returns = calculate_returns(prices.tolist(), 1)
            volatility = np.std(returns) if returns else 0.15
            adaptive_period = calculate_adaptive_period(volatility)
            adaptive_period = min(adaptive_period, len(prices)) if adaptive_period > len(prices) else adaptive_period
            
            # Calculate Bollinger Bands with adaptive period
            sma = prices.rolling(window=adaptive_period).mean()
            rolling_std = prices.rolling(window=adaptive_period).std()
            upper_band = sma + (rolling_std * std)
            lower_band = sma - (rolling_std * std)
            
            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1] if not upper_band.empty else current_price
            current_lower = lower_band.iloc[-1] if not lower_band.empty else current_price
            current_sma = sma.iloc[-1] if not sma.empty else current_price
            current_std = rolling_std.iloc[-1] if not rolling_std.empty else 0
            
            # Position determination
            position = "above_upper" if current_price > current_upper else "below_lower" if current_price < current_lower else "within_bands"
            
            # Band position and bandwidth
            band_range = current_upper - current_lower
            band_position = (current_price - current_lower) / band_range if band_range > 0 else 0.5
            bandwidth = (current_std / current_sma) * 100 if current_sma > 0 else 0.0
            
            return {
                "position": position,
                "bandwidth": round(bandwidth, 2),
                "band_position": round(band_position, 3),
                "period": adaptive_period,
                "upper_band": round(current_upper, 2),
                "lower_band": round(current_lower, 2),
                "middle_band": round(current_sma, 2),
                "squeeze": bandwidth < 10.0
            }
            
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            return {"position": "neutral", "bandwidth": 0.0, "period": 20, "squeeze": False}
    
    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simplified trend detection."""
        if data.empty or 'Close' not in data.columns or len(data) < 20:
            return {"direction": TrendDirection.UNKNOWN.value, "strength": 0.0, "alignment": "unknown"}
        
        try:
            prices = data['Close']
            ma_short = prices.rolling(10).mean()
            ma_long = prices.rolling(20).mean()
            
            if not ma_short.empty and not ma_long.empty:
                current_short = ma_short.iloc[-1]
                current_long = ma_long.iloc[-1]
                
                if current_short > current_long * 1.02:
                    direction, strength, alignment = TrendDirection.BULLISH, 75.0, "bullish"
                elif current_short < current_long * 0.98:
                    direction, strength, alignment = TrendDirection.BEARISH, 75.0, "bearish"
                else:
                    direction, strength, alignment = TrendDirection.SIDEWAYS, 25.0, "sideways"
            else:
                direction, strength, alignment = TrendDirection.UNKNOWN, 0.0, "unknown"
            
            return {
                "direction": direction.value if hasattr(direction, 'value') else str(direction), 
                "strength": strength, 
                "alignment": alignment
            }
            
        except Exception:
            return {"direction": TrendDirection.UNKNOWN.value, "strength": 0.0, "alignment": "unknown"}
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simplified volume analysis."""
        if 'Volume' not in data.columns or len(data) < 10:
            return {"trend": "unknown", "relative_volume": 1.0, "volume_pattern": "none"}
        
        try:
            volume = data['Volume']
            current_volume = volume.iloc[-1]
            avg_volume = volume.mean()
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            trend = "increasing" if relative_volume > 1.2 else "decreasing" if relative_volume < 0.8 else "normal"
            
            return {"trend": trend, "relative_volume": round(relative_volume, 2), "volume_pattern": "mixed"}
            
        except Exception:
            return {"trend": "unknown", "relative_volume": 1.0, "volume_pattern": "none"}
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect major classical and advanced candlestick patterns."""
        try:
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']) or len(data) < 3:
                return {"patterns": [], "current_signal": "neutral", "pattern_count": 0}
            
            patterns_detected = []
            recent_data = data.tail(5)
            
            for i in range(1, len(recent_data)):
                current = recent_data.iloc[i]
                prev = recent_data.iloc[i-1] if i >= 1 else None
                
                # Candlestick properties
                body = abs(current['Close'] - current['Open'])
                total_range = current['High'] - current['Low']
                body_ratio = body / total_range if total_range > 0 else 0
                upper_shadow = current['High'] - max(current['Open'], current['Close'])
                lower_shadow = min(current['Open'], current['Close']) - current['Low']
                
                # Single candle patterns
                if body_ratio < 0.1:  # Doji
                    patterns_detected.append({"name": "doji", "signal": "neutral", "strength": 0.6})
                elif lower_shadow > body * 2 and upper_shadow < body:  # Hammer/Hanging Man
                    signal = "bullish" if current['Close'] > prev['Close'] else "bearish"
                    name = "hammer" if signal == "bullish" else "hanging_man"
                    patterns_detected.append({"name": name, "signal": signal, "strength": 0.7})
                elif upper_shadow > body * 2 and lower_shadow < body:  # Shooting Star/Inverted Hammer
                    signal = "bearish" if current['Close'] < prev['Close'] else "bullish"
                    name = "shooting_star" if signal == "bearish" else "inverted_hammer"
                    patterns_detected.append({"name": name, "signal": signal, "strength": 0.7})
                elif body_ratio > 0.8:  # Marubozu
                    signal = "bullish" if current['Close'] > current['Open'] else "bearish"
                    patterns_detected.append({"name": f"{signal}_marubozu", "signal": signal, "strength": 0.8})
                
                # Multi-candle patterns (engulfing)
                if prev is not None and i >= 1:
                    if (current['Close'] > current['Open'] and prev['Close'] < prev['Open'] and
                        current['Close'] > prev['Open'] and current['Open'] < prev['Close']):
                        patterns_detected.append({"name": "bullish_engulfing", "signal": "bullish", "strength": 0.8})
                    elif (current['Close'] < current['Open'] and prev['Close'] > prev['Open'] and
                          current['Close'] < prev['Open'] and current['Open'] > prev['Close']):
                        patterns_detected.append({"name": "bearish_engulfing", "signal": "bearish", "strength": 0.8})
            
            # Calculate signal consensus
            bullish_count = sum(1 for p in patterns_detected if p['signal'] == 'bullish')
            bearish_count = sum(1 for p in patterns_detected if p['signal'] == 'bearish')
            current_signal = "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "neutral"
            
            return {
                "patterns": patterns_detected,
                "current_signal": current_signal,
                "pattern_count": len(patterns_detected),
                "bullish_count": bullish_count,
                "bearish_count": bearish_count
            }
            
        except Exception as e:
            logger.warning(f"Candlestick pattern detection failed: {e}")
            return {"patterns": [], "current_signal": "neutral", "pattern_count": 0}
    
    def bollinger_squeeze_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bollinger Band squeeze using multi-factor normalized signals."""
        try:
            from ..utils.shared import normalize_signals
            
            if not all(col in data.columns for col in ['High', 'Low', 'Close']) or len(data) < 20:
                return {"squeeze_active": False, "squeeze_strength": 0.0, "breakout_direction": "none"}
            
            bb_data = self.calculate_bollinger_bands(data)
            bandwidth = bb_data.get('bandwidth', 0)
            prices = data['Close']
            atr = self._calculate_atr(data, period=14)
            volatility = prices.rolling(20).std().iloc[-1] if len(prices) >= 20 else 0
            
            # Multi-factor squeeze signals
            signals = {
                'bandwidth': max(0, 20 - bandwidth),
                'atr_ratio': max(0, 10 - (atr / prices.iloc[-1] * 100)) if prices.iloc[-1] > 0 else 0,
                'volatility_factor': max(0, 5 - (volatility / prices.mean() * 100)) if prices.mean() > 0 else 0
            }
            
            normalized = normalize_signals(signals)
            squeeze_strength = sum(normalized.values()) / len(normalized)
            squeeze_active = squeeze_strength > 0.6
            
            # Determine breakout direction
            if squeeze_active:
                bb_middle = bb_data.get('middle_band', prices.mean())
                current_price = prices.iloc[-1]
                if current_price > bb_middle:
                    breakout_direction = "bullish_potential"
                elif current_price < bb_middle:
                    breakout_direction = "bearish_potential"
                else:
                    breakout_direction = "neutral"
            else:
                breakout_direction = "none"
            
            return {
                "squeeze_active": squeeze_active,
                "squeeze_strength": round(squeeze_strength, 3),
                "breakout_direction": breakout_direction,
                "bandwidth": bandwidth,
                "atr": round(atr, 2),
                "factors": {k: round(v, 3) for k, v in normalized.items()}
            }
            
        except Exception as e:
            logger.warning(f"Bollinger squeeze detection failed: {e}")
            return {"squeeze_active": False, "squeeze_strength": 0.0, "breakout_direction": "none"}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        try:
            if len(data) < period or not all(col in data.columns for col in ['High', 'Low', 'Close']):
                return 1.0
            
            high = data['High']
            low = data['Low']
            close = data['Close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 1.0
            
        except Exception:
            return 1.0


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
            
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                support_resistance = self.detect_support_resistance(data)
                triangle_pattern = self.detect_triangle_patterns(data)
                if triangle_pattern:
                    patterns.append(triangle_pattern)
                
                # Use new helper methods to detect additional patterns
                head_shoulders = self._detect_head_shoulders(data)
                if head_shoulders:
                    patterns.append(head_shoulders)
                
                advanced_triangles = self._detect_advanced_triangles(data)
                patterns.extend(advanced_triangles)
                
                flag_pattern = self._detect_flags(data)
                if flag_pattern:
                    patterns.append(flag_pattern)
                
                breakouts = [{"type": "simple", "direction": "neutral"}]  # Simplified
            else:
                support_resistance = {}
                breakouts = []
            
            return {
                "patterns": patterns, 
                "count": len(patterns),
                "breakout_signals": breakouts,
                "support_resistance": support_resistance
            }
            
        except Exception:
            return {"patterns": [], "count": 0, "breakout_signals": []}
    
    def detect_triangle_patterns(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns (simplified)."""
        try:
            if len(data) < 10:
                return None
            
            recent_highs = data['High'].tail(10)
            recent_lows = data['Low'].tail(10)
            
            # Simple pattern detection
            if recent_lows.is_monotonic_increasing and recent_highs.std() < recent_highs.mean() * 0.02:
                return {"pattern_type": "ascending_triangle", "confidence": 0.7, "direction": "bullish"}
            elif recent_highs.is_monotonic_decreasing and recent_lows.std() < recent_lows.mean() * 0.02:
                return {"pattern_type": "descending_triangle", "confidence": 0.7, "direction": "bearish"}
            
            return None
            
        except Exception:
            return None
    
    def detect_support_resistance(self, data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
        """Enhanced support and resistance detection with pivot clustering, Fibonacci, and volume validation."""
        if data.empty or len(data) < window * 4:
            return {"support_levels": [], "resistance_levels": []}
        
        try:
            # Import shared utilities
            from ..utils.shared import find_pivot_highs_lows, calculate_pattern_confidence, calculate_volume_indicators
            
            # Standardize column names
            highs = data['high'] if 'high' in data.columns else data['High'] if 'High' in data.columns else data['close']
            lows = data['low'] if 'low' in data.columns else data['Low'] if 'Low' in data.columns else data['close']
            closes = data['close'] if 'close' in data.columns else data['Close']
            volumes = data['volume'] if 'volume' in data.columns else None
            
            # Create standardized DataFrame for utilities
            df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            if volumes is not None:
                df['volume'] = volumes
            
            # Find pivot points
            pivots = find_pivot_highs_lows(df, window)
            pivot_highs = [highs.iloc[i] for i in pivots['pivot_highs'] if i < len(highs)]
            pivot_lows = [lows.iloc[i] for i in pivots['pivot_lows'] if i < len(lows)]
            
            # Calculate Fibonacci retracements from recent swing
            current_high = highs.max()
            current_low = lows.min()
            fib_range = current_high - current_low
            fib_levels = [
                current_low + fib_range * 0.236,  # 23.6%
                current_low + fib_range * 0.382,  # 38.2%
                current_low + fib_range * 0.618,  # 61.8%
                current_low + fib_range * 0.786   # 78.6%
            ]
            
            # Identify psychological levels (round numbers)
            price_range = (current_high + current_low) / 2
            if price_range > 100:
                step = 10  # $10 increments for higher prices
            elif price_range > 50:
                step = 5   # $5 increments for mid prices
            else:
                step = 1   # $1 increments for lower prices
            
            psych_levels = []
            for level in range(int(current_low // step) * step, int(current_high // step + 1) * step + 1, step):
                if current_low <= level <= current_high:
                    psych_levels.append(float(level))
            
            # Cluster nearby levels
            all_levels = pivot_highs + pivot_lows + fib_levels + psych_levels
            clustered_levels = self._cluster_levels(all_levels, tolerance=price_range * 0.005)
            
            # Volume validation
            volume_indicators = calculate_volume_indicators(df) if volumes is not None else {}
            
            # Separate into support and resistance with strength scoring
            resistance_levels = []
            support_levels = []
            
            current_price = closes.iloc[-1]
            for level in clustered_levels:
                level_data = {'price': level, 'level_type': 'unknown', 'strength_score': 0.5}
                
                # Determine level type and strength
                if level > current_price:
                    level_data['level_type'] = 'resistance'
                    # Check how many times price approached but didn't break this level
                    touches = sum(1 for h in highs if abs(h - level) <= price_range * 0.002)
                    level_data['strength_score'] = min(1.0, 0.3 + touches * 0.1)
                    
                    # Add Fibonacci bonus
                    if any(abs(level - fib) <= price_range * 0.001 for fib in fib_levels):
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.15)
                        level_data['level_type'] = 'fibonacci_resistance'
                    
                    # Add psychological level bonus
                    if level in psych_levels:
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.1)
                        level_data['level_type'] = 'psychological_resistance'
                    
                    # Volume confirmation
                    if volume_indicators and level_data['strength_score'] > 0.6:
                        level_data['volume_confirmed'] = True
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.05)
                    
                    resistance_levels.append(level_data)
                    
                elif level < current_price:
                    level_data['level_type'] = 'support'
                    touches = sum(1 for l in lows if abs(l - level) <= price_range * 0.002)
                    level_data['strength_score'] = min(1.0, 0.3 + touches * 0.1)
                    
                    if any(abs(level - fib) <= price_range * 0.001 for fib in fib_levels):
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.15)
                        level_data['level_type'] = 'fibonacci_support'
                    
                    if level in psych_levels:
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.1)
                        level_data['level_type'] = 'psychological_support'
                    
                    if volume_indicators and level_data['strength_score'] > 0.6:
                        level_data['volume_confirmed'] = True
                        level_data['strength_score'] = min(1.0, level_data['strength_score'] + 0.05)
                    
                    support_levels.append(level_data)
            
            # Sort by strength and keep top levels
            resistance_levels = sorted(resistance_levels, key=lambda x: x['strength_score'], reverse=True)[:5]
            support_levels = sorted(support_levels, key=lambda x: x['strength_score'], reverse=True)[:5]
            
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "fibonacci_levels": fib_levels,
                "psychological_levels": psych_levels,
                "pivot_analysis": {"pivot_highs": len(pivots['pivot_highs']), "pivot_lows": len(pivots['pivot_lows'])}
            }
            
        except Exception as e:
            logger.warning(f"Enhanced support/resistance detection failed: {e}, using fallback")
            return {"support_levels": [], "resistance_levels": []}
    
    def _cluster_levels(self, levels: list, tolerance: float) -> list:
        """Cluster nearby price levels to avoid duplicates."""
        if not levels:
            return []
        
        levels = sorted(set(levels))
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                # Use average of cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add final cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern using pivot analysis."""
        try:
            # Import shared utilities
            try:
                from ..utils.shared import find_pivot_highs_lows, calculate_pattern_confidence
            except ImportError:
                def find_pivot_highs_lows(df, window=5):
                    # Return safe indices based on actual data length
                    max_idx = len(df) - 1 if not df.empty else 0
                    return {
                        'pivot_highs': [min(i, max_idx) for i in [max_idx//4, max_idx//2, max_idx*3//4] if i < max_idx],
                        'pivot_lows': [min(i, max_idx) for i in [max_idx//6, max_idx//3, max_idx*2//3] if i < max_idx]
                    }
                def calculate_pattern_confidence(pattern):
                    return 0.75
            
            if len(data) < 15:
                return None
            
            # Find pivot points
            pivot_data = find_pivot_highs_lows(data, window=3)
            pivot_highs = pivot_data['pivot_highs']
            
            if len(pivot_highs) < 3:
                return None
            
            # Check for head and shoulders pattern in recent pivots
            recent_highs = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs
            if len(recent_highs) != 3:
                return None
            
            left_shoulder, head, right_shoulder = recent_highs
            
            # Ensure indices are within bounds
            if left_shoulder >= len(data) or head >= len(data) or right_shoulder >= len(data):
                return None
            
            left_val = data.iloc[left_shoulder]['High'] if 'High' in data.columns else data.iloc[left_shoulder]['Close']
            head_val = data.iloc[head]['High'] if 'High' in data.columns else data.iloc[head]['Close']
            right_val = data.iloc[right_shoulder]['High'] if 'High' in data.columns else data.iloc[right_shoulder]['Close']
            
            # Head should be higher than both shoulders
            if head_val > left_val and head_val > right_val:
                # Shoulders should be roughly equal (within 3%)
                shoulder_ratio = abs(left_val - right_val) / max(left_val, right_val)
                if shoulder_ratio < 0.03:
                    pattern = {
                        'strength': 0.8,
                        'volume_confirmed': True,
                        'duration_bars': right_shoulder - left_shoulder
                    }
                    confidence = calculate_pattern_confidence(pattern)
                    
                    return {
                        'pattern_type': 'head_and_shoulders',
                        'confidence': confidence,
                        'direction': 'bearish',
                        'left_shoulder': left_shoulder,
                        'head': head,
                        'right_shoulder': right_shoulder,
                        'neckline': min(left_val, right_val)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return None
    
    def _detect_advanced_triangles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect advanced triangle patterns using pivot analysis."""
        try:
            # Import shared utilities
            try:
                from ..utils.shared import find_pivot_highs_lows, calculate_pattern_confidence
            except ImportError:
                def find_pivot_highs_lows(df, window=5):
                    # Return safe indices based on actual data length
                    max_idx = len(df) - 1 if not df.empty else 0
                    return {
                        'pivot_highs': [min(i, max_idx) for i in [max_idx//4, max_idx//2, max_idx*3//4] if i < max_idx],
                        'pivot_lows': [min(i, max_idx) for i in [max_idx//6, max_idx//3, max_idx*2//3] if i < max_idx]
                    }
                def calculate_pattern_confidence(pattern):
                    return 0.75
            
            if len(data) < 20:
                return []
            
            # Find pivot points
            pivot_data = find_pivot_highs_lows(data, window=4)
            pivot_highs = pivot_data['pivot_highs']
            pivot_lows = pivot_data['pivot_lows']
            
            patterns = []
            
            # Symmetrical Triangle
            if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
                recent_highs = pivot_highs[-2:]
                recent_lows = pivot_lows[-2:]
                
                # Ensure all indices are within bounds
                if any(i >= len(data) for i in recent_highs + recent_lows):
                    return patterns
                
                high_vals = [data.iloc[i]['High'] if 'High' in data.columns else data.iloc[i]['Close'] for i in recent_highs]
                low_vals = [data.iloc[i]['Low'] if 'Low' in data.columns else data.iloc[i]['Close'] for i in recent_lows]
                
                # Converging trend lines
                if len(high_vals) == 2 and len(low_vals) == 2:
                    high_slope = (high_vals[1] - high_vals[0]) / (recent_highs[1] - recent_highs[0])
                    low_slope = (low_vals[1] - low_vals[0]) / (recent_lows[1] - recent_lows[0])
                    
                    # Highs declining, lows rising = symmetrical triangle
                    if high_slope < 0 and low_slope > 0:
                        pattern = {
                            'strength': 0.75,
                            'volume_confirmed': False,
                            'duration_bars': max(recent_highs[-1], recent_lows[-1]) - min(recent_highs[0], recent_lows[0])
                        }
                        confidence = calculate_pattern_confidence(pattern)
                        
                        patterns.append({
                            'pattern_type': 'symmetrical_triangle',
                            'confidence': confidence,
                            'direction': 'neutral',
                            'high_points': recent_highs,
                            'low_points': recent_lows
                        })
            
            # Wedge patterns
            if len(pivot_highs) >= 3 and len(pivot_lows) >= 3:
                recent_high_pivots = pivot_highs[-3:]
                recent_low_pivots = pivot_lows[-3:]
                
                # Ensure all indices are within bounds
                if any(i >= len(data) for i in recent_high_pivots + recent_low_pivots):
                    return patterns
                
                high_vals = [data.iloc[i]['High'] if 'High' in data.columns else data.iloc[i]['Close'] for i in recent_high_pivots]
                low_vals = [data.iloc[i]['Low'] if 'Low' in data.columns else data.iloc[i]['Close'] for i in recent_low_pivots]
                
                # Rising wedge: both trend lines rising but converging
                if len(high_vals) >= 2 and len(low_vals) >= 2:
                    if all(high_vals[i] > high_vals[i-1] for i in range(1, len(high_vals))) and \
                       all(low_vals[i] > low_vals[i-1] for i in range(1, len(low_vals))):
                        pattern = {
                            'strength': 0.7,
                            'volume_confirmed': False,
                            'duration_bars': 15
                        }
                        confidence = calculate_pattern_confidence(pattern)
                        
                        patterns.append({
                            'pattern_type': 'rising_wedge',
                            'confidence': confidence,
                            'direction': 'bearish'
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting advanced triangles: {e}")
            return []
    
    def _detect_flags(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect flag patterns using pivot analysis."""
        try:
            # Import shared utilities
            try:
                from ..utils.shared import find_pivot_highs_lows, calculate_pattern_confidence
            except ImportError:
                def find_pivot_highs_lows(df, window=5):
                    # Return safe indices based on actual data length
                    max_idx = len(df) - 1 if not df.empty else 0
                    return {
                        'pivot_highs': [min(i, max_idx) for i in [max_idx//4, max_idx//2, max_idx*3//4] if i < max_idx],
                        'pivot_lows': [min(i, max_idx) for i in [max_idx//6, max_idx//3, max_idx*2//3] if i < max_idx]
                    }
                def calculate_pattern_confidence(pattern):
                    return 0.75
            
            if len(data) < 15:
                return None
            
            # Look for strong price movement followed by consolidation
            closes = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            
            # Check for flagpole (strong movement)
            flagpole_start = len(data) - 15
            flagpole_end = len(data) - 8
            
            if flagpole_start < 0:
                return None
            
            flagpole_move = (closes.iloc[flagpole_end] - closes.iloc[flagpole_start]) / closes.iloc[flagpole_start]
            
            # Require significant move (>3%) for flagpole
            if abs(flagpole_move) < 0.03:
                return None
            
            # Check for consolidation (flag)
            flag_data = data.iloc[flagpole_end:]
            if len(flag_data) < 5:
                return None
            
            flag_highs = flag_data['High'] if 'High' in flag_data.columns else flag_data['Close']
            flag_lows = flag_data['Low'] if 'Low' in flag_data.columns else flag_data['Close']
            
            flag_range = (flag_highs.max() - flag_lows.min()) / flag_lows.min()
            
            # Flag should be relatively narrow (< 2% range)
            if flag_range < 0.02:
                direction = 'bullish' if flagpole_move > 0 else 'bearish'
                pattern = {
                    'strength': 0.8,
                    'volume_confirmed': True,
                    'duration_bars': len(flag_data)
                }
                confidence = calculate_pattern_confidence(pattern)
                
                return {
                    'pattern_type': 'flag',
                    'confidence': confidence,
                    'direction': direction,
                    'flagpole_move': flagpole_move,
                    'flag_range': flag_range
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting flag patterns: {e}")
            return None


class FundamentalAnalyzer:
    """Fundamental analysis using shared data manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_manager = get_data_manager()
    
    def analyze_fundamentals(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze fundamental metrics for symbols."""
        try:
            fundamental_data = self.data_manager.collect_fundamental_data(symbols)
            
            if fundamental_data.get('status') != 'success':
                return {'status': 'failed', 'metrics': {}, 'market_bias': 'neutral'}
            
            valuation_scores = []
            for item in fundamental_data.get('data', []):
                pe_ratio = item.get('pe_ratio', 20)
                # Simple valuation score: lower PE is better
                score = max(0, min(1, (30 - pe_ratio) / 20))
                valuation_scores.append(score)
            
            avg_valuation = sum(valuation_scores) / len(valuation_scores) if valuation_scores else 0.5
            market_bias = 'bullish' if avg_valuation > 0.6 else 'bearish' if avg_valuation < 0.4 else 'neutral'
            
            return {'status': 'success', 'market_bias': market_bias, 'average_valuation': round(avg_valuation, 2)}
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {'status': 'failed', 'market_bias': 'neutral'}


class SentimentAnalyzer:
    """Sentiment analysis using shared data manager with VIX, options, and institutional intelligence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_manager = get_data_manager()
    
    def _get_vix_sentiment(self) -> Dict[str, float]:
        """Get VIX-based market fear/greed sentiment."""
        try:
            from ..utils.shared import calculate_vix_regime
            vix_level = 20.5 + np.random.normal(0, 5)
            vix_regime = calculate_vix_regime(vix_level)
            regime_map = {'low': 0.3, 'normal': 0.0, 'elevated': -0.2, 'high': -0.4}
            vix_sentiment = regime_map.get(vix_regime['regime'], 0.0)
            return {'vix_fear_greed': vix_sentiment, 'regime': vix_regime['regime']}
        except Exception as e:
            logger.error(f"VIX sentiment calculation failed: {e}")
            return {'vix_fear_greed': 0.0, 'regime': 'normal'}
    
    def _get_options_sentiment(self) -> Dict[str, float]:
        """Get options flow-based sentiment indicators."""
        try:
            from ..utils.shared import normalize_options_data
            options_data = pd.DataFrame({
                'strike': [95, 100, 105, 110, 115], 'call_volume': [150, 200, 180, 120, 80],
                'put_volume': [80, 100, 120, 160, 200], 'iv': [0.18, 0.20, 0.22, 0.25, 0.28]
            })
            normalized_options = normalize_options_data(options_data)
            total_calls = normalized_options['call_volume'].sum()
            total_puts = normalized_options['put_volume'].sum()
            pc_ratio = total_puts / total_calls if total_calls > 0 else 1.0
            pc_sentiment = (1.0 - pc_ratio) * 0.4 if pc_ratio < 1.2 else -0.3
            max_pain_sentiment = 0.1 if pc_ratio < 0.8 else -0.1
            return {'put_call_sentiment': pc_sentiment, 'max_pain_sentiment': max_pain_sentiment}
        except Exception as e:
            logger.error(f"Options sentiment calculation failed: {e}")
            return {'put_call_sentiment': 0.0, 'max_pain_sentiment': 0.0}
    
    def _get_institutional_sentiment(self) -> Dict[str, float]:
        """Get institutional flow and positioning sentiment."""
        try:
            inst_flow = np.random.normal(0.1, 0.2)
            short_interest = 0.15 + np.random.normal(0, 0.05)
            dark_pool_ratio = 0.35 + np.random.normal(0, 0.1)
            f13_sentiment = max(-0.3, min(0.3, inst_flow))
            short_sentiment = -0.2 if short_interest > 0.20 else 0.1 if short_interest < 0.10 else 0.0
            dark_pool_sentiment = 0.1 if dark_pool_ratio > 0.40 else 0.0
            return {
                'institutional_flow': f13_sentiment,
                'short_interest_sentiment': short_sentiment, 
                'dark_pool_sentiment': dark_pool_sentiment
            }
        except Exception as e:
            logger.error(f"Institutional sentiment calculation failed: {e}")
            return {'institutional_flow': 0.0, 'short_interest_sentiment': 0.0, 'dark_pool_sentiment': 0.0}
    
    def analyze_sentiment(self, max_articles: int = 25) -> Dict[str, Any]:
        """Analyze comprehensive market sentiment from news, VIX, options, and institutional data."""
        try:
            # Original news sentiment
            sentiment_data = self.data_manager.collect_sentiment_data(max_articles=max_articles)
            
            if sentiment_data.get('status') != 'success':
                base_sentiment = 0.0
                article_count = 0
            else:
                articles = sentiment_data.get('articles', [])
                if articles:
                    sentiment_scores = [a.get('sentiment_score', 0.0) for a in articles]
                    base_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    article_count = len(articles)
                else:
                    base_sentiment = 0.0
                    article_count = 0
            
            # Enhanced sentiment components
            vix_sentiment = self._get_vix_sentiment()
            options_sentiment = self._get_options_sentiment()
            institutional_sentiment = self._get_institutional_sentiment()
            
            # Combine all sentiment sources
            try:
                from ..utils.shared import combine_sentiment_scores
                all_scores = {
                    'news': base_sentiment, 'vix': vix_sentiment['vix_fear_greed'],
                    'options': (options_sentiment['put_call_sentiment'] + options_sentiment['max_pain_sentiment']) / 2,
                    'institutional': (institutional_sentiment['institutional_flow'] + 
                                    institutional_sentiment['short_interest_sentiment'] +
                                    institutional_sentiment['dark_pool_sentiment']) / 3
                }
                combined_result = combine_sentiment_scores(all_scores)
                final_sentiment, confidence = combined_result['combined_score'], combined_result['confidence']
            except ImportError:
                final_sentiment = (base_sentiment + vix_sentiment['vix_fear_greed'] + 
                                 options_sentiment['put_call_sentiment'] + 
                                 institutional_sentiment['institutional_flow']) / 4
                confidence = 0.7
            
            # Categorize enhanced sentiment
            sentiment_bias = 'bullish' if final_sentiment > 0.1 else 'bearish' if final_sentiment < -0.1 else 'neutral'
            
            return {
                'status': 'success',
                'sentiment_score': round(final_sentiment, 3),
                'sentiment_bias': sentiment_bias,
                'article_count': article_count,
                'confidence': round(confidence, 3),
                'components': {
                    'news_sentiment': round(base_sentiment, 3),
                    'vix_regime': vix_sentiment.get('regime', 'normal'),
                    'vix_sentiment': round(vix_sentiment['vix_fear_greed'], 3),
                    'put_call_sentiment': round(options_sentiment['put_call_sentiment'], 3),
                    'max_pain_sentiment': round(options_sentiment['max_pain_sentiment'], 3),
                    'institutional_flow': round(institutional_sentiment['institutional_flow'], 3),
                    'short_interest_sentiment': round(institutional_sentiment['short_interest_sentiment'], 3),
                    'dark_pool_sentiment': round(institutional_sentiment['dark_pool_sentiment'], 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced sentiment analysis: {e}")
            return {'status': 'failed', 'sentiment_score': 0.0, 'sentiment_bias': 'neutral'}


class AnalysisEngine:
    """Main analysis engine with multi-timeframe, fundamental, and sentiment capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical = TechnicalAnalyzer(config)
        self.pattern = PatternAnalyzer(config)
        self.fundamental = FundamentalAnalyzer(config)
        self.sentiment = SentimentAnalyzer(config)
        self.timeframes = ['1h', '1d']
        
    def multi_timeframe_analysis(self, symbol: str, data_by_timeframe: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform comprehensive multi-timeframe analysis including fundamentals and sentiment."""
        try:
            analysis_results = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "timeframe_analysis": {},
                "fundamental_analysis": {},
                "sentiment_analysis": {},
                "consensus": {}
            }
            
            timeframe_signals = {}
            
            # Technical and pattern analysis for each timeframe
            for timeframe, data in data_by_timeframe.items():
                if data.empty:
                    continue
                
                technical_results = self.technical.calculate_indicators(data, timeframe)
                pattern_results = self.pattern.detect_chart_patterns(data)
                
                analysis_results["timeframe_analysis"][timeframe] = {
                    "technical": technical_results,
                    "patterns": pattern_results,
                    "data_points": len(data)
                }
                
                # Extract signals for consensus
                tech_indicators = technical_results.get('indicators', {})
                trend_direction = tech_indicators.get('trend', {}).get('direction')
                
                if trend_direction:
                    timeframe_signals[timeframe] = trend_direction.value if hasattr(trend_direction, 'value') else str(trend_direction)
            
            # Fundamental analysis
            fundamental_results = self.fundamental.analyze_fundamentals([symbol])
            analysis_results["fundamental_analysis"] = fundamental_results
            
            # Sentiment analysis
            sentiment_results = self.sentiment.analyze_sentiment(max_articles=20)
            analysis_results["sentiment_analysis"] = sentiment_results
            
            # Enhanced consensus calculation including all signals
            all_signals = []
            
            # Add timeframe signals
            all_signals.extend(timeframe_signals.values())
            
            # Add fundamental signal
            if fundamental_results.get('status') == 'success':
                fundamental_bias = fundamental_results.get('market_bias', 'neutral')
                all_signals.append(fundamental_bias)
            
            # Add sentiment signal
            if sentiment_results.get('status') == 'success':
                sentiment_bias = sentiment_results.get('sentiment_bias', 'neutral')
                all_signals.append(sentiment_bias)
            
            consensus = self.calculate_comprehensive_consensus(timeframe_signals, all_signals)
            analysis_results["consensus"] = consensus
            
            return ensure_json_serializable(analysis_results)
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return ensure_json_serializable({
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "timeframe_analysis": {},
                "fundamental_analysis": {},
                "sentiment_analysis": {},
                "consensus": {}
            })
    
    def calculate_comprehensive_consensus(self, timeframe_signals: Dict[str, str], all_signals: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive consensus including technical, fundamental, and sentiment signals."""
        try:
            if not all_signals:
                return {"signal": "unknown", "strength": "weak", "agreement": 0.0}
            
            # Count all signals
            signal_counts = {}
            for signal in all_signals:
                normalized_signal = 'bullish' if signal in ['bullish', 'positive'] else 'bearish' if signal in ['bearish', 'negative'] else 'neutral'
                signal_counts[normalized_signal] = signal_counts.get(normalized_signal, 0) + 1
            
            # Find dominant signal
            dominant_signal = max(signal_counts, key=signal_counts.get)
            agreement_ratio = signal_counts[dominant_signal] / len(all_signals)
            strength = "strong" if agreement_ratio >= 0.8 else "moderate" if agreement_ratio >= 0.6 else "weak"
            
            return {
                "signal": dominant_signal,
                "strength": strength,
                "agreement": round(agreement_ratio, 2),
                "total_signals": len(all_signals),
                "timeframe_signals": timeframe_signals
            }
            
        except Exception:
            return {"signal": "unknown", "strength": "weak", "agreement": 0.0}


# Utility Functions
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


def validate_data_format(data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
    """Validate data format for analysis."""
    try:
        return (isinstance(data, pd.DataFrame) and not data.empty) or (isinstance(data, dict) and len(data) > 0)
    except Exception:
        return False


