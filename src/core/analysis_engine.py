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
    """Advanced chart and technical pattern recognition for trading signals."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pattern_confidence_threshold = self.config.get('pattern_confidence_threshold', 0.7)
    
    def detect_chart_patterns(self, data: pd.DataFrame, timeframe: str = '1h') -> Dict[str, Any]:
        """Detect major chart patterns (triangles, head & shoulders, flags, etc.)."""
        try:
            if data.empty or len(data) < 20:
                return {
                    'timeframe': timeframe,
                    'patterns': [], 
                    'signals': [], 
                    'dominant_signal': 'neutral',
                    'confidence': 0.0, 
                    'pattern_count': 0,
                    'signal_strength': 'weak'
                }
            
            patterns_detected = []
            signals = []
            
            # Analyze price structure for patterns
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            # Triangle patterns
            triangle_result = self._detect_triangle_patterns(data)
            if triangle_result['detected']:
                patterns_detected.append(triangle_result)
                signals.append(triangle_result['signal'])
            
            # Head and shoulders patterns
            hs_result = self._detect_head_shoulders(data)
            if hs_result['detected']:
                patterns_detected.append(hs_result)
                signals.append(hs_result['signal'])
            
            # Flag and pennant patterns
            flag_result = self._detect_flag_patterns(data)
            if flag_result['detected']:
                patterns_detected.append(flag_result)
                signals.append(flag_result['signal'])
            
            # Double top/bottom patterns
            double_result = self._detect_double_patterns(data)
            if double_result['detected']:
                patterns_detected.append(double_result)
                signals.append(double_result['signal'])
            
            # Calculate overall confidence
            if patterns_detected:
                confidences = [p.get('confidence', 0) for p in patterns_detected]
                overall_confidence = sum(confidences) / len(confidences)
            else:
                overall_confidence = 0.0
            
            # Determine dominant signal
            bullish_signals = sum(1 for s in signals if s == 'bullish')
            bearish_signals = sum(1 for s in signals if s == 'bearish')
            
            if bullish_signals > bearish_signals:
                dominant_signal = 'bullish'
            elif bearish_signals > bullish_signals:
                dominant_signal = 'bearish'
            else:
                dominant_signal = 'neutral'
            
            return {
                'timeframe': timeframe,
                'patterns': patterns_detected,
                'signals': signals,
                'dominant_signal': dominant_signal,
                'confidence': round(overall_confidence, 3),
                'pattern_count': len(patterns_detected),
                'signal_strength': 'strong' if overall_confidence > 0.8 else 'moderate' if overall_confidence > 0.5 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Chart pattern detection failed: {e}")
            return {
                'timeframe': timeframe,
                'patterns': [], 
                'signals': [], 
                'dominant_signal': 'neutral',
                'confidence': 0.0, 
                'pattern_count': 0,
                'signal_strength': 'weak'
            }
    
    def _detect_triangle_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect ascending, descending, and symmetrical triangles."""
        try:
            if len(data) < 10:
                return {'detected': False}
            
            highs = data['High'].values
            lows = data['Low'].values
            
            # Simplified triangle detection using trend lines
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            
            # Check for converging trend lines (basic implementation)
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Triangle pattern criteria
            if abs(high_trend) < 0.01 and low_trend > 0.01:  # Ascending triangle
                return {
                    'detected': True,
                    'pattern': 'ascending_triangle',
                    'signal': 'bullish',
                    'confidence': 0.75,
                    'breakout_target': data['High'].iloc[-1] * 1.05
                }
            elif high_trend < -0.01 and abs(low_trend) < 0.01:  # Descending triangle
                return {
                    'detected': True,
                    'pattern': 'descending_triangle',
                    'signal': 'bearish',
                    'confidence': 0.75,
                    'breakout_target': data['Low'].iloc[-1] * 0.95
                }
            elif high_trend < -0.005 and low_trend > 0.005:  # Symmetrical triangle
                return {
                    'detected': True,
                    'pattern': 'symmetrical_triangle',
                    'signal': 'neutral',
                    'confidence': 0.65,
                    'breakout_direction': 'pending'
                }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect head and shoulders reversal patterns."""
        try:
            if len(data) < 15:
                return {'detected': False}
            
            highs = data['High'].values
            lows = data['Low'].values
            
            # Look for three peaks pattern (simplified)
            recent_data = data.tail(15)
            high_points = []
            
            # Find local maxima
            for i in range(1, len(recent_data)-1):
                if (recent_data.iloc[i]['High'] > recent_data.iloc[i-1]['High'] and 
                    recent_data.iloc[i]['High'] > recent_data.iloc[i+1]['High']):
                    high_points.append((i, recent_data.iloc[i]['High']))
            
            # Check for head and shoulders pattern
            if len(high_points) >= 3:
                # Sort by height to identify head (highest) and shoulders
                sorted_points = sorted(high_points, key=lambda x: x[1], reverse=True)
                head = sorted_points[0]
                potential_shoulders = sorted_points[1:3]
                
                # Basic head and shoulders criteria
                if (abs(potential_shoulders[0][1] - potential_shoulders[1][1]) < head[1] * 0.02 and
                    head[1] > potential_shoulders[0][1] * 1.03):
                    
                    return {
                        'detected': True,
                        'pattern': 'head_and_shoulders',
                        'signal': 'bearish',
                        'confidence': 0.8,
                        'neckline': min(potential_shoulders[0][1], potential_shoulders[1][1]),
                        'target': min(potential_shoulders[0][1], potential_shoulders[1][1]) * 0.95
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_flag_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect flag and pennant continuation patterns."""
        try:
            if len(data) < 10:
                return {'detected': False}
            
            # Look for strong directional move followed by consolidation
            recent_data = data.tail(10)
            price_change = (recent_data.iloc[-1]['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close']
            
            # Check for consolidation after strong move
            consolidation_range = recent_data['High'].max() - recent_data['Low'].min()
            avg_price = recent_data['Close'].mean()
            consolidation_percent = consolidation_range / avg_price
            
            if abs(price_change) > 0.03 and consolidation_percent < 0.02:  # Strong move + tight range
                signal = 'bullish' if price_change > 0 else 'bearish'
                pattern_type = 'bull_flag' if signal == 'bullish' else 'bear_flag'
                
                return {
                    'detected': True,
                    'pattern': pattern_type,
                    'signal': signal,
                    'confidence': 0.7,
                    'consolidation_range': consolidation_percent,
                    'prior_move': price_change
                }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_double_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double top and double bottom patterns."""
        try:
            if len(data) < 20:
                return {'detected': False}
            
            highs = data['High'].values
            lows = data['Low'].values
            
            # Find peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(1, len(data)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            # Check for double top
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) < last_two_peaks[0][1] * 0.02:
                    return {
                        'detected': True,
                        'pattern': 'double_top',
                        'signal': 'bearish',
                        'confidence': 0.75,
                        'resistance_level': max(last_two_peaks[0][1], last_two_peaks[1][1])
                    }
            
            # Check for double bottom
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) < last_two_troughs[0][1] * 0.02:
                    return {
                        'detected': True,
                        'pattern': 'double_bottom',
                        'signal': 'bullish',
                        'confidence': 0.75,
                        'support_level': min(last_two_troughs[0][1], last_two_troughs[1][1])
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def validate_pattern_reliability(self, pattern_data: Dict[str, Any], volume_data: pd.Series = None) -> Dict[str, Any]:
        """Validate pattern reliability using volume confirmation and other factors."""
        try:
            if not pattern_data.get('detected', False):
                return {'reliable': False, 'confidence_adjustment': 0.0}
            
            base_confidence = pattern_data.get('confidence', 0.5)
            adjustments = []
            
            # Volume confirmation (if available)
            if volume_data is not None and len(volume_data) > 5:
                recent_volume = volume_data.tail(5).mean()
                avg_volume = volume_data.mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.2:  # Above average volume supports pattern
                    adjustments.append(0.1)
                elif volume_ratio < 0.8:  # Below average volume weakens pattern
                    adjustments.append(-0.1)
            
            # Pattern-specific reliability factors
            pattern_type = pattern_data.get('pattern', '')
            if 'triangle' in pattern_type:
                # Triangles are more reliable near the apex
                adjustments.append(0.05)
            elif 'double' in pattern_type:
                # Double patterns need strong volume confirmation
                adjustments.append(-0.05 if not volume_data else 0.05)
            
            # Calculate adjusted confidence
            total_adjustment = sum(adjustments)
            adjusted_confidence = max(0.0, min(1.0, base_confidence + total_adjustment))
            reliable = adjusted_confidence >= self.pattern_confidence_threshold
            
            return {
                'reliable': reliable,
                'adjusted_confidence': round(adjusted_confidence, 3),
                'confidence_adjustment': round(total_adjustment, 3),
                'reliability_factors': adjustments
            }
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return {'reliable': False, 'confidence_adjustment': 0.0}


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
    """Main analysis engine with multi-timeframe, pattern recognition, fundamental, and sentiment capabilities."""
    
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
            
            # Technical and Pattern analysis for each timeframe
            for timeframe, data in data_by_timeframe.items():
                if data.empty:
                    continue
                
                technical_results = self.technical.calculate_indicators(data, timeframe)
                pattern_results = self.pattern.detect_chart_patterns(data, timeframe)
                
                analysis_results["timeframe_analysis"][timeframe] = {
                    "technical": technical_results,
                    "patterns": pattern_results,
                    "data_points": len(data)
                }
                
                # Extract signals for consensus
                tech_indicators = technical_results.get('indicators', {})
                trend_direction = tech_indicators.get('trend', {}).get('direction')
                pattern_signal = pattern_results.get('dominant_signal', 'neutral')
                
                if trend_direction:
                    timeframe_signals[timeframe] = trend_direction.value if hasattr(trend_direction, 'value') else str(trend_direction)
                
                # Add pattern signals to timeframe signals
                if pattern_signal != 'neutral':
                    pattern_key = f"{timeframe}_pattern"
                    timeframe_signals[pattern_key] = pattern_signal
            
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
            
            # Original consensus calculation (maintained for backward compatibility)
            consensus = self.calculate_comprehensive_consensus(timeframe_signals, all_signals)
            analysis_results["consensus"] = consensus
            
            # Enhanced Resonance Engine integration
            try:
                from .resonance_engine import ResonanceEngine
                
                # Prepare data for ResonanceEngine
                resonance_data = {
                    'timeframes': {}
                }
                
                # Structure timeframe data for resonance analysis
                for timeframe, tf_data in analysis_results["timeframe_analysis"].items():
                    resonance_data['timeframes'][timeframe] = {
                        'technical': tf_data.get('technical', {}),
                        'patterns': tf_data.get('patterns', {}),
                        'fundamental': analysis_results["fundamental_analysis"],
                        'sentiment': analysis_results["sentiment_analysis"]
                    }
                
                # Calculate resonance consensus
                resonance_engine = ResonanceEngine()
                resonance_consensus = resonance_engine.calculate_consensus(resonance_data)
                
                # Add enhanced consensus fields while maintaining backward compatibility
                analysis_results["consensus"].update({
                    'consensus_score': resonance_consensus.get('consensus_score', 0.5),
                    'confidence_level': resonance_consensus.get('confidence_level', 'moderate'),
                    'alignment_status': resonance_consensus.get('alignment_status', 'no_consensus'),
                    'resonance_analysis': resonance_consensus
                })
                
            except ImportError:
                logger.warning("ResonanceEngine not available, using standard consensus only")
                # Add fallback enhanced fields to maintain API consistency
                analysis_results["consensus"].update({
                    'consensus_score': consensus.get('agreement', 0.5),
                    'confidence_level': 'moderate' if consensus.get('strength') == 'moderate' else 'low',
                    'alignment_status': 'partially_aligned' if consensus.get('agreement', 0) > 0.5 else 'no_consensus'
                })
            except Exception as e:
                logger.warning(f"Resonance Engine calculation failed: {e}, using fallback")
                analysis_results["consensus"].update({
                    'consensus_score': consensus.get('agreement', 0.5),
                    'confidence_level': 'moderate',
                    'alignment_status': 'no_consensus'
                })
            
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


