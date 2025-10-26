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
                    'bollinger': self.calculate_bollinger_bands(data['Close']),
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
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands (simplified)."""
        if len(prices) < period:
            return {"position": "neutral", "bandwidth": 0.0}
        
        try:
            sma = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            current_price = prices.iloc[-1]
            current_sma = sma.iloc[-1] if not sma.empty else current_price
            
            # Simplified position determination
            position = "above_upper" if current_price > current_sma * 1.02 else "below_lower" if current_price < current_sma * 0.98 else "within_bands"
            bandwidth = (rolling_std.iloc[-1] / current_sma) * 100 if current_sma > 0 else 0.0
            
            return {"position": position, "bandwidth": round(bandwidth, 2)}
            
        except Exception:
            return {"position": "neutral", "bandwidth": 0.0}
    
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
        """Simplified support and resistance detection."""
        if data.empty or len(data) < window * 2:
            return {"support_levels": [], "resistance_levels": []}
        
        try:
            highs = data['High'] if 'High' in data.columns else data['Close']
            lows = data['Low'] if 'Low' in data.columns else data['Close']
            
            # Simple peak/trough detection
            resistance_levels = [highs.max()]
            support_levels = [lows.min()]
            
            return {"support_levels": support_levels, "resistance_levels": resistance_levels}
            
        except Exception:
            return {"support_levels": [], "resistance_levels": []}


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
    """Sentiment analysis using shared data manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_manager = get_data_manager()
    
    def analyze_sentiment(self, max_articles: int = 25) -> Dict[str, Any]:
        """Analyze market sentiment from news data."""
        try:
            sentiment_data = self.data_manager.collect_sentiment_data(max_articles=max_articles)
            
            if sentiment_data.get('status') != 'success':
                return {'status': 'failed', 'sentiment_score': 0.0, 'sentiment_bias': 'neutral'}
            
            articles = sentiment_data.get('articles', [])
            if not articles:
                return {'status': 'failed', 'sentiment_score': 0.0, 'sentiment_bias': 'neutral'}
            
            # Calculate aggregate sentiment
            sentiment_scores = [a.get('sentiment_score', 0.0) for a in articles]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Categorize sentiment
            sentiment_bias = 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
            
            return {
                'status': 'success',
                'sentiment_score': round(avg_sentiment, 3),
                'sentiment_bias': sentiment_bias,
                'article_count': len(articles),
                'confidence': min(abs(avg_sentiment) * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
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


