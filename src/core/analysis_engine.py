"""
Analysis Engine Module
Consolidated analysis functionality including technical, fundamental, pattern, and sentiment analysis.

Consolidates functionality from:
- src/core/technical_analysis.py
- src/core/pattern_recognition.py  
- src/core/candlestick_patterns.py
- src/core/correlation_engine.py
- src/core/sector_analysis.py
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum

# Optional imports with fallbacks
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("sklearn/scipy unavailable. Using fallback implementations.")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logging.warning("TextBlob unavailable. Using keyword-based sentiment.")

from ..config import settings

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class TechnicalAnalyzer:
    """
    Technical analysis engine consolidating technical_analysis.py functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        if data.empty or len(data) < 20:
            return {}
        
        try:
            indicators = {}
            
            if 'Close' in data.columns:
                indicators['rsi'] = self.calculate_rsi(data['Close'])
                indicators['macd'] = self.calculate_macd(data['Close'])
                indicators['bollinger'] = self.calculate_bollinger_bands(data['Close'])
                indicators['trend'] = self.detect_trend(data)
                
            if 'Volume' in data.columns:
                indicators['volume'] = self.analyze_volume(data)
                
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
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
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return {"current": 50.0, "signal": "neutral"}
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """Calculate MACD indicator."""
        if len(prices) < max(fast, slow, signal) + 1:
            return {"crossover": "neutral"}
        
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
            
            return {"crossover": crossover}
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {"crossover": "neutral"}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"position": "neutral"}
        
        try:
            sma = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper = sma + (rolling_std * std)
            lower = sma - (rolling_std * std)
            
            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1] if not upper.empty else current_price
            current_lower = lower.iloc[-1] if not lower.empty else current_price
            
            if current_price > current_upper:
                position = "above_upper"
            elif current_price < current_lower:
                position = "below_lower"
            else:
                position = "within_bands"
            
            return {"position": position}
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {"position": "neutral"}
    
    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect price trend direction."""
        if data.empty or 'Close' not in data.columns or len(data) < 20:
            return {"direction": TrendDirection.UNKNOWN, "strength": 0.0}
        
        try:
            prices = data['Close']
            ma_short = prices.rolling(10).mean()
            ma_long = prices.rolling(20).mean()
            
            if not ma_short.empty and not ma_long.empty:
                current_short = ma_short.iloc[-1]
                current_long = ma_long.iloc[-1]
                
                if current_short > current_long:
                    direction = TrendDirection.BULLISH
                    strength = min((current_short - current_long) / current_long * 100, 100)
                elif current_short < current_long:
                    direction = TrendDirection.BEARISH
                    strength = min((current_long - current_short) / current_long * 100, 100)
                else:
                    direction = TrendDirection.SIDEWAYS
                    strength = 0.0
            else:
                direction = TrendDirection.UNKNOWN
                strength = 0.0
            
            return {"direction": direction, "strength": round(strength, 2)}
            
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return {"direction": TrendDirection.UNKNOWN, "strength": 0.0}
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if 'Volume' not in data.columns or len(data) < 10:
            return {"trend": "unknown", "relative_volume": 1.0}
        
        try:
            volume = data['Volume']
            volume_ma = volume.rolling(10).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_ma.iloc[-1] if not volume_ma.empty else 1
            
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if relative_volume > 1.5:
                trend = "increasing"
            elif relative_volume < 0.5:
                trend = "decreasing"
            else:
                trend = "normal"
            
            return {"trend": trend, "relative_volume": round(relative_volume, 2)}
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"trend": "unknown", "relative_volume": 1.0}


class FundamentalAnalyzer:
    """
    Fundamental analysis engine for financial metrics and valuation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def calculate_financial_ratios(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive financial ratios."""
        try:
            ratios = {}
            
            # Valuation ratios
            pe_ratio = financial_data.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:
                    ratios['pe_signal'] = "undervalued"
                elif pe_ratio > 25:
                    ratios['pe_signal'] = "overvalued"
                else:
                    ratios['pe_signal'] = "fair_value"
                ratios['pe_ratio'] = pe_ratio
            
            # Profitability ratios
            profit_margins = financial_data.get('profit_margins')
            if profit_margins:
                ratios['profit_margins'] = profit_margins
                ratios['profitability_signal'] = "strong" if profit_margins > 0.15 else "weak"
            
            # Liquidity ratios
            current_ratio = financial_data.get('current_ratio')
            if current_ratio:
                ratios['current_ratio'] = current_ratio
                ratios['liquidity_signal'] = "strong" if current_ratio > 1.5 else "weak"
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return {}
    
    def calculate_valuation_metrics(self, financial_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate valuation metrics."""
        try:
            valuation = {}
            
            market_cap = market_data.get('market_cap')
            revenue = financial_data.get('latest_revenue')
            
            if market_cap and revenue and revenue > 0:
                price_to_sales = market_cap / revenue
                valuation['price_to_sales'] = round(price_to_sales, 2)
                
                if price_to_sales < 2:
                    valuation['ps_assessment'] = "undervalued"
                elif price_to_sales > 5:
                    valuation['ps_assessment'] = "overvalued"
                else:
                    valuation['ps_assessment'] = "fair_value"
            
            return valuation
            
        except Exception as e:
            logger.error(f"Error calculating valuation metrics: {e}")
            return {}


class PatternAnalyzer:
    """
    Pattern recognition engine consolidating pattern_recognition.py and candlestick_patterns.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def detect_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns."""
        if data.empty or len(data) < 20:
            return {"patterns": [], "count": 0}
        
        try:
            patterns = []
            
            if 'High' in data.columns and 'Low' in data.columns:
                support_resistance = self.detect_support_resistance(data)
                
                # Simple ascending triangle detection
                if len(data) >= 10:
                    recent_highs = data['High'].tail(10)
                    recent_lows = data['Low'].tail(10)
                    
                    if (recent_lows.is_monotonic_increasing and 
                        recent_highs.std() < recent_highs.mean() * 0.02):
                        patterns.append({
                            "pattern_type": "ascending_triangle",
                            "confidence": 0.7,
                            "direction": "bullish"
                        })
            
            return {"patterns": patterns, "count": len(patterns)}
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
            return {"patterns": [], "count": 0}
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect candlestick patterns."""
        if data.empty or len(data) < 3:
            return {"patterns": [], "bullish_count": 0, "bearish_count": 0}
        
        try:
            patterns = []
            
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                for i in range(1, len(data)):
                    open_price = data['Open'].iloc[i]
                    close_price = data['Close'].iloc[i]
                    high_price = data['High'].iloc[i]
                    low_price = data['Low'].iloc[i]
                    
                    body_size = abs(close_price - open_price)
                    total_range = high_price - low_price
                    
                    # Doji pattern
                    if total_range > 0 and body_size / total_range < 0.1:
                        patterns.append({
                            "pattern": "doji",
                            "signal": "reversal",
                            "confidence": 0.6
                        })
            
            bullish_count = sum(1 for p in patterns if p.get("signal") == "bullish")
            bearish_count = sum(1 for p in patterns if p.get("signal") == "bearish")
            
            return {
                "patterns": patterns[-5:],  # Last 5 patterns
                "bullish_count": bullish_count,
                "bearish_count": bearish_count
            }
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {"patterns": [], "bullish_count": 0, "bearish_count": 0}
    
    def detect_support_resistance(self, data: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
        """Detect support and resistance levels."""
        if data.empty or len(data) < window * 2:
            return {"support_levels": [], "resistance_levels": []}
        
        try:
            highs = data['High'] if 'High' in data.columns else data['Close']
            lows = data['Low'] if 'Low' in data.columns else data['Close']
            
            resistance_levels = []
            support_levels = []
            
            # Simple peak/trough detection
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
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {"support_levels": [], "resistance_levels": []}


class SentimentAnalyzer:
    """
    Sentiment analysis engine for news and market sentiment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not news_data:
            return {"overall_sentiment": "neutral", "sentiment_score": 0.0, "article_count": 0}
        
        try:
            sentiment_scores = []
            
            for article in news_data:
                text = ""
                if 'title' in article:
                    text += article['title'] + " "
                if 'content' in article:
                    text += article.get('content', '')
                
                if text.strip():
                    sentiment = self.calculate_sentiment_score(text)
                    sentiment_scores.append(sentiment['score'])
            
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                if avg_sentiment > 0.1:
                    overall_sentiment = "positive"
                elif avg_sentiment < -0.1:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"
            else:
                avg_sentiment = 0.0
                overall_sentiment = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": round(avg_sentiment, 3),
                "article_count": len(news_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {"overall_sentiment": "neutral", "sentiment_score": 0.0, "article_count": len(news_data)}
    
    def calculate_sentiment_score(self, text: str) -> Dict[str, Any]:
        """Calculate sentiment score for text."""
        if not text or not text.strip():
            return {"score": 0.0, "label": "neutral"}
        
        try:
            if HAS_TEXTBLOB:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                    
            else:
                # Keyword-based fallback
                positive_keywords = ['good', 'great', 'positive', 'gain', 'profit', 'growth', 'up', 'rise']
                negative_keywords = ['bad', 'negative', 'loss', 'decline', 'down', 'fall', 'crash']
                
                text_lower = text.lower()
                positive_score = sum(1 for word in positive_keywords if word in text_lower)
                negative_score = sum(1 for word in negative_keywords if word in text_lower)
                
                total_words = len(text.split())
                if total_words > 0:
                    polarity = (positive_score - negative_score) / total_words * 5
                    polarity = max(-1, min(1, polarity))
                else:
                    polarity = 0
                
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
            
            return {"score": round(polarity, 3), "label": label}
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return {"score": 0.0, "label": "neutral"}


class AnalysisEngine:
    """
    Main analysis engine coordinating all analysis modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical = TechnicalAnalyzer(config)
        self.fundamental = FundamentalAnalyzer(config)
        self.pattern = PatternAnalyzer(config)
        self.sentiment = SentimentAnalyzer(config)
        
    def comprehensive_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using all engines."""
        try:
            analysis_results = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "analysis_components": {}
            }
            
            # Technical analysis
            if 'price_data' in data and not data['price_data'].empty:
                technical_results = self.technical.calculate_indicators(data['price_data'])
                analysis_results["analysis_components"]["technical"] = technical_results
            
            # Fundamental analysis
            if 'fundamental_data' in data:
                fundamental_results = self.fundamental.calculate_financial_ratios(data['fundamental_data'])
                analysis_results["analysis_components"]["fundamental"] = fundamental_results
                
                if 'market_data' in data:
                    valuation_results = self.fundamental.calculate_valuation_metrics(
                        data['fundamental_data'], data['market_data']
                    )
                    analysis_results["analysis_components"]["valuation"] = valuation_results
            
            # Pattern analysis
            if 'price_data' in data and not data['price_data'].empty:
                pattern_results = self.pattern.detect_chart_patterns(data['price_data'])
                candlestick_results = self.pattern.detect_candlestick_patterns(data['price_data'])
                
                analysis_results["analysis_components"]["patterns"] = {
                    "chart_patterns": pattern_results,
                    "candlestick_patterns": candlestick_results
                }
            
            # Sentiment analysis
            if 'news_data' in data:
                sentiment_results = self.sentiment.analyze_news_sentiment(data['news_data'])
                analysis_results["analysis_components"]["sentiment"] = sentiment_results
            
            # Generate signals
            signals = self.generate_signals(analysis_results)
            analysis_results["signals"] = signals
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "analysis_components": {},
                "signals": {}
            }
    
    def generate_signals(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from analysis results."""
        try:
            signals = {
                "overall_signal": "hold",
                "signal_strength": "weak",
                "component_signals": {},
                "signal_count": {"buy": 0, "sell": 0, "hold": 0}
            }
            
            components = analysis_results.get("analysis_components", {})
            
            # Technical signals
            if "technical" in components:
                tech = components["technical"]
                tech_signal = "hold"
                
                # RSI signal
                if "rsi" in tech:
                    rsi_signal = tech["rsi"].get("signal", "neutral")
                    if rsi_signal == "oversold":
                        tech_signal = "buy"
                    elif rsi_signal == "overbought":
                        tech_signal = "sell"
                
                # Trend signal
                if "trend" in tech:
                    trend_dir = tech["trend"].get("direction")
                    if trend_dir == TrendDirection.BULLISH:
                        tech_signal = "buy" if tech_signal != "sell" else "hold"
                    elif trend_dir == TrendDirection.BEARISH:
                        tech_signal = "sell" if tech_signal != "buy" else "hold"
                
                signals["component_signals"]["technical"] = tech_signal
                signals["signal_count"][tech_signal] += 1
            
            # Fundamental signals
            if "fundamental" in components:
                fund = components["fundamental"]
                fund_signal = "hold"
                
                pe_signal = fund.get("pe_signal")
                if pe_signal == "undervalued":
                    fund_signal = "buy"
                elif pe_signal == "overvalued":
                    fund_signal = "sell"
                
                signals["component_signals"]["fundamental"] = fund_signal
                signals["signal_count"][fund_signal] += 1
            
            # Sentiment signals
            if "sentiment" in components:
                sent = components["sentiment"]
                sent_signal = "hold"
                
                overall_sentiment = sent.get("overall_sentiment", "neutral")
                if overall_sentiment == "positive":
                    sent_signal = "buy"
                elif overall_sentiment == "negative":
                    sent_signal = "sell"
                
                signals["component_signals"]["sentiment"] = sent_signal
                signals["signal_count"][sent_signal] += 1
            
            # Determine overall signal
            buy_count = signals["signal_count"]["buy"]
            sell_count = signals["signal_count"]["sell"]
            total_signals = buy_count + sell_count + signals["signal_count"]["hold"]
            
            if total_signals > 0:
                if buy_count > sell_count and buy_count / total_signals > 0.5:
                    signals["overall_signal"] = "buy"
                    signals["signal_strength"] = "strong" if buy_count / total_signals > 0.7 else "moderate"
                elif sell_count > buy_count and sell_count / total_signals > 0.5:
                    signals["overall_signal"] = "sell"
                    signals["signal_strength"] = "strong" if sell_count / total_signals > 0.7 else "moderate"
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                "overall_signal": "hold",
                "signal_strength": "weak",
                "component_signals": {},
                "signal_count": {"buy": 0, "sell": 0, "hold": 0}
            }


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
        
    except Exception as e:
        logger.error(f"Error calculating composite score: {e}")
        return 0.0