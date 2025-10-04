"""Candlestick pattern recognition system for technical analysis.

This module implements detection algorithms for classic candlestick patterns including:
- Single candlestick patterns (Doji, Hammer, Shooting Star, etc.)
- Two-candlestick patterns (Engulfing, Harami, etc.)
- Three-candlestick patterns (Morning Star, Evening Star, Three White Soldiers, etc.)
- Complex multi-candlestick patterns

Each pattern includes confidence scoring and historical validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CandlestickPatternType(Enum):
    """Enumeration of supported candlestick patterns."""
    # Single candlestick patterns
    DOJI = "doji"
    LONG_LEGGED_DOJI = "long_legged_doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    SPINNING_TOP = "spinning_top"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"
    
    # Two candlestick patterns
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    BULLISH_HARAMI = "bullish_harami"
    BEARISH_HARAMI = "bearish_harami"
    PIERCING_PATTERN = "piercing_pattern"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"
    
    # Three candlestick patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    MORNING_DOJI_STAR = "morning_doji_star"
    EVENING_DOJI_STAR = "evening_doji_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    THREE_OUTSIDE_UP = "three_outside_up"
    THREE_OUTSIDE_DOWN = "three_outside_down"
    
    # Advanced patterns
    ABANDONED_BABY_BULLISH = "abandoned_baby_bullish"
    ABANDONED_BABY_BEARISH = "abandoned_baby_bearish"
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"

class CandlestickDirection(Enum):
    """Candlestick pattern direction classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class CandlestickDetection:
    """Complete candlestick pattern detection result."""
    pattern_type: CandlestickPatternType
    direction: CandlestickDirection
    confidence: float
    start_index: int
    end_index: int
    pattern_strength: str  # 'weak', 'moderate', 'strong'
    context_support: float  # How well the pattern fits market context
    volume_confirmation: bool
    trend_context: str  # 'uptrend', 'downtrend', 'sideways'
    reliability_score: float  # Historical reliability of this pattern type
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    metadata: Dict

class CandlestickAnalyzer:
    """Main class for detecting candlestick patterns in OHLCV data."""
    
    def __init__(self,
                 min_body_ratio: float = 0.1,
                 doji_threshold: float = 0.05,
                 volume_confirmation: bool = True,
                 trend_lookback: int = 10):
        """
        Initialize candlestick analyzer.
        
        Args:
            min_body_ratio: Minimum body size as ratio of total range
            doji_threshold: Maximum body size for doji patterns (as ratio of range)
            volume_confirmation: Whether to require volume confirmation
            trend_lookback: Number of periods to look back for trend analysis
        """
        self.min_body_ratio = min_body_ratio
        self.doji_threshold = doji_threshold
        self.volume_confirmation = volume_confirmation
        self.trend_lookback = trend_lookback
        
        # Pattern reliability scores (historical success rates)
        self.pattern_reliability = {
            CandlestickPatternType.DOJI: 0.65,
            CandlestickPatternType.HAMMER: 0.72,
            CandlestickPatternType.HANGING_MAN: 0.68,
            CandlestickPatternType.SHOOTING_STAR: 0.70,
            CandlestickPatternType.BULLISH_ENGULFING: 0.78,
            CandlestickPatternType.BEARISH_ENGULFING: 0.76,
            CandlestickPatternType.MORNING_STAR: 0.84,
            CandlestickPatternType.EVENING_STAR: 0.82,
            CandlestickPatternType.THREE_WHITE_SOLDIERS: 0.86,
            CandlestickPatternType.THREE_BLACK_CROWS: 0.84,
            # Add more as needed
        }
    
    def detect_patterns(self, data: pd.DataFrame, 
                       symbol: str = "UNKNOWN") -> List[CandlestickDetection]:
        """
        Detect all candlestick patterns in the given price data.
        
        Args:
            data: OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            symbol: Symbol being analyzed for context
            
        Returns:
            List of detected patterns with confidence scores
        """
        if len(data) < 3:
            logger.warning(f"Insufficient data for candlestick pattern detection: {len(data)} periods")
            return []
        
        patterns = []
        
        try:
            # Calculate candlestick properties
            candle_data = self._calculate_candlestick_properties(data)
            
            # Detect single candlestick patterns
            patterns.extend(self._detect_single_patterns(data, candle_data))
            
            # Detect two-candlestick patterns
            patterns.extend(self._detect_two_patterns(data, candle_data))
            
            # Detect three-candlestick patterns
            patterns.extend(self._detect_three_patterns(data, candle_data))
            
            # Detect advanced patterns
            patterns.extend(self._detect_advanced_patterns(data, candle_data))
            
            # Add context and final scoring
            for pattern in patterns:
                self._enhance_pattern_with_context(pattern, data, candle_data)
            
            # Sort by confidence score
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            # Filter weak patterns
            patterns = [p for p in patterns if p.confidence >= 0.3]
            
        except Exception as e:
            logger.error(f"Error in candlestick pattern detection: {e}")
        
        return patterns
    
    def _calculate_candlestick_properties(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick properties for pattern detection."""
        
        candle_props = pd.DataFrame(index=data.index)
        
        # Basic OHLC
        candle_props['open'] = data['Open']
        candle_props['high'] = data['High']
        candle_props['low'] = data['Low']
        candle_props['close'] = data['Close']
        candle_props['volume'] = data.get('Volume', 0)
        
        # Candlestick components
        candle_props['body_size'] = abs(data['Close'] - data['Open'])
        candle_props['upper_shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
        candle_props['lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        candle_props['total_range'] = data['High'] - data['Low']
        
        # Ratios and percentages
        candle_props['body_ratio'] = np.where(
            candle_props['total_range'] > 0,
            candle_props['body_size'] / candle_props['total_range'],
            0
        )
        candle_props['upper_shadow_ratio'] = np.where(
            candle_props['total_range'] > 0,
            candle_props['upper_shadow'] / candle_props['total_range'],
            0
        )
        candle_props['lower_shadow_ratio'] = np.where(
            candle_props['total_range'] > 0,
            candle_props['lower_shadow'] / candle_props['total_range'],
            0
        )
        
        # Candlestick classification
        candle_props['is_bullish'] = data['Close'] > data['Open']
        candle_props['is_bearish'] = data['Close'] < data['Open']
        candle_props['is_doji'] = candle_props['body_ratio'] <= self.doji_threshold
        
        # Size classification
        candle_props['is_long_body'] = candle_props['body_ratio'] >= 0.7
        candle_props['is_short_body'] = candle_props['body_ratio'] <= 0.3
        candle_props['is_long_upper_shadow'] = candle_props['upper_shadow_ratio'] >= 0.6
        candle_props['is_long_lower_shadow'] = candle_props['lower_shadow_ratio'] >= 0.6
        
        # Price change and volatility
        candle_props['price_change'] = data['Close'] - data['Open']
        candle_props['price_change_pct'] = np.where(
            data['Open'] > 0,
            (data['Close'] - data['Open']) / data['Open'],
            0
        )
        
        # Volume analysis
        if 'Volume' in data.columns:
            candle_props['volume_sma'] = data['Volume'].rolling(window=10, min_periods=1).mean()
            candle_props['volume_ratio'] = np.where(
                candle_props['volume_sma'] > 0,
                data['Volume'] / candle_props['volume_sma'],
                1
            )
            candle_props['high_volume'] = candle_props['volume_ratio'] >= 1.5
        else:
            candle_props['volume_ratio'] = 1
            candle_props['high_volume'] = False
        
        return candle_props
    
    def _detect_single_patterns(self, data: pd.DataFrame, 
                              candle_data: pd.DataFrame) -> List[CandlestickDetection]:
        """Detect single candlestick patterns."""
        patterns = []
        
        for i in range(len(data)):
            # Skip if insufficient data for context
            if i < 1:
                continue
            
            candle = candle_data.iloc[i]
            
            # Doji patterns
            if candle['is_doji']:
                doji_pattern = self._analyze_doji_pattern(data, candle_data, i)
                if doji_pattern:
                    patterns.append(doji_pattern)
            
            # Hammer patterns
            elif (candle['lower_shadow_ratio'] >= 0.6 and 
                  candle['upper_shadow_ratio'] <= 0.1 and
                  candle['body_ratio'] <= 0.3):
                hammer_pattern = self._analyze_hammer_pattern(data, candle_data, i)
                if hammer_pattern:
                    patterns.append(hammer_pattern)
            
            # Shooting star / Inverted hammer
            elif (candle['upper_shadow_ratio'] >= 0.6 and 
                  candle['lower_shadow_ratio'] <= 0.1 and
                  candle['body_ratio'] <= 0.3):
                star_pattern = self._analyze_star_pattern(data, candle_data, i)
                if star_pattern:
                    patterns.append(star_pattern)
            
            # Spinning top
            elif (candle['upper_shadow_ratio'] >= 0.25 and 
                  candle['lower_shadow_ratio'] >= 0.25 and
                  candle['body_ratio'] <= 0.3):
                spinning_pattern = self._analyze_spinning_top(data, candle_data, i)
                if spinning_pattern:
                    patterns.append(spinning_pattern)
            
            # Marubozu
            elif (candle['body_ratio'] >= 0.9 and
                  candle['upper_shadow_ratio'] <= 0.05 and
                  candle['lower_shadow_ratio'] <= 0.05):
                marubozu_pattern = self._analyze_marubozu(data, candle_data, i)
                if marubozu_pattern:
                    patterns.append(marubozu_pattern)
        
        return patterns
    
    def _analyze_doji_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                            index: int) -> Optional[CandlestickDetection]:
        """Analyze doji pattern and its variations."""
        
        candle = candle_data.iloc[index]
        
        # Determine doji subtype
        if (candle['upper_shadow_ratio'] >= 0.6 and 
            candle['lower_shadow_ratio'] >= 0.6):
            pattern_type = CandlestickPatternType.LONG_LEGGED_DOJI
            base_confidence = 0.7
        elif (candle['lower_shadow_ratio'] >= 0.6 and 
              candle['upper_shadow_ratio'] <= 0.1):
            pattern_type = CandlestickPatternType.DRAGONFLY_DOJI
            base_confidence = 0.75
        elif (candle['upper_shadow_ratio'] >= 0.6 and 
              candle['lower_shadow_ratio'] <= 0.1):
            pattern_type = CandlestickPatternType.GRAVESTONE_DOJI
            base_confidence = 0.75
        else:
            pattern_type = CandlestickPatternType.DOJI
            base_confidence = 0.65
        
        # Determine direction based on trend context
        trend = self._analyze_trend_context(data, index)
        
        if pattern_type == CandlestickPatternType.DRAGONFLY_DOJI:
            direction = CandlestickDirection.BULLISH if trend == 'downtrend' else CandlestickDirection.NEUTRAL
        elif pattern_type == CandlestickPatternType.GRAVESTONE_DOJI:
            direction = CandlestickDirection.BEARISH if trend == 'uptrend' else CandlestickDirection.NEUTRAL
        else:
            direction = CandlestickDirection.NEUTRAL
        
        # Calculate confidence
        confidence = self._calculate_single_pattern_confidence(
            candle, base_confidence, trend, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_single_pattern_targets(
            data, index, direction, candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.6),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'body_ratio': candle['body_ratio'],
                'upper_shadow_ratio': candle['upper_shadow_ratio'],
                'lower_shadow_ratio': candle['lower_shadow_ratio'],
                'volume_ratio': candle['volume_ratio']
            }
        )
    
    def _analyze_hammer_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                              index: int) -> Optional[CandlestickDetection]:
        """Analyze hammer pattern (hammer or hanging man)."""
        
        candle = candle_data.iloc[index]
        trend = self._analyze_trend_context(data, index)
        
        # Determine if it's a hammer (bullish in downtrend) or hanging man (bearish in uptrend)
        if trend == 'downtrend':
            pattern_type = CandlestickPatternType.HAMMER
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.72
        elif trend == 'uptrend':
            pattern_type = CandlestickPatternType.HANGING_MAN
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.68
        else:
            # In sideways market, lower confidence
            pattern_type = CandlestickPatternType.HAMMER if candle['is_bullish'] else CandlestickPatternType.HANGING_MAN
            direction = CandlestickDirection.BULLISH if candle['is_bullish'] else CandlestickDirection.BEARISH
            base_confidence = 0.55
        
        # Calculate confidence
        confidence = self._calculate_single_pattern_confidence(
            candle, base_confidence, trend, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_single_pattern_targets(
            data, index, direction, candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.7),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'lower_shadow_length': candle['lower_shadow'] / candle['body_size'] if candle['body_size'] > 0 else float('inf'),
                'body_position': 'upper' if candle['close'] > candle['open'] else 'lower',
                'shadow_body_ratio': candle['lower_shadow_ratio'] / candle['body_ratio'] if candle['body_ratio'] > 0 else float('inf')
            }
        )
    
    def _analyze_star_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                            index: int) -> Optional[CandlestickDetection]:
        """Analyze shooting star or inverted hammer pattern."""
        
        candle = candle_data.iloc[index]
        trend = self._analyze_trend_context(data, index)
        
        # Determine pattern type based on trend context
        if trend == 'uptrend':
            pattern_type = CandlestickPatternType.SHOOTING_STAR
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.70
        elif trend == 'downtrend':
            pattern_type = CandlestickPatternType.INVERTED_HAMMER
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.68
        else:
            # In sideways market
            pattern_type = CandlestickPatternType.SHOOTING_STAR if trend != 'downtrend' else CandlestickPatternType.INVERTED_HAMMER
            direction = CandlestickDirection.BEARISH if pattern_type == CandlestickPatternType.SHOOTING_STAR else CandlestickDirection.BULLISH
            base_confidence = 0.55
        
        # Calculate confidence
        confidence = self._calculate_single_pattern_confidence(
            candle, base_confidence, trend, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_single_pattern_targets(
            data, index, direction, candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.69),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'upper_shadow_length': candle['upper_shadow'] / candle['body_size'] if candle['body_size'] > 0 else float('inf'),
                'body_color': 'green' if candle['is_bullish'] else 'red',
                'gap_above': data['Low'].iloc[index] > data['High'].iloc[index - 1] if index > 0 else False
            }
        )
    
    def _analyze_spinning_top(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                            index: int) -> Optional[CandlestickDetection]:
        """Analyze spinning top pattern."""
        
        candle = candle_data.iloc[index]
        trend = self._analyze_trend_context(data, index)
        
        pattern_type = CandlestickPatternType.SPINNING_TOP
        direction = CandlestickDirection.NEUTRAL
        base_confidence = 0.60
        
        # Spinning tops indicate indecision, higher confidence in trending markets
        if trend in ['uptrend', 'downtrend']:
            base_confidence = 0.65
        
        confidence = self._calculate_single_pattern_confidence(
            candle, base_confidence, trend, pattern_type
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.60),
            target_price=None,
            stop_loss=None,
            risk_reward_ratio=None,
            metadata={
                'shadow_symmetry': 1 - abs(candle['upper_shadow_ratio'] - candle['lower_shadow_ratio']),
                'body_small': candle['body_ratio'] < 0.2,
                'indecision_score': (candle['upper_shadow_ratio'] + candle['lower_shadow_ratio']) / 2
            }
        )
    
    def _analyze_marubozu(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                        index: int) -> Optional[CandlestickDetection]:
        """Analyze marubozu pattern."""
        
        candle = candle_data.iloc[index]
        trend = self._analyze_trend_context(data, index)
        
        if candle['is_bullish']:
            pattern_type = CandlestickPatternType.MARUBOZU_BULLISH
            direction = CandlestickDirection.BULLISH
        else:
            pattern_type = CandlestickPatternType.MARUBOZU_BEARISH
            direction = CandlestickDirection.BEARISH
        
        base_confidence = 0.75
        
        # Higher confidence if aligned with trend
        if ((direction == CandlestickDirection.BULLISH and trend == 'uptrend') or
            (direction == CandlestickDirection.BEARISH and trend == 'downtrend')):
            base_confidence = 0.80
        
        confidence = self._calculate_single_pattern_confidence(
            candle, base_confidence, trend, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_single_pattern_targets(
            data, index, direction, candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.75),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'body_strength': candle['body_ratio'],
                'price_change_pct': candle['price_change_pct'],
                'continuation_signal': direction.value == trend
            }
        )
    
    def _detect_two_patterns(self, data: pd.DataFrame, 
                           candle_data: pd.DataFrame) -> List[CandlestickDetection]:
        """Detect two-candlestick patterns."""
        patterns = []
        
        for i in range(1, len(data)):
            prev_candle = candle_data.iloc[i - 1]
            curr_candle = candle_data.iloc[i]
            
            # Engulfing patterns
            engulfing = self._analyze_engulfing_pattern(data, candle_data, i)
            if engulfing:
                patterns.append(engulfing)
            
            # Harami patterns
            harami = self._analyze_harami_pattern(data, candle_data, i)
            if harami:
                patterns.append(harami)
            
            # Piercing pattern / Dark cloud cover
            piercing = self._analyze_piercing_pattern(data, candle_data, i)
            if piercing:
                patterns.append(piercing)
            
            # Tweezer patterns
            tweezer = self._analyze_tweezer_pattern(data, candle_data, i)
            if tweezer:
                patterns.append(tweezer)
        
        return patterns
    
    def _analyze_engulfing_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                                 index: int) -> Optional[CandlestickDetection]:
        """Analyze bullish/bearish engulfing patterns."""
        
        if index < 1:
            return None
        
        prev_candle = candle_data.iloc[index - 1]
        curr_candle = candle_data.iloc[index]
        
        # Check for engulfing criteria
        prev_body_top = max(prev_candle['open'], prev_candle['close'])
        prev_body_bottom = min(prev_candle['open'], prev_candle['close'])
        curr_body_top = max(curr_candle['open'], curr_candle['close'])
        curr_body_bottom = min(curr_candle['open'], curr_candle['close'])
        
        is_bullish_engulfing = (
            prev_candle['is_bearish'] and 
            curr_candle['is_bullish'] and
            curr_body_bottom < prev_body_bottom and
            curr_body_top > prev_body_top
        )
        
        is_bearish_engulfing = (
            prev_candle['is_bullish'] and 
            curr_candle['is_bearish'] and
            curr_body_top > prev_body_top and
            curr_body_bottom < prev_body_bottom
        )
        
        if not (is_bullish_engulfing or is_bearish_engulfing):
            return None
        
        # Determine pattern type
        if is_bullish_engulfing:
            pattern_type = CandlestickPatternType.BULLISH_ENGULFING
            direction = CandlestickDirection.BULLISH
        else:
            pattern_type = CandlestickPatternType.BEARISH_ENGULFING
            direction = CandlestickDirection.BEARISH
        
        # Calculate confidence
        base_confidence = 0.78 if is_bullish_engulfing else 0.76
        confidence = self._calculate_two_pattern_confidence(
            prev_candle, curr_candle, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_two_pattern_targets(
            data, index, direction, curr_candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 1,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.77),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'engulfing_ratio': curr_candle['body_size'] / prev_candle['body_size'] if prev_candle['body_size'] > 0 else float('inf'),
                'size_difference': curr_candle['body_size'] - prev_candle['body_size'],
                'gap_present': self._check_gap_between_candles(prev_candle, curr_candle),
                'volume_surge': curr_candle['volume_ratio'] > 1.5
            }
        )
    
    def _analyze_harami_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                              index: int) -> Optional[CandlestickDetection]:
        """Analyze bullish/bearish harami patterns."""
        
        if index < 1:
            return None
        
        prev_candle = candle_data.iloc[index - 1]
        curr_candle = candle_data.iloc[index]
        
        # Check for harami criteria (current candle body inside previous candle body)
        prev_body_top = max(prev_candle['open'], prev_candle['close'])
        prev_body_bottom = min(prev_candle['open'], prev_candle['close'])
        curr_body_top = max(curr_candle['open'], curr_candle['close'])
        curr_body_bottom = min(curr_candle['open'], curr_candle['close'])
        
        is_harami = (
            curr_body_top < prev_body_top and
            curr_body_bottom > prev_body_bottom and
            prev_candle['body_ratio'] >= 0.6 and  # Previous candle should be large
            curr_candle['body_ratio'] >= 0.1  # Current candle should have some body
        )
        
        if not is_harami:
            return None
        
        # Determine pattern type based on previous candle color and trend
        trend = self._analyze_trend_context(data, index)
        
        if prev_candle['is_bearish'] and trend == 'downtrend':
            pattern_type = CandlestickPatternType.BULLISH_HARAMI
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.70
        elif prev_candle['is_bullish'] and trend == 'uptrend':
            pattern_type = CandlestickPatternType.BEARISH_HARAMI
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.68
        else:
            # Lower confidence if not in appropriate trend
            if prev_candle['is_bearish']:
                pattern_type = CandlestickPatternType.BULLISH_HARAMI
                direction = CandlestickDirection.BULLISH
            else:
                pattern_type = CandlestickPatternType.BEARISH_HARAMI
                direction = CandlestickDirection.BEARISH
            base_confidence = 0.55
        
        confidence = self._calculate_two_pattern_confidence(
            prev_candle, curr_candle, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_two_pattern_targets(
            data, index, direction, prev_candle['total_range']
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 1,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=trend,
            reliability_score=self.pattern_reliability.get(pattern_type, 0.69),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'containment_ratio': curr_candle['body_size'] / prev_candle['body_size'],
                'mother_candle_strength': prev_candle['body_ratio'],
                'baby_candle_position': 'upper' if curr_body_top > (prev_body_top + prev_body_bottom) / 2 else 'lower',
                'trend_reversal_signal': direction.value != trend
            }
        )
    
    def _analyze_piercing_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                                index: int) -> Optional[CandlestickDetection]:
        """Analyze piercing pattern and dark cloud cover."""
        
        if index < 1:
            return None
        
        prev_candle = candle_data.iloc[index - 1]
        curr_candle = candle_data.iloc[index]
        
        # Check for piercing pattern (bullish)
        is_piercing = (
            prev_candle['is_bearish'] and
            curr_candle['is_bullish'] and
            curr_candle['open'] < prev_candle['close'] and
            curr_candle['close'] > (prev_candle['open'] + prev_candle['close']) / 2 and
            curr_candle['close'] < prev_candle['open']
        )
        
        # Check for dark cloud cover (bearish)
        is_dark_cloud = (
            prev_candle['is_bullish'] and
            curr_candle['is_bearish'] and
            curr_candle['open'] > prev_candle['close'] and
            curr_candle['close'] < (prev_candle['open'] + prev_candle['close']) / 2 and
            curr_candle['close'] > prev_candle['open']
        )
        
        if not (is_piercing or is_dark_cloud):
            return None
        
        if is_piercing:
            pattern_type = CandlestickPatternType.PIERCING_PATTERN
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.72
        else:
            pattern_type = CandlestickPatternType.DARK_CLOUD_COVER
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.70
        
        # Calculate penetration percentage
        if is_piercing:
            penetration = (curr_candle['close'] - prev_candle['close']) / prev_candle['body_size']
        else:
            penetration = (prev_candle['close'] - curr_candle['close']) / prev_candle['body_size']
        
        # Adjust confidence based on penetration depth
        penetration_bonus = min(0.1, penetration * 0.2)  # Max 10% bonus
        base_confidence += penetration_bonus
        
        confidence = self._calculate_two_pattern_confidence(
            prev_candle, curr_candle, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_two_pattern_targets(
            data, index, direction, max(prev_candle['total_range'], curr_candle['total_range'])
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 1,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.71),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'penetration_ratio': penetration,
                'gap_size': abs(curr_candle['open'] - prev_candle['close']) / prev_candle['close'],
                'body_size_ratio': curr_candle['body_size'] / prev_candle['body_size'],
                'depth_score': penetration
            }
        )
    
    def _analyze_tweezer_pattern(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                               index: int) -> Optional[CandlestickDetection]:
        """Analyze tweezer top/bottom patterns."""
        
        if index < 1:
            return None
        
        prev_candle = candle_data.iloc[index - 1]
        curr_candle = candle_data.iloc[index]
        
        # Calculate price tolerance for "equal" levels
        avg_price = (prev_candle['close'] + curr_candle['close']) / 2
        tolerance = avg_price * 0.01  # 1% tolerance
        
        # Check for tweezer top (equal highs)
        is_tweezer_top = (
            abs(prev_candle['high'] - curr_candle['high']) <= tolerance and
            prev_candle['is_bullish'] and
            curr_candle['is_bearish'] and
            self._analyze_trend_context(data, index) == 'uptrend'
        )
        
        # Check for tweezer bottom (equal lows)
        is_tweezer_bottom = (
            abs(prev_candle['low'] - curr_candle['low']) <= tolerance and
            prev_candle['is_bearish'] and
            curr_candle['is_bullish'] and
            self._analyze_trend_context(data, index) == 'downtrend'
        )
        
        if not (is_tweezer_top or is_tweezer_bottom):
            return None
        
        if is_tweezer_top:
            pattern_type = CandlestickPatternType.TWEEZER_TOP
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.68
        else:
            pattern_type = CandlestickPatternType.TWEEZER_BOTTOM
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.70
        
        confidence = self._calculate_two_pattern_confidence(
            prev_candle, curr_candle, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        target_price, stop_loss = self._calculate_two_pattern_targets(
            data, index, direction, max(prev_candle['total_range'], curr_candle['total_range'])
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 1,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.69),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'level_precision': 1 - abs(prev_candle['high'] - curr_candle['high']) / avg_price if is_tweezer_top 
                                  else 1 - abs(prev_candle['low'] - curr_candle['low']) / avg_price,
                'reversal_strength': abs(prev_candle['price_change']) + abs(curr_candle['price_change']),
                'equal_level': prev_candle['high'] if is_tweezer_top else prev_candle['low']
            }
        )
    
    def _detect_three_patterns(self, data: pd.DataFrame, 
                             candle_data: pd.DataFrame) -> List[CandlestickDetection]:
        """Detect three-candlestick patterns."""
        patterns = []
        
        for i in range(2, len(data)):
            # Morning/Evening Star patterns
            star_pattern = self._analyze_star_patterns(data, candle_data, i)
            if star_pattern:
                patterns.append(star_pattern)
            
            # Three soldiers/crows patterns
            soldiers_pattern = self._analyze_soldiers_crows(data, candle_data, i)
            if soldiers_pattern:
                patterns.append(soldiers_pattern)
            
            # Three inside patterns
            inside_pattern = self._analyze_three_inside(data, candle_data, i)
            if inside_pattern:
                patterns.append(inside_pattern)
            
            # Three outside patterns
            outside_pattern = self._analyze_three_outside(data, candle_data, i)
            if outside_pattern:
                patterns.append(outside_pattern)
        
        return patterns
    
    def _analyze_star_patterns(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                             index: int) -> Optional[CandlestickDetection]:
        """Analyze morning star, evening star, and doji star patterns."""
        
        if index < 2:
            return None
        
        candle1 = candle_data.iloc[index - 2]  # First candle
        candle2 = candle_data.iloc[index - 1]  # Star candle
        candle3 = candle_data.iloc[index]      # Third candle
        
        # Check for morning star pattern
        is_morning_star = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            candle2['body_size'] < candle1['body_size'] * 0.5 and  # Small middle candle
            candle3['is_bullish'] and candle3['body_ratio'] >= 0.6 and
            candle3['close'] > (candle1['open'] + candle1['close']) / 2  # Recovery above midpoint
        )
        
        # Check for evening star pattern
        is_evening_star = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            candle2['body_size'] < candle1['body_size'] * 0.5 and  # Small middle candle
            candle3['is_bearish'] and candle3['body_ratio'] >= 0.6 and
            candle3['close'] < (candle1['open'] + candle1['close']) / 2  # Decline below midpoint
        )
        
        # Check for doji star variations
        is_morning_doji_star = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_doji'] and
            candle3['is_bullish'] and candle3['body_ratio'] >= 0.6
        )
        
        is_evening_doji_star = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_doji'] and
            candle3['is_bearish'] and candle3['body_ratio'] >= 0.6
        )
        
        if not (is_morning_star or is_evening_star or is_morning_doji_star or is_evening_doji_star):
            return None
        
        # Determine pattern type
        if is_morning_star:
            pattern_type = CandlestickPatternType.MORNING_STAR
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.84
        elif is_evening_star:
            pattern_type = CandlestickPatternType.EVENING_STAR
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.82
        elif is_morning_doji_star:
            pattern_type = CandlestickPatternType.MORNING_DOJI_STAR
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.80
        else:  # evening_doji_star
            pattern_type = CandlestickPatternType.EVENING_DOJI_STAR
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.78
        
        # Calculate confidence
        confidence = self._calculate_three_pattern_confidence(
            candle1, candle2, candle3, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        pattern_height = max(candle1['total_range'], candle2['total_range'], candle3['total_range'])
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, pattern_height
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 2,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.81),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'star_isolation': min(
                    abs(candle2['high'] - candle1['close']) / candle1['close'],
                    abs(candle2['high'] - candle3['open']) / candle3['open']
                ),
                'recovery_strength': abs(candle3['close'] - candle2['close']) / candle2['close'],
                'gap_up': candle2['low'] > max(candle1['close'], candle1['open']),
                'gap_down': candle2['high'] < min(candle1['close'], candle1['open'])
            }
        )
    
    def _analyze_soldiers_crows(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                              index: int) -> Optional[CandlestickDetection]:
        """Analyze three white soldiers and three black crows patterns."""
        
        if index < 2:
            return None
        
        candle1 = candle_data.iloc[index - 2]
        candle2 = candle_data.iloc[index - 1]
        candle3 = candle_data.iloc[index]
        
        # Check for three white soldiers
        is_three_soldiers = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_bullish'] and candle2['body_ratio'] >= 0.6 and
            candle3['is_bullish'] and candle3['body_ratio'] >= 0.6 and
            candle2['close'] > candle1['close'] and
            candle3['close'] > candle2['close'] and
            candle2['open'] >= candle1['body_size'] * 0.3 + min(candle1['open'], candle1['close']) and
            candle3['open'] >= candle2['body_size'] * 0.3 + min(candle2['open'], candle2['close'])
        )
        
        # Check for three black crows
        is_three_crows = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_bearish'] and candle2['body_ratio'] >= 0.6 and
            candle3['is_bearish'] and candle3['body_ratio'] >= 0.6 and
            candle2['close'] < candle1['close'] and
            candle3['close'] < candle2['close'] and
            candle2['open'] <= max(candle1['open'], candle1['close']) - candle1['body_size'] * 0.3 and
            candle3['open'] <= max(candle2['open'], candle2['close']) - candle2['body_size'] * 0.3
        )
        
        if not (is_three_soldiers or is_three_crows):
            return None
        
        if is_three_soldiers:
            pattern_type = CandlestickPatternType.THREE_WHITE_SOLDIERS
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.86
        else:
            pattern_type = CandlestickPatternType.THREE_BLACK_CROWS
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.84
        
        # Calculate confidence
        confidence = self._calculate_three_pattern_confidence(
            candle1, candle2, candle3, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        total_move = abs(candle3['close'] - candle1['open'])
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, total_move
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 2,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.85),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'progression_strength': (candle3['close'] - candle1['open']) / candle1['open'],
                'body_consistency': min(candle1['body_ratio'], candle2['body_ratio'], candle3['body_ratio']),
                'opening_progression': all([
                    candle2['open'] > candle1['close'] * 0.995,
                    candle3['open'] > candle2['close'] * 0.995
                ]) if is_three_soldiers else all([
                    candle2['open'] < candle1['close'] * 1.005,
                    candle3['open'] < candle2['close'] * 1.005
                ]),
                'momentum_acceleration': abs(candle3['price_change']) > abs(candle1['price_change'])
            }
        )
    
    def _analyze_three_inside(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                            index: int) -> Optional[CandlestickDetection]:
        """Analyze three inside up/down patterns."""
        
        if index < 2:
            return None
        
        candle1 = candle_data.iloc[index - 2]
        candle2 = candle_data.iloc[index - 1]
        candle3 = candle_data.iloc[index]
        
        # First check if we have a harami pattern in first two candles
        harami_bullish = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_bullish'] and
            max(candle2['open'], candle2['close']) < max(candle1['open'], candle1['close']) and
            min(candle2['open'], candle2['close']) > min(candle1['open'], candle1['close'])
        )
        
        harami_bearish = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            candle2['is_bearish'] and
            max(candle2['open'], candle2['close']) < max(candle1['open'], candle1['close']) and
            min(candle2['open'], candle2['close']) > min(candle1['open'], candle1['close'])
        )
        
        # Check for three inside up
        is_three_inside_up = (
            harami_bullish and
            candle3['is_bullish'] and
            candle3['close'] > candle1['close']
        )
        
        # Check for three inside down
        is_three_inside_down = (
            harami_bearish and
            candle3['is_bearish'] and
            candle3['close'] < candle1['close']
        )
        
        if not (is_three_inside_up or is_three_inside_down):
            return None
        
        if is_three_inside_up:
            pattern_type = CandlestickPatternType.THREE_INSIDE_UP
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.75
        else:
            pattern_type = CandlestickPatternType.THREE_INSIDE_DOWN
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.73
        
        confidence = self._calculate_three_pattern_confidence(
            candle1, candle2, candle3, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        pattern_height = candle1['total_range']
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, pattern_height
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 2,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.74),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'harami_strength': candle2['body_size'] / candle1['body_size'],
                'confirmation_strength': abs(candle3['close'] - candle1['close']) / candle1['close'],
                'breakout_confirmed': candle3['close'] > candle1['close'] if is_three_inside_up else candle3['close'] < candle1['close']
            }
        )
    
    def _analyze_three_outside(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                             index: int) -> Optional[CandlestickDetection]:
        """Analyze three outside up/down patterns."""
        
        if index < 2:
            return None
        
        candle1 = candle_data.iloc[index - 2]
        candle2 = candle_data.iloc[index - 1]
        candle3 = candle_data.iloc[index]
        
        # Check for engulfing pattern in first two candles
        bullish_engulfing = (
            candle1['is_bearish'] and
            candle2['is_bullish'] and
            min(candle2['open'], candle2['close']) < min(candle1['open'], candle1['close']) and
            max(candle2['open'], candle2['close']) > max(candle1['open'], candle1['close'])
        )
        
        bearish_engulfing = (
            candle1['is_bullish'] and
            candle2['is_bearish'] and
            max(candle2['open'], candle2['close']) > max(candle1['open'], candle1['close']) and
            min(candle2['open'], candle2['close']) < min(candle1['open'], candle1['close'])
        )
        
        # Check for three outside up
        is_three_outside_up = (
            bullish_engulfing and
            candle3['is_bullish'] and
            candle3['close'] > candle2['close']
        )
        
        # Check for three outside down
        is_three_outside_down = (
            bearish_engulfing and
            candle3['is_bearish'] and
            candle3['close'] < candle2['close']
        )
        
        if not (is_three_outside_up or is_three_outside_down):
            return None
        
        if is_three_outside_up:
            pattern_type = CandlestickPatternType.THREE_OUTSIDE_UP
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.78
        else:
            pattern_type = CandlestickPatternType.THREE_OUTSIDE_DOWN
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.76
        
        confidence = self._calculate_three_pattern_confidence(
            candle1, candle2, candle3, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets
        pattern_height = max(candle1['total_range'], candle2['total_range'])
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, pattern_height
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 2,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.77),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'engulfing_strength': candle2['body_size'] / candle1['body_size'],
                'continuation_strength': abs(candle3['close'] - candle2['close']) / candle2['close'],
                'momentum_confirmed': candle3['close'] > candle2['close'] if is_three_outside_up else candle3['close'] < candle2['close']
            }
        )
    
    def _detect_advanced_patterns(self, data: pd.DataFrame, 
                                candle_data: pd.DataFrame) -> List[CandlestickDetection]:
        """Detect advanced multi-candlestick patterns."""
        patterns = []
        
        # These patterns require more complex analysis
        for i in range(4, len(data)):
            # Abandoned baby patterns
            abandoned_baby = self._analyze_abandoned_baby(data, candle_data, i)
            if abandoned_baby:
                patterns.append(abandoned_baby)
            
            # Rising/Falling three methods
            if i >= 6:  # Need at least 5 candles
                three_methods = self._analyze_three_methods(data, candle_data, i)
                if three_methods:
                    patterns.append(three_methods)
        
        return patterns
    
    def _analyze_abandoned_baby(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                              index: int) -> Optional[CandlestickDetection]:
        """Analyze abandoned baby pattern (rare but very reliable)."""
        
        if index < 2:
            return None
        
        candle1 = candle_data.iloc[index - 2]
        candle2 = candle_data.iloc[index - 1]  # Doji
        candle3 = candle_data.iloc[index]
        
        # Abandoned baby requires gaps and a doji in middle
        if not candle2['is_doji']:
            return None
        
        # Check for bullish abandoned baby
        gap_down = candle2['high'] < candle1['low']
        gap_up = candle2['low'] > candle3['high']
        
        is_bullish_abandoned = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            gap_down and gap_up and
            candle3['is_bullish'] and candle3['body_ratio'] >= 0.6 and
            self._analyze_trend_context(data, index) == 'downtrend'
        )
        
        # Check for bearish abandoned baby
        gap_up_1 = candle2['low'] > candle1['high']
        gap_down_2 = candle2['high'] < candle3['low']
        
        is_bearish_abandoned = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            gap_up_1 and gap_down_2 and
            candle3['is_bearish'] and candle3['body_ratio'] >= 0.6 and
            self._analyze_trend_context(data, index) == 'uptrend'
        )
        
        if not (is_bullish_abandoned or is_bearish_abandoned):
            return None
        
        if is_bullish_abandoned:
            pattern_type = CandlestickPatternType.ABANDONED_BABY_BULLISH
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.90  # Very high reliability
        else:
            pattern_type = CandlestickPatternType.ABANDONED_BABY_BEARISH
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.88
        
        confidence = self._calculate_three_pattern_confidence(
            candle1, candle2, candle3, base_confidence, data, index, pattern_type
        )
        
        # Calculate targets (more aggressive due to high reliability)
        pattern_height = abs(candle3['close'] - candle1['close'])
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, pattern_height * 1.5  # More aggressive target
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=index - 2,
            end_index=index,
            pattern_strength='strong',  # Always strong due to rarity and reliability
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=0.90,  # Very high
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'gap_size_1': abs(candle2['high'] - candle1['low']) / candle1['close'] if is_bullish_abandoned 
                             else abs(candle2['low'] - candle1['high']) / candle1['close'],
                'gap_size_2': abs(candle3['high'] - candle2['low']) / candle2['close'] if is_bullish_abandoned 
                             else abs(candle3['low'] - candle2['high']) / candle2['close'],
                'isolation_quality': min(gap_down, gap_up) if is_bullish_abandoned else min(gap_up_1, gap_down_2),
                'reversal_strength': abs(candle3['close'] - candle1['close']) / candle1['close']
            }
        )
    
    def _analyze_three_methods(self, data: pd.DataFrame, candle_data: pd.DataFrame, 
                             index: int) -> Optional[CandlestickDetection]:
        """Analyze rising/falling three methods pattern."""
        
        if index < 4:  # Need 5 candles minimum
            return None
        
        # Pattern structure: Large candle, 3 small contrary candles, large continuation candle
        candle1 = candle_data.iloc[index - 4]  # First large candle
        small_candles = candle_data.iloc[index - 3:index]  # Three small candles
        candle5 = candle_data.iloc[index]  # Final large candle
        
        # Check for rising three methods
        is_rising_methods = (
            candle1['is_bullish'] and candle1['body_ratio'] >= 0.6 and
            all(candle['is_bearish'] for _, candle in small_candles.iterrows()) and
            all(candle['body_ratio'] <= 0.4 for _, candle in small_candles.iterrows()) and
            all(candle['close'] > candle1['close'] * 0.95 for _, candle in small_candles.iterrows()) and  # Stay above first candle
            all(candle['open'] < candle1['close'] * 1.05 for _, candle in small_candles.iterrows()) and
            candle5['is_bullish'] and candle5['body_ratio'] >= 0.6 and
            candle5['close'] > candle1['close']
        )
        
        # Check for falling three methods
        is_falling_methods = (
            candle1['is_bearish'] and candle1['body_ratio'] >= 0.6 and
            all(candle['is_bullish'] for _, candle in small_candles.iterrows()) and
            all(candle['body_ratio'] <= 0.4 for _, candle in small_candles.iterrows()) and
            all(candle['close'] < candle1['close'] * 1.05 for _, candle in small_candles.iterrows()) and  # Stay below first candle
            all(candle['open'] > candle1['close'] * 0.95 for _, candle in small_candles.iterrows()) and
            candle5['is_bearish'] and candle5['body_ratio'] >= 0.6 and
            candle5['close'] < candle1['close']
        )
        
        if not (is_rising_methods or is_falling_methods):
            return None
        
        if is_rising_methods:
            pattern_type = CandlestickPatternType.RISING_THREE_METHODS
            direction = CandlestickDirection.BULLISH
            base_confidence = 0.80
        else:
            pattern_type = CandlestickPatternType.FALLING_THREE_METHODS
            direction = CandlestickDirection.BEARISH
            base_confidence = 0.78
        
        # Calculate confidence based on pattern quality
        continuation_strength = abs(candle5['close'] - candle1['close']) / candle1['close']
        consolidation_quality = 1 - (small_candles['body_size'].max() / candle1['body_size'])
        
        confidence = base_confidence + min(0.1, continuation_strength * 0.5) + min(0.1, consolidation_quality * 0.2)
        
        # Calculate targets
        pattern_height = abs(candle5['close'] - candle1['open'])
        target_price, stop_loss = self._calculate_three_pattern_targets(
            data, index, direction, pattern_height
        )
        
        return CandlestickDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=min(1.0, confidence),
            start_index=index - 4,
            end_index=index,
            pattern_strength=self._classify_pattern_strength(confidence),
            context_support=self._calculate_context_support(data, index, direction),
            volume_confirmation=self._check_volume_confirmation(candle_data, index, direction),
            trend_context=self._analyze_trend_context(data, index),
            reliability_score=self.pattern_reliability.get(pattern_type, 0.79),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=self._calculate_risk_reward(
                data['Close'].iloc[index], target_price, stop_loss
            ),
            metadata={
                'continuation_strength': continuation_strength,
                'consolidation_quality': consolidation_quality,
                'small_candles_count': len(small_candles),
                'pattern_duration': 5,
                'trend_consistency': direction.value == self._analyze_trend_context(data, index)
            }
        )
    
    # Helper methods for pattern analysis
    
    def _analyze_trend_context(self, data: pd.DataFrame, index: int) -> str:
        """Analyze the trend context around the pattern."""
        
        lookback = min(self.trend_lookback, index)
        if lookback < 3:
            return 'sideways'
        
        start_idx = index - lookback
        trend_data = data.iloc[start_idx:index + 1]
        
        # Simple trend analysis using close prices
        closes = trend_data['Close']
        start_price = closes.iloc[0]
        end_price = closes.iloc[-1]
        
        price_change = (end_price - start_price) / start_price
        
        # Calculate trend strength using linear regression
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Normalize slope by price level
        normalized_slope = slope / start_price
        
        if normalized_slope > 0.01:  # 1% upward slope
            return 'uptrend'
        elif normalized_slope < -0.01:  # 1% downward slope
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_single_pattern_confidence(self, candle: pd.Series, base_confidence: float, 
                                           trend: str, pattern_type: CandlestickPatternType) -> float:
        """Calculate confidence for single candlestick patterns."""
        
        confidence_adjustments = []
        
        # 1. Volume confirmation (+/- 10%)
        if candle['high_volume']:
            confidence_adjustments.append(0.1)
        else:
            confidence_adjustments.append(-0.05)
        
        # 2. Pattern quality based on proportions (+/- 15%)
        if pattern_type in [CandlestickPatternType.HAMMER, CandlestickPatternType.HANGING_MAN]:
            # Lower shadow should be at least 2x body size
            shadow_quality = min(0.15, candle['lower_shadow_ratio'] / candle['body_ratio'] * 0.05)
            confidence_adjustments.append(shadow_quality)
        elif pattern_type in [CandlestickPatternType.SHOOTING_STAR, CandlestickPatternType.INVERTED_HAMMER]:
            # Upper shadow should be at least 2x body size
            shadow_quality = min(0.15, candle['upper_shadow_ratio'] / candle['body_ratio'] * 0.05)
            confidence_adjustments.append(shadow_quality)
        
        # 3. Trend context alignment (+/- 10%)
        if pattern_type == CandlestickPatternType.HAMMER and trend == 'downtrend':
            confidence_adjustments.append(0.1)
        elif pattern_type == CandlestickPatternType.HANGING_MAN and trend == 'uptrend':
            confidence_adjustments.append(0.1)
        elif pattern_type == CandlestickPatternType.SHOOTING_STAR and trend == 'uptrend':
            confidence_adjustments.append(0.1)
        elif pattern_type == CandlestickPatternType.INVERTED_HAMMER and trend == 'downtrend':
            confidence_adjustments.append(0.1)
        else:
            confidence_adjustments.append(-0.05)
        
        # 4. Size significance (+/- 5%)
        if candle['total_range'] > candle.get('range_sma', candle['total_range']) * 1.5:
            confidence_adjustments.append(0.05)
        elif candle['total_range'] < candle.get('range_sma', candle['total_range']) * 0.5:
            confidence_adjustments.append(-0.05)
        
        final_confidence = base_confidence + sum(confidence_adjustments)
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_two_pattern_confidence(self, candle1: pd.Series, candle2: pd.Series, 
                                        base_confidence: float, data: pd.DataFrame, 
                                        index: int, pattern_type: CandlestickPatternType) -> float:
        """Calculate confidence for two-candlestick patterns."""
        
        confidence_adjustments = []
        
        # 1. Volume confirmation - second candle should have higher volume
        if candle2['volume_ratio'] > candle1['volume_ratio']:
            confidence_adjustments.append(0.1)
        else:
            confidence_adjustments.append(-0.05)
        
        # 2. Size relationship
        size_ratio = candle2['body_size'] / candle1['body_size'] if candle1['body_size'] > 0 else 1
        
        if pattern_type in [CandlestickPatternType.BULLISH_ENGULFING, CandlestickPatternType.BEARISH_ENGULFING]:
            # Engulfing patterns: larger second candle is better
            if size_ratio > 1.2:
                confidence_adjustments.append(0.1)
            elif size_ratio < 0.8:
                confidence_adjustments.append(-0.1)
        elif pattern_type in [CandlestickPatternType.BULLISH_HARAMI, CandlestickPatternType.BEARISH_HARAMI]:
            # Harami patterns: smaller second candle is better
            if size_ratio < 0.5:
                confidence_adjustments.append(0.1)
            elif size_ratio > 0.8:
                confidence_adjustments.append(-0.1)
        
        # 3. Trend context
        trend = self._analyze_trend_context(data, index)
        reversal_patterns = [
            CandlestickPatternType.BULLISH_ENGULFING, CandlestickPatternType.BEARISH_ENGULFING,
            CandlestickPatternType.BULLISH_HARAMI, CandlestickPatternType.BEARISH_HARAMI,
            CandlestickPatternType.PIERCING_PATTERN, CandlestickPatternType.DARK_CLOUD_COVER
        ]
        
        if pattern_type in reversal_patterns:
            if ((pattern_type in [CandlestickPatternType.BULLISH_ENGULFING, 
                                CandlestickPatternType.BULLISH_HARAMI, 
                                CandlestickPatternType.PIERCING_PATTERN] and trend == 'downtrend') or
                (pattern_type in [CandlestickPatternType.BEARISH_ENGULFING, 
                                CandlestickPatternType.BEARISH_HARAMI, 
                                CandlestickPatternType.DARK_CLOUD_COVER] and trend == 'uptrend')):
                confidence_adjustments.append(0.1)
            else:
                confidence_adjustments.append(-0.1)
        
        # 4. Gap between candles
        gap_size = abs(candle2['open'] - candle1['close']) / candle1['close']
        if gap_size > 0.02:  # 2% gap
            confidence_adjustments.append(0.05)
        
        final_confidence = base_confidence + sum(confidence_adjustments)
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_three_pattern_confidence(self, candle1: pd.Series, candle2: pd.Series, 
                                          candle3: pd.Series, base_confidence: float, 
                                          data: pd.DataFrame, index: int, 
                                          pattern_type: CandlestickPatternType) -> float:
        """Calculate confidence for three-candlestick patterns."""
        
        confidence_adjustments = []
        
        # 1. Volume progression - should increase on breakout candle
        volumes = [candle1['volume_ratio'], candle2['volume_ratio'], candle3['volume_ratio']]
        if candle3['volume_ratio'] > max(candle1['volume_ratio'], candle2['volume_ratio']):
            confidence_adjustments.append(0.1)
        elif candle3['volume_ratio'] < min(candle1['volume_ratio'], candle2['volume_ratio']):
            confidence_adjustments.append(-0.05)
        
        # 2. Pattern symmetry and progression
        if pattern_type in [CandlestickPatternType.MORNING_STAR, CandlestickPatternType.EVENING_STAR]:
            # Middle candle should be small
            middle_ratio = candle2['body_size'] / max(candle1['body_size'], candle3['body_size'])
            if middle_ratio < 0.3:
                confidence_adjustments.append(0.1)
            elif middle_ratio > 0.6:
                confidence_adjustments.append(-0.1)
        
        # 3. Trend context alignment
        trend = self._analyze_trend_context(data, index)
        bullish_patterns = [
            CandlestickPatternType.MORNING_STAR, CandlestickPatternType.MORNING_DOJI_STAR,
            CandlestickPatternType.THREE_WHITE_SOLDIERS, CandlestickPatternType.THREE_INSIDE_UP,
            CandlestickPatternType.THREE_OUTSIDE_UP
        ]
        bearish_patterns = [
            CandlestickPatternType.EVENING_STAR, CandlestickPatternType.EVENING_DOJI_STAR,
            CandlestickPatternType.THREE_BLACK_CROWS, CandlestickPatternType.THREE_INSIDE_DOWN,
            CandlestickPatternType.THREE_OUTSIDE_DOWN
        ]
        
        if ((pattern_type in bullish_patterns and trend == 'downtrend') or
            (pattern_type in bearish_patterns and trend == 'uptrend')):
            confidence_adjustments.append(0.1)
        elif ((pattern_type in bullish_patterns and trend == 'uptrend') or
              (pattern_type in bearish_patterns and trend == 'downtrend')):
            confidence_adjustments.append(-0.05)  # Continuation patterns get slight penalty
        
        # 4. Completion strength
        if pattern_type in bullish_patterns:
            completion_strength = (candle3['close'] - candle1['open']) / candle1['open']
        else:
            completion_strength = (candle1['open'] - candle3['close']) / candle1['open']
        
        if completion_strength > 0.05:  # 5% move
            confidence_adjustments.append(0.1)
        elif completion_strength < 0.02:  # Less than 2% move
            confidence_adjustments.append(-0.05)
        
        final_confidence = base_confidence + sum(confidence_adjustments)
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_context_support(self, data: pd.DataFrame, index: int, 
                                 direction: CandlestickDirection) -> float:
        """Calculate how well the pattern fits the market context."""
        
        support_factors = []
        
        # 1. Support/Resistance levels
        lookback = min(20, index)
        if lookback >= 10:
            recent_data = data.iloc[index - lookback:index + 1]
            current_price = data['Close'].iloc[index]
            
            # Check if we're near support (for bullish patterns) or resistance (for bearish patterns)
            if direction == CandlestickDirection.BULLISH:
                support_level = recent_data['Low'].min()
                distance_to_support = (current_price - support_level) / current_price
                if distance_to_support < 0.02:  # Within 2% of support
                    support_factors.append(0.3)
            elif direction == CandlestickDirection.BEARISH:
                resistance_level = recent_data['High'].max()
                distance_to_resistance = (resistance_level - current_price) / current_price
                if distance_to_resistance < 0.02:  # Within 2% of resistance
                    support_factors.append(0.3)
        
        # 2. Moving average context
        if len(data) >= 20:
            ma20 = data['Close'].rolling(20).mean().iloc[index]
            current_price = data['Close'].iloc[index]
            
            if direction == CandlestickDirection.BULLISH and current_price < ma20:
                support_factors.append(0.2)  # Bullish pattern below MA
            elif direction == CandlestickDirection.BEARISH and current_price > ma20:
                support_factors.append(0.2)  # Bearish pattern above MA
        
        # 3. Volatility context
        if len(data) >= 10:
            recent_ranges = data['High'].iloc[index-9:index+1] - data['Low'].iloc[index-9:index+1]
            avg_range = recent_ranges.mean()
            current_range = data['High'].iloc[index] - data['Low'].iloc[index]
            
            if current_range > avg_range * 1.5:  # High volatility day
                support_factors.append(0.2)
        
        return min(1.0, sum(support_factors))
    
    def _check_volume_confirmation(self, candle_data: pd.DataFrame, index: int, 
                                 direction: CandlestickDirection) -> bool:
        """Check if volume confirms the pattern."""
        
        if not self.volume_confirmation or 'volume_ratio' not in candle_data.columns:
            return False
        
        current_volume = candle_data['volume_ratio'].iloc[index]
        
        # Volume should be above average for significant patterns
        return current_volume >= 1.2
    
    def _check_gap_between_candles(self, candle1: pd.Series, candle2: pd.Series) -> bool:
        """Check if there's a gap between two candles."""
        
        gap_up = candle2['low'] > candle1['high']
        gap_down = candle2['high'] < candle1['low']
        
        return gap_up or gap_down
    
    def _classify_pattern_strength(self, confidence: float) -> str:
        """Classify pattern strength based on confidence score."""
        
        if confidence >= 0.8:
            return 'strong'
        elif confidence >= 0.6:
            return 'moderate'
        else:
            return 'weak'
    
    def _calculate_single_pattern_targets(self, data: pd.DataFrame, index: int, 
                                        direction: CandlestickDirection, 
                                        pattern_range: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss for single candlestick patterns."""
        
        if direction == CandlestickDirection.NEUTRAL:
            return None, None
        
        current_price = data['Close'].iloc[index]
        
        if direction == CandlestickDirection.BULLISH:
            target_price = current_price + pattern_range * 1.5
            stop_loss = data['Low'].iloc[index] - pattern_range * 0.2
        else:  # BEARISH
            target_price = current_price - pattern_range * 1.5
            stop_loss = data['High'].iloc[index] + pattern_range * 0.2
        
        return target_price, stop_loss
    
    def _calculate_two_pattern_targets(self, data: pd.DataFrame, index: int, 
                                     direction: CandlestickDirection, 
                                     pattern_range: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss for two-candlestick patterns."""
        
        current_price = data['Close'].iloc[index]
        
        if direction == CandlestickDirection.BULLISH:
            target_price = current_price + pattern_range * 2.0
            stop_loss = min(data['Low'].iloc[index-1], data['Low'].iloc[index]) - pattern_range * 0.1
        else:  # BEARISH
            target_price = current_price - pattern_range * 2.0
            stop_loss = max(data['High'].iloc[index-1], data['High'].iloc[index]) + pattern_range * 0.1
        
        return target_price, stop_loss
    
    def _calculate_three_pattern_targets(self, data: pd.DataFrame, index: int, 
                                       direction: CandlestickDirection, 
                                       pattern_range: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss for three-candlestick patterns."""
        
        current_price = data['Close'].iloc[index]
        
        if direction == CandlestickDirection.BULLISH:
            target_price = current_price + pattern_range * 2.5
            stop_loss = data['Low'].iloc[index-2:index+1].min() - pattern_range * 0.1
        else:  # BEARISH
            target_price = current_price - pattern_range * 2.5
            stop_loss = data['High'].iloc[index-2:index+1].max() + pattern_range * 0.1
        
        return target_price, stop_loss
    
    def _calculate_risk_reward(self, entry_price: float, target_price: Optional[float], 
                             stop_loss: Optional[float]) -> Optional[float]:
        """Calculate risk-reward ratio."""
        
        if target_price is None or stop_loss is None:
            return None
        
        reward = abs(target_price - entry_price)
        risk = abs(entry_price - stop_loss)
        
        if risk == 0:
            return None
        
        return reward / risk
    
    def _enhance_pattern_with_context(self, pattern: CandlestickDetection, 
                                    data: pd.DataFrame, candle_data: pd.DataFrame):
        """Enhance pattern detection with additional context and validation."""
        
        # Add market context
        index = pattern.end_index
        
        # Check for confluence with technical levels
        if len(data) >= 50:
            # Simple support/resistance check
            lookback_data = data.iloc[max(0, index-50):index]
            current_price = data['Close'].iloc[index]
            
            # Find nearby support/resistance levels
            resistance_levels = []
            support_levels = []
            
            for i in range(len(lookback_data) - 2):
                if (lookback_data['High'].iloc[i] < lookback_data['High'].iloc[i+1] > 
                    lookback_data['High'].iloc[i+2]):
                    resistance_levels.append(lookback_data['High'].iloc[i+1])
                
                if (lookback_data['Low'].iloc[i] > lookback_data['Low'].iloc[i+1] < 
                    lookback_data['Low'].iloc[i+2]):
                    support_levels.append(lookback_data['Low'].iloc[i+1])
            
            # Check for confluence
            confluence_bonus = 0
            for level in resistance_levels + support_levels:
                if abs(current_price - level) / current_price < 0.01:  # Within 1%
                    confluence_bonus += 0.05
            
            pattern.confidence = min(1.0, pattern.confidence + min(0.15, confluence_bonus))
        
        # Update pattern strength classification
        pattern.pattern_strength = self._classify_pattern_strength(pattern.confidence)

class CandlestickScanner:
    """Scanner for finding candlestick patterns across multiple symbols."""
    
    def __init__(self, analyzer: CandlestickAnalyzer):
        """Initialize scanner with a candlestick analyzer."""
        self.analyzer = analyzer
        self.scan_results = []
    
    def scan_symbols(self, symbol_data: Dict[str, pd.DataFrame], 
                    min_confidence: float = 0.5) -> Dict[str, List[CandlestickDetection]]:
        """
        Scan multiple symbols for candlestick patterns.
        
        Args:
            symbol_data: Dictionary of symbol -> OHLCV data
            min_confidence: Minimum confidence threshold for patterns
            
        Returns:
            Dictionary of symbol -> list of detected patterns
        """
        results = {}
        
        for symbol, data in symbol_data.items():
            try:
                patterns = self.analyzer.detect_patterns(data, symbol)
                # Filter by confidence
                filtered_patterns = [p for p in patterns if p.confidence >= min_confidence]
                results[symbol] = filtered_patterns
                
                # Store for historical analysis
                for pattern in filtered_patterns:
                    self.scan_results.append({
                        'symbol': symbol,
                        'pattern': pattern,
                        'scan_time': datetime.now()
                    })
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                results[symbol] = []
        
        return results
    
    def get_top_patterns(self, n: int = 10) -> List[Dict]:
        """Get top N patterns by confidence across all symbols."""
        
        if not self.scan_results:
            return []
        
        # Sort by confidence and return top N
        sorted_results = sorted(self.scan_results, 
                              key=lambda x: x['pattern'].confidence, reverse=True)
        
        return sorted_results[:n]
    
    def get_patterns_by_type(self, pattern_type: CandlestickPatternType) -> List[Dict]:
        """Get all patterns of a specific type."""
        
        return [result for result in self.scan_results 
                if result['pattern'].pattern_type == pattern_type]
    
    def clear_results(self):
        """Clear scan results."""
        self.scan_results = []