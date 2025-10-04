"""Chart pattern recognition system for technical analysis.

This module implements detection algorithms for classic chart patterns including:
- Head and Shoulders (and inverse)
- Triangle patterns (Ascending, Descending, Symmetrical)
- Flag and Pennant patterns
- Wedge patterns (Rising, Falling)
- Double Top/Bottom patterns
- Rectangle/Channel patterns

Each pattern includes confidence scoring and historical validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt
import warnings

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Enumeration of supported chart patterns."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    RECTANGLE = "rectangle"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"

class PatternDirection(Enum):
    """Pattern direction classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class PatternPoint:
    """Represents a significant point in a pattern."""
    index: int
    price: float
    timestamp: pd.Timestamp
    point_type: str  # 'peak', 'trough', 'support', 'resistance'

@dataclass
class PatternDetection:
    """Complete pattern detection result."""
    pattern_type: PatternType
    direction: PatternDirection
    confidence: float
    start_index: int
    end_index: int
    key_points: List[PatternPoint]
    support_level: Optional[float]
    resistance_level: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    pattern_height: float
    formation_time: int  # Number of periods
    breakout_confirmation: bool
    volume_confirmation: bool
    metadata: Dict

class ChartPatternDetector:
    """Main class for detecting chart patterns in price data."""
    
    def __init__(self, 
                 min_pattern_length: int = 20,
                 max_pattern_length: int = 200,
                 peak_distance: int = 5,
                 price_tolerance: float = 0.02,
                 volume_confirmation: bool = True):
        """
        Initialize pattern detector.
        
        Args:
            min_pattern_length: Minimum number of periods for pattern formation
            max_pattern_length: Maximum number of periods for pattern formation
            peak_distance: Minimum distance between peaks/troughs
            price_tolerance: Price tolerance for pattern validation (as percentage)
            volume_confirmation: Whether to require volume confirmation
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.peak_distance = peak_distance
        self.price_tolerance = price_tolerance
        self.volume_confirmation = volume_confirmation
        
        # Pattern-specific parameters
        self.head_shoulders_ratio = 0.05  # 5% difference between shoulders
        self.triangle_slope_tolerance = 0.01
        self.flag_retrace_min = 0.382  # Minimum 38.2% retracement
        self.flag_retrace_max = 0.618  # Maximum 61.8% retracement
        
    def detect_patterns(self, data: pd.DataFrame, 
                       timeframe: str = "1D") -> List[PatternDetection]:
        """
        Detect all patterns in the given price data.
        
        Args:
            data: OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            timeframe: Timeframe for pattern analysis
            
        Returns:
            List of detected patterns with confidence scores
        """
        if len(data) < self.min_pattern_length:
            logger.warning(f"Insufficient data for pattern detection: {len(data)} periods")
            return []
            
        patterns = []
        
        try:
            # Find significant peaks and troughs
            peaks, troughs = self._find_significant_points(data)
            
            if len(peaks) < 2 or len(troughs) < 2:
                logger.info("Insufficient peaks/troughs for pattern detection")
                return patterns
            
            # Detect different pattern types
            patterns.extend(self._detect_head_and_shoulders(data, peaks, troughs))
            patterns.extend(self._detect_double_patterns(data, peaks, troughs))
            patterns.extend(self._detect_triple_patterns(data, peaks, troughs))
            patterns.extend(self._detect_triangles(data, peaks, troughs))
            patterns.extend(self._detect_flags_and_pennants(data, peaks, troughs))
            patterns.extend(self._detect_wedges(data, peaks, troughs))
            patterns.extend(self._detect_rectangles(data, peaks, troughs))
            
            # Sort by confidence score
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            # Filter overlapping patterns (keep highest confidence)
            patterns = self._filter_overlapping_patterns(patterns)
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            
        return patterns
    
    def _find_significant_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs in price data."""
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # Find peaks and troughs using scipy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            peaks, _ = find_peaks(high_prices, distance=self.peak_distance)
            troughs, _ = find_peaks(-low_prices, distance=self.peak_distance)
        
        # Filter by significance (must be local extrema)
        significant_peaks = []
        significant_troughs = []
        
        for peak in peaks:
            if self._is_significant_peak(high_prices, peak):
                significant_peaks.append(peak)
                
        for trough in troughs:
            if self._is_significant_trough(low_prices, trough):
                significant_troughs.append(trough)
        
        return significant_peaks, significant_troughs
    
    def _is_significant_peak(self, prices: np.ndarray, index: int) -> bool:
        """Check if a peak is significant based on surrounding prices."""
        if index < 2 or index >= len(prices) - 2:
            return False
            
        current_price = prices[index]
        left_max = np.max(prices[max(0, index-5):index])
        right_max = np.max(prices[index+1:min(len(prices), index+6)])
        
        return current_price > left_max and current_price > right_max
    
    def _is_significant_trough(self, prices: np.ndarray, index: int) -> bool:
        """Check if a trough is significant based on surrounding prices."""
        if index < 2 or index >= len(prices) - 2:
            return False
            
        current_price = prices[index]
        left_min = np.min(prices[max(0, index-5):index])
        right_min = np.min(prices[index+1:min(len(prices), index+6)])
        
        return current_price < left_min and current_price < right_min
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame, 
                                 peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Head and Shoulders patterns."""
        patterns = []
        
        if len(peaks) < 3 or len(troughs) < 2:
            return patterns
            
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Find corresponding troughs
            left_trough = None
            right_trough = None
            
            for t in troughs:
                if left_shoulder < t < head and left_trough is None:
                    left_trough = t
                elif head < t < right_shoulder and right_trough is None:
                    right_trough = t
            
            if left_trough is None or right_trough is None:
                continue
                
            # Validate Head and Shoulders pattern
            pattern = self._validate_head_and_shoulders(
                data, left_shoulder, head, right_shoulder, left_trough, right_trough
            )
            
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _validate_head_and_shoulders(self, data: pd.DataFrame,
                                   left_shoulder: int, head: int, right_shoulder: int,
                                   left_trough: int, right_trough: int) -> Optional[PatternDetection]:
        """Validate and score Head and Shoulders pattern."""
        
        # Get price levels
        ls_price = data['High'].iloc[left_shoulder]
        head_price = data['High'].iloc[head]
        rs_price = data['High'].iloc[right_shoulder]
        lt_price = data['Low'].iloc[left_trough]
        rt_price = data['Low'].iloc[right_trough]
        
        # Check pattern criteria
        # 1. Head should be higher than both shoulders
        if not (head_price > ls_price and head_price > rs_price):
            return None
        
        # 2. Shoulders should be roughly equal (within tolerance)
        shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
        if shoulder_diff > self.head_shoulders_ratio:
            return None
        
        # 3. Neckline should be roughly horizontal
        neckline_slope = abs(lt_price - rt_price) / max(lt_price, rt_price)
        if neckline_slope > self.price_tolerance:
            return None
        
        # Calculate confidence score
        confidence = self._calculate_head_shoulders_confidence(
            data, left_shoulder, head, right_shoulder, left_trough, right_trough
        )
        
        # Create pattern points
        key_points = [
            PatternPoint(left_shoulder, ls_price, data.index[left_shoulder], 'peak'),
            PatternPoint(head, head_price, data.index[head], 'peak'),
            PatternPoint(right_shoulder, rs_price, data.index[right_shoulder], 'peak'),
            PatternPoint(left_trough, lt_price, data.index[left_trough], 'trough'),
            PatternPoint(right_trough, rt_price, data.index[right_trough], 'trough'),
        ]
        
        # Calculate targets
        neckline = (lt_price + rt_price) / 2
        pattern_height = head_price - neckline
        target_price = neckline - pattern_height
        stop_loss = head_price
        
        # Check for breakout
        breakout_confirmed = self._check_neckline_break(data, right_trough, neckline, 'down')
        
        # Volume confirmation
        volume_confirmed = self._check_volume_confirmation(
            data, left_shoulder, right_shoulder, 'bearish'
        )
        
        return PatternDetection(
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            start_index=left_shoulder,
            end_index=right_shoulder,
            key_points=key_points,
            support_level=neckline,
            resistance_level=head_price,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_height=pattern_height,
            formation_time=right_shoulder - left_shoulder,
            breakout_confirmation=breakout_confirmed,
            volume_confirmation=volume_confirmed,
            metadata={
                'neckline': neckline,
                'shoulder_symmetry': 1 - shoulder_diff,
                'head_prominence': (head_price - max(ls_price, rs_price)) / head_price
            }
        )
    
    def _calculate_head_shoulders_confidence(self, data: pd.DataFrame,
                                           left_shoulder: int, head: int, right_shoulder: int,
                                           left_trough: int, right_trough: int) -> float:
        """Calculate confidence score for Head and Shoulders pattern."""
        
        ls_price = data['High'].iloc[left_shoulder]
        head_price = data['High'].iloc[head]
        rs_price = data['High'].iloc[right_shoulder]
        lt_price = data['Low'].iloc[left_trough]
        rt_price = data['Low'].iloc[right_trough]
        
        confidence_factors = []
        
        # 1. Shoulder symmetry (0-30 points)
        shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
        symmetry_score = max(0, 30 * (1 - shoulder_diff / self.head_shoulders_ratio))
        confidence_factors.append(symmetry_score)
        
        # 2. Head prominence (0-25 points)
        head_prominence = (head_price - max(ls_price, rs_price)) / head_price
        prominence_score = min(25, 25 * head_prominence / 0.1)  # 10% prominence = full score
        confidence_factors.append(prominence_score)
        
        # 3. Neckline quality (0-20 points)
        neckline_slope = abs(lt_price - rt_price) / max(lt_price, rt_price)
        neckline_score = max(0, 20 * (1 - neckline_slope / self.price_tolerance))
        confidence_factors.append(neckline_score)
        
        # 4. Pattern duration (0-15 points)
        duration = right_shoulder - left_shoulder
        duration_score = min(15, 15 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        # 5. Volume confirmation (0-10 points)
        volume_score = 10 if self._check_volume_confirmation(
            data, left_shoulder, right_shoulder, 'bearish'
        ) else 0
        confidence_factors.append(volume_score)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100  # Normalize to 0-1
    
    def _detect_double_patterns(self, data: pd.DataFrame, 
                              peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []
        
        # Double Top detection
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                pattern = self._validate_double_top(data, peaks[i], peaks[i + 1], troughs)
                if pattern:
                    patterns.append(pattern)
        
        # Double Bottom detection  
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                pattern = self._validate_double_bottom(data, troughs[i], troughs[i + 1], peaks)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _validate_double_top(self, data: pd.DataFrame, peak1: int, peak2: int, 
                           troughs: List[int]) -> Optional[PatternDetection]:
        """Validate Double Top pattern."""
        
        # Find trough between peaks
        valley = None
        for t in troughs:
            if peak1 < t < peak2:
                valley = t
                break
        
        if valley is None:
            return None
        
        peak1_price = data['High'].iloc[peak1]
        peak2_price = data['High'].iloc[peak2]
        valley_price = data['Low'].iloc[valley]
        
        # Check if peaks are roughly equal
        peak_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
        if peak_diff > self.price_tolerance:
            return None
        
        # Check minimum retracement
        retracement = (max(peak1_price, peak2_price) - valley_price) / max(peak1_price, peak2_price)
        if retracement < 0.1:  # Minimum 10% retracement
            return None
        
        # Calculate confidence
        confidence = self._calculate_double_pattern_confidence(
            peak1_price, peak2_price, valley_price, peak2 - peak1, 'top'
        )
        
        # Create pattern
        avg_peak = (peak1_price + peak2_price) / 2
        pattern_height = avg_peak - valley_price
        target_price = valley_price - pattern_height
        
        key_points = [
            PatternPoint(peak1, peak1_price, data.index[peak1], 'peak'),
            PatternPoint(peak2, peak2_price, data.index[peak2], 'peak'),
            PatternPoint(valley, valley_price, data.index[valley], 'trough'),
        ]
        
        return PatternDetection(
            pattern_type=PatternType.DOUBLE_TOP,
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            start_index=peak1,
            end_index=peak2,
            key_points=key_points,
            support_level=valley_price,
            resistance_level=avg_peak,
            target_price=target_price,
            stop_loss=avg_peak,
            pattern_height=pattern_height,
            formation_time=peak2 - peak1,
            breakout_confirmation=self._check_neckline_break(data, peak2, valley_price, 'down'),
            volume_confirmation=self._check_volume_confirmation(data, peak1, peak2, 'bearish'),
            metadata={'peak_symmetry': 1 - peak_diff, 'retracement': retracement}
        )
    
    def _validate_double_bottom(self, data: pd.DataFrame, trough1: int, trough2: int, 
                              peaks: List[int]) -> Optional[PatternDetection]:
        """Validate Double Bottom pattern."""
        
        # Find peak between troughs
        peak = None
        for p in peaks:
            if trough1 < p < trough2:
                peak = p
                break
        
        if peak is None:
            return None
        
        trough1_price = data['Low'].iloc[trough1]
        trough2_price = data['Low'].iloc[trough2]
        peak_price = data['High'].iloc[peak]
        
        # Check if troughs are roughly equal
        trough_diff = abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)
        if trough_diff > self.price_tolerance:
            return None
        
        # Check minimum retracement
        retracement = (peak_price - min(trough1_price, trough2_price)) / peak_price
        if retracement < 0.1:  # Minimum 10% retracement
            return None
        
        # Calculate confidence
        confidence = self._calculate_double_pattern_confidence(
            trough1_price, trough2_price, peak_price, trough2 - trough1, 'bottom'
        )
        
        # Create pattern
        avg_trough = (trough1_price + trough2_price) / 2
        pattern_height = peak_price - avg_trough
        target_price = peak_price + pattern_height
        
        key_points = [
            PatternPoint(trough1, trough1_price, data.index[trough1], 'trough'),
            PatternPoint(trough2, trough2_price, data.index[trough2], 'trough'),
            PatternPoint(peak, peak_price, data.index[peak], 'peak'),
        ]
        
        return PatternDetection(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            direction=PatternDirection.BULLISH,
            confidence=confidence,
            start_index=trough1,
            end_index=trough2,
            key_points=key_points,
            support_level=avg_trough,
            resistance_level=peak_price,
            target_price=target_price,
            stop_loss=avg_trough,
            pattern_height=pattern_height,
            formation_time=trough2 - trough1,
            breakout_confirmation=self._check_neckline_break(data, trough2, peak_price, 'up'),
            volume_confirmation=self._check_volume_confirmation(data, trough1, trough2, 'bullish'),
            metadata={'trough_symmetry': 1 - trough_diff, 'retracement': retracement}
        )
    
    def _calculate_double_pattern_confidence(self, level1: float, level2: float, 
                                           middle: float, duration: int, pattern_type: str) -> float:
        """Calculate confidence for double patterns."""
        
        confidence_factors = []
        
        # 1. Level symmetry (0-40 points)
        level_diff = abs(level1 - level2) / max(level1, level2)
        symmetry_score = max(0, 40 * (1 - level_diff / self.price_tolerance))
        confidence_factors.append(symmetry_score)
        
        # 2. Retracement depth (0-30 points)
        if pattern_type == 'top':
            retracement = (max(level1, level2) - middle) / max(level1, level2)
        else:
            retracement = (middle - min(level1, level2)) / middle
        
        retracement_score = min(30, 30 * retracement / 0.3)  # 30% retracement = full score
        confidence_factors.append(retracement_score)
        
        # 3. Pattern duration (0-20 points)
        duration_score = min(20, 20 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        # 4. Base score (10 points for valid pattern)
        confidence_factors.append(10)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100
    
    def _detect_triple_patterns(self, data: pd.DataFrame, 
                              peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Triple Top and Triple Bottom patterns."""
        patterns = []
        
        # Triple Top detection
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                pattern = self._validate_triple_top(data, peaks[i:i+3], troughs)
                if pattern:
                    patterns.append(pattern)
        
        # Triple Bottom detection
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                pattern = self._validate_triple_bottom(data, troughs[i:i+3], peaks)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _validate_triple_top(self, data: pd.DataFrame, peaks: List[int], 
                           troughs: List[int]) -> Optional[PatternDetection]:
        """Validate Triple Top pattern."""
        
        if len(peaks) != 3:
            return None
        
        peak1, peak2, peak3 = peaks
        
        # Find valleys between peaks
        valley1 = valley2 = None
        for t in troughs:
            if peak1 < t < peak2 and valley1 is None:
                valley1 = t
            elif peak2 < t < peak3 and valley2 is None:
                valley2 = t
        
        if valley1 is None or valley2 is None:
            return None
        
        # Get prices
        p1 = data['High'].iloc[peak1]
        p2 = data['High'].iloc[peak2]
        p3 = data['High'].iloc[peak3]
        v1 = data['Low'].iloc[valley1]
        v2 = data['Low'].iloc[valley2]
        
        # Check if all peaks are roughly equal
        avg_peak = (p1 + p2 + p3) / 3
        max_deviation = max(abs(p1 - avg_peak), abs(p2 - avg_peak), abs(p3 - avg_peak))
        if max_deviation / avg_peak > self.price_tolerance:
            return None
        
        # Calculate confidence
        confidence = self._calculate_triple_pattern_confidence(
            [p1, p2, p3], [v1, v2], peak3 - peak1, 'top'
        )
        
        # Create pattern
        support_level = min(v1, v2)
        pattern_height = avg_peak - support_level
        target_price = support_level - pattern_height
        
        key_points = [
            PatternPoint(peak1, p1, data.index[peak1], 'peak'),
            PatternPoint(peak2, p2, data.index[peak2], 'peak'),
            PatternPoint(peak3, p3, data.index[peak3], 'peak'),
            PatternPoint(valley1, v1, data.index[valley1], 'trough'),
            PatternPoint(valley2, v2, data.index[valley2], 'trough'),
        ]
        
        return PatternDetection(
            pattern_type=PatternType.TRIPLE_TOP,
            direction=PatternDirection.BEARISH,
            confidence=confidence,
            start_index=peak1,
            end_index=peak3,
            key_points=key_points,
            support_level=support_level,
            resistance_level=avg_peak,
            target_price=target_price,
            stop_loss=avg_peak,
            pattern_height=pattern_height,
            formation_time=peak3 - peak1,
            breakout_confirmation=self._check_neckline_break(data, peak3, support_level, 'down'),
            volume_confirmation=self._check_volume_confirmation(data, peak1, peak3, 'bearish'),
            metadata={'peak_uniformity': 1 - (max_deviation / avg_peak)}
        )
    
    def _validate_triple_bottom(self, data: pd.DataFrame, troughs: List[int], 
                              peaks: List[int]) -> Optional[PatternDetection]:
        """Validate Triple Bottom pattern."""
        
        if len(troughs) != 3:
            return None
        
        trough1, trough2, trough3 = troughs
        
        # Find peaks between troughs
        peak1 = peak2 = None
        for p in peaks:
            if trough1 < p < trough2 and peak1 is None:
                peak1 = p
            elif trough2 < p < trough3 and peak2 is None:
                peak2 = p
        
        if peak1 is None or peak2 is None:
            return None
        
        # Get prices
        t1 = data['Low'].iloc[trough1]
        t2 = data['Low'].iloc[trough2]
        t3 = data['Low'].iloc[trough3]
        p1 = data['High'].iloc[peak1]
        p2 = data['High'].iloc[peak2]
        
        # Check if all troughs are roughly equal
        avg_trough = (t1 + t2 + t3) / 3
        max_deviation = max(abs(t1 - avg_trough), abs(t2 - avg_trough), abs(t3 - avg_trough))
        if max_deviation / avg_trough > self.price_tolerance:
            return None
        
        # Calculate confidence
        confidence = self._calculate_triple_pattern_confidence(
            [t1, t2, t3], [p1, p2], trough3 - trough1, 'bottom'
        )
        
        # Create pattern
        resistance_level = max(p1, p2)
        pattern_height = resistance_level - avg_trough
        target_price = resistance_level + pattern_height
        
        key_points = [
            PatternPoint(trough1, t1, data.index[trough1], 'trough'),
            PatternPoint(trough2, t2, data.index[trough2], 'trough'),
            PatternPoint(trough3, t3, data.index[trough3], 'trough'),
            PatternPoint(peak1, p1, data.index[peak1], 'peak'),
            PatternPoint(peak2, p2, data.index[peak2], 'peak'),
        ]
        
        return PatternDetection(
            pattern_type=PatternType.TRIPLE_BOTTOM,
            direction=PatternDirection.BULLISH,
            confidence=confidence,
            start_index=trough1,
            end_index=trough3,
            key_points=key_points,
            support_level=avg_trough,
            resistance_level=resistance_level,
            target_price=target_price,
            stop_loss=avg_trough,
            pattern_height=pattern_height,
            formation_time=trough3 - trough1,
            breakout_confirmation=self._check_neckline_break(data, trough3, resistance_level, 'up'),
            volume_confirmation=self._check_volume_confirmation(data, trough1, trough3, 'bullish'),
            metadata={'trough_uniformity': 1 - (max_deviation / avg_trough)}
        )
    
    def _calculate_triple_pattern_confidence(self, levels: List[float], middles: List[float], 
                                           duration: int, pattern_type: str) -> float:
        """Calculate confidence for triple patterns."""
        
        confidence_factors = []
        
        # 1. Level uniformity (0-45 points)
        avg_level = sum(levels) / len(levels)
        max_deviation = max(abs(level - avg_level) for level in levels)
        uniformity_score = max(0, 45 * (1 - max_deviation / (avg_level * self.price_tolerance)))
        confidence_factors.append(uniformity_score)
        
        # 2. Retracement quality (0-25 points)
        if pattern_type == 'top':
            avg_retracement = sum((max(levels) - middle) / max(levels) for middle in middles) / len(middles)
        else:
            avg_retracement = sum((middle - min(levels)) / middle for middle in middles) / len(middles)
        
        retracement_score = min(25, 25 * avg_retracement / 0.25)  # 25% retracement = full score
        confidence_factors.append(retracement_score)
        
        # 3. Pattern duration (0-20 points)
        duration_score = min(20, 20 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        # 4. Base score (10 points for valid pattern)
        confidence_factors.append(10)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100
    
    def _detect_triangles(self, data: pd.DataFrame, 
                         peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect triangle patterns (Ascending, Descending, Symmetrical)."""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Try different combinations of peaks and troughs for triangles
        for i in range(len(peaks) - 1):
            for j in range(len(troughs) - 1):
                # Check if peaks and troughs are interleaved properly for triangle
                if self._is_valid_triangle_sequence(peaks[i:i+2], troughs[j:j+2]):
                    triangle = self._analyze_triangle(data, peaks[i:i+2], troughs[j:j+2])
                    if triangle:
                        patterns.append(triangle)
        
        return patterns
    
    def _is_valid_triangle_sequence(self, peaks: List[int], troughs: List[int]) -> bool:
        """Check if peaks and troughs form a valid triangle sequence."""
        all_points = sorted(peaks + troughs)
        
        # Should have alternating peaks and troughs
        peak_positions = [all_points.index(p) for p in peaks]
        trough_positions = [all_points.index(t) for t in troughs]
        
        # Check if they alternate reasonably
        return len(set(peak_positions + trough_positions)) == 4
    
    def _analyze_triangle(self, data: pd.DataFrame, peaks: List[int], 
                         troughs: List[int]) -> Optional[PatternDetection]:
        """Analyze triangle pattern formation."""
        
        if len(peaks) != 2 or len(troughs) != 2:
            return None
        
        # Get price levels
        peak_prices = [data['High'].iloc[p] for p in peaks]
        trough_prices = [data['Low'].iloc[t] for t in troughs]
        
        # Calculate trend lines
        peak_slope = self._calculate_slope(peaks, peak_prices)
        trough_slope = self._calculate_slope(troughs, trough_prices)
        
        # Classify triangle type
        triangle_type = self._classify_triangle(peak_slope, trough_slope)
        
        if triangle_type is None:
            return None
        
        # Calculate confidence
        confidence = self._calculate_triangle_confidence(
            data, peaks, troughs, peak_slope, trough_slope, triangle_type
        )
        
        # Determine direction
        direction = self._get_triangle_direction(triangle_type, peak_slope, trough_slope)
        
        # Calculate targets and levels
        all_points = sorted(peaks + troughs)
        start_idx = min(all_points)
        end_idx = max(all_points)
        
        # Estimate breakout point and target
        apex_x = self._calculate_triangle_apex(peaks, troughs, peak_slope, trough_slope)
        current_resistance = self._get_resistance_at_point(peaks, peak_prices, peak_slope, end_idx)
        current_support = self._get_support_at_point(troughs, trough_prices, trough_slope, end_idx)
        
        pattern_height = abs(max(peak_prices + trough_prices) - min(peak_prices + trough_prices))
        
        if direction == PatternDirection.BULLISH:
            target_price = current_resistance + pattern_height * 0.75
            stop_loss = current_support
        else:
            target_price = current_support - pattern_height * 0.75
            stop_loss = current_resistance
        
        # Create key points
        key_points = []
        for i, peak in enumerate(peaks):
            key_points.append(PatternPoint(peak, peak_prices[i], data.index[peak], 'peak'))
        for i, trough in enumerate(troughs):
            key_points.append(PatternPoint(trough, trough_prices[i], data.index[trough], 'trough'))
        
        return PatternDetection(
            pattern_type=triangle_type,
            direction=direction,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            key_points=key_points,
            support_level=current_support,
            resistance_level=current_resistance,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_height=pattern_height,
            formation_time=end_idx - start_idx,
            breakout_confirmation=False,  # Would need real-time data to confirm
            volume_confirmation=self._check_triangle_volume(data, start_idx, end_idx),
            metadata={
                'peak_slope': peak_slope,
                'trough_slope': trough_slope,
                'apex_x': apex_x,
                'convergence': abs(current_resistance - current_support) / max(current_resistance, current_support)
            }
        )
    
    def _calculate_slope(self, x_points: List[int], y_points: List[float]) -> float:
        """Calculate slope of trend line through points."""
        if len(x_points) != len(y_points) or len(x_points) < 2:
            return 0.0
        
        n = len(x_points)
        sum_x = sum(x_points)
        sum_y = sum(y_points)
        sum_xy = sum(x * y for x, y in zip(x_points, y_points))
        sum_x2 = sum(x * x for x in x_points)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _classify_triangle(self, peak_slope: float, trough_slope: float) -> Optional[PatternType]:
        """Classify triangle type based on trend line slopes."""
        
        # Use slope tolerance to determine pattern type
        tolerance = self.triangle_slope_tolerance
        
        if abs(peak_slope) < tolerance and trough_slope > tolerance:
            return PatternType.ASCENDING_TRIANGLE
        elif peak_slope < -tolerance and abs(trough_slope) < tolerance:
            return PatternType.DESCENDING_TRIANGLE
        elif peak_slope < -tolerance and trough_slope > tolerance:
            # Check if slopes are converging
            if abs(abs(peak_slope) - abs(trough_slope)) / max(abs(peak_slope), abs(trough_slope)) < 0.5:
                return PatternType.SYMMETRICAL_TRIANGLE
        
        return None
    
    def _get_triangle_direction(self, triangle_type: PatternType, 
                              peak_slope: float, trough_slope: float) -> PatternDirection:
        """Determine expected breakout direction for triangle."""
        
        if triangle_type == PatternType.ASCENDING_TRIANGLE:
            return PatternDirection.BULLISH
        elif triangle_type == PatternType.DESCENDING_TRIANGLE:
            return PatternDirection.BEARISH
        else:  # Symmetrical triangle
            # Direction depends on prior trend or can be neutral
            return PatternDirection.NEUTRAL
    
    def _calculate_triangle_confidence(self, data: pd.DataFrame, peaks: List[int], troughs: List[int],
                                     peak_slope: float, trough_slope: float, 
                                     triangle_type: PatternType) -> float:
        """Calculate confidence score for triangle pattern."""
        
        confidence_factors = []
        
        # 1. Trend line quality (0-30 points each)
        peak_r2 = self._calculate_trendline_r2(peaks, [data['High'].iloc[p] for p in peaks])
        trough_r2 = self._calculate_trendline_r2(troughs, [data['Low'].iloc[t] for t in troughs])
        
        peak_quality = 30 * peak_r2
        trough_quality = 30 * trough_r2
        confidence_factors.extend([peak_quality, trough_quality])
        
        # 2. Convergence quality (0-25 points)
        current_resistance = self._get_resistance_at_point(peaks, [data['High'].iloc[p] for p in peaks], 
                                                         peak_slope, max(peaks + troughs))
        current_support = self._get_support_at_point(troughs, [data['Low'].iloc[t] for t in troughs], 
                                                   trough_slope, max(peaks + troughs))
        convergence = 1 - abs(current_resistance - current_support) / max(current_resistance, current_support)
        convergence_score = 25 * convergence
        confidence_factors.append(convergence_score)
        
        # 3. Pattern duration (0-15 points)
        duration = max(peaks + troughs) - min(peaks + troughs)
        duration_score = min(15, 15 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100
    
    def _calculate_trendline_r2(self, x_points: List[int], y_points: List[float]) -> float:
        """Calculate R-squared for trend line fit."""
        if len(x_points) < 2:
            return 0.0
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x_points, y_points)
            return r_value ** 2
        except:
            return 0.0
    
    def _get_resistance_at_point(self, x_points: List[int], y_points: List[float], 
                               slope: float, target_x: int) -> float:
        """Get resistance level at a specific point using trend line."""
        if not x_points or not y_points:
            return 0.0
        
        # Use first point and slope to calculate resistance at target_x
        x0, y0 = x_points[0], y_points[0]
        return y0 + slope * (target_x - x0)
    
    def _get_support_at_point(self, x_points: List[int], y_points: List[float], 
                            slope: float, target_x: int) -> float:
        """Get support level at a specific point using trend line."""
        if not x_points or not y_points:
            return 0.0
        
        # Use first point and slope to calculate support at target_x
        x0, y0 = x_points[0], y_points[0]
        return y0 + slope * (target_x - x0)
    
    def _calculate_triangle_apex(self, peaks: List[int], troughs: List[int],
                               peak_slope: float, trough_slope: float) -> float:
        """Calculate where triangle trend lines would converge (apex)."""
        
        if abs(peak_slope - trough_slope) < 1e-10:
            return float('inf')  # Parallel lines
        
        # Use first points of each trend line
        x1, y1 = peaks[0], 0  # We'll use relative positioning
        x2, y2 = troughs[0], 0
        
        # Find intersection point
        # y1 + peak_slope * (x - x1) = y2 + trough_slope * (x - x2)
        # Solving for x
        apex_x = (y2 - y1 + peak_slope * x1 - trough_slope * x2) / (peak_slope - trough_slope)
        
        return apex_x
    
    def _check_triangle_volume(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Check if volume confirms triangle pattern (should decrease during formation)."""
        
        if not self.volume_confirmation or 'Volume' not in data.columns:
            return False
        
        pattern_volume = data['Volume'].iloc[start_idx:end_idx + 1]
        
        if len(pattern_volume) < 3:
            return False
        
        # Check if volume generally decreases during pattern formation
        early_volume = pattern_volume.iloc[:len(pattern_volume)//3].mean()
        late_volume = pattern_volume.iloc[-len(pattern_volume)//3:].mean()
        
        return late_volume < early_volume * 0.8  # 20% decrease
    
    def _detect_flags_and_pennants(self, data: pd.DataFrame, 
                                  peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Flag and Pennant patterns."""
        patterns = []
        
        # Look for strong moves followed by consolidation
        for i in range(len(data) - self.min_pattern_length):
            # Check for strong preceding move
            lookback = min(20, i)
            if lookback < 10:
                continue
            
            prior_move = self._identify_strong_move(data, i - lookback, i)
            if prior_move is None:
                continue
            
            # Look for consolidation pattern after the move
            consolidation_end = min(i + self.max_pattern_length, len(data) - 1)
            flag_pattern = self._analyze_flag_pattern(
                data, i, consolidation_end, prior_move, peaks, troughs
            )
            
            if flag_pattern:
                patterns.append(flag_pattern)
        
        return patterns
    
    def _identify_strong_move(self, data: pd.DataFrame, start: int, end: int) -> Optional[Dict]:
        """Identify if there's a strong price move in the given period."""
        
        if end <= start:
            return None
        
        start_price = data['Close'].iloc[start]
        end_price = data['Close'].iloc[end]
        high_price = data['High'].iloc[start:end + 1].max()
        low_price = data['Low'].iloc[start:end + 1].min()
        
        # Calculate move magnitude
        if start_price == 0:
            return None
        
        price_change = (end_price - start_price) / start_price
        volatility = (high_price - low_price) / start_price
        
        # Criteria for "strong move"
        min_change = 0.05  # 5% minimum move
        min_volatility = 0.08  # 8% minimum volatility
        
        if abs(price_change) >= min_change and volatility >= min_volatility:
            return {
                'direction': 'up' if price_change > 0 else 'down',
                'magnitude': abs(price_change),
                'start_price': start_price,
                'end_price': end_price,
                'high': high_price,
                'low': low_price
            }
        
        return None
    
    def _analyze_flag_pattern(self, data: pd.DataFrame, start: int, end: int, 
                            prior_move: Dict, peaks: List[int], troughs: List[int]) -> Optional[PatternDetection]:
        """Analyze potential flag/pennant pattern."""
        
        if end - start < self.min_pattern_length:
            return None
        
        # Get consolidation data
        consolidation_data = data.iloc[start:end + 1]
        
        # Find peaks and troughs within consolidation
        local_peaks = [p for p in peaks if start <= p <= end]
        local_troughs = [t for t in troughs if start <= t <= end]
        
        if len(local_peaks) < 2 or len(local_troughs) < 2:
            return None
        
        # Analyze trend lines within consolidation
        peak_prices = [data['High'].iloc[p] for p in local_peaks]
        trough_prices = [data['Low'].iloc[t] for t in local_troughs]
        
        peak_slope = self._calculate_slope(local_peaks, peak_prices)
        trough_slope = self._calculate_slope(local_troughs, trough_prices)
        
        # Classify as flag or pennant
        pattern_type, confidence = self._classify_flag_pennant(
            prior_move, peak_slope, trough_slope, consolidation_data
        )
        
        if pattern_type is None:
            return None
        
        # Calculate retracement
        consolidation_high = consolidation_data['High'].max()
        consolidation_low = consolidation_data['Low'].min()
        
        if prior_move['direction'] == 'up':
            retracement = (prior_move['end_price'] - consolidation_low) / (prior_move['end_price'] - prior_move['start_price'])
        else:
            retracement = (consolidation_high - prior_move['end_price']) / (prior_move['start_price'] - prior_move['end_price'])
        
        # Check if retracement is within acceptable range
        if not (self.flag_retrace_min <= retracement <= self.flag_retrace_max):
            confidence *= 0.7  # Reduce confidence for poor retracement
        
        # Determine direction
        direction = PatternDirection.BULLISH if prior_move['direction'] == 'up' else PatternDirection.BEARISH
        
        # Calculate targets
        flagpole_height = abs(prior_move['end_price'] - prior_move['start_price'])
        if direction == PatternDirection.BULLISH:
            target_price = consolidation_high + flagpole_height
            stop_loss = consolidation_low
            resistance_level = consolidation_high
            support_level = consolidation_low
        else:
            target_price = consolidation_low - flagpole_height
            stop_loss = consolidation_high
            resistance_level = consolidation_high
            support_level = consolidation_low
        
        # Create key points
        key_points = []
        for i, peak in enumerate(local_peaks):
            key_points.append(PatternPoint(peak, peak_prices[i], data.index[peak], 'peak'))
        for i, trough in enumerate(local_troughs):
            key_points.append(PatternPoint(trough, trough_prices[i], data.index[trough], 'trough'))
        
        return PatternDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=start,
            end_index=end,
            key_points=key_points,
            support_level=support_level,
            resistance_level=resistance_level,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_height=flagpole_height,
            formation_time=end - start,
            breakout_confirmation=False,
            volume_confirmation=self._check_flag_volume(data, start, end),
            metadata={
                'flagpole_height': flagpole_height,
                'retracement': retracement,
                'prior_move': prior_move,
                'peak_slope': peak_slope,
                'trough_slope': trough_slope
            }
        )
    
    def _classify_flag_pennant(self, prior_move: Dict, peak_slope: float, 
                             trough_slope: float, consolidation_data: pd.DataFrame) -> Tuple[Optional[PatternType], float]:
        """Classify consolidation as flag or pennant and calculate confidence."""
        
        confidence_factors = []
        
        # Check if slopes are roughly parallel (flag) or converging (pennant)
        slope_diff = abs(abs(peak_slope) - abs(trough_slope))
        parallel_threshold = 0.001
        
        is_parallel = slope_diff < parallel_threshold
        
        # Check slope direction relative to prior move
        if prior_move['direction'] == 'up':
            # Bull flag should slope down slightly, bull pennant converges
            if is_parallel and peak_slope < 0 and trough_slope < 0:
                pattern_type = PatternType.BULL_FLAG
                confidence_factors.append(30)  # Good flag characteristics
            elif not is_parallel and peak_slope < 0 and trough_slope > 0:
                pattern_type = PatternType.BULL_PENNANT
                confidence_factors.append(25)  # Good pennant characteristics
            else:
                return None, 0.0
        else:
            # Bear flag should slope up slightly, bear pennant converges
            if is_parallel and peak_slope > 0 and trough_slope > 0:
                pattern_type = PatternType.BEAR_FLAG
                confidence_factors.append(30)
            elif not is_parallel and peak_slope < 0 and trough_slope > 0:
                pattern_type = PatternType.BEAR_PENNANT
                confidence_factors.append(25)
            else:
                return None, 0.0
        
        # Additional confidence factors
        
        # 1. Consolidation tightness (0-25 points)
        price_range = consolidation_data['High'].max() - consolidation_data['Low'].min()
        avg_price = (consolidation_data['High'].max() + consolidation_data['Low'].min()) / 2
        tightness = 1 - (price_range / avg_price) / 0.1  # 10% range = 0 points
        tightness_score = max(0, 25 * tightness)
        confidence_factors.append(tightness_score)
        
        # 2. Duration appropriateness (0-20 points)
        duration = len(consolidation_data)
        ideal_duration = 15  # Ideally 10-20 periods
        duration_score = max(0, 20 * (1 - abs(duration - ideal_duration) / ideal_duration))
        confidence_factors.append(duration_score)
        
        # 3. Volume pattern (0-15 points) - should decrease during consolidation
        if 'Volume' in consolidation_data.columns:
            early_volume = consolidation_data['Volume'].iloc[:len(consolidation_data)//3].mean()
            late_volume = consolidation_data['Volume'].iloc[-len(consolidation_data)//3:].mean()
            
            if late_volume < early_volume:
                volume_score = 15 * (1 - late_volume / early_volume)
            else:
                volume_score = 0
            confidence_factors.append(volume_score)
        
        # 4. Base score (10 points)
        confidence_factors.append(10)
        
        total_confidence = sum(confidence_factors)
        return pattern_type, min(100, total_confidence) / 100
    
    def _check_flag_volume(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Check volume confirmation for flag patterns."""
        
        if not self.volume_confirmation or 'Volume' not in data.columns:
            return False
        
        flag_volume = data['Volume'].iloc[start:end + 1]
        
        if len(flag_volume) < 3:
            return False
        
        # Volume should decrease during flag formation
        early_volume = flag_volume.iloc[:len(flag_volume)//3].mean()
        late_volume = flag_volume.iloc[-len(flag_volume)//3:].mean()
        
        return late_volume < early_volume * 0.8
    
    def _detect_wedges(self, data: pd.DataFrame, 
                      peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Rising and Falling Wedge patterns."""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Analyze different combinations for wedge patterns
        for i in range(len(peaks) - 1):
            for j in range(len(troughs) - 1):
                wedge = self._analyze_wedge_pattern(data, peaks[i:i+2], troughs[j:j+2])
                if wedge:
                    patterns.append(wedge)
        
        return patterns
    
    def _analyze_wedge_pattern(self, data: pd.DataFrame, peaks: List[int], 
                             troughs: List[int]) -> Optional[PatternDetection]:
        """Analyze potential wedge pattern."""
        
        if len(peaks) != 2 or len(troughs) != 2:
            return None
        
        # Ensure proper sequence
        all_points = sorted(peaks + troughs)
        if len(all_points) != 4:
            return None
        
        # Get prices and calculate slopes
        peak_prices = [data['High'].iloc[p] for p in peaks]
        trough_prices = [data['Low'].iloc[t] for t in troughs]
        
        peak_slope = self._calculate_slope(peaks, peak_prices)
        trough_slope = self._calculate_slope(troughs, trough_prices)
        
        # Classify wedge type
        wedge_type = self._classify_wedge(peak_slope, trough_slope)
        
        if wedge_type is None:
            return None
        
        # Calculate confidence
        confidence = self._calculate_wedge_confidence(data, peaks, troughs, peak_slope, trough_slope)
        
        # Determine direction (wedges are typically reversal patterns)
        if wedge_type == PatternType.RISING_WEDGE:
            direction = PatternDirection.BEARISH
        else:  # FALLING_WEDGE
            direction = PatternDirection.BULLISH
        
        # Calculate levels and targets
        start_idx = min(all_points)
        end_idx = max(all_points)
        
        current_resistance = self._get_resistance_at_point(peaks, peak_prices, peak_slope, end_idx)
        current_support = self._get_support_at_point(troughs, trough_prices, trough_slope, end_idx)
        
        pattern_height = abs(max(peak_prices + trough_prices) - min(peak_prices + trough_prices))
        
        if direction == PatternDirection.BULLISH:
            target_price = current_resistance + pattern_height * 0.8
            stop_loss = current_support
        else:
            target_price = current_support - pattern_height * 0.8
            stop_loss = current_resistance
        
        # Create key points
        key_points = []
        for i, peak in enumerate(peaks):
            key_points.append(PatternPoint(peak, peak_prices[i], data.index[peak], 'peak'))
        for i, trough in enumerate(troughs):
            key_points.append(PatternPoint(trough, trough_prices[i], data.index[trough], 'trough'))
        
        return PatternDetection(
            pattern_type=wedge_type,
            direction=direction,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            key_points=key_points,
            support_level=current_support,
            resistance_level=current_resistance,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_height=pattern_height,
            formation_time=end_idx - start_idx,
            breakout_confirmation=False,
            volume_confirmation=self._check_wedge_volume(data, start_idx, end_idx),
            metadata={
                'peak_slope': peak_slope,
                'trough_slope': trough_slope,
                'convergence_rate': abs(peak_slope - trough_slope)
            }
        )
    
    def _classify_wedge(self, peak_slope: float, trough_slope: float) -> Optional[PatternType]:
        """Classify wedge type based on slope directions."""
        
        # Rising wedge: both slopes positive, converging upward
        if peak_slope > 0 and trough_slope > 0 and trough_slope > peak_slope:
            return PatternType.RISING_WEDGE
        
        # Falling wedge: both slopes negative, converging downward
        if peak_slope < 0 and trough_slope < 0 and peak_slope < trough_slope:
            return PatternType.FALLING_WEDGE
        
        return None
    
    def _calculate_wedge_confidence(self, data: pd.DataFrame, peaks: List[int], troughs: List[int],
                                  peak_slope: float, trough_slope: float) -> float:
        """Calculate confidence score for wedge pattern."""
        
        confidence_factors = []
        
        # 1. Slope consistency (0-35 points)
        peak_r2 = self._calculate_trendline_r2(peaks, [data['High'].iloc[p] for p in peaks])
        trough_r2 = self._calculate_trendline_r2(troughs, [data['Low'].iloc[t] for t in troughs])
        
        slope_quality = 35 * (peak_r2 + trough_r2) / 2
        confidence_factors.append(slope_quality)
        
        # 2. Convergence rate (0-30 points)
        convergence_rate = abs(trough_slope - peak_slope)
        # Ideal convergence rate is moderate (not too fast, not too slow)
        ideal_rate = 0.005
        convergence_score = max(0, 30 * (1 - abs(convergence_rate - ideal_rate) / ideal_rate))
        confidence_factors.append(convergence_score)
        
        # 3. Volume pattern (0-20 points) - should decrease in wedges
        all_points = sorted(peaks + troughs)
        start_idx = min(all_points)
        end_idx = max(all_points)
        
        if self._check_wedge_volume(data, start_idx, end_idx):
            volume_score = 20
        else:
            volume_score = 5
        confidence_factors.append(volume_score)
        
        # 4. Duration (0-15 points)
        duration = end_idx - start_idx
        duration_score = min(15, 15 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100
    
    def _check_wedge_volume(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Check volume confirmation for wedge patterns."""
        
        if not self.volume_confirmation or 'Volume' not in data.columns:
            return False
        
        wedge_volume = data['Volume'].iloc[start_idx:end_idx + 1]
        
        if len(wedge_volume) < 3:
            return False
        
        # Volume should generally decrease during wedge formation
        early_volume = wedge_volume.iloc[:len(wedge_volume)//3].mean()
        late_volume = wedge_volume.iloc[-len(wedge_volume)//3:].mean()
        
        return late_volume < early_volume * 0.85
    
    def _detect_rectangles(self, data: pd.DataFrame, 
                          peaks: List[int], troughs: List[int]) -> List[PatternDetection]:
        """Detect Rectangle/Channel patterns."""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for horizontal support and resistance levels
        for i in range(len(peaks) - 1):
            for j in range(len(troughs) - 1):
                rectangle = self._analyze_rectangle_pattern(data, peaks[i:i+2], troughs[j:j+2])
                if rectangle:
                    patterns.append(rectangle)
        
        return patterns
    
    def _analyze_rectangle_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                 troughs: List[int]) -> Optional[PatternDetection]:
        """Analyze potential rectangle pattern."""
        
        if len(peaks) != 2 or len(troughs) != 2:
            return None
        
        # Get price levels
        peak_prices = [data['High'].iloc[p] for p in peaks]
        trough_prices = [data['Low'].iloc[t] for t in troughs]
        
        # Check if peaks are roughly horizontal (resistance)
        peak_diff = abs(peak_prices[0] - peak_prices[1]) / max(peak_prices)
        if peak_diff > self.price_tolerance:
            return None
        
        # Check if troughs are roughly horizontal (support)
        trough_diff = abs(trough_prices[0] - trough_prices[1]) / min(trough_prices)
        if trough_diff > self.price_tolerance:
            return None
        
        # Check for sufficient separation between support and resistance
        avg_resistance = sum(peak_prices) / len(peak_prices)
        avg_support = sum(trough_prices) / len(trough_prices)
        channel_width = (avg_resistance - avg_support) / avg_support
        
        if channel_width < 0.05:  # Minimum 5% channel width
            return None
        
        # Calculate confidence
        confidence = self._calculate_rectangle_confidence(
            data, peaks, troughs, peak_diff, trough_diff, channel_width
        )
        
        # Determine pattern type and direction
        all_points = sorted(peaks + troughs)
        start_idx = min(all_points)
        end_idx = max(all_points)
        
        # Check overall trend to classify as channel
        overall_slope = (data['Close'].iloc[end_idx] - data['Close'].iloc[start_idx]) / (end_idx - start_idx)
        
        if abs(overall_slope) < 0.001:
            pattern_type = PatternType.RECTANGLE
            direction = PatternDirection.NEUTRAL
        elif overall_slope > 0:
            pattern_type = PatternType.CHANNEL_UP
            direction = PatternDirection.BULLISH
        else:
            pattern_type = PatternType.CHANNEL_DOWN
            direction = PatternDirection.BEARISH
        
        # Calculate targets (breakout targets)
        channel_height = avg_resistance - avg_support
        if direction == PatternDirection.BULLISH or direction == PatternDirection.NEUTRAL:
            target_price = avg_resistance + channel_height
            stop_loss = avg_support
        else:
            target_price = avg_support - channel_height
            stop_loss = avg_resistance
        
        # Create key points
        key_points = []
        for i, peak in enumerate(peaks):
            key_points.append(PatternPoint(peak, peak_prices[i], data.index[peak], 'resistance'))
        for i, trough in enumerate(troughs):
            key_points.append(PatternPoint(trough, trough_prices[i], data.index[trough], 'support'))
        
        return PatternDetection(
            pattern_type=pattern_type,
            direction=direction,
            confidence=confidence,
            start_index=start_idx,
            end_index=end_idx,
            key_points=key_points,
            support_level=avg_support,
            resistance_level=avg_resistance,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_height=channel_height,
            formation_time=end_idx - start_idx,
            breakout_confirmation=False,
            volume_confirmation=self._check_rectangle_volume(data, start_idx, end_idx),
            metadata={
                'channel_width': channel_width,
                'resistance_consistency': 1 - peak_diff,
                'support_consistency': 1 - trough_diff,
                'touch_points': len(peaks) + len(troughs)
            }
        )
    
    def _calculate_rectangle_confidence(self, data: pd.DataFrame, peaks: List[int], troughs: List[int],
                                      peak_diff: float, trough_diff: float, channel_width: float) -> float:
        """Calculate confidence score for rectangle pattern."""
        
        confidence_factors = []
        
        # 1. Level consistency (0-40 points)
        resistance_consistency = max(0, 20 * (1 - peak_diff / self.price_tolerance))
        support_consistency = max(0, 20 * (1 - trough_diff / self.price_tolerance))
        confidence_factors.extend([resistance_consistency, support_consistency])
        
        # 2. Channel width (0-25 points)
        # Ideal channel width is 8-15%
        ideal_width = 0.1
        width_score = max(0, 25 * (1 - abs(channel_width - ideal_width) / ideal_width))
        confidence_factors.append(width_score)
        
        # 3. Number of touches (0-20 points)
        total_touches = len(peaks) + len(troughs)
        touch_score = min(20, 5 * total_touches)  # 5 points per touch, max 20
        confidence_factors.append(touch_score)
        
        # 4. Duration (0-15 points)
        all_points = sorted(peaks + troughs)
        duration = max(all_points) - min(all_points)
        duration_score = min(15, 15 * (duration - self.min_pattern_length) / 
                           (self.max_pattern_length - self.min_pattern_length))
        confidence_factors.append(duration_score)
        
        total_confidence = sum(confidence_factors)
        return min(100, total_confidence) / 100
    
    def _check_rectangle_volume(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Check volume confirmation for rectangle patterns."""
        
        if not self.volume_confirmation or 'Volume' not in data.columns:
            return False
        
        # Volume pattern in rectangles can vary, but generally should be consistent
        pattern_volume = data['Volume'].iloc[start_idx:end_idx + 1]
        
        if len(pattern_volume) < 3:
            return False
        
        # Check for reasonable volume consistency (not too volatile)
        volume_std = pattern_volume.std()
        volume_mean = pattern_volume.mean()
        
        coefficient_of_variation = volume_std / volume_mean if volume_mean > 0 else float('inf')
        
        return coefficient_of_variation < 1.0  # Reasonable volume consistency
    
    def _check_neckline_break(self, data: pd.DataFrame, pattern_end: int, 
                            neckline: float, direction: str) -> bool:
        """Check if neckline has been broken after pattern completion."""
        
        # Look for breakout in subsequent periods
        lookforward = min(10, len(data) - pattern_end - 1)
        if lookforward <= 0:
            return False
        
        post_pattern = data.iloc[pattern_end + 1:pattern_end + 1 + lookforward]
        
        if direction == 'down':
            # Check for break below neckline
            return post_pattern['Low'].min() < neckline * 0.98  # 2% buffer
        else:
            # Check for break above neckline
            return post_pattern['High'].max() > neckline * 1.02  # 2% buffer
    
    def _check_volume_confirmation(self, data: pd.DataFrame, start: int, end: int, 
                                 expected_direction: str) -> bool:
        """Check volume confirmation for patterns."""
        
        if not self.volume_confirmation or 'Volume' not in data.columns:
            return False
        
        pattern_volume = data['Volume'].iloc[start:end + 1]
        
        if len(pattern_volume) < 3:
            return False
        
        # For bearish patterns, expect volume to increase on declines
        # For bullish patterns, expect volume to increase on advances
        
        if expected_direction == 'bearish':
            # Check if volume increases on down days
            down_days = data.iloc[start:end + 1]['Close'] < data.iloc[start:end + 1]['Open']
            if down_days.sum() == 0:
                return False
            
            down_volume = pattern_volume[down_days].mean()
            up_volume = pattern_volume[~down_days].mean()
            
            return down_volume > up_volume * 1.2  # 20% more volume on down days
        
        elif expected_direction == 'bullish':
            # Check if volume increases on up days
            up_days = data.iloc[start:end + 1]['Close'] > data.iloc[start:end + 1]['Open']
            if up_days.sum() == 0:
                return False
            
            up_volume = pattern_volume[up_days].mean()
            down_volume = pattern_volume[~up_days].mean()
            
            return up_volume > down_volume * 1.2  # 20% more volume on up days
        
        return False
    
    def _filter_overlapping_patterns(self, patterns: List[PatternDetection]) -> List[PatternDetection]:
        """Filter out overlapping patterns, keeping the highest confidence ones."""
        
        if len(patterns) <= 1:
            return patterns
        
        # Sort by confidence descending
        sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
        filtered_patterns = []
        
        for pattern in sorted_patterns:
            # Check if this pattern overlaps significantly with any already selected
            overlaps = False
            
            for selected in filtered_patterns:
                # Calculate overlap
                overlap_start = max(pattern.start_index, selected.start_index)
                overlap_end = min(pattern.end_index, selected.end_index)
                
                if overlap_end > overlap_start:
                    overlap_length = overlap_end - overlap_start
                    pattern_length = pattern.end_index - pattern.start_index
                    selected_length = selected.end_index - selected.start_index
                    
                    # If overlap is more than 50% of either pattern, consider it overlapping
                    overlap_ratio = overlap_length / min(pattern_length, selected_length)
                    if overlap_ratio > 0.5:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_patterns.append(pattern)
        
        return filtered_patterns

class PatternValidator:
    """Validates pattern detections and calculates historical success rates."""
    
    def __init__(self, lookforward_periods: int = 20):
        """
        Initialize pattern validator.
        
        Args:
            lookforward_periods: Number of periods to look forward for validation
        """
        self.lookforward_periods = lookforward_periods
        self.validation_history = []
    
    def validate_pattern(self, pattern: PatternDetection, data: pd.DataFrame) -> Dict:
        """
        Validate a pattern detection against subsequent price action.
        
        Args:
            pattern: Pattern detection to validate
            data: Complete price data including periods after pattern
            
        Returns:
            Validation results with success metrics
        """
        validation_result = {
            'pattern_id': id(pattern),
            'pattern_type': pattern.pattern_type.value,
            'confidence': pattern.confidence,
            'target_reached': False,
            'stop_loss_hit': False,
            'max_favorable_move': 0.0,
            'max_adverse_move': 0.0,
            'actual_outcome': 'pending',
            'days_to_target': None,
            'days_to_stop': None,
            'success_score': 0.0
        }
        
        # Get post-pattern data
        pattern_end = pattern.end_index
        lookforward_end = min(pattern_end + self.lookforward_periods, len(data) - 1)
        
        if lookforward_end <= pattern_end:
            validation_result['actual_outcome'] = 'insufficient_data'
            return validation_result
        
        post_data = data.iloc[pattern_end:lookforward_end + 1]
        entry_price = data['Close'].iloc[pattern_end]
        
        # Track price movement
        if pattern.direction == PatternDirection.BULLISH:
            # For bullish patterns
            max_high = post_data['High'].max()
            min_low = post_data['Low'].min()
            
            validation_result['max_favorable_move'] = (max_high - entry_price) / entry_price
            validation_result['max_adverse_move'] = (entry_price - min_low) / entry_price
            
            # Check target and stop loss
            if pattern.target_price and max_high >= pattern.target_price:
                validation_result['target_reached'] = True
                # Find when target was reached
                target_reached_idx = post_data[post_data['High'] >= pattern.target_price].index[0]
                validation_result['days_to_target'] = data.index.get_loc(target_reached_idx) - pattern_end
            
            if pattern.stop_loss and min_low <= pattern.stop_loss:
                validation_result['stop_loss_hit'] = True
                stop_hit_idx = post_data[post_data['Low'] <= pattern.stop_loss].index[0]
                validation_result['days_to_stop'] = data.index.get_loc(stop_hit_idx) - pattern_end
        
        elif pattern.direction == PatternDirection.BEARISH:
            # For bearish patterns
            max_high = post_data['High'].max()
            min_low = post_data['Low'].min()
            
            validation_result['max_favorable_move'] = (entry_price - min_low) / entry_price
            validation_result['max_adverse_move'] = (max_high - entry_price) / entry_price
            
            # Check target and stop loss
            if pattern.target_price and min_low <= pattern.target_price:
                validation_result['target_reached'] = True
                target_reached_idx = post_data[post_data['Low'] <= pattern.target_price].index[0]
                validation_result['days_to_target'] = data.index.get_loc(target_reached_idx) - pattern_end
            
            if pattern.stop_loss and max_high >= pattern.stop_loss:
                validation_result['stop_loss_hit'] = True
                stop_hit_idx = post_data[post_data['High'] >= pattern.stop_loss].index[0]
                validation_result['days_to_stop'] = data.index.get_loc(stop_hit_idx) - pattern_end
        
        # Determine outcome
        if validation_result['target_reached'] and not validation_result['stop_loss_hit']:
            validation_result['actual_outcome'] = 'success'
            validation_result['success_score'] = 1.0
        elif validation_result['stop_loss_hit'] and not validation_result['target_reached']:
            validation_result['actual_outcome'] = 'failure'
            validation_result['success_score'] = 0.0
        elif validation_result['target_reached'] and validation_result['stop_loss_hit']:
            # Both hit - check which came first
            if validation_result['days_to_target'] < validation_result['days_to_stop']:
                validation_result['actual_outcome'] = 'success'
                validation_result['success_score'] = 0.8  # Slightly lower for hitting stop after
            else:
                validation_result['actual_outcome'] = 'failure'
                validation_result['success_score'] = 0.2  # Some credit for eventually reaching target
        else:
            # Neither hit - evaluate based on favorable movement
            max_favorable = validation_result['max_favorable_move']
            max_adverse = validation_result['max_adverse_move']
            
            if max_favorable > max_adverse * 1.5:
                validation_result['actual_outcome'] = 'partial_success'
                validation_result['success_score'] = 0.6
            elif max_adverse > max_favorable * 1.5:
                validation_result['actual_outcome'] = 'partial_failure'
                validation_result['success_score'] = 0.3
            else:
                validation_result['actual_outcome'] = 'neutral'
                validation_result['success_score'] = 0.5
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def get_pattern_statistics(self, pattern_type: PatternType = None) -> Dict:
        """
        Get historical statistics for pattern performance.
        
        Args:
            pattern_type: Specific pattern type to analyze (None for all)
            
        Returns:
            Statistical summary of pattern performance
        """
        if not self.validation_history:
            return {'error': 'No validation history available'}
        
        # Filter by pattern type if specified
        if pattern_type:
            filtered_history = [v for v in self.validation_history 
                              if v['pattern_type'] == pattern_type.value]
        else:
            filtered_history = self.validation_history
        
        if not filtered_history:
            return {'error': f'No data for pattern type {pattern_type}'}
        
        # Calculate statistics
        total_patterns = len(filtered_history)
        successful_patterns = len([v for v in filtered_history if v['actual_outcome'] == 'success'])
        failed_patterns = len([v for v in filtered_history if v['actual_outcome'] == 'failure'])
        
        success_rate = successful_patterns / total_patterns
        failure_rate = failed_patterns / total_patterns
        
        avg_success_score = sum(v['success_score'] for v in filtered_history) / total_patterns
        avg_confidence = sum(v['confidence'] for v in filtered_history) / total_patterns
        
        # Target/stop statistics
        targets_reached = len([v for v in filtered_history if v['target_reached']])
        stops_hit = len([v for v in filtered_history if v['stop_loss_hit']])
        
        target_reach_rate = targets_reached / total_patterns
        stop_hit_rate = stops_hit / total_patterns
        
        # Movement statistics
        avg_favorable_move = sum(v['max_favorable_move'] for v in filtered_history) / total_patterns
        avg_adverse_move = sum(v['max_adverse_move'] for v in filtered_history) / total_patterns
        
        # Timing statistics
        days_to_target_list = [v['days_to_target'] for v in filtered_history if v['days_to_target'] is not None]
        days_to_stop_list = [v['days_to_stop'] for v in filtered_history if v['days_to_stop'] is not None]
        
        avg_days_to_target = sum(days_to_target_list) / len(days_to_target_list) if days_to_target_list else None
        avg_days_to_stop = sum(days_to_stop_list) / len(days_to_stop_list) if days_to_stop_list else None
        
        return {
            'pattern_type': pattern_type.value if pattern_type else 'all',
            'total_patterns': total_patterns,
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'avg_success_score': avg_success_score,
            'avg_confidence': avg_confidence,
            'target_reach_rate': target_reach_rate,
            'stop_hit_rate': stops_hit / total_patterns,
            'avg_favorable_move': avg_favorable_move,
            'avg_adverse_move': avg_adverse_move,
            'avg_days_to_target': avg_days_to_target,
            'avg_days_to_stop': avg_days_to_stop,
            'outcome_distribution': {
                'success': successful_patterns,
                'failure': failed_patterns,
                'partial_success': len([v for v in filtered_history if v['actual_outcome'] == 'partial_success']),
                'partial_failure': len([v for v in filtered_history if v['actual_outcome'] == 'partial_failure']),
                'neutral': len([v for v in filtered_history if v['actual_outcome'] == 'neutral']),
                'pending': len([v for v in filtered_history if v['actual_outcome'] == 'pending'])
            }
        }
    
    def get_confidence_calibration(self) -> Dict:
        """
        Analyze how well pattern confidence scores correlate with actual success.
        
        Returns:
            Confidence calibration analysis
        """
        if not self.validation_history:
            return {'error': 'No validation history available'}
        
        # Group patterns by confidence ranges
        confidence_ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibration_data = {}
        
        for i in range(len(confidence_ranges) - 1):
            range_min = confidence_ranges[i]
            range_max = confidence_ranges[i + 1]
            range_name = f"{range_min:.1f}-{range_max:.1f}"
            
            range_patterns = [v for v in self.validation_history 
                            if range_min <= v['confidence'] < range_max]
            
            if range_patterns:
                avg_success_score = sum(v['success_score'] for v in range_patterns) / len(range_patterns)
                avg_confidence = sum(v['confidence'] for v in range_patterns) / len(range_patterns)
                
                calibration_data[range_name] = {
                    'count': len(range_patterns),
                    'avg_confidence': avg_confidence,
                    'avg_success_score': avg_success_score,
                    'calibration_error': abs(avg_confidence - avg_success_score)
                }
        
        # Calculate overall calibration error
        total_error = sum(data['calibration_error'] * data['count'] 
                         for data in calibration_data.values())
        total_count = sum(data['count'] for data in calibration_data.values())
        overall_calibration_error = total_error / total_count if total_count > 0 else 0
        
        return {
            'calibration_by_range': calibration_data,
            'overall_calibration_error': overall_calibration_error,
            'total_patterns_analyzed': total_count
        }