"""Pattern performance tracking and confidence calibration system.

This module provides comprehensive tracking of pattern detection performance including:
- Historical success rate calculation
- Confidence score calibration
- Pattern reliability scoring
- Performance analytics and reporting
- Real-time pattern alerts and notifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .pattern_recognition import PatternDetection, PatternType, PatternDirection
from .candlestick_patterns import CandlestickDetection, CandlestickPatternType, CandlestickDirection

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of pattern alerts."""
    PATTERN_DETECTED = "pattern_detected"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    TARGET_REACHED = "target_reached"
    STOP_LOSS_HIT = "stop_loss_hit"
    CONFLUENCE_DETECTED = "confluence_detected"

class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PatternAlert:
    """Pattern alert notification."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str
    confidence: float
    current_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    message: str
    metadata: Dict
    created_at: datetime
    sent: bool = False

@dataclass
class PatternPerformanceRecord:
    """Record of pattern performance for tracking."""
    pattern_id: str
    symbol: str
    pattern_type: str
    direction: str
    confidence: float
    detected_at: datetime
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    current_status: str  # 'active', 'target_hit', 'stop_hit', 'expired'
    outcome: Optional[str]  # 'success', 'failure', 'partial', None
    max_favorable_move: float = 0.0
    max_adverse_move: float = 0.0
    target_hit_date: Optional[datetime] = None
    stop_hit_date: Optional[datetime] = None
    final_price: Optional[float] = None
    roi: Optional[float] = None
    days_active: Optional[int] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PatternPerformanceTracker:
    """Tracks pattern performance and calculates success rates."""
    
    def __init__(self, lookback_days: int = 90, min_samples: int = 10):
        """
        Initialize performance tracker.
        
        Args:
            lookback_days: Days to look back for performance calculation
            min_samples: Minimum samples needed for reliable statistics
        """
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.performance_records: List[PatternPerformanceRecord] = []
        self.pattern_stats: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        # Cache for performance calculations
        self._stats_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = timedelta(hours=1)
    
    def add_pattern_detection(self, pattern: Union[PatternDetection, CandlestickDetection], 
                            symbol: str, current_price: float) -> str:
        """
        Add a new pattern detection for tracking.
        
        Args:
            pattern: Pattern detection result
            symbol: Symbol being tracked
            current_price: Current market price
            
        Returns:
            Pattern ID for tracking
        """
        pattern_id = f"{symbol}_{pattern.pattern_type.value}_{int(datetime.now().timestamp())}"
        
        # Extract common attributes regardless of pattern type
        if hasattr(pattern, 'pattern_type'):
            if isinstance(pattern.pattern_type, (PatternType, CandlestickPatternType)):
                pattern_type = pattern.pattern_type.value
            else:
                pattern_type = str(pattern.pattern_type)
        else:
            pattern_type = "unknown"
        
        if hasattr(pattern, 'direction'):
            if isinstance(pattern.direction, (PatternDirection, CandlestickDirection)):
                direction = pattern.direction.value
            else:
                direction = str(pattern.direction)
        else:
            direction = "neutral"
        
        record = PatternPerformanceRecord(
            pattern_id=pattern_id,
            symbol=symbol,
            pattern_type=pattern_type,
            direction=direction,
            confidence=pattern.confidence,
            detected_at=datetime.now(),
            entry_price=current_price,
            target_price=getattr(pattern, 'target_price', None),
            stop_loss=getattr(pattern, 'stop_loss', None),
            current_status='active',
            outcome=None,
            metadata={
                'pattern_data': asdict(pattern) if hasattr(pattern, '__dataclass_fields__') else str(pattern),
                'detection_method': type(pattern).__name__
            }
        )
        
        with self._lock:
            self.performance_records.append(record)
        
        logger.info(f"Added pattern tracking: {pattern_id} for {symbol}")
        return pattern_id
    
    def update_pattern_status(self, pattern_id: str, current_price: float, 
                            timestamp: Optional[datetime] = None) -> Optional[PatternPerformanceRecord]:
        """
        Update pattern status based on current price.
        
        Args:
            pattern_id: ID of pattern to update
            current_price: Current market price
            timestamp: Current timestamp (defaults to now)
            
        Returns:
            Updated pattern record if found
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            record = None
            for r in self.performance_records:
                if r.pattern_id == pattern_id:
                    record = r
                    break
            
            if not record or record.current_status != 'active':
                return None
            
            # Calculate moves
            if record.direction == 'bullish':
                favorable_move = (current_price - record.entry_price) / record.entry_price
                adverse_move = (record.entry_price - current_price) / record.entry_price if current_price < record.entry_price else 0
            elif record.direction == 'bearish':
                favorable_move = (record.entry_price - current_price) / record.entry_price
                adverse_move = (current_price - record.entry_price) / record.entry_price if current_price > record.entry_price else 0
            else:  # neutral
                favorable_move = abs(current_price - record.entry_price) / record.entry_price
                adverse_move = 0
            
            # Update max moves
            record.max_favorable_move = max(record.max_favorable_move, favorable_move)
            record.max_adverse_move = max(record.max_adverse_move, adverse_move)
            
            # Check for target/stop hits
            if record.target_price and not record.target_hit_date:
                if ((record.direction == 'bullish' and current_price >= record.target_price) or
                    (record.direction == 'bearish' and current_price <= record.target_price)):
                    record.target_hit_date = timestamp
                    record.current_status = 'target_hit'
                    record.outcome = 'success'
                    record.final_price = current_price
                    record.roi = favorable_move
            
            if record.stop_loss and not record.stop_hit_date:
                if ((record.direction == 'bullish' and current_price <= record.stop_loss) or
                    (record.direction == 'bearish' and current_price >= record.stop_loss)):
                    record.stop_hit_date = timestamp
                    if record.current_status != 'target_hit':  # Target wasn't hit first
                        record.current_status = 'stop_hit'
                        record.outcome = 'failure'
                        record.final_price = current_price
                        record.roi = -adverse_move
            
            # Check for expiration (patterns older than 30 days)
            if record.current_status == 'active' and (timestamp - record.detected_at).days > 30:
                record.current_status = 'expired'
                record.final_price = current_price
                record.roi = favorable_move if favorable_move > adverse_move else -adverse_move
                
                # Determine outcome based on movement
                if record.max_favorable_move > 0.05:  # 5% favorable move
                    record.outcome = 'partial'
                elif record.max_adverse_move > 0.05:  # 5% adverse move
                    record.outcome = 'failure'
                else:
                    record.outcome = 'neutral'
            
            # Update days active
            record.days_active = (timestamp - record.detected_at).days
            
        return record
    
    def calculate_pattern_statistics(self, pattern_type: Optional[str] = None, 
                                   min_confidence: Optional[float] = None) -> Dict:
        """
        Calculate performance statistics for patterns.
        
        Args:
            pattern_type: Specific pattern type to analyze (None for all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with performance statistics
        """
        # Check cache
        cache_key = f"{pattern_type}_{min_confidence}"
        if (self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl and
            cache_key in self._stats_cache):
            return self._stats_cache[cache_key]
        
        with self._lock:
            # Filter records
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            records = [r for r in self.performance_records 
                      if r.detected_at >= cutoff_date and r.outcome is not None]
            
            if pattern_type:
                records = [r for r in records if r.pattern_type == pattern_type]
            
            if min_confidence:
                records = [r for r in records if r.confidence >= min_confidence]
            
            if len(records) < self.min_samples:
                stats = {
                    'error': f'Insufficient samples: {len(records)} < {self.min_samples}',
                    'total_patterns': len(records)
                }
            else:
                stats = self._calculate_stats(records)
        
        # Cache result
        self._stats_cache[cache_key] = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def _calculate_stats(self, records: List[PatternPerformanceRecord]) -> Dict:
        """Calculate detailed statistics from performance records."""
        
        if not records:
            return {'error': 'No records provided'}
        
        total_patterns = len(records)
        successful = len([r for r in records if r.outcome == 'success'])
        failed = len([r for r in records if r.outcome == 'failure'])
        partial = len([r for r in records if r.outcome == 'partial'])
        neutral = len([r for r in records if r.outcome == 'neutral'])
        
        success_rate = successful / total_patterns
        failure_rate = failed / total_patterns
        
        # ROI statistics
        rois = [r.roi for r in records if r.roi is not None]
        avg_roi = np.mean(rois) if rois else 0
        median_roi = np.median(rois) if rois else 0
        
        # Winning/losing streaks
        outcomes = [r.outcome for r in sorted(records, key=lambda x: x.detected_at)]
        win_streak, loss_streak = self._calculate_streaks(outcomes)
        
        # Time to resolution
        resolved_records = [r for r in records if r.days_active is not None]
        avg_days_active = np.mean([r.days_active for r in resolved_records]) if resolved_records else 0
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(records)
        
        # Direction-specific stats
        bullish_records = [r for r in records if r.direction == 'bullish']
        bearish_records = [r for r in records if r.direction == 'bearish']
        
        bullish_success = len([r for r in bullish_records if r.outcome == 'success']) / len(bullish_records) if bullish_records else 0
        bearish_success = len([r for r in bearish_records if r.outcome == 'success']) / len(bearish_records) if bearish_records else 0
        
        # Risk-reward analysis
        successful_records = [r for r in records if r.outcome == 'success' and r.roi is not None]
        failed_records = [r for r in records if r.outcome == 'failure' and r.roi is not None]
        
        avg_win = np.mean([r.roi for r in successful_records]) if successful_records else 0
        avg_loss = np.mean([abs(r.roi) for r in failed_records]) if failed_records else 0
        
        profit_factor = (avg_win * successful) / (avg_loss * failed) if failed > 0 and avg_loss > 0 else float('inf')
        
        stats = {
            'total_patterns': total_patterns,
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'partial_rate': partial / total_patterns,
            'neutral_rate': neutral / total_patterns,
            
            'roi_stats': {
                'average_roi': avg_roi,
                'median_roi': median_roi,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor
            },
            
            'timing_stats': {
                'avg_days_active': avg_days_active,
                'max_win_streak': win_streak,
                'max_loss_streak': loss_streak
            },
            
            'direction_stats': {
                'bullish_success_rate': bullish_success,
                'bearish_success_rate': bearish_success,
                'bullish_count': len(bullish_records),
                'bearish_count': len(bearish_records)
            },
            
            'confidence_calibration': confidence_calibration,
            
            'outcome_distribution': {
                'success': successful,
                'failure': failed,
                'partial': partial,
                'neutral': neutral
            },
            
            'generated_at': datetime.now().isoformat(),
            'lookback_days': self.lookback_days
        }
        
        return stats
    
    def _calculate_streaks(self, outcomes: List[str]) -> Tuple[int, int]:
        """Calculate maximum winning and losing streaks."""
        if not outcomes:
            return 0, 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for outcome in outcomes:
            if outcome == 'success':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif outcome == 'failure':
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0
        
        return max_win_streak, max_loss_streak
    
    def _calculate_confidence_calibration(self, records: List[PatternPerformanceRecord]) -> Dict:
        """Calculate how well confidence scores correlate with actual success."""
        
        confidence_ranges = np.arange(0.0, 1.1, 0.1)
        calibration_data = {}
        
        for i in range(len(confidence_ranges) - 1):
            range_min = confidence_ranges[i]
            range_max = confidence_ranges[i + 1]
            
            range_records = [r for r in records 
                           if range_min <= r.confidence < range_max]
            
            if range_records:
                success_count = len([r for r in range_records if r.outcome == 'success'])
                actual_success_rate = success_count / len(range_records)
                avg_confidence = np.mean([r.confidence for r in range_records])
                
                calibration_error = abs(avg_confidence - actual_success_rate)
                
                calibration_data[f"{range_min:.1f}-{range_max:.1f}"] = {
                    'count': len(range_records),
                    'avg_confidence': avg_confidence,
                    'actual_success_rate': actual_success_rate,
                    'calibration_error': calibration_error
                }
        
        # Overall calibration error
        total_error = sum(data['calibration_error'] * data['count'] 
                         for data in calibration_data.values())
        total_count = sum(data['count'] for data in calibration_data.values())
        overall_error = total_error / total_count if total_count > 0 else 0
        
        return {
            'by_range': calibration_data,
            'overall_calibration_error': overall_error,
            'total_patterns': total_count
        }
    
    def get_pattern_reliability_score(self, pattern_type: str, confidence: float) -> float:
        """
        Get reliability score for a pattern type and confidence level.
        
        Args:
            pattern_type: Type of pattern
            confidence: Pattern confidence score
            
        Returns:
            Adjusted reliability score (0.0 to 1.0)
        """
        stats = self.calculate_pattern_statistics(pattern_type=pattern_type)
        
        if 'error' in stats:
            # Use default reliability if insufficient data
            default_scores = {
                'morning_star': 0.75,
                'evening_star': 0.73,
                'bullish_engulfing': 0.70,
                'bearish_engulfing': 0.68,
                'head_and_shoulders': 0.72,
                'inverse_head_and_shoulders': 0.74,
                'ascending_triangle': 0.69,
                'descending_triangle': 0.67,
                'bull_flag': 0.71,
                'bear_flag': 0.69
            }
            base_reliability = default_scores.get(pattern_type, 0.65)
        else:
            base_reliability = stats['success_rate']
        
        # Adjust based on confidence calibration
        if 'confidence_calibration' in stats and stats['confidence_calibration']['by_range']:
            # Find appropriate confidence range
            for range_key, range_data in stats['confidence_calibration']['by_range'].items():
                range_min = float(range_key.split('-')[0])
                range_max = float(range_key.split('-')[1])
                
                if range_min <= confidence < range_max:
                    # Adjust reliability based on calibration
                    calibration_factor = 1 - range_data['calibration_error']
                    base_reliability *= calibration_factor
                    break
        
        # Apply confidence multiplier
        confidence_multiplier = 0.8 + (confidence * 0.4)  # Scale from 0.8 to 1.2
        
        return min(1.0, base_reliability * confidence_multiplier)

class PatternAlertSystem:
    """Real-time pattern alert and notification system."""
    
    def __init__(self, performance_tracker: PatternPerformanceTracker):
        """Initialize alert system with performance tracker."""
        self.performance_tracker = performance_tracker
        self.active_alerts: List[PatternAlert] = []
        self.alert_queue = deque(maxlen=1000)
        self.notification_handlers = []
        self._alert_counter = 0
        self._lock = threading.Lock()
        
        # Alert configuration
        self.min_confidence_for_alert = 0.6
        self.min_reliability_for_high_priority = 0.75
        self.confluence_bonus = 0.15
        
    def add_notification_handler(self, handler):
        """Add a notification handler function."""
        self.notification_handlers.append(handler)
    
    def create_pattern_alert(self, pattern: Union[PatternDetection, CandlestickDetection],
                           symbol: str, timeframe: str, current_price: float,
                           alert_type: AlertType = AlertType.PATTERN_DETECTED,
                           metadata: Optional[Dict] = None) -> Optional[PatternAlert]:
        """
        Create a pattern alert.
        
        Args:
            pattern: Pattern detection result
            symbol: Symbol
            timeframe: Timeframe
            current_price: Current price
            alert_type: Type of alert
            metadata: Additional metadata
            
        Returns:
            Created alert or None if below threshold
        """
        if pattern.confidence < self.min_confidence_for_alert:
            return None
        
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(datetime.now().timestamp())}"
        
        # Get pattern reliability
        pattern_type = pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type)
        reliability = self.performance_tracker.get_pattern_reliability_score(
            pattern_type, pattern.confidence
        )
        
        # Determine priority
        if reliability >= self.min_reliability_for_high_priority and pattern.confidence >= 0.8:
            priority = AlertPriority.HIGH
        elif reliability >= 0.65 and pattern.confidence >= 0.7:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        # Create alert message
        direction = pattern.direction.value if hasattr(pattern.direction, 'value') else str(pattern.direction)
        message = self._generate_alert_message(pattern_type, direction, symbol, timeframe, 
                                             pattern.confidence, reliability)
        
        alert = PatternAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            priority=priority,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            direction=direction,
            confidence=pattern.confidence,
            current_price=current_price,
            target_price=getattr(pattern, 'target_price', None),
            stop_loss=getattr(pattern, 'stop_loss', None),
            risk_reward_ratio=getattr(pattern, 'risk_reward_ratio', None),
            message=message,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        with self._lock:
            self.active_alerts.append(alert)
            self.alert_queue.append(alert)
        
        logger.info(f"Created {priority.value} priority alert: {alert_id}")
        return alert
    
    def create_confluence_alert(self, chart_pattern: PatternDetection, 
                              candlestick_pattern: CandlestickDetection,
                              symbol: str, timeframe: str, current_price: float,
                              confluence_score: float) -> Optional[PatternAlert]:
        """Create alert for pattern confluence."""
        
        adjusted_confidence = min(1.0, confluence_score + self.confluence_bonus)
        
        if adjusted_confidence < self.min_confidence_for_alert:
            return None
        
        with self._lock:
            self._alert_counter += 1
            alert_id = f"confluence_{self._alert_counter}_{int(datetime.now().timestamp())}"
        
        # Confluence patterns get high priority if both patterns are strong
        if (chart_pattern.confidence >= 0.7 and 
            candlestick_pattern.confidence >= 0.7 and
            confluence_score >= 0.8):
            priority = AlertPriority.HIGH
        elif confluence_score >= 0.7:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        # Create confluence message
        chart_type = chart_pattern.pattern_type.value if hasattr(chart_pattern.pattern_type, 'value') else str(chart_pattern.pattern_type)
        candlestick_type = candlestick_pattern.pattern_type.value if hasattr(candlestick_pattern.pattern_type, 'value') else str(candlestick_pattern.pattern_type)
        direction = chart_pattern.direction.value if hasattr(chart_pattern.direction, 'value') else str(chart_pattern.direction)
        
        message = (f"🔥 CONFLUENCE ALERT: {chart_type.upper()} + {candlestick_type.upper()} "
                  f"detected on {symbol} ({timeframe}). "
                  f"{direction.capitalize()} signal with {confluence_score:.1%} confluence score.")
        
        alert = PatternAlert(
            alert_id=alert_id,
            alert_type=AlertType.CONFLUENCE_DETECTED,
            priority=priority,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=f"{chart_type}+{candlestick_type}",
            direction=direction,
            confidence=adjusted_confidence,
            current_price=current_price,
            target_price=chart_pattern.target_price,
            stop_loss=chart_pattern.stop_loss,
            risk_reward_ratio=chart_pattern.risk_reward_ratio,
            message=message,
            metadata={
                'chart_pattern': asdict(chart_pattern) if hasattr(chart_pattern, '__dict__') else str(chart_pattern),
                'candlestick_pattern': asdict(candlestick_pattern) if hasattr(candlestick_pattern, '__dict__') else str(candlestick_pattern),
                'confluence_score': confluence_score
            },
            created_at=datetime.now()
        )
        
        with self._lock:
            self.active_alerts.append(alert)
            self.alert_queue.append(alert)
        
        logger.info(f"Created confluence alert: {alert_id}")
        return alert
    
    def _generate_alert_message(self, pattern_type: str, direction: str, symbol: str, 
                              timeframe: str, confidence: float, reliability: float) -> str:
        """Generate formatted alert message."""
        
        emoji_map = {
            'bullish': '🚀',
            'bearish': '📉',
            'neutral': '⚖️'
        }
        
        emoji = emoji_map.get(direction.lower(), '📊')
        
        message = (f"{emoji} {pattern_type.upper().replace('_', ' ')} pattern detected on {symbol} "
                  f"({timeframe}). {direction.capitalize()} signal with {confidence:.1%} confidence "
                  f"and {reliability:.1%} historical reliability.")
        
        return message
    
    async def send_alert(self, alert: PatternAlert) -> bool:
        """Send alert through all notification handlers."""
        
        if alert.sent:
            return True
        
        success = True
        
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert through handler {handler}: {e}")
                success = False
        
        alert.sent = success
        
        if success:
            logger.info(f"Successfully sent alert {alert.alert_id}")
        
        return success
    
    async def process_alert_queue(self, max_alerts: int = 10):
        """Process pending alerts in queue."""
        
        processed = 0
        
        with self._lock:
            pending_alerts = [alert for alert in self.active_alerts if not alert.sent]
        
        # Sort by priority and time
        priority_order = {AlertPriority.CRITICAL: 4, AlertPriority.HIGH: 3, 
                         AlertPriority.MEDIUM: 2, AlertPriority.LOW: 1}
        
        pending_alerts.sort(key=lambda x: (priority_order[x.priority], x.created_at), reverse=True)
        
        for alert in pending_alerts[:max_alerts]:
            try:
                await self.send_alert(alert)
                processed += 1
            except Exception as e:
                logger.error(f"Error processing alert {alert.alert_id}: {e}")
        
        return processed
    
    def get_active_alerts(self, priority: Optional[AlertPriority] = None, 
                         symbol: Optional[str] = None) -> List[PatternAlert]:
        """Get active alerts with optional filtering."""
        
        with self._lock:
            alerts = list(self.active_alerts)
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        return alerts
    
    def clear_old_alerts(self, max_age_hours: int = 24):
        """Clear alerts older than specified hours."""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            self.active_alerts = [a for a in self.active_alerts if a.created_at >= cutoff_time]
        
        logger.info(f"Cleared alerts older than {max_age_hours} hours")

class BreakoutMonitor:
    """Monitor patterns for breakout confirmations."""
    
    def __init__(self, alert_system: PatternAlertSystem):
        """Initialize breakout monitor."""
        self.alert_system = alert_system
        self.monitored_patterns: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def add_pattern_for_monitoring(self, pattern_id: str, symbol: str, 
                                 pattern_data: Dict, breakout_level: float,
                                 direction: str):
        """Add pattern for breakout monitoring."""
        
        with self._lock:
            self.monitored_patterns[pattern_id] = {
                'symbol': symbol,
                'pattern_data': pattern_data,
                'breakout_level': breakout_level,
                'direction': direction,
                'added_at': datetime.now(),
                'confirmed': False
            }
        
        logger.info(f"Added pattern {pattern_id} for breakout monitoring at ${breakout_level:.2f}")
    
    def check_breakouts(self, price_data: Dict[str, float]) -> List[PatternAlert]:
        """Check for breakout confirmations and generate alerts."""
        
        breakout_alerts = []
        
        with self._lock:
            for pattern_id, data in list(self.monitored_patterns.items()):
                symbol = data['symbol']
                current_price = price_data.get(symbol)
                
                if current_price is None:
                    continue
                
                breakout_level = data['breakout_level']
                direction = data['direction']
                
                # Check for breakout
                breakout_confirmed = False
                
                if direction == 'bullish' and current_price > breakout_level * 1.01:  # 1% above breakout
                    breakout_confirmed = True
                elif direction == 'bearish' and current_price < breakout_level * 0.99:  # 1% below breakout
                    breakout_confirmed = True
                
                if breakout_confirmed and not data['confirmed']:
                    # Create breakout alert
                    alert = PatternAlert(
                        alert_id=f"breakout_{pattern_id}_{int(datetime.now().timestamp())}",
                        alert_type=AlertType.BREAKOUT_CONFIRMED,
                        priority=AlertPriority.HIGH,
                        symbol=symbol,
                        timeframe=data['pattern_data'].get('timeframe', 'unknown'),
                        pattern_type=data['pattern_data'].get('pattern_type', 'unknown'),
                        direction=direction,
                        confidence=data['pattern_data'].get('confidence', 0.8),
                        current_price=current_price,
                        target_price=data['pattern_data'].get('target_price'),
                        stop_loss=data['pattern_data'].get('stop_loss'),
                        risk_reward_ratio=data['pattern_data'].get('risk_reward_ratio'),
                        message=f"🎯 BREAKOUT CONFIRMED: {symbol} broke {direction} through ${breakout_level:.2f} resistance/support. Current price: ${current_price:.2f}",
                        metadata={'original_pattern_id': pattern_id, 'breakout_level': breakout_level},
                        created_at=datetime.now()
                    )
                    
                    breakout_alerts.append(alert)
                    data['confirmed'] = True
                    
                    logger.info(f"Breakout confirmed for pattern {pattern_id}: {symbol} @ ${current_price:.2f}")
        
        return breakout_alerts
    
    def cleanup_old_patterns(self, max_age_days: int = 7):
        """Remove old patterns from monitoring."""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        with self._lock:
            old_patterns = [pid for pid, data in self.monitored_patterns.items() 
                          if data['added_at'] < cutoff_date]
            
            for pattern_id in old_patterns:
                del self.monitored_patterns[pattern_id]
        
        logger.info(f"Removed {len(old_patterns)} old patterns from breakout monitoring")

# Example notification handlers

async def console_notification_handler(alert: PatternAlert):
    """Simple console notification handler."""
    print(f"\n{'='*60}")
    print(f"PATTERN ALERT - {alert.priority.value.upper()}")
    print(f"{'='*60}")
    print(f"Symbol: {alert.symbol}")
    print(f"Pattern: {alert.pattern_type}")
    print(f"Direction: {alert.direction}")
    print(f"Confidence: {alert.confidence:.2%}")
    print(f"Current Price: ${alert.current_price:.2f}")
    if alert.target_price:
        print(f"Target: ${alert.target_price:.2f}")
    if alert.stop_loss:
        print(f"Stop Loss: ${alert.stop_loss:.2f}")
    print(f"Message: {alert.message}")
    print(f"Time: {alert.created_at}")
    print(f"{'='*60}\n")

def webhook_notification_handler(alert: PatternAlert, webhook_url: str):
    """Webhook notification handler."""
    import requests
    
    payload = {
        'alert_id': alert.alert_id,
        'type': alert.alert_type.value,
        'priority': alert.priority.value,
        'symbol': alert.symbol,
        'pattern_type': alert.pattern_type,
        'direction': alert.direction,
        'confidence': alert.confidence,
        'current_price': alert.current_price,
        'target_price': alert.target_price,
        'stop_loss': alert.stop_loss,
        'message': alert.message,
        'timestamp': alert.created_at.isoformat()
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
    except Exception as e:
        logger.error(f"Failed to send webhook notification: {e}")
        raise

class PatternPerformanceReporter:
    """Generate performance reports and analytics."""
    
    def __init__(self, performance_tracker: PatternPerformanceTracker):
        """Initialize reporter with performance tracker."""
        self.performance_tracker = performance_tracker
    
    def generate_daily_report(self) -> Dict:
        """Generate daily performance report."""
        
        # Get today's patterns
        today = datetime.now().date()
        with self.performance_tracker._lock:
            today_patterns = [r for r in self.performance_tracker.performance_records
                            if r.detected_at.date() == today]
        
        # Overall statistics
        overall_stats = self.performance_tracker.calculate_pattern_statistics()
        
        # Pattern type breakdown
        pattern_types = set(r.pattern_type for r in today_patterns)
        pattern_breakdown = {}
        
        for pattern_type in pattern_types:
            type_stats = self.performance_tracker.calculate_pattern_statistics(pattern_type=pattern_type)
            pattern_breakdown[pattern_type] = type_stats
        
        # Top performers
        top_patterns = []
        if not overall_stats.get('error'):
            for pattern_type in pattern_types:
                stats = pattern_breakdown.get(pattern_type, {})
                if not stats.get('error') and stats.get('success_rate', 0) > 0.6:
                    top_patterns.append({
                        'pattern_type': pattern_type,
                        'success_rate': stats['success_rate'],
                        'total_patterns': stats['total_patterns']
                    })
        
        top_patterns.sort(key=lambda x: x['success_rate'], reverse=True)
        
        report = {
            'report_date': today.isoformat(),
            'patterns_detected_today': len(today_patterns),
            'overall_performance': overall_stats,
            'pattern_breakdown': pattern_breakdown,
            'top_performing_patterns': top_patterns[:5],
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def export_performance_data(self, filename: Optional[str] = None) -> str:
        """Export performance data to JSON file."""
        
        if filename is None:
            filename = f"pattern_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self.performance_tracker._lock:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(self.performance_tracker.performance_records),
                'performance_records': [asdict(r) for r in self.performance_tracker.performance_records],
                'overall_statistics': self.performance_tracker.calculate_pattern_statistics()
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance data exported to {filename}")
        return filename


class NotificationHandler:
    """Handles sending notifications for pattern alerts."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize notification handler.
        
        Args:
            webhook_url: Optional webhook URL for sending notifications
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    def send_console_notification(self, alert: PatternAlert) -> bool:
        """Send notification to console/logs."""
        try:
            priority_emoji = {
                AlertPriority.LOW: "ℹ️",
                AlertPriority.MEDIUM: "⚠️", 
                AlertPriority.HIGH: "🚨",
                AlertPriority.CRITICAL: "🔥"
            }
            
            emoji = priority_emoji.get(alert.priority, "📊")
            
            notification_msg = f"""
{emoji} PATTERN ALERT [{alert.priority.value.upper()}]
{'='*50}
Symbol: {alert.symbol}
Timeframe: {alert.timeframe}
Pattern: {alert.pattern_type}
Direction: {alert.direction}
Confidence: {alert.confidence:.1%}
Current Price: ${alert.current_price:.2f}
{'Target: $' + str(alert.target_price) if alert.target_price else ''}
{'Stop Loss: $' + str(alert.stop_loss) if alert.stop_loss else ''}
Message: {alert.message}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
            """
            
            self.logger.info(notification_msg)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send console notification: {e}")
            return False
    
    def send_webhook_notification(self, alert: PatternAlert, webhook_url: Optional[str] = None) -> bool:
        """Send notification via webhook."""
        try:
            import requests
            
            url = webhook_url or self.webhook_url
            if not url:
                self.logger.warning("No webhook URL configured")
                return False
            
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "priority": alert.priority.value,
                "symbol": alert.symbol,
                "timeframe": alert.timeframe,
                "pattern_type": alert.pattern_type,
                "direction": alert.direction,
                "confidence": alert.confidence,
                "current_price": alert.current_price,
                "target_price": alert.target_price,
                "stop_loss": alert.stop_loss,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully sent webhook notification for alert {alert.alert_id}")
                return True
            else:
                self.logger.error(f"Webhook notification failed with status {response.status_code}")
                return False
                
        except ImportError:
            self.logger.error("requests library not available for webhook notifications")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def send_email_notification(self, alert: PatternAlert, recipient: str) -> bool:
        """Send notification via email (placeholder implementation)."""
        try:
            # This would integrate with an email service
            # For now, just log the email content
            
            subject = f"Pattern Alert: {alert.pattern_type} on {alert.symbol}"
            
            body = f"""
Pattern Alert Notification

Symbol: {alert.symbol}
Timeframe: {alert.timeframe}
Pattern Type: {alert.pattern_type}
Direction: {alert.direction}
Confidence: {alert.confidence:.1%}
Priority: {alert.priority.value.upper()}

Current Price: ${alert.current_price:.2f}
{f'Target Price: ${alert.target_price:.2f}' if alert.target_price else ''}
{f'Stop Loss: ${alert.stop_loss:.2f}' if alert.stop_loss else ''}

Message: {alert.message}

Alert generated at: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.logger.info(f"Email notification prepared for {recipient}: {subject}")
            # In production, would send actual email here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False