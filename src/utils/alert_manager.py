"""
Intelligent alert management with prioritization and routing.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertManager:
    """Intelligent alert management with prioritization, rate limiting, and routing."""
    
    def __init__(self):
        self.alert_history = deque(maxlen=10000)
        self.rate_limits = defaultdict(lambda: {'count': 0, 'window_start': time.time()})
        self.alert_handlers = {}
        self.escalation_rules = {}
        self.lock = threading.RLock()
        
        # Default rate limits (alerts per hour)
        self.default_rate_limits = {
            AlertSeverity.CRITICAL: 50,
            AlertSeverity.HIGH: 100,
            AlertSeverity.MEDIUM: 200,
            AlertSeverity.LOW: 500,
            AlertSeverity.INFO: 1000
        }
    
    def send_prioritized_alert(self, message: str, severity: str = "medium", 
                             context: Optional[Dict[str, Any]] = None,
                             channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send prioritized alert with intelligent routing.
        
        Args:
            message: Alert message
            severity: Alert severity level
            context: Additional context information
            channels: Specific channels to send to
            
        Returns:
            Alert processing result
        """
        # Validate severity first (let ValueError bubble up)
        severity_enum = AlertSeverity(severity.lower())
        
        try:
            with self.lock:
                # Check rate limits
                if not self._check_rate_limit(message, severity_enum):
                    return {
                        'status': 'rate_limited',
                        'message': 'Alert rate limit exceeded',
                        'severity': severity,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Check for duplicates
                if self._is_duplicate_alert(message, severity_enum):
                    return {
                        'status': 'duplicate',
                        'message': 'Duplicate alert suppressed',
                        'severity': severity,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Create alert record
                alert = {
                    'id': f"alert_{int(time.time() * 1000)}",
                    'message': message,
                    'severity': severity,
                    'context': context or {},
                    'timestamp': datetime.now().isoformat(),
                    'channels_sent': [],
                    'status': 'sent'
                }
                
                # Route alert to appropriate channels
                sent_channels = self._route_alert(alert, channels)
                alert['channels_sent'] = sent_channels
                
                # Store in history
                self.alert_history.append(alert)
                
                return alert
                
        except Exception as e:
            logger.error(f"Error sending prioritized alert: {e}")
            return {
                'status': 'error',
                'message': f"Failed to send alert: {e}",
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
    
    def send_performance_alert(self, performance_data: Dict[str, Any], 
                             thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send performance-based alerts with intelligent threshold checking.
        
        Args:
            performance_data: Performance metrics
            thresholds: Custom alert thresholds
            
        Returns:
            Alert result
        """
        try:
            if not thresholds:
                thresholds = {
                    'max_drawdown_critical': 0.20,  # 20% drawdown
                    'max_drawdown_high': 0.15,      # 15% drawdown
                    'negative_return_days': 5,       # 5 consecutive negative days
                    'low_win_rate': 0.40            # 40% win rate
                }
            
            total_return = performance_data.get('total_return_pct', 0)
            max_drawdown = performance_data.get('max_drawdown_pct', 0)
            win_rate = performance_data.get('win_rate', 50) / 100
            
            alerts_sent = []
            
            # Critical drawdown alert
            if max_drawdown > thresholds.get('max_drawdown_critical', 20) * 100:
                alert = self.send_prioritized_alert(
                    f"CRITICAL: Portfolio drawdown reached {max_drawdown:.1f}%",
                    severity="critical",
                    context={'performance_data': performance_data, 'alert_type': 'drawdown_critical'}
                )
                alerts_sent.append(alert)
            
            # High drawdown alert
            elif max_drawdown > thresholds.get('max_drawdown_high', 15) * 100:
                alert = self.send_prioritized_alert(
                    f"HIGH: Portfolio drawdown at {max_drawdown:.1f}%",
                    severity="high", 
                    context={'performance_data': performance_data, 'alert_type': 'drawdown_high'}
                )
                alerts_sent.append(alert)
            
            # Low win rate alert
            if win_rate < thresholds.get('low_win_rate', 0.40):
                alert = self.send_prioritized_alert(
                    f"MEDIUM: Low win rate detected: {win_rate:.1%}",
                    severity="medium",
                    context={'performance_data': performance_data, 'alert_type': 'low_win_rate'}
                )
                alerts_sent.append(alert)
            
            return {
                'alerts_sent': len(alerts_sent),
                'alert_details': alerts_sent,
                'performance_summary': performance_data
            }
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
            return {
                'alerts_sent': 0, 
                'alert_details': [],
                'performance_summary': performance_data,
                'error': str(e)
            }
    
    def format_trading_alert(self, signal: Dict[str, Any], confidence: float) -> str:
        """
        Format trading signal as alert message.
        
        Args:
            signal: Trading signal data
            confidence: Signal confidence level
            
        Returns:
            Formatted alert message
        """
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            action = signal.get('action', 'HOLD').upper()
            price = signal.get('price', 0)
            reason = signal.get('reason', 'Technical analysis')
            
            confidence_text = f"{confidence:.0%}"
            
            if confidence >= 0.8:
                priority = "🔥 HIGH CONFIDENCE"
            elif confidence >= 0.6:
                priority = "⚡ MEDIUM CONFIDENCE"
            else:
                priority = "📊 LOW CONFIDENCE"
            
            message = f"{priority} {action} signal for {symbol} at ${price:.2f} (Confidence: {confidence_text}) - {reason}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting trading alert: {e}")
            return f"Trading signal formatting error: {e}"
    
    def register_alert_handler(self, channel: str, handler: Callable[[Dict[str, Any]], bool]) -> None:
        """Register custom alert handler for a channel."""
        self.alert_handlers[channel] = handler
    
    def set_escalation_rule(self, alert_pattern: str, escalation_time: int, escalation_channels: List[str]) -> None:
        """Set escalation rule for unacknowledged alerts."""
        self.escalation_rules[alert_pattern] = {
            'escalation_time': escalation_time,
            'channels': escalation_channels
        }
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]
            
            if not recent_alerts:
                return {'total_alerts': 0, 'by_severity': {}, 'recent_period_hours': hours}
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in recent_alerts:
                severity_counts[alert['severity']] += 1
            
            # Calculate alert rate
            alert_rate = len(recent_alerts) / hours
            
            return {
                'total_alerts': len(recent_alerts),
                'by_severity': dict(severity_counts),
                'alert_rate_per_hour': alert_rate,
                'recent_period_hours': hours,
                'most_recent': recent_alerts[-1] if recent_alerts else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {'total_alerts': 0, 'error': str(e)}
    
    def _check_rate_limit(self, message: str, severity: AlertSeverity) -> bool:
        """Check if alert is within rate limits."""
        try:
            rate_key = f"{severity.value}_{hash(message) % 1000}"  # Group similar messages
            current_time = time.time()
            window_duration = 3600  # 1 hour window
            
            rate_info = self.rate_limits[rate_key]
            
            # Reset window if expired
            if current_time - rate_info['window_start'] > window_duration:
                rate_info['count'] = 0
                rate_info['window_start'] = current_time
            
            # Check limit
            limit = self.default_rate_limits.get(severity, 100)
            if rate_info['count'] >= limit:
                return False
            
            # Increment counter
            rate_info['count'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow alert on error
    
    def _is_duplicate_alert(self, message: str, severity: AlertSeverity, window_minutes: int = 10) -> bool:
        """Check if this is a duplicate alert within the time window."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            
            for alert in reversed(self.alert_history):
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time < cutoff_time:
                    break
                
                if (alert['message'] == message and 
                    alert['severity'] == severity.value):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate alert: {e}")
            return False
    
    def _route_alert(self, alert: Dict[str, Any], specific_channels: Optional[List[str]] = None) -> List[str]:
        """Route alert to appropriate channels based on severity and configuration."""
        try:
            severity = AlertSeverity(alert['severity'])
            channels_sent = []
            
            # Determine channels based on severity
            if specific_channels:
                target_channels = specific_channels
            else:
                if severity == AlertSeverity.CRITICAL:
                    target_channels = ['log', 'email', 'webhook']
                elif severity == AlertSeverity.HIGH:
                    target_channels = ['log', 'webhook']
                elif severity == AlertSeverity.MEDIUM:
                    target_channels = ['log']
                else:
                    target_channels = ['log']
            
            # Send to each channel
            for channel in target_channels:
                try:
                    success = self._send_to_channel(alert, channel)
                    if success:
                        channels_sent.append(channel)
                except Exception as e:
                    logger.error(f"Failed to send alert to channel {channel}: {e}")
            
            return channels_sent
            
        except Exception as e:
            logger.error(f"Error routing alert: {e}")
            return []
    
    def _send_to_channel(self, alert: Dict[str, Any], channel: str) -> bool:
        """Send alert to specific channel."""
        try:
            if channel == 'log':
                # Send to logging system
                severity = alert['severity']
                message = alert['message']
                
                if severity == 'critical':
                    logger.critical(f"ALERT: {message}")
                elif severity == 'high':
                    logger.error(f"ALERT: {message}")
                elif severity == 'medium':
                    logger.warning(f"ALERT: {message}")
                else:
                    logger.info(f"ALERT: {message}")
                
                return True
            
            elif channel in self.alert_handlers:
                # Use custom handler
                return self.alert_handlers[channel](alert)
            
            else:
                # Default: log as info
                logger.info(f"Alert sent to {channel}: {alert['message']}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending to channel {channel}: {e}")
            return False


# Global alert manager instance
alert_manager = AlertManager()

# Convenience functions
def send_prioritized_alert(message: str, severity: str = "medium", 
                         context: Optional[Dict[str, Any]] = None,
                         channels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Send prioritized alert."""
    return alert_manager.send_prioritized_alert(message, severity, context, channels)

def send_performance_alert(performance_data: Dict[str, Any], 
                         thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send performance alert."""
    return alert_manager.send_performance_alert(performance_data, thresholds)

def format_trading_alert(signal: Dict[str, Any], confidence: float) -> str:
    """Format trading alert message."""
    return alert_manager.format_trading_alert(signal, confidence)