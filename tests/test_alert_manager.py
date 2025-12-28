"""Tests for alert manager utilities."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.utils.alert_manager import AlertManager, AlertSeverity, send_prioritized_alert


class TestAlertManager:
    """Test alert manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()
        
        # Sample performance data
        self.sample_performance = {
            'total_return_pct': 5.2,
            'max_drawdown_pct': 8.5,
            'win_rate': 65.0,
            'sharpe_ratio': 1.2
        }
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        assert len(self.alert_manager.alert_history) == 0
        assert len(self.alert_manager.rate_limits) == 0
        assert len(self.alert_manager.alert_handlers) == 0
        assert AlertSeverity.CRITICAL in self.alert_manager.default_rate_limits
    
    def test_send_prioritized_alert_basic(self):
        """Test basic prioritized alert sending."""
        result = self.alert_manager.send_prioritized_alert(
            "Test alert message",
            severity="high",
            context={'test': 'data'}
        )
        
        assert result['status'] == 'sent'
        assert result['message'] == "Test alert message"
        assert result['severity'] == "high"
        assert 'id' in result
        assert 'timestamp' in result
        assert 'channels_sent' in result
        
        # Should be in history
        assert len(self.alert_manager.alert_history) == 1
    
    def test_send_prioritized_alert_invalid_severity(self):
        """Test alert with invalid severity level."""
        with pytest.raises(ValueError):
            self.alert_manager.send_prioritized_alert("Test", severity="invalid")
    
    def test_send_prioritized_alert_rate_limiting(self):
        """Test alert rate limiting functionality."""
        # Send many similar alerts quickly
        message = "Rate limit test message"
        
        # First alert should succeed
        result1 = self.alert_manager.send_prioritized_alert(message, severity="critical")
        assert result1['status'] == 'sent'
        
        # Send many more to trigger rate limit (critical = 50/hour)
        for i in range(60):
            self.alert_manager.send_prioritized_alert(message, severity="critical")
        
        # Should eventually hit rate limit
        result_final = self.alert_manager.send_prioritized_alert(message, severity="critical")
        assert result_final['status'] == 'rate_limited'
    
    def test_send_prioritized_alert_duplicate_suppression(self):
        """Test duplicate alert suppression."""
        message = "Duplicate test message"
        severity = "medium"
        
        # Send first alert
        result1 = self.alert_manager.send_prioritized_alert(message, severity=severity)
        assert result1['status'] == 'sent'
        
        # Send same alert immediately
        result2 = self.alert_manager.send_prioritized_alert(message, severity=severity)
        assert result2['status'] == 'duplicate'
    
    def test_send_performance_alert_critical_drawdown(self):
        """Test performance alert for critical drawdown."""
        high_drawdown_performance = self.sample_performance.copy()
        high_drawdown_performance['max_drawdown_pct'] = 25.0  # 25% drawdown
        
        result = self.alert_manager.send_performance_alert(high_drawdown_performance)
        
        assert result['alerts_sent'] > 0
        assert len(result['alert_details']) > 0
        
        # Should have critical alert
        critical_alert = result['alert_details'][0]
        assert 'CRITICAL' in critical_alert['message']
        assert critical_alert['severity'] == 'critical'
    
    def test_send_performance_alert_high_drawdown(self):
        """Test performance alert for high drawdown."""
        moderate_drawdown_performance = self.sample_performance.copy()
        moderate_drawdown_performance['max_drawdown_pct'] = 18.0  # 18% drawdown
        
        result = self.alert_manager.send_performance_alert(moderate_drawdown_performance)
        
        assert result['alerts_sent'] > 0
        high_alert = result['alert_details'][0]
        assert 'HIGH' in high_alert['message']
        assert high_alert['severity'] == 'high'
    
    def test_send_performance_alert_low_win_rate(self):
        """Test performance alert for low win rate."""
        low_win_rate_performance = self.sample_performance.copy()
        low_win_rate_performance['win_rate'] = 35.0  # 35% win rate
        
        result = self.alert_manager.send_performance_alert(low_win_rate_performance)
        
        assert result['alerts_sent'] > 0
        win_rate_alert = result['alert_details'][0]
        assert 'win rate' in win_rate_alert['message'].lower()
        assert win_rate_alert['severity'] == 'medium'
    
    def test_send_performance_alert_custom_thresholds(self):
        """Test performance alert with custom thresholds."""
        custom_thresholds = {
            'max_drawdown_critical': 0.10,  # 10% (lower than default)
            'low_win_rate': 0.70           # 70% (higher than default)
        }
        
        result = self.alert_manager.send_performance_alert(
            self.sample_performance, 
            thresholds=custom_thresholds
        )
        
        # Should trigger win rate alert with custom threshold
        assert result['alerts_sent'] > 0
    
    def test_format_trading_alert_high_confidence(self):
        """Test trading alert formatting with high confidence."""
        signal = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'price': 150.25,
            'reason': 'Bullish breakout with volume confirmation'
        }
        confidence = 0.85
        
        alert_message = self.alert_manager.format_trading_alert(signal, confidence)
        
        assert 'HIGH CONFIDENCE' in alert_message
        assert 'BUY' in alert_message
        assert 'AAPL' in alert_message
        assert '$150.25' in alert_message
        assert '85%' in alert_message
    
    def test_format_trading_alert_medium_confidence(self):
        """Test trading alert formatting with medium confidence."""
        signal = {
            'symbol': 'GOOGL',
            'action': 'SELL',
            'price': 2100.50,
            'reason': 'Technical resistance'
        }
        confidence = 0.65
        
        alert_message = self.alert_manager.format_trading_alert(signal, confidence)
        
        assert 'MEDIUM CONFIDENCE' in alert_message
        assert 'SELL' in alert_message
        assert 'GOOGL' in alert_message
        assert '65%' in alert_message
    
    def test_format_trading_alert_low_confidence(self):
        """Test trading alert formatting with low confidence."""
        signal = {
            'symbol': 'MSFT',
            'action': 'HOLD',
            'price': 300.00,
            'reason': 'Mixed signals'
        }
        confidence = 0.45
        
        alert_message = self.alert_manager.format_trading_alert(signal, confidence)
        
        assert 'LOW CONFIDENCE' in alert_message
        assert 'HOLD' in alert_message
        assert '45%' in alert_message
    
    def test_register_alert_handler(self):
        """Test registering custom alert handler."""
        mock_handler = Mock(return_value=True)
        
        self.alert_manager.register_alert_handler('custom_channel', mock_handler)
        
        # Send alert to custom channel
        result = self.alert_manager.send_prioritized_alert(
            "Test custom handler",
            channels=['custom_channel']
        )
        
        assert 'custom_channel' in result['channels_sent']
        mock_handler.assert_called_once()
    
    def test_get_alert_summary_recent_period(self):
        """Test alert summary for recent period."""
        # Send alerts with different severities
        self.alert_manager.send_prioritized_alert("Critical alert", severity="critical")
        self.alert_manager.send_prioritized_alert("High alert", severity="high")
        self.alert_manager.send_prioritized_alert("Medium alert", severity="medium")
        
        summary = self.alert_manager.get_alert_summary(hours=24)
        
        assert summary['total_alerts'] == 3
        assert 'by_severity' in summary
        assert summary['by_severity']['critical'] == 1
        assert summary['by_severity']['high'] == 1
        assert summary['by_severity']['medium'] == 1
        assert 'alert_rate_per_hour' in summary
        assert 'most_recent' in summary
    
    def test_get_alert_summary_empty_period(self):
        """Test alert summary with no recent alerts."""
        summary = self.alert_manager.get_alert_summary(hours=1)
        
        assert summary['total_alerts'] == 0
        assert summary['by_severity'] == {}
    
    def test_alert_routing_by_severity(self):
        """Test alert routing based on severity levels."""
        # Critical alert should go to multiple channels
        result_critical = self.alert_manager.send_prioritized_alert(
            "Critical test", severity="critical"
        )
        assert len(result_critical['channels_sent']) >= 1
        
        # Info alert should go to fewer channels
        result_info = self.alert_manager.send_prioritized_alert(
            "Info test", severity="info"
        )
        assert len(result_info['channels_sent']) >= 1
    
    def test_rate_limit_reset_after_window(self):
        """Test rate limit reset after time window."""
        message = "Rate limit reset test"
        
        # Fill rate limit
        for i in range(10):
            self.alert_manager.send_prioritized_alert(message, severity="info")
        
        # Clear history to avoid duplicate detection and mock time passage
        self.alert_manager.alert_history.clear()
        with patch('time.time', return_value=time.time() + 3700):  # 1 hour + 100 seconds
            result = self.alert_manager.send_prioritized_alert(message, severity="info")
            assert result['status'] == 'sent'  # Should work after reset
    
    def test_error_handling_in_alert_processing(self):
        """Test error handling in alert processing."""
        # Mock an error in routing
        with patch.object(self.alert_manager, '_route_alert', side_effect=Exception("Routing error")):
            result = self.alert_manager.send_prioritized_alert("Error test")
            assert result['status'] == 'error'
            assert 'Routing error' in result['message']
    
    def test_error_handling_in_performance_alerts(self):
        """Test error handling in performance alerts."""
        # Test with malformed performance data that causes an exception
        malformed_data = None  # This will cause an exception when calling .get()
        
        result = self.alert_manager.send_performance_alert(malformed_data)
        assert result['alerts_sent'] == 0
        assert 'error' in result


class TestAlertManagerChannelHandling:
    """Test alert manager channel handling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()
    
    def test_send_to_log_channel(self):
        """Test sending alerts to log channel."""
        alert = {
            'message': 'Test log message',
            'severity': 'medium',
            'id': 'test_id',
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('src.utils.alert_manager.logger') as mock_logger:
            success = self.alert_manager._send_to_channel(alert, 'log')
            assert success is True
            mock_logger.warning.assert_called_once()
    
    def test_send_to_custom_handler(self):
        """Test sending alerts to custom handler."""
        mock_handler = Mock(return_value=True)
        self.alert_manager.register_alert_handler('test_channel', mock_handler)
        
        alert = {
            'message': 'Test custom message',
            'severity': 'high',
            'id': 'test_id'
        }
        
        success = self.alert_manager._send_to_channel(alert, 'test_channel')
        assert success is True
        mock_handler.assert_called_once_with(alert)
    
    def test_send_to_unknown_channel(self):
        """Test sending alerts to unknown channel."""
        alert = {
            'message': 'Test unknown channel',
            'severity': 'info'
        }
        
        with patch('src.utils.alert_manager.logger') as mock_logger:
            success = self.alert_manager._send_to_channel(alert, 'unknown_channel')
            assert success is True  # Should default to log
            mock_logger.info.assert_called()
    
    def test_channel_error_handling(self):
        """Test error handling in channel operations."""
        failing_handler = Mock(side_effect=Exception("Handler error"))
        self.alert_manager.register_alert_handler('failing_channel', failing_handler)
        
        alert = {'message': 'Test failing channel', 'severity': 'medium'}
        
        success = self.alert_manager._send_to_channel(alert, 'failing_channel')
        assert success is False


class TestAlertManagerConvenienceFunctions:
    """Test convenience functions for alert manager."""
    
    def test_send_prioritized_alert_function(self):
        """Test send_prioritized_alert convenience function."""
        result = send_prioritized_alert("Test convenience function", severity="low")
        
        assert result['status'] == 'sent'
        assert result['message'] == "Test convenience function"
        assert result['severity'] == "low"


class TestAlertManagerAdvanced:
    """Test advanced alert manager features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()
    
    def test_concurrent_alert_processing(self):
        """Test concurrent alert processing."""
        import threading
        results = []
        
        def send_alert(thread_id):
            result = self.alert_manager.send_prioritized_alert(
                f"Concurrent alert {thread_id}",
                severity="medium"
            )
            results.append(result)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=send_alert, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All alerts should be processed successfully
        assert len(results) == 10
        assert all(r['status'] == 'sent' for r in results)
    
    def test_alert_escalation_rules(self):
        """Test alert escalation rule setting."""
        self.alert_manager.set_escalation_rule(
            alert_pattern="CRITICAL*",
            escalation_time=300,  # 5 minutes
            escalation_channels=['email', 'webhook']
        )
        
        assert 'CRITICAL*' in self.alert_manager.escalation_rules
        rule = self.alert_manager.escalation_rules['CRITICAL*']
        assert rule['escalation_time'] == 300
        assert rule['channels'] == ['email', 'webhook']
    
    def test_alert_history_management(self):
        """Test alert history management and storage."""
        # Send multiple alerts
        for i in range(15):
            self.alert_manager.send_prioritized_alert(
                f"History test {i}",
                severity="info"
            )
        
        # All should be in history
        assert len(self.alert_manager.alert_history) == 15
        
        # Verify chronological order
        timestamps = [alert['timestamp'] for alert in self.alert_manager.alert_history]
        assert timestamps == sorted(timestamps)
    
    def test_performance_alert_multiple_thresholds(self):
        """Test performance alerts with multiple threshold violations."""
        poor_performance = {
            'total_return_pct': -15.0,
            'max_drawdown_pct': 22.0,  # Above critical
            'win_rate': 30.0,         # Below threshold
            'sharpe_ratio': -0.5
        }
        
        result = self.alert_manager.send_performance_alert(poor_performance)
        
        # Should generate multiple alerts
        assert result['alerts_sent'] >= 2  # Drawdown + win rate
        
        # Should have different severity levels
        severities = [alert['severity'] for alert in result['alert_details']]
        assert 'critical' in severities  # For high drawdown
        assert 'medium' in severities    # For low win rate