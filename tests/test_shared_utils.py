"""
Tests for shared utility functions.
Tests the common functions used across multiple DAGs and modules.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.shared import (
    get_data_manager,
    validate_data_quality,
    calculate_returns,
    calculate_volatility,
    log_performance,
    send_alerts,
    aggregate_data_quality_scores
)


class TestGetDataManager:
    """Test data manager factory function."""
    
    def test_get_data_manager_fallback(self):
        """Test that fallback mock data manager is returned when imports fail."""
        data_manager = get_data_manager()
        
        # Should return a mock data manager
        assert hasattr(data_manager, 'collect_market_data')
        assert hasattr(data_manager, 'collect_fundamental_data')
        assert hasattr(data_manager, 'collect_sentiment_data')
        
        # Test mock functionality
        market_result = data_manager.collect_market_data(['AAPL', 'SPY'])
        assert market_result['status'] == 'success'
        assert market_result['symbols_collected'] == 2
        assert 'data' in market_result


class TestValidateDataQuality:
    """Test data quality validation function."""
    
    def test_validate_empty_data(self):
        """Test validation of empty data."""
        result = validate_data_quality(None, 'market')
        
        assert result['quality_score'] == 0.0
        assert 'No data available' in result['issues']
        assert result['data_type'] == 'market'
    
    def test_validate_market_data_success(self):
        """Test validation of successful market data."""
        market_data = {
            'status': 'success',
            'success_rate': 0.9,
            'data': {'AAPL': {'price': 150}}
        }
        
        result = validate_data_quality(market_data, 'market')
        
        assert result['quality_score'] == 0.9
        assert len(result['issues']) == 0
        assert result['data_type'] == 'market'
    
    def test_validate_sentiment_data_low_quality(self):
        """Test validation of low quality sentiment data."""
        sentiment_data = {
            'status': 'partial',
            'article_count': 5  # Low article count
        }
        
        result = validate_data_quality(sentiment_data, 'sentiment', min_threshold=0.8)
        
        assert result['quality_score'] == 0.25  # 5/20 = 0.25
        assert len(result['issues']) > 0
        assert 'Low quality score' in result['issues'][0]
    
    def test_validate_fundamental_data_symbols(self):
        """Test validation based on symbols processed."""
        fundamental_data = {
            'status': 'success',
            'symbols_processed': 8
        }
        
        result = validate_data_quality(fundamental_data, 'fundamental')
        
        assert result['quality_score'] == 0.8  # 8/10 = 0.8
        assert result['data_type'] == 'fundamental'


class TestCalculateReturns:
    """Test return calculation functions."""
    
    def test_calculate_returns_simple(self):
        """Test simple return calculation."""
        prices = [100, 105, 110, 108, 112]
        returns = calculate_returns(prices, periods=1)
        
        expected = [0.05, 0.047619, -0.018182, 0.037037]  # Approximate values
        assert len(returns) == 4
        assert abs(returns[0] - 0.05) < 0.001  # 5% return
    
    def test_calculate_returns_insufficient_data(self):
        """Test return calculation with insufficient data."""
        prices = [100]
        returns = calculate_returns(prices, periods=1)
        
        assert returns == []
    
    def test_calculate_returns_zero_price(self):
        """Test return calculation with zero price."""
        prices = [0, 100, 105]
        returns = calculate_returns(prices, periods=1)
        
        assert returns[0] == 0.0  # Should handle division by zero
        assert abs(returns[1] - 0.05) < 0.001


class TestCalculateVolatility:
    """Test volatility calculation."""
    
    def test_calculate_volatility_normal(self):
        """Test normal volatility calculation."""
        returns = [0.01, -0.02, 0.015, -0.005, 0.03]
        volatility = calculate_volatility(returns)
        
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_calculate_volatility_insufficient_data(self):
        """Test volatility with insufficient data."""
        returns = [0.01]
        volatility = calculate_volatility(returns)
        
        assert volatility == 0.0
    
    def test_calculate_volatility_empty(self):
        """Test volatility with empty returns."""
        returns = []
        volatility = calculate_volatility(returns)
        
        assert volatility == 0.0


class TestLogPerformance:
    """Test performance logging function."""
    
    def test_log_performance_success(self):
        """Test successful performance logging."""
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 0, 5)  # 5 seconds later
        
        result = log_performance('test_operation', start_time, end_time, 'success', {'records': 100})
        
        assert result['operation'] == 'test_operation'
        assert result['duration_seconds'] == 5.0
        assert result['status'] == 'success'
        assert result['metrics']['records'] == 100
    
    def test_log_performance_no_metrics(self):
        """Test performance logging without metrics."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2)
        
        result = log_performance('simple_operation', start_time, end_time)
        
        assert result['operation'] == 'simple_operation'
        assert result['status'] == 'success'  # Default status
        assert result['metrics'] == {}


class TestSendAlerts:
    """Test alert sending function."""
    
    @patch('src.utils.shared.logger')
    def test_send_alerts_info(self, mock_logger):
        """Test sending info alerts."""
        success = send_alerts('data_quality', 'Test message', 'info', {'context': 'test'})
        
        assert success is True
        mock_logger.info.assert_called_once()
    
    @patch('src.utils.shared.logger')
    def test_send_alerts_critical(self, mock_logger):
        """Test sending critical alerts."""
        success = send_alerts('system_error', 'Critical issue', 'critical')
        
        assert success is True
        mock_logger.critical.assert_called_once()
    
    @patch('src.utils.shared.logger')
    def test_send_alerts_warning(self, mock_logger):
        """Test sending warning alerts."""
        success = send_alerts('performance', 'Performance degraded', 'warning')
        
        assert success is True
        mock_logger.warning.assert_called_once()


class TestAggregateDataQualityScores:
    """Test data quality score aggregation."""
    
    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        result = aggregate_data_quality_scores([])
        
        assert result['overall_score'] == 0.0
        assert result['grade'] == 'poor'
        assert result['total_issues'] == 0
    
    def test_aggregate_mixed_quality(self):
        """Test aggregation with mixed quality scores."""
        quality_results = [
            {'quality_score': 0.9, 'issues': []},
            {'quality_score': 0.7, 'issues': ['Minor issue']},
            {'quality_score': 0.5, 'issues': ['Major issue', 'Another issue']},
            {'quality_score': 0.8, 'issues': []}
        ]
        
        result = aggregate_data_quality_scores(quality_results)
        
        assert abs(result['overall_score'] - 0.725) < 0.001  # (0.9 + 0.7 + 0.5 + 0.8) / 4
        assert result['grade'] == 'good'  # 0.725 >= 0.7
        assert result['total_issues'] == 3
        assert result['data_completeness'] == 0.5  # 2 sources >= 0.8 out of 4 (0.9, 0.8)
        assert result['source_count'] == 4
    
    def test_aggregate_excellent_quality(self):
        """Test aggregation with excellent quality."""
        quality_results = [
            {'quality_score': 0.95, 'issues': []},
            {'quality_score': 0.92, 'issues': []}
        ]
        
        result = aggregate_data_quality_scores(quality_results)
        
        assert result['overall_score'] == 0.935
        assert result['grade'] == 'excellent'
        assert result['total_issues'] == 0
        assert result['data_completeness'] == 1.0
    
    def test_aggregate_poor_quality(self):
        """Test aggregation with poor quality."""
        quality_results = [
            {'quality_score': 0.3, 'issues': ['Critical error']},
            {'quality_score': 0.2, 'issues': ['System failure', 'Data corruption']}
        ]
        
        result = aggregate_data_quality_scores(quality_results)
        
        assert result['overall_score'] == 0.25
        assert result['grade'] == 'poor'
        assert result['total_issues'] == 3
        assert result['data_completeness'] == 0.0  # No sources >= 0.8