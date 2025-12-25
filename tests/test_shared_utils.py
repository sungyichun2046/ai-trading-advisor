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
    aggregate_data_quality_scores,
    calculate_vix_regime,
    normalize_options_data,
    detect_unusual_volume,
    calculate_pattern_confidence,
    normalize_signals
)
import pandas as pd
import numpy as np


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


class TestVixRegimeCalculation:
    """Test VIX regime calculation function."""
    
    def test_vix_regime_low(self):
        """Test low volatility regime classification."""
        result = calculate_vix_regime(10.0)
        
        assert result['regime'] == 'low'
        assert result['vix_level'] == 10.0
        assert result['description'] == 'Low volatility environment'
        assert 'percentile' in result
    
    def test_vix_regime_normal(self):
        """Test normal volatility regime classification."""
        result = calculate_vix_regime(15.0)
        
        assert result['regime'] == 'normal'
        assert result['vix_level'] == 15.0
        assert result['description'] == 'Normal volatility environment'
    
    def test_vix_regime_elevated(self):
        """Test elevated volatility regime classification."""
        result = calculate_vix_regime(25.0)
        
        assert result['regime'] == 'elevated'
        assert result['vix_level'] == 25.0
        assert result['description'] == 'Elevated volatility environment'
    
    def test_vix_regime_high(self):
        """Test high volatility regime classification."""
        result = calculate_vix_regime(35.0)
        
        assert result['regime'] == 'high'
        assert result['vix_level'] == 35.0
        assert result['description'] == 'High volatility environment'
    
    def test_vix_regime_none_input(self):
        """Test VIX regime with None input."""
        result = calculate_vix_regime(None)
        
        assert result['regime'] == 'normal'
        assert result['vix_level'] == 18.5
    
    def test_vix_regime_nan_input(self):
        """Test VIX regime with NaN input."""
        result = calculate_vix_regime(np.nan)
        
        assert result['regime'] == 'normal'
        assert result['vix_level'] == 18.5


class TestOptionsNormalization:
    """Test options data normalization function."""
    
    def test_options_normalization_real_data(self):
        """Test normalization with real options data."""
        df = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'call_volume': [1000, 800, 600, 400, 200],
            'put_volume': [200, 400, 600, 800, 1000],
            'iv': [0.15, 0.18, 0.22, 0.25, 0.30]
        })
        
        result = normalize_options_data(df)
        
        assert len(result) == 5
        assert 'call_volume' in result.columns
        assert 'put_volume' in result.columns
        assert 'iv' in result.columns
        assert result['iv'].max() <= 2.0  # Clipped max
    
    def test_options_normalization_empty_data(self):
        """Test normalization with empty DataFrame."""
        df = pd.DataFrame()
        
        result = normalize_options_data(df)
        
        assert len(result) == 5
        assert 'strike' in result.columns
        assert 'call_volume' in result.columns
        assert 'put_volume' in result.columns
        assert 'iv' in result.columns
    
    def test_options_normalization_volume_scaling(self):
        """Test volume normalization to 0-1 range."""
        df = pd.DataFrame({
            'strike': [100, 110],
            'call_volume': [0, 1000],  # Min-max for testing
            'put_volume': [500, 1500],
            'iv': [0.2, 0.3]
        })
        
        result = normalize_options_data(df)
        
        # Call volume should be normalized to 0-1
        assert result['call_volume'].min() == 0.0
        assert abs(result['call_volume'].max() - 1.0) < 1e-10


class TestVolumeAnomalyDetection:
    """Test volume anomaly detection function."""
    
    def test_volume_anomaly_normal(self):
        """Test normal volume detection."""
        result = detect_unusual_volume(1000000, 1000000)
        
        assert result['volume_ratio'] == 1.0
        assert result['classification'] == 'normal'
        assert result['is_unusual'] is False
    
    def test_volume_anomaly_high(self):
        """Test high volume detection."""
        result = detect_unusual_volume(1800000, 1000000)
        
        assert result['volume_ratio'] == 1.8
        assert result['classification'] == 'high'
        assert result['is_unusual'] is True
    
    def test_volume_anomaly_very_high(self):
        """Test very high volume detection."""
        result = detect_unusual_volume(2500000, 1000000)
        
        assert result['volume_ratio'] == 2.5
        assert result['classification'] == 'very_high'
        assert result['is_unusual'] is True
    
    def test_volume_anomaly_extremely_high(self):
        """Test extremely high volume detection."""
        result = detect_unusual_volume(3500000, 1000000)
        
        assert result['volume_ratio'] == 3.5
        assert result['classification'] == 'extremely_high'
        assert result['is_unusual'] is True
    
    def test_volume_anomaly_low(self):
        """Test low volume detection."""
        result = detect_unusual_volume(400000, 1000000)
        
        assert result['volume_ratio'] == 0.4
        assert result['classification'] == 'low'
        assert result['is_unusual'] is True
    
    def test_volume_anomaly_none_input(self):
        """Test volume anomaly with None inputs."""
        result = detect_unusual_volume(None, None)
        
        assert 'volume' in result
        assert 'classification' in result
        assert result['volume'] == 1500000
        assert result['avg_volume'] == 1000000


class TestPatternConfidence:
    """Test pattern confidence calculation function."""
    
    def test_pattern_confidence_high(self):
        """Test high confidence pattern."""
        pattern = {
            'strength': 0.9,
            'volume_confirmed': True,
            'duration_bars': 12
        }
        
        result = calculate_pattern_confidence(pattern)
        
        assert 0.8 <= result <= 1.0
        assert isinstance(result, float)
    
    def test_pattern_confidence_medium(self):
        """Test medium confidence pattern."""
        pattern = {
            'strength': 0.6,
            'volume_confirmed': False,
            'duration_bars': 3
        }
        
        result = calculate_pattern_confidence(pattern)
        
        assert 0.4 <= result <= 0.8
    
    def test_pattern_confidence_low(self):
        """Test low confidence pattern."""
        pattern = {
            'strength': 0.3,
            'volume_confirmed': False,
            'duration_bars': 2
        }
        
        result = calculate_pattern_confidence(pattern)
        
        assert 0.0 <= result <= 0.6
    
    def test_pattern_confidence_empty_pattern(self):
        """Test confidence calculation with empty pattern."""
        result = calculate_pattern_confidence({})
        
        assert result == 0.75
    
    def test_pattern_confidence_missing_strength(self):
        """Test confidence calculation with missing strength."""
        pattern = {
            'volume_confirmed': True,
            'duration_bars': 8
        }
        
        result = calculate_pattern_confidence(pattern)
        
        assert result == 0.75  # Fallback value


class TestSignalNormalization:
    """Test signal normalization function."""
    
    def test_signal_normalization_rsi(self):
        """Test RSI signal normalization."""
        signals = {
            'rsi': 70,
            'stoch': 85
        }
        
        result = normalize_signals(signals)
        
        assert result['rsi'] == 0.7  # 70/100
        assert result['stoch'] == 0.85  # 85/100
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_signal_normalization_macd(self):
        """Test MACD signal normalization."""
        signals = {
            'macd': 0.5,
            'momentum': -0.2
        }
        
        result = normalize_signals(signals)
        
        assert result['macd'] == 0.75  # (0.5 + 1) / 2
        assert result['momentum'] == 0.4  # (-0.2 + 1) / 2
    
    def test_signal_normalization_mixed(self):
        """Test mixed signal normalization."""
        signals = {
            'rsi': 65,
            'macd': 0.3,
            'custom': 1.5
        }
        
        result = normalize_signals(signals)
        
        assert result['rsi'] == 0.65
        assert result['macd'] == 0.65  # (0.3 + 1) / 2
        assert result['custom'] == 1.0  # Clipped to 1.0
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_signal_normalization_empty(self):
        """Test normalization with empty signals."""
        result = normalize_signals({})
        
        assert 'rsi' in result
        assert 'macd' in result
        assert 'momentum' in result
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_signal_normalization_none_values(self):
        """Test normalization with None values."""
        signals = {
            'rsi': None,
            'macd': 0.2,
            'momentum': None
        }
        
        result = normalize_signals(signals)
        
        assert result['rsi'] == 0.5  # None -> 0.5
        assert result['macd'] == 0.6  # (0.2 + 1) / 2
        assert result['momentum'] == 0.5  # None -> 0.5