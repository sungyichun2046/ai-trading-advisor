"""
Tests for DataManager module including monitoring functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

from src.core.data_manager import DataManager, get_data_manager, validate_symbols


class TestDataManagerInitialization:
    """Test DataManager initialization and setup."""
    
    def test_initialization_default_config(self):
        """Test DataManager initialization with default config."""
        manager = DataManager()
        assert manager.config == {} and manager.retry_attempts == 3 and manager.retry_delay == 1
        assert 'host' in manager.connection_params and manager.sentiment_method in ['dummy', 'textblob', 'finbert']
    
    def test_initialization_missing_use_real_data_flag(self):
        """Test DataManager initialization fails without USE_REAL_DATA flag."""
        with patch('src.core.data_manager.settings') as mock_settings:
            if hasattr(mock_settings, 'use_real_data'): delattr(mock_settings, 'use_real_data')
            with pytest.raises(ValueError, match="USE_REAL_DATA flag required"): DataManager()
    
    @patch.dict('os.environ', {'POSTGRES_HOST': 'custom_host', 'POSTGRES_PORT': '5433', 'POSTGRES_DB': 'custom_db'})
    def test_initialization_custom_env_vars(self):
        """Test DataManager initialization with custom environment variables."""
        manager = DataManager()
        assert manager.connection_params['host'] == 'custom_host' and manager.connection_params['port'] == '5433' and manager.connection_params['database'] == 'custom_db'


class TestMarketDataCollection:
    """Test market data collection functionality."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_market_data_dummy_mode(self, mock_settings):
        """Test market data collection in dummy mode."""
        mock_settings.use_real_data = False
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = self.manager.collect_market_data(symbols)
        
        assert result['status'] == 'success' and result['symbols_collected'] == 3 and result['total_symbols'] == 3
        assert len(result['data']) == 3 and len(result['errors']) == 0
        
        for symbol in symbols:
            assert symbol in result['data']
            data = result['data'][symbol]
            assert data['symbol'] == symbol and data['status'] == 'success' and 'price' in data and data['data_source'] == 'dummy'
    
    @patch('src.core.data_manager.settings')
    @patch('src.core.data_manager.YFINANCE_AVAILABLE', True)
    @patch('src.core.data_manager.yf')
    def test_collect_market_data_yfinance_success(self, mock_yf, mock_settings):
        """Test successful market data collection with yfinance."""
        mock_settings.use_real_data = True
        
        # Mock yfinance ticker
        mock_ticker = Mock()
        mock_hist = pd.DataFrame({'Open': [100.0], 'High': [105.0], 'Low': [99.0], 'Close': [102.0], 'Volume': [1000000]})
        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker
        
        result = self.manager.collect_market_data(['AAPL'])
        assert result['status'] == 'success' and result['symbols_collected'] == 1


class TestFundamentalDataCollection:
    """Test fundamental data collection."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_fundamental_data_dummy_mode(self, mock_settings):
        """Test fundamental data collection in dummy mode."""
        mock_settings.use_real_data = False
        symbols = ['AAPL', 'MSFT']
        result = self.manager.collect_fundamental_data(symbols)
        
        assert result['status'] == 'success' and result['symbols_collected'] == 2 and len(result['data']) == 2
        for item in result['data']:
            assert 'symbol' in item and 'pe_ratio' in item and item['data_source'] == 'dummy'


class TestSentimentDataCollection:
    """Test sentiment data collection."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_sentiment_data_dummy_mode(self, mock_settings):
        """Test sentiment data collection in dummy mode."""
        mock_settings.use_real_data = False
        result = self.manager.collect_sentiment_data(max_articles=5)
        
        assert result['status'] == 'success' and result['article_count'] == 5 and len(result['articles']) == 5
        for article in result['articles']:
            assert 'title' in article and 'sentiment_score' in article and 'sentiment_label' in article
            assert article['sentiment_label'] in ['positive', 'negative', 'neutral']


class TestDatabaseOperations:
    """Test database operations."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.psycopg2')
    def test_get_connection_success(self, mock_psycopg2):
        """Test successful database connection."""
        mock_conn = Mock()
        mock_psycopg2.connect.return_value = mock_conn
        
        with self.manager.get_connection() as conn:
            assert conn == mock_conn
        mock_psycopg2.connect.assert_called_once_with(**self.manager.connection_params)
    
    @patch('src.core.data_manager.psycopg2')
    def test_get_connection_fallback_to_localhost(self, mock_psycopg2):
        """Test database connection fallback to localhost."""
        mock_psycopg2.OperationalError = Exception
        mock_psycopg2.connect.side_effect = [Exception("Connection failed"), Mock()]
        
        self.manager.connection_params['host'] = 'postgres'
        with self.manager.get_connection() as conn:
            assert conn is not None
        
        calls = mock_psycopg2.connect.call_args_list
        assert len(calls) == 2 and calls[1][1]['host'] == 'localhost'
    
    @patch('src.core.data_manager.psycopg2')
    def test_store_market_data_success(self, mock_psycopg2):
        """Test successful market data storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        market_data = {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000, 'timestamp': datetime.now()}
        execution_date = datetime.now()
        
        result = self.manager.store_market_data(market_data, execution_date)
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class TestDataMonitoring:
    """Test enhanced monitoring functionality."""
    
    def setup_method(self): self.manager = DataManager()
    
    def test_monitor_data_quality_success(self):
        """Test successful data quality monitoring."""
        with patch.object(self.manager, '_check_market_data_quality', return_value={'quality_score': 0.9, 'symbols_tested': 3, 'successful_collections': 3, 'errors': 0}):
            with patch.object(self.manager, '_check_news_data_quality', return_value={'quality_score': 0.8, 'articles_collected': 5, 'newsapi_available': False}):
                with patch.object(self.manager, '_check_database_health', return_value={'healthy': True, 'response_time': 0.1}):
                    result = self.manager.monitor_data_quality()
                    
                    assert result['status'] == 'success' and 'overall_quality_score' in result and 'component_metrics' in result
                    assert 0.8 <= result['overall_quality_score'] <= 1.0
                    assert 'market_data' in result['component_metrics'] and 'news_data' in result['component_metrics']
    
    def test_monitor_data_quality_low_scores(self):
        """Test data quality monitoring with low scores."""
        with patch.object(self.manager, '_check_market_data_quality', return_value={'quality_score': 0.5, 'symbols_tested': 3, 'successful_collections': 1, 'errors': 2}):
            with patch.object(self.manager, '_check_news_data_quality', return_value={'quality_score': 0.3, 'articles_collected': 0, 'newsapi_available': False}):
                with patch.object(self.manager, '_check_database_health', return_value={'healthy': False, 'issues': ['Connection timeout']}):
                    with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                        result = self.manager.monitor_data_quality()
                        
                        assert result['status'] == 'success' and result['overall_quality_score'] < 0.7
                        assert result['alerts_generated'] >= 2
                        assert mock_send_alerts.call_count >= 2
    
    def test_monitor_data_freshness_success(self):
        """Test successful data freshness monitoring."""
        mock_conn = MagicMock()
        mock_cursor = Mock()
        recent_time = datetime.now() - timedelta(minutes=30)
        mock_cursor.fetchone.side_effect = [[recent_time], [recent_time]]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        with patch.object(self.manager, 'get_connection', return_value=mock_conn):
            result = self.manager.monitor_data_freshness(max_age_hours=2)
            
            assert result['status'] == 'success' and result['is_all_fresh'] is True and len(result['stale_sources']) == 0
            assert 'freshness_check' in result and 'market_data' in result['freshness_check'] and 'news_data' in result['freshness_check']
    
    def test_monitor_data_freshness_stale_data(self):
        """Test data freshness monitoring with stale data."""
        mock_conn = MagicMock()
        mock_cursor = Mock()
        old_time = datetime.now() - timedelta(hours=5)
        mock_cursor.fetchone.side_effect = [[old_time], [old_time]]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        with patch.object(self.manager, 'get_connection', return_value=mock_conn):
            with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                result = self.manager.monitor_data_freshness(max_age_hours=2)
                
                assert result['status'] == 'success' and result['is_all_fresh'] is False and len(result['stale_sources']) == 2
                assert 'market_data' in result['stale_sources'] and 'news_data' in result['stale_sources']
                mock_send_alerts.assert_called_once()
    
    def test_monitor_system_health_success(self):
        """Test successful system health monitoring."""
        with patch.object(self.manager, '_check_database_performance', return_value={'healthy': True, 'response_time': 0.2, 'market_data_records': 1000, 'news_data_records': 500}):
            with patch.object(self.manager, '_check_api_availability', return_value={'yahoo_finance': {'available': True, 'response_time': 1.0}, 'newsapi': {'available': False, 'reason': 'not_configured'}}):
                with patch.object(self.manager, '_check_collection_systems', return_value={'yfinance_available': True, 'newsapi_available': False, 'errors': 1}):
                    result = self.manager.monitor_system_health()
                    
                    assert result['status'] == 'success' and 'overall_health' in result and 'components' in result
                    assert result['overall_health'] in ['healthy', 'warning', 'critical']
                    assert 'database' in result['components'] and 'apis' in result['components']
    
    def test_monitor_system_health_critical_issues(self):
        """Test system health monitoring with critical issues."""
        with patch.object(self.manager, '_check_database_performance', return_value={'healthy': False, 'error': 'Database connection failed'}):
            with patch.object(self.manager, '_check_api_availability', return_value={'yahoo_finance': {'available': False, 'error': 'Timeout'}}):
                with patch.object(self.manager, '_check_collection_systems', return_value={'errors': 3}):
                    with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                        result = self.manager.monitor_system_health()
                        
                        assert result['status'] == 'success' and result['overall_health'] == 'critical' and len(result['critical_issues']) > 0
                        mock_send_alerts.assert_called()
    
    def test_monitor_system_health_warnings(self):
        """Test system health monitoring with warnings only."""
        with patch.object(self.manager, '_check_database_performance', return_value={'healthy': True, 'response_time': 6.0}):
            with patch.object(self.manager, '_check_api_availability', return_value={'yahoo_finance': {'available': False}}):
                with patch.object(self.manager, '_check_collection_systems', return_value={'errors': 1}):
                    with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                        result = self.manager.monitor_system_health()
                        
                        assert result['status'] == 'success' and result['overall_health'] == 'warning' and len(result['warnings']) > 0
                        mock_send_alerts.assert_called()


class TestHealthCheck:
    """Test health check functionality."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.psycopg2')
    @patch('src.core.data_manager.YFINANCE_AVAILABLE', True)
    def test_health_check_all_healthy(self, mock_psycopg2):
        """Test health check when all components are healthy."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        self.manager.newsapi_client = Mock()
        
        result = self.manager.health_check()
        assert result['status'] == 'healthy' and 'components' in result
        assert result['components']['database'] == 'healthy' and result['components']['yfinance'] == 'available'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_data_manager_factory(self):
        """Test get_data_manager factory function."""
        manager = get_data_manager()
        assert isinstance(manager, DataManager)
        
        config = {'test': 'value'}
        manager_with_config = get_data_manager(config)
        assert manager_with_config.config == config
    
    def test_validate_symbols_function(self):
        """Test validate_symbols utility function."""
        symbols = ['aapl', ' MSFT ', 'googl', '', 'TSLA']
        validated = validate_symbols(symbols)
        
        assert validated == ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        assert len(validated) == 4  # Empty string should be filtered out
    
    def test_validate_symbols_empty_list(self):
        """Test validate_symbols with empty list."""
        result = validate_symbols([])
        assert result == []
        
        result = validate_symbols(['', ' ', '  '])
        assert result == []


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self): self.manager = DataManager()
    
    def test_monitor_data_quality_error_handling(self):
        """Test data quality monitoring error handling."""
        with patch.object(self.manager, '_check_market_data_quality', side_effect=Exception("Test error")):
            with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                result = self.manager.monitor_data_quality()
                
                assert result['status'] == 'error' and 'error' in result
                mock_send_alerts.assert_called()
    
    def test_monitor_data_freshness_error_handling(self):
        """Test data freshness monitoring error handling."""
        with patch.object(self.manager, 'get_connection', side_effect=Exception("DB error")):
            with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                result = self.manager.monitor_data_freshness()
                
                # The method continues even with DB errors, but should record them
                assert result['status'] == 'success'  
                assert 'freshness_check' in result
                # Both market_data and news_data should have error status
                assert result['freshness_check']['market_data'].get('status') == 'error'
                assert result['freshness_check']['news_data'].get('status') == 'error'
    
    def test_monitor_system_health_error_handling(self):
        """Test system health monitoring error handling."""
        with patch.object(self.manager, '_check_database_performance', side_effect=Exception("Test error")):
            with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                result = self.manager.monitor_system_health()
                
                assert result['status'] == 'error' and 'error' in result
                mock_send_alerts.assert_called()
    
    def test_monitor_data_collection_performance_success(self):
        """Test successful data collection performance monitoring."""
        mock_market_result = {'status': 'success', 'symbols_collected': 3, 'total_symbols': 3}
        mock_fundamental_result = {'status': 'success', 'data': [{'symbol': 'AAPL'}, {'symbol': 'SPY'}, {'symbol': 'QQQ'}]}
        mock_sentiment_result = {'status': 'success', 'article_count': 5}
        mock_health_result = {'status': 'healthy'}
        
        with patch.object(self.manager, 'collect_market_data', return_value=mock_market_result):
            with patch.object(self.manager, 'collect_fundamental_data', return_value=mock_fundamental_result):
                with patch.object(self.manager, 'collect_sentiment_data', return_value=mock_sentiment_result):
                    with patch.object(self.manager, 'health_check', return_value=mock_health_result):
                        result = self.manager.monitor_data_collection_performance()
                        
                        assert result['status'] == 'success' and 'overall_performance_score' in result and 'metrics' in result
                        assert result['overall_performance_score'] == 1.0  # Perfect scores
                        assert result['metrics']['market_success_rate'] == 1.0
                        assert result['metrics']['fundamental_success_rate'] == 1.0
    
    def test_monitor_data_collection_performance_degraded(self):
        """Test data collection performance monitoring with degraded performance."""
        mock_market_result = {'status': 'success', 'symbols_collected': 1, 'total_symbols': 3}  # 33% success
        mock_fundamental_result = {'status': 'success', 'data': [{'symbol': 'AAPL'}]}  # 33% success
        mock_sentiment_result = {'status': 'success', 'article_count': 2}
        mock_health_result = {'status': 'degraded'}
        
        with patch.object(self.manager, 'collect_market_data', return_value=mock_market_result):
            with patch.object(self.manager, 'collect_fundamental_data', return_value=mock_fundamental_result):
                with patch.object(self.manager, 'collect_sentiment_data', return_value=mock_sentiment_result):
                    with patch.object(self.manager, 'health_check', return_value=mock_health_result):
                        with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                            result = self.manager.monitor_data_collection_performance()
                            
                            assert result['status'] == 'success' and result['overall_performance_score'] < 0.7
                            assert result['metrics']['market_success_rate'] < 1.0
                            assert result['metrics']['fundamental_success_rate'] < 1.0
                            mock_send_alerts.assert_called_once()
    
    def test_monitor_data_collection_performance_error_handling(self):
        """Test data collection performance monitoring error handling."""
        with patch.object(self.manager, 'collect_market_data', side_effect=Exception("Collection error")):
            with patch('src.core.data_manager.send_alerts') as mock_send_alerts:
                result = self.manager.monitor_data_collection_performance()
                
                assert result['status'] == 'error' and 'error' in result
                mock_send_alerts.assert_called()


class TestIntegration:
    """Integration tests for DataManager with monitoring."""
    
    def setup_method(self): self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_full_monitoring_workflow(self, mock_settings):
        """Test complete monitoring workflow."""
        mock_settings.use_real_data = False
        
        # Run all monitoring functions
        quality_result = self.manager.monitor_data_quality()
        freshness_result = self.manager.monitor_data_freshness()
        health_result = self.manager.monitor_system_health()
        performance_result = self.manager.monitor_data_collection_performance()
        
        # Verify all monitoring functions return valid results
        assert quality_result['status'] == 'success' and 'overall_quality_score' in quality_result
        assert freshness_result['status'] in ['success', 'error'] and 'freshness_check' in freshness_result or 'error' in freshness_result
        assert health_result['status'] == 'success' and 'overall_health' in health_result
        assert performance_result['status'] == 'success' and 'overall_performance_score' in performance_result and 'metrics' in performance_result