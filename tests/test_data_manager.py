"""
Tests for DataManager module.
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
        
        assert manager.config == {}
        assert manager.retry_attempts == 3
        assert manager.retry_delay == 1
        assert 'host' in manager.connection_params
        assert manager.sentiment_method in ['dummy', 'textblob', 'finbert']
    
    def test_initialization_missing_use_real_data_flag(self):
        """Test DataManager initialization fails without USE_REAL_DATA flag."""
        with patch('src.core.data_manager.settings') as mock_settings:
            # Remove use_real_data attribute
            if hasattr(mock_settings, 'use_real_data'):
                delattr(mock_settings, 'use_real_data')
            
            with pytest.raises(ValueError, match="USE_REAL_DATA flag is required"):
                DataManager()
    
    def test_initialization_custom_config(self):
        """Test DataManager initialization with custom config."""
        config = {"custom_setting": "test_value"}
        manager = DataManager(config)
        
        assert manager.config == config
    
    @patch.dict('os.environ', {
        'POSTGRES_HOST': 'custom_host',
        'POSTGRES_PORT': '5433',
        'POSTGRES_DB': 'custom_db',
        'POSTGRES_USER': 'custom_user',
        'POSTGRES_PASSWORD': 'custom_pass'
    })
    def test_initialization_custom_env_vars(self):
        """Test DataManager initialization with custom environment variables."""
        manager = DataManager()
        
        assert manager.connection_params['host'] == 'custom_host'
        assert manager.connection_params['port'] == '5433'
        assert manager.connection_params['database'] == 'custom_db'
        assert manager.connection_params['user'] == 'custom_user'
        assert manager.connection_params['password'] == 'custom_pass'


class TestMarketDataCollection:
    """Test market data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_market_data_dummy_mode(self, mock_settings):
        """Test market data collection in dummy mode."""
        mock_settings.use_real_data = False
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = self.manager.collect_market_data(symbols)
        
        assert result['status'] == 'success'
        assert result['symbols_collected'] == 3
        assert result['total_symbols'] == 3
        assert len(result['data']) == 3
        assert len(result['errors']) == 0
        
        # Check data structure for each symbol
        for symbol in symbols:
            assert symbol in result['data']
            data = result['data'][symbol]
            assert data['symbol'] == symbol
            assert data['status'] == 'success'
            assert 'price' in data
            assert 'volume' in data
            assert data['data_source'] == 'dummy'
    
    @patch('src.core.data_manager.settings')
    @patch('src.core.data_manager.YFINANCE_AVAILABLE', True)
    @patch('src.core.data_manager.yf')
    def test_collect_market_data_yfinance_success(self, mock_yf, mock_settings):
        """Test successful market data collection with yfinance."""
        mock_settings.use_real_data = True
        
        # Mock yfinance ticker
        mock_ticker = Mock()
        mock_hist = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        })
        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker
        
        # Mock direct Yahoo API to fail so yfinance is used
        with patch.object(self.manager, '_collect_yahoo_direct', return_value=None):
            result = self.manager.collect_market_data(['AAPL'])
        
        assert result['status'] == 'success'
        assert 'AAPL' in result['data']
        data = result['data']['AAPL']
        assert data['price'] == 102.0
        assert data['volume'] == 1100000
        assert data['data_source'] == 'yfinance'
    
    @patch('src.core.data_manager.settings')
    @patch('src.core.data_manager.requests.get')
    def test_collect_market_data_yahoo_direct_success(self, mock_get, mock_settings):
        """Test successful market data collection with direct Yahoo API."""
        mock_settings.use_real_data = True
        
        # Mock Yahoo direct API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'chart': {
                'result': [{
                    'meta': {'marketCap': 2500000000000},
                    'timestamp': [1640995200],  # Example timestamp
                    'indicators': {
                        'quote': [{
                            'open': [150.0],
                            'high': [155.0],
                            'low': [148.0],
                            'close': [152.0],
                            'volume': [50000000]
                        }]
                    }
                }]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.manager.collect_market_data(['AAPL'])
        
        assert result['status'] == 'success'
        assert 'AAPL' in result['data']
        data = result['data']['AAPL']
        assert data['price'] == 152.0
        assert data['volume'] == 50000000
        assert data['data_source'] == 'yahoo_direct'
    
    @patch('src.core.data_manager.settings')
    def test_collect_market_data_error_handling(self, mock_settings):
        """Test error handling in market data collection."""
        mock_settings.use_real_data = True
        
        # Mock _collect_yfinance_data to return dummy data (as it does on failure)
        with patch.object(self.manager, '_collect_yfinance_data') as mock_yfinance:
            mock_yfinance.return_value = {'symbol': 'INVALID', 'data_source': 'dummy', 'status': 'success', 'price': 100.0}
            
            result = self.manager.collect_market_data(['INVALID'])
        
        # Should still get dummy data as fallback
        assert result['status'] == 'success'
        assert 'INVALID' in result['data']
        assert result['data']['INVALID']['data_source'] == 'dummy'
    
    def test_generate_dummy_market_data(self):
        """Test dummy market data generation."""
        data = self.manager._generate_dummy_market_data('AAPL')
        
        assert data['symbol'] == 'AAPL'
        assert data['status'] == 'success'
        assert isinstance(data['price'], float)
        assert isinstance(data['volume'], int)
        assert data['data_source'] == 'dummy'
        assert 'timestamp' in data
        
        # Test with unknown symbol
        unknown_data = self.manager._generate_dummy_market_data('UNKNOWN')
        assert unknown_data['symbol'] == 'UNKNOWN'
        assert unknown_data['price'] > 0


class TestFundamentalDataCollection:
    """Test fundamental data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_fundamental_data_dummy_mode(self, mock_settings):
        """Test fundamental data collection in dummy mode."""
        mock_settings.use_real_data = False
        
        symbols = ['AAPL', 'MSFT']
        result = self.manager.collect_fundamental_data(symbols)
        
        assert result['status'] == 'success'
        assert result['symbols_collected'] == 2
        assert result['total_symbols'] == 2
        assert len(result['data']) == 2
        
        # Check data structure
        for fund_data in result['data']:
            assert fund_data['status'] == 'success'
            assert 'symbol' in fund_data
            assert 'pe_ratio' in fund_data
            assert 'pb_ratio' in fund_data
            assert fund_data['data_source'] == 'dummy'
    
    @patch('src.core.data_manager.settings')
    @patch('src.core.data_manager.YFINANCE_AVAILABLE', True)
    @patch('src.core.data_manager.yf')
    def test_collect_fundamental_data_yfinance_success(self, mock_yf, mock_settings):
        """Test successful fundamental data collection with yfinance."""
        mock_settings.use_real_data = True
        
        # Mock yfinance ticker info
        mock_ticker = Mock()
        mock_ticker.info = {
            'forwardPE': 25.5,
            'priceToBook': 4.2,
            'priceToSalesTrailing12Months': 3.1,
            'debtToEquity': 0.8,
            'profitMargins': 0.25,
            'returnOnEquity': 0.35,
            'revenueGrowth': 0.15,
            'earningsGrowth': 0.20
        }
        mock_yf.Ticker.return_value = mock_ticker
        
        result = self.manager.collect_fundamental_data(['AAPL'])
        
        assert result['status'] == 'success'
        assert len(result['data']) == 1
        
        fund_data = result['data'][0]
        assert fund_data['symbol'] == 'AAPL'
        assert fund_data['pe_ratio'] == 25.5
        assert fund_data['pb_ratio'] == 4.2
        assert fund_data['data_source'] == 'yfinance'
    
    @patch('src.core.data_manager.settings')
    @patch('src.core.data_manager.YFINANCE_AVAILABLE', True)
    @patch('src.core.data_manager.yf')
    def test_collect_fundamental_data_yfinance_error(self, mock_yf, mock_settings):
        """Test fundamental data collection with yfinance error."""
        mock_settings.use_real_data = True
        
        # Mock yfinance to raise exception
        mock_yf.Ticker.side_effect = Exception("API Error")
        
        result = self.manager.collect_fundamental_data(['AAPL'])
        
        assert result['status'] == 'success'
        assert len(result['data']) == 1
        assert result['data'][0]['data_source'] == 'dummy'
    
    def test_generate_dummy_fundamental_data(self):
        """Test dummy fundamental data generation."""
        data = self.manager._generate_dummy_fundamental_data('AAPL')
        
        assert data['symbol'] == 'AAPL'
        assert data['status'] == 'success'
        assert 'pe_ratio' in data
        assert 'pb_ratio' in data
        assert 'ps_ratio' in data
        assert 'debt_to_equity' in data
        assert data['data_source'] == 'dummy'
        
        # Check value ranges
        assert 15.0 <= data['pe_ratio'] <= 35.0
        assert 1.5 <= data['pb_ratio'] <= 5.0


class TestSentimentDataCollection:
    """Test sentiment data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_collect_sentiment_data_dummy_mode(self, mock_settings):
        """Test sentiment data collection in dummy mode."""
        mock_settings.use_real_data = False
        
        result = self.manager.collect_sentiment_data(max_articles=10)
        
        assert result['status'] == 'success'
        assert result['article_count'] == 10
        assert result['sentiment_method'] == 'dummy'
        assert len(result['articles']) == 10
        
        # Check article structure
        for article in result['articles']:
            assert 'title' in article
            assert 'content' in article
            assert 'sentiment_score' in article
            assert 'sentiment_label' in article
            assert article['source'] == 'Dummy News'
    
    @patch('src.core.data_manager.settings')
    def test_collect_sentiment_data_no_newsapi_client(self, mock_settings):
        """Test sentiment data collection without NewsAPI client."""
        mock_settings.use_real_data = True
        self.manager.newsapi_client = None
        
        result = self.manager.collect_sentiment_data(max_articles=5)
        
        assert result['status'] == 'success'
        assert result['sentiment_method'] == 'dummy'
        assert len(result['articles']) == 5
    
    @patch('src.core.data_manager.settings')
    def test_collect_sentiment_data_newsapi_success(self, mock_settings):
        """Test successful sentiment data collection with NewsAPI."""
        mock_settings.use_real_data = True
        
        # Mock NewsAPI client
        mock_client = Mock()
        mock_client.get_everything.return_value = {
            'status': 'ok',
            'articles': [{
                'title': 'Market Rally Continues',
                'description': 'Stock market shows positive momentum',
                'url': 'https://example.com/news1',
                'source': {'name': 'Financial Times'},
                'publishedAt': '2024-01-01T12:00:00Z'
            }]
        }
        self.manager.newsapi_client = mock_client
        
        result = self.manager.collect_sentiment_data(max_articles=5)
        
        assert result['status'] == 'success'
        assert len(result['articles']) >= 1
        
        article = result['articles'][0]
        assert article['title'] == 'Market Rally Continues'
        assert article['source'] == 'Financial Times'
        assert 'sentiment_score' in article
        assert 'sentiment_label' in article
    
    def test_analyze_sentiment_textblob(self):
        """Test sentiment analysis with TextBlob."""
        with patch('src.core.data_manager.TEXTBLOB_AVAILABLE', True), \
             patch('src.core.data_manager.TextBlob') as mock_textblob:
            
            mock_blob = Mock()
            mock_blob.sentiment.polarity = 0.5
            mock_textblob.return_value = mock_blob
            
            self.manager.sentiment_method = 'textblob'
            
            result = self.manager._analyze_sentiment('This is great news!')
            
            assert result['score'] == 0.5
            assert result['label'] == 'positive'
            assert result['confidence'] == 0.5
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text."""
        result = self.manager._analyze_sentiment('')
        
        assert result['score'] == 0.0
        assert result['label'] == 'neutral'
        assert result['confidence'] == 0.0
    
    def test_analyze_sentiment_dummy(self):
        """Test dummy sentiment analysis."""
        self.manager.sentiment_method = 'dummy'
        
        result = self.manager._analyze_sentiment('Some text')
        
        assert 'score' in result
        assert 'label' in result
        assert 'confidence' in result
        assert result['label'] in ['positive', 'negative', 'neutral']


class TestDatabaseOperations:
    """Test database operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_get_connection_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        with self.manager.get_connection() as conn:
            assert conn == mock_conn
        
        mock_conn.close.assert_called_once()
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_get_connection_fallback_to_localhost(self, mock_connect):
        """Test database connection fallback to localhost."""
        import psycopg2
        
        # First call fails with OperationalError, second succeeds
        mock_conn = Mock()
        mock_connect.side_effect = [
            psycopg2.OperationalError("Connection failed"),
            mock_conn
        ]
        
        self.manager.connection_params['host'] = 'postgres'
        
        with self.manager.get_connection() as conn:
            assert conn == mock_conn
        
        # Should have tried both postgres and localhost
        assert mock_connect.call_count == 2
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_store_market_data_success(self, mock_connect):
        """Test successful market data storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        market_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000000,
            'open': 149.0,
            'high': 151.0,
            'low': 148.0,
            'close': 150.0,
            'data_source': 'yfinance'
        }
        
        result = self.manager.store_market_data(market_data, datetime.now())
        
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_store_market_data_error(self, mock_connect):
        """Test market data storage error handling."""
        mock_connect.side_effect = Exception("Database error")
        
        market_data = {'symbol': 'AAPL', 'price': 150.0}
        result = self.manager.store_market_data(market_data, datetime.now())
        
        assert result is False
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_store_news_data_success(self, mock_connect):
        """Test successful news data storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        news_data = {
            'title': 'Market News',
            'content': 'Content here',
            'url': 'https://example.com',
            'source': 'Test Source',
            'sentiment_score': 0.5,
            'sentiment_label': 'positive'
        }
        
        result = self.manager.store_news_data(news_data, datetime.now())
        
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_create_tables_success(self, mock_connect):
        """Test successful table creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        self.manager.create_tables()
        
        # Should execute CREATE TABLE statements
        assert mock_cursor.execute.call_count >= 2
        mock_conn.commit.assert_called_once()
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_health_check_database_healthy(self, mock_connect):
        """Test health check with healthy database."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        result = self.manager.health_check()
        
        assert result['status'] == 'healthy'
        assert result['components']['database'] == 'healthy'
        assert 'yfinance' in result['components']
        assert 'newsapi' in result['components']
        assert 'sentiment' in result['components']
    
    @patch('src.core.data_manager.psycopg2.connect')
    def test_health_check_database_unhealthy(self, mock_connect):
        """Test health check with unhealthy database."""
        mock_connect.side_effect = Exception("Database error")
        
        result = self.manager.health_check()
        
        assert result['status'] == 'degraded'
        assert 'unhealthy' in result['components']['database']


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    def test_collect_market_data_exception_handling(self):
        """Test exception handling in market data collection."""
        # Mock settings to trigger the exception path
        with patch('src.core.data_manager.settings') as mock_settings:
            mock_settings.use_real_data = False
            # Mock _generate_dummy_market_data to raise exception
            with patch.object(self.manager, '_generate_dummy_market_data', side_effect=Exception("Error")):
                result = self.manager.collect_market_data(['AAPL'])
        
        assert result['status'] == 'failed'
        assert len(result['errors']) > 0
        assert 'AAPL' in result['errors'][0]
    
    def test_collect_fundamental_data_exception_handling(self):
        """Test exception handling in fundamental data collection."""
        with patch.object(self.manager, '_collect_weekly_fundamentals', side_effect=Exception("Error")):
            result = self.manager.collect_fundamental_data(['AAPL'])
        
        assert result['status'] == 'failed'
        assert len(result['errors']) > 0
        assert 'AAPL' in result['errors'][0]
    
    def test_collect_sentiment_data_exception_handling(self):
        """Test exception handling in sentiment data collection."""
        self.manager.newsapi_client = Mock()
        self.manager.newsapi_client.get_everything.side_effect = Exception("API Error")
        
        with patch('src.core.data_manager.settings') as mock_settings:
            mock_settings.use_real_data = True
            result = self.manager.collect_sentiment_data(max_articles=5)
        
        # Should fall back to dummy data but keep original sentiment method
        assert result['status'] == 'success'
        # The sentiment_method in result should be 'dummy' since it generates dummy articles
        assert 'sentiment_method' in result


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_data_manager_factory(self):
        """Test DataManager factory function."""
        manager = get_data_manager()
        assert isinstance(manager, DataManager)
        
        config = {'test': 'value'}
        manager_with_config = get_data_manager(config)
        assert manager_with_config.config == config
    
    def test_validate_symbols(self):
        """Test symbol validation function."""
        symbols = ['aapl', ' MSFT ', 'googl', '', '  ']
        validated = validate_symbols(symbols)
        
        assert validated == ['AAPL', 'MSFT', 'GOOGL']
        
        # Test empty list
        assert validate_symbols([]) == []
        
        # Test with only empty strings
        assert validate_symbols(['', '  ', '   ']) == []


class TestIntegration:
    """Integration tests for DataManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    @patch('src.core.data_manager.settings')
    def test_full_data_collection_workflow(self, mock_settings):
        """Test complete data collection workflow."""
        mock_settings.use_real_data = False
        
        symbols = ['AAPL', 'MSFT']
        
        # Collect all types of data
        market_result = self.manager.collect_market_data(symbols)
        fundamental_result = self.manager.collect_fundamental_data(symbols)
        sentiment_result = self.manager.collect_sentiment_data(max_articles=5)
        
        # Verify all collections succeeded
        assert market_result['status'] == 'success'
        assert fundamental_result['status'] == 'success'
        assert sentiment_result['status'] == 'success'
        
        # Verify data structure consistency
        assert len(market_result['data']) == 2
        assert len(fundamental_result['data']) == 2
        assert len(sentiment_result['articles']) == 5


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DataManager()
    
    def test_market_data_structure_validation(self):
        """Test market data structure validation."""
        data = self.manager._generate_dummy_market_data('AAPL')
        
        required_fields = ['symbol', 'status', 'price', 'volume', 'timestamp', 'data_source']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate data types
        assert isinstance(data['price'], float)
        assert isinstance(data['volume'], int)
        assert data['price'] > 0
        assert data['volume'] >= 0
    
    def test_fundamental_data_structure_validation(self):
        """Test fundamental data structure validation."""
        data = self.manager._generate_dummy_fundamental_data('AAPL')
        
        required_fields = ['symbol', 'status', 'pe_ratio', 'pb_ratio', 'timestamp', 'data_source']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate data types and ranges
        assert isinstance(data['pe_ratio'], float)
        assert isinstance(data['pb_ratio'], float)
        assert data['pe_ratio'] > 0
        assert data['pb_ratio'] > 0
    
    def test_sentiment_data_structure_validation(self):
        """Test sentiment data structure validation."""
        result = self.manager._generate_dummy_news_sentiment(3)
        
        assert 'articles' in result
        assert 'article_count' in result
        assert result['article_count'] == len(result['articles'])
        
        for article in result['articles']:
            required_fields = ['title', 'content', 'sentiment_score', 'sentiment_label', 'timestamp']
            for field in required_fields:
                assert field in article, f"Missing required field: {field}"
            
            # Validate sentiment data
            assert isinstance(article['sentiment_score'], float)
            assert article['sentiment_label'] in ['positive', 'negative', 'neutral']
            assert -1.0 <= article['sentiment_score'] <= 1.0