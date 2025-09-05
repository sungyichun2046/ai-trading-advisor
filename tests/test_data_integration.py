"""Integration tests for data collection pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import psycopg2
import os

from src.data.database import DatabaseManager, MarketDataStorage, NewsStorage
from src.data.collectors import MarketDataCollector, NewsCollector


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture(scope="session")
    def db_connection_params(self):
        """Database connection parameters for testing."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'),
            'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
        }

    def test_database_manager_init(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager()
        assert manager.connection_params is not None

    @patch('psycopg2.connect')
    def test_create_tables(self, mock_connect):
        """Test table creation."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        manager = DatabaseManager()
        manager.create_tables()
        
        # Verify that cursor.execute was called for table creation
        assert mock_cursor.execute.call_count >= 4  # At least 4 tables
        mock_conn.commit.assert_called_once()

    @patch('psycopg2.connect')
    def test_market_data_storage(self, mock_connect):
        """Test market data storage operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        storage = MarketDataStorage()
        
        # Mock market data
        market_data = {
            "AAPL": {
                "symbol": "AAPL",
                "status": "success",
                "price": 150.0,
                "volume": 1000000,
                "open": 149.0,
                "high": 152.0,
                "low": 148.0,
                "close": 151.0,
                "market_cap": 2500000000000,
                "pe_ratio": 25.5,
                "data_source": "yfinance",
                "timestamp": "2024-01-01T10:00:00"
            }
        }
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        result = storage.store_market_data(market_data, execution_date)
        
        # Verify storage was attempted
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
        assert result["count"] == 1

    @patch('psycopg2.connect')
    def test_news_data_storage(self, mock_connect):
        """Test news data storage operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        storage = NewsStorage()
        
        # Mock news data
        news_data = {
            "status": "success",
            "articles": [
                {
                    "title": "Test Article",
                    "content": "Test content",
                    "source": "Test Source",
                    "sentiment_score": 0.5,
                    "data_source": "newsapi",
                    "timestamp": "2024-01-01T10:00:00"
                }
            ]
        }
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        result = storage.store_news_data(news_data, execution_date)
        
        # Verify storage was attempted
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
        assert result["count"] == 1

    @patch('psycopg2.connect')
    def test_database_connection_failure(self, mock_connect):
        """Test handling of database connection failures."""
        mock_connect.side_effect = psycopg2.Error("Connection failed")
        
        storage = MarketDataStorage()
        
        market_data = {
            "AAPL": {"status": "success", "price": 150.0}
        }
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        
        # Should handle connection failure gracefully
        with pytest.raises(psycopg2.Error):
            storage.store_market_data(market_data, execution_date)


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    @patch('src.data.collectors.settings')
    @patch('psycopg2.connect')
    def test_complete_pipeline_dummy_mode(self, mock_connect, mock_settings):
        """Test complete pipeline in dummy mode."""
        # Configure dummy mode
        mock_settings.use_real_data = False
        
        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Initialize components
        market_collector = MarketDataCollector()
        news_collector = NewsCollector()
        market_storage = MarketDataStorage()
        news_storage = NewsStorage()
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        
        # Step 1: Collect market data
        market_data = {}
        for symbol in ["SPY", "AAPL"]:
            market_data[symbol] = market_collector.collect_real_time_data(symbol)
        
        # Verify market data collection
        assert len(market_data) == 2
        assert all(data["status"] == "success" for data in market_data.values())
        assert all(data["data_source"] == "dummy" for data in market_data.values())
        
        # Step 2: Collect news data
        news_articles = news_collector.collect_financial_news()
        sentiment_results = news_collector.analyze_sentiment(news_articles)
        
        # Verify news collection
        assert sentiment_results["status"] == "success"
        assert sentiment_results["data_source"] == "dummy"
        assert len(sentiment_results["articles"]) >= 3
        
        # Step 3: Store data
        market_result = market_storage.store_market_data(market_data, execution_date)
        news_result = news_storage.store_news_data(sentiment_results, execution_date)
        
        # Verify storage
        assert market_result["count"] == 2
        assert news_result["count"] >= 3

    @patch('src.data.collectors.yf.Ticker')
    @patch('src.data.collectors.NewsApiClient')
    @patch('src.data.collectors.pipeline')
    @patch('src.data.collectors.settings')
    @patch('psycopg2.connect')
    def test_complete_pipeline_real_mode(self, mock_connect, mock_settings, 
                                        mock_pipeline, mock_newsapi, mock_yf):
        """Test complete pipeline with mocked real APIs."""
        # Configure real mode
        mock_settings.use_real_data = True
        mock_settings.newsapi_key = "test_key"
        mock_settings.max_news_articles = 2
        
        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock yfinance
        import pandas as pd
        mock_hist = pd.DataFrame({
            'Open': [150.0], 'High': [152.0], 'Low': [149.0],
            'Close': [151.0], 'Volume': [1000000]
        })
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker_instance.info = {'currentPrice': 151.0, 'marketCap': 2500000000000}
        mock_yf.return_value = mock_ticker_instance
        
        # Mock NewsAPI
        mock_articles = {
            'articles': [
                {
                    'title': 'Test Article',
                    'description': 'Test description', 
                    'source': {'name': 'Test Source'},
                    'publishedAt': '2024-01-01T10:00:00Z'
                }
            ]
        }
        mock_client = Mock()
        mock_client.get_everything.return_value = mock_articles
        mock_newsapi.return_value = mock_client
        
        # Mock sentiment analyzer
        mock_analyzer = Mock()
        mock_analyzer.return_value = [[
            {'label': 'positive', 'score': 0.7},
            {'label': 'negative', 'score': 0.2}
        ]]
        mock_pipeline.return_value = mock_analyzer
        
        # Initialize components
        market_collector = MarketDataCollector()
        news_collector = NewsCollector()
        market_storage = MarketDataStorage()
        news_storage = NewsStorage()
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        
        # Run pipeline
        market_data = {"AAPL": market_collector.collect_real_time_data("AAPL")}
        news_articles = news_collector.collect_financial_news()
        sentiment_results = news_collector.analyze_sentiment(news_articles)
        
        # Store data
        market_result = market_storage.store_market_data(market_data, execution_date)
        news_result = news_storage.store_news_data(sentiment_results, execution_date)
        
        # Verify real API integration
        assert market_data["AAPL"]["data_source"] == "yfinance"
        assert sentiment_results["data_source"] == "finbert"
        assert market_result["count"] == 1
        assert news_result["count"] == 1

    def test_data_validation_integration(self):
        """Test data validation integration."""
        from src.data.processors import DataValidator
        
        # This would be implemented when DataValidator is created
        # For now, just verify the import works
        validator = DataValidator()
        assert validator is not None


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @patch('src.data.collectors.settings')
    def test_collector_resilience(self, mock_settings):
        """Test collector resilience to various failures."""
        mock_settings.use_real_data = False
        
        collector = MarketDataCollector()
        
        # Test with invalid symbol
        result = collector.collect_real_time_data("INVALID_SYMBOL")
        assert result is not None
        assert result["status"] == "success"  # Falls back to dummy data
        
        # Test with empty symbol
        result = collector.collect_real_time_data("")
        assert result is not None

    @patch('psycopg2.connect')
    def test_storage_error_handling(self, mock_connect):
        """Test storage error handling."""
        # Mock connection that fails during operation
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = psycopg2.Error("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        storage = MarketDataStorage()
        
        market_data = {
            "AAPL": {"status": "success", "price": 150.0}
        }
        
        execution_date = datetime(2024, 1, 1, 10, 0, 0)
        result = storage.store_market_data(market_data, execution_date)
        
        # Should handle error gracefully
        assert result["count"] == 0
        assert "error" in result

    @patch('src.data.collectors.settings')
    def test_mixed_success_failure_scenarios(self, mock_settings):
        """Test handling of mixed success/failure scenarios."""
        mock_settings.use_real_data = False
        
        collector = MarketDataCollector()
        
        # Simulate collecting data for multiple symbols
        symbols = ["SPY", "AAPL", "INVALID"]
        results = {}
        
        for symbol in symbols:
            results[symbol] = collector.collect_real_time_data(symbol)
        
        # All should succeed in dummy mode
        assert all(result["status"] == "success" for result in results.values())
        assert len(results) == 3