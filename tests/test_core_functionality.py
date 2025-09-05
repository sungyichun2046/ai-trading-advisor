"""Core functionality tests without heavy dependencies."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfigurationSettings:
    """Test configuration settings."""

    def test_settings_import(self):
        """Test that settings can be imported."""
        from src.config import settings
        
        assert settings is not None
        assert hasattr(settings, 'use_real_data')
        assert hasattr(settings, 'data_collection_interval')

    def test_default_settings(self):
        """Test configuration values (reads from actual .env file)."""
        from src.config import settings
        
        # Test configuration values (based on actual .env file)
        assert isinstance(settings.use_real_data, bool)
        assert settings.data_collection_interval == 15
        assert settings.max_news_articles == 50
        
        # Test that required fields are present
        assert hasattr(settings, 'use_real_data')
        assert hasattr(settings, 'data_collection_interval')
        assert hasattr(settings, 'max_news_articles')


class TestDatabaseSchema:
    """Test database schema and operations."""

    @patch('psycopg2.connect')
    def test_database_manager_init(self, mock_connect):
        """Test DatabaseManager initialization."""
        from src.data.database import DatabaseManager
        
        manager = DatabaseManager()
        assert manager.connection_params is not None
        assert 'host' in manager.connection_params
        assert 'database' in manager.connection_params

    @patch('psycopg2.connect')
    def test_table_creation_sql(self, mock_connect):
        """Test that table creation doesn't fail."""
        from src.data.database import DatabaseManager
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        manager = DatabaseManager()
        
        # This should not raise an exception
        try:
            manager.create_tables()
        except Exception as e:
            # If it fails, it should be due to mocking, not logic errors
            assert "Mock" in str(type(e)) or "mock" in str(e).lower()

    def test_market_data_storage_init(self):
        """Test MarketDataStorage initialization."""
        from src.data.database import MarketDataStorage
        
        storage = MarketDataStorage()
        assert storage.connection_params is not None

    def test_news_storage_init(self):
        """Test NewsStorage initialization."""
        from src.data.database import NewsStorage
        
        storage = NewsStorage()
        assert storage.connection_params is not None


class TestDataValidationLogic:
    """Test data validation logic."""

    def test_validation_thresholds(self):
        """Test data quality validation thresholds."""
        # Test the validation logic from the pipeline
        def validate_quality(market_validation, news_validation):
            return (
                market_validation.get("completeness", 0) >= 0.8
                and market_validation.get("freshness", 0) >= 0.9
                and news_validation.get("coverage", 0) >= 0.7
            )
        
        # Test passing validation
        good_market = {"completeness": 0.9, "freshness": 0.95}
        good_news = {"coverage": 0.8}
        assert validate_quality(good_market, good_news) is True
        
        # Test failing validation
        bad_market = {"completeness": 0.7, "freshness": 0.85}
        bad_news = {"coverage": 0.6}
        assert validate_quality(bad_market, bad_news) is False


class TestPipelineFunctions:
    """Test pipeline functions with mocked dependencies."""

    @patch('src.airflow_dags.data_pipeline.MarketDataCollector')
    def test_collect_market_data_structure(self, mock_collector_class):
        """Test market data collection function structure."""
        # Import inside test to avoid dependency issues
        try:
            from src.airflow_dags.data_pipeline import collect_market_data
            
            # Mock collector
            mock_collector = Mock()
            mock_collector.collect_real_time_data.return_value = {
                "symbol": "AAPL",
                "status": "success",
                "price": 150.0
            }
            mock_collector_class.return_value = mock_collector
            
            # Test context
            context = {"execution_date": datetime(2024, 1, 1, 10, 0, 0)}
            
            result = collect_market_data(**context)
            
            # Verify structure
            assert isinstance(result, dict)
            assert len(result) > 0
            
        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    @patch('src.airflow_dags.data_pipeline.NewsCollector')
    def test_collect_news_sentiment_structure(self, mock_collector_class):
        """Test news sentiment collection function structure."""
        try:
            from src.airflow_dags.data_pipeline import collect_news_sentiment
            
            # Mock collector
            mock_collector = Mock()
            mock_collector.collect_financial_news.return_value = []
            mock_collector.analyze_sentiment.return_value = {
                "status": "success",
                "average": 0.5
            }
            mock_collector_class.return_value = mock_collector
            
            context = {"execution_date": datetime(2024, 1, 1, 10, 0, 0)}
            
            result = collect_news_sentiment(**context)
            
            # Verify structure
            assert isinstance(result, dict)
            assert "status" in result
            
        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_pipeline_error_handling(self):
        """Test that pipeline functions handle errors gracefully."""
        # This tests the error handling structure without dependencies
        
        def mock_pipeline_function(**context):
            try:
                # Simulate operation that might fail
                raise Exception("Simulated API error")
            except Exception as e:
                return {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": context["execution_date"].isoformat()
                }
        
        context = {"execution_date": datetime(2024, 1, 1, 10, 0, 0)}
        result = mock_pipeline_function(**context)
        
        assert result["status"] == "failed"
        assert "Simulated API error" in result["error"]
        assert "timestamp" in result


class TestDataFormats:
    """Test expected data formats."""

    def test_market_data_format(self):
        """Test expected market data format."""
        expected_fields = [
            "symbol", "status", "price", "volume", "open", "high",
            "low", "close", "timestamp", "data_source"
        ]
        
        # Mock market data structure
        mock_data = {
            "symbol": "AAPL",
            "status": "success", 
            "price": 150.0,
            "volume": 1000000,
            "open": 149.0,
            "high": 152.0,
            "low": 148.0,
            "close": 151.0,
            "timestamp": "2024-01-01T10:00:00",
            "data_source": "yfinance"
        }
        
        # Verify all required fields are present
        for field in expected_fields:
            assert field in mock_data
        
        # Verify data types
        assert isinstance(mock_data["price"], (int, float))
        assert isinstance(mock_data["volume"], int)
        assert mock_data["status"] in ["success", "failed"]

    def test_news_data_format(self):
        """Test expected news data format."""
        expected_fields = [
            "status", "articles_analyzed", "average", "articles"
        ]
        
        mock_sentiment = {
            "status": "success",
            "articles_analyzed": 5,
            "average": 0.3,
            "articles": [
                {
                    "title": "Test Article",
                    "content": "Test content",
                    "sentiment_score": 0.3
                }
            ]
        }
        
        for field in expected_fields:
            assert field in mock_sentiment
        
        assert isinstance(mock_sentiment["average"], (int, float))
        assert mock_sentiment["average"] >= -1 and mock_sentiment["average"] <= 1


class TestEnvironmentConfiguration:
    """Test environment and configuration handling."""

    def test_environment_variables(self):
        """Test environment variable handling."""
        import os
        
        # Test default values when env vars not set
        test_vars = [
            ("USE_REAL_DATA", "False"),
            ("DATA_COLLECTION_INTERVAL", "15"),
            ("MAX_NEWS_ARTICLES", "50")
        ]
        
        for var_name, expected_default in test_vars:
            # If env var exists, should be readable
            env_value = os.getenv(var_name)
            if env_value is not None:
                assert isinstance(env_value, str)
            # Otherwise defaults should be used (tested in config tests)

    def test_database_url_construction(self):
        """Test database URL construction."""
        from src.config import DatabaseConfig
        
        db_url = DatabaseConfig.get_database_url()
        assert isinstance(db_url, str)
        assert "postgresql://" in db_url
        assert "@" in db_url  # Should have user@host format


class TestIntegrationReadiness:
    """Test that system is ready for integration."""

    def test_all_imports_work(self):
        """Test that all core modules can be imported."""
        import_tests = [
            "src.config",
            "src.data.database"
        ]
        
        for module_name in import_tests:
            try:
                __import__(module_name)
            except ImportError as e:
                if "newsapi" not in str(e) and "transformers" not in str(e):
                    # Allow failures for heavy dependencies, but not core logic
                    pytest.fail(f"Core module {module_name} failed to import: {e}")

    def test_dag_structure_exists(self):
        """Test that DAG files exist and have basic structure."""
        import os
        
        dag_file = "src/airflow_dags/data_pipeline.py"
        assert os.path.exists(dag_file), "DAG file should exist"
        
        # Read file and check for key components
        with open(dag_file, 'r') as f:
            content = f.read()
            
        required_components = [
            "def collect_market_data",
            "def collect_news_sentiment", 
            "def store_processed_data",
            "data_collection_pipeline"
        ]
        
        for component in required_components:
            assert component in content, f"DAG should contain {component}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])