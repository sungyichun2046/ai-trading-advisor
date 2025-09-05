"""Tests for Airflow data pipeline DAG."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from airflow.models import DagBag

# Test imports
from src.airflow_dags.data_pipeline import (
    collect_market_data,
    collect_news_sentiment,
    initialize_database,
    store_processed_data,
    validate_data_quality
)


class TestDAGStructure:
    """Test DAG structure and configuration."""

    def test_dag_loaded_successfully(self):
        """Test that the DAG loads without errors."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        
        # Check for import errors
        assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
        
        # Check that our DAG is loaded
        assert "data_collection_pipeline" in dag_bag.dags
        
        dag = dag_bag.get_dag("data_collection_pipeline")
        assert dag is not None

    def test_dag_configuration(self):
        """Test DAG configuration and properties."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Test DAG properties
        assert dag.dag_id == "data_collection_pipeline"
        assert dag.schedule_interval == "*/15 * * * *"  # Every 15 minutes
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        
        # Test tags
        expected_tags = ["data", "collection", "market"]
        assert all(tag in dag.tags for tag in expected_tags)

    def test_dag_tasks_structure(self):
        """Test DAG task structure and dependencies."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Expected tasks
        expected_tasks = [
            "initialize_database",
            "collect_market_data",
            "collect_news_sentiment", 
            "validate_data_quality",
            "store_processed_data",
            "pipeline_health_check"
        ]
        
        actual_tasks = list(dag.task_ids)
        
        for task_id in expected_tasks:
            assert task_id in actual_tasks, f"Missing task: {task_id}"

    def test_task_dependencies(self):
        """Test task dependencies are correct."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Get tasks
        init_db = dag.get_task("initialize_database")
        collect_market = dag.get_task("collect_market_data")
        collect_news = dag.get_task("collect_news_sentiment")
        validate = dag.get_task("validate_data_quality")
        store = dag.get_task("store_processed_data")
        health_check = dag.get_task("pipeline_health_check")
        
        # Check dependencies
        assert len(init_db.upstream_task_ids) == 0  # First task
        assert collect_market.upstream_task_ids == {"initialize_database"}
        assert collect_news.upstream_task_ids == {"initialize_database"}
        assert validate.upstream_task_ids == {"collect_market_data", "collect_news_sentiment"}
        assert store.upstream_task_ids == {"validate_data_quality"}
        assert health_check.upstream_task_ids == {"store_processed_data"}


class TestPipelineFunctions:
    """Test individual pipeline functions."""

    @patch('src.airflow_dags.data_pipeline.MarketDataCollector')
    def test_collect_market_data_success(self, mock_collector_class):
        """Test successful market data collection."""
        # Mock collector
        mock_collector = Mock()
        mock_collector.collect_real_time_data.return_value = {
            "symbol": "AAPL",
            "status": "success", 
            "price": 150.0,
            "volume": 1000000,
            "data_source": "yfinance"
        }
        mock_collector_class.return_value = mock_collector
        
        # Mock context
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = collect_market_data(**context)
        
        # Verify results
        assert isinstance(result, dict)
        assert len(result) == 8  # Number of symbols configured
        assert "SPY" in result
        assert "AAPL" in result
        assert result["AAPL"]["status"] == "success"

    @patch('src.airflow_dags.data_pipeline.MarketDataCollector')
    def test_collect_market_data_failure(self, mock_collector_class):
        """Test market data collection with API failure."""
        # Mock collector that raises exception
        mock_collector = Mock()
        mock_collector.collect_real_time_data.side_effect = Exception("API Error")
        mock_collector_class.return_value = mock_collector
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = collect_market_data(**context)
        
        # Verify error handling
        assert isinstance(result, dict)
        for symbol_result in result.values():
            assert symbol_result["status"] == "failed"
            assert "API Error" in symbol_result["error"]

    @patch('src.airflow_dags.data_pipeline.NewsCollector')
    def test_collect_news_sentiment_success(self, mock_collector_class):
        """Test successful news sentiment collection."""
        # Mock collector
        mock_collector = Mock()
        mock_collector.collect_financial_news.return_value = [
            {"title": "Test Article", "content": "Test content"}
        ]
        mock_collector.analyze_sentiment.return_value = {
            "status": "success",
            "average": 0.5,
            "articles_analyzed": 1,
            "articles": [
                {"title": "Test Article", "sentiment_score": 0.5}
            ]
        }
        mock_collector_class.return_value = mock_collector
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = collect_news_sentiment(**context)
        
        # Verify results
        assert result["status"] == "success"
        assert result["articles_count"] == 1
        assert "average" in result

    @patch('src.airflow_dags.data_pipeline.NewsCollector')
    def test_collect_news_sentiment_failure(self, mock_collector_class):
        """Test news sentiment collection failure."""
        # Mock collector that raises exception
        mock_collector = Mock()
        mock_collector.collect_financial_news.side_effect = Exception("News API Error")
        mock_collector_class.return_value = mock_collector
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = collect_news_sentiment(**context)
        
        # Verify error handling
        assert result["status"] == "failed"
        assert "News API Error" in result["error"]

    @patch('src.airflow_dags.data_pipeline.DatabaseManager')
    def test_initialize_database_success(self, mock_db_class):
        """Test successful database initialization."""
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = initialize_database(**context)
        
        # Verify results
        assert result["status"] == "success"
        mock_db.create_tables.assert_called_once()

    @patch('src.airflow_dags.data_pipeline.DatabaseManager')
    def test_initialize_database_failure(self, mock_db_class):
        """Test database initialization failure."""
        mock_db = Mock()
        mock_db.create_tables.side_effect = Exception("Database Error")
        mock_db_class.return_value = mock_db
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0)
        }
        
        result = initialize_database(**context)
        
        # Verify error handling
        assert result["status"] == "failed"
        assert "Database Error" in result["error"]

    @patch('src.airflow_dags.data_pipeline.NewsStorage')
    @patch('src.airflow_dags.data_pipeline.MarketDataStorage')
    def test_store_processed_data_success(self, mock_market_storage_class, mock_news_storage_class):
        """Test successful data storage."""
        # Mock storage classes
        mock_market_storage = Mock()
        mock_market_storage.store_market_data.return_value = {"count": 2}
        mock_market_storage_class.return_value = mock_market_storage
        
        mock_news_storage = Mock()
        mock_news_storage.store_news_data.return_value = {"count": 5}
        mock_news_storage_class.return_value = mock_news_storage
        
        # Mock context with task instance
        mock_task_instance = Mock()
        mock_task_instance.xcom_pull.side_effect = [
            {"AAPL": {"status": "success", "price": 150.0}},  # market data
            {"status": "success", "articles": []}  # news data
        ]
        
        context = {
            "execution_date": datetime(2024, 1, 1, 10, 0, 0),
            "task_instance": mock_task_instance
        }
        
        result = store_processed_data(**context)
        
        # Verify results
        assert result["status"] == "success"
        assert result["market_records_stored"] == 2
        assert result["news_records_stored"] == 5

    @patch('src.data.processors.DataValidator')
    def test_validate_data_quality_success(self, mock_validator_class):
        """Test successful data quality validation."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_market_data.return_value = {
            "completeness": 0.9,
            "freshness": 0.95
        }
        mock_validator.validate_news_data.return_value = {
            "coverage": 0.8
        }
        mock_validator_class.return_value = mock_validator
        
        # Mock context
        mock_task_instance = Mock()
        mock_task_instance.xcom_pull.side_effect = [
            {"AAPL": {"status": "success"}},  # market data
            {"status": "success"}  # news data
        ]
        
        context = {
            "task_instance": mock_task_instance
        }
        
        result = validate_data_quality(**context)
        
        # Verify validation passes
        assert result is True

    @patch('src.data.processors.DataValidator')
    def test_validate_data_quality_failure(self, mock_validator_class):
        """Test data quality validation failure."""
        # Mock validator with poor quality data
        mock_validator = Mock()
        mock_validator.validate_market_data.return_value = {
            "completeness": 0.5,  # Below 80% threshold
            "freshness": 0.7      # Below 90% threshold
        }
        mock_validator.validate_news_data.return_value = {
            "coverage": 0.6       # Below 70% threshold
        }
        mock_validator_class.return_value = mock_validator
        
        # Mock context
        mock_task_instance = Mock()
        mock_task_instance.xcom_pull.side_effect = [
            {"AAPL": {"status": "failed"}},
            {"status": "failed"}
        ]
        
        context = {
            "task_instance": mock_task_instance
        }
        
        # Should raise ValueError for failed validation
        with pytest.raises(ValueError, match="Data quality validation failed"):
            validate_data_quality(**context)


class TestDAGExecution:
    """Test DAG execution scenarios."""

    def test_dag_execution_order(self):
        """Test that tasks execute in correct order."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Simulate DAG run
        from airflow.utils.state import DagRunState
        from airflow.utils.types import DagRunType
        
        # This would require a test Airflow environment
        # For now, just verify the structure is correct
        assert dag is not None
        
        # Check that data collection tasks run in parallel
        collect_market = dag.get_task("collect_market_data")
        collect_news = dag.get_task("collect_news_sentiment")
        
        # Both should depend only on init_db
        assert collect_market.upstream_task_ids == {"initialize_database"}
        assert collect_news.upstream_task_ids == {"initialize_database"}
        
        # Validate should wait for both collection tasks
        validate = dag.get_task("validate_data_quality")
        assert "collect_market_data" in validate.upstream_task_ids
        assert "collect_news_sentiment" in validate.upstream_task_ids

    def test_dag_timeout_configuration(self):
        """Test DAG timeout and retry configuration."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Check default args
        default_args = dag.default_args
        assert default_args["retries"] == 2
        assert default_args["retry_delay"] == timedelta(minutes=5)
        assert default_args["execution_timeout"] == timedelta(hours=1)

    def test_dag_failure_handling(self):
        """Test DAG failure handling configuration."""
        dag_bag = DagBag(dag_folder="src/airflow_dags", include_examples=False)
        dag = dag_bag.get_dag("data_collection_pipeline")
        
        # Check email notifications
        default_args = dag.default_args
        assert default_args["email_on_failure"] is True
        assert default_args["email_on_retry"] is False


class TestDataValidation:
    """Test data validation logic."""

    def test_market_data_validation_logic(self):
        """Test market data validation thresholds."""
        # This tests the validation logic embedded in validate_data_quality
        
        # Test case: Good quality data
        good_validation = {
            "completeness": 0.9,  # > 0.8
            "freshness": 0.95     # > 0.9
        }
        
        # Test case: Poor quality data  
        poor_validation = {
            "completeness": 0.7,  # < 0.8
            "freshness": 0.85     # < 0.9
        }
        
        # The actual validation happens in the function
        # Here we verify the threshold logic
        def check_quality(market_val, news_val):
            return (
                market_val.get("completeness", 0) >= 0.8
                and market_val.get("freshness", 0) >= 0.9  
                and news_val.get("coverage", 0) >= 0.7
            )
        
        assert check_quality(good_validation, {"coverage": 0.8}) is True
        assert check_quality(poor_validation, {"coverage": 0.8}) is False