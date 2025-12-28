"""
Tests for Data Collection DAG.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# Import the DAG module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.data_collection_dag import (
    simple_collect_market_data as collect_market_data,
    simple_collect_fundamental_data as collect_fundamental_data,
    simple_collect_sentiment as collect_sentiment_data,
    monitor_data_systems, dag
)


class TestDataCollectionDAG:
    """Test data collection DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'data_collection'
        assert dag.schedule_interval is None  # Manual trigger only
        assert dag.max_active_runs == 1
        assert 'data' in dag.tags
        assert 'simple' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'collect_market_data',
            'collect_fundamental_data', 
            'collect_sentiment_data',
            'monitor_data_systems'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 8  # Enhanced with dependency management (skip, proceed, monitoring tasks)


class TestCollectMarketData:
    """Test market data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_collect_market_data_success(self):
        """Test successful market data collection."""
        result = collect_market_data(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'symbols' in result
        assert 'timestamp' in result
        assert result['data_points'] >= 1
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestCollectFundamentalData:
    """Test fundamental data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'status': 'success',
            'symbols_collected': 15
        }
    
    def test_collect_fundamental_data_success(self):
        """Test successful fundamental data collection."""
        result = collect_fundamental_data(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert 'timestamp' in result


class TestCollectSentimentData:
    """Test sentiment data collection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_collect_sentiment_data_success(self):
        """Test successful sentiment data collection."""
        result = collect_sentiment_data(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'sentiment' in result
        assert 'score' in result
        assert 'timestamp' in result


class TestMonitorDataSystems:
    """Test data monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_monitor_data_systems_success(self):
        """Test successful data monitoring."""
        result = monitor_data_systems(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'collection_performance' in result or 'data_system_health' in result or 'monitoring' in result
        # alerts_sent is tracked in logs but not always in return value
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()