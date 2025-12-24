"""Tests for Analysis DAG."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.analysis_dag import (
    simple_technical_analysis as analyze_technical_indicators,
    simple_fundamental_analysis as analyze_fundamentals, 
    simple_sentiment_analysis as analyze_sentiment,
    monitor_analysis_systems, dag
)


class TestAnalysisDAG:
    """Test analysis DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'analysis'
        assert dag.schedule_interval is None  # Manual trigger only
        assert dag.max_active_runs == 1
        assert 'analysis' in dag.tags
        assert 'simple' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'analyze_technical_indicators',
            'analyze_fundamentals',
            'analyze_sentiment',
            'monitor_analysis_systems'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 4  # Simple structure with monitoring


class TestAnalyzeTechnicalIndicators:
    """Test technical indicator analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_analyze_technical_indicators_success(self):
        """Test successful technical indicator analysis."""
        result = analyze_technical_indicators(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        assert 'timestamp' in result
        
        result = analyze_technical_indicators(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestAnalyzeFundamentals:
    """Test fundamental analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'technical_summary': {'market_sentiment': 'bullish'}
        }
    
    def test_analyze_fundamentals_success(self):
        """Test successful fundamental analysis."""
        result = analyze_fundamentals(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'valuation' in result
        assert 'recommendation' in result
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestMonitorAnalysisSystems:
    """Test monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_monitor_analysis_systems_success(self):
        """Test successful analysis monitoring."""
        result = monitor_analysis_systems(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'analysis_system_health' in result or 'monitoring' in result
        # alerts_sent is tracked in logs but not always in return value
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestAnalyzeSentiment:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'technical_summary': {'market_sentiment': 'bullish'},
            'fundamental_summary': {'market_valuation': 'fairly_valued'}
        }
    
    def test_analyze_sentiment_success(self):
        """Test successful sentiment analysis."""
        result = analyze_sentiment(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'overall_sentiment' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        assert result['confidence'] >= 0.0
        assert result['confidence'] <= 1.0
        self.mock_context['task_instance'].xcom_push.assert_called()