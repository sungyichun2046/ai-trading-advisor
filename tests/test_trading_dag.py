"""
Tests for Consolidated Trading DAG with Task Groups.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# Import the DAG module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.trading_dag import (
    # Data collection functions
    simple_collect_market_data,
    simple_collect_fundamental_data,
    simple_collect_sentiment,
    monitor_data_systems,
    
    # Analysis functions
    simple_technical_analysis,
    simple_fundamental_analysis,
    simple_sentiment_analysis,
    calculate_consensus_signals,
    monitor_analysis_systems,
    
    # Trading functions
    simple_generate_signals,
    simple_assess_risk,
    simple_execute_trades,
    monitor_trading_systems,
    
    # DAG
    dag
)


class TestConsolidatedTradingDAG:
    """Test consolidated trading DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'trading_workflow'
        assert dag.schedule_interval is None  # Manual trigger only
        assert dag.max_active_runs == 1
        assert 'trading' in dag.tags
        assert 'consolidated' in dag.tags
        assert 'workflow' in dag.tags

    def test_dag_tasks_exist(self):
        """Test that all expected tasks exist in the DAG."""
        task_ids = [t.task_id for t in dag.tasks]
        
        # Data collection tasks
        assert 'collect_data_tasks.collect_market_data' in task_ids
        assert 'collect_data_tasks.collect_fundamental_data' in task_ids
        assert 'collect_data_tasks.collect_sentiment_data' in task_ids
        assert 'collect_data_tasks.monitor_data_systems' in task_ids
        
        # Analysis tasks
        assert 'analyze_data_tasks.analyze_technical_indicators' in task_ids
        assert 'analyze_data_tasks.analyze_fundamentals' in task_ids
        assert 'analyze_data_tasks.analyze_sentiment' in task_ids
        assert 'analyze_data_tasks.calculate_consensus_signals' in task_ids
        assert 'analyze_data_tasks.monitor_analysis_systems' in task_ids
        
        # Trading tasks
        assert 'execute_trades_tasks.generate_trading_signals' in task_ids
        assert 'execute_trades_tasks.assess_portfolio_risk' in task_ids
        assert 'execute_trades_tasks.execute_paper_trades' in task_ids
        assert 'execute_trades_tasks.monitor_trading_systems' in task_ids

    def test_task_groups_exist(self):
        """Test that expected task groups are present."""
        task_groups = []
        for task in dag.tasks:
            if hasattr(task, 'task_group') and task.task_group:
                group_id = task.task_group.group_id
                if group_id not in task_groups:
                    task_groups.append(group_id)
        
        expected_groups = ['collect_data_tasks', 'analyze_data_tasks', 'execute_trades_tasks']
        for group in expected_groups:
            assert group in task_groups

    def test_task_count(self):
        """Test that we have the expected number of tasks."""
        assert len(dag.tasks) == 13  # 4 data + 5 analysis + 4 trading


class TestDataCollectionFunctions:
    """Test data collection functions."""
    
    def test_collect_market_data_success(self):
        """Test successful market data collection."""
        mock_context = {'task_instance': Mock()}
        result = simple_collect_market_data(**mock_context)
        
        assert result['status'] == 'success'
        assert 'symbols' in result
        assert 'data_points' in result
        assert result['symbols'] == ['AAPL', 'SPY', 'QQQ']
        mock_context['task_instance'].xcom_push.assert_called_once()

    def test_collect_fundamental_data_success(self):
        """Test successful fundamental data collection."""
        mock_context = {'task_instance': Mock()}
        result = simple_collect_fundamental_data(**mock_context)
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert 'pe_ratio' in result['metrics']
        mock_context['task_instance'].xcom_push.assert_called_once()

    def test_collect_sentiment_success(self):
        """Test successful sentiment data collection."""
        mock_context = {'task_instance': Mock()}
        result = simple_collect_sentiment(**mock_context)
        
        assert result['status'] == 'success'
        assert 'sentiment' in result
        assert 'score' in result
        mock_context['task_instance'].xcom_push.assert_called_once()


class TestAnalysisFunctions:
    """Test analysis functions."""
    
    def test_technical_analysis_success(self):
        """Test successful technical analysis."""
        mock_ti = Mock()
        # Configure mock to return realistic market data structure
        mock_ti.xcom_pull.return_value = {
            'status': 'success',
            'data_source': 'yahoo_finance',
            'symbols': ['AAPL', 'SPY', 'QQQ'],
            'data_points': 100,
            'data': {
                'AAPL': {'price': 150.0, 'volume': 1000000},
                'SPY': {'price': 400.0, 'volume': 2000000},
                'QQQ': {'price': 350.0, 'volume': 1500000}
            }
        }
        mock_context = {'task_instance': mock_ti}
        result = simple_technical_analysis(**mock_context)
        
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        mock_ti.xcom_push.assert_called_once()

    def test_fundamental_analysis_success(self):
        """Test successful fundamental analysis."""
        mock_ti = Mock()
        # Configure mock to return realistic fundamental data structure
        mock_ti.xcom_pull.return_value = {
            'status': 'success',
            'data_source': 'yahoo_finance',
            'symbols_collected': 1,
            'data': [{
                'symbol': 'AAPL',
                'pe_ratio': 25.5,
                'market_cap': 2500000000000,
                'revenue': 400000000000,
                'profit_margin': 0.25,
                'debt_to_equity': 1.5
            }],
            'metrics': {'pe_ratio': 25.5}
        }
        mock_context = {'task_instance': mock_ti}
        result = simple_fundamental_analysis(**mock_context)
        
        assert result['status'] == 'success'
        assert 'valuation' in result
        assert 'recommendation' in result
        mock_ti.xcom_push.assert_called_once()

    def test_sentiment_analysis_success(self):
        """Test successful sentiment analysis."""
        mock_ti = Mock()
        # Configure mock to return realistic sentiment data structure
        mock_ti.xcom_pull.return_value = {
            'status': 'success',
            'data_source': 'newsapi',
            'article_count': 15,
            'sentiment_method': 'textblob',
            'articles': [
                {'title': 'Market Up', 'sentiment_score': 0.2, 'sentiment_label': 'positive'},
                {'title': 'Tech Strong', 'sentiment_score': 0.15, 'sentiment_label': 'positive'},
            ],
            'sentiment': 0.175,
            'score': 0.175
        }
        mock_context = {'task_instance': mock_ti}
        result = simple_sentiment_analysis(**mock_context)
        
        assert result['status'] == 'success'
        assert 'overall_sentiment' in result
        assert 'confidence' in result
        mock_ti.xcom_push.assert_called_once()

    def test_consensus_signals_success(self):
        """Test successful consensus signals calculation."""
        mock_ti = Mock()
        mock_ti.xcom_pull.return_value = {'status': 'success'}
        mock_context = {'task_instance': mock_ti}
        
        result = calculate_consensus_signals(**mock_context)
        
        assert result['status'] == 'success'
        assert 'consensus_score' in result
        assert 'confidence_level' in result
        mock_ti.xcom_push.assert_called_once()


class TestTradingFunctions:
    """Test trading functions."""
    
    def test_generate_signals_success(self):
        """Test successful signal generation."""
        mock_ti = Mock()
        # Configure mock to return realistic analysis data structures
        def mock_xcom_pull(key=None, task_ids=None):
            if 'technical' in str(task_ids):
                return {
                    'status': 'success',
                    'signal': 'buy',
                    'confidence': 0.8,
                    'indicators': {'rsi': 65, 'macd': 0.5}
                }
            elif 'fundamental' in str(task_ids):
                return {
                    'status': 'success',
                    'recommendation': 'buy',
                    'valuation': 'undervalued',
                    'confidence': 0.75
                }
            elif 'sentiment' in str(task_ids):
                return {
                    'status': 'success',
                    'overall_sentiment': 0.2,
                    'confidence': 0.7
                }
            elif 'consensus' in str(task_ids):
                return {
                    'status': 'success',
                    'consensus_score': 0.75,
                    'confidence_level': 'high',
                    'alignment_status': 'aligned'
                }
            return {'status': 'success'}
        
        mock_ti.xcom_pull.side_effect = mock_xcom_pull
        mock_context = {'task_instance': mock_ti}
        result = simple_generate_signals(**mock_context)
        
        assert result['status'] == 'success'
        assert 'signals' in result
        assert 'overall_confidence' in result
        mock_ti.xcom_push.assert_called_once()

    def test_assess_risk_success(self):
        """Test successful risk assessment."""
        mock_context = {'task_instance': Mock()}
        result = simple_assess_risk(**mock_context)
        
        assert result['status'] == 'success'
        assert 'portfolio_risk' in result
        assert 'recommendation' in result
        mock_context['task_instance'].xcom_push.assert_called_once()

    def test_execute_trades_success(self):
        """Test successful trade execution."""
        mock_context = {'task_instance': Mock()}
        result = simple_execute_trades(**mock_context)
        
        assert result['status'] == 'success'
        assert 'trades_executed' in result
        assert 'total_value' in result
        mock_context['task_instance'].xcom_push.assert_called_once()


class TestMonitoringFunctions:
    """Test monitoring functions."""
    
    @patch('src.core.data_manager.DataManager')
    def test_monitor_data_systems_success(self, mock_data_manager):
        """Test successful data systems monitoring."""
        mock_dm = Mock()
        mock_dm.monitor_data_quality.return_value = {'score': 0.9}
        mock_dm.monitor_data_freshness.return_value = {'fresh': True}
        mock_dm.monitor_system_health.return_value = {'healthy': True}
        mock_dm.monitor_data_collection_performance.return_value = {'fast': True}
        mock_data_manager.return_value = mock_dm
        
        mock_context = {'task_instance': Mock()}
        result = monitor_data_systems(**mock_context)
        
        assert result['status'] == 'success'
        mock_context['task_instance'].xcom_push.assert_called_once()

    @patch('src.core.data_manager.DataManager')
    def test_monitor_analysis_systems_success(self, mock_data_manager):
        """Test successful analysis systems monitoring."""
        mock_dm = Mock()
        mock_dm.monitor_system_health.return_value = {'healthy': True}
        mock_dm.monitor_data_quality.return_value = {'score': 0.9}
        mock_data_manager.return_value = mock_dm
        
        mock_context = {'task_instance': Mock()}
        result = monitor_analysis_systems(**mock_context)
        
        assert result['status'] == 'success'
        mock_context['task_instance'].xcom_push.assert_called_once()

    @patch('src.core.data_manager.DataManager')
    def test_monitor_trading_systems_success(self, mock_data_manager):
        """Test successful trading systems monitoring."""
        mock_dm = Mock()
        mock_dm.monitor_system_health.return_value = {'healthy': True}
        mock_dm.monitor_data_freshness.return_value = {'fresh': True}
        mock_data_manager.return_value = mock_dm
        
        mock_context = {'task_instance': Mock()}
        result = monitor_trading_systems(**mock_context)
        
        assert result['status'] == 'success'
        mock_context['task_instance'].xcom_push.assert_called_once()


class TestTaskGroupIntegration:
    """Test task group integration and workflow."""
    
    def test_no_external_task_sensor_dependencies(self):
        """Test that no ExternalTaskSensor dependencies exist."""
        for task in dag.tasks:
            assert 'ExternalTaskSensor' not in str(type(task))
            assert 'wait_for_' not in task.task_id

    def test_task_group_workflow_structure(self):
        """Test that task groups have proper internal structure."""
        # Get tasks by group
        collect_tasks = [t for t in dag.tasks if t.task_id.startswith('collect_data_tasks.')]
        analyze_tasks = [t for t in dag.tasks if t.task_id.startswith('analyze_data_tasks.')]
        trading_tasks = [t for t in dag.tasks if t.task_id.startswith('execute_trades_tasks.')]
        
        # Verify each group has expected number of tasks
        assert len(collect_tasks) == 4  # 3 collection + 1 monitoring
        assert len(analyze_tasks) == 5  # 3 analysis + 1 consensus + 1 monitoring  
        assert len(trading_tasks) == 4  # 3 trading + 1 monitoring

    def test_consolidated_dag_eliminates_sensors(self):
        """Test that consolidation eliminated all ExternalTaskSensor complexity."""
        # No external task sensors should exist
        sensor_tasks = [t for t in dag.tasks if 'sensor' in t.task_id.lower() or 'wait' in t.task_id.lower()]
        assert len(sensor_tasks) == 0
        
        # All tasks should belong to one of the three main groups
        for task in dag.tasks:
            assert (task.task_id.startswith('collect_data_tasks.') or
                   task.task_id.startswith('analyze_data_tasks.') or
                   task.task_id.startswith('execute_trades_tasks.'))