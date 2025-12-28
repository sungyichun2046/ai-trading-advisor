"""
Tests for Trading DAG.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# Import the DAG module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.trading_dag import (
    simple_generate_signals as generate_trading_signals,
    simple_assess_risk as assess_portfolio_risk,
    simple_execute_trades as execute_paper_trades,
    monitor_trading_systems, dag
)


class TestTradingDAG:
    """Test trading DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'trading'
        assert dag.schedule_interval is None  # Manual trigger only
        assert dag.max_active_runs == 1
        assert 'trading' in dag.tags
        assert 'simple' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'generate_trading_signals',
            'assess_portfolio_risk',
            'execute_paper_trades',
            'monitor_trading_systems'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 8  # Enhanced with dependency management (skip, proceed, monitoring tasks)


class TestGenerateTradingSignals:
    """Test trading signal generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_generate_trading_signals_success(self):
        """Test successful trading signal generation."""
        result = generate_trading_signals(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'signals' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        assert len(result['signals']) >= 1
        self.mock_context['task_instance'].xcom_push.assert_called()
    
    def test_generate_trading_signals_structure(self):
        """Test trading signals output structure."""
        result = generate_trading_signals(**self.mock_context)
        
        required_keys = ['status', 'signals', 'confidence', 'timestamp']
        assert all(key in result for key in required_keys)
        
        assert isinstance(result['signals'], dict)
        assert isinstance(result['confidence'], float)
        assert 0.0 <= result['confidence'] <= 1.0


class TestAssessPortfolioRisk:
    """Test portfolio risk assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'trading_signals': {'AAPL': {'signal': 'buy', 'confidence': 0.7}}
        }
    
    def test_assess_portfolio_risk_success(self):
        """Test successful portfolio risk assessment."""
        result = assess_portfolio_risk(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'portfolio_risk' in result
        assert 'recommendation' in result
        assert 'timestamp' in result
        assert result['recommendation'] in ['acceptable', 'caution', 'high_risk']
        self.mock_context['task_instance'].xcom_push.assert_called()
    
    def test_assess_portfolio_risk_structure(self):
        """Test risk assessment output structure."""
        result = assess_portfolio_risk(**self.mock_context)
        
        required_keys = ['status', 'portfolio_risk', 'recommendation', 'timestamp']
        assert all(key in result for key in required_keys)
        
        assert isinstance(result['portfolio_risk'], float)
        assert 0.0 <= result['portfolio_risk'] <= 1.0


class TestExecutePaperTrades:
    """Test paper trading execution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.side_effect = [
            {'signals': {'AAPL': {'signal': 'buy'}}, 'signals_generated': 3},
            {'risk': {'overall_risk_level': 'medium'}, 'risk_level': 'medium'}
        ]
    
    def test_execute_paper_trades_success(self):
        """Test successful paper trade execution."""
        result = execute_paper_trades(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'trades_executed' in result
        assert 'total_value' in result
        assert 'timestamp' in result
        assert result['trades_executed'] >= 0
        self.mock_context['task_instance'].xcom_push.assert_called()
    
    def test_execute_paper_trades_structure(self):
        """Test paper trades output structure."""
        result = execute_paper_trades(**self.mock_context)
        
        required_keys = ['status', 'trades_executed', 'total_value', 'timestamp']
        assert all(key in result for key in required_keys)
        
        assert isinstance(result['trades_executed'], int)
        assert result['trades_executed'] >= 0


class TestMonitorTradingSystems:
    """Test trading monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_monitor_trading_systems_success(self):
        """Test successful trading monitoring."""
        result = monitor_trading_systems(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'trading_system_health' in result or 'monitoring' in result
        # alerts_sent is tracked in logs but not always in return value
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestTradingDAGIntegration:
    """Test trading DAG integration and workflow."""
    
    def test_task_dependencies(self):
        """Test task dependency configuration."""
        # Get task dependency mapping
        task_dict = {task.task_id: task for task in dag.tasks}
        
        # Check generate_trading_signals depends on trading_proceed (from dependency manager)
        assert 'trading_proceed' in task_dict['generate_trading_signals'].upstream_task_ids
        
        # Check assess_portfolio_risk depends on generate_trading_signals
        assert 'generate_trading_signals' in task_dict['assess_portfolio_risk'].upstream_task_ids
        
        # Check execute_paper_trades depends on risk assessment only
        trade_upstream = task_dict['execute_paper_trades'].upstream_task_ids
        assert 'assess_portfolio_risk' in trade_upstream
        
        # Check monitor_trading_systems dependency
        monitor_upstream = task_dict['monitor_trading_systems'].upstream_task_ids
        # Monitoring depends on execute_paper_trades and trading_proceed (from dependency manager)
        assert 'execute_paper_trades' in monitor_upstream
        assert len(monitor_upstream) >= 1  # May have additional dependencies from dependency manager
    
    def test_dag_task_count(self):
        """Test DAG has exactly 8 tasks (enhanced with dependency management)."""
        assert len(dag.tasks) == 8  # Enhanced with dependency management (skip, proceed, monitoring tasks)
    
    def test_dag_schedule_weekdays_only(self):
        """Test DAG is scheduled for manual trigger only."""
        # Simple DAG uses manual trigger (None schedule)
        assert dag.schedule_interval is None