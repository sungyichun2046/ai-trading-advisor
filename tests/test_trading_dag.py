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
    generate_trading_signals, assess_portfolio_risk, calculate_position_sizes,
    manage_portfolio, send_alerts, dag
)


class TestTradingDAG:
    """Test trading DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'trading'
        assert dag.schedule_interval == '0 9,15 * * 1-5'  # 9 AM and 3 PM weekdays
        assert dag.max_active_runs == 1
        assert 'trading' in dag.tags
        assert 'portfolio' in dag.tags
        assert 'risk' in dag.tags
        assert 'signals' in dag.tags
        assert 'alerts' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'generate_trading_signals',
            'assess_portfolio_risk',
            'calculate_position_sizes',
            'manage_portfolio',
            'send_alerts'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 5


class TestGenerateTradingSignals:
    """Test trading signal generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    @patch('src.dags.trading_dag.get_trading_engine')
    def test_generate_trading_signals_success(self, mock_get_trading_engine):
        """Test successful trading signal generation."""
        mock_trading_engine = Mock()
        # Mock the strategy methods
        mock_trading_engine.momentum_strategy.return_value = {'signal': 'buy', 'confidence': 0.8, 'reasoning': 'Mock momentum'}
        mock_trading_engine.mean_reversion_strategy.return_value = {'signal': 'hold', 'confidence': 0.5, 'reasoning': 'Mock mean reversion'}
        mock_trading_engine.breakout_strategy.return_value = {'signal': 'sell', 'confidence': 0.7, 'reasoning': 'Mock breakout'}
        mock_trading_engine.value_strategy.return_value = {'signal': 'buy', 'confidence': 0.9, 'reasoning': 'Mock value'}
        mock_get_trading_engine.return_value = mock_trading_engine
        
        result = generate_trading_signals(**self.mock_context)
        
        assert 'strategy_results' in result
        assert 'overall_signal' in result
        assert 'confidence' in result
        assert result['overall_signal'] in ['buy', 'sell', 'hold']
        self.mock_context['task_instance'].xcom_push.assert_called_once()
    
    def test_generate_trading_signals_structure(self):
        """Test trading signals output structure."""
        result = generate_trading_signals(**self.mock_context)
        
        required_keys = ['timestamp', 'strategy_results', 'overall_signal', 'confidence', 'signal_distribution', 'performance_metrics']
        assert all(key in result for key in required_keys)
        
        assert isinstance(result['strategy_results'], dict)
        assert len(result['strategy_results']) == 4  # 4 strategies
        assert isinstance(result['signal_distribution'], dict)


class TestAssessPortfolioRisk:
    """Test portfolio risk assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'trading_signals': {'AAPL': {'signal': 'buy', 'confidence': 0.7}}
        }
    
    @patch('src.dags.trading_dag.RiskManager')
    def test_assess_portfolio_risk_success(self, mock_risk_manager):
        """Test successful portfolio risk assessment."""
        mock_manager = Mock()
        mock_manager.assess_portfolio_risk.return_value = {
            'status': 'success',
            'risk_level': 'moderate'
        }
        mock_risk_manager.return_value = mock_manager
        
        result = assess_portfolio_risk(**self.mock_context)
        
        assert 'risk_metrics' in result
        assert 'daily_loss_status' in result
        assert 'portfolio_analysis' in result
        self.mock_context['task_instance'].xcom_push.assert_called_once()
    
    def test_assess_portfolio_risk_structure(self):
        """Test risk assessment output structure."""
        result = assess_portfolio_risk(**self.mock_context)
        
        required_keys = ['timestamp', 'risk_metrics', 'daily_loss_status', 'portfolio_analysis', 'performance_metrics', 'risk_recommendations']
        assert all(key in result for key in required_keys)
        
        assert isinstance(result['risk_metrics'], dict)
        assert isinstance(result['portfolio_analysis'], dict)


class TestCalculatePositionSizes:
    """Test position sizing calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.side_effect = [
            {'trading_signals': {'AAPL': {'signal': 'buy', 'confidence': 0.7}}},
            {'portfolio_risk_assessment': {'overall_risk_level': 'moderate'}}
        ]
    
    @patch('src.dags.trading_dag.PositionSizingCalculator')
    def test_calculate_position_sizes_success(self, mock_calculator):
        """Test successful position size calculation."""
        mock_calc = Mock()
        mock_calc.calculate_position_sizes.return_value = {
            'status': 'success',
            'calculated_positions': 3
        }
        mock_calculator.return_value = mock_calc
        
        result = calculate_position_sizes(**self.mock_context)
        
        assert 'position_sizing' in result
        assert 'calculated_positions' in result
        assert 'sizing_constraints' in result
        assert result['position_sizing']['number_of_positions'] == 3
        self.mock_context['task_instance'].xcom_push.assert_called_once()
    
    def test_calculate_position_sizes_allocation(self):
        """Test position sizing allocation logic."""
        result = calculate_position_sizes(**self.mock_context)
        
        assert result['position_sizing']['total_portfolio_allocation'] <= 1.0
        assert result['position_sizing']['cash_remaining_pct'] >= 0.0
        
        for symbol, position in result['calculated_positions'].items():
            assert 'dollar_amount' in position
            assert 'portfolio_percentage' in position
            assert 'shares' in position
            assert position['portfolio_percentage'] <= 0.05  # Max 5% per position


class TestManagePortfolio:
    """Test portfolio management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.side_effect = [
            {'calculated_positions': {'AAPL': {'dollar_amount': 5000}}},
            {'portfolio_risk_assessment': {'overall_risk_level': 'moderate'}}
        ]
    
    @patch('src.dags.trading_dag.PortfolioManager')
    def test_manage_portfolio_success(self, mock_manager):
        """Test successful portfolio management."""
        mock_pm = Mock()
        mock_pm.manage_portfolio.return_value = {
            'status': 'success',
            'rebalanced': True
        }
        mock_manager.return_value = mock_pm
        
        result = manage_portfolio(**self.mock_context)
        
        assert 'portfolio_management' in result
        assert 'executed_trades' in result
        assert 'portfolio_status' in result
        assert isinstance(result['executed_trades'], list)
        self.mock_context['task_instance'].xcom_push.assert_called_once()
    
    def test_manage_portfolio_trade_execution(self):
        """Test trade execution logic."""
        result = manage_portfolio(**self.mock_context)
        
        assert 'portfolio_turnover_pct' in result['portfolio_management']
        assert 'total_trade_value' in result['portfolio_management']
        
        for trade in result['executed_trades']:
            assert 'symbol' in trade
            assert 'action' in trade
            assert trade['action'] in ['buy', 'sell']
            assert 'amount' in trade


class TestSendAlerts:
    """Test alert and notification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.side_effect = [
            {'trading_signals': {'AAPL': {'signal': 'buy'}}, 'signal_summary': {'high_confidence_signals': 3}},
            {'portfolio_risk_assessment': {'risk_violations': ['AAPL: 3% exceeds 2% limit']}},
            {'calculated_positions': {'AAPL': {'dollar_amount': 5000}}},
            {'executed_trades': [{'symbol': 'AAPL', 'action': 'buy', 'amount': 1000}]}
        ]
    
    @patch('src.dags.trading_dag.AlertManager')
    def test_send_alerts_success(self, mock_alert_manager):
        """Test successful alert sending."""
        mock_am = Mock()
        mock_am.send_alerts.return_value = {
            'status': 'success',
            'alerts_sent': 3
        }
        mock_alert_manager.return_value = mock_am
        
        result = send_alerts(**self.mock_context)
        
        assert 'alert_summary' in result
        assert 'alert_data_processed' in result
        assert 'notifications_sent' in result
        assert result['alert_summary']['total_alerts_generated'] >= 0
        self.mock_context['task_instance'].xcom_push.assert_called_once()
    
    def test_send_alerts_types(self):
        """Test different alert types generation."""
        result = send_alerts(**self.mock_context)
        
        # Check alert data was processed
        alert_data = result['alert_data_processed']
        
        # Should process different types of alert data
        assert 'risk_violations' in alert_data
        assert 'strong_signals' in alert_data
        assert 'performance_issues' in alert_data
        
        # Check that we have alert summary
        assert 'alert_summary' in result
        assert 'total_alerts_generated' in result['alert_summary']


class TestTradingDAGIntegration:
    """Test trading DAG integration and workflow."""
    
    def test_task_dependencies(self):
        """Test task dependency configuration."""
        # Get task dependency mapping
        task_dict = {task.task_id: task for task in dag.tasks}
        
        # Check generate_trading_signals has no upstream dependencies
        assert len(task_dict['generate_trading_signals'].upstream_task_ids) == 0
        
        # Check assess_portfolio_risk depends on generate_trading_signals
        assert 'generate_trading_signals' in task_dict['assess_portfolio_risk'].upstream_task_ids
        
        # Check calculate_position_sizes depends on both signals and risk
        calc_upstream = task_dict['calculate_position_sizes'].upstream_task_ids
        assert 'generate_trading_signals' in calc_upstream
        assert 'assess_portfolio_risk' in calc_upstream
        
        # Check manage_portfolio depends on calculate_position_sizes
        assert 'calculate_position_sizes' in task_dict['manage_portfolio'].upstream_task_ids
        
        # Check send_alerts depends on risk and portfolio management
        alerts_upstream = task_dict['send_alerts'].upstream_task_ids
        assert 'assess_portfolio_risk' in alerts_upstream
        assert 'manage_portfolio' in alerts_upstream
    
    def test_dag_task_count(self):
        """Test DAG has exactly 5 tasks."""
        assert len(dag.tasks) == 5
    
    def test_dag_schedule_weekdays_only(self):
        """Test DAG is scheduled for weekdays only."""
        # Schedule: '0 9,15 * * 1-5' means 9 AM and 3 PM on weekdays
        assert '1-5' in dag.schedule_interval  # Monday to Friday
        assert '9,15' in dag.schedule_interval  # 9 AM and 3 PM