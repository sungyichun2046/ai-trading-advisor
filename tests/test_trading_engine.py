"""
Tests for Enhanced Trading Engine
Testing explanation, scoring, and attribution functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.trading_engine import (
    TradingEngine, AlertPriority, Alert, calculate_returns, log_performance
)


class TestTradingEngine:
    """Test suite for enhanced TradingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TradingEngine()
        self.sample_data = {
            'technical': {
                'price_data': pd.Series([100, 102, 98, 105, 103, 107, 104, 110, 112, 115, 108, 120, 118, 125, 123, 130, 128, 135, 133, 140], 
                                       index=pd.date_range('2024-01-01', periods=20)),
                'volume_data': pd.Series([1000, 1200, 800, 1500, 1100, 1300, 900, 1600, 1400, 1700, 1000, 1800, 1200, 1900, 1300, 2000, 1100, 2100, 1400, 2200],
                                        index=pd.date_range('2024-01-01', periods=20))
            },
            'fundamental': {
                'financial_metrics': {
                    'pe_ratio': 12.5,
                    'pb_ratio': 1.3,
                    'debt_to_equity': 0.4,
                    'roe': 0.18,
                    'current_ratio': 2.1
                }
            }
        }
        
        self.strategy_results = {
            'momentum': {
                'signal': 'buy',
                'confidence': 0.8,
                'reasoning': 'Strong upward momentum detected',
                'performance': {'momentum_value': 0.025}
            },
            'mean_reversion': {
                'signal': 'hold',
                'confidence': 0.3,
                'reasoning': 'Price near mean',
                'performance': {'z_score': 0.1}
            },
            'breakout': {
                'signal': 'buy',
                'confidence': 0.7,
                'reasoning': 'Price breakout confirmed',
                'performance': {'volume_confirmed': True}
            },
            'value': {
                'signal': 'buy',
                'confidence': 0.6,
                'reasoning': 'Good fundamental metrics',
                'performance': {'value_score': 4}
            }
        }
        
        self.consensus_result = {
            'overall_signal': 'buy',
            'confidence': 0.7,
            'signal_distribution': {'buy': 3, 'sell': 0, 'hold': 1}
        }

    def test_trading_engine_initialization(self):
        """Test TradingEngine initialization."""
        engine = TradingEngine({'test': 'config'})
        assert engine.config == {'test': 'config'}
        assert hasattr(engine, 'logger')

    def test_momentum_strategy_buy_signal(self):
        """Test momentum strategy generating buy signal."""
        result = self.engine.momentum_strategy(self.sample_data)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert 'performance' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        assert result['signal'] in ['buy', 'sell', 'hold']

    def test_momentum_strategy_insufficient_data(self):
        """Test momentum strategy with insufficient data."""
        empty_data = {
            'technical': {'price_data': pd.Series(dtype=float)},
            'fundamental': {}
        }
        result = self.engine.momentum_strategy(empty_data)
        
        assert result['signal'] == 'hold'
        assert result['confidence'] == 0.0
        assert 'No price data' in result['reasoning']

    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy."""
        result = self.engine.mean_reversion_strategy(self.sample_data)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert 'z_score' in result['performance']
        assert isinstance(result['performance']['z_score'], (int, float))

    def test_breakout_strategy_with_volume(self):
        """Test breakout strategy with volume confirmation."""
        result = self.engine.breakout_strategy(self.sample_data)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'volume_confirmed' in result['performance']
        assert isinstance(result['performance']['volume_confirmed'], (bool, np.bool_))

    def test_value_strategy_with_fundamentals(self):
        """Test value strategy with fundamental data."""
        result = self.engine.value_strategy(self.sample_data)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'value_score' in result['performance']
        assert isinstance(result['performance']['value_score'], int)

    def test_value_strategy_no_fundamentals(self):
        """Test value strategy without fundamental data."""
        no_fundamental_data = {
            'technical': self.sample_data['technical'],
            'fundamental': {}
        }
        result = self.engine.value_strategy(no_fundamental_data)
        
        assert result['signal'] == 'hold'
        assert result['confidence'] == 0.0
        assert 'No fundamental data' in result['reasoning']

    def test_generate_explanation_success(self):
        """Test explanation generation."""
        result = self.engine.generate_explanation(self.strategy_results, self.consensus_result)
        
        assert result['status'] == 'success'
        assert 'explanation' in result
        assert 'timestamp' in result
        
        explanation = result['explanation']
        assert 'decision_summary' in explanation
        assert 'strategy_breakdown' in explanation
        assert 'risk_analysis' in explanation
        assert 'confidence_attribution' in explanation
        assert 'recommendations' in explanation

    def test_generate_explanation_error_handling(self):
        """Test explanation generation with invalid input."""
        with patch.object(self.engine, '_create_decision_summary', side_effect=Exception("Test error")):
            result = self.engine.generate_explanation({}, {})
            assert result['status'] == 'error'
            assert 'error' in result['explanation']

    def test_calculate_strategy_scores(self):
        """Test strategy score calculation."""
        scores = self.engine.calculate_strategy_scores(self.strategy_results)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(self.strategy_results)
        
        for strategy_name, score in scores.items():
            assert isinstance(score, float)
            assert 0 <= score <= 100
            
        # Buy/sell signals should have higher scores than hold
        assert scores['momentum'] > scores['mean_reversion']

    def test_attribute_performance(self):
        """Test performance attribution."""
        portfolio_performance = {'total_return': 0.15}
        result = self.engine.attribute_performance(self.strategy_results, portfolio_performance)
        
        assert 'attribution' in result
        assert 'total_attribution' in result
        assert 'timestamp' in result
        
        attribution = result['attribution']
        assert len(attribution) == len(self.strategy_results)
        
        # Check weights sum to 1
        total_weight = sum(attr['weight'] for attr in attribution.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check attribution sums to total return
        total_attributed = sum(attr['attributed_return'] for attr in attribution.values())
        assert abs(total_attributed - 0.15) < 0.01

    def test_attribute_performance_zero_confidence(self):
        """Test performance attribution with zero total confidence."""
        zero_confidence_results = {
            'momentum': {'signal': 'hold', 'confidence': 0.0},
            'value': {'signal': 'hold', 'confidence': 0.0}
        }
        portfolio_performance = {'total_return': 0.10}
        
        result = self.engine.attribute_performance(zero_confidence_results, portfolio_performance)
        assert result['attribution'] == {}
        assert result['total_attribution'] == 0.0

    def test_send_alerts_risk_violations(self):
        """Test alert system with risk violations."""
        alert_data = {
            'risk_violations': ['Position size exceeded', 'Daily loss limit reached'],
            'strong_signals': [],
            'performance_issues': []
        }
        
        result = self.engine.send_alerts(alert_data)
        
        assert result['status'] == 'success'
        assert result['alerts_generated'] == 2
        assert result['notifications_sent'] == 2
        assert len(result['notifications']) == 2
        
        # Check high priority for risk violations
        for notification in result['notifications']:
            if 'Risk Violation' in notification['message']:
                assert notification['channel'] == 'email'

    def test_send_alerts_strong_signals(self):
        """Test alert system with strong signals."""
        alert_data = {
            'risk_violations': [],
            'strong_signals': [
                {'symbol': 'AAPL', 'signal': 'buy', 'confidence': 0.9},
                {'symbol': 'MSFT', 'signal': 'sell', 'confidence': 0.7}
            ],
            'performance_issues': []
        }
        
        result = self.engine.send_alerts(alert_data)
        
        assert result['status'] == 'success'
        assert result['alerts_generated'] == 2
        
        # High confidence signal should be medium priority (email)
        high_conf_alert = next(n for n in result['notifications'] if 'AAPL' in n['message'])
        assert high_conf_alert['channel'] == 'email'

    def test_send_alerts_performance_issues(self):
        """Test alert system with performance issues."""
        alert_data = {
            'risk_violations': [],
            'strong_signals': [],
            'performance_issues': ['High drawdown detected', 'Low Sharpe ratio']
        }
        
        result = self.engine.send_alerts(alert_data)
        
        assert result['status'] == 'success'
        assert result['alerts_generated'] == 2
        assert all(n['priority'] == 'medium' for n in result['notifications'])

    def test_send_alerts_error_handling(self):
        """Test alert system error handling."""
        # Test with invalid input
        with patch.object(Alert, '__init__', side_effect=Exception("Test error")):
            result = self.engine.send_alerts({'risk_violations': ['test']})
            assert result['status'] == 'error'
            assert 'error' in result

    def test_create_decision_summary(self):
        """Test decision summary creation."""
        # High confidence buy
        high_conf_result = {'overall_signal': 'buy', 'confidence': 0.8}
        summary = self.engine._create_decision_summary(high_conf_result)
        assert 'BUY' in summary
        assert 'high confidence' in summary
        
        # Low confidence hold
        low_conf_result = {'overall_signal': 'hold', 'confidence': 0.2}
        summary = self.engine._create_decision_summary(low_conf_result)
        assert 'HOLD' in summary
        assert 'low confidence' in summary

    def test_analyze_strategy_breakdown(self):
        """Test strategy breakdown analysis."""
        breakdown = self.engine._analyze_strategy_breakdown(self.strategy_results)
        
        assert isinstance(breakdown, dict)
        assert len(breakdown) == len(self.strategy_results)
        
        for strategy_name, description in breakdown.items():
            assert strategy_name in self.strategy_results
            assert isinstance(description, str)
            assert len(description) > 0

    def test_assess_risk_factors(self):
        """Test risk factor assessment."""
        # Test low consensus scenario
        low_consensus_result = {
            'signal_distribution': {'buy': 2, 'sell': 2, 'hold': 1},
            'confidence': 0.3
        }
        
        risk_factors = self.engine._assess_risk_factors(self.strategy_results, low_consensus_result)
        
        assert isinstance(risk_factors, list)
        assert any('Low strategy consensus' in factor for factor in risk_factors)
        assert any('Low overall confidence' in factor for factor in risk_factors)

    def test_attribute_confidence(self):
        """Test confidence attribution."""
        attribution = self.engine._attribute_confidence(self.strategy_results)
        
        assert isinstance(attribution, dict)
        assert len(attribution) == len(self.strategy_results)
        
        # Check percentages sum to 100
        total_percentage = sum(attribution.values())
        assert abs(total_percentage - 100.0) < 0.1
        
        # Highest confidence strategy should have highest attribution
        max_attribution_strategy = max(attribution.keys(), key=lambda k: attribution[k])
        max_confidence_strategy = max(self.strategy_results.keys(), 
                                    key=lambda k: self.strategy_results[k]['confidence'])
        assert max_attribution_strategy == max_confidence_strategy

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # High confidence buy
        buy_result = {'overall_signal': 'buy', 'confidence': 0.8}
        recommendations = self.engine._generate_recommendations(buy_result)
        assert any('long position' in rec for rec in recommendations)
        
        # High confidence sell
        sell_result = {'overall_signal': 'sell', 'confidence': 0.8}
        recommendations = self.engine._generate_recommendations(sell_result)
        assert any('exiting long' in rec or 'short' in rec for rec in recommendations)
        
        # Low confidence hold
        hold_result = {'overall_signal': 'hold', 'confidence': 0.3}
        recommendations = self.engine._generate_recommendations(hold_result)
        assert any('Wait for clearer signals' in rec for rec in recommendations)


class TestSharedUtilities:
    """Test suite for shared utility functions."""

    def test_calculate_returns_simple(self):
        """Test simple returns calculation."""
        prices = pd.Series([100, 102, 98, 105])
        returns = calculate_returns(prices)
        
        expected_returns = pd.Series([0.0, 0.02, -0.0392157, 0.0714286])
        
        assert len(returns) == len(prices)
        assert abs(returns.iloc[1] - 0.02) < 0.01
        assert abs(returns.iloc[2] - (-0.0392)) < 0.01

    def test_calculate_returns_log(self):
        """Test log returns calculation."""
        prices = pd.Series([100, 102, 98])
        returns = calculate_returns(prices, returns_type="log")
        
        assert len(returns) == len(prices)
        assert returns.iloc[0] == 0.0  # First return is always 0
        assert returns.iloc[1] > 0  # Price increased
        assert returns.iloc[2] < 0  # Price decreased

    def test_calculate_returns_empty_series(self):
        """Test returns calculation with empty series."""
        empty_prices = pd.Series(dtype=float)
        returns = calculate_returns(empty_prices)
        
        assert returns.empty

    def test_calculate_returns_single_price(self):
        """Test returns calculation with single price."""
        single_price = pd.Series([100])
        returns = calculate_returns(single_price)
        
        assert returns.empty

    @patch('src.core.trading_engine.logger')
    def test_log_performance_float_values(self, mock_logger):
        """Test performance logging with float values."""
        performance_data = {
            'return': 0.15,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'trades': 50
        }
        
        log_performance('Test Strategy', performance_data)
        
        # Verify logger.info was called
        assert mock_logger.info.called
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        
        # Check strategy name logging
        assert any('Test Strategy' in arg for arg in call_args)

    @patch('src.core.trading_engine.logger')
    def test_log_performance_error_handling(self, mock_logger):
        """Test performance logging error handling."""
        # Simulate error in logging
        mock_logger.info.side_effect = Exception("Logging error")
        
        performance_data = {'return': 0.15}
        
        # Should not raise exception
        log_performance('Error Strategy', performance_data)
        
        # Should log error
        assert mock_logger.error.called


class TestAlertDataStructures:
    """Test suite for Alert data structures."""

    def test_alert_creation(self):
        """Test Alert dataclass creation."""
        alert = Alert(
            message="Test alert",
            priority=AlertPriority.HIGH,
            timestamp=datetime.now(),
            category="test",
            details={'key': 'value'}
        )
        
        assert alert.message == "Test alert"
        assert alert.priority == AlertPriority.HIGH
        assert alert.category == "test"
        assert alert.details == {'key': 'value'}

    def test_alert_priority_enum(self):
        """Test AlertPriority enum values."""
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.MEDIUM.value == "medium"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.CRITICAL.value == "critical"

    def test_alert_without_details(self):
        """Test Alert creation without details."""
        alert = Alert(
            message="Simple alert",
            priority=AlertPriority.MEDIUM,
            timestamp=datetime.now(),
            category="simple"
        )
        
        assert alert.details is None


class TestErrorScenarios:
    """Test suite for error scenarios and edge cases."""

    def test_strategy_with_exception(self):
        """Test strategy execution with exception."""
        engine = TradingEngine()
        
        # Invalid data that should cause errors
        invalid_data = {'technical': {'price_data': 'invalid'}}
        
        result = engine.momentum_strategy(invalid_data)
        assert result['signal'] == 'hold'
        assert result['confidence'] == 0.0
        assert 'Error:' in result['reasoning']

    def test_explanation_with_empty_results(self):
        """Test explanation generation with empty strategy results."""
        engine = TradingEngine()
        
        result = engine.generate_explanation({}, {})
        assert result['status'] == 'success'
        assert 'explanation' in result

    def test_strategy_scores_empty_input(self):
        """Test strategy score calculation with empty input."""
        engine = TradingEngine()
        
        scores = engine.calculate_strategy_scores({})
        assert scores == {}

    def test_send_alerts_empty_data(self):
        """Test alert system with empty data."""
        engine = TradingEngine()
        
        result = engine.send_alerts({})
        assert result['status'] == 'success'
        assert result['alerts_generated'] == 0
        assert result['notifications_sent'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])