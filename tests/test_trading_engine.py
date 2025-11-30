"""Tests for Enhanced Trading Engine"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.core.trading_engine import TradingEngine, AlertPriority, Alert, calculate_returns, log_performance

class TestTradingEngine:
    def setup_method(self):
        self.engine = TradingEngine()
        self.sample_data = {
            'technical': {
                'price_data': pd.Series([100, 102, 98, 105, 103, 107, 104, 110, 112, 115, 108, 120, 118, 125, 123, 130, 128, 135, 133, 140], 
                                       index=pd.date_range('2024-01-01', periods=20)),
                'volume_data': pd.Series([1000, 1200, 800, 1500, 1100, 1300, 900, 1600, 1400, 1700, 1000, 1800, 1200, 1900, 1300, 2000, 1100, 2100, 1400, 2200],
                                        index=pd.date_range('2024-01-01', periods=20))
            },
            'fundamental': {'financial_metrics': {'pe_ratio': 12.5, 'pb_ratio': 1.3, 'debt_to_equity': 0.4, 'roe': 0.18, 'current_ratio': 2.1}}
        }
        self.strategy_results = {
            'momentum': {'signal': 'buy', 'confidence': 0.8, 'reasoning': 'Strong upward momentum detected', 'performance': {'momentum_value': 0.025}},
            'mean_reversion': {'signal': 'hold', 'confidence': 0.3, 'reasoning': 'Price near mean', 'performance': {'z_score': 0.1}},
            'breakout': {'signal': 'buy', 'confidence': 0.7, 'reasoning': 'Price breakout confirmed', 'performance': {'volume_confirmed': True}},
            'value': {'signal': 'buy', 'confidence': 0.6, 'reasoning': 'Good fundamental metrics', 'performance': {'value_score': 4}}
        }
        self.consensus_result = {'overall_signal': 'buy', 'confidence': 0.7, 'signal_distribution': {'buy': 3, 'sell': 0, 'hold': 1}}

    def test_trading_strategies_comprehensive(self):
        engine = TradingEngine({'test': 'config'})
        assert engine.config == {'test': 'config'} and hasattr(engine, 'logger')
        
        # Test all strategies with sample data
        momentum = self.engine.momentum_strategy(self.sample_data)
        assert all(k in momentum for k in ['signal', 'confidence', 'reasoning', 'performance'])
        assert isinstance(momentum['confidence'], float) and 0 <= momentum['confidence'] <= 1
        
        mean_rev = self.engine.mean_reversion_strategy(self.sample_data)
        assert 'z_score' in mean_rev['performance']
        
        breakout = self.engine.breakout_strategy(self.sample_data)
        assert 'volume_confirmed' in breakout['performance']
        assert isinstance(breakout['performance']['volume_confirmed'], (bool, np.bool_))
        
        value = self.engine.value_strategy(self.sample_data)
        assert 'value_score' in value['performance']
        
        # Test with insufficient data
        empty_data = {'technical': {'price_data': pd.Series(dtype=float)}, 'fundamental': {}}
        empty_result = self.engine.momentum_strategy(empty_data)
        assert empty_result['signal'] == 'hold' and empty_result['confidence'] == 0.0
        
        # Test value strategy without fundamentals
        no_fundamental = {'technical': self.sample_data['technical'], 'fundamental': {}}
        empty_value = self.engine.value_strategy(no_fundamental)
        assert empty_value['signal'] == 'hold' and empty_value['confidence'] == 0.0

    def test_explanation_and_scoring(self):
        # Test explanation generation
        result = self.engine.generate_explanation(self.strategy_results, self.consensus_result)
        assert result['status'] == 'success' and 'explanation' in result and 'timestamp' in result
        explanation = result['explanation']
        assert all(k in explanation for k in ['decision_summary', 'strategy_breakdown', 'risk_analysis', 'confidence_attribution', 'recommendations'])
        
        # Test error handling
        with patch.object(self.engine, '_create_decision_summary', side_effect=Exception("Test error")):
            error_result = self.engine.generate_explanation({}, {})
            assert error_result['status'] == 'error' and 'error' in error_result['explanation']
        
        # Test strategy scoring
        scores = self.engine.calculate_strategy_scores(self.strategy_results)
        assert isinstance(scores, dict) and len(scores) == len(self.strategy_results)
        for strategy_name, score in scores.items():
            assert isinstance(score, float) and 0 <= score <= 100
        assert scores['momentum'] > scores['mean_reversion']  # Buy/sell signals > hold

    def test_performance_attribution(self):
        portfolio_performance = {'total_return': 0.15}
        result = self.engine.attribute_performance(self.strategy_results, portfolio_performance)
        assert all(k in result for k in ['attribution', 'total_attribution', 'timestamp'])
        attribution = result['attribution']
        assert len(attribution) == len(self.strategy_results)
        
        # Check weights sum to 1 and attribution sums to total return
        total_weight = sum(attr['weight'] for attr in attribution.values())
        total_attributed = sum(attr['attributed_return'] for attr in attribution.values())
        assert abs(total_weight - 1.0) < 0.01 and abs(total_attributed - 0.15) < 0.01
        
        # Test zero confidence case
        zero_confidence_results = {'momentum': {'signal': 'hold', 'confidence': 0.0}, 'value': {'signal': 'hold', 'confidence': 0.0}}
        zero_result = self.engine.attribute_performance(zero_confidence_results, {'total_return': 0.10})
        assert zero_result['attribution'] == {} and zero_result['total_attribution'] == 0.0

    def test_alert_system_comprehensive(self):
        alert_data = {'risk_violations': ['Position size exceeded'], 'strong_signals': [{'symbol': 'AAPL', 'signal': 'buy', 'confidence': 0.9}], 'performance_issues': ['High drawdown detected']}
        result = self.engine.send_alerts(alert_data)
        assert result['status'] == 'success' and result['alerts_generated'] == 3 and result['notifications_sent'] == 3
        risk_alert = next(n for n in result['notifications'] if 'Risk Violation' in n['message'])
        assert risk_alert['channel'] == 'email'
        
        # Test error handling
        with patch.object(Alert, '__init__', side_effect=Exception("Test error")):
            error_result = self.engine.send_alerts({'risk_violations': ['test']})
            assert error_result['status'] == 'error' and 'error' in error_result

    def test_helper_methods_comprehensive(self):
        summary = self.engine._create_decision_summary({'overall_signal': 'buy', 'confidence': 0.8})
        assert 'BUY' in summary and 'high confidence' in summary
        breakdown = self.engine._analyze_strategy_breakdown(self.strategy_results)
        assert isinstance(breakdown, dict) and len(breakdown) == len(self.strategy_results)
        low_consensus = {'signal_distribution': {'buy': 2, 'sell': 2, 'hold': 1}, 'confidence': 0.3}
        risk_factors = self.engine._assess_risk_factors(self.strategy_results, low_consensus)
        assert any('Low strategy consensus' in factor for factor in risk_factors)
        attribution = self.engine._attribute_confidence(self.strategy_results)
        assert abs(sum(attribution.values()) - 100.0) < 0.1
        recommendations = self.engine._generate_recommendations({'overall_signal': 'buy', 'confidence': 0.8})
        assert any('long position' in rec for rec in recommendations)

class TestSharedUtilities:
    def test_calculate_returns_comprehensive(self):
        # Simple returns
        prices = pd.Series([100, 102, 98, 105])
        returns = calculate_returns(prices)
        assert len(returns) == len(prices) and abs(returns.iloc[1] - 0.02) < 0.01
        
        # Log returns
        log_returns = calculate_returns(prices, returns_type="log")
        assert log_returns.iloc[0] == 0.0 and log_returns.iloc[1] > 0 and log_returns.iloc[2] < 0
        
        # Edge cases
        assert calculate_returns(pd.Series(dtype=float)).empty
        assert calculate_returns(pd.Series([100])).empty

    def test_log_performance_comprehensive(self):
        start_time = datetime.now()
        end_time = datetime.now()
        performance_data = {'return': 0.15, 'sharpe_ratio': 1.25, 'max_drawdown': 0.08, 'trades': 50}
        
        # Test that log_performance executes without error
        result = log_performance('Test Strategy', start_time, end_time, 'success', performance_data)
        assert result is not None  # Function should return something
        
        # Test error case
        error_result = log_performance('Error Strategy', start_time, end_time, 'error', {'return': 0.15})
        assert error_result is not None

class TestAlertDataStructures:
    def test_alert_comprehensive(self):
        alert = Alert("Test alert", AlertPriority.HIGH, datetime.now(), "test", {'key': 'value'})
        assert alert.message == "Test alert" and alert.priority == AlertPriority.HIGH and alert.category == "test" and alert.details == {'key': 'value'}
        
        # Test enum values
        assert AlertPriority.LOW.value == "low" and AlertPriority.MEDIUM.value == "medium" and AlertPriority.HIGH.value == "high" and AlertPriority.CRITICAL.value == "critical"
        
        # Test without details
        simple_alert = Alert("Simple alert", AlertPriority.MEDIUM, datetime.now(), "simple")
        assert simple_alert.details is None

class TestPaperTradingEngine:
    def setup_method(self):
        self.engine = TradingEngine()
        self.signal_data = {'symbol': 'AAPL', 'signal': 'buy', 'confidence': 0.8, 'price': 150.0, 'strategy_source': 'momentum'}
        self.portfolio_state = {'total_value': 100000, 'cash': 50000, 'positions': {'AAPL': 50, 'SPY': 100}}
        self.market_data = {'AAPL': {'price': 150.0, 'status': 'success'}, 'SPY': {'price': 450.0, 'status': 'success'}}

    def test_paper_trading_comprehensive(self):
        # Buy signal execution
        result = self.engine.execute_paper_trade(self.signal_data, self.portfolio_state)
        assert result['status'] == 'executed' and 'trade' in result and 'portfolio_state' in result and 'quality_check' in result
        trade = result['trade']
        assert trade['symbol'] == 'AAPL' and trade['action'] == 'buy' and trade['shares'] > 0 and trade['value'] == trade['shares'] * trade['price'] and trade['confidence'] == 0.8
        
        # Sell signal execution
        sell_signal = self.signal_data.copy()
        sell_signal['signal'] = 'sell'
        sell_result = self.engine.execute_paper_trade(sell_signal, self.portfolio_state)
        assert sell_result['status'] == 'executed' and sell_result['trade']['action'] == 'sell'
        assert sell_result['portfolio_state']['positions']['AAPL'] <= self.portfolio_state['positions']['AAPL']
        
        # Low confidence rejection
        low_conf_signal = self.signal_data.copy()
        low_conf_signal['confidence'] = 0.3
        low_result = self.engine.execute_paper_trade(low_conf_signal, self.portfolio_state)
        assert low_result['status'] == 'no_action' and low_result['trade'] is None and 'below threshold' in low_result['reason']
        
        # Hold signal
        hold_signal = self.signal_data.copy()
        hold_signal['signal'] = 'hold'
        hold_result = self.engine.execute_paper_trade(hold_signal, self.portfolio_state)
        assert hold_result['status'] == 'no_action' and hold_result['trade'] is None

    def test_portfolio_metrics_comprehensive(self):
        result = self.engine.calculate_portfolio_metrics(self.portfolio_state, self.market_data)
        assert result['status'] == 'success' and 'metrics' in result
        metrics = result['metrics']
        assert all(k in metrics for k in ['total_value', 'diversification_score', 'concentration_risk', 'data_quality', 'timestamp'])
        
        # Test with empty positions
        empty_portfolio = {'total_value': 50000, 'cash': 50000, 'positions': {}}
        empty_result = self.engine.calculate_portfolio_metrics(empty_portfolio, self.market_data)
        assert empty_result['status'] == 'success'
        empty_metrics = empty_result['metrics']
        assert empty_metrics['total_value'] == 50000 and empty_metrics['diversification_score'] == 1.0 and empty_metrics['concentration_risk'] == 0.0

class TestErrorScenarios:
    def test_error_handling_comprehensive(self):
        engine = TradingEngine()
        
        # Strategy with exception
        invalid_data = {'technical': {'price_data': 'invalid'}}
        result = engine.momentum_strategy(invalid_data)
        assert result['signal'] == 'hold' and result['confidence'] == 0.0 and 'Error:' in result['reasoning']
        
        # Paper trading error
        with patch('src.core.trading_engine.validate_data_quality', side_effect=Exception("Test error")):
            trade_result = engine.execute_paper_trade({}, {})
            assert trade_result['status'] == 'error' and 'error' in trade_result and trade_result['trade'] is None
        
        # Portfolio metrics error
        with patch('src.core.trading_engine.validate_data_quality', side_effect=Exception("Test error")):
            metrics_result = engine.calculate_portfolio_metrics({}, {})
            assert metrics_result['status'] == 'error' and 'error' in metrics_result and metrics_result['metrics'] == {}

if __name__ == '__main__':
    pytest.main([__file__, '-v'])