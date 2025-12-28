"""Tests for portfolio tracking utilities."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.utils.portfolio_tracker import PortfolioTracker, calculate_portfolio_performance


class TestPortfolioTracker:
    """Test portfolio tracking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio_tracker = PortfolioTracker()
        
        # Sample positions data
        self.sample_positions = {
            'AAPL': {'quantity': 100, 'avg_cost': 150.0, 'current_price': 160.0},
            'GOOGL': {'quantity': 50, 'avg_cost': 2000.0, 'current_price': 2100.0},
            'MSFT': {'quantity': 75, 'avg_cost': 300.0, 'current_price': 290.0}
        }
        
        # Sample trade history
        self.sample_trades = [
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 100, 'price': 150.0, 'timestamp': datetime.now() - timedelta(days=10)},
            {'symbol': 'GOOGL', 'action': 'buy', 'quantity': 50, 'price': 2000.0, 'timestamp': datetime.now() - timedelta(days=8)},
            {'symbol': 'MSFT', 'action': 'buy', 'quantity': 75, 'price': 300.0, 'timestamp': datetime.now() - timedelta(days=5)}
        ]
    
    def test_portfolio_tracker_initialization(self):
        """Test portfolio tracker initialization."""
        assert len(self.portfolio_tracker.positions) == 0
        assert len(self.portfolio_tracker.trade_history) == 0
        assert len(self.portfolio_tracker.performance_history) == 0
    
    def test_calculate_portfolio_performance_basic(self):
        """Test basic portfolio performance calculation."""
        result = self.portfolio_tracker.calculate_portfolio_performance(self.sample_positions)
        
        assert 'total_cost_basis' in result
        assert 'total_market_value' in result
        assert 'total_unrealized_pnl' in result
        assert 'total_return_pct' in result
        assert 'positions' in result
        
        # Verify calculations
        expected_cost = (100 * 150.0) + (50 * 2000.0) + (75 * 300.0)  # 137,500
        expected_market = (100 * 160.0) + (50 * 2100.0) + (75 * 290.0)  # 142,750
        expected_pnl = expected_market - expected_cost  # 5,250
        
        assert result['total_cost_basis'] == expected_cost
        assert result['total_market_value'] == expected_market
        assert result['total_unrealized_pnl'] == expected_pnl
        assert result['position_count'] == 3
    
    def test_calculate_portfolio_performance_empty_positions(self):
        """Test portfolio performance with empty positions."""
        result = self.portfolio_tracker.calculate_portfolio_performance({})
        
        assert result['total_cost_basis'] == 0
        assert result['total_market_value'] == 0
        assert result['total_unrealized_pnl'] == 0
        assert result['total_return_pct'] == 0
        assert result['positions'] == []
    
    def test_calculate_portfolio_performance_with_current_prices(self):
        """Test portfolio performance with updated current prices."""
        current_prices = {'AAPL': 165.0, 'GOOGL': 2050.0, 'MSFT': 295.0}
        
        result = self.portfolio_tracker.calculate_portfolio_performance(
            self.sample_positions, current_prices
        )
        
        # Verify prices were updated
        aapl_position = next(p for p in result['positions'] if p['symbol'] == 'AAPL')
        assert aapl_position['current_price'] == 165.0
    
    def test_track_portfolio_positions_from_trades(self):
        """Test position tracking from trade history."""
        positions = self.portfolio_tracker.track_portfolio_positions(self.sample_trades)
        
        assert 'AAPL' in positions
        assert 'GOOGL' in positions
        assert 'MSFT' in positions
        
        # Verify AAPL position
        aapl_pos = positions['AAPL']
        assert aapl_pos['quantity'] == 100
        assert aapl_pos['avg_cost'] == 150.0
        assert aapl_pos['total_cost'] == 15000.0
    
    def test_track_portfolio_positions_with_sells(self):
        """Test position tracking with sell trades."""
        trades_with_sells = self.sample_trades + [
            {'symbol': 'AAPL', 'action': 'sell', 'quantity': 50, 'price': 160.0, 'timestamp': datetime.now()}
        ]
        
        positions = self.portfolio_tracker.track_portfolio_positions(trades_with_sells)
        
        # AAPL position should be reduced
        aapl_pos = positions['AAPL']
        assert aapl_pos['quantity'] == 50  # 100 - 50
        assert aapl_pos['avg_cost'] == 150.0  # Should remain same
    
    def test_track_portfolio_positions_complete_sell(self):
        """Test position tracking with complete position closure."""
        trades_with_complete_sell = self.sample_trades + [
            {'symbol': 'AAPL', 'action': 'sell', 'quantity': 100, 'price': 160.0, 'timestamp': datetime.now()}
        ]
        
        positions = self.portfolio_tracker.track_portfolio_positions(trades_with_complete_sell)
        
        # AAPL should be removed from positions
        assert 'AAPL' not in positions
        assert 'GOOGL' in positions  # Others should remain
    
    def test_calculate_performance_metrics_basic(self):
        """Test basic performance metrics calculation."""
        returns_data = [0.02, -0.01, 0.03, 0.015, -0.005]  # 5 days of returns
        
        metrics = self.portfolio_tracker.calculate_performance_metrics(returns_data)
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Verify win rate calculation
        positive_returns = sum(1 for r in returns_data if r > 0)
        expected_win_rate = (positive_returns / len(returns_data)) * 100
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01
    
    def test_calculate_performance_metrics_with_benchmark(self):
        """Test performance metrics calculation with benchmark."""
        returns_data = [0.02, -0.01, 0.03, 0.015, -0.005]
        benchmark_returns = [0.015, -0.005, 0.025, 0.01, 0.0]
        
        metrics = self.portfolio_tracker.calculate_performance_metrics(
            returns_data, benchmark_returns
        )
        
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'benchmark_return' in metrics
        assert 'excess_return' in metrics
    
    def test_calculate_performance_metrics_empty_returns(self):
        """Test performance metrics with empty returns data."""
        metrics = self.portfolio_tracker.calculate_performance_metrics([])
        assert metrics == {}
    
    def test_get_position_summary(self):
        """Test position summary retrieval."""
        # Set some positions
        self.portfolio_tracker.positions = {
            'AAPL': {'total_cost': 15000.0},
            'GOOGL': {'total_cost': 100000.0}
        }
        
        summary = self.portfolio_tracker.get_position_summary()
        
        assert summary['total_positions'] == 2
        assert summary['total_value'] == 115000.0
        assert len(summary['positions']) == 2
        assert 'timestamp' in summary
    
    def test_get_position_summary_empty(self):
        """Test position summary with empty positions."""
        summary = self.portfolio_tracker.get_position_summary()
        
        assert summary['total_positions'] == 0
        assert summary['total_value'] == 0
        assert summary['positions'] == []
    
    def test_get_performance_trend(self):
        """Test performance trend analysis."""
        # Add some performance history
        for i in range(10):
            self.portfolio_tracker.performance_history.append({
                'total_return_pct': i * 0.5,  # Increasing trend
                'timestamp': (datetime.now() - timedelta(days=10-i)).isoformat()
            })
        
        trend = self.portfolio_tracker.get_performance_trend(days=30)
        
        assert 'trend' in trend
        assert 'data_points' in trend
        assert 'latest_return' in trend
        assert 'avg_return' in trend
        assert trend['data_points'] == 10
        assert trend['trend'] in ['upward', 'downward', 'sideways']
    
    def test_get_performance_trend_insufficient_data(self):
        """Test performance trend with insufficient data."""
        trend = self.portfolio_tracker.get_performance_trend()
        
        assert trend['trend'] == 'no_data'
        assert trend['data_points'] == 0
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        position_details = [
            {'symbol': 'AAPL', 'market_value': 16000.0},
            {'symbol': 'GOOGL', 'market_value': 105000.0},
            {'symbol': 'MSFT', 'market_value': 21750.0}
        ]
        
        risk_metrics = self.portfolio_tracker._calculate_risk_metrics(position_details)
        
        assert 'concentration_risk' in risk_metrics
        assert 'max_position_weight' in risk_metrics
        assert 'diversification_score' in risk_metrics
        assert 'effective_positions' in risk_metrics
        
        # GOOGL should be the largest position
        total_value = sum(p['market_value'] for p in position_details)
        googl_weight = 105000.0 / total_value
        assert abs(risk_metrics['max_position_weight'] - (googl_weight * 100)) < 0.01
    
    def test_performance_history_trimming(self):
        """Test performance history trimming to prevent memory bloat."""
        # Add many entries to trigger trimming
        for i in range(1200):  # More than max_entries of 1000
            self.portfolio_tracker.performance_history.append({
                'total_return_pct': i * 0.1,
                'timestamp': datetime.now().isoformat()
            })
        
        self.portfolio_tracker._trim_history(max_entries=500)
        
        # Should be trimmed to max_entries
        assert len(self.portfolio_tracker.performance_history) == 500
    
    def test_error_handling_in_performance_calculation(self):
        """Test error handling in performance calculations."""
        # Test with malformed positions data
        malformed_positions = {
            'INVALID': {'quantity': 'not_a_number', 'avg_cost': None}
        }
        
        result = self.portfolio_tracker.calculate_portfolio_performance(malformed_positions)
        
        # Should return empty metrics structure on error
        assert result['total_cost_basis'] == 0
        assert result['total_market_value'] == 0
        assert 'timestamp' in result
    
    def test_error_handling_in_performance_metrics(self):
        """Test error handling in performance metrics calculation."""
        # Test with invalid returns data
        with patch('numpy.array', side_effect=Exception("NumPy error")):
            metrics = self.portfolio_tracker.calculate_performance_metrics([0.1, 0.2])
            assert metrics == {}


class TestPortfolioTrackerConvenienceFunctions:
    """Test convenience functions for portfolio tracking."""
    
    def test_calculate_portfolio_performance_function(self):
        """Test calculate_portfolio_performance convenience function."""
        positions = {'AAPL': {'quantity': 100, 'avg_cost': 150.0, 'current_price': 160.0}}
        
        result = calculate_portfolio_performance(positions)
        
        assert 'total_cost_basis' in result
        assert 'total_market_value' in result
        assert result['total_cost_basis'] == 15000.0
        assert result['total_market_value'] == 16000.0


class TestPortfolioTrackerAdvanced:
    """Test advanced portfolio tracking features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio_tracker = PortfolioTracker()
    
    def test_complex_trading_scenario(self):
        """Test complex trading scenario with multiple buys and sells."""
        complex_trades = [
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 100, 'price': 140.0, 'timestamp': datetime.now() - timedelta(days=20)},
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 50, 'price': 160.0, 'timestamp': datetime.now() - timedelta(days=15)},
            {'symbol': 'AAPL', 'action': 'sell', 'quantity': 75, 'price': 155.0, 'timestamp': datetime.now() - timedelta(days=10)},
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 25, 'price': 150.0, 'timestamp': datetime.now() - timedelta(days=5)}
        ]
        
        positions = self.portfolio_tracker.track_portfolio_positions(complex_trades)
        
        # Should have net position
        assert 'AAPL' in positions
        aapl_pos = positions['AAPL']
        
        # Net quantity: 100 + 50 - 75 + 25 = 100
        assert aapl_pos['quantity'] == 100
        
        # Should have updated average cost
        assert aapl_pos['avg_cost'] > 0
    
    def test_portfolio_performance_with_mixed_positions(self):
        """Test portfolio performance with profitable and losing positions."""
        mixed_positions = {
            'WINNER': {'quantity': 100, 'avg_cost': 100.0, 'current_price': 120.0},  # +20% gain
            'LOSER': {'quantity': 50, 'avg_cost': 200.0, 'current_price': 180.0},    # -10% loss
            'FLAT': {'quantity': 200, 'avg_cost': 50.0, 'current_price': 50.0}       # No change
        }
        
        result = self.portfolio_tracker.calculate_portfolio_performance(mixed_positions)
        
        # Verify individual position calculations
        positions = result['positions']
        
        winner = next(p for p in positions if p['symbol'] == 'WINNER')
        assert winner['unrealized_pnl'] == 2000.0  # (120-100) * 100
        assert abs(winner['unrealized_pnl_pct'] - 20.0) < 0.01
        
        loser = next(p for p in positions if p['symbol'] == 'LOSER')
        assert loser['unrealized_pnl'] == -1000.0  # (180-200) * 50
        assert abs(loser['unrealized_pnl_pct'] - (-10.0)) < 0.01
        
        flat = next(p for p in positions if p['symbol'] == 'FLAT')
        assert flat['unrealized_pnl'] == 0.0
        assert flat['unrealized_pnl_pct'] == 0.0
    
    def test_performance_metrics_edge_cases(self):
        """Test performance metrics calculation edge cases."""
        # Test with all zero returns
        zero_returns = [0.0] * 10
        metrics = self.portfolio_tracker.calculate_performance_metrics(zero_returns)
        assert metrics['total_return'] == 0.0
        assert metrics['win_rate'] == 0.0
        
        # Test with all positive returns
        positive_returns = [0.01] * 10
        metrics = self.portfolio_tracker.calculate_performance_metrics(positive_returns)
        assert metrics['win_rate'] == 100.0
        assert metrics['max_drawdown'] == 0.0
        
        # Test with single return
        single_return = [0.05]
        metrics = self.portfolio_tracker.calculate_performance_metrics(single_return)
        assert 'total_return' in metrics
        assert metrics['total_trades'] == 1