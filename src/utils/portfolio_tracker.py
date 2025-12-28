"""
Real-time portfolio performance tracking and metrics calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Real-time portfolio performance tracking and analytics."""
    
    def __init__(self):
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        self.benchmark_data = {}
        self.lock = threading.RLock()
    
    def calculate_portfolio_performance(self, positions: Dict[str, Any], current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            positions: Dictionary of positions {symbol: {quantity, avg_cost, current_price}}
            current_prices: Optional current market prices
            
        Returns:
            Portfolio performance metrics
        """
        try:
            if not positions:
                return self._get_empty_portfolio_metrics()
            
            with self.lock:
                # Update current prices if provided
                if current_prices:
                    for symbol in positions:
                        if symbol in current_prices:
                            positions[symbol]['current_price'] = current_prices[symbol]
                
                # Calculate portfolio values
                total_cost = 0
                total_market_value = 0
                position_details = []
                
                for symbol, position in positions.items():
                    quantity = position.get('quantity', 0)
                    avg_cost = position.get('avg_cost', 0)
                    current_price = position.get('current_price', avg_cost)
                    
                    cost_basis = quantity * avg_cost
                    market_value = quantity * current_price
                    unrealized_pnl = market_value - cost_basis
                    
                    if cost_basis > 0:
                        pnl_pct = (unrealized_pnl / cost_basis) * 100
                    else:
                        pnl_pct = 0
                    
                    position_details.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'avg_cost': avg_cost,
                        'current_price': current_price,
                        'cost_basis': cost_basis,
                        'market_value': market_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': pnl_pct
                    })
                    
                    total_cost += cost_basis
                    total_market_value += market_value
                
                # Portfolio-level metrics
                total_pnl = total_market_value - total_cost
                total_return_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
                
                # Calculate additional metrics
                metrics = {
                    'total_cost_basis': total_cost,
                    'total_market_value': total_market_value,
                    'total_unrealized_pnl': total_pnl,
                    'total_return_pct': total_return_pct,
                    'positions': position_details,
                    'position_count': len(positions),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add risk metrics if we have enough data
                if len(position_details) > 1:
                    metrics.update(self._calculate_risk_metrics(position_details))
                
                # Store in history for trend analysis
                self.performance_history.append(metrics)
                self._trim_history()
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return self._get_empty_portfolio_metrics()
    
    def track_portfolio_positions(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Track portfolio positions from trade history.
        
        Args:
            trades: List of trade dictionaries {symbol, action, quantity, price, timestamp}
            
        Returns:
            Current positions dictionary
        """
        try:
            with self.lock:
                positions = {}
                
                for trade in sorted(trades, key=lambda x: x.get('timestamp', datetime.now())):
                    symbol = trade.get('symbol', '')
                    action = trade.get('action', '').lower()
                    quantity = trade.get('quantity', 0)
                    price = trade.get('price', 0)
                    
                    if symbol not in positions:
                        positions[symbol] = {'quantity': 0, 'total_cost': 0, 'avg_cost': 0}
                    
                    pos = positions[symbol]
                    
                    if action in ['buy', 'long']:
                        pos['quantity'] += quantity
                        pos['total_cost'] += quantity * price
                        if pos['quantity'] > 0:
                            pos['avg_cost'] = pos['total_cost'] / pos['quantity']
                    
                    elif action in ['sell', 'short']:
                        # For simplicity, handle as position reduction
                        if pos['quantity'] >= quantity:
                            sell_ratio = quantity / pos['quantity']
                            pos['quantity'] -= quantity
                            pos['total_cost'] *= (1 - sell_ratio)
                            if pos['quantity'] > 0:
                                pos['avg_cost'] = pos['total_cost'] / pos['quantity']
                            else:
                                pos['avg_cost'] = 0
                
                # Remove positions with zero quantity
                positions = {k: v for k, v in positions.items() if v['quantity'] > 0}
                
                self.positions = positions
                return positions
                
        except Exception as e:
            logger.error(f"Error tracking portfolio positions: {e}")
            return {}
    
    def calculate_performance_metrics(self, returns_data: List[float], benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics.
        
        Args:
            returns_data: List of portfolio returns
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if not returns_data:
                return {}
            
            returns = np.array(returns_data)
            
            # Basic metrics
            total_return = np.prod(1 + returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Win rate
            win_rate = np.sum(returns > 0) / len(returns) * 100
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'max_drawdown_pct': abs(max_drawdown) * 100,
                'win_rate': win_rate,
                'total_trades': len(returns),
                'avg_return': np.mean(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns)
            }
            
            # Benchmark comparison if provided
            if benchmark_returns and len(benchmark_returns) == len(returns):
                bench_returns = np.array(benchmark_returns)
                alpha = annualized_return - ((1 + np.prod(1 + bench_returns)) ** (252 / len(bench_returns)) - 1)
                
                # Beta calculation
                covariance = np.cov(returns, bench_returns)[0][1]
                benchmark_variance = np.var(bench_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                metrics.update({
                    'alpha': alpha,
                    'beta': beta,
                    'benchmark_return': np.prod(1 + bench_returns) - 1,
                    'excess_return': total_return - (np.prod(1 + bench_returns) - 1)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary."""
        try:
            with self.lock:
                if not self.positions:
                    return {'total_positions': 0, 'total_value': 0, 'positions': []}
                
                total_value = sum(pos.get('total_cost', 0) for pos in self.positions.values())
                
                return {
                    'total_positions': len(self.positions),
                    'total_value': total_value,
                    'positions': list(self.positions.items()),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {'total_positions': 0, 'total_value': 0, 'positions': []}
    
    def get_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trend over specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_history = [
                h for h in self.performance_history
                if datetime.fromisoformat(h['timestamp']) > cutoff_date
            ]
            
            if not recent_history:
                return {'trend': 'no_data', 'data_points': 0}
            
            returns = [h.get('total_return_pct', 0) for h in recent_history]
            
            # Calculate trend
            if len(returns) > 1:
                trend_slope = np.polyfit(range(len(returns)), returns, 1)[0]
                if trend_slope > 0.1:
                    trend = 'upward'
                elif trend_slope < -0.1:
                    trend = 'downward'
                else:
                    trend = 'sideways'
            else:
                trend = 'insufficient_data'
            
            return {
                'trend': trend,
                'data_points': len(recent_history),
                'latest_return': returns[-1] if returns else 0,
                'avg_return': np.mean(returns) if returns else 0,
                'trend_slope': trend_slope if len(returns) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return {'trend': 'error', 'data_points': 0}
    
    def _calculate_risk_metrics(self, position_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        try:
            market_values = [pos['market_value'] for pos in position_details]
            total_value = sum(market_values)
            
            if total_value == 0:
                return {}
            
            # Concentration risk
            weights = [mv / total_value for mv in market_values]
            max_weight = max(weights)
            concentration_risk = 'high' if max_weight > 0.4 else 'moderate' if max_weight > 0.25 else 'low'
            
            # Diversification score (Herfindahl index)
            herfindahl_index = sum(w ** 2 for w in weights)
            diversification_score = (1 - herfindahl_index) * 100
            
            return {
                'concentration_risk': concentration_risk,
                'max_position_weight': max_weight * 100,
                'diversification_score': diversification_score,
                'effective_positions': 1 / herfindahl_index if herfindahl_index > 0 else len(position_details)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _get_empty_portfolio_metrics(self) -> Dict[str, Any]:
        """Get empty portfolio metrics structure."""
        return {
            'total_cost_basis': 0,
            'total_market_value': 0,
            'total_unrealized_pnl': 0,
            'total_return_pct': 0,
            'positions': [],
            'position_count': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _trim_history(self, max_entries: int = 1000):
        """Trim performance history to prevent memory bloat."""
        if len(self.performance_history) > max_entries:
            self.performance_history = self.performance_history[-max_entries:]


# Global portfolio tracker instance
portfolio_tracker = PortfolioTracker()

# Convenience functions
def calculate_portfolio_performance(positions: Dict[str, Any], current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Calculate portfolio performance metrics."""
    return portfolio_tracker.calculate_portfolio_performance(positions, current_prices)

def track_portfolio_positions(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Track portfolio positions from trades."""
    return portfolio_tracker.track_portfolio_positions(trades)

def calculate_performance_metrics(returns_data: List[float], benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
    """Calculate advanced performance metrics."""
    return portfolio_tracker.calculate_performance_metrics(returns_data, benchmark_returns)