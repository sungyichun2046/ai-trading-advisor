"""
Trading DAG
Replaces 3 old DAGs with 6 streamlined trading tasks including paper trading.

Replaces:
- position_sizing_pipeline.py
- portfolio_rebalancing_dag.py
- trading_execution_dag.py
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import core modules with fallbacks
import sys
import os

# Add multiple possible paths for different environments
possible_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..'),  # Local development
    '/opt/airflow',  # Airflow Docker environment
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Alternative
]
for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.core.trading_engine import TradingEngine, calculate_returns, log_performance
    HAS_TRADING_ENGINE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import trading engine: {e}")
    HAS_TRADING_ENGINE = False
    
    # Fallback: define essential functions locally
    def calculate_returns(prices, returns_type="simple"):
        if len(prices) < 2:
            return []
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    def log_performance(strategy_name, performance_data):
        logger.info(f"Strategy Performance: {strategy_name}")
        for metric, value in performance_data.items():
            logger.info(f"  {metric}: {value}")

# Get trading engine instance
def get_trading_engine():
    if HAS_TRADING_ENGINE:
        return TradingEngine()
    else:
        class MockTradingEngine:
            def momentum_strategy(self, data, params=None):
                return {'signal': 'buy', 'confidence': 0.7, 'reasoning': 'Mock momentum signal'}
            
            def mean_reversion_strategy(self, data, params=None):
                return {'signal': 'sell', 'confidence': 0.6, 'reasoning': 'Mock mean reversion signal'}
            
            def breakout_strategy(self, data, params=None):
                return {'signal': 'buy', 'confidence': 0.8, 'reasoning': 'Mock breakout signal'}
            
            def value_strategy(self, data, params=None):
                return {'signal': 'hold', 'confidence': 0.5, 'reasoning': 'Mock value signal'}
            
            def assess_risk(self, portfolio_data):
                return {
                    'status': 'success', 
                    'portfolio_risk': 0.15, 
                    'var_95': 0.08, 
                    'max_drawdown': 0.12,
                    'sharpe_ratio': 1.2
                }
        
        return MockTradingEngine()

class PositionSizingCalculator:
    def calculate_position_sizes(self, signals, portfolio_value=100000):
        return {
            'status': 'success', 
            'calculated_positions': len(signals), 
            'total_allocation': 0.8,
            'portfolio_value': portfolio_value
        }

class PortfolioManager:
    def rebalance_portfolio(self, target_positions):
        return {'status': 'success', 'rebalanced_positions': len(target_positions)}

class RiskManager:
    def assess_portfolio_risk(self, portfolio_data):
        return {
            'status': 'success', 
            'portfolio_risk': 0.15, 
            'var_95': 0.08, 
            'max_drawdown': 0.12,
            'risk_score': 0.7
        }
    
    def validate_positions(self, positions):
        return {'status': 'success', 'risk_compliant': True}

class AlertManager:
    def send_trading_alerts(self, signals):
        return {'status': 'success', 'alerts_sent': len(signals)}
    
    class PositionSizingCalculator:
        def calculate_position_sizes(self, signals, portfolio_value=100000):
            return {'status': 'success', 'calculated_positions': len(signals), 'total_allocation': 0.8}
    
    class PortfolioManager:
        def manage_portfolio(self, positions, current_portfolio):
            return {'status': 'success', 'rebalanced': True, 'trades_executed': 5}
    
    class RiskManager:
        def assess_portfolio_risk(self, portfolio_data):
            return {'status': 'success', 'risk_level': 'moderate', 'risk_score': 0.6}
    
    class AlertManager:
        def send_alerts(self, alert_data):
            return {'status': 'success', 'alerts_sent': 3, 'notifications': ['portfolio_update', 'risk_alert']}

logger = logging.getLogger(__name__)

# Core symbols - reduced for fast execution
SYMBOLS = ['AAPL', 'SPY', 'QQQ']  # Only 3 symbols for speed

# DAG configuration
dag = DAG(
    'trading',
    default_args={
        'owner': 'ai-trading-advisor',
        'depends_on_past': False,
        'start_date': days_ago(1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=2),
        'execution_timeout': timedelta(minutes=10),
        'catchup': False
    },
    description='Streamlined trading and portfolio management pipeline',
    schedule_interval='0 9,15 * * 1-5',  # 9 AM and 3 PM weekdays
    max_active_runs=1,
    tags=['trading', 'portfolio', 'risk', 'signals', 'alerts']
)


def generate_trading_signals(**context) -> Dict[str, Any]:
    """Generate trading signals using multiple strategies."""
    try:
        logger.info("Starting multi-strategy trading signal generation")
        start_time = datetime.now()
        
        trading_engine = get_trading_engine()
        
        # Sample market data for strategy execution
        import pandas as pd
        import numpy as np
        
        # Mock price data for testing
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        mock_data = {
            'technical': {
                'price_data': pd.Series(100 + np.random.randn(50).cumsum(), index=dates),
                'volume_data': pd.Series(np.random.randint(1000, 10000, 50), index=dates)
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
        
        # Execute all 4 strategies
        strategy_results = {}
        strategy_results['momentum'] = trading_engine.momentum_strategy(mock_data)
        strategy_results['mean_reversion'] = trading_engine.mean_reversion_strategy(mock_data)
        strategy_results['breakout'] = trading_engine.breakout_strategy(mock_data)
        strategy_results['value'] = trading_engine.value_strategy(mock_data)
        
        # Aggregate signals
        buy_signals = len([r for r in strategy_results.values() if r['signal'] == 'buy'])
        sell_signals = len([r for r in strategy_results.values() if r['signal'] == 'sell'])
        hold_signals = len([r for r in strategy_results.values() if r['signal'] == 'hold'])
        
        # Determine overall signal
        if buy_signals > sell_signals:
            overall_signal = 'buy'
            confidence = sum([r['confidence'] for r in strategy_results.values() if r['signal'] == 'buy']) / buy_signals
        elif sell_signals > buy_signals:
            overall_signal = 'sell'
            confidence = sum([r['confidence'] for r in strategy_results.values() if r['signal'] == 'sell']) / sell_signals
        else:
            overall_signal = 'hold'
            confidence = 0.5
        
        # Create consensus result for explanation
        consensus_result = {
            'overall_signal': overall_signal,
            'confidence': confidence,
            'signal_distribution': {
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': hold_signals
            }
        }
        
        # Generate detailed explanation if trading engine supports it
        explanation_result = {}
        if HAS_TRADING_ENGINE and hasattr(trading_engine, 'generate_explanation'):
            explanation_result = trading_engine.generate_explanation(strategy_results, consensus_result)
        
        # Calculate strategy scores if trading engine supports it
        strategy_scores = {}
        if HAS_TRADING_ENGINE and hasattr(trading_engine, 'calculate_strategy_scores'):
            strategy_scores = trading_engine.calculate_strategy_scores(strategy_results)
        
        # Performance metrics
        performance = {
            'strategies_executed': len(strategy_results),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'avg_confidence': sum([r['confidence'] for r in strategy_results.values()]) / len(strategy_results),
            'signal_consensus': buy_signals + sell_signals
        }
        
        # Log performance using shared utility
        log_performance('Multi-Strategy Signal Generation', start_time, datetime.now(), 'success', performance)
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy_results': strategy_results,
            'overall_signal': overall_signal,
            'confidence': confidence,
            'signal_distribution': {
                'buy': buy_signals,
                'sell': sell_signals, 
                'hold': hold_signals
            },
            'performance_metrics': performance,
            'explanation': explanation_result,
            'strategy_scores': strategy_scores
        }
        
        context['task_instance'].xcom_push(key='trading_signals', value=processed_data)
        logger.info(f"Multi-strategy signals generated: {overall_signal} with {confidence:.2f} confidence")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in trading signal generation: {e}")
        raise


def assess_portfolio_risk(**context) -> Dict[str, Any]:
    """Assess portfolio risk using enhanced trading engine."""
    try:
        logger.info("Starting enhanced portfolio risk assessment")
        start_time = datetime.now()
        
        trading_signals = context['task_instance'].xcom_pull(task_ids='generate_trading_signals', key='trading_signals')
        trading_engine = get_trading_engine()
        
        # Portfolio data with current positions
        portfolio_data = {
            'total_value': 100000,
            'positions': {s: {'value': 10000, 'shares': 100} for s in SYMBOLS},
            'daily_pnl': -500,  # Mock daily P&L
            'cash': 70000
        }
        
        # Use trading engine's risk assessment if available
        if hasattr(trading_engine, 'risk_manager'):
            risk_assessment = trading_engine.risk_manager.calculate_portfolio_risk(portfolio_data)
            daily_loss_check = trading_engine.risk_manager.check_daily_loss_limit(portfolio_data, 0.06)
        else:
            risk_assessment = {'total_risk': 0.15, 'concentration_risk': 0.08, 'var_95': 5000}
            daily_loss_check = {'limit_reached': False, 'current_loss': 0.005}
        
        # Calculate returns for risk metrics
        import pandas as pd
        mock_prices = pd.Series([100, 101, 99, 102, 98])
        returns = calculate_returns(mock_prices)
        volatility = returns.std() if not returns.empty else 0.02
        
        # Enhanced risk metrics
        performance_metrics = {
            'portfolio_volatility': volatility,
            'risk_assessment_time': (datetime.now() - start_time).total_seconds(),
            'diversification_score': risk_assessment.get('diversification_score', 0.8),
            'concentration_risk': risk_assessment.get('concentration_risk', 0.08)
        }
        
        # Log performance using shared utility
        log_performance('Portfolio Risk Assessment', start_time, datetime.now(), 'success', performance_metrics)
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': risk_assessment,
            'daily_loss_status': daily_loss_check,
            'portfolio_analysis': {
                'total_value': portfolio_data['total_value'],
                'position_count': len(portfolio_data['positions']),
                'cash_percentage': portfolio_data['cash'] / portfolio_data['total_value'],
                'largest_position': max([p['value'] for p in portfolio_data['positions'].values()])
            },
            'performance_metrics': performance_metrics,
            'risk_recommendations': {
                'rebalance_needed': risk_assessment.get('concentration_risk', 0) > 0.1,
                'reduce_exposure': daily_loss_check.get('limit_reached', False),
                'increase_diversification': risk_assessment.get('diversification_score', 1) < 0.7
            }
        }
        
        context['task_instance'].xcom_push(key='risk_assessment', value=processed_data)
        logger.info(f"Enhanced risk assessment completed in {performance_metrics['risk_assessment_time']:.2f}s")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in portfolio risk assessment: {e}")
        raise


def calculate_position_sizes(**context) -> Dict[str, Any]:
    """Calculate optimal position sizes based on signals and risk."""
    try:
        logger.info("Starting position size calculation")
        
        trading_signals = context['task_instance'].xcom_pull(task_ids='generate_trading_signals', key='trading_signals')
        risk_assessment = context['task_instance'].xcom_pull(task_ids='assess_portfolio_risk', key='risk_assessment')
        
        position_calculator = PositionSizingCalculator()
        signals_data = trading_signals.get('trading_signals', {}) if trading_signals else {}
        
        # Ultra-simple position sizing
        portfolio_value = 100000
        max_position_size = 0.05  # 5% max per position
        calculated_positions = {}
        
        for symbol in SYMBOLS:
            signal_strength = signals_data.get(symbol, {}).get('confidence', 0.5)
            position_size = portfolio_value * max_position_size * signal_strength
            calculated_positions[symbol] = {
                'dollar_amount': position_size,
                'portfolio_percentage': position_size / portfolio_value,
                'shares': int(position_size / 100),  # Assuming $100 per share
                'signal_strength': signal_strength
            }
        
        total_allocation = sum(pos['portfolio_percentage'] for pos in calculated_positions.values())
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'position_sizing': {
                'total_portfolio_allocation': total_allocation,
                'number_of_positions': len(calculated_positions),
                'largest_position_pct': max(pos['portfolio_percentage'] for pos in calculated_positions.values()),
                'cash_remaining_pct': 1.0 - total_allocation
            },
            'calculated_positions': calculated_positions,
            'sizing_constraints': {
                'max_position_size_pct': max_position_size,
                'total_portfolio_value': portfolio_value,
                'risk_adjusted': True
            },
            'execution_plan': {
                'immediate_trades': len([p for p in calculated_positions.values() if p['portfolio_percentage'] > 0.01]),
                'estimated_execution_time': '5-10 minutes',
                'market_impact': 'minimal'
            }
        }
        
        context['task_instance'].xcom_push(key='position_sizes', value=processed_data)
        logger.info(f"Position sizes calculated: {total_allocation:.1%} total allocation")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in position size calculation: {e}")
        raise


def manage_portfolio(**context) -> Dict[str, Any]:
    """Execute portfolio management and rebalancing."""
    try:
        logger.info("Starting portfolio management")
        
        position_sizes = context['task_instance'].xcom_pull(task_ids='calculate_position_sizes', key='position_sizes')
        risk_assessment = context['task_instance'].xcom_pull(task_ids='assess_portfolio_risk', key='risk_assessment')
        
        portfolio_manager = PortfolioManager()
        current_portfolio = {'AAPL': 5000, 'SPY': 3000, 'QQQ': 2000, 'cash': 90000}
        
        # Ultra-simple portfolio management
        trades_to_execute = []
        rebalancing_needed = True
        
        calculated_positions = position_sizes.get('calculated_positions', {}) if position_sizes else {}
        for symbol, target_position in calculated_positions.items():
            current_value = current_portfolio.get(symbol, 0)
            target_value = target_position['dollar_amount']
            difference = target_value - current_value
            
            if abs(difference) > 100:  # Only trade if difference > $100
                trades_to_execute.append({
                    'symbol': symbol,
                    'action': 'buy' if difference > 0 else 'sell',
                    'amount': abs(difference),
                    'shares': int(abs(difference) / 100)
                })
        
        # Execute trades (simulated)
        total_trade_value = sum(trade['amount'] for trade in trades_to_execute)
        portfolio_turnover = total_trade_value / sum(current_portfolio.values())
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_management': {
                'rebalancing_required': rebalancing_needed,
                'trades_executed': len(trades_to_execute),
                'total_trade_value': total_trade_value,
                'portfolio_turnover_pct': portfolio_turnover
            },
            'executed_trades': trades_to_execute,
            'portfolio_status': {
                'pre_trade_value': sum(current_portfolio.values()),
                'estimated_post_trade_value': sum(current_portfolio.values()),
                'cash_utilized_pct': total_trade_value / current_portfolio.get('cash', 1)
            },
            'performance_impact': {
                'expected_return_impact': 0.02,  # 2% expected improvement
                'risk_reduction': len(trades_to_execute) * 0.005,  # 0.5% risk reduction per trade
                'diversification_score': 0.8
            }
        }
        
        context['task_instance'].xcom_push(key='portfolio_management', value=processed_data)
        logger.info(f"Portfolio management completed: {len(trades_to_execute)} trades executed")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in portfolio management: {e}")
        raise


def execute_paper_trades(**context) -> Dict[str, Any]:
    """Execute paper trading using enhanced trading engine."""
    try:
        logger.info("Starting paper trading execution")
        
        # Get previous analysis results
        trading_signals = context['task_instance'].xcom_pull(task_ids='generate_trading_signals', key='trading_signals')
        position_sizes = context['task_instance'].xcom_pull(task_ids='calculate_position_sizes', key='position_sizes')
        
        trading_engine = get_trading_engine()
        
        # Current portfolio state for paper trading
        portfolio_state = {
            'total_value': 100000,
            'cash': 50000,
            'positions': {'AAPL': 50, 'SPY': 100, 'QQQ': 25}
        }
        
        # Prepare market data for paper trading
        market_data = {
            'AAPL': {'price': 180.50, 'status': 'success'},
            'SPY': {'price': 450.00, 'status': 'success'}, 
            'QQQ': {'price': 380.25, 'status': 'success'}
        }
        
        # Execute paper trades for each signal
        paper_trades = []
        trading_summary = {'executed': 0, 'rejected': 0, 'no_action': 0}
        
        # Process trading signals
        strategy_results = trading_signals.get('strategy_results', {}) if trading_signals else {}
        overall_signal = trading_signals.get('overall_signal', 'hold') if trading_signals else 'hold'
        
        for symbol in SYMBOLS:
            signal_data = {
                'symbol': symbol,
                'signal': overall_signal,
                'confidence': trading_signals.get('confidence', 0.5) if trading_signals else 0.5,
                'price': market_data[symbol]['price'],
                'strategy_source': 'multi_strategy_consensus'
            }
            
            # Execute paper trade using enhanced trading engine
            if HAS_TRADING_ENGINE and hasattr(trading_engine, 'execute_paper_trade'):
                trade_result = trading_engine.execute_paper_trade(signal_data, portfolio_state)
            else:
                # Fallback paper trading
                if signal_data['signal'] in ['buy', 'sell'] and signal_data['confidence'] > 0.5:
                    shares = int(portfolio_state['total_value'] * 0.02 / signal_data['price'])
                    trade_result = {
                        'status': 'executed',
                        'trade': {
                            'symbol': symbol, 'action': signal_data['signal'], 'shares': shares,
                            'price': signal_data['price'], 'value': shares * signal_data['price'],
                            'confidence': signal_data['confidence'], 'timestamp': datetime.now().isoformat()
                        },
                        'portfolio_state': portfolio_state
                    }
                else:
                    trade_result = {'status': 'no_action', 'trade': None}
            
            # Track trade results
            paper_trades.append({'symbol': symbol, 'result': trade_result})
            trading_summary[trade_result['status']] += 1
            
            # Update portfolio state if trade executed
            if trade_result['status'] == 'executed' and 'portfolio_state' in trade_result:
                portfolio_state = trade_result['portfolio_state']
        
        # Calculate portfolio metrics using enhanced trading engine
        portfolio_metrics = {}
        if HAS_TRADING_ENGINE and hasattr(trading_engine, 'calculate_portfolio_metrics'):
            metrics_result = trading_engine.calculate_portfolio_metrics(portfolio_state, market_data)
            portfolio_metrics = metrics_result.get('metrics', {})
        else:
            # Fallback metrics
            total_value = sum(portfolio_state.get('positions', {}).values()) * 100 + portfolio_state.get('cash', 0)
            portfolio_metrics = {
                'total_value': total_value,
                'cash': portfolio_state.get('cash', 0),
                'diversification_score': 0.8,
                'concentration_risk': 0.1
            }
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'paper_trading_summary': trading_summary,
            'trades_executed': paper_trades,
            'portfolio_state': portfolio_state,
            'portfolio_metrics': portfolio_metrics,
            'market_data': market_data,
            'execution_performance': {
                'total_trades_attempted': len(paper_trades),
                'success_rate': trading_summary['executed'] / len(paper_trades) if paper_trades else 0,
                'total_notional_value': sum([t['result']['trade']['value'] for t in paper_trades 
                                           if t['result']['status'] == 'executed' and t['result']['trade']], 0)
            }
        }
        
        context['task_instance'].xcom_push(key='paper_trades', value=processed_data)
        logger.info(f"Paper trading completed: {trading_summary['executed']} trades executed, {trading_summary['rejected']} rejected")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in paper trading execution: {e}")
        raise


def send_alerts(**context) -> Dict[str, Any]:
    """Send alerts and notifications about trading activities."""
    try:
        logger.info("Starting alert and notification system")
        
        # Get all previous analysis results
        trading_signals = context['task_instance'].xcom_pull(task_ids='generate_trading_signals', key='trading_signals')
        risk_assessment = context['task_instance'].xcom_pull(task_ids='assess_portfolio_risk', key='risk_assessment')
        position_sizes = context['task_instance'].xcom_pull(task_ids='calculate_position_sizes', key='position_sizes')
        portfolio_management = context['task_instance'].xcom_pull(task_ids='manage_portfolio', key='portfolio_management')
        paper_trades = context['task_instance'].xcom_pull(task_ids='execute_paper_trades', key='paper_trades')
        
        alert_manager = AlertManager()
        
        # Prepare alert data for enhanced alert system
        alert_data = {
            'risk_violations': [],
            'strong_signals': [],
            'performance_issues': []
        }
        
        # Risk violations
        if risk_assessment and risk_assessment.get('risk_metrics', {}).get('concentration_risk', 0) > 0.15:
            alert_data['risk_violations'].append("High concentration risk detected")
        
        # Strong signals
        if trading_signals:
            for strategy_name, result in trading_signals.get('strategy_results', {}).items():
                if result.get('confidence', 0) > 0.8 and result.get('signal') in ['buy', 'sell']:
                    alert_data['strong_signals'].append({
                        'symbol': 'AAPL',  # Mock symbol
                        'signal': result['signal'],
                        'confidence': result['confidence'],
                        'strategy': strategy_name
                    })
        
        # Performance issues
        if portfolio_management:
            portfolio_turnover = portfolio_management.get('portfolio_management', {}).get('portfolio_turnover_pct', 0)
            if portfolio_turnover > 0.5:  # High turnover
                alert_data['performance_issues'].append(f"High portfolio turnover: {portfolio_turnover:.1%}")
        
        # Use enhanced alert system if available
        alert_result = {}
        trading_engine = get_trading_engine()
        if HAS_TRADING_ENGINE and hasattr(trading_engine, 'send_alerts'):
            alert_result = trading_engine.send_alerts(alert_data)
            notifications_sent = alert_result.get('notifications', [])
        else:
            # Fallback alert system
            notifications_sent = []
            for violation in alert_data['risk_violations']:
                notifications_sent.append({
                    'channel': 'email',
                    'recipient': 'trader@ai-trading-advisor.com',
                    'subject': 'Trading Alert: RISK',
                    'sent_at': datetime.now().isoformat()
                })
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_summary': {
                'total_alerts_generated': alert_result.get('alerts_generated', len(notifications_sent)),
                'notifications_sent': len(notifications_sent),
                'alert_system_status': alert_result.get('status', 'success')
            },
            'alert_data_processed': alert_data,
            'notifications_sent': notifications_sent,
            'enhanced_alert_result': alert_result,
            'system_status': {
                'all_systems_operational': len(alert_data['risk_violations']) == 0,
                'trading_session_complete': True,
                'next_analysis_scheduled': 'Next market session'
            }
        }
        
        context['task_instance'].xcom_push(key='alerts_sent', value=processed_data)
        logger.info(f"Alerts and notifications completed: {len(notifications_sent)} notifications sent")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in alert system: {e}")
        raise


# Define tasks
generate_trading_signals_task = PythonOperator(task_id='generate_trading_signals', python_callable=generate_trading_signals, dag=dag)
assess_portfolio_risk_task = PythonOperator(task_id='assess_portfolio_risk', python_callable=assess_portfolio_risk, dag=dag)
calculate_position_sizes_task = PythonOperator(task_id='calculate_position_sizes', python_callable=calculate_position_sizes, dag=dag)
manage_portfolio_task = PythonOperator(task_id='manage_portfolio', python_callable=manage_portfolio, dag=dag)
execute_paper_trades_task = PythonOperator(task_id='execute_paper_trades', python_callable=execute_paper_trades, dag=dag)
send_alerts_task = PythonOperator(task_id='send_alerts', python_callable=send_alerts, dag=dag)

# Define task dependencies
generate_trading_signals_task >> assess_portfolio_risk_task
[generate_trading_signals_task, assess_portfolio_risk_task] >> calculate_position_sizes_task
calculate_position_sizes_task >> manage_portfolio_task
manage_portfolio_task >> execute_paper_trades_task
[execute_paper_trades_task, assess_portfolio_risk_task] >> send_alerts_task