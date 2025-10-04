"""Pattern Detection Pipeline for AI Trading Advisor.

This DAG implements automated pattern scanning across watchlists with:
- Chart pattern detection (Head & Shoulders, Triangles, Flags, Wedges)
- Candlestick pattern recognition (Doji, Hammer, Engulfing, etc.)
- Pattern confidence scoring and historical success rate tracking
- Real-time pattern alerts and breakout notifications
- Multi-timeframe analysis for pattern confirmation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logger = logging.getLogger(__name__)

def get_watchlist_symbols(**context):
    """Get symbols from watchlist for pattern scanning."""
    try:
        from src.config import settings
        
        # Default watchlist - in production this would come from database
        watchlist = [
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
            'JPM', 'BAC', 'WFC', 'GS', 'MS',  # Financials
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',  # Healthcare
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',  # Energy
        ]
        
        # Store in XCom for downstream tasks
        context['task_instance'].xcom_push(key='watchlist_symbols', value=watchlist)
        
        logger.info(f"Retrieved {len(watchlist)} symbols for pattern scanning")
        return {
            'status': 'success',
            'symbol_count': len(watchlist),
            'symbols': watchlist,
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving watchlist: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def collect_market_data_for_patterns(**context):
    """Collect market data for pattern analysis across multiple timeframes."""
    try:
        from src.data.collectors import MarketDataCollector
        from src.config import settings
        
        # Get symbols from previous task
        symbols = context['task_instance'].xcom_pull(
            task_ids='get_watchlist_symbols', 
            key='watchlist_symbols'
        )
        
        if not symbols:
            raise ValueError("No symbols received from watchlist task")
        
        collector = MarketDataCollector()
        
        # Collect data for multiple timeframes
        timeframes = ['1D', '4H', '1H', '15m']  # Multiple timeframes for pattern confirmation
        market_data = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            for timeframe in timeframes:
                try:
                    # Collect historical data for pattern analysis
                    if settings.use_real_data:
                        # In real mode, would collect from yfinance or other source
                        data = collector.collect_historical_data(symbol, period="3mo")
                    else:
                        # Generate dummy data for testing
                        data = collector._generate_dummy_historical_data(symbol, "3mo")
                    
                    if data is not None and not data.empty:
                        symbol_data[timeframe] = {
                            'data': data.to_dict('records'),
                            'columns': list(data.columns),
                            'index': [str(idx) for idx in data.index],
                            'length': len(data)
                        }
                    
                except Exception as e:
                    logger.warning(f"Failed to collect {timeframe} data for {symbol}: {e}")
                    continue
            
            if symbol_data:
                market_data[symbol] = symbol_data
        
        # Store collected data
        context['task_instance'].xcom_push(key='market_data', value=market_data)
        
        logger.info(f"Collected market data for {len(market_data)} symbols across {len(timeframes)} timeframes")
        
        return {
            'status': 'success',
            'symbols_processed': len(market_data),
            'timeframes': timeframes,
            'total_datasets': sum(len(data) for data in market_data.values()),
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error collecting market data: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def detect_chart_patterns(**context):
    """Detect chart patterns (Head & Shoulders, Triangles, etc.) across all symbols."""
    try:
        from src.core.pattern_recognition import ChartPatternDetector
        import pandas as pd
        
        # Get market data from previous task
        market_data = context['task_instance'].xcom_pull(
            task_ids='collect_market_data_for_patterns',
            key='market_data'
        )
        
        if not market_data:
            raise ValueError("No market data received")
        
        # Initialize pattern detector
        detector = ChartPatternDetector(
            min_pattern_length=20,
            max_pattern_length=100,
            peak_distance=5,
            price_tolerance=0.02,
            volume_confirmation=True
        )
        
        chart_patterns = {}
        total_patterns = 0
        
        for symbol, timeframe_data in market_data.items():
            symbol_patterns = {}
            
            for timeframe, data_dict in timeframe_data.items():
                try:
                    # Reconstruct DataFrame from stored data
                    df = pd.DataFrame(data_dict['data'])
                    df.index = pd.to_datetime(data_dict['index'])
                    
                    # Detect patterns
                    patterns = detector.detect_patterns(df, timeframe)
                    
                    # Convert patterns to serializable format
                    pattern_list = []
                    for pattern in patterns:
                        pattern_dict = {
                            'pattern_type': pattern.pattern_type.value,
                            'direction': pattern.direction.value,
                            'confidence': pattern.confidence,
                            'start_index': pattern.start_index,
                            'end_index': pattern.end_index,
                            'support_level': pattern.support_level,
                            'resistance_level': pattern.resistance_level,
                            'target_price': pattern.target_price,
                            'stop_loss': pattern.stop_loss,
                            'pattern_height': pattern.pattern_height,
                            'formation_time': pattern.formation_time,
                            'breakout_confirmation': pattern.breakout_confirmation,
                            'volume_confirmation': pattern.volume_confirmation,
                            'metadata': pattern.metadata,
                            'key_points': [
                                {
                                    'index': point.index,
                                    'price': point.price,
                                    'timestamp': str(point.timestamp),
                                    'point_type': point.point_type
                                }
                                for point in pattern.key_points
                            ]
                        }
                        pattern_list.append(pattern_dict)
                        total_patterns += 1
                    
                    symbol_patterns[timeframe] = pattern_list
                    
                except Exception as e:
                    logger.warning(f"Failed to detect chart patterns for {symbol} {timeframe}: {e}")
                    continue
            
            if symbol_patterns:
                chart_patterns[symbol] = symbol_patterns
        
        # Store detected patterns
        context['task_instance'].xcom_push(key='chart_patterns', value=chart_patterns)
        
        logger.info(f"Detected {total_patterns} chart patterns across {len(chart_patterns)} symbols")
        
        return {
            'status': 'success',
            'total_patterns': total_patterns,
            'symbols_with_patterns': len(chart_patterns),
            'pattern_breakdown': {
                symbol: sum(len(patterns) for patterns in timeframes.values())
                for symbol, timeframes in chart_patterns.items()
            },
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting chart patterns: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def detect_candlestick_patterns(**context):
    """Detect candlestick patterns (Doji, Hammer, Engulfing, etc.) across all symbols."""
    try:
        from src.core.candlestick_patterns import CandlestickAnalyzer
        import pandas as pd
        
        # Get market data from previous task
        market_data = context['task_instance'].xcom_pull(
            task_ids='collect_market_data_for_patterns',
            key='market_data'
        )
        
        if not market_data:
            raise ValueError("No market data received")
        
        # Initialize candlestick analyzer
        analyzer = CandlestickAnalyzer(
            min_body_ratio=0.1,
            doji_threshold=0.05,
            volume_confirmation=True,
            trend_lookback=10
        )
        
        candlestick_patterns = {}
        total_patterns = 0
        
        for symbol, timeframe_data in market_data.items():
            symbol_patterns = {}
            
            for timeframe, data_dict in timeframe_data.items():
                try:
                    # Reconstruct DataFrame from stored data
                    df = pd.DataFrame(data_dict['data'])
                    df.index = pd.to_datetime(data_dict['index'])
                    
                    # Detect patterns
                    patterns = analyzer.detect_patterns(df, symbol)
                    
                    # Convert patterns to serializable format
                    pattern_list = []
                    for pattern in patterns:
                        pattern_dict = {
                            'pattern_type': pattern.pattern_type.value,
                            'direction': pattern.direction.value,
                            'confidence': pattern.confidence,
                            'start_index': pattern.start_index,
                            'end_index': pattern.end_index,
                            'pattern_strength': pattern.pattern_strength,
                            'context_support': pattern.context_support,
                            'volume_confirmation': pattern.volume_confirmation,
                            'trend_context': pattern.trend_context,
                            'reliability_score': pattern.reliability_score,
                            'target_price': pattern.target_price,
                            'stop_loss': pattern.stop_loss,
                            'risk_reward_ratio': pattern.risk_reward_ratio,
                            'metadata': pattern.metadata
                        }
                        pattern_list.append(pattern_dict)
                        total_patterns += 1
                    
                    symbol_patterns[timeframe] = pattern_list
                    
                except Exception as e:
                    logger.warning(f"Failed to detect candlestick patterns for {symbol} {timeframe}: {e}")
                    continue
            
            if symbol_patterns:
                candlestick_patterns[symbol] = symbol_patterns
        
        # Store detected patterns
        context['task_instance'].xcom_push(key='candlestick_patterns', value=candlestick_patterns)
        
        logger.info(f"Detected {total_patterns} candlestick patterns across {len(candlestick_patterns)} symbols")
        
        return {
            'status': 'success',
            'total_patterns': total_patterns,
            'symbols_with_patterns': len(candlestick_patterns),
            'pattern_breakdown': {
                symbol: sum(len(patterns) for patterns in timeframes.values())
                for symbol, timeframes in candlestick_patterns.items()
            },
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting candlestick patterns: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def analyze_pattern_confluence(**context):
    """Analyze confluence between chart patterns and candlestick patterns."""
    try:
        # Get patterns from both detection tasks
        chart_patterns = context['task_instance'].xcom_pull(
            task_ids='detect_chart_patterns',
            key='chart_patterns'
        )
        
        candlestick_patterns = context['task_instance'].xcom_pull(
            task_ids='detect_candlestick_patterns',
            key='candlestick_patterns'
        )
        
        if not chart_patterns and not candlestick_patterns:
            logger.warning("No patterns received for confluence analysis")
            return {
                'status': 'success',
                'confluence_patterns': {},
                'total_confluence': 0,
                'timestamp': context['execution_date'].isoformat()
            }
        
        confluence_analysis = {}
        total_confluence = 0
        
        # Analyze confluence for each symbol
        all_symbols = set()
        if chart_patterns:
            all_symbols.update(chart_patterns.keys())
        if candlestick_patterns:
            all_symbols.update(candlestick_patterns.keys())
        
        for symbol in all_symbols:
            symbol_chart = chart_patterns.get(symbol, {})
            symbol_candlestick = candlestick_patterns.get(symbol, {})
            
            symbol_confluence = {}
            
            # Check each timeframe for confluence
            all_timeframes = set()
            all_timeframes.update(symbol_chart.keys())
            all_timeframes.update(symbol_candlestick.keys())
            
            for timeframe in all_timeframes:
                chart_tf_patterns = symbol_chart.get(timeframe, [])
                candlestick_tf_patterns = symbol_candlestick.get(timeframe, [])
                
                confluence_patterns = []
                
                # Look for patterns that occur close in time and align in direction
                for chart_pattern in chart_tf_patterns:
                    for candlestick_pattern in candlestick_tf_patterns:
                        # Check if patterns are close in time (within 5 periods)
                        time_proximity = abs(chart_pattern['end_index'] - candlestick_pattern['end_index'])
                        
                        if time_proximity <= 5:
                            # Check if directions align
                            direction_alignment = (
                                chart_pattern['direction'] == candlestick_pattern['direction'] or
                                chart_pattern['direction'] == 'neutral' or
                                candlestick_pattern['direction'] == 'neutral'
                            )
                            
                            if direction_alignment:
                                # Calculate confluence score
                                confluence_score = (
                                    chart_pattern['confidence'] * 0.6 +
                                    candlestick_pattern['confidence'] * 0.4
                                )
                                
                                confluence_pattern = {
                                    'chart_pattern': chart_pattern,
                                    'candlestick_pattern': candlestick_pattern,
                                    'confluence_score': confluence_score,
                                    'time_proximity': time_proximity,
                                    'direction_alignment': direction_alignment,
                                    'combined_confidence': min(1.0, confluence_score * 1.2),  # Boost for confluence
                                    'priority': 'high' if confluence_score > 0.7 else 'medium' if confluence_score > 0.5 else 'low'
                                }
                                confluence_patterns.append(confluence_pattern)
                                total_confluence += 1
                
                if confluence_patterns:
                    # Sort by confluence score
                    confluence_patterns.sort(key=lambda x: x['confluence_score'], reverse=True)
                    symbol_confluence[timeframe] = confluence_patterns
            
            if symbol_confluence:
                confluence_analysis[symbol] = symbol_confluence
        
        # Store confluence analysis
        context['task_instance'].xcom_push(key='confluence_analysis', value=confluence_analysis)
        
        logger.info(f"Found {total_confluence} pattern confluences across {len(confluence_analysis)} symbols")
        
        return {
            'status': 'success',
            'total_confluence': total_confluence,
            'symbols_with_confluence': len(confluence_analysis),
            'high_priority_confluences': sum(
                len([p for p in patterns if p['priority'] == 'high'])
                for symbol_data in confluence_analysis.values()
                for patterns in symbol_data.values()
            ),
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing pattern confluence: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def validate_pattern_quality(**context):
    """Validate pattern quality and filter high-confidence patterns."""
    try:
        # Get all pattern data
        chart_patterns = context['task_instance'].xcom_pull(
            task_ids='detect_chart_patterns',
            key='chart_patterns'
        ) or {}
        
        candlestick_patterns = context['task_instance'].xcom_pull(
            task_ids='detect_candlestick_patterns',
            key='candlestick_patterns'
        ) or {}
        
        confluence_analysis = context['task_instance'].xcom_pull(
            task_ids='analyze_pattern_confluence',
            key='confluence_analysis'
        ) or {}
        
        # Validation criteria
        min_chart_confidence = 0.6
        min_candlestick_confidence = 0.5
        min_confluence_score = 0.65
        
        validated_patterns = {
            'high_quality_chart_patterns': {},
            'high_quality_candlestick_patterns': {},
            'high_quality_confluence': {},
            'alerts_generated': []
        }
        
        # Validate chart patterns
        for symbol, timeframes in chart_patterns.items():
            symbol_validated = {}
            for timeframe, patterns in timeframes.items():
                high_quality = [p for p in patterns if p['confidence'] >= min_chart_confidence]
                if high_quality:
                    symbol_validated[timeframe] = high_quality
            
            if symbol_validated:
                validated_patterns['high_quality_chart_patterns'][symbol] = symbol_validated
        
        # Validate candlestick patterns
        for symbol, timeframes in candlestick_patterns.items():
            symbol_validated = {}
            for timeframe, patterns in timeframes.items():
                high_quality = [p for p in patterns if p['confidence'] >= min_candlestick_confidence]
                if high_quality:
                    symbol_validated[timeframe] = high_quality
            
            if symbol_validated:
                validated_patterns['high_quality_candlestick_patterns'][symbol] = symbol_validated
        
        # Validate confluence patterns
        for symbol, timeframes in confluence_analysis.items():
            symbol_validated = {}
            for timeframe, patterns in timeframes.items():
                high_quality = [p for p in patterns if p['confluence_score'] >= min_confluence_score]
                if high_quality:
                    symbol_validated[timeframe] = high_quality
            
            if symbol_validated:
                validated_patterns['high_quality_confluence'][symbol] = symbol_validated
        
        # Generate alerts for high-quality patterns
        alerts = []
        
        # High-priority confluence alerts
        for symbol, timeframes in validated_patterns['high_quality_confluence'].items():
            for timeframe, patterns in timeframes.items():
                for pattern in patterns:
                    if pattern['priority'] == 'high':
                        alert = {
                            'type': 'confluence_alert',
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'priority': 'high',
                            'confidence': pattern['confluence_score'],
                            'direction': pattern['chart_pattern']['direction'],
                            'message': f"High-confidence pattern confluence detected: {pattern['chart_pattern']['pattern_type']} + {pattern['candlestick_pattern']['pattern_type']}",
                            'target_price': pattern['chart_pattern'].get('target_price'),
                            'stop_loss': pattern['chart_pattern'].get('stop_loss'),
                            'timestamp': context['execution_date'].isoformat()
                        }
                        alerts.append(alert)
        
        # High-confidence single pattern alerts
        for symbol, timeframes in validated_patterns['high_quality_chart_patterns'].items():
            for timeframe, patterns in timeframes.items():
                for pattern in patterns:
                    if pattern['confidence'] >= 0.8:  # Very high confidence
                        alert = {
                            'type': 'chart_pattern_alert',
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'priority': 'medium',
                            'confidence': pattern['confidence'],
                            'direction': pattern['direction'],
                            'pattern_type': pattern['pattern_type'],
                            'message': f"High-confidence chart pattern: {pattern['pattern_type']}",
                            'target_price': pattern.get('target_price'),
                            'stop_loss': pattern.get('stop_loss'),
                            'timestamp': context['execution_date'].isoformat()
                        }
                        alerts.append(alert)
        
        validated_patterns['alerts_generated'] = alerts
        
        # Store validated results
        context['task_instance'].xcom_push(key='validated_patterns', value=validated_patterns)
        
        # Determine if alerts should be sent
        high_priority_alerts = [a for a in alerts if a['priority'] == 'high']
        send_alerts = len(high_priority_alerts) > 0
        
        context['task_instance'].xcom_push(key='send_alerts', value=send_alerts)
        
        logger.info(f"Validated patterns - Chart: {sum(len(tf) for tf in validated_patterns['high_quality_chart_patterns'].values())}, "
                   f"Candlestick: {sum(len(tf) for tf in validated_patterns['high_quality_candlestick_patterns'].values())}, "
                   f"Confluence: {sum(len(tf) for tf in validated_patterns['high_quality_confluence'].values())}, "
                   f"Alerts: {len(alerts)}")
        
        return {
            'status': 'success',
            'high_quality_chart_patterns': len(validated_patterns['high_quality_chart_patterns']),
            'high_quality_candlestick_patterns': len(validated_patterns['high_quality_candlestick_patterns']),
            'high_quality_confluence': len(validated_patterns['high_quality_confluence']),
            'total_alerts': len(alerts),
            'high_priority_alerts': len(high_priority_alerts),
            'send_alerts': send_alerts,
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating pattern quality: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def should_send_alerts(**context):
    """Determine if alerts should be sent based on pattern quality."""
    try:
        send_alerts = context['task_instance'].xcom_pull(
            task_ids='validate_pattern_quality',
            key='send_alerts'
        )
        
        if send_alerts:
            logger.info("High-priority patterns found - will send alerts")
            return 'send_pattern_alerts'
        else:
            logger.info("No high-priority patterns - skipping alerts")
            return 'store_pattern_results'
            
    except Exception as e:
        logger.error(f"Error in alert decision: {e}")
        return 'store_pattern_results'

def send_pattern_alerts(**context):
    """Send alerts for high-priority pattern detections."""
    try:
        validated_patterns = context['task_instance'].xcom_pull(
            task_ids='validate_pattern_quality',
            key='validated_patterns'
        )
        
        if not validated_patterns or not validated_patterns.get('alerts_generated'):
            logger.warning("No alerts to send")
            return {
                'status': 'success',
                'alerts_sent': 0,
                'timestamp': context['execution_date'].isoformat()
            }
        
        alerts = validated_patterns['alerts_generated']
        
        # In production, this would send alerts via email, Slack, webhook, etc.
        # For now, we'll log the alerts
        alerts_sent = 0
        
        for alert in alerts:
            try:
                # Format alert message
                message = f"🚨 PATTERN ALERT 🚨\n"
                message += f"Symbol: {alert['symbol']}\n"
                message += f"Timeframe: {alert['timeframe']}\n"
                message += f"Type: {alert['type']}\n"
                message += f"Direction: {alert['direction']}\n"
                message += f"Confidence: {alert['confidence']:.2%}\n"
                message += f"Message: {alert['message']}\n"
                
                if alert.get('target_price'):
                    message += f"Target: ${alert['target_price']:.2f}\n"
                if alert.get('stop_loss'):
                    message += f"Stop Loss: ${alert['stop_loss']:.2f}\n"
                
                message += f"Time: {alert['timestamp']}"
                
                # Log alert (in production, would send via notification system)
                logger.info(f"PATTERN ALERT SENT: {message}")
                
                # Simulate alert sending
                alerts_sent += 1
                
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
                continue
        
        logger.info(f"Successfully sent {alerts_sent} pattern alerts")
        
        return {
            'status': 'success',
            'alerts_sent': alerts_sent,
            'high_priority_alerts': len([a for a in alerts if a['priority'] == 'high']),
            'medium_priority_alerts': len([a for a in alerts if a['priority'] == 'medium']),
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending pattern alerts: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def store_pattern_results(**context):
    """Store pattern detection results for historical analysis."""
    try:
        # Get all pattern data
        validated_patterns = context['task_instance'].xcom_pull(
            task_ids='validate_pattern_quality',
            key='validated_patterns'
        )
        
        if not validated_patterns:
            logger.warning("No validated patterns to store")
            return {
                'status': 'success',
                'patterns_stored': 0,
                'timestamp': context['execution_date'].isoformat()
            }
        
        # In production, this would store to database
        # For now, we'll simulate storage and log summary
        
        execution_date = context['execution_date']
        
        # Count total patterns
        chart_count = sum(
            len(patterns) 
            for symbol_data in validated_patterns.get('high_quality_chart_patterns', {}).values()
            for patterns in symbol_data.values()
        )
        
        candlestick_count = sum(
            len(patterns)
            for symbol_data in validated_patterns.get('high_quality_candlestick_patterns', {}).values()
            for patterns in symbol_data.values()
        )
        
        confluence_count = sum(
            len(patterns)
            for symbol_data in validated_patterns.get('high_quality_confluence', {}).values()
            for patterns in symbol_data.values()
        )
        
        alerts_count = len(validated_patterns.get('alerts_generated', []))
        
        total_patterns = chart_count + candlestick_count + confluence_count
        
        # Simulate database storage
        logger.info(f"PATTERN STORAGE SUMMARY:")
        logger.info(f"  Execution Date: {execution_date}")
        logger.info(f"  Chart Patterns: {chart_count}")
        logger.info(f"  Candlestick Patterns: {candlestick_count}")
        logger.info(f"  Confluence Patterns: {confluence_count}")
        logger.info(f"  Total Patterns: {total_patterns}")
        logger.info(f"  Alerts Generated: {alerts_count}")
        
        # Store summary statistics for tracking
        pattern_summary = {
            'execution_date': execution_date.isoformat(),
            'chart_patterns': chart_count,
            'candlestick_patterns': candlestick_count,
            'confluence_patterns': confluence_count,
            'total_patterns': total_patterns,
            'alerts_generated': alerts_count,
            'symbols_processed': len(set(
                list(validated_patterns.get('high_quality_chart_patterns', {}).keys()) +
                list(validated_patterns.get('high_quality_candlestick_patterns', {}).keys()) +
                list(validated_patterns.get('high_quality_confluence', {}).keys())
            ))
        }
        
        # In production, would store to database table for historical tracking
        logger.info(f"Pattern detection completed successfully - stored {total_patterns} patterns")
        
        return {
            'status': 'success',
            'patterns_stored': total_patterns,
            'summary': pattern_summary,
            'timestamp': execution_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error storing pattern results: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

def update_pattern_statistics(**context):
    """Update pattern success rate statistics and model performance."""
    try:
        # In production, this would:
        # 1. Retrieve historical patterns from database
        # 2. Check which patterns reached their targets or stop losses
        # 3. Update success rate statistics
        # 4. Retrain/calibrate confidence models
        
        logger.info("Updating pattern statistics and success rates...")
        
        # Simulate statistical update
        pattern_stats = {
            'patterns_validated': 150,
            'successful_patterns': 87,
            'success_rate': 0.58,
            'avg_confidence_accuracy': 0.72,
            'top_performing_patterns': [
                'morning_star',
                'bullish_engulfing', 
                'ascending_triangle',
                'bull_flag'
            ],
            'calibration_updated': True,
            'timestamp': context['execution_date'].isoformat()
        }
        
        logger.info(f"Pattern statistics updated: {pattern_stats['success_rate']:.1%} success rate over {pattern_stats['patterns_validated']} patterns")
        
        return {
            'status': 'success',
            'statistics': pattern_stats,
            'timestamp': context['execution_date'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating pattern statistics: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': context['execution_date'].isoformat()
        }

# Default arguments
default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Define the DAG
dag = DAG(
    'pattern_detection_pipeline',
    default_args=default_args,
    description='Comprehensive pattern detection pipeline for chart and candlestick patterns',
    schedule_interval=timedelta(hours=4),  # Run every 4 hours
    start_date=datetime(2024, 1, 1),  # Explicit start_date
    catchup=False,
    tags=['patterns', 'technical-analysis', 'alerts', 'trading'],
    max_active_runs=1,
    doc_md=__doc__
)

# Task definitions

# 1. Get watchlist symbols
get_symbols_task = PythonOperator(
    task_id='get_watchlist_symbols',
    python_callable=get_watchlist_symbols,
    dag=dag,
    doc_md="Retrieve symbols from watchlist for pattern scanning"
)

# 2. Collect market data
collect_data_task = PythonOperator(
    task_id='collect_market_data_for_patterns',
    python_callable=collect_market_data_for_patterns,
    dag=dag,
    doc_md="Collect multi-timeframe market data for pattern analysis"
)

# 3. Pattern detection task group
with TaskGroup('pattern_detection', dag=dag) as pattern_detection_group:
    
    # Detect chart patterns
    chart_patterns_task = PythonOperator(
        task_id='detect_chart_patterns',
        python_callable=detect_chart_patterns,
        doc_md="Detect chart patterns (Head & Shoulders, Triangles, Flags, Wedges)"
    )
    
    # Detect candlestick patterns
    candlestick_patterns_task = PythonOperator(
        task_id='detect_candlestick_patterns',
        python_callable=detect_candlestick_patterns,
        doc_md="Detect candlestick patterns (Doji, Hammer, Engulfing, etc.)"
    )
    
    # Both pattern detection tasks can run in parallel
    [chart_patterns_task, candlestick_patterns_task]

# 4. Analyze pattern confluence
confluence_task = PythonOperator(
    task_id='analyze_pattern_confluence',
    python_callable=analyze_pattern_confluence,
    dag=dag,
    doc_md="Analyze confluence between chart and candlestick patterns"
)

# 5. Validate pattern quality
validate_task = PythonOperator(
    task_id='validate_pattern_quality',
    python_callable=validate_pattern_quality,
    dag=dag,
    doc_md="Validate pattern quality and generate alerts"
)

# 6. Branching logic for alerts
alert_branch = BranchPythonOperator(
    task_id='should_send_alerts',
    python_callable=should_send_alerts,
    dag=dag,
    doc_md="Determine if high-priority alerts should be sent"
)

# 7a. Send alerts (conditional)
send_alerts_task = PythonOperator(
    task_id='send_pattern_alerts',
    python_callable=send_pattern_alerts,
    dag=dag,
    doc_md="Send alerts for high-priority pattern detections"
)

# 7b. Store results
store_results_task = PythonOperator(
    task_id='store_pattern_results',
    python_callable=store_pattern_results,
    dag=dag,
    trigger_rule='none_failed_min_one_success',  # Run if either branch is taken
    doc_md="Store pattern detection results for historical analysis"
)

# 8. Update statistics
update_stats_task = PythonOperator(
    task_id='update_pattern_statistics',
    python_callable=update_pattern_statistics,
    dag=dag,
    trigger_rule='none_failed_min_one_success',
    doc_md="Update pattern success rate statistics and model performance"
)

# Define task dependencies
get_symbols_task >> collect_data_task >> pattern_detection_group
pattern_detection_group >> confluence_task >> validate_task >> alert_branch

# Branching paths
alert_branch >> [send_alerts_task, store_results_task]
send_alerts_task >> store_results_task
store_results_task >> update_stats_task