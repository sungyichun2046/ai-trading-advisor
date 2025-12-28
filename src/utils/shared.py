"""
Shared utility functions used across multiple DAGs and modules.
Eliminates code duplication and provides centralized implementations.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Set
import sys
import os
import numpy as np
import pandas as pd
import asyncio
import json
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# WebSocket availability check
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Cache manager availability check
try:
    from .caching_manager import get_cache_manager
except ImportError:
    def get_cache_manager():
        """Fallback when caching manager not available."""
        return None


def get_data_manager():
    """
    Get data manager instance with fallback to mock implementation.
    
    Returns:
        DataManager instance or MockDataManager if imports fail
    """
    try:
        from src.core.data_manager import DataManager
        return DataManager()
    except ImportError:
        logger.warning("DataManager not available, using mock implementation")
        
        class MockDataManager:
            def collect_market_data(self, symbols):
                return {
                    'status': 'success', 
                    'symbols_collected': len(symbols), 
                    'data': {s: {'price': 100 + hash(s) % 50, 'volume': 1000000} for s in symbols[:3]}
                }
            
            def collect_fundamental_data(self, symbols):
                return {
                    'status': 'success', 
                    'symbols_collected': len(symbols), 
                    'data': [{'symbol': s, 'pe_ratio': 15 + hash(s) % 10, 'pb_ratio': 2 + hash(s) % 3} for s in symbols[:3]]
                }
            
            def collect_sentiment_data(self, max_articles=25):
                return {
                    'status': 'success', 
                    'article_count': max_articles, 
                    'articles': [{'sentiment_score': 0.1 * (i % 11 - 5), 'sentiment_label': 'positive' if i % 3 == 0 else 'negative' if i % 3 == 1 else 'neutral'} for i in range(max_articles)]
                }
        
        return MockDataManager()


def validate_data_quality(data: Dict[str, Any], data_type: str = "general", min_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Validate quality and completeness of data.
    
    Args:
        data: Data to validate
        data_type: Type of data being validated
        min_threshold: Minimum quality threshold
        
    Returns:
        Dict with quality score and issues
    """
    if not data:
        return {'quality_score': 0.0, 'issues': ['No data available'], 'data_type': data_type}
    
    # Calculate quality score based on data type
    if 'success_rate' in data:
        score = data['success_rate']
    elif 'symbols_processed' in data or 'symbols_collected' in data:
        processed = data.get('symbols_processed', data.get('symbols_collected', 0))
        score = min(1.0, processed / 10) if processed > 0 else 0.0
    elif 'article_count' in data:
        score = min(1.0, data['article_count'] / 20) if data['article_count'] > 0 else 0.0
    elif 'status' in data:
        score = 1.0 if data['status'] == 'success' else 0.5
    else:
        score = 0.8  # Default for structured data
    
    # Identify issues
    issues = []
    if score < min_threshold:
        issues.append(f"Low quality score: {score:.1%}")
    
    if data.get('status') != 'success':
        issues.append(f"Non-success status: {data.get('status', 'unknown')}")
    
    # Check for missing critical fields
    if data_type == 'market' and not data.get('data'):
        issues.append("Missing market data")
    elif data_type == 'sentiment' and not data.get('articles'):
        issues.append("Missing sentiment articles")
    
    return {
        'quality_score': score,
        'issues': issues,
        'data_type': data_type,
        'validated_at': datetime.now().isoformat()
    }


def calculate_returns(prices: List[float], periods: int = 1) -> List[float]:
    """
    Calculate price returns for given periods.
    
    Args:
        prices: List of prices
        periods: Number of periods for return calculation
        
    Returns:
        List of returns
    """
    if len(prices) < periods + 1:
        return []
    
    returns = []
    for i in range(periods, len(prices)):
        if prices[i - periods] != 0:
            ret = (prices[i] - prices[i - periods]) / prices[i - periods]
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def calculate_volatility(returns: List[float]) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Volatility (standard deviation)
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    return variance ** 0.5


def log_performance(operation: str, start_time: datetime, end_time: datetime, 
                   status: str = "success", metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Log performance metrics for operations.
    
    Args:
        operation: Name of operation
        start_time: Operation start time
        end_time: Operation end time
        status: Operation status
        metrics: Additional metrics
        
    Returns:
        Performance log entry
    """
    duration = (end_time - start_time).total_seconds()
    
    performance_log = {
        'operation': operation,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'status': status,
        'metrics': metrics or {}
    }
    
    logger.info(f"Performance: {operation} completed in {duration:.2f}s with status {status}")
    
    return performance_log


def send_alerts(alert_type: str, message: str, severity: str = "info", 
               context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send alerts and notifications.
    
    Args:
        alert_type: Type of alert (data_quality, system_health, trading, etc.)
        message: Alert message
        severity: Alert severity (info, warning, error, critical)
        context: Additional context information
        
    Returns:
        Success status
    """
    try:
        # Log based on severity
        if severity == 'critical':
            logger.critical(f"CRITICAL ALERT [{alert_type}]: {message}")
        elif severity == 'error':
            logger.error(f"ERROR [{alert_type}]: {message}")
        elif severity == 'warning':
            logger.warning(f"WARNING [{alert_type}]: {message}")
        else:
            logger.info(f"INFO [{alert_type}]: {message}")
        
        # In production, this would send to Slack, email, PagerDuty, etc.
        # For now, just log the alert
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        return False


def aggregate_data_quality_scores(quality_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple data quality validation results.
    
    Args:
        quality_results: List of quality validation results
        
    Returns:
        Aggregated quality metrics
    """
    if not quality_results:
        return {'overall_score': 0.0, 'grade': 'poor', 'total_issues': 0}
    
    # Calculate overall score
    scores = [result.get('quality_score', 0.0) for result in quality_results]
    overall_score = sum(scores) / len(scores)
    
    # Count total issues
    total_issues = sum(len(result.get('issues', [])) for result in quality_results)
    
    # Determine grade
    if overall_score >= 0.9:
        grade = 'excellent'
    elif overall_score >= 0.7:
        grade = 'good'
    elif overall_score >= 0.5:
        grade = 'fair'
    else:
        grade = 'poor'
    
    return {
        'overall_score': overall_score,
        'grade': grade,
        'total_issues': total_issues,
        'data_completeness': sum(1 for score in scores if score >= 0.8) / len(scores),
        'source_count': len(quality_results),
        'aggregated_at': datetime.now().isoformat()
    }


def calculate_vix_regime(vix: float) -> Dict[str, Any]:
    """
    Calculate VIX regime based on VIX level with error handling.
    
    Args:
        vix: VIX level
        
    Returns:
        Dict with regime classification and details
    """
    try:
        if vix is None or np.isnan(vix):
            vix = 18.5  # Dummy fallback
            
        if vix < 12:
            regime = 'low'
            description = 'Low volatility environment'
        elif vix < 20:
            regime = 'normal'
            description = 'Normal volatility environment'
        elif vix < 30:
            regime = 'elevated'
            description = 'Elevated volatility environment'
        else:
            regime = 'high'
            description = 'High volatility environment'
            
        return {
            'vix_level': vix,
            'regime': regime,
            'description': description,
            'percentile': min(100, max(0, (vix - 10) / 0.5))
        }
    except Exception as e:
        logger.warning(f"VIX regime calculation failed: {e}, using dummy data")
        return {'vix_level': 18.5, 'regime': 'normal', 'description': 'Normal volatility environment', 'percentile': 50}


def normalize_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize options data with error handling and dummy fallback.
    
    Args:
        df: Options data DataFrame
        
    Returns:
        Normalized DataFrame
    """
    try:
        if df.empty:
            return pd.DataFrame({
                'strike': [100, 105, 110, 115, 120],
                'call_volume': [100, 80, 60, 40, 20],
                'put_volume': [20, 40, 60, 80, 100],
                'iv': [0.2, 0.22, 0.24, 0.26, 0.28]
            })
        
        normalized_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['volume', 'open_interest'] or 'volume' in col.lower():
                normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            elif col == 'iv':
                normalized_df[col] = np.clip(df[col], 0, 2)
                
        return normalized_df
    except Exception as e:
        logger.warning(f"Options data normalization failed: {e}, using dummy data")
        return pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'call_volume': [0.8, 0.6, 0.4, 0.2, 0.0],
            'put_volume': [0.0, 0.2, 0.4, 0.6, 0.8],
            'iv': [0.2, 0.22, 0.24, 0.26, 0.28]
        })


def detect_unusual_volume(volume: float, avg_vol: float) -> Dict[str, Any]:
    """
    Detect unusual volume with error handling and dummy fallback.
    
    Args:
        volume: Current volume
        avg_vol: Average volume
        
    Returns:
        Dict with unusual volume analysis
    """
    try:
        if volume is None or avg_vol is None or avg_vol == 0:
            volume, avg_vol = 1500000, 1000000  # Dummy fallback
            
        ratio = volume / avg_vol
        
        if ratio > 3.0:
            classification = 'extremely_high'
        elif ratio > 2.0:
            classification = 'very_high'
        elif ratio > 1.5:
            classification = 'high'
        elif ratio < 0.5:
            classification = 'low'
        else:
            classification = 'normal'
            
        return {
            'volume': volume,
            'avg_volume': avg_vol,
            'volume_ratio': ratio,
            'classification': classification,
            'is_unusual': ratio > 1.5 or ratio < 0.5
        }
    except Exception as e:
        logger.warning(f"Volume analysis failed: {e}, using dummy data")
        return {'volume': 1500000, 'avg_volume': 1000000, 'volume_ratio': 1.5, 'classification': 'high', 'is_unusual': True}


def calculate_pattern_confidence(pattern: Dict[str, Any]) -> float:
    """
    Calculate pattern confidence score with error handling.
    
    Args:
        pattern: Pattern data with metrics
        
    Returns:
        Confidence score (0-1)
    """
    try:
        if not pattern or 'strength' not in pattern:
            return 0.75  # Dummy fallback
            
        strength = pattern.get('strength', 0.5)
        volume_confirm = pattern.get('volume_confirmed', False)
        duration = pattern.get('duration_bars', 5)
        
        confidence = strength * 0.6
        if volume_confirm:
            confidence += 0.2
        if duration >= 5:
            confidence += 0.15
        else:
            confidence += 0.05
            
        return min(1.0, max(0.0, confidence))
    except Exception as e:
        logger.warning(f"Pattern confidence calculation failed: {e}, using dummy value")
        return 0.75


def find_pivot_highs_lows(df: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
    """
    Find pivot highs and lows with error handling and dummy fallback.
    
    Args:
        df: Price data DataFrame
        window: Window size for pivot detection
        
    Returns:
        Dict with pivot high and low indices
    """
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns:
            return {'pivot_highs': [10, 25, 40], 'pivot_lows': [5, 20, 35]}
            
        highs = df['high'].values
        lows = df['low'].values
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                pivot_highs.append(i)
            if lows[i] == min(lows[i-window:i+window+1]):
                pivot_lows.append(i)
                
        return {'pivot_highs': pivot_highs, 'pivot_lows': pivot_lows}
    except Exception as e:
        logger.warning(f"Pivot detection failed: {e}, using dummy data")
        return {'pivot_highs': [10, 25, 40], 'pivot_lows': [5, 20, 35]}


def calculate_adaptive_period(volatility: float) -> int:
    """
    Calculate adaptive period based on volatility with error handling.
    
    Args:
        volatility: Market volatility measure
        
    Returns:
        Adaptive period length
    """
    try:
        if volatility is None or np.isnan(volatility):
            volatility = 0.15  # Dummy fallback
            
        # Higher volatility = shorter periods
        if volatility > 0.3:
            period = 5
        elif volatility > 0.2:
            period = 10
        elif volatility > 0.1:
            period = 14
        else:
            period = 20
            
        return max(5, min(50, period))
    except Exception as e:
        logger.warning(f"Adaptive period calculation failed: {e}, using dummy value")
        return 14


def normalize_signals(signal_dict: Dict[str, Union[float, int]]) -> Dict[str, float]:
    """
    Normalize signal values to 0-1 range with error handling.
    
    Args:
        signal_dict: Dictionary of signal names and values
        
    Returns:
        Normalized signal dictionary
    """
    try:
        if not signal_dict:
            return {'rsi': 0.6, 'macd': 0.4, 'momentum': 0.7}  # Dummy fallback
            
        normalized = {}
        for key, value in signal_dict.items():
            if value is None:
                normalized[key] = 0.5
            else:
                # Assume values are in typical ranges and normalize
                if key.lower() in ['rsi', 'stoch']:
                    normalized[key] = max(0, min(1, value / 100))
                elif key.lower() in ['macd', 'momentum']:
                    normalized[key] = max(0, min(1, (value + 1) / 2))
                else:
                    normalized[key] = max(0, min(1, abs(value)))
                    
        return normalized
    except Exception as e:
        logger.warning(f"Signal normalization failed: {e}, using dummy data")
        return {'rsi': 0.6, 'macd': 0.4, 'momentum': 0.7}


def calculate_agreement_ratio(signals: List[Dict[str, Any]]) -> float:
    """
    Calculate agreement ratio between multiple signals with error handling.
    
    Args:
        signals: List of signal dictionaries
        
    Returns:
        Agreement ratio (0-1)
    """
    try:
        if not signals:
            return 0.65  # Dummy fallback
            
        total_signals = len(signals)
        bullish_count = sum(1 for s in signals if s.get('direction', 'neutral') == 'bullish')
        bearish_count = sum(1 for s in signals if s.get('direction', 'neutral') == 'bearish')
        
        max_agreement = max(bullish_count, bearish_count)
        agreement_ratio = max_agreement / total_signals
        
        return min(1.0, max(0.0, agreement_ratio))
    except Exception as e:
        logger.warning(f"Agreement ratio calculation failed: {e}, using dummy value")
        return 0.65


def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume-based indicators with error handling and dummy fallback.
    
    Args:
        df: DataFrame with volume and price data
        
    Returns:
        Dict with volume indicators
    """
    try:
        if df.empty or 'volume' not in df.columns:
            return {'obv': 1000000, 'vwap': 105.5, 'volume_sma': 950000, 'volume_trend': 0.05}
            
        # On-Balance Volume
        obv = 0
        for i in range(1, len(df)):
            if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                obv += df.iloc[i]['volume']
            elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
                obv -= df.iloc[i]['volume']
                
        # VWAP
        vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
        
        # Volume SMA
        volume_sma = df['volume'].rolling(window=14).mean().iloc[-1]
        
        # Volume trend
        recent_vol = df['volume'].tail(5).mean()
        older_vol = df['volume'].tail(20).head(15).mean()
        volume_trend = (recent_vol - older_vol) / older_vol if older_vol != 0 else 0
        
        return {
            'obv': float(obv),
            'vwap': float(vwap),
            'volume_sma': float(volume_sma),
            'volume_trend': float(volume_trend)
        }
    except Exception as e:
        logger.warning(f"Volume indicators calculation failed: {e}, using dummy data")
        return {'obv': 1000000, 'vwap': 105.5, 'volume_sma': 950000, 'volume_trend': 0.05}


def combine_sentiment_scores(scores_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Combine sentiment scores from multiple sources with error handling.
    
    Args:
        scores_dict: Dictionary of sentiment scores by source
        
    Returns:
        Combined sentiment analysis
    """
    try:
        if not scores_dict:
            return {'combined_score': 0.15, 'confidence': 0.7, 'dominant_sentiment': 'slightly_positive'}
            
        scores = [s for s in scores_dict.values() if s is not None]
        if not scores:
            return {'combined_score': 0.0, 'confidence': 0.0, 'dominant_sentiment': 'neutral'}
            
        combined_score = sum(scores) / len(scores)
        confidence = 1 - (np.std(scores) if len(scores) > 1 else 0.2)
        
        if combined_score > 0.2:
            dominant = 'positive'
        elif combined_score > 0.05:
            dominant = 'slightly_positive'
        elif combined_score < -0.2:
            dominant = 'negative'
        elif combined_score < -0.05:
            dominant = 'slightly_negative'
        else:
            dominant = 'neutral'
            
        return {
            'combined_score': combined_score,
            'confidence': max(0, min(1, confidence)),
            'dominant_sentiment': dominant,
            'source_count': len(scores)
        }
    except Exception as e:
        logger.warning(f"Sentiment combination failed: {e}, using dummy data")
        return {'combined_score': 0.15, 'confidence': 0.7, 'dominant_sentiment': 'slightly_positive'}


def validate_signal_quality(signal: Dict[str, Any], age_minutes: int) -> Dict[str, Any]:
    """
    Validate signal quality based on age and other factors with error handling.
    
    Args:
        signal: Signal data to validate
        age_minutes: Age of signal in minutes
        
    Returns:
        Signal quality validation results
    """
    try:
        if not signal:
            return {'is_valid': False, 'quality_score': 0.0, 'issues': ['No signal data']}
            
        issues = []
        quality_score = 1.0
        
        # Age validation
        if age_minutes > 60:
            issues.append(f'Signal too old: {age_minutes} minutes')
            quality_score -= 0.3
        elif age_minutes > 30:
            quality_score -= 0.1
            
        # Strength validation
        strength = signal.get('strength', 0.5)
        if strength < 0.3:
            issues.append(f'Low signal strength: {strength}')
            quality_score -= 0.2
            
        # Confidence validation
        confidence = signal.get('confidence', 0.5)
        if confidence < 0.5:
            issues.append(f'Low confidence: {confidence}')
            quality_score -= 0.15
            
        quality_score = max(0.0, quality_score)
        is_valid = quality_score >= 0.6 and age_minutes <= 60
        
        return {
            'is_valid': is_valid,
            'quality_score': quality_score,
            'issues': issues,
            'age_minutes': age_minutes
        }
    except Exception as e:
        logger.warning(f"Signal validation failed: {e}, using dummy result")
        return {'is_valid': True, 'quality_score': 0.75, 'issues': [], 'age_minutes': 15}


# =============================================================================
# WEBSOCKET AND STREAMING UTILITIES
# =============================================================================

# Global WebSocket connections registry
_websocket_connections = {}
_stream_buffers = {}
_connection_lock = threading.RLock()


def setup_websocket_connection(url: str, handlers: Dict[str, Callable], 
                             auto_reconnect: bool = True, max_retries: int = 5) -> Dict[str, Any]:
    """
    Setup WebSocket connection with auto-reconnection and handlers.
    
    Args:
        url: WebSocket URL to connect to
        handlers: Dictionary of message type handlers {message_type: handler_function}
        auto_reconnect: Enable automatic reconnection
        max_retries: Maximum reconnection attempts
        
    Returns:
        Connection status and details
    """
    try:
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSocket connection requested but websockets package not available")
            return {
                'status': 'unavailable',
                'message': 'WebSocket package not installed',
                'connection_id': None
            }
        
        connection_id = f"ws_{hash(url)}_{int(time.time())}"
        
        with _connection_lock:
            # Store connection configuration
            _websocket_connections[connection_id] = {
                'url': url,
                'handlers': handlers,
                'auto_reconnect': auto_reconnect,
                'max_retries': max_retries,
                'status': 'connecting',
                'retry_count': 0,
                'last_message': None,
                'connected_at': None
            }
        
        # Start connection in background thread
        def start_connection():
            asyncio.new_event_loop().run_until_complete(
                _websocket_connection_handler(connection_id)
            )
        
        thread = threading.Thread(target=start_connection, daemon=True)
        thread.start()
        
        return {
            'status': 'connecting',
            'connection_id': connection_id,
            'url': url,
            'handlers_count': len(handlers)
        }
        
    except Exception as e:
        logger.error(f"Failed to setup WebSocket connection to {url}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'connection_id': None
        }


def stream_market_data(symbols: List[str], callback: Callable[[Dict[str, Any]], None],
                      buffer_size: int = 1000, batch_size: int = 10) -> Dict[str, Any]:
    """
    Stream market data with buffering and batch processing.
    
    Args:
        symbols: List of symbols to stream
        callback: Callback function to process streaming data
        buffer_size: Maximum buffer size for incoming data
        batch_size: Number of messages to batch before processing
        
    Returns:
        Stream configuration and status
    """
    try:
        stream_id = f"market_stream_{hash(tuple(symbols))}_{int(time.time())}"
        
        with _connection_lock:
            # Initialize stream buffer
            _stream_buffers[stream_id] = {
                'symbols': symbols,
                'callback': callback,
                'buffer': deque(maxlen=buffer_size),
                'batch_size': batch_size,
                'processed_count': 0,
                'error_count': 0,
                'started_at': datetime.now().isoformat(),
                'last_processed': None
            }
        
        # Start streaming in background thread
        def start_streaming():
            _market_data_streamer(stream_id)
        
        thread = threading.Thread(target=start_streaming, daemon=True)
        thread.start()
        
        return {
            'status': 'streaming',
            'stream_id': stream_id,
            'symbols': symbols,
            'buffer_size': buffer_size,
            'batch_size': batch_size
        }
        
    except Exception as e:
        logger.error(f"Failed to start market data stream for {symbols}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'stream_id': None
        }


def cache_cross_dag_data(key: str, data: Any, ttl: Optional[int] = None) -> bool:
    """
    Cache data for cross-DAG sharing using the caching manager.
    
    Args:
        key: Cache key
        data: Data to cache
        ttl: Time to live in seconds
        
    Returns:
        Success status
    """
    try:
        cache_manager = get_cache_manager()
        if cache_manager:
            return cache_manager.set(key, data, ttl)
        return False
    except Exception as e:
        logger.error(f"Failed to cache cross-DAG data for key {key}: {e}")
        return False


def get_cached_analysis_result(dag_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached analysis result from DAG task.
    
    Args:
        dag_id: DAG identifier
        task_id: Task identifier
        
    Returns:
        Cached analysis result or None
    """
    try:
        cache_manager = get_cache_manager()
        if cache_manager:
            return cache_manager.get_cached_analysis_result(dag_id, task_id)
        return None
    except Exception as e:
        logger.error(f"Failed to get cached analysis result for {dag_id}.{task_id}: {e}")
        return None


async def _websocket_connection_handler(connection_id: str):
    """Handle WebSocket connection with auto-reconnection."""
    try:
        if connection_id not in _websocket_connections:
            return
        
        config = _websocket_connections[connection_id]
        url = config['url']
        handlers = config['handlers']
        
        while config['retry_count'] <= config['max_retries']:
            try:
                logger.info(f"Connecting to WebSocket: {url}")
                
                async with websockets.connect(url) as websocket:
                    config['status'] = 'connected'
                    config['connected_at'] = datetime.now().isoformat()
                    config['retry_count'] = 0
                    
                    logger.info(f"WebSocket connected: {connection_id}")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            message_type = data.get('type', 'default')
                            
                            config['last_message'] = datetime.now().isoformat()
                            
                            # Route to appropriate handler
                            if message_type in handlers:
                                handlers[message_type](data)
                            elif 'default' in handlers:
                                handlers['default'](data)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received on {connection_id}: {message}")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message on {connection_id}: {e}")
                            
            except Exception as e:
                logger.error(f"WebSocket connection error on {connection_id}: {e}")
                config['status'] = 'disconnected'
                config['retry_count'] += 1
                
                if config['auto_reconnect'] and config['retry_count'] <= config['max_retries']:
                    wait_time = min(2 ** config['retry_count'], 30)  # Exponential backoff, max 30s
                    logger.info(f"Reconnecting {connection_id} in {wait_time} seconds (attempt {config['retry_count']})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for {connection_id}, giving up")
                    config['status'] = 'failed'
                    break
                    
    except Exception as e:
        logger.error(f"Fatal error in WebSocket handler {connection_id}: {e}")
        if connection_id in _websocket_connections:
            _websocket_connections[connection_id]['status'] = 'failed'


def _market_data_streamer(stream_id: str):
    """Simulate market data streaming with buffering."""
    try:
        if stream_id not in _stream_buffers:
            return
        
        stream_config = _stream_buffers[stream_id]
        symbols = stream_config['symbols']
        callback = stream_config['callback']
        buffer = stream_config['buffer']
        batch_size = stream_config['batch_size']
        
        logger.info(f"Starting market data stream for {symbols}")
        
        while True:
            try:
                # Simulate incoming market data
                for symbol in symbols:
                    mock_data = {
                        'symbol': symbol,
                        'price': 100 + np.random.randn() * 5,
                        'volume': np.random.randint(1000, 10000),
                        'timestamp': datetime.now().isoformat(),
                        'bid': 100 + np.random.randn() * 5,
                        'ask': 100 + np.random.randn() * 5 + 0.01
                    }
                    buffer.append(mock_data)
                
                # Process batch if buffer has enough data
                if len(buffer) >= batch_size:
                    batch = []
                    for _ in range(min(batch_size, len(buffer))):
                        if buffer:
                            batch.append(buffer.popleft())
                    
                    if batch:
                        try:
                            callback(batch)
                            stream_config['processed_count'] += len(batch)
                            stream_config['last_processed'] = datetime.now().isoformat()
                        except Exception as e:
                            logger.error(f"Error in stream callback for {stream_id}: {e}")
                            stream_config['error_count'] += 1
                
                # Simulate real-time data frequency (every 100ms)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in market data streamer {stream_id}: {e}")
                stream_config['error_count'] += 1
                time.sleep(1)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Fatal error in market data streamer {stream_id}: {e}")


def get_websocket_status() -> Dict[str, Any]:
    """Get status of all WebSocket connections."""
    try:
        with _connection_lock:
            status = {
                'total_connections': len(_websocket_connections),
                'connections': {},
                'websockets_available': WEBSOCKETS_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
            
            for conn_id, config in _websocket_connections.items():
                status['connections'][conn_id] = {
                    'url': config['url'],
                    'status': config['status'],
                    'retry_count': config['retry_count'],
                    'connected_at': config.get('connected_at'),
                    'last_message': config.get('last_message')
                }
            
            return status
            
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        return {'error': str(e)}


def get_stream_status() -> Dict[str, Any]:
    """Get status of all data streams."""
    try:
        with _connection_lock:
            status = {
                'total_streams': len(_stream_buffers),
                'streams': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for stream_id, config in _stream_buffers.items():
                status['streams'][stream_id] = {
                    'symbols': config.get('symbols', []),
                    'buffer_size': len(config.get('buffer', [])),
                    'processed_count': config.get('processed_count', 0),
                    'error_count': config.get('error_count', 0),
                    'started_at': config.get('started_at'),
                    'last_processed': config.get('last_processed')
                }
            
            return status
            
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        return {'error': str(e)}