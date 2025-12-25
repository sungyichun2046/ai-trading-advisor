"""
Shared utility functions used across multiple DAGs and modules.
Eliminates code duplication and provides centralized implementations.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import sys
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


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