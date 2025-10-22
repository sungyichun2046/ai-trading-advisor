"""
Shared utility functions used across multiple DAGs and modules.
Eliminates code duplication and provides centralized implementations.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os

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
        alert_entry = {
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
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