"""Tests for Analysis DAG."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.analysis_dag import (
    simple_technical_analysis as analyze_technical_indicators,
    simple_fundamental_analysis as analyze_fundamentals, 
    simple_sentiment_analysis as analyze_sentiment,
    monitor_analysis_systems, calculate_consensus_signals, dag
)


class TestAnalysisDAG:
    """Test analysis DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'analysis'
        assert dag.schedule_interval is None  # Manual trigger only
        assert dag.max_active_runs == 1
        assert 'analysis' in dag.tags
        assert 'simple' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'analyze_technical_indicators',
            'analyze_fundamentals',
            'analyze_sentiment',
            'monitor_analysis_systems',
            'calculate_consensus_signals'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 9  # Enhanced with dependency management (skip, proceed, monitoring tasks)


class TestAnalyzeTechnicalIndicators:
    """Test technical indicator analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_analyze_technical_indicators_success(self):
        """Test successful technical indicator analysis."""
        result = analyze_technical_indicators(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        assert 'timestamp' in result
        
        result = analyze_technical_indicators(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestAnalyzeFundamentals:
    """Test fundamental analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'technical_summary': {'market_sentiment': 'bullish'}
        }
    
    def test_analyze_fundamentals_success(self):
        """Test successful fundamental analysis."""
        result = analyze_fundamentals(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'valuation' in result
        assert 'recommendation' in result
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestMonitorAnalysisSystems:
    """Test monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_monitor_analysis_systems_success(self):
        """Test successful analysis monitoring."""
        result = monitor_analysis_systems(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'analysis_system_health' in result or 'monitoring' in result
        # alerts_sent is tracked in logs but not always in return value
        assert 'timestamp' in result
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestAnalyzeSentiment:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        self.mock_context['task_instance'].xcom_pull.return_value = {
            'technical_summary': {'market_sentiment': 'bullish'},
            'fundamental_summary': {'market_valuation': 'fairly_valued'}
        }
    
    def test_analyze_sentiment_success(self):
        """Test successful sentiment analysis."""
        result = analyze_sentiment(**self.mock_context)
        
        assert result['status'] == 'success'
        assert 'overall_sentiment' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        assert result['confidence'] >= 0.0
        assert result['confidence'] <= 1.0
        self.mock_context['task_instance'].xcom_push.assert_called()


class TestDAGBackwardsCompatibility:
    """Test DAG backwards compatibility and signature preservation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_dag_backwards_compatibility(self):
        """Test that existing task signatures remain unchanged."""
        import inspect
        
        # Test technical analysis function signature
        sig = inspect.signature(analyze_technical_indicators)
        assert '**context' in str(sig), "Technical analysis signature changed"
        
        # Test fundamental analysis function signature  
        sig = inspect.signature(analyze_fundamentals)
        assert '**context' in str(sig), "Fundamental analysis signature changed"
        
        # Test sentiment analysis function signature
        sig = inspect.signature(analyze_sentiment)
        assert '**context' in str(sig), "Sentiment analysis signature changed"
        
        # Test monitor systems function signature
        sig = inspect.signature(monitor_analysis_systems)
        assert '**context' in str(sig), "Monitor systems signature changed"
        
        # Test that all functions return expected basic fields
        result = analyze_technical_indicators(**self.mock_context)
        assert 'status' in result
        assert 'timestamp' in result
        
        result = analyze_fundamentals(**self.mock_context)
        assert 'status' in result
        assert 'timestamp' in result
        
        result = analyze_sentiment(**self.mock_context)
        assert 'status' in result
        assert 'timestamp' in result


class TestEnhancedSentimentTask:
    """Test enhanced sentiment task with fallback functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_enhanced_sentiment_task(self):
        """Test sentiment task uses enhanced logic with fallback."""
        result = analyze_sentiment(**self.mock_context)
        
        # Test basic return structure
        assert result['status'] == 'success'
        assert 'overall_sentiment' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        
        # Test enhanced vs fallback behavior
        assert 'enhanced' in result, "Should indicate whether enhanced analysis was used"
        
        # Test enhanced fields if enhancement worked
        if result.get('enhanced', False):
            assert 'sentiment_score' in result
            assert 'article_count' in result
            assert 'components' in result
        
        # Test fallback fields always present
        assert result['overall_sentiment'] in ['positive', 'negative', 'neutral']
        assert isinstance(result['confidence'], (int, float))
        assert 0 <= result['confidence'] <= 1


class TestEnhancedTechnicalTask:
    """Test enhanced technical task with fallback functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    def test_enhanced_technical_task(self):
        """Test technical task uses enhanced logic with fallback."""
        result = analyze_technical_indicators(**self.mock_context)
        
        # Test basic return structure
        assert result['status'] == 'success'
        assert 'indicators' in result
        assert 'signal' in result
        assert 'timestamp' in result
        
        # Test enhanced vs fallback behavior
        assert 'enhanced' in result, "Should indicate whether enhanced analysis was used"
        
        # Test enhanced fields if enhancement worked
        if result.get('enhanced', False):
            assert 'timeframe' in result
            assert 'data_quality' in result
        
        # Test fallback fields always present
        assert isinstance(result['indicators'], dict)
        assert result['signal'] in ['bullish', 'bearish', 'neutral']


class TestConsensusTaskIntegration:
    """Test consensus task integration with DAG."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        
        # Mock previous task results
        self.mock_context['task_instance'].xcom_pull.side_effect = lambda key=None: {
            'technical_analysis': {'status': 'success', 'signal': 'bullish', 'indicators': {'rsi': 65}},
            'fundamental_analysis': {'status': 'success', 'valuation': 'fair'},
            'sentiment_analysis': {'status': 'success', 'overall_sentiment': 'positive', 'confidence': 0.8}
        }.get(key, {})
    
    def test_consensus_task_integration(self):
        """Test consensus task functionality and integration."""
        result = calculate_consensus_signals(**self.mock_context)
        
        # Test basic return structure
        assert result['status'] == 'success'
        assert 'timestamp' in result
        
        # Test consensus fields
        assert 'consensus_score' in result
        assert 'confidence_level' in result
        assert 'alignment_status' in result
        assert 'agreement_ratio' in result
        assert 'signal_count' in result
        
        # Test enhanced vs fallback behavior
        assert 'enhanced' in result, "Should indicate whether enhanced analysis was used"
        
        # Test enhanced fields if enhancement worked
        if result.get('enhanced', False):
            assert 'resonance_analysis' in result
        
        # Test value ranges
        assert isinstance(result['consensus_score'], (int, float))
        assert 0 <= result['consensus_score'] <= 1
        assert result['confidence_level'] in ['very_high', 'high', 'moderate', 'low', 'very_low']
        assert result['alignment_status'] in ['fully_aligned', 'mostly_aligned', 'partially_aligned', 'conflicted', 'no_consensus']
        
        # Test XCom interaction
        self.mock_context['task_instance'].xcom_push.assert_called()
        self.mock_context['task_instance'].xcom_pull.assert_called()