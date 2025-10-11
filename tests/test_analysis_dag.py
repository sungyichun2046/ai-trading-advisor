"""Tests for Analysis DAG."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dags.analysis_dag import (
    analyze_technical_indicators, analyze_fundamentals, detect_patterns,
    analyze_sentiment, classify_market_regime, dag
)


class TestAnalysisDAG:
    """Test analysis DAG structure and configuration."""
    
    def test_dag_configuration(self):
        """Test DAG configuration parameters."""
        assert dag.dag_id == 'analysis'
        assert dag.schedule_interval.total_seconds() == 3600  # 1 hour
        assert dag.max_active_runs == 1
        assert 'analysis' in dag.tags
        assert 'technical' in dag.tags
        assert 'patterns' in dag.tags
        assert 'sentiment' in dag.tags
        assert 'regime' in dag.tags
    
    def test_dag_tasks_exist(self):
        """Test all required tasks exist in DAG."""
        task_ids = {task.task_id for task in dag.tasks}
        
        expected_tasks = {
            'analyze_technical_indicators',
            'analyze_fundamentals',
            'detect_patterns',
            'analyze_sentiment',
            'classify_market_regime'
        }
        
        assert expected_tasks.issubset(task_ids)
        assert len(dag.tasks) == 5


class TestAnalyzeTechnicalIndicators:
    """Test technical indicator analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    @patch('src.dags.analysis_dag.MarketDataCollector')
    @patch('src.dags.analysis_dag.TechnicalIndicators')
    @patch('src.dags.analysis_dag.MultiTimeframeAnalysis')
    def test_analyze_technical_indicators_success(self, mock_multi_tf, mock_tech_ind, mock_collector_class):
        """Test successful technical indicator analysis."""
        mock_collector = Mock()
        mock_collector.get_recent_data.return_value = {
            'AAPL': Mock(),
            'MSFT': Mock()
        }
        mock_collector_class.return_value = mock_collector
        
        mock_indicators = Mock()
        mock_indicators.calculate_all.return_value = {
            'rsi': {'value': 65.0, 'signal': 'bullish'},
            'macd': {'signal': 'bullish', 'histogram': 0.5}
        }
        mock_tech_ind.return_value = mock_indicators
        
        mock_timeframe = Mock()
        mock_timeframe.analyze_timeframes.return_value = {
            '1h': {'trend': 'bullish', 'strength': 0.8}
        }
        mock_multi_tf.return_value = mock_timeframe
        
        result = analyze_technical_indicators(**self.mock_context)
        
        assert result['symbols_analyzed'] == 2
        assert 'technical_summary' in result
        assert 'market_sentiment' in result['technical_summary']
        self.mock_context['task_instance'].xcom_push.assert_called_once()


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
        
        assert result['symbols_analyzed'] == 10
        assert 'fundamental_summary' in result
        assert 'market_valuation' in result['fundamental_summary']
        assert 'symbol_fundamentals' in result
        self.mock_context['task_instance'].xcom_push.assert_called_once()


class TestDetectPatterns:
    """Test pattern detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
    
    @patch('src.dags.analysis_dag.MarketDataCollector')
    @patch('src.dags.analysis_dag.ChartPatternDetector')
    def test_detect_patterns_success(self, mock_detector_class, mock_collector_class):
        """Test successful pattern detection."""
        mock_collector = Mock()
        mock_collector.get_recent_data.return_value = {
            'AAPL': Mock(),
            'MSFT': Mock()
        }
        mock_collector_class.return_value = mock_collector
        
        mock_detector = Mock()
        mock_detector.detect_patterns.return_value = {
            'patterns_found': ['double_bottom', 'ascending_triangle'],
            'confidence': [0.8, 0.6],
            'breakout_probability': 0.75
        }
        mock_detector_class.return_value = mock_detector
        
        result = detect_patterns(**self.mock_context)
        
        assert result['symbols_analyzed'] == 2
        assert 'pattern_summary' in result
        assert 'market_pattern_bias' in result['pattern_summary']
        assert 'symbol_patterns' in result
        self.mock_context['task_instance'].xcom_push.assert_called_once()


class TestAnalyzeSentiment:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        
        def mock_xcom_pull(task_ids, key):
            if task_ids == 'analyze_technical_indicators':
                return {'technical_summary': {'market_sentiment': 'bullish'}}
            elif task_ids == 'detect_patterns':
                return {'pattern_summary': {'market_pattern_bias': 'bullish'}}
            return None
        
        self.mock_context['task_instance'].xcom_pull.side_effect = mock_xcom_pull
    
    @patch('src.dags.analysis_dag.SentimentAnalyzer')
    def test_analyze_sentiment_success(self, mock_analyzer_class):
        """Test successful sentiment analysis."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_market_sentiment.return_value = {
            'overall_sentiment': 'positive',
            'sentiment_score': 0.15,
            'fear_greed_index': 65,
            'social_sentiment': 'bullish'
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        result = analyze_sentiment(**self.mock_context)
        
        assert 'sentiment_analysis' in result
        assert 'consensus_sentiment' in result['sentiment_analysis']
        assert 'sentiment_signals' in result
        assert 'market_psychology' in result
        self.mock_context['task_instance'].xcom_push.assert_called_once()


class TestClassifyMarketRegime:
    """Test market regime classification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = {'task_instance': Mock()}
        
        def mock_xcom_pull(task_ids, key):
            if task_ids == 'analyze_technical_indicators':
                return {
                    'technical_summary': {
                        'market_sentiment': 'bullish',
                        'consensus_level': 0.8
                    }
                }
            elif task_ids == 'analyze_fundamentals':
                return {
                    'fundamental_summary': {
                        'market_outlook': 'fairly_valued',
                        'market_valuation': 65
                    }
                }
            elif task_ids == 'detect_patterns':
                return {
                    'pattern_summary': {'market_pattern_bias': 'bullish'},
                    'trading_signals': {
                        'strong_buy_signals': 3,
                        'strong_sell_signals': 1
                    }
                }
            elif task_ids == 'analyze_sentiment':
                return {
                    'sentiment_analysis': {
                        'consensus_sentiment': 'bullish',
                        'consensus_strength': 0.75
                    }
                }
            return None
        
        self.mock_context['task_instance'].xcom_pull.side_effect = mock_xcom_pull
    
    @patch('src.dags.analysis_dag.MarketDataCollector')
    @patch('src.dags.analysis_dag.MarketRegimeClassifier')
    def test_classify_market_regime_success(self, mock_classifier_class, mock_collector_class):
        """Test successful market regime classification."""
        mock_collector = Mock()
        mock_collector.get_recent_data.return_value = {
            'SPY': Mock(),
            'QQQ': Mock(),
            'IWM': Mock()
        }
        mock_collector_class.return_value = mock_collector
        
        mock_classifier = Mock()
        mock_classifier.classify_regime.return_value = {
            'current_regime': 'trending_bull',
            'confidence': 0.85,
            'regime_duration': 25,
            'transition_probability': 0.15
        }
        mock_classifier_class.return_value = mock_classifier
        
        result = classify_market_regime(**self.mock_context)
        
        assert 'regime_classification' in result
        assert 'current_regime' in result['regime_classification']
        assert 'regime_factors' in result
        assert 'trading_implications' in result
        self.mock_context['task_instance'].xcom_push.assert_called_once()