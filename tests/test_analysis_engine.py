"""
Tests for Analysis Engine Module.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.analysis_engine import (
    AnalysisEngine, TechnicalAnalyzer, FundamentalAnalyzer, PatternAnalyzer,
    SentimentAnalyzer, TrendDirection, validate_data_format, calculate_composite_score
)


class TestTechnicalAnalyzer:
    """Test technical analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 30),
            'High': np.random.uniform(100, 110, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(95, 105, 30),
            'Volume': np.random.randint(100000, 1000000, 30)
        }, index=dates)
        
        # Ensure OHLC logic
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.iloc[i][['Open', 'High', 'Close']])
            low = min(self.sample_data.iloc[i][['Open', 'Low', 'Close']])
            self.sample_data.at[self.sample_data.index[i], 'High'] = high
            self.sample_data.at[self.sample_data.index[i], 'Low'] = low
    
    def test_analyzer_initialization(self):
        """Test TechnicalAnalyzer initialization."""
        assert self.analyzer.config == {}
    
    def test_calculate_indicators_complete(self):
        """Test complete indicator calculation."""
        result = self.analyzer.calculate_indicators(self.sample_data)
        
        assert isinstance(result, dict)
        assert 'rsi' in result
        assert 'macd' in result
        assert 'bollinger' in result
        assert 'trend' in result
        assert 'volume' in result
    
    def test_calculate_indicators_insufficient_data(self):
        """Test indicator calculation with insufficient data."""
        small_data = self.sample_data.head(5)
        result = self.analyzer.calculate_indicators(small_data)
        assert result == {}
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        result = self.analyzer.calculate_rsi(self.sample_data['Close'])
        
        assert 'current' in result
        assert 'signal' in result
        assert isinstance(result['current'], float)
        assert result['signal'] in ['overbought', 'oversold', 'neutral']
        assert 0 <= result['current'] <= 100
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        result = self.analyzer.calculate_macd(self.sample_data['Close'])
        
        assert 'crossover' in result
        assert result['crossover'] in ['bullish', 'bearish', 'neutral']
    
    def test_detect_trend(self):
        """Test trend detection."""
        result = self.analyzer.detect_trend(self.sample_data)
        
        assert 'direction' in result
        assert 'strength' in result
        assert isinstance(result['direction'], TrendDirection)
        assert 0 <= result['strength'] <= 100


class TestFundamentalAnalyzer:
    """Test fundamental analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FundamentalAnalyzer()
        self.sample_financial_data = {
            'pe_ratio': 22.5,
            'pb_ratio': 3.2,
            'profit_margins': 0.18,
            'current_ratio': 1.8,
            'latest_revenue': 1000000000
        }
        self.sample_market_data = {
            'market_cap': 2500000000000,
            'price': 180.50
        }
    
    def test_analyzer_initialization(self):
        """Test FundamentalAnalyzer initialization."""
        assert self.analyzer.config == {}
    
    def test_calculate_financial_ratios(self):
        """Test financial ratios calculation."""
        result = self.analyzer.calculate_financial_ratios(self.sample_financial_data)
        
        assert 'pe_signal' in result
        assert 'pe_ratio' in result
        assert 'profitability_signal' in result
        assert 'liquidity_signal' in result
        
        assert result['pe_signal'] in ['undervalued', 'overvalued', 'fair_value']
        assert result['profitability_signal'] in ['strong', 'weak']
    
    def test_calculate_valuation_metrics(self):
        """Test valuation metrics calculation."""
        result = self.analyzer.calculate_valuation_metrics(
            self.sample_financial_data, self.sample_market_data
        )
        
        assert 'price_to_sales' in result
        assert 'ps_assessment' in result
        assert result['ps_assessment'] in ['undervalued', 'overvalued', 'fair_value']


class TestPatternAnalyzer:
    """Test pattern analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102] * 10,
            'High': [101, 102, 103] * 10,
            'Low': [99, 100, 101] * 10,
            'Close': [100.5, 101.5, 102.5] * 10,
            'Volume': [100000] * 30
        }, index=dates)
    
    def test_detect_chart_patterns(self):
        """Test chart pattern detection."""
        result = self.analyzer.detect_chart_patterns(self.sample_data)
        
        assert 'patterns' in result
        assert 'count' in result
        assert isinstance(result['patterns'], list)
        assert isinstance(result['count'], int)
    
    def test_detect_candlestick_patterns(self):
        """Test candlestick pattern detection."""
        result = self.analyzer.detect_candlestick_patterns(self.sample_data)
        
        assert 'patterns' in result
        assert 'bullish_count' in result
        assert 'bearish_count' in result
        assert isinstance(result['patterns'], list)
    
    def test_detect_support_resistance(self):
        """Test support and resistance detection."""
        result = self.analyzer.detect_support_resistance(self.sample_data)
        
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        assert isinstance(result['support_levels'], list)
        assert isinstance(result['resistance_levels'], list)


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.sample_news_data = [
            {'title': 'Market Rally Continues', 'content': 'Strong gains across sectors.'},
            {'title': 'Economic Concerns Rise', 'content': 'Inflation fears weigh on sentiment.'},
            {'title': 'Neutral Market Update', 'content': 'Markets traded sideways today.'}
        ]
    
    def test_analyzer_initialization(self):
        """Test SentimentAnalyzer initialization."""
        assert self.analyzer.config == {}
    
    def test_analyze_news_sentiment(self):
        """Test news sentiment analysis."""
        result = self.analyzer.analyze_news_sentiment(self.sample_news_data)
        
        assert 'overall_sentiment' in result
        assert 'sentiment_score' in result
        assert 'article_count' in result
        
        assert result['overall_sentiment'] in ['positive', 'negative', 'neutral']
        assert result['article_count'] == 3
        assert -1 <= result['sentiment_score'] <= 1
    
    def test_analyze_news_sentiment_empty(self):
        """Test news sentiment analysis with empty data."""
        result = self.analyzer.analyze_news_sentiment([])
        
        assert result['overall_sentiment'] == 'neutral'
        assert result['sentiment_score'] == 0.0
        assert result['article_count'] == 0
    
    def test_calculate_sentiment_score(self):
        """Test sentiment score calculation."""
        positive_text = "Great news! Stock prices are rising."
        negative_text = "Terrible market crash with huge losses."
        neutral_text = "The market traded unchanged today."
        
        pos_result = self.analyzer.calculate_sentiment_score(positive_text)
        neg_result = self.analyzer.calculate_sentiment_score(negative_text)
        neu_result = self.analyzer.calculate_sentiment_score(neutral_text)
        
        assert pos_result['label'] in ['positive', 'negative', 'neutral']
        assert neg_result['label'] in ['positive', 'negative', 'neutral']
        assert neu_result['label'] in ['positive', 'negative', 'neutral']
        
        assert -1 <= pos_result['score'] <= 1
        assert -1 <= neg_result['score'] <= 1
        assert -1 <= neu_result['score'] <= 1
    
    def test_calculate_sentiment_score_empty(self):
        """Test sentiment score with empty text."""
        result = self.analyzer.calculate_sentiment_score("")
        
        assert result['score'] == 0.0
        assert result['label'] == 'neutral'


class TestAnalysisEngine:
    """Test integrated analysis engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        self.sample_data = {
            'price_data': pd.DataFrame({
                'Open': np.random.uniform(95, 105, 30),
                'High': np.random.uniform(100, 110, 30),
                'Low': np.random.uniform(90, 100, 30),
                'Close': np.random.uniform(95, 105, 30),
                'Volume': np.random.randint(100000, 1000000, 30)
            }, index=dates),
            'fundamental_data': {
                'pe_ratio': 22.5,
                'profit_margins': 0.18,
                'latest_revenue': 1000000000
            },
            'market_data': {
                'market_cap': 2500000000000,
                'price': 180.50
            },
            'news_data': [
                {'title': 'Positive News', 'content': 'Great earnings report'},
                {'title': 'Market Concerns', 'content': 'Economic uncertainty ahead'}
            ]
        }
    
    def test_engine_initialization(self):
        """Test AnalysisEngine initialization."""
        assert hasattr(self.engine, 'technical')
        assert hasattr(self.engine, 'fundamental')
        assert hasattr(self.engine, 'pattern')
        assert hasattr(self.engine, 'sentiment')
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        result = self.engine.comprehensive_analysis('AAPL', self.sample_data)
        
        assert 'symbol' in result
        assert 'timestamp' in result
        assert 'analysis_components' in result
        assert 'signals' in result
        
        assert result['symbol'] == 'AAPL'
        
        # Check analysis components
        components = result['analysis_components']
        assert 'technical' in components
        assert 'fundamental' in components
        assert 'patterns' in components
        assert 'sentiment' in components
    
    def test_generate_signals(self):
        """Test signal generation."""
        analysis_results = self.engine.comprehensive_analysis('AAPL', self.sample_data)
        signals = self.engine.generate_signals(analysis_results)
        
        assert 'overall_signal' in signals
        assert 'signal_strength' in signals
        assert 'component_signals' in signals
        assert 'signal_count' in signals
        
        assert signals['overall_signal'] in ['buy', 'sell', 'hold']
        assert signals['signal_strength'] in ['weak', 'moderate', 'strong']
    
    def test_comprehensive_analysis_error_handling(self):
        """Test error handling in comprehensive analysis."""
        invalid_data = {'price_data': pd.DataFrame()}  # Empty DataFrame
        
        result = self.engine.comprehensive_analysis('AAPL', invalid_data)
        
        assert 'symbol' in result
        assert result['symbol'] == 'AAPL'
        assert 'analysis_components' in result or 'error' in result


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_data_format_dataframe(self):
        """Test data format validation for DataFrame."""
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        empty_df = pd.DataFrame()
        
        assert validate_data_format(valid_df) is True
        assert validate_data_format(empty_df) is False
    
    def test_validate_data_format_dict(self):
        """Test data format validation for dictionary."""
        valid_dict = {'key': 'value'}
        empty_dict = {}
        
        assert validate_data_format(valid_dict) is True
        assert validate_data_format(empty_dict) is False
    
    def test_validate_data_format_invalid(self):
        """Test data format validation for invalid types."""
        assert validate_data_format("string") is False
        assert validate_data_format(123) is False
        assert validate_data_format(None) is False
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        scores = {'tech': 0.8, 'fund': 0.6, 'sent': 0.7}
        weights = {'tech': 1.0, 'fund': 1.5, 'sent': 0.5}
        
        result = calculate_composite_score(scores, weights)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_calculate_composite_score_equal_weights(self):
        """Test composite score with equal weights."""
        scores = {'a': 0.5, 'b': 0.7, 'c': 0.9}
        
        result = calculate_composite_score(scores)
        
        expected = (0.5 + 0.7 + 0.9) / 3
        assert abs(result - expected) < 0.001
    
    def test_calculate_composite_score_empty(self):
        """Test composite score with empty scores."""
        result = calculate_composite_score({})
        assert result == 0.0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_technical_analyzer_error_handling(self):
        """Test technical analyzer error handling."""
        analyzer = TechnicalAnalyzer()
        
        # Test with empty data
        result = analyzer.calculate_indicators(pd.DataFrame())
        assert result == {}
        
        # Test RSI with insufficient data
        result = analyzer.calculate_rsi(pd.Series([]))
        assert result['current'] == 50.0
        assert result['signal'] == 'neutral'
    
    def test_sentiment_analyzer_error_handling(self):
        """Test sentiment analyzer error handling."""
        analyzer = SentimentAnalyzer()
        
        # Test with None input
        result = analyzer.calculate_sentiment_score(None)
        assert result['score'] == 0.0
        assert result['label'] == 'neutral'
        
        # Test with empty news data
        result = analyzer.analyze_news_sentiment([])
        assert result['overall_sentiment'] == 'neutral'
        assert result['article_count'] == 0
    
    def test_pattern_analyzer_error_handling(self):
        """Test pattern analyzer error handling."""
        analyzer = PatternAnalyzer()
        
        # Test with empty data
        result = analyzer.detect_chart_patterns(pd.DataFrame())
        assert result['patterns'] == []
        assert result['count'] == 0
        
        # Test candlestick patterns with insufficient data
        small_data = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100.5]})
        result = analyzer.detect_candlestick_patterns(small_data)
        assert result['patterns'] == []


class TestEnumValues:
    """Test enumeration values."""
    
    def test_trend_direction_enum(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.BULLISH.value == "bullish"
        assert TrendDirection.BEARISH.value == "bearish"
        assert TrendDirection.SIDEWAYS.value == "sideways"
        assert TrendDirection.UNKNOWN.value == "unknown"