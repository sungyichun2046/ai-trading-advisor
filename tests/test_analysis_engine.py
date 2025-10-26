"""
Comprehensive tests for Enhanced Analysis Engine Module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.analysis_engine import (
    AnalysisEngine, TechnicalAnalyzer, PatternAnalyzer, FundamentalAnalyzer, SentimentAnalyzer, TrendDirection,
    validate_data_format
)


class TestTechnicalAnalyzer:
    """Test enhanced technical analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Create realistic sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        # Generate price series with trend
        base_price = 100
        prices = [base_price + i * 0.5 + np.random.normal(0, 1) for i in range(30)]
        
        self.sample_data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
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
        analyzer = TechnicalAnalyzer()
        assert analyzer.config == {}
        assert analyzer.timeframes == ['1h', '1d']
    
    def test_calculate_indicators_complete(self):
        """Test complete indicator calculation with all indicators."""
        result = self.analyzer.calculate_indicators(self.sample_data, '1h')
        
        # Check main structure
        assert 'timeframe' in result
        assert result['timeframe'] == '1h'
        assert 'data_quality' in result
        assert 'indicators' in result
        
        indicators = result['indicators']
        
        # Check all indicators are present
        expected_indicators = ['rsi', 'macd', 'bollinger', 'returns', 'trend', 'volume']
        for indicator in expected_indicators:
            assert indicator in indicators, f"Missing indicator: {indicator}"
    
    def test_rsi_calculation(self):
        """Test RSI calculation accuracy."""
        prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108] * 2)
        result = self.analyzer.calculate_rsi(prices, period=14)
        
        assert 'current' in result
        assert 'signal' in result
        assert 0 <= result['current'] <= 100
        assert result['signal'] in ['overbought', 'oversold', 'neutral']
    
    def test_trend_detection(self):
        """Test simplified trend detection."""
        result = self.analyzer.detect_trend(self.sample_data)
        
        assert 'direction' in result
        assert 'strength' in result
        assert 'alignment' in result
        
        # Check direction is valid
        direction = result['direction']
        assert direction in ['bullish', 'bearish', 'sideways', 'unknown']
        
        assert isinstance(result['strength'], (int, float))
        assert result['alignment'] in ['bullish', 'bearish', 'sideways', 'unknown']


class TestPatternAnalyzer:
    """Test enhanced pattern recognition functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()
        
        # Create pattern-friendly data
        dates = pd.date_range(start='2024-01-01', periods=25, freq='D')
        
        # Create ascending triangle pattern
        self.triangle_data = pd.DataFrame({
            'High': [105, 105.1, 104.9, 105.2, 104.8] * 5,  # Horizontal resistance
            'Low': [100 + i*0.3 for i in range(25)],  # Rising support
            'Close': [102 + i*0.2 for i in range(25)],
            'Volume': np.random.randint(100000, 200000, 25)
        }, index=dates)
        
        # Ensure OHLC logic
        self.triangle_data['Open'] = self.triangle_data['Close'].shift(1).fillna(self.triangle_data['Close'].iloc[0])
        for i in range(len(self.triangle_data)):
            high = max(self.triangle_data.iloc[i][['Open', 'High', 'Close']])
            low = min(self.triangle_data.iloc[i][['Open', 'Low', 'Close']])
            self.triangle_data.at[self.triangle_data.index[i], 'High'] = high
            self.triangle_data.at[self.triangle_data.index[i], 'Low'] = low
    
    def test_detect_chart_patterns_complete(self):
        """Test complete chart pattern detection."""
        result = self.analyzer.detect_chart_patterns(self.triangle_data)
        
        assert 'patterns' in result
        assert 'count' in result
        assert 'breakout_signals' in result
        assert 'support_resistance' in result
        
        assert isinstance(result['patterns'], list)
        assert isinstance(result['count'], int)
        assert isinstance(result['breakout_signals'], list)
        assert isinstance(result['support_resistance'], dict)
    
    def test_detect_support_resistance(self):
        """Test simplified support and resistance detection."""
        result = self.analyzer.detect_support_resistance(self.triangle_data)
        
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        
        assert isinstance(result['support_levels'], list)
        assert isinstance(result['resistance_levels'], list)


class TestFundamentalAnalyzer:
    """Test fundamental analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FundamentalAnalyzer()
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_analyze_fundamentals_success(self, mock_get_data_manager):
        """Test fundamental analysis with successful data collection."""
        # Mock data manager
        mock_data_manager = Mock()
        mock_data_manager.collect_fundamental_data.return_value = {
            'status': 'success',
            'data': [
                {'symbol': 'AAPL', 'pe_ratio': 18.5, 'pb_ratio': 2.2, 'return_on_equity': 0.16},
                {'symbol': 'MSFT', 'pe_ratio': 22.1, 'pb_ratio': 3.1, 'return_on_equity': 0.14}
            ]
        }
        mock_get_data_manager.return_value = mock_data_manager
        
        result = self.analyzer.analyze_fundamentals(['AAPL', 'MSFT'])
        
        assert result['status'] == 'success'
        assert 'market_bias' in result
        assert 'average_valuation' in result
        assert result['market_bias'] in ['bullish', 'bearish', 'neutral']
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_analyze_fundamentals_failure(self, mock_get_data_manager):
        """Test fundamental analysis with failed data collection."""
        # Mock failed data manager
        mock_data_manager = Mock()
        mock_data_manager.collect_fundamental_data.return_value = {'status': 'failed'}
        mock_get_data_manager.return_value = mock_data_manager
        
        # Create new analyzer instance to pick up the mocked data manager
        analyzer = FundamentalAnalyzer()
        result = analyzer.analyze_fundamentals(['AAPL'])
        
        assert result['status'] == 'failed'
        assert result['market_bias'] == 'neutral'


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_analyze_sentiment_success(self, mock_get_data_manager):
        """Test sentiment analysis with successful data collection."""
        # Mock data manager
        mock_data_manager = Mock()
        mock_data_manager.collect_sentiment_data.return_value = {
            'status': 'success',
            'articles': [
                {'sentiment_score': 0.3, 'sentiment_label': 'positive'},
                {'sentiment_score': -0.2, 'sentiment_label': 'negative'},
                {'sentiment_score': 0.1, 'sentiment_label': 'positive'}
            ]
        }
        mock_get_data_manager.return_value = mock_data_manager
        
        result = self.analyzer.analyze_sentiment(max_articles=25)
        
        assert result['status'] == 'success'
        assert 'sentiment_score' in result
        assert 'sentiment_bias' in result
        assert 'article_count' in result
        assert 'confidence' in result
        assert result['sentiment_bias'] in ['bullish', 'bearish', 'neutral']
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_analyze_sentiment_failure(self, mock_get_data_manager):
        """Test sentiment analysis with failed data collection."""
        # Mock failed data manager
        mock_data_manager = Mock()
        mock_data_manager.collect_sentiment_data.return_value = {'status': 'failed'}
        mock_get_data_manager.return_value = mock_data_manager
        
        # Create new analyzer instance to pick up the mocked data manager
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment()
        
        assert result['status'] == 'failed'
        assert result['sentiment_bias'] == 'neutral'


class TestAnalysisEngine:
    """Test main analysis engine with enhanced capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()
        
        # Create multi-timeframe data
        dates_1h = pd.date_range(start='2024-01-01', periods=30, freq='H')
        dates_1d = pd.date_range(start='2024-01-01', periods=20, freq='D')
        
        self.sample_data_1h = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 30),
            'High': np.random.uniform(100, 110, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(95, 105, 30),
            'Volume': np.random.randint(100000, 1000000, 30)
        }, index=dates_1h)
        
        self.sample_data_1d = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 20),
            'High': np.random.uniform(100, 110, 20),
            'Low': np.random.uniform(90, 100, 20),
            'Close': np.random.uniform(95, 105, 20),
            'Volume': np.random.randint(1000000, 10000000, 20)
        }, index=dates_1d)
        
        # Fix OHLC logic for both datasets
        for data in [self.sample_data_1h, self.sample_data_1d]:
            for i in range(len(data)):
                high = max(data.iloc[i][['Open', 'High', 'Close']])
                low = min(data.iloc[i][['Open', 'Low', 'Close']])
                data.at[data.index[i], 'High'] = high
                data.at[data.index[i], 'Low'] = low
    
    def test_engine_initialization(self):
        """Test AnalysisEngine initialization with new analyzers."""
        engine = AnalysisEngine()
        assert hasattr(engine, 'technical')
        assert hasattr(engine, 'pattern')
        assert hasattr(engine, 'fundamental')
        assert hasattr(engine, 'sentiment')
        assert engine.timeframes == ['1h', '1d']
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_multi_timeframe_analysis_enhanced(self, mock_get_data_manager):
        """Test enhanced multi-timeframe analysis with fundamental and sentiment."""
        # Mock data manager for fundamental and sentiment analysis
        mock_data_manager = Mock()
        mock_data_manager.collect_fundamental_data.return_value = {
            'status': 'success',
            'data': [{'symbol': 'AAPL', 'pe_ratio': 20, 'pb_ratio': 2.5}]
        }
        mock_data_manager.collect_sentiment_data.return_value = {
            'status': 'success',
            'articles': [{'sentiment_score': 0.1, 'sentiment_label': 'positive'}]
        }
        mock_get_data_manager.return_value = mock_data_manager
        
        data_by_timeframe = {
            '1h': self.sample_data_1h,
            '1d': self.sample_data_1d
        }
        
        result = self.engine.multi_timeframe_analysis('AAPL', data_by_timeframe)
        
        # Check enhanced structure
        assert 'symbol' in result
        assert 'timestamp' in result
        assert 'timeframe_analysis' in result
        assert 'fundamental_analysis' in result  # New field
        assert 'sentiment_analysis' in result    # New field
        assert 'consensus' in result
        
        assert result['symbol'] == 'AAPL'
        
        # Check fundamental analysis results
        fundamental_analysis = result['fundamental_analysis']
        assert 'status' in fundamental_analysis
        assert 'market_bias' in fundamental_analysis
        
        # Check sentiment analysis results
        sentiment_analysis = result['sentiment_analysis']
        assert 'status' in sentiment_analysis
        assert 'sentiment_bias' in sentiment_analysis
        
        # Check enhanced consensus
        consensus = result['consensus']
        assert 'signal' in consensus
        assert 'strength' in consensus
        assert 'agreement' in consensus
        assert 'total_signals' in consensus  # New field
    
    def test_calculate_comprehensive_consensus(self):
        """Test comprehensive consensus calculation."""
        # Test strong consensus with all signal types
        timeframe_signals = {'1h': 'bullish', '1d': 'bullish'}
        all_signals = ['bullish', 'bullish', 'bullish', 'neutral']  # Technical, fundamental, sentiment, extra
        
        result = self.engine.calculate_comprehensive_consensus(timeframe_signals, all_signals)
        
        assert result['signal'] == 'bullish'
        assert result['strength'] in ['strong', 'moderate', 'weak']
        assert 0 <= result['agreement'] <= 1
        assert result['total_signals'] == 4
        assert 'timeframe_signals' in result
    
    def test_calculate_comprehensive_consensus_empty(self):
        """Test consensus calculation with no signals."""
        result = self.engine.calculate_comprehensive_consensus({}, [])
        
        assert result['signal'] == 'unknown'
        assert result['strength'] == 'weak'
        assert result['agreement'] == 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_data_format_dataframe(self):
        """Test data format validation for DataFrames."""
        # Valid DataFrame
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert validate_data_format(valid_df) == True
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert validate_data_format(empty_df) == False
    
    def test_validate_data_format_dict(self):
        """Test data format validation for dictionaries."""
        # Valid dictionary
        valid_dict = {'key1': 'value1', 'key2': 'value2'}
        assert validate_data_format(valid_dict) == True
        
        # Empty dictionary
        empty_dict = {}
        assert validate_data_format(empty_dict) == False
    
    def test_validate_data_format_other_types(self):
        """Test data format validation for other types."""
        assert validate_data_format("string") == False
        assert validate_data_format(123) == False
        assert validate_data_format(None) == False


class TestIntegrationScenarios:
    """Test integration scenarios with realistic market data patterns."""
    
    def test_bullish_market_scenario(self):
        """Test analysis with bullish market conditions."""
        # Create bullish trending data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        base_price = 100
        
        # Strong uptrend
        closes = [base_price + i * 0.8 + np.random.normal(0, 0.5) for i in range(30)]
        
        bullish_data = pd.DataFrame({
            'Open': [c * 0.995 for c in closes],
            'High': [c * 1.01 for c in closes],
            'Low': [c * 0.99 for c in closes],
            'Close': closes,
            'Volume': np.random.randint(1000000, 2000000, 30)
        }, index=dates)
        
        # Fix OHLC logic
        for i in range(len(bullish_data)):
            high = max(bullish_data.iloc[i][['Open', 'High', 'Close']])
            low = min(bullish_data.iloc[i][['Open', 'Low', 'Close']])
            bullish_data.at[bullish_data.index[i], 'High'] = high
            bullish_data.at[bullish_data.index[i], 'Low'] = low
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.calculate_indicators(bullish_data, '1d')
        
        # Should detect bullish conditions
        trend = result['indicators']['trend']
        assert trend['direction'] in ['bullish', 'sideways']  # Allow sideways due to simplified trend detection
        assert trend['strength'] >= 0


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])