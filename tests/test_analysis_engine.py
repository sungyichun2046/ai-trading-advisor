"""
Comprehensive tests for Enhanced Analysis Engine Module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.analysis_engine import (
    AnalysisEngine, TechnicalAnalyzer, PatternAnalyzer, TrendDirection,
    validate_data_format, calculate_composite_score
)


class TestTechnicalAnalyzer:
    """Test enhanced technical analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Create realistic sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        # Generate price series with trend
        base_price = 100
        price_changes = np.random.normal(0.001, 0.02, 50)
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        self.sample_data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 50)
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
        
        config = {'test': 'value'}
        analyzer_with_config = TechnicalAnalyzer(config)
        assert analyzer_with_config.config == config
    
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
        expected_indicators = ['rsi', 'macd', 'bollinger', 'atr', 'adx', 'stochastic', 'returns', 'trend', 'volume']
        for indicator in expected_indicators:
            assert indicator in indicators, f"Missing indicator: {indicator}"
        
        # Check RSI structure
        rsi = indicators['rsi']
        assert 'current' in rsi
        assert 'signal' in rsi
        assert isinstance(rsi['current'], (int, float))
        assert rsi['signal'] in ['overbought', 'oversold', 'neutral']
        
        # Check MACD structure
        macd = indicators['macd']
        assert 'crossover' in macd
        assert 'histogram' in macd
        assert macd['crossover'] in ['bullish', 'bearish', 'neutral']
        
        # Check trend structure
        trend = indicators['trend']
        assert 'direction' in trend
        assert 'strength' in trend
        assert 'alignment' in trend
    
    def test_calculate_indicators_insufficient_data(self):
        """Test indicator calculation with insufficient data."""
        small_data = self.sample_data.head(5)
        result = self.analyzer.calculate_indicators(small_data)
        
        assert result['timeframe'] == '1h'
        assert result['indicators'] == {}
    
    def test_calculate_indicators_empty_data(self):
        """Test indicator calculation with empty data."""
        empty_data = pd.DataFrame()
        result = self.analyzer.calculate_indicators(empty_data)
        
        assert result['timeframe'] == '1h'
        assert result['indicators'] == {}
    
    def test_rsi_calculation(self):
        """Test RSI calculation accuracy."""
        prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108] * 3)
        result = self.analyzer.calculate_rsi(prices, period=14)
        
        assert 'current' in result
        assert 'signal' in result
        assert 0 <= result['current'] <= 100
        assert isinstance(result['current'], (int, float))
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = pd.Series([100, 102, 101])
        result = self.analyzer.calculate_rsi(prices, period=14)
        
        assert result['current'] == 50.0
        assert result['signal'] == 'neutral'
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = pd.Series(range(100, 150))
        result = self.analyzer.calculate_macd(prices)
        
        assert 'crossover' in result
        assert 'histogram' in result
        assert result['crossover'] in ['bullish', 'bearish', 'neutral']
        assert isinstance(result['histogram'], (int, float))
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        prices = pd.Series([100 + i + np.random.normal(0, 2) for i in range(30)])
        result = self.analyzer.calculate_bollinger_bands(prices)
        
        assert 'position' in result
        assert 'bandwidth' in result
        assert result['position'] in ['above_upper', 'below_lower', 'within_bands']
        assert isinstance(result['bandwidth'], (int, float))
        assert result['bandwidth'] >= 0
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        result = self.analyzer.calculate_atr(self.sample_data)
        
        assert 'current' in result
        assert 'volatility_level' in result
        assert isinstance(result['current'], (int, float))
        assert result['current'] >= 0
        assert result['volatility_level'] in ['high', 'low', 'normal', 'unknown']
    
    def test_atr_insufficient_columns(self):
        """Test ATR with missing columns."""
        data = pd.DataFrame({'Close': [100, 101, 102]})
        result = self.analyzer.calculate_atr(data)
        
        assert result['current'] == 0.0
        assert result['volatility_level'] == 'unknown'
    
    def test_adx_calculation(self):
        """Test Average Directional Index calculation."""
        result = self.analyzer.calculate_adx(self.sample_data)
        
        assert 'current' in result
        assert 'trend_strength' in result
        assert isinstance(result['current'], (int, float))
        assert result['current'] >= 0
        assert result['trend_strength'] in ['strong', 'moderate', 'weak']
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        result = self.analyzer.calculate_stochastic(self.sample_data)
        
        assert 'k_percent' in result
        assert 'd_percent' in result
        assert 'signal' in result
        assert 0 <= result['k_percent'] <= 100
        assert 0 <= result['d_percent'] <= 100
        assert result['signal'] in ['overbought', 'oversold', 'bullish_crossover', 'bearish_crossover', 'neutral']
    
    def test_trend_detection(self):
        """Test multi-timeframe trend detection."""
        result = self.analyzer.detect_trend(self.sample_data)
        
        assert 'direction' in result
        assert 'strength' in result
        assert 'alignment' in result
        
        # Check direction is a TrendDirection enum or string
        direction = result['direction']
        if hasattr(direction, 'value'):
            assert direction.value in ['bullish', 'bearish', 'sideways', 'unknown']
        else:
            assert direction in ['bullish', 'bearish', 'sideways', 'unknown']
        
        assert isinstance(result['strength'], (int, float))
        assert result['strength'] >= 0
        assert result['alignment'] in ['strong_bullish', 'strong_bearish', 'weak_bullish', 'weak_bearish', 'sideways', 'unknown']
    
    def test_volume_analysis(self):
        """Test enhanced volume analysis."""
        result = self.analyzer.analyze_volume(self.sample_data)
        
        assert 'trend' in result
        assert 'relative_volume' in result
        assert 'volume_pattern' in result
        
        assert result['trend'] in ['increasing', 'decreasing', 'normal', 'unknown']
        assert isinstance(result['relative_volume'], (int, float))
        assert result['relative_volume'] > 0
        assert result['volume_pattern'] in ['accumulation', 'distribution', 'mixed', 'insufficient_data', 'none']
    
    def test_volume_analysis_no_volume_column(self):
        """Test volume analysis without volume data."""
        data = self.sample_data.drop('Volume', axis=1)
        result = self.analyzer.analyze_volume(data)
        
        assert result['trend'] == 'unknown'
        assert result['relative_volume'] == 1.0
        assert result['volume_pattern'] == 'none'


class TestPatternAnalyzer:
    """Test enhanced pattern recognition functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()
        
        # Create pattern-friendly data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create ascending triangle pattern
        self.triangle_data = pd.DataFrame({
            'High': [105, 105.1, 104.9, 105.2, 104.8] * 6,  # Horizontal resistance
            'Low': [100 + i*0.5 for i in range(30)],  # Rising support
            'Close': [102 + i*0.3 for i in range(30)],
            'Volume': np.random.randint(100000, 200000, 30)
        }, index=dates)
        
        # Ensure OHLC logic
        self.triangle_data['Open'] = self.triangle_data['Close'].shift(1).fillna(self.triangle_data['Close'].iloc[0])
        for i in range(len(self.triangle_data)):
            high = max(self.triangle_data.iloc[i][['Open', 'High', 'Close']])
            low = min(self.triangle_data.iloc[i][['Open', 'Low', 'Close']])
            self.triangle_data.at[self.triangle_data.index[i], 'High'] = high
            self.triangle_data.at[self.triangle_data.index[i], 'Low'] = low
    
    def test_analyzer_initialization(self):
        """Test PatternAnalyzer initialization."""
        analyzer = PatternAnalyzer()
        assert analyzer.config == {}
        
        config = {'test': 'value'}
        analyzer_with_config = PatternAnalyzer(config)
        assert analyzer_with_config.config == config
    
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
        
        assert result['count'] == len(result['patterns'])
    
    def test_detect_chart_patterns_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        small_data = self.triangle_data.head(5)
        result = self.analyzer.detect_chart_patterns(small_data)
        
        assert result['patterns'] == []
        assert result['count'] == 0
        assert result['breakout_signals'] == []
    
    def test_detect_triangle_patterns(self):
        """Test triangle pattern detection."""
        result = self.analyzer.detect_triangle_patterns(self.triangle_data)
        
        if result:  # Pattern might be detected
            assert 'pattern_type' in result
            assert 'confidence' in result
            assert 'direction' in result
            
            assert result['pattern_type'] in ['ascending_triangle', 'descending_triangle', 'symmetrical_triangle']
            assert 0 <= result['confidence'] <= 1
            assert result['direction'] in ['bullish', 'bearish', 'neutral']
    
    def test_detect_triangle_patterns_insufficient_data(self):
        """Test triangle pattern detection with insufficient data."""
        small_data = self.triangle_data.head(5)
        result = self.analyzer.detect_triangle_patterns(small_data)
        
        assert result is None
    
    def test_detect_support_resistance(self):
        """Test support and resistance level detection."""
        result = self.analyzer.detect_support_resistance(self.triangle_data)
        
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        
        assert isinstance(result['support_levels'], list)
        assert isinstance(result['resistance_levels'], list)
        
        # Should have at most 3 levels each
        assert len(result['support_levels']) <= 3
        assert len(result['resistance_levels']) <= 3
        
        # Support levels should be sorted in descending order
        if len(result['support_levels']) > 1:
            assert result['support_levels'] == sorted(result['support_levels'], reverse=True)
        
        # Resistance levels should be sorted in ascending order
        if len(result['resistance_levels']) > 1:
            assert result['resistance_levels'] == sorted(result['resistance_levels'])
    
    def test_detect_breakouts(self):
        """Test breakout detection from support/resistance levels."""
        # Create data with a clear breakout
        breakout_data = self.triangle_data.copy()
        breakout_data['Close'].iloc[-1] = 110  # Price breaks above resistance
        
        support_resistance = {
            'support_levels': [100, 102],
            'resistance_levels': [105, 107]
        }
        
        result = self.analyzer.detect_breakouts(breakout_data, support_resistance)
        
        assert isinstance(result, list)
        
        if result:  # Breakout might be detected
            breakout = result[0]
            assert 'type' in breakout
            assert 'level' in breakout
            assert 'direction' in breakout
            assert 'strength' in breakout
            
            assert breakout['type'] in ['resistance_breakout', 'support_breakdown']
            assert breakout['direction'] in ['bullish', 'bearish']
            assert breakout['strength'] in ['strong', 'weak']
    
    def test_detect_breakouts_insufficient_data(self):
        """Test breakout detection with insufficient data."""
        small_data = self.triangle_data.head(1)
        support_resistance = {'support_levels': [100], 'resistance_levels': [105]}
        
        result = self.analyzer.detect_breakouts(small_data, support_resistance)
        assert result == []


class TestAnalysisEngine:
    """Test main analysis engine with multi-timeframe capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()
        
        # Create multi-timeframe data
        dates_1h = pd.date_range(start='2024-01-01', periods=50, freq='H')
        dates_1d = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        self.sample_data_1h = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates_1h)
        
        self.sample_data_1d = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 30),
            'High': np.random.uniform(100, 110, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(95, 105, 30),
            'Volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates_1d)
        
        # Fix OHLC logic for both datasets
        for data in [self.sample_data_1h, self.sample_data_1d]:
            for i in range(len(data)):
                high = max(data.iloc[i][['Open', 'High', 'Close']])
                low = min(data.iloc[i][['Open', 'Low', 'Close']])
                data.at[data.index[i], 'High'] = high
                data.at[data.index[i], 'Low'] = low
    
    def test_engine_initialization(self):
        """Test AnalysisEngine initialization."""
        engine = AnalysisEngine()
        assert hasattr(engine, 'technical')
        assert hasattr(engine, 'pattern')
        assert engine.timeframes == ['1h', '1d']
        
        config = {'test': 'value'}
        engine_with_config = AnalysisEngine(config)
        assert engine_with_config.config == config
    
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis functionality."""
        data_by_timeframe = {
            '1h': self.sample_data_1h,
            '1d': self.sample_data_1d
        }
        
        result = self.engine.multi_timeframe_analysis('AAPL', data_by_timeframe)
        
        # Check main structure
        assert 'symbol' in result
        assert 'timestamp' in result
        assert 'timeframe_analysis' in result
        assert 'consensus' in result
        
        assert result['symbol'] == 'AAPL'
        
        # Check timeframe analysis
        timeframe_analysis = result['timeframe_analysis']
        assert '1h' in timeframe_analysis
        assert '1d' in timeframe_analysis
        
        for timeframe, analysis in timeframe_analysis.items():
            assert 'technical' in analysis
            assert 'patterns' in analysis
            assert 'data_points' in analysis
            
            assert isinstance(analysis['data_points'], int)
            assert analysis['data_points'] > 0
        
        # Check consensus
        consensus = result['consensus']
        assert 'signal' in consensus
        assert 'strength' in consensus
        assert 'agreement' in consensus
        
        assert consensus['signal'] in ['bullish', 'bearish', 'sideways', 'unknown']
        assert consensus['strength'] in ['strong', 'moderate', 'weak']
        assert isinstance(consensus['agreement'], (int, float))
        assert 0 <= consensus['agreement'] <= 1
    
    def test_multi_timeframe_analysis_empty_data(self):
        """Test multi-timeframe analysis with empty data."""
        data_by_timeframe = {
            '1h': pd.DataFrame(),
            '1d': pd.DataFrame()
        }
        
        result = self.engine.multi_timeframe_analysis('AAPL', data_by_timeframe)
        
        assert result['symbol'] == 'AAPL'
        assert result['timeframe_analysis'] == {}
        assert result['consensus']['signal'] == 'unknown'
    
    def test_calculate_timeframe_consensus(self):
        """Test timeframe consensus calculation."""
        # Test strong consensus
        timeframe_signals = {
            '1h': 'bullish',
            '1d': 'bullish',
            '4h': 'bullish'
        }
        
        result = self.engine.calculate_timeframe_consensus(timeframe_signals)
        
        assert result['signal'] == 'bullish'
        assert result['strength'] == 'strong'
        assert result['agreement'] == 1.0
        assert 'timeframe_signals' in result
        
        # Test mixed signals
        mixed_signals = {
            '1h': 'bullish',
            '1d': 'bearish',
            '4h': 'neutral'
        }
        
        result = self.engine.calculate_timeframe_consensus(mixed_signals)
        
        assert result['signal'] in ['bullish', 'bearish', 'neutral']
        assert result['strength'] == 'weak'
        assert result['agreement'] < 1.0
    
    def test_calculate_timeframe_consensus_empty(self):
        """Test consensus calculation with no signals."""
        result = self.engine.calculate_timeframe_consensus({})
        
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
        
        # DataFrame with no columns
        no_cols_df = pd.DataFrame(index=[0, 1, 2])
        assert validate_data_format(no_cols_df) == False
    
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
        assert validate_data_format([1, 2, 3]) == False
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        # Test equal weights
        scores = {'technical': 0.8, 'fundamental': 0.6, 'sentiment': 0.7}
        result = calculate_composite_score(scores)
        expected = (0.8 + 0.6 + 0.7) / 3
        assert abs(result - expected) < 0.001
        
        # Test custom weights
        weights = {'technical': 0.5, 'fundamental': 0.3, 'sentiment': 0.2}
        result = calculate_composite_score(scores, weights)
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.7 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(result - expected) < 0.001
        
        # Test empty scores
        result = calculate_composite_score({})
        assert result == 0.0
        
        # Test partial weights (missing keys should use default weight 1.0)
        partial_weights = {'technical': 0.5}
        result = calculate_composite_score(scores, partial_weights)
        expected = (0.8 * 0.5 + 0.6 * 1.0 + 0.7 * 1.0) / (0.5 + 1.0 + 1.0)
        assert abs(result - expected) < 0.001


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_technical_analyzer_error_handling(self):
        """Test technical analyzer error handling."""
        analyzer = TechnicalAnalyzer()
        
        # Test with corrupted data
        corrupted_data = pd.DataFrame({
            'Close': [np.nan, np.inf, -np.inf, 100, 101]
        })
        
        # Should not raise exception
        result = analyzer.calculate_indicators(corrupted_data)
        assert isinstance(result, dict)
    
    def test_pattern_analyzer_error_handling(self):
        """Test pattern analyzer error handling."""
        analyzer = PatternAnalyzer()
        
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        
        # Should not raise exception
        result = analyzer.detect_chart_patterns(incomplete_data)
        assert isinstance(result, dict)
        assert result['patterns'] == []
    
    def test_analysis_engine_error_handling(self):
        """Test analysis engine error handling."""
        engine = AnalysisEngine()
        
        # Test with corrupted timeframe data
        corrupted_data = {
            '1h': pd.DataFrame({'Close': [np.nan, np.inf]}),
            '1d': pd.DataFrame()
        }
        
        # Should not raise exception
        result = engine.multi_timeframe_analysis('TEST', corrupted_data)
        assert isinstance(result, dict)
        assert result['symbol'] == 'TEST'


class TestIntegrationScenarios:
    """Test integration scenarios with realistic market data patterns."""
    
    def test_bullish_market_scenario(self):
        """Test analysis with bullish market conditions."""
        # Create bullish trending data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        base_price = 100
        
        # Strong uptrend
        closes = [base_price + i * 0.5 + np.random.normal(0, 0.5) for i in range(50)]
        
        bullish_data = pd.DataFrame({
            'Open': [c * 0.995 for c in closes],
            'High': [c * 1.01 for c in closes],
            'Low': [c * 0.99 for c in closes],
            'Close': closes,
            'Volume': np.random.randint(1000000, 2000000, 50)
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
        assert trend['direction'] in [TrendDirection.BULLISH, 'bullish']
        assert trend['strength'] > 0
    
    def test_multi_timeframe_alignment(self):
        """Test multi-timeframe analysis alignment."""
        engine = AnalysisEngine()
        
        # Create aligned bullish data for both timeframes
        dates_1h = pd.date_range(start='2024-01-01', periods=100, freq='H')
        dates_1d = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Both timeframes show bullish trend
        data_1h = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [100 + i * 0.1 + 1 for i in range(100)],
            'Low': [100 + i * 0.1 - 0.5 for i in range(100)],
            'Close': [100 + i * 0.1 + 0.2 for i in range(100)],
            'Volume': np.random.randint(100000, 200000, 100)
        }, index=dates_1h)
        
        data_1d = pd.DataFrame({
            'Open': [100 + i * 0.5 for i in range(30)],
            'High': [100 + i * 0.5 + 2 for i in range(30)],
            'Low': [100 + i * 0.5 - 1 for i in range(30)],
            'Close': [100 + i * 0.5 + 0.5 for i in range(30)],
            'Volume': np.random.randint(1000000, 2000000, 30)
        }, index=dates_1d)
        
        data_by_timeframe = {'1h': data_1h, '1d': data_1d}
        
        result = engine.multi_timeframe_analysis('TEST', data_by_timeframe)
        
        # Should show strong consensus if both timeframes align
        consensus = result['consensus']
        if consensus['agreement'] > 0.8:
            assert consensus['strength'] in ['strong', 'moderate']


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])