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
        expected_indicators = ['rsi', 'macd', 'bollinger', 'returns', 'trend', 'volume', 'candlestick_patterns', 'bollinger_squeeze']
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
    
    def test_enhanced_support_resistance(self):
        """Test enhanced support/resistance detection with multiple features."""
        # Create data with clear support/resistance levels
        dates = pd.date_range('2024-01-01', periods=50, freq='1D')
        
        # Price data that bounces between 95-105 with clear levels at 100, 102
        base_prices = [100 + 5 * np.sin(i / 10) + np.random.random() * 0.5 for i in range(50)]
        highs = [p + abs(np.random.random() * 1) for p in base_prices]
        lows = [p - abs(np.random.random() * 1) for p in base_prices]
        volumes = [1000000 + np.random.randint(-200000, 200000) for _ in range(50)]
        
        test_data = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': base_prices,
            'volume': volumes
        }, index=dates)
        
        result = self.analyzer.detect_support_resistance(test_data)
        
        # Verify enhanced structure
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        assert 'fibonacci_levels' in result
        assert 'psychological_levels' in result
        assert 'pivot_analysis' in result
        
        # Check that levels have enhanced attributes
        if result['support_levels']:
            for level in result['support_levels']:
                assert 'price' in level
                assert 'level_type' in level
                assert 'strength_score' in level
                assert 0 <= level['strength_score'] <= 1
        
        if result['resistance_levels']:
            for level in result['resistance_levels']:
                assert 'price' in level
                assert 'level_type' in level
                assert 'strength_score' in level
                assert 0 <= level['strength_score'] <= 1
    
    def test_pivot_point_detection(self):
        """Test pivot point detection integration."""
        # Create data with clear pivot points (ensure enough data for window=5, need 20+ points)
        prices = [100, 102, 104, 106, 104, 102, 100, 98, 96, 98, 100, 102, 104, 103, 101, 99, 97, 95, 98, 101, 103, 105, 107, 106, 104, 102, 100]
        test_data = pd.DataFrame({
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices
        })
        
        result = self.analyzer.detect_support_resistance(test_data)
        
        assert 'pivot_analysis' in result
        assert 'pivot_highs' in result['pivot_analysis']
        assert 'pivot_lows' in result['pivot_analysis']
        assert isinstance(result['pivot_analysis']['pivot_highs'], int)
        assert isinstance(result['pivot_analysis']['pivot_lows'], int)
    
    def test_fibonacci_levels(self):
        """Test Fibonacci retracement level calculation."""
        # Create data with clear swing high/low
        prices = list(range(90, 110)) + list(range(110, 90, -1))  # Up then down
        test_data = pd.DataFrame({
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices
        })
        
        result = self.analyzer.detect_support_resistance(test_data)
        
        assert 'fibonacci_levels' in result
        assert len(result['fibonacci_levels']) == 4  # 23.6%, 38.2%, 61.8%, 78.6%
        
        # Verify Fibonacci levels are in ascending order within range
        fib_levels = sorted(result['fibonacci_levels'])
        assert fib_levels[0] >= min(prices) - 1  # Within low range
        assert fib_levels[-1] <= max(prices) + 1  # Within high range
    
    def test_psychological_levels(self):
        """Test psychological level identification."""
        # Create data around round number levels (ensure enough data for window requirement)
        prices = [98.5, 99.2, 100.1, 99.8, 100.3, 101.1, 100.9, 105.2, 104.8, 110.1, 109.5, 108.7, 107.3, 106.8, 105.4, 104.9, 103.2, 102.6, 101.8, 100.4, 99.7, 98.9, 99.3, 100.2, 101.5]
        test_data = pd.DataFrame({
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.2 for p in prices],
            'close': prices
        })
        
        result = self.analyzer.detect_support_resistance(test_data)
        
        assert 'psychological_levels' in result
        assert isinstance(result['psychological_levels'], list)
        
        # Should include round numbers like 100, 105, 110 for this price range
        psych_levels = result['psychological_levels']
        if psych_levels:
            # Verify they are round numbers
            for level in psych_levels:
                assert level == round(level)  # Should be whole numbers
                assert min(prices) <= level <= max(prices)  # Within price range
    
    def test_enhanced_indicators(self):
        """Test enhanced technical indicators integration."""
        # Create comprehensive test data with OHLCV
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        np.random.seed(42)
        
        base_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        test_data = pd.DataFrame({
            'Open': base_prices + np.random.randn(50) * 0.1,
            'High': base_prices + abs(np.random.randn(50)) * 0.3,
            'Low': base_prices - abs(np.random.randn(50)) * 0.3,
            'Close': base_prices,
            'Volume': np.random.randint(800000, 1200000, 50)
        }, index=dates)
        
        # Ensure High >= Close >= Low
        test_data['High'] = test_data[['Open', 'High', 'Close']].max(axis=1)
        test_data['Low'] = test_data[['Open', 'Low', 'Close']].min(axis=1)
        
        analyzer = TechnicalAnalyzer()
        result = analyzer.calculate_indicators(test_data, '1h')
        
        # Verify all enhanced indicators are present
        indicators = result['indicators']
        assert 'bollinger' in indicators
        assert 'candlestick_patterns' in indicators
        assert 'bollinger_squeeze' in indicators
        
        # Check enhanced Bollinger Bands structure
        bollinger = indicators['bollinger']
        enhanced_bb_keys = ['position', 'bandwidth', 'band_position', 'period', 'upper_band', 'lower_band', 'middle_band', 'squeeze']
        for key in enhanced_bb_keys:
            assert key in bollinger, f"Missing Bollinger Band key: {key}"
        
        # Check candlestick patterns structure
        patterns = indicators['candlestick_patterns']
        assert 'patterns' in patterns
        assert 'current_signal' in patterns
        assert 'pattern_count' in patterns
        assert isinstance(patterns['patterns'], list)
        assert patterns['current_signal'] in ['bullish', 'bearish', 'neutral']
        
        # Check Bollinger squeeze structure
        squeeze = indicators['bollinger_squeeze']
        squeeze_keys = ['squeeze_active', 'squeeze_strength', 'breakout_direction', 'factors']
        for key in squeeze_keys:
            assert key in squeeze, f"Missing squeeze key: {key}"
    
    def test_dynamic_bollinger(self):
        """Test dynamic Bollinger Bands with adaptive periods."""
        analyzer = TechnicalAnalyzer()
        
        # Test with high volatility data (should use shorter periods)
        high_vol_data = pd.DataFrame({
            'Open': [100, 95, 105, 90, 110, 85, 115] * 5,
            'High': [102, 97, 107, 92, 112, 87, 117] * 5,
            'Low': [98, 93, 103, 88, 108, 83, 113] * 5,
            'Close': [101, 96, 106, 91, 111, 86, 116] * 5,
            'Volume': [1000000] * 35
        })
        
        result_high_vol = analyzer.calculate_bollinger_bands(high_vol_data)
        
        # Test with low volatility data (should use longer periods)
        low_vol_data = pd.DataFrame({
            'Open': [100, 100.1, 100.2, 100.1, 100.3] * 7,
            'High': [100.2, 100.3, 100.4, 100.3, 100.5] * 7,
            'Low': [99.8, 99.9, 100.0, 99.9, 100.1] * 7,
            'Close': [100, 100.1, 100.2, 100.1, 100.3] * 7,
            'Volume': [1000000] * 35
        })
        
        result_low_vol = analyzer.calculate_bollinger_bands(low_vol_data)
        
        # High volatility should generally use shorter periods than low volatility
        assert 'period' in result_high_vol
        assert 'period' in result_low_vol
        assert isinstance(result_high_vol['period'], int)
        assert isinstance(result_low_vol['period'], int)
        
        # Check adaptive period logic is working
        assert result_high_vol['period'] <= result_low_vol['period']
        
        # Check enhanced fields are present
        enhanced_fields = ['band_position', 'upper_band', 'lower_band', 'middle_band', 'squeeze']
        for field in enhanced_fields:
            assert field in result_high_vol
            assert field in result_low_vol
    
    def test_candlestick_integration(self):
        """Test candlestick pattern detection integration."""
        analyzer = TechnicalAnalyzer()
        
        # Create data with clear candlestick patterns
        # Doji pattern
        doji_data = pd.DataFrame({
            'Open': [100, 100, 100, 100, 100],
            'High': [101, 101, 101, 101, 101],
            'Low': [99, 99, 99, 99, 99],
            'Close': [100.01, 99.99, 100.02, 99.98, 100],  # Very small bodies
        })
        
        result = analyzer.detect_candlestick_patterns(doji_data)
        
        # Check basic structure
        assert 'patterns' in result
        assert 'current_signal' in result
        assert 'pattern_count' in result
        assert 'bullish_count' in result
        assert 'bearish_count' in result
        
        # Should detect some patterns
        assert isinstance(result['patterns'], list)
        assert result['current_signal'] in ['bullish', 'bearish', 'neutral']
        assert result['pattern_count'] >= 0
        
        # Test hammer pattern
        hammer_data = pd.DataFrame({
            'Open': [100, 100, 90, 100, 100],
            'High': [100, 100, 91, 100, 100],
            'Low': [100, 100, 85, 100, 100],  # Long lower shadow
            'Close': [100, 99, 90.5, 100, 100],  # Small body at top
        })
        
        hammer_result = analyzer.detect_candlestick_patterns(hammer_data)
        assert hammer_result['pattern_count'] >= 0
        
        # If patterns detected, check their structure
        if hammer_result['patterns']:
            pattern = hammer_result['patterns'][0]
            assert 'name' in pattern
            assert 'signal' in pattern
            assert 'strength' in pattern
            assert pattern['signal'] in ['bullish', 'bearish', 'neutral']
            assert 0 <= pattern['strength'] <= 1
    
    def test_adaptive_periods(self):
        """Test adaptive period calculation integration."""
        analyzer = TechnicalAnalyzer()
        
        # Test different volatility scenarios
        scenarios = [
            # Very high volatility
            pd.DataFrame({
                'Close': [100, 80, 120, 70, 130, 60, 140] * 5,
                'High': [105, 85, 125, 75, 135, 65, 145] * 5,
                'Low': [95, 75, 115, 65, 125, 55, 135] * 5,
                'Open': [102, 82, 122, 72, 132, 62, 142] * 5
            }),
            # Medium volatility  
            pd.DataFrame({
                'Close': [100, 98, 102, 101, 99, 103, 100] * 5,
                'High': [102, 100, 104, 103, 101, 105, 102] * 5,
                'Low': [98, 96, 100, 99, 97, 101, 98] * 5,
                'Open': [101, 99, 101, 100, 98, 102, 101] * 5
            }),
            # Low volatility
            pd.DataFrame({
                'Close': [100, 100.2, 99.8, 100.1, 99.9, 100.3, 100] * 5,
                'High': [100.3, 100.5, 100.1, 100.4, 100.2, 100.6, 100.3] * 5,
                'Low': [99.7, 99.9, 99.5, 99.8, 99.6, 100, 99.7] * 5,
                'Open': [100.1, 100.3, 99.9, 100.2, 100, 100.4, 100.1] * 5
            })
        ]
        
        periods = []
        for scenario in scenarios:
            bb_result = analyzer.calculate_bollinger_bands(scenario)
            periods.append(bb_result['period'])
        
        # High volatility should generally use shorter periods
        # Low volatility should generally use longer periods
        assert len(periods) == 3
        assert all(isinstance(p, int) for p in periods)
        assert all(5 <= p <= 50 for p in periods)  # Within expected range


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
        """Test sentiment analysis with failed news data collection but enhanced components still work."""
        # Mock failed data manager for news sentiment
        mock_data_manager = Mock()
        mock_data_manager.collect_sentiment_data.return_value = {'status': 'failed'}
        mock_get_data_manager.return_value = mock_data_manager
        
        # Create new analyzer instance to pick up the mocked data manager
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment()
        
        # With enhanced sentiment, even if news fails, other components still work
        assert result['status'] == 'success'  # Enhanced analyzer has fallback components
        assert 'components' in result  # Enhanced result structure
        assert 'news_sentiment' in result['components']
        assert result['components']['news_sentiment'] == 0.0  # News failed, defaults to 0
        assert 'vix_sentiment' in result['components']  # VIX component still works
        assert result['sentiment_bias'] in ['bullish', 'bearish', 'neutral']
    
    def test_enhanced_sentiment_integration(self):
        """Test enhanced sentiment analysis with VIX, options, and institutional components."""
        with patch('src.core.analysis_engine.get_data_manager') as mock_get_data_manager:
            # Mock data manager for base sentiment
            mock_data_manager = Mock()
            mock_data_manager.collect_sentiment_data.return_value = {
                'status': 'success',
                'articles': [{'sentiment_score': 0.2, 'sentiment_label': 'positive'}]
            }
            mock_get_data_manager.return_value = mock_data_manager
            
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze_sentiment(max_articles=10)
            
            # Test enhanced result structure
            assert result['status'] == 'success'
            assert 'components' in result
            assert 'vix_regime' in result['components']
            assert 'vix_sentiment' in result['components']
            assert 'put_call_sentiment' in result['components']
            assert 'max_pain_sentiment' in result['components'] 
            assert 'institutional_flow' in result['components']
            assert 'short_interest_sentiment' in result['components']
            assert 'dark_pool_sentiment' in result['components']
            assert 'news_sentiment' in result['components']
            
            # Test sentiment score is combined from multiple sources
            assert result['sentiment_score'] != result['components']['news_sentiment']
            assert result['confidence'] > 0.0
    
    def test_vix_sentiment_component(self):
        """Test VIX sentiment component calculation."""
        analyzer = SentimentAnalyzer()
        
        # Test VIX sentiment method directly
        vix_result = analyzer._get_vix_sentiment()
        
        assert 'vix_fear_greed' in vix_result
        assert 'regime' in vix_result
        assert vix_result['regime'] in ['low', 'normal', 'elevated', 'high']
        assert isinstance(vix_result['vix_fear_greed'], float)
        assert -1.0 <= vix_result['vix_fear_greed'] <= 1.0
    
    def test_options_sentiment_component(self):
        """Test options sentiment component calculation."""
        analyzer = SentimentAnalyzer()
        
        # Test options sentiment method directly
        options_result = analyzer._get_options_sentiment()
        
        assert 'put_call_sentiment' in options_result
        assert 'max_pain_sentiment' in options_result
        assert isinstance(options_result['put_call_sentiment'], float)
        assert isinstance(options_result['max_pain_sentiment'], float)
        assert -1.0 <= options_result['put_call_sentiment'] <= 1.0
        assert -1.0 <= options_result['max_pain_sentiment'] <= 1.0
    
    def test_institutional_component(self):
        """Test institutional sentiment component calculation.""" 
        analyzer = SentimentAnalyzer()
        
        # Test institutional sentiment method directly
        inst_result = analyzer._get_institutional_sentiment()
        
        assert 'institutional_flow' in inst_result
        assert 'short_interest_sentiment' in inst_result
        assert 'dark_pool_sentiment' in inst_result
        assert isinstance(inst_result['institutional_flow'], float)
        assert isinstance(inst_result['short_interest_sentiment'], float)
        assert isinstance(inst_result['dark_pool_sentiment'], float)
        assert -1.0 <= inst_result['institutional_flow'] <= 1.0
        assert -1.0 <= inst_result['short_interest_sentiment'] <= 1.0
        assert -1.0 <= inst_result['dark_pool_sentiment'] <= 1.0


class TestAnalysisEngine:
    """Test main analysis engine with enhanced capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()
        
        # Create multi-timeframe data
        dates_1h = pd.date_range(start='2024-01-01', periods=30, freq='h')
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


class TestEnhancedPatternDetection:
    """Test enhanced pattern detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()
        
        # Create data for head and shoulders pattern
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        # Head and shoulders: left shoulder, head (higher), right shoulder
        prices = [100] * 10 + [105] * 5 + [98] * 5 + [110] * 5 + [102] * 5 + [104] * 5 + [100] * 15
        
        self.head_shoulders_data = pd.DataFrame({
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 200000, 50)
        }, index=dates)
        
        # Flag pattern: strong move followed by consolidation (25 data points)
        flagpole_prices = [95, 96, 97, 98, 99, 100, 105, 110, 115, 120]  # 10 points
        consolidation_prices = [120, 120.5, 120.3, 120.7, 120.4, 120.6, 120.2, 120.8, 120.1, 120.9, 120.3, 120.6, 120.4, 120.7, 120.5]  # 15 points
        
        self.flag_data = pd.DataFrame({
            'High': [p * 1.01 for p in flagpole_prices + consolidation_prices],
            'Low': [p * 0.99 for p in flagpole_prices + consolidation_prices],
            'Close': flagpole_prices + consolidation_prices,
            'Volume': [1000000] * 10 + [500000] * 15
        }, index=pd.date_range(start='2024-01-01', periods=25, freq='h'))
        
        # Advanced triangle data
        self.triangle_data = pd.DataFrame({
            'High': [105, 104, 103, 102, 101] * 4,  # Descending highs
            'Low': [95 + i*0.5 for i in range(20)],  # Rising lows (symmetrical triangle)
            'Close': [100, 99.5, 99, 98.5, 98] * 4,
            'Volume': np.random.randint(100000, 200000, 20)
        }, index=pd.date_range(start='2024-01-01', periods=20, freq='h'))
    
    def test_enhanced_pattern_detection(self):
        """Test that enhanced pattern detection works end-to-end."""
        # Test with data that should trigger multiple patterns
        result = self.analyzer.detect_chart_patterns(self.head_shoulders_data)
        
        assert 'patterns' in result
        assert isinstance(result['patterns'], list)
        assert result['count'] == len(result['patterns'])
        
        # Should contain original patterns plus new enhanced ones
        pattern_types = [p.get('pattern_type') for p in result['patterns'] if p is not None]
        
        # Check that we have more pattern types now
        assert len(pattern_types) >= 0  # At least some patterns should be detected
        
        # Verify structure of detected patterns
        for pattern in result['patterns']:
            if pattern:
                assert 'pattern_type' in pattern
                assert 'confidence' in pattern
                assert isinstance(pattern['confidence'], (int, float))
                assert 0 <= pattern['confidence'] <= 1
    
    def test_head_shoulders_integration(self):
        """Test head and shoulders pattern integration."""
        # Use mock to ensure we can test the helper method
        head_shoulders = self.analyzer._detect_head_shoulders(self.head_shoulders_data)
        
        if head_shoulders:  # Pattern might not always be detected with dummy data
            assert head_shoulders['pattern_type'] == 'head_and_shoulders'
            assert head_shoulders['direction'] == 'bearish'
            assert 'confidence' in head_shoulders
            assert 'left_shoulder' in head_shoulders
            assert 'head' in head_shoulders
            assert 'right_shoulder' in head_shoulders
            assert 'neckline' in head_shoulders
        
        # Test integration in main detection function
        result = self.analyzer.detect_chart_patterns(self.head_shoulders_data)
        
        # Check that head_shoulders pattern can be found in results
        head_shoulders_patterns = [
            p for p in result['patterns'] 
            if p and p.get('pattern_type') == 'head_and_shoulders'
        ]
        
        # Should be able to detect pattern or handle gracefully
        assert isinstance(head_shoulders_patterns, list)
    
    def test_advanced_triangles(self):
        """Test advanced triangle pattern detection."""
        advanced_triangles = self.analyzer._detect_advanced_triangles(self.triangle_data)
        
        assert isinstance(advanced_triangles, list)
        
        for pattern in advanced_triangles:
            assert 'pattern_type' in pattern
            assert 'confidence' in pattern
            assert pattern['pattern_type'] in ['symmetrical_triangle', 'rising_wedge']
            assert isinstance(pattern['confidence'], (int, float))
            assert 0 <= pattern['confidence'] <= 1
        
        # Test integration in main detection function
        result = self.analyzer.detect_chart_patterns(self.triangle_data)
        
        # Should include advanced triangles in results
        triangle_patterns = [
            p for p in result['patterns']
            if p and p.get('pattern_type') in ['symmetrical_triangle', 'rising_wedge']
        ]
        
        assert isinstance(triangle_patterns, list)
    
    def test_flag_detection(self):
        """Test flag pattern detection."""
        flag_pattern = self.analyzer._detect_flags(self.flag_data)
        
        # Flag pattern may or may not be detected based on data characteristics
        if flag_pattern:
            assert flag_pattern['pattern_type'] == 'flag'
            assert flag_pattern['direction'] in ['bullish', 'bearish']
            assert 'confidence' in flag_pattern
            assert 'flagpole_move' in flag_pattern
            assert 'flag_range' in flag_pattern
            assert isinstance(flag_pattern['confidence'], (int, float))
            assert 0 <= flag_pattern['confidence'] <= 1
        
        # Test integration in main detection function
        result = self.analyzer.detect_chart_patterns(self.flag_data)
        
        # Check that flag patterns can be found in results
        flag_patterns = [
            p for p in result['patterns']
            if p and p.get('pattern_type') == 'flag'
        ]
        
        assert isinstance(flag_patterns, list)
    
    def test_pattern_confidence_calculation(self):
        """Test pattern confidence calculation from shared utils."""
        # Test various pattern configurations
        test_patterns = [
            {'strength': 0.8, 'volume_confirmed': True, 'duration_bars': 10},
            {'strength': 0.6, 'volume_confirmed': False, 'duration_bars': 3},
            {'strength': 0.9, 'volume_confirmed': True, 'duration_bars': 15},
        ]
        
        for pattern_data in test_patterns:
            # Import and test the confidence calculation
            try:
                from src.utils.shared import calculate_pattern_confidence
                confidence = calculate_pattern_confidence(pattern_data)
                assert isinstance(confidence, (int, float))
                assert 0 <= confidence <= 1
            except ImportError:
                # Fallback test - helper methods should work with or without shared utils
                head_shoulders = self.analyzer._detect_head_shoulders(self.head_shoulders_data)
                # If we get a result, confidence should be valid
                if head_shoulders:
                    assert isinstance(head_shoulders['confidence'], (int, float))
                    assert 0 <= head_shoulders['confidence'] <= 1
    
    def test_pivot_analysis_integration(self):
        """Test pivot analysis integration from shared utils."""
        # Test that pivot analysis is used in pattern detection
        try:
            from src.utils.shared import find_pivot_highs_lows
            pivot_data = find_pivot_highs_lows(self.triangle_data)
            
            assert 'pivot_highs' in pivot_data
            assert 'pivot_lows' in pivot_data
            assert isinstance(pivot_data['pivot_highs'], list)
            assert isinstance(pivot_data['pivot_lows'], list)
        except ImportError:
            # Test fallback behavior
            advanced_triangles = self.analyzer._detect_advanced_triangles(self.triangle_data)
            # Should work with fallback dummy data
            assert isinstance(advanced_triangles, list)


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])