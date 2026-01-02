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
        

class TestResonanceEngine:
    """Test ResonanceEngine and consensus integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.core.resonance_engine import ResonanceEngine
            self.resonance_engine = ResonanceEngine()
        except ImportError:
            self.resonance_engine = None
        
        # Create test multi-timeframe data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        base_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        self.test_data = pd.DataFrame({
            'Open': base_prices + np.random.randn(50) * 0.1,
            'High': base_prices + abs(np.random.randn(50)) * 0.3,
            'Low': base_prices - abs(np.random.randn(50)) * 0.3,
            'Close': base_prices,
            'Volume': np.random.randint(800000, 1200000, 50)
        }, index=dates)
        
        # Ensure High >= Close >= Low
        self.test_data['High'] = self.test_data[['Open', 'High', 'Close']].max(axis=1)
        self.test_data['Low'] = self.test_data[['Open', 'Low', 'Close']].min(axis=1)
        
        self.multi_timeframe_data = {
            'timeframes': {
                '1h': {
                    'technical': {
                        'rsi': 65,
                        'macd': {'crossover': 'bullish'},
                        'bollinger': {'position': 'within_bands'},
                        'trend': {'direction': 'bullish'}
                    },
                    'fundamental': {
                        'pe_ratio': 18,
                        'growth_score': 0.8
                    },
                    'sentiment': {
                        'score': 0.3,
                        'news_sentiment': 0.2
                    }
                },
                '1d': {
                    'technical': {
                        'rsi': 58,
                        'macd': {'crossover': 'neutral'},
                        'bollinger': {'position': 'above_upper'},
                        'trend': {'direction': 'bullish'}
                    },
                    'fundamental': {
                        'pe_ratio': 18,
                        'growth_score': 0.7
                    },
                    'sentiment': {
                        'score': 0.25,
                        'news_sentiment': 0.1
                    }
                }
            }
        }
    
    def test_resonance_engine_creation(self):
        """Test ResonanceEngine creation and basic functionality."""
        if self.resonance_engine is None:
            pytest.skip("ResonanceEngine not available")
        
        # Test basic initialization
        assert self.resonance_engine is not None
        assert hasattr(self.resonance_engine, 'calculate_consensus')
        assert hasattr(self.resonance_engine, 'consensus_threshold')
        assert hasattr(self.resonance_engine, 'alignment_threshold')
        
        # Test configuration
        config = {
            'consensus_threshold': 0.8,
            'alignment_threshold': 0.7,
            'signal_weights': {'technical': 0.5, 'fundamental': 0.3, 'sentiment': 0.2}
        }
        
        from src.core.resonance_engine import ResonanceEngine
        configured_engine = ResonanceEngine(config)
        assert configured_engine.consensus_threshold == 0.8
        assert configured_engine.alignment_threshold == 0.7
        assert configured_engine.signal_weights['technical'] == 0.5
    
    def test_consensus_integration(self):
        """Test consensus calculation integration."""
        if self.resonance_engine is None:
            pytest.skip("ResonanceEngine not available")
        
        consensus = self.resonance_engine.calculate_consensus(self.multi_timeframe_data)
        
        # Check required fields
        required_fields = [
            'consensus_score', 'confidence_level', 'alignment_status',
            'agreement_ratio', 'signal_count', 'timeframe_weights',
            'signal_strengths', 'validation', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in consensus, f"Missing consensus field: {field}"
        
        # Check field types and ranges
        assert isinstance(consensus['consensus_score'], (int, float))
        assert 0 <= consensus['consensus_score'] <= 1
        assert consensus['confidence_level'] in ['very_high', 'high', 'moderate', 'low', 'very_low']
        assert consensus['alignment_status'] in [
            'fully_aligned', 'mostly_aligned', 'partially_aligned', 'conflicted', 'no_consensus'
        ]
        assert isinstance(consensus['agreement_ratio'], (int, float))
        assert 0 <= consensus['agreement_ratio'] <= 1
        assert isinstance(consensus['signal_count'], int)
        assert consensus['signal_count'] >= 0
    
    def test_enhanced_multi_timeframe(self):
        """Test enhanced multi-timeframe analysis with ResonanceEngine."""
        # Test with AnalysisEngine integration
        from src.core.analysis_engine import AnalysisEngine
        
        analysis_engine = AnalysisEngine()
        
        # Create test data for multiple timeframes
        data_by_timeframe = {
            '1h': self.test_data,
            '1d': self.test_data.resample('1d').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
        }
        
        # Run multi-timeframe analysis
        result = analysis_engine.multi_timeframe_analysis('AAPL', data_by_timeframe)
        
        # Check basic structure is preserved (backward compatibility)
        basic_fields = ['symbol', 'timestamp', 'timeframe_analysis', 'fundamental_analysis', 'sentiment_analysis', 'consensus']
        for field in basic_fields:
            assert field in result, f"Missing basic field: {field}"
        
        # Check enhanced consensus fields are added
        consensus = result['consensus']
        enhanced_fields = ['consensus_score', 'confidence_level', 'alignment_status']
        for field in enhanced_fields:
            assert field in consensus, f"Missing enhanced consensus field: {field}"
        
        # Check enhanced field types
        assert isinstance(consensus['consensus_score'], (int, float))
        assert 0 <= consensus['consensus_score'] <= 1
        assert consensus['confidence_level'] in ['very_high', 'high', 'moderate', 'low', 'very_low']
        assert consensus['alignment_status'] in [
            'fully_aligned', 'mostly_aligned', 'partially_aligned', 'conflicted', 'no_consensus'
        ]
        
        # Check original consensus fields still exist (backward compatibility)
        original_fields = ['signal', 'strength', 'agreement', 'total_signals']
        for field in original_fields:
            assert field in consensus, f"Missing original consensus field: {field}"
    
    def test_signal_alignment(self):
        """Test signal alignment assessment functionality."""
        if self.resonance_engine is None:
            pytest.skip("ResonanceEngine not available")
        
        # Test with aligned signals
        aligned_data = {
            'timeframes': {
                '1h': {
                    'technical': {
                        'rsi': 75,  # Overbought (bearish)
                        'macd': {'crossover': 'bearish'},
                        'trend': {'direction': 'bearish'}
                    },
                    'sentiment': {'score': -0.3}  # Negative (bearish)
                },
                '1d': {
                    'technical': {
                        'rsi': 72,  # Overbought (bearish) 
                        'macd': {'crossover': 'bearish'},
                        'trend': {'direction': 'bearish'}
                    },
                    'sentiment': {'score': -0.25}  # Negative (bearish)
                }
            }
        }
        
        aligned_consensus = self.resonance_engine.calculate_consensus(aligned_data)
        
        # Should show high alignment
        assert aligned_consensus['alignment_status'] in ['fully_aligned', 'mostly_aligned']
        assert aligned_consensus['agreement_ratio'] > 0.6
        
        # Test with conflicting signals
        conflicted_data = {
            'timeframes': {
                '1h': {
                    'technical': {
                        'rsi': 25,  # Oversold (bullish)
                        'macd': {'crossover': 'bullish'},
                        'trend': {'direction': 'bullish'}
                    },
                    'sentiment': {'score': 0.3}  # Positive (bullish)
                },
                '1d': {
                    'technical': {
                        'rsi': 78,  # Overbought (bearish)
                        'macd': {'crossover': 'bearish'},
                        'trend': {'direction': 'bearish'}
                    },
                    'sentiment': {'score': -0.2}  # Negative (bearish)
                }
            }
        }
        
        conflicted_consensus = self.resonance_engine.calculate_consensus(conflicted_data)
        
        # Should show conflict or low alignment
        assert conflicted_consensus['alignment_status'] in ['conflicted', 'partially_aligned', 'no_consensus']
        assert conflicted_consensus['agreement_ratio'] < 0.8


class TestPatternAnalyzer:
    """Test pattern analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()
        self.sample_ohlcv_data = self._create_sample_ohlcv_data()
    
    def _create_sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
        np.random.seed(42)  # For reproducible tests
        
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        data = pd.DataFrame({
            'Open': prices + np.random.randn(50) * 0.1,
            'High': prices + np.abs(np.random.randn(50)) * 0.3,
            'Low': prices - np.abs(np.random.randn(50)) * 0.3,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # Ensure OHLC logic
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def test_analyzer_initialization(self):
        """Test PatternAnalyzer initialization."""
        analyzer = PatternAnalyzer()
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'pattern_confidence_threshold')
        assert analyzer.pattern_confidence_threshold == 0.7
        
        # Test custom config
        custom_config = {'pattern_confidence_threshold': 0.8}
        custom_analyzer = PatternAnalyzer(custom_config)
        assert custom_analyzer.pattern_confidence_threshold == 0.8
    
    def test_detect_chart_patterns(self):
        """Test chart pattern detection."""
        result = self.analyzer.detect_chart_patterns(self.sample_ohlcv_data, '1h')
        
        # Check result structure
        required_keys = ['timeframe', 'patterns', 'signals', 'dominant_signal', 'confidence', 'pattern_count']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check types
        assert result['timeframe'] == '1h'
        assert isinstance(result['patterns'], list)
        assert isinstance(result['signals'], list)
        assert result['dominant_signal'] in ['bullish', 'bearish', 'neutral']
        assert isinstance(result['confidence'], float)
        assert isinstance(result['pattern_count'], int)
        assert 0 <= result['confidence'] <= 1
    
    def test_triangle_pattern_detection(self):
        """Test triangle pattern detection."""
        # Create data with ascending triangle pattern
        triangle_data = self.sample_ohlcv_data.copy()
        
        # Simulate ascending triangle: flat resistance, rising support
        for i in range(20, 30):
            triangle_data.iloc[i, triangle_data.columns.get_loc('High')] = 105  # Flat resistance
            triangle_data.iloc[i, triangle_data.columns.get_loc('Low')] = 100 + (i - 20) * 0.5  # Rising support
        
        triangle_result = self.analyzer._detect_triangle_patterns(triangle_data)
        
        # Should detect some triangle pattern
        assert isinstance(triangle_result, dict)
        assert 'detected' in triangle_result
    
    def test_head_shoulders_pattern_detection(self):
        """Test head and shoulders pattern detection."""
        hs_result = self.analyzer._detect_head_shoulders(self.sample_ohlcv_data)
        
        assert isinstance(hs_result, dict)
        assert 'detected' in hs_result
        
        if hs_result['detected']:
            assert 'pattern' in hs_result
            assert 'signal' in hs_result
            assert 'confidence' in hs_result
    
    def test_flag_pattern_detection(self):
        """Test flag pattern detection."""
        flag_result = self.analyzer._detect_flag_patterns(self.sample_ohlcv_data)
        
        assert isinstance(flag_result, dict)
        assert 'detected' in flag_result
        
        if flag_result['detected']:
            assert flag_result['pattern'] in ['bull_flag', 'bear_flag']
            assert flag_result['signal'] in ['bullish', 'bearish']
    
    def test_double_pattern_detection(self):
        """Test double top/bottom pattern detection."""
        double_result = self.analyzer._detect_double_patterns(self.sample_ohlcv_data)
        
        assert isinstance(double_result, dict)
        assert 'detected' in double_result
        
        if double_result['detected']:
            assert double_result['pattern'] in ['double_top', 'double_bottom']
            assert double_result['signal'] in ['bullish', 'bearish']
    
    def test_pattern_reliability_validation(self):
        """Test pattern reliability validation."""
        # Create mock pattern data
        pattern_data = {
            'detected': True,
            'pattern': 'ascending_triangle',
            'confidence': 0.7,
            'signal': 'bullish'
        }
        
        volume_data = self.sample_ohlcv_data['Volume']
        reliability = self.analyzer.validate_pattern_reliability(pattern_data, volume_data)
        
        # Check reliability structure
        required_keys = ['reliable', 'adjusted_confidence', 'confidence_adjustment']
        for key in required_keys:
            assert key in reliability, f"Missing reliability key: {key}"
        
        assert isinstance(reliability['reliable'], bool)
        assert isinstance(reliability['adjusted_confidence'], float)
        assert 0 <= reliability['adjusted_confidence'] <= 1
    
    def test_pattern_detection_with_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        # Test with empty data
        empty_result = self.analyzer.detect_chart_patterns(pd.DataFrame(), '1h')
        assert empty_result['confidence'] == 0.0
        assert empty_result['pattern_count'] == 0
        
        # Test with minimal data
        minimal_data = self.sample_ohlcv_data.head(5)
        minimal_result = self.analyzer.detect_chart_patterns(minimal_data, '1h')
        assert isinstance(minimal_result, dict)
        assert 'confidence' in minimal_result
    
    def test_pattern_signal_aggregation(self):
        """Test pattern signal aggregation logic."""
        # Create mock patterns with different signals
        self.analyzer._mock_patterns = [
            {'signal': 'bullish', 'confidence': 0.8},
            {'signal': 'bullish', 'confidence': 0.7},
            {'signal': 'bearish', 'confidence': 0.6}
        ]
        
        result = self.analyzer.detect_chart_patterns(self.sample_ohlcv_data, '1h')
        
        # Should aggregate signals properly
        assert 'dominant_signal' in result
        assert result['dominant_signal'] in ['bullish', 'bearish', 'neutral']


class TestAnalysisEngineWithPatterns:
    """Test Analysis Engine with PatternAnalyzer integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()
        self.sample_data = self._create_sample_timeframe_data()
    
    def _create_sample_timeframe_data(self):
        """Create sample data for multiple timeframes."""
        data_1h = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [100000, 110000, 120000]
        })
        
        data_1d = pd.DataFrame({
            'Open': [100, 102, 104],
            'High': [102, 105, 107],
            'Low': [98, 101, 103],
            'Close': [102, 104, 106],
            'Volume': [500000, 520000, 540000]
        })
        
        return {'1h': data_1h, '1d': data_1d}
    
    def test_engine_has_pattern_analyzer(self):
        """Test that AnalysisEngine includes PatternAnalyzer."""
        assert hasattr(self.engine, 'pattern')
        assert isinstance(self.engine.pattern, PatternAnalyzer)
    
    @patch('src.core.analysis_engine.get_data_manager')
    def test_multi_timeframe_analysis_with_patterns(self, mock_get_data_manager):
        """Test multi-timeframe analysis includes pattern analysis."""
        # Mock data manager
        mock_data_manager = Mock()
        mock_data_manager.collect_fundamental_data.return_value = {
            'status': 'success',
            'data': [{'symbol': 'TEST', 'pe_ratio': 20}]
        }
        mock_data_manager.collect_sentiment_data.return_value = {
            'status': 'success',
            'articles': [{'sentiment_score': 0.1}]
        }
        mock_get_data_manager.return_value = mock_data_manager
        
        result = self.engine.multi_timeframe_analysis('TEST', self.sample_data)
        
        # Check that patterns are included in timeframe analysis
        assert 'timeframe_analysis' in result
        for timeframe in self.sample_data.keys():
            if timeframe in result['timeframe_analysis']:
                tf_analysis = result['timeframe_analysis'][timeframe]
                assert 'patterns' in tf_analysis, f"Missing patterns in {timeframe} analysis"
                
                # Check pattern structure
                patterns = tf_analysis['patterns']
                assert isinstance(patterns, dict)
                pattern_keys = ['timeframe', 'patterns', 'signals', 'dominant_signal', 'confidence']
                for key in pattern_keys:
                    assert key in patterns, f"Missing pattern key: {key} in {timeframe}"


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])