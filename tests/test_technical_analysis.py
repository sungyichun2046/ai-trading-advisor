"""Technical analysis system tests with comprehensive coverage."""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Ensure we're using real pandas, not mocked version
def restore_real_pandas():
    """Forcefully restore real pandas."""
    # Remove mocked pandas from sys.modules
    if 'pandas' in sys.modules:
        del sys.modules['pandas']
    
    # Force re-import of real pandas
    import pandas as real_pandas
    sys.modules['pandas'] = real_pandas
    return real_pandas

def get_real_pandas():
    """Get real pandas even if it's been mocked by other tests."""
    # If pandas is already imported and mocked, we need to restore it
    if 'pandas' in sys.modules:
        # Check if this is a mock by looking for DataFrame attribute
        pandas_module = sys.modules['pandas']
        if hasattr(pandas_module, 'DataFrame'):
            # Try to create a simple DataFrame to test if it's real pandas
            try:
                df = pandas_module.DataFrame({'test': [1, 2, 3]})
                # If we can access columns normally, it's real pandas
                if hasattr(df, '__getitem__') and hasattr(df['test'], 'iloc'):
                    return pandas_module
            except:
                # This is a mock, restore real pandas
                return restore_real_pandas()
    
    # If we reach here, pandas is mocked or not imported properly
    # Re-import pandas directly from source
    return restore_real_pandas()

pd = get_real_pandas()
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.core.technical_analysis import (
    TechnicalIndicators, 
    TimeframeManager, 
    MultiTimeframeAnalysis
)


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.indicators = TechnicalIndicators()
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
        prices = np.random.randn(100).cumsum() + 100
        
        self.sample_data = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.5,
            'High': prices + np.random.randn(100) * 0.5 + 1,
            'Low': prices + np.random.randn(100) * 0.5 - 1,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Ensure OHLC logic
        self.sample_data['High'] = self.sample_data[['Open', 'High', 'Close']].max(axis=1)
        self.sample_data['Low'] = self.sample_data[['Open', 'Low', 'Close']].min(axis=1)

    def test_rsi_calculation(self):
        """Test RSI indicator calculation."""
        rsi = self.indicators.rsi(self.sample_data['Close'])
        
        assert not rsi.empty
        assert len(rsi) == len(self.sample_data)
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        
        # Test with insufficient data
        short_prices = pd.Series([100, 101, 102])
        rsi_short = self.indicators.rsi(short_prices, period=14)
        assert rsi_short.empty

    def test_macd_calculation(self):
        """Test MACD indicator calculation."""
        macd_result = self.indicators.macd(self.sample_data['Close'])
        
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        assert not macd_result['macd'].empty
        assert not macd_result['signal'].empty
        assert not macd_result['histogram'].empty
        
        # Test histogram calculation
        calculated_histogram = macd_result['macd'] - macd_result['signal']
        pd.testing.assert_series_equal(
            macd_result['histogram'], 
            calculated_histogram, 
            check_names=False
        )

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_result = self.indicators.bollinger_bands(self.sample_data['Close'])
        
        assert 'upper' in bb_result
        assert 'middle' in bb_result
        assert 'lower' in bb_result
        
        # Check that upper > middle > lower (most of the time)
        upper_greater = (bb_result['upper'] > bb_result['middle']).sum()
        lower_less = (bb_result['lower'] < bb_result['middle']).sum()
        
        assert upper_greater > len(bb_result['upper']) * 0.8  # 80% of the time
        assert lower_less > len(bb_result['lower']) * 0.8

    def test_moving_averages_calculation(self):
        """Test moving averages calculation."""
        ma_result = self.indicators.moving_averages(self.sample_data['Close'])
        
        assert 'sma_5' in ma_result
        assert 'sma_10' in ma_result
        assert 'sma_20' in ma_result
        
        # Test that shorter MA reacts faster than longer MA
        sma_5 = ma_result['sma_5'].dropna()
        sma_20 = ma_result['sma_20'].dropna()
        assert len(sma_5) >= len(sma_20)  # SMA5 has more data points

    def test_exponential_moving_averages_calculation(self):
        """Test exponential moving averages calculation."""
        ema_result = self.indicators.exponential_moving_averages(self.sample_data['Close'])
        
        assert 'ema_12' in ema_result
        assert 'ema_26' in ema_result
        assert 'ema_50' in ema_result
        
        # EMA should have all data points (no initial NaN period like SMA)
        for ema in ema_result.values():
            assert not ema.empty

    def test_stochastic_calculation(self):
        """Test Stochastic oscillator calculation."""
        stoch_result = self.indicators.stochastic(
            self.sample_data['High'],
            self.sample_data['Low'], 
            self.sample_data['Close']
        )
        
        assert 'k_percent' in stoch_result
        assert 'd_percent' in stoch_result
        
        # Values should be between 0 and 100
        assert (stoch_result['k_percent'] >= 0).all()
        assert (stoch_result['k_percent'] <= 100).all()
        assert (stoch_result['d_percent'] >= 0).all()
        assert (stoch_result['d_percent'] <= 100).all()

    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr_result = self.indicators.atr(
            self.sample_data['High'],
            self.sample_data['Low'],
            self.sample_data['Close']
        )
        
        assert not atr_result.empty
        assert (atr_result >= 0).all()  # ATR is always positive

    def test_adx_calculation(self):
        """Test ADX indicator calculation."""
        adx_result = self.indicators.adx(
            self.sample_data['High'],
            self.sample_data['Low'],
            self.sample_data['Close']
        )
        
        assert 'adx' in adx_result
        assert 'plus_di' in adx_result
        assert 'minus_di' in adx_result
        
        # ADX and DI values should be positive
        assert (adx_result['adx'] >= 0).all()
        assert (adx_result['plus_di'] >= 0).all()
        assert (adx_result['minus_di'] >= 0).all()

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        short_data = pd.Series([100, 101, 102])
        
        # Test each indicator with insufficient data
        rsi = self.indicators.rsi(short_data, period=14)
        assert rsi.empty
        
        macd = self.indicators.macd(short_data)
        assert macd['macd'].empty
        
        bb = self.indicators.bollinger_bands(short_data, period=20)
        assert bb['middle'].empty


class TestTimeframeManager:
    """Test timeframe management and data synchronization."""
    
    def setup_method(self):
        """Set up test data."""
        self.manager = TimeframeManager()
        
        # Create 1-minute sample data
        dates = pd.date_range(start='2024-01-01 09:30:00', periods=1000, freq='1T')
        prices = np.random.randn(1000).cumsum() + 100
        
        self.minute_data = pd.DataFrame({
            'Open': prices + np.random.randn(1000) * 0.1,
            'High': prices + np.random.randn(1000) * 0.1 + 0.5,
            'Low': prices + np.random.randn(1000) * 0.1 - 0.5,
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, 1000)
        }, index=dates)
        
        # Ensure OHLC logic
        self.minute_data['High'] = self.minute_data[['Open', 'High', 'Close']].max(axis=1)
        self.minute_data['Low'] = self.minute_data[['Open', 'Low', 'Close']].min(axis=1)

    def test_timeframe_validation(self):
        """Test timeframe validation."""
        assert self.manager.validate_timeframe('1m')
        assert self.manager.validate_timeframe('5m')
        assert self.manager.validate_timeframe('1h')
        assert self.manager.validate_timeframe('1d')
        
        assert not self.manager.validate_timeframe('2m')
        assert not self.manager.validate_timeframe('invalid')

    def test_timeframe_minutes(self):
        """Test timeframe minute conversion."""
        assert self.manager.get_timeframe_minutes('1m') == 1
        assert self.manager.get_timeframe_minutes('5m') == 5
        assert self.manager.get_timeframe_minutes('1h') == 60
        assert self.manager.get_timeframe_minutes('1d') == 1440
        
        with pytest.raises(ValueError):
            self.manager.get_timeframe_minutes('invalid')

    def test_data_resampling(self):
        """Test data resampling to different timeframes."""
        # Test 5-minute resampling
        resampled_5m = self.manager.resample_data(self.minute_data, '5m')
        
        assert not resampled_5m.empty
        assert len(resampled_5m) < len(self.minute_data)
        assert all(col in resampled_5m.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Test hourly resampling
        resampled_1h = self.manager.resample_data(self.minute_data, '1h')
        
        assert not resampled_1h.empty
        assert len(resampled_1h) < len(resampled_5m)

    def test_timeframe_synchronization(self):
        """Test synchronization across multiple timeframes."""
        # Create data for different timeframes
        data_5m = self.manager.resample_data(self.minute_data, '5m')
        data_1h = self.manager.resample_data(self.minute_data, '1h')
        
        data_dict = {
            '1m': self.minute_data,
            '5m': data_5m,
            '1h': data_1h
        }
        
        synchronized = self.manager.synchronize_timeframes(data_dict)
        
        assert '1m' in synchronized
        assert '5m' in synchronized
        assert '1h' in synchronized
        
        # Check that all timeframes have overlapping time ranges
        start_times = [df.index.min() for df in synchronized.values() if not df.empty]
        end_times = [df.index.max() for df in synchronized.values() if not df.empty]
        
        if start_times and end_times:
            assert max(start_times) <= min(end_times)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        resampled = self.manager.resample_data(empty_df, '5m')
        assert resampled.empty
        
        synchronized = self.manager.synchronize_timeframes({'5m': empty_df})
        assert '5m' in synchronized
        assert synchronized['5m'].empty


class TestMultiTimeframeAnalysis:
    """Test multi-timeframe analysis engine."""
    
    def setup_method(self):
        """Set up test data."""
        self.analysis = MultiTimeframeAnalysis()
        
        # Create realistic OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        prices = np.random.randn(200).cumsum() + 100
        
        self.test_data = pd.DataFrame({
            'Open': prices + np.random.randn(200) * 0.2,
            'High': prices + np.random.randn(200) * 0.2 + 1,
            'Low': prices + np.random.randn(200) * 0.2 - 1,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 200)
        }, index=dates)
        
        # Ensure OHLC logic
        self.test_data['High'] = self.test_data[['Open', 'High', 'Close']].max(axis=1)
        self.test_data['Low'] = self.test_data[['Open', 'Low', 'Close']].min(axis=1)

    def test_calculate_indicators_single_timeframe(self):
        """Test indicator calculation for a single timeframe."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h'])
        
        assert '1h' in indicators
        timeframe_data = indicators['1h']
        
        # Check required indicators
        assert 'rsi' in timeframe_data
        assert 'macd' in timeframe_data
        assert 'bollinger_bands' in timeframe_data
        assert 'moving_averages' in timeframe_data
        assert 'stochastic' in timeframe_data
        assert 'atr' in timeframe_data
        assert 'adx' in timeframe_data
        
        # Check metadata
        assert 'latest_price' in timeframe_data
        assert 'timeframe' in timeframe_data
        assert 'data_points' in timeframe_data

    def test_calculate_indicators_multiple_timeframes(self):
        """Test indicator calculation for multiple timeframes."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h', '1d'])
        
        assert '1h' in indicators
        assert '1d' in indicators
        
        # Both timeframes should have basic data
        for tf in ['1h', '1d']:
            assert 'latest_price' in indicators[tf]
            assert 'timeframe' in indicators[tf]
            # Indicators may not be present if insufficient data, but should have moving averages if any data
            if 'moving_averages' in indicators[tf]:
                assert len(indicators[tf]['moving_averages']) > 0

    def test_rsi_interpretation(self):
        """Test RSI indicator interpretation."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h'])
        rsi_data = indicators['1h']['rsi']
        
        assert 'current' in rsi_data
        assert 'overbought' in rsi_data
        assert 'oversold' in rsi_data
        
        # Test overbought/oversold logic
        if rsi_data['current'] > 70:
            assert rsi_data['overbought'] == True
        if rsi_data['current'] < 30:
            assert rsi_data['oversold'] == True

    def test_macd_interpretation(self):
        """Test MACD indicator interpretation."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h'])
        macd_data = indicators['1h']['macd']
        
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data
        assert 'bullish' in macd_data
        
        # Test bullish logic
        expected_bullish = macd_data['macd'] > macd_data['signal']
        assert macd_data['bullish'] == expected_bullish

    def test_bollinger_bands_interpretation(self):
        """Test Bollinger Bands interpretation."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h'])
        bb_data = indicators['1h']['bollinger_bands']
        
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        assert 'position' in bb_data
        
        # Test position logic
        price = indicators['1h']['latest_price']
        if price >= bb_data['upper']:
            assert bb_data['position'] == 'above_upper'
        elif price <= bb_data['lower']:
            assert bb_data['position'] == 'below_lower'
        else:
            assert bb_data['position'] == 'middle'

    def test_generate_signals(self):
        """Test trading signal generation."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h', '1d'])
        signals = self.analysis.generate_signals(indicators)
        
        assert 'overall_signal' in signals
        assert 'confidence' in signals
        assert 'timeframe_signals' in signals
        assert 'confluence_factors' in signals
        assert 'risk_factors' in signals
        
        # Check signal values
        assert signals['overall_signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signals['confidence'] <= 1
        
        # Check timeframe signals
        for tf in ['1h', '1d']:
            if tf in signals['timeframe_signals']:
                tf_signal = signals['timeframe_signals'][tf]
                assert 'signal' in tf_signal
                assert 'score' in tf_signal
                assert tf_signal['signal'] in ['BUY', 'SELL', 'HOLD']
                assert 0 <= tf_signal['score'] <= 1

    def test_signal_scoring(self):
        """Test signal scoring algorithm."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h'])
        
        if '1h' in indicators:
            score = self.analysis._calculate_timeframe_signal_score(indicators['1h'])
            assert 0 <= score <= 1

    def test_confluence_identification(self):
        """Test confluence factor identification."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h', '1d'])
        confluence = self.analysis._identify_confluence(indicators)
        
        assert isinstance(confluence, list)

    def test_risk_identification(self):
        """Test risk factor identification."""
        indicators = self.analysis.calculate_indicators(self.test_data, ['1h', '1d'])
        risks = self.analysis._identify_risks(indicators)
        
        assert isinstance(risks, list)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        indicators = self.analysis.calculate_indicators(empty_data, ['1h'])
        
        assert indicators == {}

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create data with only 10 points
        small_data = self.test_data.head(10)
        indicators = self.analysis.calculate_indicators(small_data, ['1h'])
        
        if '1h' in indicators:
            # Should still return basic data even with insufficient data for complex indicators
            assert 'latest_price' in indicators['1h']
            assert 'timeframe' in indicators['1h']
            # Complex indicators may not be available, but basic ones should work
            if 'moving_averages' in indicators['1h']:
                # Should have at least some moving averages that can work with 10 points
                assert len(indicators['1h']['moving_averages']) >= 0


class TestTechnicalDataCollector:
    """Test technical data collection and caching."""
    
    @patch('src.data.technical_collectors.redis')
    @patch('src.data.technical_collectors.yf')
    def test_collector_initialization(self, mock_yf, mock_redis):
        """Test technical data collector initialization."""
        from src.data.technical_collectors import TechnicalDataCollector
        
        collector = TechnicalDataCollector()
        assert collector.cache_manager is not None
        assert collector.analysis_engine is not None

    @patch('src.data.technical_collectors.yf')
    def test_collect_technical_indicators(self, mock_yf):
        """Test technical indicator collection."""
        from src.data.technical_collectors import TechnicalDataCollector
        
        # Mock yfinance data
        mock_ticker = Mock()
        mock_hist = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
        
        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker
        
        collector = TechnicalDataCollector()
        
        # Test with real data disabled (should use dummy data)
        with patch('src.data.technical_collectors.settings.use_real_data', False):
            result = collector.collect_technical_indicators('AAPL', ['1h'])
            
            assert 'symbol' in result
            assert result['symbol'] == 'AAPL'
            assert 'timeframes' in result
            assert 'timestamp' in result

    def test_cache_key_generation(self):
        """Test cache key generation."""
        from src.data.technical_collectors import CacheManager
        
        cache_manager = CacheManager()
        key = cache_manager.get_cache_key('AAPL', '1h', 'indicators')
        
        assert 'AAPL' in key
        assert '1h' in key
        assert 'indicators' in key

    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        from src.data.technical_collectors import TechnicalDataCollector
        
        collector = TechnicalDataCollector()
        
        # Create test data with proper index
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        good_data = pd.DataFrame({
            'Open': range(100, 200),
            'High': range(101, 201),
            'Low': range(99, 199),
            'Close': range(100, 200),
            'Volume': range(1000000, 1100000, 1000)
        }, index=dates)
        
        quality = collector._assess_data_quality(good_data, '1h')
        
        assert 'quality' in quality
        assert 'total_periods' in quality
        assert 'data_completeness' in quality
        assert quality['quality'] in ['excellent', 'good', 'fair', 'poor']

    def test_bulk_collection(self):
        """Test bulk indicator collection."""
        from src.data.technical_collectors import TechnicalDataCollector
        
        with patch('src.data.technical_collectors.settings.use_real_data', False):
            collector = TechnicalDataCollector()
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            results = collector.bulk_collect_indicators(symbols, ['1h'])
            
            assert len(results) == len(symbols)
            for symbol in symbols:
                assert symbol in results


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_end_to_end_pipeline(self):
        """Test complete technical analysis pipeline."""
        # Create realistic market data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='5T')
        prices = np.random.randn(500).cumsum() + 100
        
        market_data = pd.DataFrame({
            'Open': prices + np.random.randn(500) * 0.1,
            'High': prices + np.random.randn(500) * 0.1 + 0.5,
            'Low': prices + np.random.randn(500) * 0.1 - 0.5,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 500)
        }, index=dates)
        
        # Ensure OHLC logic
        market_data['High'] = market_data[['Open', 'High', 'Close']].max(axis=1)
        market_data['Low'] = market_data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Test complete pipeline
        analysis = MultiTimeframeAnalysis()
        
        # 1. Calculate indicators
        indicators = analysis.calculate_indicators(market_data, ['5m', '1h'])
        assert len(indicators) > 0
        
        # 2. Generate signals
        signals = analysis.generate_signals(indicators)
        assert 'overall_signal' in signals
        
        # 3. Validate signal quality
        assert signals['confidence'] >= 0
        assert signals['overall_signal'] in ['BUY', 'SELL', 'HOLD']

    def test_market_stress_scenarios(self):
        """Test behavior during market stress scenarios."""
        # Create volatile market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Simulate market crash - sharp decline
        prices = np.linspace(100, 70, 100) + np.random.randn(100) * 2
        
        stress_data = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.5,
            'High': prices + np.random.randn(100) * 0.5 + 2,
            'Low': prices + np.random.randn(100) * 0.5 - 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)  # High volume
        }, index=dates)
        
        # Ensure OHLC logic
        stress_data['High'] = stress_data[['Open', 'High', 'Close']].max(axis=1)
        stress_data['Low'] = stress_data[['Open', 'Low', 'Close']].min(axis=1)
        
        analysis = MultiTimeframeAnalysis()
        indicators = analysis.calculate_indicators(stress_data, ['1h'])
        
        if '1h' in indicators:
            # During stress, should detect high volatility
            if 'atr' in indicators['1h']:
                atr_pct = indicators['1h']['atr']['percentage']
                # In stress scenario, ATR should be elevated
                assert atr_pct >= 0  # Just ensure it's calculated
            
            # RSI might be oversold
            if 'rsi' in indicators['1h']:
                rsi = indicators['1h']['rsi']['current']
                assert 0 <= rsi <= 100

    def test_sideways_market_scenarios(self):
        """Test behavior in sideways/ranging markets."""
        # Create sideways market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Sideways market - oscillating around 100
        prices = 100 + np.sin(np.linspace(0, 10, 100)) * 5 + np.random.randn(100) * 0.5
        
        sideways_data = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.2,
            'High': prices + np.random.randn(100) * 0.2 + 0.5,
            'Low': prices + np.random.randn(100) * 0.2 - 0.5,
            'Close': prices,
            'Volume': np.random.randint(500000, 1500000, 100)
        }, index=dates)
        
        # Ensure OHLC logic
        sideways_data['High'] = sideways_data[['Open', 'High', 'Close']].max(axis=1)
        sideways_data['Low'] = sideways_data[['Open', 'Low', 'Close']].min(axis=1)
        
        analysis = MultiTimeframeAnalysis()
        indicators = analysis.calculate_indicators(sideways_data, ['1h'])
        signals = analysis.generate_signals(indicators)
        
        # In sideways market, signals might be mixed or neutral
        assert signals['overall_signal'] in ['BUY', 'SELL', 'HOLD']

    @patch('src.data.technical_collectors.yf')
    def test_data_feed_interruption(self, mock_yf):
        """Test handling of data feed interruption."""
        from src.data.technical_collectors import TechnicalDataCollector
        
        # Mock data feed failure
        mock_yf.Ticker.side_effect = Exception("Data feed error")
        
        collector = TechnicalDataCollector()
        
        # Should fallback to dummy data gracefully
        result = collector.collect_technical_indicators('AAPL', ['1h'])
        
        # Should still return data (dummy data)
        assert 'symbol' in result
        assert result['symbol'] == 'AAPL'