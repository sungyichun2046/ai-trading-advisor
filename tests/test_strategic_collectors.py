"""Tests for strategic data collectors (fundamental and volatility monitoring)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from src.data.collectors import FundamentalDataCollector, VolatilityMonitor


class TestFundamentalDataCollector:
    """Test FundamentalDataCollector functionality."""
    
    def test_init(self):
        """Test FundamentalDataCollector initialization."""
        collector = FundamentalDataCollector()
        assert collector.retry_attempts == 3
        assert collector.retry_delay == 2
    
    @patch('src.config.settings.use_real_data', False)
    def test_collect_weekly_fundamentals_dummy_mode(self):
        """Test fundamental data collection in dummy mode."""
        collector = FundamentalDataCollector()
        result = collector.collect_weekly_fundamentals("AAPL")
        
        assert result is not None
        assert result["status"] == "success"
        assert result["symbol"] == "AAPL"
        assert result["data_source"] == "dummy"
        assert "pe_ratio" in result
        assert "market_cap" in result
        assert "dividend_yield" in result
        assert isinstance(result["market_cap"], int)
        assert isinstance(result["pe_ratio"], (int, float))
    
    @patch('src.config.settings.use_real_data', True)
    @patch('yfinance.Ticker')
    def test_collect_weekly_fundamentals_real_mode_success(self, mock_ticker_class):
        """Test successful fundamental data collection in real mode."""
        # Mock ticker object
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Mock ticker.info
        mock_ticker.info = {
            'marketCap': 3000000000000,
            'forwardPE': 28.5,
            'priceToBook': 6.8,
            'profitMargins': 0.234,
            'returnOnEquity': 0.456,
            'debtToEquity': 1.23,
            'dividendYield': 0.004,
            'beta': 1.2
        }
        
        # Mock empty DataFrames for financials (to avoid index errors)
        mock_ticker.quarterly_financials = pd.DataFrame()
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_cashflow = pd.DataFrame()
        
        collector = FundamentalDataCollector()
        result = collector.collect_weekly_fundamentals("AAPL")
        
        assert result is not None
        assert result["status"] == "success"
        assert result["symbol"] == "AAPL"
        assert result["data_source"] == "yfinance"
        assert result["market_cap"] == 3000000000000
        assert result["pe_ratio"] == 28.5
        assert result["pb_ratio"] == 6.8
        assert result["profit_margins"] == 0.234
    
    @patch('src.config.settings.use_real_data', True)
    @patch('yfinance.Ticker')
    def test_collect_weekly_fundamentals_real_mode_failure(self, mock_ticker_class):
        """Test fundamental data collection failure in real mode."""
        mock_ticker_class.side_effect = Exception("API Error")
        
        collector = FundamentalDataCollector()
        result = collector.collect_weekly_fundamentals("INVALID")
        
        # Should fallback to dummy data after all retries fail
        assert result is not None
        assert result["data_source"] == "dummy"
        assert result["symbol"] == "INVALID"
    
    def test_generate_dummy_fundamentals_various_symbols(self):
        """Test dummy fundamental data generation for various symbols."""
        collector = FundamentalDataCollector()
        
        symbols = ["SPY", "AAPL", "MSFT", "TSLA", "UNKNOWN"]
        
        for symbol in symbols:
            result = collector._generate_dummy_fundamentals(symbol)
            
            assert result["symbol"] == symbol
            assert result["status"] == "success"
            assert result["data_source"] == "dummy"
            
            # Validate data types and ranges
            assert isinstance(result["market_cap"], int)
            assert 0 < result["pe_ratio"] < 100
            assert 0 < result["pb_ratio"] < 20
            assert 0 < result["profit_margins"] < 1
            assert 0 < result["current_ratio"] < 5
    
    def test_get_latest_financial_item(self):
        """Test extraction of financial items from DataFrame."""
        collector = FundamentalDataCollector()
        
        # Create mock financial DataFrame - each column represents a quarter
        data = {
            '2024-Q1': [1000000, 200000],
            '2024-Q2': [950000, 180000], 
            '2024-Q3': [900000, 160000]
        }
        financials = pd.DataFrame(data, index=['Total Revenue', 'Net Income'])
        
        # Test successful extraction
        revenue = collector._get_latest_financial_item(financials, 'Total Revenue')
        assert revenue == 1000000.0
        
        # Test missing item
        missing = collector._get_latest_financial_item(financials, 'Missing Item')
        assert missing is None
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        empty_result = collector._get_latest_financial_item(empty_df, 'Total Revenue')
        assert empty_result is None


class TestVolatilityMonitor:
    """Test VolatilityMonitor functionality."""
    
    def test_init(self):
        """Test VolatilityMonitor initialization."""
        monitor = VolatilityMonitor()
        assert monitor.retry_attempts == 2
        assert monitor.retry_delay == 1
        assert monitor.vix_high_threshold == 30.0
        assert monitor.vix_extreme_threshold == 40.0
        assert monitor.volume_spike_threshold == 2.0
        assert monitor.price_movement_threshold == 0.05
    
    @patch('src.config.settings.use_real_data', False)
    def test_check_market_volatility_dummy_mode(self):
        """Test volatility monitoring in dummy mode."""
        monitor = VolatilityMonitor()
        result = monitor.check_market_volatility()
        
        assert result is not None
        assert result["status"] == "success"
        assert result["data_source"] == "dummy"
        assert "vix_current" in result
        assert "alerts" in result
        assert "triggers" in result
        assert "volatility_level" in result
        assert isinstance(result["vix_current"], (int, float))
        assert isinstance(result["alerts"], list)
        assert isinstance(result["triggers"], list)
    
    @patch('src.config.settings.use_real_data', True)
    @patch('yfinance.Ticker')
    def test_check_market_volatility_real_mode_normal(self, mock_ticker_class):
        """Test volatility monitoring in real mode with normal conditions."""
        # Mock VIX ticker
        mock_vix_ticker = Mock()
        mock_spy_ticker = Mock()
        
        def ticker_side_effect(symbol):
            if symbol == "^VIX":
                return mock_vix_ticker
            elif symbol == "SPY":
                return mock_spy_ticker
            else:
                return Mock()
        
        mock_ticker_class.side_effect = ticker_side_effect
        
        # Mock VIX data (normal volatility)
        vix_data = pd.DataFrame({
            'Close': [18.5, 19.2, 18.8, 18.0, 17.5],
        })
        mock_vix_ticker.history.return_value = vix_data
        
        # Mock SPY data (normal volume)
        spy_data = pd.DataFrame({
            'Close': [450.0] * 25,
            'High': [452.0] * 25,
            'Low': [448.0] * 25,
            'Volume': [50000000] * 25,
        })
        mock_spy_ticker.history.return_value = spy_data
        
        monitor = VolatilityMonitor()
        result = monitor.check_market_volatility()
        
        assert result["status"] == "success"
        assert result["data_source"] == "yfinance"
        assert result["vix_current"] == 17.5  # Last VIX value
        assert result["volatility_level"] == "LOW"  # Normal conditions
        assert len(result["alerts"]) == 0  # No alerts for normal conditions
    
    @patch('src.config.settings.use_real_data', True)
    @patch('yfinance.Ticker')
    def test_check_market_volatility_real_mode_high_vix(self, mock_ticker_class):
        """Test volatility monitoring with high VIX."""
        mock_vix_ticker = Mock()
        mock_ticker_class.return_value = mock_vix_ticker
        
        # Mock high VIX data
        vix_data = pd.DataFrame({
            'Close': [35.0, 36.2, 35.8, 34.0, 33.5],
        })
        mock_vix_ticker.history.return_value = vix_data
        
        monitor = VolatilityMonitor()
        result = monitor._get_vix_data()
        
        assert result is not None
        assert result["vix_current"] == 33.5
        assert result["vix_5day_avg"] > 30  # Should be above high threshold
    
    @patch('src.config.settings.use_real_data', True) 
    @patch('yfinance.Ticker')
    def test_check_market_volatility_extreme_conditions(self, mock_ticker_class):
        """Test volatility monitoring with extreme market conditions."""
        # Mock tickers for VIX and indices
        mock_vix_ticker = Mock()
        mock_spy_ticker = Mock()
        mock_qqq_ticker = Mock()
        mock_iwm_ticker = Mock()
        
        def ticker_side_effect(symbol):
            if symbol == "^VIX":
                return mock_vix_ticker
            elif symbol == "SPY":
                return mock_spy_ticker
            elif symbol == "QQQ":
                return mock_qqq_ticker  
            elif symbol == "IWM":
                return mock_iwm_ticker
            else:
                return Mock()
        
        mock_ticker_class.side_effect = ticker_side_effect
        
        # Mock extreme VIX data
        vix_data = pd.DataFrame({
            'Close': [45.0, 46.2, 45.8, 44.0, 43.5],
        })
        mock_vix_ticker.history.return_value = vix_data
        
        # Mock SPY with volume spike and price movement
        spy_data = pd.DataFrame({
            'Close': [400.0] * 20 + [380.0] * 4 + [375.0],  # 6.25% drop
            'High': [402.0] * 20 + [385.0] * 4 + [380.0],
            'Low': [398.0] * 20 + [375.0] * 4 + [370.0],
            'Volume': [50000000] * 20 + [150000000] * 5,  # 3x volume spike
        })
        mock_spy_ticker.history.return_value = spy_data
        
        # Mock other indices with normal data
        normal_data = pd.DataFrame({
            'Close': [300.0] * 25,
            'High': [302.0] * 25,
            'Low': [298.0] * 25,
            'Volume': [30000000] * 25,
        })
        mock_qqq_ticker.history.return_value = normal_data
        mock_iwm_ticker.history.return_value = normal_data
        
        monitor = VolatilityMonitor()
        result = monitor.check_market_volatility()
        
        assert result["status"] == "success"
        assert result["vix_current"] == 43.5
        assert result["volatility_level"] == "EXTREME"
        assert "EXTREME_VIX" in result["alerts"]
        assert "SPY_VOLUME_SPIKE" in result["alerts"]
        assert "emergency_analysis" in result["triggers"]
    
    def test_determine_volatility_level(self):
        """Test volatility level determination logic."""
        monitor = VolatilityMonitor()
        
        # Test extreme level
        extreme_data = {"alerts": ["EXTREME_VIX", "SPY_VOLUME_SPIKE"]}
        assert monitor._determine_volatility_level(extreme_data) == "EXTREME"
        
        # Test high level
        high_data = {"alerts": ["HIGH_VIX", "SPY_VOLUME_SPIKE", "QQQ_PRICE_MOVEMENT"]}
        assert monitor._determine_volatility_level(high_data) == "HIGH"
        
        # Test elevated level
        elevated_data = {"alerts": ["SPY_VOLUME_SPIKE", "QQQ_PRICE_MOVEMENT"]}
        assert monitor._determine_volatility_level(elevated_data) == "ELEVATED"
        
        # Test moderate level
        moderate_data = {"alerts": ["SPY_VOLUME_SPIKE"]}
        assert monitor._determine_volatility_level(moderate_data) == "MODERATE"
        
        # Test low level
        low_data = {"alerts": []}
        assert monitor._determine_volatility_level(low_data) == "LOW"
    
    def test_generate_dummy_volatility_extreme_conditions(self):
        """Test dummy volatility generation with extreme conditions."""
        monitor = VolatilityMonitor()
        
        # Test multiple times to check random generation
        for _ in range(10):
            result = monitor._generate_dummy_volatility()
            
            assert result["status"] == "success"
            assert result["data_source"] == "dummy"
            assert isinstance(result["vix_current"], (int, float))
            assert 15.0 <= result["vix_current"] <= 45.0
            assert isinstance(result["alerts"], list)
            assert isinstance(result["triggers"], list)
            assert result["volatility_level"] in ["LOW", "MODERATE", "ELEVATED", "HIGH", "EXTREME"]
    
    @patch('yfinance.Ticker')
    def test_check_index_volatility(self, mock_ticker_class):
        """Test individual index volatility checking."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Mock normal market data
        normal_data = pd.DataFrame({
            'Close': [400.0] * 25,
            'High': [402.0] * 25,
            'Low': [398.0] * 25,
            'Volume': [50000000] * 25,
        })
        mock_ticker.history.return_value = normal_data
        
        monitor = VolatilityMonitor()
        result = monitor._check_index_volatility("SPY")
        
        assert result is not None
        assert result["symbol"] == "SPY"
        assert result["latest_volume"] == 50000000
        assert result["volume_ratio"] == 1.0  # No spike
        assert not result["volume_spike"]
        assert not result["price_movement_alert"]
    
    @patch('yfinance.Ticker')
    def test_check_index_volatility_insufficient_data(self, mock_ticker_class):
        """Test index volatility checking with insufficient data."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Mock insufficient data (less than 20 records needed)
        insufficient_data = pd.DataFrame({
            'Close': [400.0] * 5,
            'Volume': [50000000] * 5,
        })
        mock_ticker.history.return_value = insufficient_data
        
        monitor = VolatilityMonitor()
        result = monitor._check_index_volatility("SPY")
        
        assert result is None  # Should return None for insufficient data
    
    def test_volatility_thresholds(self):
        """Test that volatility thresholds are properly configured."""
        monitor = VolatilityMonitor()
        
        # Verify thresholds are reasonable
        assert 0 < monitor.vix_high_threshold < monitor.vix_extreme_threshold
        assert monitor.volume_spike_threshold > 1.0
        assert 0 < monitor.price_movement_threshold < 1.0
        
        # Test threshold logic
        assert monitor.vix_high_threshold == 30.0
        assert monitor.vix_extreme_threshold == 40.0
        assert monitor.volume_spike_threshold == 2.0
        assert monitor.price_movement_threshold == 0.05


class TestIntegration:
    """Integration tests for strategic collectors."""
    
    def test_fundamental_and_volatility_collectors_compatibility(self):
        """Test that both collectors can work together."""
        fundamental_collector = FundamentalDataCollector()
        volatility_monitor = VolatilityMonitor()
        
        # Test that both can collect data simultaneously
        fundamental_result = fundamental_collector.collect_weekly_fundamentals("SPY")
        volatility_result = volatility_monitor.check_market_volatility()
        
        assert fundamental_result is not None
        assert volatility_result is not None
        assert fundamental_result["status"] == "success"
        assert volatility_result["status"] == "success"
    
    def test_data_format_consistency(self):
        """Test that data formats are consistent across collectors."""
        fundamental_collector = FundamentalDataCollector()
        volatility_monitor = VolatilityMonitor()
        
        fundamental_result = fundamental_collector.collect_weekly_fundamentals("AAPL")
        volatility_result = volatility_monitor.check_market_volatility()
        
        # Both should have required fields
        required_fields = ["status", "timestamp", "data_source"]
        
        for field in required_fields:
            assert field in fundamental_result
            assert field in volatility_result
        
        # Timestamps should be in ISO format
        for result in [fundamental_result, volatility_result]:
            timestamp = result["timestamp"]
            assert isinstance(timestamp, str)
            # Basic ISO format check
            assert "T" in timestamp or "-" in timestamp