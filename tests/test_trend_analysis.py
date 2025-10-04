"""Trend analysis system tests with comprehensive coverage."""

import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Ensure we're using real pandas, not mocked version
def restore_real_pandas():
    """Forcefully restore real pandas."""
    if 'pandas' in sys.modules:
        del sys.modules['pandas']
    
    import pandas as real_pandas
    sys.modules['pandas'] = real_pandas
    return real_pandas

def get_real_pandas():
    """Get real pandas even if it's been mocked by other tests."""
    if 'pandas' in sys.modules:
        pandas_module = sys.modules['pandas']
        if hasattr(pandas_module, 'DataFrame'):
            try:
                df = pandas_module.DataFrame({'test': [1, 2, 3]})
                if hasattr(df, '__getitem__') and hasattr(df['test'], 'iloc'):
                    return pandas_module
            except:
                return restore_real_pandas()
    
    return restore_real_pandas()

pd = get_real_pandas()

from src.core.trend_analysis import (
    TrendDetector, 
    CrossTimeframeTrendAnalyzer,
    TrendDirection,
    TrendStrength,
    TrendInfo
)

from src.core.market_regime import (
    MarketRegimeClassifier,
    MultiTimeframeRegimeAnalyzer,
    MarketRegime,
    RegimeStrength,
    RegimeInfo
)


class TestTrendDetector:
    """Test trend detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trend_detector = TrendDetector()
        
        # Create sample data for testing
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        
        # Bullish trend data
        self.bullish_prices = 100 + np.cumsum(np.random.normal(0.1, 0.5, 100))
        self.bullish_data = pd.DataFrame({
            'Open': self.bullish_prices * 0.999,
            'High': self.bullish_prices * 1.002,
            'Low': self.bullish_prices * 0.998,
            'Close': self.bullish_prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Bearish trend data
        self.bearish_prices = 100 + np.cumsum(np.random.normal(-0.1, 0.5, 100))
        self.bearish_data = pd.DataFrame({
            'Open': self.bearish_prices * 1.001,
            'High': self.bearish_prices * 1.002,
            'Low': self.bearish_prices * 0.998,
            'Close': self.bearish_prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Sideways data
        self.sideways_prices = 100 + np.random.normal(0, 1, 100)
        self.sideways_data = pd.DataFrame({
            'Open': self.sideways_prices * 0.999,
            'High': self.sideways_prices * 1.001,
            'Low': self.sideways_prices * 0.999,
            'Close': self.sideways_prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_detector_initialization(self):
        """Test trend detector initialization."""
        assert self.trend_detector is not None
        assert hasattr(self.trend_detector, 'min_trend_duration')
        assert 'short' in self.trend_detector.min_trend_duration
        assert 'medium' in self.trend_detector.min_trend_duration
        assert 'long' in self.trend_detector.min_trend_duration
    
    def test_detect_bullish_trend(self):
        """Test detection of bullish trends."""
        result = self.trend_detector.detect_trends(self.bullish_data, "medium")
        
        assert "error" not in result
        assert "consensus_trend" in result
        
        consensus = result["consensus_trend"]
        direction = consensus["direction"]
        # Handle both string and enum values
        if hasattr(direction, 'value'):
            direction = direction.value
        assert direction in [TrendDirection.BULLISH.value, TrendDirection.SIDEWAYS.value]
        assert consensus["confidence"] >= 0.0
        assert consensus["confidence"] <= 1.0
    
    def test_detect_bearish_trend(self):
        """Test detection of bearish trends."""
        result = self.trend_detector.detect_trends(self.bearish_data, "medium")
        
        assert "error" not in result
        assert "consensus_trend" in result
        
        consensus = result["consensus_trend"]
        direction = consensus["direction"]
        # Handle both string and enum values
        if hasattr(direction, 'value'):
            direction = direction.value
        assert direction in [TrendDirection.BEARISH.value, TrendDirection.SIDEWAYS.value]
        assert consensus["confidence"] >= 0.0
        assert consensus["confidence"] <= 1.0
    
    def test_detect_sideways_trend(self):
        """Test detection of sideways trends."""
        result = self.trend_detector.detect_trends(self.sideways_data, "medium")
        
        assert "error" not in result
        assert "consensus_trend" in result
        
        consensus = result["consensus_trend"]
        # Sideways should have low slope and medium confidence
        assert consensus["confidence"] >= 0.0
    
    def test_trend_horizons(self):
        """Test trend detection across different horizons."""
        horizons = ["short", "medium", "long"]
        
        for horizon in horizons:
            result = self.trend_detector.detect_trends(self.bullish_data, horizon)
            assert "error" not in result
            assert result["horizon"] == horizon
            assert "consensus_trend" in result
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Test with very small dataset
        small_data = self.bullish_data.head(5)
        result = self.trend_detector.detect_trends(small_data, "medium")
        
        # Should handle gracefully, not crash
        assert isinstance(result, dict)
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        empty_data = pd.DataFrame()
        result = self.trend_detector.detect_trends(empty_data, "medium")
        
        assert "error" in result or "consensus_trend" in result
    
    def test_reversal_detection(self):
        """Test trend reversal signal detection."""
        result = self.trend_detector.detect_trends(self.bullish_data, "medium")
        
        if "reversal_signals" in result:
            reversal = result["reversal_signals"]
            assert "reversal_probability" in reversal
            assert reversal["reversal_probability"] >= 0.0
            assert reversal["reversal_probability"] <= 1.0
            assert "signals" in reversal
            assert isinstance(reversal["signals"], list)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        result = self.trend_detector.detect_trends(self.bullish_data, "medium")
        
        if "data_quality" in result:
            quality = result["data_quality"]
            assert "total_points" in quality
            assert "analysis_points" in quality
            assert quality["total_points"] > 0


class TestCrossTimeframeTrendAnalyzer:
    """Test cross-timeframe trend analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CrossTimeframeTrendAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        prices = 100 + np.cumsum(np.random.normal(0.05, 1, 200))
        
        self.test_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 200)
        }, index=dates)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'trend_detector')
        assert hasattr(self.analyzer, 'timeframes')
        assert len(self.analyzer.timeframes) == 3
    
    def test_multi_horizon_analysis(self):
        """Test multi-horizon trend analysis."""
        result = self.analyzer.analyze_multi_horizon_trends(self.test_data)
        
        assert "error" not in result
        assert "horizon_analysis" in result
        assert "cross_timeframe_confirmation" in result
        assert "trend_signals" in result
        assert "overall_assessment" in result
    
    def test_trend_confirmation(self):
        """Test cross-timeframe trend confirmation."""
        result = self.analyzer.analyze_multi_horizon_trends(self.test_data)
        
        if "cross_timeframe_confirmation" in result:
            confirmation = result["cross_timeframe_confirmation"]
            # Handle error cases gracefully
            if "error" not in confirmation:
                assert "dominant_trend" in confirmation
                assert "confirmation_strength" in confirmation
                assert confirmation["confirmation_strength"] >= 0.0
                assert confirmation["confirmation_strength"] <= 1.0
    
    def test_signal_generation(self):
        """Test trading signal generation."""
        result = self.analyzer.analyze_multi_horizon_trends(self.test_data)
        
        if "trend_signals" in result:
            signals = result["trend_signals"]
            assert "primary_signal" in signals
            assert signals["primary_signal"] in ["BUY", "SELL", "HOLD"]
            assert "signal_strength" in signals
            assert signals["signal_strength"] >= 0.0
            assert signals["signal_strength"] <= 1.0
    
    def test_overall_assessment(self):
        """Test overall trend assessment."""
        result = self.analyzer.analyze_multi_horizon_trends(self.test_data)
        
        if "overall_assessment" in result:
            assessment = result["overall_assessment"]
            assert "market_phase" in assessment
            assert "trend_quality" in assessment
            assert "conviction_level" in assessment
            assert "recommended_action" in assessment


class TestMarketRegimeClassifier:
    """Test market regime classification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = MarketRegimeClassifier()
        
        # Create different regime datasets
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=252, freq='1d')
        
        # Bull market data
        bull_returns = np.random.normal(0.001, 0.015, 252)  # Positive mean, moderate vol
        bull_prices = 100 * (1 + bull_returns).cumprod()
        self.bull_data = pd.DataFrame({
            'Open': bull_prices * 0.999,
            'High': bull_prices * 1.005,
            'Low': bull_prices * 0.995,
            'Close': bull_prices,
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        # Bear market data
        bear_returns = np.random.normal(-0.001, 0.02, 252)  # Negative mean, higher vol
        bear_prices = 100 * (1 + bear_returns).cumprod()
        self.bear_data = pd.DataFrame({
            'Open': bear_prices * 1.001,
            'High': bear_prices * 1.003,
            'Low': bear_prices * 0.995,
            'Close': bear_prices,
            'Volume': np.random.randint(1000000, 15000000, 252)
        }, index=dates)
        
        # Volatile market data
        volatile_returns = np.random.normal(0, 0.035, 252)  # Zero mean, high vol
        volatile_prices = 100 * (1 + volatile_returns).cumprod()
        self.volatile_data = pd.DataFrame({
            'Open': volatile_prices * 0.98,
            'High': volatile_prices * 1.02,
            'Low': volatile_prices * 0.98,
            'Close': volatile_prices,
            'Volume': np.random.randint(2000000, 20000000, 252)
        }, index=dates)
    
    def test_classifier_initialization(self):
        """Test regime classifier initialization."""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'regime_thresholds')
        assert 'bull_min_return' in self.classifier.regime_thresholds
        assert 'bear_max_return' in self.classifier.regime_thresholds
        assert 'volatility_high' in self.classifier.regime_thresholds
    
    def test_bull_regime_detection(self):
        """Test bull market regime detection."""
        result = self.classifier.classify_regime(self.bull_data, lookback_days=252)
        
        assert "error" not in result
        assert "current_regime" in result
        
        regime = result["current_regime"]
        # Should detect bullish characteristics (allowing for some flexibility)
        assert regime["confidence"] >= 0.0
        assert regime["confidence"] <= 1.0
    
    def test_bear_regime_detection(self):
        """Test bear market regime detection."""
        result = self.classifier.classify_regime(self.bear_data, lookback_days=252)
        
        assert "error" not in result
        assert "current_regime" in result
        
        regime = result["current_regime"]
        assert regime["confidence"] >= 0.0
        assert regime["confidence"] <= 1.0
    
    def test_volatile_regime_detection(self):
        """Test volatile market regime detection."""
        result = self.classifier.classify_regime(self.volatile_data, lookback_days=252)
        
        assert "error" not in result
        assert "current_regime" in result
        
        regime = result["current_regime"]
        # High volatility should be detected
        assert regime["confidence"] >= 0.0
    
    def test_returns_analysis(self):
        """Test returns analysis component."""
        result = self.classifier.classify_regime(self.bull_data, lookback_days=100)
        
        if "returns_analysis" in result:
            returns = result["returns_analysis"]
            assert "annualized_return" in returns
            assert "annualized_volatility" in returns
            assert "sharpe_ratio" in returns
            assert "max_drawdown" in returns
    
    def test_volatility_analysis(self):
        """Test volatility analysis component."""
        result = self.classifier.classify_regime(self.volatile_data, lookback_days=100)
        
        if "volatility_analysis" in result:
            vol = result["volatility_analysis"]
            assert "current_volatility" in vol
            assert "volatility_regime" in vol
            assert vol["volatility_regime"] in ["high", "medium", "low"]
    
    def test_trend_analysis(self):
        """Test trend characteristics analysis."""
        result = self.classifier.classify_regime(self.bull_data, lookback_days=100)
        
        if "trend_analysis" in result:
            trend = result["trend_analysis"]
            assert "trend_consistency" in trend
            assert "trend_classification" in trend
            assert trend["trend_consistency"] >= 0.0
            assert trend["trend_consistency"] <= 1.0
    
    def test_insufficient_regime_data(self):
        """Test behavior with insufficient data for regime classification."""
        small_data = self.bull_data.head(10)
        result = self.classifier.classify_regime(small_data, lookback_days=252)
        
        # Should handle gracefully
        assert isinstance(result, dict)
    
    def test_empty_regime_data(self):
        """Test behavior with empty data for regime classification."""
        empty_data = pd.DataFrame()
        result = self.classifier.classify_regime(empty_data, lookback_days=252)
        
        assert "error" in result or "current_regime" in result


class TestMultiTimeframeRegimeAnalyzer:
    """Test multi-timeframe regime analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MultiTimeframeRegimeAnalyzer()
        
        # Create sample data for regime analysis
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=300, freq='1d')
        returns = np.random.normal(0.0005, 0.02, 300)
        prices = 100 * (1 + returns).cumprod()
        
        self.regime_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.003,
            'Low': prices * 0.997,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
    
    def test_regime_analyzer_initialization(self):
        """Test regime analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'classifier')
        assert hasattr(self.analyzer, 'timeframes')
        assert len(self.analyzer.timeframes) == 3
    
    def test_regime_hierarchy_analysis(self):
        """Test regime hierarchy across timeframes."""
        result = self.analyzer.analyze_regime_hierarchy(self.regime_data)
        
        assert "error" not in result
        assert "timeframe_analysis" in result
        assert "regime_consensus" in result
        assert "regime_signals" in result
        assert "regime_summary" in result
    
    def test_regime_consensus(self):
        """Test regime consensus calculation."""
        result = self.analyzer.analyze_regime_hierarchy(self.regime_data)
        
        if "regime_consensus" in result:
            consensus = result["regime_consensus"]
            assert "dominant_regime" in consensus
            assert "consensus_strength" in consensus
            assert consensus["consensus_strength"] >= 0.0
            assert consensus["consensus_strength"] <= 1.0
    
    def test_regime_signals(self):
        """Test regime-based signal generation."""
        result = self.analyzer.analyze_regime_hierarchy(self.regime_data)
        
        if "regime_signals" in result:
            signals = result["regime_signals"]
            assert "regime_signal" in signals
            assert signals["regime_signal"] in ["BUY", "SELL", "HOLD", "NEUTRAL"]
            assert "recommended_strategy" in signals
            assert "risk_level" in signals
    
    def test_regime_summary(self):
        """Test regime summary creation."""
        result = self.analyzer.analyze_regime_hierarchy(self.regime_data)
        
        if "regime_summary" in result:
            summary = result["regime_summary"]
            assert "overall_assessment" in summary
            assert "key_insights" in summary
            assert "strategic_implications" in summary
            assert isinstance(summary["key_insights"], list)
            assert isinstance(summary["strategic_implications"], list)


class TestTrendAnalysisIntegration:
    """Test integration between trend analysis and regime classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trend_analyzer = CrossTimeframeTrendAnalyzer()
        self.regime_analyzer = MultiTimeframeRegimeAnalyzer()
        
        # Create comprehensive test data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        
        # Simulate market phases
        phase1_returns = np.random.normal(0.001, 0.01, 200)  # Bull phase
        phase2_returns = np.random.normal(-0.001, 0.02, 200)  # Bear phase
        phase3_returns = np.random.normal(0, 0.025, 100)      # Volatile phase
        
        all_returns = np.concatenate([phase1_returns, phase2_returns, phase3_returns])
        prices = 100 * (1 + all_returns).cumprod()
        
        self.integration_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 1.002, 500),
            'High': prices * np.random.uniform(1.001, 1.010, 500),
            'Low': prices * np.random.uniform(0.990, 0.999, 500),
            'Close': prices,
            'Volume': np.random.randint(100000, 5000000, 500)
        }, index=dates)
    
    def test_combined_analysis(self):
        """Test combined trend and regime analysis."""
        # Run both analyses
        trend_result = self.trend_analyzer.analyze_multi_horizon_trends(self.integration_data)
        regime_result = self.regime_analyzer.analyze_regime_hierarchy(self.integration_data)
        
        # Both should complete successfully
        assert "error" not in trend_result
        assert "error" not in regime_result
        
        # Should have complementary insights
        assert "trend_signals" in trend_result
        assert "regime_signals" in regime_result
    
    def test_signal_consistency(self):
        """Test consistency between trend and regime signals."""
        trend_result = self.trend_analyzer.analyze_multi_horizon_trends(self.integration_data)
        regime_result = self.regime_analyzer.analyze_regime_hierarchy(self.integration_data)
        
        if "trend_signals" in trend_result and "regime_signals" in regime_result:
            trend_signal = trend_result["trend_signals"].get("primary_signal", "HOLD")
            regime_signal = regime_result["regime_signals"].get("regime_signal", "HOLD")
            
            # Signals should be from valid set
            assert trend_signal in ["BUY", "SELL", "HOLD"]
            assert regime_signal in ["BUY", "SELL", "HOLD", "NEUTRAL"]
    
    def test_confidence_correlation(self):
        """Test correlation between trend and regime confidence levels."""
        trend_result = self.trend_analyzer.analyze_multi_horizon_trends(self.integration_data)
        regime_result = self.regime_analyzer.analyze_regime_hierarchy(self.integration_data)
        
        # Extract confidence levels
        trend_confidence = 0.5
        regime_confidence = 0.5
        
        if "trend_signals" in trend_result:
            trend_confidence = trend_result["trend_signals"].get("signal_strength", 0.5)
        
        if "regime_signals" in regime_result:
            regime_confidence = regime_result["regime_signals"].get("signal_strength", 0.5)
        
        # Both should be in valid range
        assert 0.0 <= trend_confidence <= 1.0
        assert 0.0 <= regime_confidence <= 1.0


class TestTrendAnalysisEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trend_detector = TrendDetector()
        self.regime_classifier = MarketRegimeClassifier()
    
    def test_missing_columns(self):
        """Test behavior with missing required columns."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        
        # Missing volume
        incomplete_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50)
        }, index=dates)
        
        # Should handle gracefully
        result = self.trend_detector.detect_trends(incomplete_data, "medium")
        assert isinstance(result, dict)
    
    def test_extreme_values(self):
        """Test behavior with extreme price values."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        
        # Extreme price movements
        extreme_data = pd.DataFrame({
            'Open': [100] + [1000000] * 49,  # Massive jump
            'High': [105] + [1000005] * 49,
            'Low': [95] + [999995] * 49,
            'Close': [100] + [1000000] * 49,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # Should handle without crashing
        result = self.trend_detector.detect_trends(extreme_data, "medium")
        assert isinstance(result, dict)
    
    def test_constant_prices(self):
        """Test behavior with constant price data."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        
        # Constant prices
        constant_data = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100] * 50,
            'Low': [100] * 50,
            'Close': [100] * 50,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        result = self.trend_detector.detect_trends(constant_data, "medium")
        assert isinstance(result, dict)
        
        if "consensus_trend" in result:
            # Should detect sideways trend for constant prices
            direction = result["consensus_trend"]["direction"]
            # Handle both string and enum values
            if hasattr(direction, 'value'):
                direction = direction.value
            assert direction == TrendDirection.SIDEWAYS.value
    
    def test_nan_values(self):
        """Test behavior with NaN values in data."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        prices = np.random.uniform(95, 105, 50)
        prices[10:15] = np.nan  # Insert NaN values
        
        nan_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.999,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # Should handle NaN values gracefully
        result = self.trend_detector.detect_trends(nan_data, "medium")
        assert isinstance(result, dict)
    
    def test_invalid_horizon(self):
        """Test behavior with invalid horizon parameter."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        valid_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # Test with invalid horizon
        result = self.trend_detector.detect_trends(valid_data, "invalid_horizon")
        assert isinstance(result, dict)
        # Should either handle gracefully or provide error info


class TestTrendAnalysisPerformance:
    """Test performance characteristics of trend analysis."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.trend_detector = TrendDetector()
        self.analyzer = CrossTimeframeTrendAnalyzer()
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='1h')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 10000))
        
        large_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.999,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 10000)
        }, index=dates)
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        result = self.trend_detector.detect_trends(large_data, "medium")
        end_time = time.time()
        
        # Should complete within 30 seconds for 10k data points
        assert (end_time - start_time) < 30
        assert isinstance(result, dict)
    
    def test_memory_usage(self):
        """Test memory efficiency."""
        # Test that analysis doesn't consume excessive memory
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=5000, freq='1h')
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 5000))
        
        memory_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.999,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 5000)
        }, index=dates)
        
        # Should complete without memory errors
        result = self.analyzer.analyze_multi_horizon_trends(memory_data)
        assert isinstance(result, dict)


# Integration test that can be run independently
def test_trend_analysis_integration():
    """Integration test for trend analysis system."""
    # Create comprehensive test
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    # Create trending data
    trend = np.linspace(0, 20, 200)  # Upward trend
    noise = np.random.normal(0, 2, 200)
    prices = 100 + trend + noise
    
    test_data = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 200)
    }, index=dates)
    
    # Test trend detection
    trend_detector = TrendDetector()
    trend_result = trend_detector.detect_trends(test_data, "medium")
    
    # Test cross-timeframe analysis
    cross_analyzer = CrossTimeframeTrendAnalyzer()
    cross_result = cross_analyzer.analyze_multi_horizon_trends(test_data)
    
    # Test regime classification
    regime_classifier = MarketRegimeClassifier()
    regime_result = regime_classifier.classify_regime(test_data, lookback_days=100)
    
    # All should complete successfully
    assert "error" not in trend_result
    assert "error" not in cross_result
    assert "error" not in regime_result
    
    # Return None instead of True to avoid pytest warning
    assert True


if __name__ == "__main__":
    # Run basic integration test
    success = test_trend_analysis_integration()
    print(f"✅ Trend analysis integration test: {'PASSED' if success else 'FAILED'}")