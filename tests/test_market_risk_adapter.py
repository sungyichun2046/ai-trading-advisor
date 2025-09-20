"""Tests for market risk adapter functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Mock heavy dependencies before imports
mock_modules = {
    'yfinance': Mock(),
    'newsapi': Mock(),
    'transformers': Mock(),
    'torch': Mock(),
    'textblob': Mock(),
    'pandas': Mock()
}

for module_name, mock_obj in mock_modules.items():
    sys.modules[module_name] = mock_obj

from src.core.market_risk_adapter import (
    MarketRiskAdapter, MarketRegime, VolatilityLevel
)
from src.config import settings


class TestMarketRiskAdapter:
    """Test MarketRiskAdapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = MarketRiskAdapter()

    def test_init(self):
        """Test MarketRiskAdapter initialization."""
        assert self.adapter.volatility_monitor is not None
        assert self.adapter.risk_engine is not None
        assert isinstance(self.adapter.vix_thresholds, dict)
        assert isinstance(self.adapter.risk_multipliers, dict)
        assert isinstance(self.adapter.regime_multipliers, dict)

    def test_vix_thresholds_configuration(self):
        """Test VIX threshold configuration."""
        thresholds = self.adapter.vix_thresholds
        assert thresholds["low"] < thresholds["moderate"]
        assert thresholds["moderate"] < thresholds["elevated"]
        assert thresholds["elevated"] < thresholds["high"]
        assert thresholds["high"] < thresholds["extreme"]
        
        # Verify reasonable threshold values
        assert 10 <= thresholds["low"] <= 20
        assert 35 <= thresholds["extreme"] <= 50

    def test_risk_multipliers_configuration(self):
        """Test risk multiplier configuration."""
        multipliers = self.adapter.risk_multipliers
        
        # Verify all volatility levels have multipliers
        for level in VolatilityLevel:
            assert level in multipliers
            assert isinstance(multipliers[level], (int, float))
            assert 0.1 <= multipliers[level] <= 2.0
        
        # Verify multipliers decrease with higher volatility
        assert multipliers[VolatilityLevel.LOW] >= multipliers[VolatilityLevel.MODERATE]
        assert multipliers[VolatilityLevel.MODERATE] >= multipliers[VolatilityLevel.HIGH]
        assert multipliers[VolatilityLevel.HIGH] >= multipliers[VolatilityLevel.EXTREME]

    def test_classify_volatility_level(self):
        """Test VIX-based volatility level classification."""
        # Test low volatility
        assert self.adapter.classify_volatility_level(12.0) == VolatilityLevel.LOW
        
        # Test moderate volatility
        assert self.adapter.classify_volatility_level(18.0) == VolatilityLevel.MODERATE
        
        # Test elevated volatility
        assert self.adapter.classify_volatility_level(23.0) == VolatilityLevel.ELEVATED
        
        # Test high volatility
        assert self.adapter.classify_volatility_level(35.0) == VolatilityLevel.HIGH
        
        # Test extreme volatility
        assert self.adapter.classify_volatility_level(45.0) == VolatilityLevel.EXTREME

    def test_detect_market_regime_crisis(self):
        """Test crisis regime detection."""
        crisis_data = {
            "vix_current": 55.0,
            "alerts": ["EXTREME_VIX", "SPY_VOLUME_SPIKE"],
            "volatility_level": "EXTREME"
        }
        
        regime = self.adapter.detect_market_regime(crisis_data)
        assert regime == MarketRegime.CRISIS

    def test_detect_market_regime_bear_high_vol(self):
        """Test bear high volatility regime detection."""
        bear_data = {
            "vix_current": 35.0,
            "alerts": ["SPY_PRICE_MOVEMENT", "QQQ_VOLUME_SPIKE"],
            "volatility_level": "HIGH"
        }
        
        regime = self.adapter.detect_market_regime(bear_data)
        assert regime == MarketRegime.BEAR_HIGH_VOL

    def test_detect_market_regime_bull_low_vol(self):
        """Test bull low volatility regime detection."""
        bull_data = {
            "vix_current": 16.0,
            "alerts": [],
            "volatility_level": "LOW"
        }
        
        regime = self.adapter.detect_market_regime(bull_data)
        assert regime == MarketRegime.BULL_LOW_VOL

    def test_calculate_risk_multiplier_normal_conditions(self):
        """Test risk multiplier calculation under normal conditions."""
        normal_data = {
            "vix_current": 20.0,
            "alerts": [],
            "volatility_level": "MODERATE"
        }
        
        multiplier = self.adapter.calculate_risk_multiplier(normal_data)
        assert isinstance(multiplier, float)
        assert 0.8 <= multiplier <= 1.2  # Normal range

    def test_calculate_risk_multiplier_high_volatility(self):
        """Test risk multiplier calculation during high volatility."""
        high_vol_data = {
            "vix_current": 35.0,
            "alerts": ["HIGH_VIX", "SPY_VOLUME_SPIKE"],
            "volatility_level": "HIGH"
        }
        
        multiplier = self.adapter.calculate_risk_multiplier(high_vol_data)
        assert multiplier < 0.8  # Should reduce risk

    def test_calculate_risk_multiplier_circuit_breaker(self):
        """Test circuit breaker activation."""
        extreme_data = {
            "vix_current": 55.0,
            "alerts": ["EXTREME_VIX", "SPY_VOLUME_SPIKE", "QQQ_PRICE_MOVEMENT"],
            "volatility_level": "EXTREME",
            "triggers": ["emergency_analysis"]
        }
        
        multiplier = self.adapter.calculate_risk_multiplier(extreme_data)
        assert multiplier <= 0.1  # Circuit breaker should activate

    def test_should_apply_circuit_breaker(self):
        """Test circuit breaker conditions."""
        # Test VIX circuit breaker
        vix_extreme = {"vix_current": 55.0, "alerts": [], "triggers": []}
        assert self.adapter._should_apply_circuit_breaker(vix_extreme) is True
        
        # Test multiple extreme alerts
        multiple_alerts = {
            "vix_current": 25.0,
            "alerts": ["EXTREME_VIX", "EXTREME_VOLUME"],
            "triggers": []
        }
        assert self.adapter._should_apply_circuit_breaker(multiple_alerts) is True
        
        # Test emergency triggers
        emergency = {
            "vix_current": 25.0,
            "alerts": [],
            "triggers": ["emergency_analysis"]
        }
        assert self.adapter._should_apply_circuit_breaker(emergency) is True
        
        # Test normal conditions
        normal = {"vix_current": 20.0, "alerts": [], "triggers": []}
        assert self.adapter._should_apply_circuit_breaker(normal) is False

    @patch('src.core.market_risk_adapter.VolatilityMonitor')
    def test_get_current_market_conditions_success(self, mock_volatility_monitor):
        """Test successful market conditions retrieval."""
        mock_monitor = Mock()
        mock_volatility_monitor.return_value = mock_monitor
        
        expected_data = {
            "status": "success",
            "vix_current": 22.5,
            "volatility_level": "MODERATE",
            "alerts": ["SPY_VOLUME_SPIKE"],
            "triggers": []
        }
        mock_monitor.check_market_volatility.return_value = expected_data
        
        adapter = MarketRiskAdapter()
        result = adapter.get_current_market_conditions()
        
        assert result == expected_data

    @patch('src.core.market_risk_adapter.VolatilityMonitor')
    def test_get_current_market_conditions_failure(self, mock_volatility_monitor):
        """Test market conditions retrieval failure."""
        mock_monitor = Mock()
        mock_volatility_monitor.return_value = mock_monitor
        mock_monitor.check_market_volatility.side_effect = Exception("API Error")
        
        adapter = MarketRiskAdapter()
        result = adapter.get_current_market_conditions()
        
        # Should return default conditions
        assert result["status"] == "success"
        assert result["data_source"] == "default"
        assert "vix_current" in result

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_adjust_position_size_normal_conditions(self, mock_market_conditions):
        """Test position size adjustment under normal conditions."""
        mock_market_conditions.return_value = {
            "status": "success",
            "vix_current": 20.0,
            "volatility_level": "MODERATE",
            "alerts": [],
            "triggers": []
        }
        
        base_size = 10000.0
        account_balance = 100000.0
        
        result = self.adapter.adjust_position_size(base_size, account_balance, "AAPL")
        
        assert "original_position_size" in result
        assert "adjusted_position_size" in result
        assert "risk_multiplier" in result
        assert "volatility_level" in result
        assert "market_regime" in result
        assert result["original_position_size"] == base_size
        assert isinstance(result["adjusted_position_size"], float)
        assert 0.1 <= result["risk_multiplier"] <= 2.0

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_adjust_position_size_high_volatility(self, mock_market_conditions):
        """Test position size adjustment during high volatility."""
        mock_market_conditions.return_value = {
            "status": "success",
            "vix_current": 35.0,
            "volatility_level": "HIGH",
            "alerts": ["HIGH_VIX", "SPY_VOLUME_SPIKE"],
            "triggers": []
        }
        
        base_size = 10000.0
        account_balance = 100000.0
        
        result = self.adapter.adjust_position_size(base_size, account_balance, "SPY")
        
        # Should reduce position size during high volatility
        assert result["adjusted_position_size"] < result["original_position_size"]
        assert result["risk_multiplier"] < 1.0
        assert result["volatility_level"] == "HIGH"

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_adjust_position_size_circuit_breaker(self, mock_market_conditions):
        """Test position size adjustment with circuit breaker."""
        mock_market_conditions.return_value = {
            "status": "success",
            "vix_current": 55.0,
            "volatility_level": "EXTREME",
            "alerts": ["EXTREME_VIX", "SPY_VOLUME_SPIKE"],
            "triggers": ["emergency_analysis"]
        }
        
        base_size = 10000.0
        account_balance = 100000.0
        
        result = self.adapter.adjust_position_size(base_size, account_balance, "SPY")
        
        # Circuit breaker should drastically reduce position
        assert result["adjusted_position_size"] <= base_size * 0.2
        assert result["circuit_breaker_active"] is True
        assert result["risk_multiplier"] <= 0.1

    def test_get_risk_parameters_for_regime(self):
        """Test risk parameter adjustment for different regimes."""
        # Test crisis regime
        crisis_params = self.adapter.get_risk_parameters_for_regime(MarketRegime.CRISIS)
        assert all(param in crisis_params for param in [
            "max_risk_per_trade", "max_portfolio_risk", 
            "max_position_size", "daily_loss_limit"
        ])
        
        # Crisis should have very conservative parameters
        assert crisis_params["max_risk_per_trade"] < settings.max_risk_per_trade
        assert crisis_params["daily_loss_limit"] < settings.daily_loss_limit
        
        # Test bull low vol regime
        bull_params = self.adapter.get_risk_parameters_for_regime(MarketRegime.BULL_LOW_VOL)
        assert bull_params["max_risk_per_trade"] >= crisis_params["max_risk_per_trade"]

    @patch.object(MarketRiskAdapter, 'adjust_position_size')
    def test_validate_trade_against_market_conditions_approved(self, mock_adjust):
        """Test trade validation - approved trade."""
        mock_adjust.return_value = {
            "adjusted_position_size": 9500.0,
            "risk_multiplier": 0.95,
            "vix_current": 18.0,
            "volatility_level": "MODERATE",
            "market_regime": "BULL_LOW_VOL"
        }
        
        result = self.adapter.validate_trade_against_market_conditions(
            trade_size=10000.0,
            account_balance=100000.0,
            symbol="AAPL"
        )
        
        assert result["approval_status"] == "approved"
        assert result["proposed_size"] == 10000.0
        assert result["recommended_size"] == 9500.0

    @patch.object(MarketRiskAdapter, 'adjust_position_size')
    def test_validate_trade_against_market_conditions_rejected(self, mock_adjust):
        """Test trade validation - rejected trade."""
        mock_adjust.return_value = {
            "adjusted_position_size": 3000.0,  # Much smaller recommended size
            "risk_multiplier": 0.3,
            "vix_current": 45.0,
            "volatility_level": "EXTREME",
            "market_regime": "CRISIS"
        }
        
        result = self.adapter.validate_trade_against_market_conditions(
            trade_size=10000.0,
            account_balance=100000.0,
            symbol="SPY"
        )
        
        assert result["approval_status"] == "rejected"
        assert "too large" in result["message"].lower()

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_get_market_stress_score_low_stress(self, mock_market_conditions):
        """Test market stress score calculation - low stress."""
        mock_market_conditions.return_value = {
            "vix_current": 15.0,
            "alerts": [],
            "volatility_level": "LOW"
        }
        
        result = self.adapter.get_market_stress_score()
        
        assert "stress_score" in result
        assert "stress_level" in result
        assert result["stress_score"] <= 30  # Should be low stress
        assert result["stress_level"] in ["MINIMAL", "LOW"]

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_get_market_stress_score_high_stress(self, mock_market_conditions):
        """Test market stress score calculation - high stress."""
        mock_market_conditions.return_value = {
            "vix_current": 40.0,
            "alerts": ["EXTREME_VIX", "SPY_VOLUME_SPIKE", "QQQ_PRICE_MOVEMENT"],
            "volatility_level": "EXTREME"
        }
        
        result = self.adapter.get_market_stress_score()
        
        assert result["stress_score"] >= 70  # Should be high stress
        assert result["stress_level"] in ["HIGH", "EXTREME"]
        assert result["vix_contribution"] > 50
        assert result["alert_contribution"] > 0

    def test_get_adjustment_reason(self):
        """Test adjustment reason generation."""
        # Test circuit breaker reason
        extreme_data = {
            "vix_current": 55.0,
            "volatility_level": "EXTREME",
            "alerts": ["EXTREME_VIX"]
        }
        reason = self.adapter._get_adjustment_reason(extreme_data, 0.1)
        assert "circuit breaker" in reason.lower()
        
        # Test risk reduction reason
        high_vol_data = {
            "vix_current": 35.0,
            "volatility_level": "HIGH",
            "alerts": []
        }
        reason = self.adapter._get_adjustment_reason(high_vol_data, 0.6)
        assert "risk reduced" in reason.lower()
        
        # Test risk increase reason
        low_vol_data = {
            "vix_current": 12.0,
            "volatility_level": "LOW",
            "alerts": []
        }
        reason = self.adapter._get_adjustment_reason(low_vol_data, 1.3)
        assert "risk increased" in reason.lower()


class TestVolatilityLevelEnum:
    """Test VolatilityLevel enum."""

    def test_volatility_levels_exist(self):
        """Test that all expected volatility levels exist."""
        expected_levels = ["LOW", "MODERATE", "ELEVATED", "HIGH", "EXTREME"]
        
        for level in expected_levels:
            assert hasattr(VolatilityLevel, level)
            assert VolatilityLevel[level].value == level


class TestMarketRegimeEnum:
    """Test MarketRegime enum."""

    def test_market_regimes_exist(self):
        """Test that all expected market regimes exist."""
        expected_regimes = [
            "BULL_LOW_VOL", "BULL_HIGH_VOL", "BEAR_LOW_VOL", 
            "BEAR_HIGH_VOL", "SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "CRISIS"
        ]
        
        for regime in expected_regimes:
            assert hasattr(MarketRegime, regime)


class TestIntegration:
    """Integration tests for market risk adapter."""

    @patch('src.core.market_risk_adapter.VolatilityMonitor')
    def test_end_to_end_risk_adjustment(self, mock_volatility_monitor):
        """Test end-to-end risk adjustment workflow."""
        # Mock volatility monitor
        mock_monitor = Mock()
        mock_volatility_monitor.return_value = mock_monitor
        mock_monitor.check_market_volatility.return_value = {
            "status": "success",
            "vix_current": 25.0,
            "volatility_level": "ELEVATED",
            "alerts": ["SPY_VOLUME_SPIKE"],
            "triggers": []
        }
        
        adapter = MarketRiskAdapter()
        
        # Test complete workflow
        base_position = 15000.0
        account_balance = 100000.0
        
        # Step 1: Get market conditions
        conditions = adapter.get_current_market_conditions()
        assert conditions["status"] == "success"
        
        # Step 2: Adjust position size
        adjustment = adapter.adjust_position_size(base_position, account_balance)
        assert "adjusted_position_size" in adjustment
        
        # Step 3: Validate trade
        validation = adapter.validate_trade_against_market_conditions(
            base_position, account_balance
        )
        assert "approval_status" in validation
        
        # Step 4: Get stress score
        stress = adapter.get_market_stress_score()
        assert "stress_score" in stress

    def test_risk_parameter_consistency(self):
        """Test consistency across risk parameters."""
        adapter = MarketRiskAdapter()
        
        # Test all regimes have valid multipliers
        for regime in MarketRegime:
            params = adapter.get_risk_parameters_for_regime(regime)
            
            # All parameters should be positive
            for param_name, param_value in params.items():
                assert param_value > 0, f"{param_name} should be positive for {regime}"
                
            # Risk parameters should be reasonable
            assert params["max_risk_per_trade"] <= 0.1  # Max 10% per trade
            assert params["max_portfolio_risk"] <= 1.0   # Max 100% portfolio risk
            assert params["daily_loss_limit"] <= 0.2     # Max 20% daily loss

    @patch.object(MarketRiskAdapter, 'get_current_market_conditions')
    def test_position_size_bounds_checking(self, mock_market_conditions):
        """Test that position sizes stay within reasonable bounds."""
        mock_market_conditions.return_value = {
            "status": "success",
            "vix_current": 20.0,
            "volatility_level": "MODERATE",
            "alerts": [],
            "triggers": []
        }
        
        adapter = MarketRiskAdapter()
        account_balance = 50000.0
        
        # Test various base position sizes
        test_sizes = [1000, 5000, 10000, 25000, 50000]
        
        for base_size in test_sizes:
            result = adapter.adjust_position_size(base_size, account_balance)
            
            # Adjusted size should be positive
            assert result["adjusted_position_size"] > 0
            
            # Should not exceed account balance
            assert result["adjusted_position_size"] <= account_balance
            
            # Risk percentage should be reasonable
            assert result["position_risk_percentage"] <= 1.0


class TestErrorHandling:
    """Test error handling in market risk adapter."""

    @patch('src.core.market_risk_adapter.VolatilityMonitor')
    def test_volatility_monitor_failure_handling(self, mock_volatility_monitor):
        """Test handling of volatility monitor failures."""
        mock_monitor = Mock()
        mock_volatility_monitor.return_value = mock_monitor
        mock_monitor.check_market_volatility.side_effect = Exception("Monitor failed")
        
        adapter = MarketRiskAdapter()
        
        # Should not crash and should return default conditions
        conditions = adapter.get_current_market_conditions()
        assert conditions["status"] == "success"
        assert conditions["data_source"] == "default"

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        adapter = MarketRiskAdapter()
        
        # Test with negative position size
        result = adapter.adjust_position_size(-1000.0, 50000.0)
        assert result["adjusted_position_size"] >= 0
        
        # Test with zero account balance
        result = adapter.adjust_position_size(1000.0, 0.0)
        assert result["adjusted_position_size"] >= 0

    def test_missing_market_data_fields(self):
        """Test handling of missing market data fields."""
        adapter = MarketRiskAdapter()
        
        # Test with minimal market data
        incomplete_data = {"status": "success"}
        
        # Should not crash when calculating risk multiplier
        multiplier = adapter.calculate_risk_multiplier(incomplete_data)
        assert isinstance(multiplier, float)
        assert 0.1 <= multiplier <= 2.0
        
        # Should not crash when detecting regime
        regime = adapter.detect_market_regime(incomplete_data)
        assert isinstance(regime, MarketRegime)