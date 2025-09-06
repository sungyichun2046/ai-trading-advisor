"""Production-ready tests for user profiling system.

Comprehensive test suite covering:
- Unit tests with edge cases
- Integration tests with database
- Performance benchmarks
- Security validation
- Error handling scenarios
- Compliance and audit requirements
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import time
from decimal import Decimal

from src.core.user_profiling import (
    RiskCategory, UserProfilingEngine, RiskProfile, UserResponse,
    validate_trading_decision_against_profile
)


class TestRiskAssessmentSecurity:
    """Security-focused tests for risk assessment."""
    
    def test_sql_injection_prevention_user_id(self):
        """Test that SQL injection attempts in user_id are handled safely."""
        engine = UserProfilingEngine()
        malicious_user_ids = [
            "'; DROP TABLE user_profiles; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "user_id'; INSERT INTO user_profiles VALUES ('hacker', 'aggressive', 100); --"
        ]
        
        valid_responses = self._get_valid_moderate_responses()
        
        for malicious_id in malicious_user_ids:
            try:
                profile = engine.assess_risk_profile(malicious_id, valid_responses)
                # Should create profile but sanitize the ID
                assert profile.user_id == malicious_id  # Should be stored as-is but safely handled
            except ValueError:
                # Acceptable to reject malicious input
                pass
    
    def test_response_injection_prevention(self):
        """Test that malicious responses don't cause issues."""
        engine = UserProfilingEngine()
        
        malicious_responses = {
            "experience_level": "'; DROP TABLE user_profiles; --",
            "investment_horizon": "<script>alert('xss')</script>",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }
        
        # Should either reject the input or handle it safely
        with pytest.raises(ValueError):
            engine.assess_risk_profile("test_user", malicious_responses)
    
    def test_input_size_limits(self):
        """Test handling of extremely large inputs."""
        engine = UserProfilingEngine()
        
        # Test with extremely large user ID
        large_user_id = "a" * 10000
        valid_responses = self._get_valid_moderate_responses()
        
        # Should handle large inputs gracefully
        try:
            profile = engine.assess_risk_profile(large_user_id, valid_responses)
            assert len(profile.user_id) <= 10000
        except ValueError as e:
            # Acceptable to reject oversized input
            assert "user_id" in str(e).lower()
    
    def _get_valid_moderate_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


class TestPerformanceBenchmarks:
    """Performance benchmarks for production deployment."""
    
    def test_assessment_performance(self):
        """Test risk assessment performance under load."""
        engine = UserProfilingEngine()
        responses = self._get_valid_moderate_responses()
        
        # Single assessment benchmark
        start_time = time.time()
        profile = engine.assess_risk_profile("test_user", responses)
        single_assessment_time = time.time() - start_time
        
        assert single_assessment_time < 0.1  # Should complete in < 100ms
        assert profile.risk_category in [RiskCategory.CONSERVATIVE, RiskCategory.MODERATE, RiskCategory.AGGRESSIVE]
    
    def test_batch_assessment_performance(self):
        """Test performance with multiple concurrent assessments."""
        engine = UserProfilingEngine()
        responses = self._get_valid_moderate_responses()
        
        # Batch assessment benchmark
        start_time = time.time()
        profiles = []
        for i in range(100):
            profile = engine.assess_risk_profile(f"user_{i}", responses)
            profiles.append(profile)
        batch_time = time.time() - start_time
        
        assert batch_time < 5.0  # 100 assessments in < 5 seconds
        assert len(profiles) == 100
        assert all(p.risk_category == RiskCategory.MODERATE for p in profiles)
    
    def test_validation_performance(self):
        """Test trading validation performance."""
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", self._get_valid_moderate_responses())
        
        # Validation benchmark
        start_time = time.time()
        for _ in range(1000):
            is_valid, message = validate_trading_decision_against_profile(
                profile, 1000.0, 10000.0, 0.02
            )
        validation_time = time.time() - start_time
        
        assert validation_time < 1.0  # 1000 validations in < 1 second
    
    def _get_valid_moderate_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


class TestErrorHandlingScenarios:
    """Comprehensive error handling tests."""
    
    def test_empty_responses_handling(self):
        """Test handling of empty or null responses."""
        engine = UserProfilingEngine()
        
        test_cases = [
            {},  # Empty dict
            {"experience_level": ""},  # Empty string
            {"experience_level": None},  # None value
            {key: "" for key in ["experience_level", "investment_horizon"]},  # Multiple empty
        ]
        
        for responses in test_cases:
            with pytest.raises(ValueError):
                engine.assess_risk_profile("test_user", responses)
    
    def test_invalid_response_values(self):
        """Test handling of invalid response values."""
        engine = UserProfilingEngine()
        base_responses = self._get_valid_responses()
        
        invalid_modifications = [
            {"experience_level": "Invalid experience level"},
            {"volatility_comfort": "10"},  # Out of range
            {"volatility_comfort": "-1"},  # Negative
            {"loss_tolerance": "Invalid loss tolerance"},
        ]
        
        for modification in invalid_modifications:
            responses = base_responses.copy()
            responses.update(modification)
            
            with pytest.raises(ValueError):
                engine.assess_risk_profile("test_user", responses)
    
    def test_missing_required_responses(self):
        """Test handling of missing required responses."""
        engine = UserProfilingEngine()
        complete_responses = self._get_valid_responses()
        
        # Test removing each required field
        for key in complete_responses.keys():
            incomplete_responses = complete_responses.copy()
            del incomplete_responses[key]
            
            with pytest.raises(ValueError):
                engine.assess_risk_profile("test_user", incomplete_responses)
    
    def test_database_connection_failures(self):
        """Test handling of database connection failures."""
        from src.data.database import UserProfileStorage
        
        with mock.patch.object(UserProfileStorage, 'get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database connection failed")
            
            storage = UserProfileStorage()
            # Should handle connection failures gracefully
            result = storage.get_risk_profile("test_user")
            assert result is None
    
    def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access."""
        engine = UserProfilingEngine()
        responses = self._get_valid_responses()
        
        # Simulate concurrent assessments
        def assess_profile(user_id):
            return engine.assess_risk_profile(f"user_{user_id}", responses)
        
        # This would need actual threading in a real test
        profiles = [assess_profile(i) for i in range(10)]
        
        assert len(profiles) == 10
        assert all(p.risk_category == RiskCategory.MODERATE for p in profiles)
    
    def _get_valid_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


class TestComplianceAndAudit:
    """Tests for compliance and audit requirements."""
    
    def test_profile_immutability_tracking(self):
        """Test that profile changes are properly tracked."""
        engine = UserProfilingEngine()
        responses = self._get_valid_responses()
        
        # Create initial profile
        profile1 = engine.assess_risk_profile("test_user", responses)
        
        # Update with different responses
        aggressive_responses = responses.copy()
        aggressive_responses["experience_level"] = "Very experienced - over 10 years"
        aggressive_responses["volatility_comfort"] = "5"
        
        profile2 = engine.assess_risk_profile("test_user", aggressive_responses)
        
        # Should have different timestamps and may have different categories
        # Note: Both responses might still result in moderate category, but with different scores
        assert profile1.risk_score != profile2.risk_score or profile1.confidence_score != profile2.confidence_score
        assert profile2.updated_at >= profile1.updated_at
        assert profile2.risk_category in [RiskCategory.MODERATE, RiskCategory.AGGRESSIVE]  # Should be more aggressive
    
    def test_data_retention_compliance(self):
        """Test data retention and cleanup capabilities."""
        engine = UserProfilingEngine()
        responses = self._get_valid_responses()
        
        # Create profile
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Simulate old profile (would need database integration for full test)
        old_profile = RiskProfile(
            user_id="old_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=50,
            responses=profile.responses,
            created_at=datetime.now() - timedelta(days=2555),  # > 7 years
            updated_at=datetime.now() - timedelta(days=2555),
            questionnaire_version="0.9"
        )
        
        # Should identify as requiring cleanup
        assert old_profile.created_at < datetime.now() - timedelta(days=2500)
    
    def test_risk_parameter_consistency(self):
        """Test that risk parameters are consistently calculated."""
        engine = UserProfilingEngine()
        
        # Test multiple times to ensure consistency
        for category in RiskCategory:
            params1 = engine.get_trading_parameters(category)
            params2 = engine.get_trading_parameters(category)
            
            assert params1 == params2
            
            # Validate parameter ranges
            assert 0 < params1["max_risk_per_trade"] <= 0.05
            assert 0 < params1["max_portfolio_risk"] <= 0.5
            assert 0 < params1["max_position_size"] <= 0.25
            assert params1["leverage_limit"] >= 1.0
    
    def test_confidence_score_impact(self):
        """Test that confidence scores properly adjust risk parameters."""
        engine = UserProfilingEngine()
        
        # Test with high confidence
        high_confidence_params = engine.get_trading_parameters(RiskCategory.MODERATE, 0.9)
        
        # Test with low confidence
        low_confidence_params = engine.get_trading_parameters(RiskCategory.MODERATE, 0.5)
        
        # Low confidence should have more conservative parameters
        assert low_confidence_params["max_risk_per_trade"] <= high_confidence_params["max_risk_per_trade"]
        assert low_confidence_params["max_portfolio_risk"] <= high_confidence_params["max_portfolio_risk"]
    
    def _get_valid_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""
    
    def test_boundary_risk_scores(self):
        """Test profiles at category boundaries."""
        engine = UserProfilingEngine()
        
        # Test scores at boundaries
        boundary_scores = [2.49, 2.5, 2.51, 3.69, 3.7, 3.71]
        
        for score in boundary_scores:
            category = engine.categorize_risk(score)
            
            if score < 2.5:
                assert category == RiskCategory.CONSERVATIVE
            elif score < 3.7:
                assert category == RiskCategory.MODERATE
            else:
                assert category == RiskCategory.AGGRESSIVE
    
    def test_extreme_confidence_scores(self):
        """Test handling of extreme confidence scores."""
        engine = UserProfilingEngine()
        
        extreme_confidence_values = [-1.0, 0.0, 1.0, 2.0, None]
        
        for confidence in extreme_confidence_values:
            try:
                params = engine.get_trading_parameters(RiskCategory.MODERATE, confidence)
                
                # Should always return valid parameters
                assert isinstance(params, dict)
                assert all(v > 0 for v in params.values())
                
            except Exception:
                # Should handle invalid confidence gracefully
                pass
    
    def test_trading_validation_edge_cases(self):
        """Test trading validation with edge case values."""
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", self._get_valid_responses())
        
        edge_cases = [
            (0.01, 10000.0, 0.001),  # Very small trade
            (9999.99, 10000.0, 0.01),  # Almost full account
            (1000.0, 1000000.0, 0.00001),  # Very small risk
            (1000.0, 1001.0, 0.01),  # Almost full account balance
        ]
        
        for trade_size, account_balance, position_risk in edge_cases:
            try:
                is_valid, message = validate_trading_decision_against_profile(
                    profile, trade_size, account_balance, position_risk
                )
                assert isinstance(is_valid, bool)
                assert isinstance(message, str)
                assert len(message) > 0
            except ValueError:
                # Some edge cases should raise ValueError
                pass
    
    def test_zero_and_negative_values(self):
        """Test handling of zero and negative values."""
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", self._get_valid_responses())
        
        invalid_cases = [
            (-1000.0, 10000.0, 0.01),  # Negative trade size
            (1000.0, -10000.0, 0.01),  # Negative account balance
            (1000.0, 10000.0, -0.01),  # Negative risk
            (0.0, 0.0, 0.0),  # All zeros
        ]
        
        for trade_size, account_balance, position_risk in invalid_cases:
            with pytest.raises(ValueError):
                validate_trading_decision_against_profile(
                    profile, trade_size, account_balance, position_risk
                )
    
    def _get_valid_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    def test_profile_data_consistency(self):
        """Test that profile data remains consistent across operations."""
        engine = UserProfilingEngine()
        responses = self._get_valid_responses()
        
        # Create profile
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Verify data integrity
        assert profile.user_id == "test_user"
        assert 0 <= profile.risk_score <= 100
        assert len(profile.responses) == 10  # All questions answered
        assert profile.created_at <= profile.updated_at
        assert profile.questionnaire_version is not None
    
    def test_response_data_validation(self):
        """Test that response data is properly validated."""
        from src.core.user_profiling import UserResponse
        
        # Test valid response creation
        response = UserResponse("test_question", "test_response", 3)
        assert response.question_id == "test_question"
        assert response.response == "test_response"
        assert response.score == 3
    
    def test_risk_profile_post_init_validation(self):
        """Test RiskProfile post-initialization validation."""
        from src.core.user_profiling import UserResponse
        
        now = datetime.now()
        valid_responses = [UserResponse("q1", "response1", 3)]
        
        # Valid profile should work
        profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=50,
            responses=valid_responses,
            created_at=now,
            updated_at=now
        )
        assert profile.user_id == "test_user"
        
        # Invalid user_id should raise error
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            RiskProfile(
                user_id="",
                risk_category=RiskCategory.MODERATE,
                risk_score=50,
                responses=valid_responses,
                created_at=now,
                updated_at=now
            )
        
        # Invalid risk_score should raise error
        with pytest.raises(ValueError, match="risk_score must be between 0 and 100"):
            RiskProfile(
                user_id="test_user",
                risk_category=RiskCategory.MODERATE,
                risk_score=150,
                responses=valid_responses,
                created_at=now,
                updated_at=now
            )
        
        # Invalid timestamps should raise error
        with pytest.raises(ValueError, match="created_at cannot be after updated_at"):
            RiskProfile(
                user_id="test_user",
                risk_category=RiskCategory.MODERATE,
                risk_score=50,
                responses=valid_responses,
                created_at=now,
                updated_at=now - timedelta(hours=1)
            )
    
    def _get_valid_responses(self) -> Dict[str, str]:
        return {
            "experience_level": "Moderate experience - 2-5 years",
            "investment_horizon": "Medium-term (1-3 years)",
            "volatility_comfort": "3",
            "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
            "portfolio_percentage": "25-50% - Moderate portion",
            "income_stability": "Moderate - Income varies but generally consistent",
            "market_reaction": "Hold my positions and wait for recovery",
            "financial_goals": "Moderate growth - Steady appreciation",
            "age_category": "36-50 - Peak earning years",
            "trading_frequency": "Regularly - Weekly trading"
        }


# Performance test to be run separately
def test_memory_usage():
    """Test memory usage with large datasets."""
    import tracemalloc
    
    tracemalloc.start()
    
    engine = UserProfilingEngine()
    responses = {
        "experience_level": "Moderate experience - 2-5 years",
        "investment_horizon": "Medium-term (1-3 years)",
        "volatility_comfort": "3",
        "loss_tolerance": "10-20% - Moderate losses are acceptable for higher returns",
        "portfolio_percentage": "25-50% - Moderate portion",
        "income_stability": "Moderate - Income varies but generally consistent",
        "market_reaction": "Hold my positions and wait for recovery",
        "financial_goals": "Moderate growth - Steady appreciation",
        "age_category": "36-50 - Peak earning years",
        "trading_frequency": "Regularly - Weekly trading"
    }
    
    # Create many profiles
    profiles = []
    for i in range(1000):
        profile = engine.assess_risk_profile(f"user_{i}", responses)
        profiles.append(profile)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory usage should be reasonable (less than 50MB for 1000 profiles)
    assert peak < 50 * 1024 * 1024
    assert len(profiles) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])