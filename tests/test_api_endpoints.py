"""Live API endpoint tests for risk profiling system.

Tests the actual running API server with real database integration.
These tests verify the complete request/response cycle including:
- Database operations
- Business logic validation
- Error handling
- Performance characteristics
"""

import pytest
import requests
import json
import time
from typing import Dict


class TestLiveAPIEndpoints:
    """Tests for live API endpoints with real database."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.base_url = "http://localhost:8000/api/v1/risk-profile"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.session.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "risk-profiling"
        assert data["questionnaire_questions"] == 10
        assert len(data["available_categories"]) == 3
        assert "timestamp" in data
    
    def test_questionnaire_endpoint(self):
        """Test questionnaire retrieval."""
        response = self.session.get(f"{self.base_url}/questionnaire")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "questions" in data
        assert "version" in data
        assert "total_questions" in data
        assert data["total_questions"] == 10
        assert len(data["questions"]) == 10
        
        # Validate first question structure
        question = data["questions"][0]
        required_fields = ["id", "question", "type", "options", "weight", "category"]
        for field in required_fields:
            assert field in question
        
        assert len(question["options"]) > 0
        assert question["weight"] > 0
    
    def test_categories_endpoint(self):
        """Test categories endpoint."""
        response = self.session.get(f"{self.base_url}/categories")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_categories = ["conservative", "moderate", "aggressive"]
        for category in expected_categories:
            assert category in data
            assert isinstance(data[category], str)
            assert len(data[category]) > 20  # Meaningful description
    
    def test_parameters_endpoints(self):
        """Test parameters for each risk category."""
        categories = ["conservative", "moderate", "aggressive"]
        required_params = [
            "max_risk_per_trade", "max_portfolio_risk", "max_position_size",
            "daily_loss_limit", "leverage_limit", "volatility_threshold",
            "holding_period_days"
        ]
        
        for category in categories:
            response = self.session.get(f"{self.base_url}/parameters/{category}")
            
            assert response.status_code == 200
            data = response.json()
            
            for param in required_params:
                assert param in data
                assert isinstance(data[param], (int, float))
                assert data[param] > 0
        
        # Test invalid category
        response = self.session.get(f"{self.base_url}/parameters/invalid")
        assert response.status_code == 400
    
    def test_assessment_conservative_profile(self):
        """Test assessment creating conservative profile."""
        assessment_data = {
            "user_id": "test_conservative_api",
            "responses": {
                "experience_level": "No experience - I'm just starting",
                "investment_horizon": "Short-term (less than 1 year)",
                "volatility_comfort": "1",
                "loss_tolerance": "0-5% - I cannot tolerate significant losses",
                "portfolio_percentage": "Over 75% - Most of my portfolio",
                "income_stability": "Unstable - Income is very unpredictable",
                "market_reaction": "Sell everything immediately to prevent further losses",
                "financial_goals": "Capital preservation - Protect my money",
                "age_category": "Over 65 - Retirement/preservation focus",
                "trading_frequency": "Rarely - Long-term buy and hold"
            }
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_conservative_api"
        assert data["risk_category"] == "conservative"
        assert 0 <= data["risk_score"] <= 50  # Should be low
        assert "assessment_date" in data
        assert "trading_parameters" in data
        assert "confidence_score" in data
        assert "review_recommended" in data
        
        # Validate conservative trading parameters
        params = data["trading_parameters"]
        assert params["max_risk_per_trade"] <= 0.01  # 1% or less
        assert params["leverage_limit"] == 1.0  # No leverage
    
    def test_assessment_moderate_profile(self):
        """Test assessment creating moderate profile."""
        assessment_data = {
            "user_id": "test_moderate_api",
            "responses": {
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
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_moderate_api"
        assert data["risk_category"] == "moderate"
        assert 30 <= data["risk_score"] <= 80  # Should be medium
        
        # Validate moderate trading parameters
        params = data["trading_parameters"]
        assert 0.01 < params["max_risk_per_trade"] <= 0.02
        assert 1.0 < params["leverage_limit"] <= 1.5
    
    def test_assessment_aggressive_profile(self):
        """Test assessment creating aggressive profile."""
        assessment_data = {
            "user_id": "test_aggressive_api",
            "responses": {
                "experience_level": "Very experienced - over 10 years",
                "investment_horizon": "Very long-term (over 7 years)",
                "volatility_comfort": "5",
                "loss_tolerance": "30%+ - I can handle very large losses",
                "portfolio_percentage": "Less than 10% - Very small portion",
                "income_stability": "Very stable - Regular salary with job security",
                "market_reaction": "Significantly increase my investment",
                "financial_goals": "Aggressive growth - Maximum returns",
                "age_category": "25-35 - Building wealth phase",
                "trading_frequency": "Frequently - Daily trading"
            }
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_aggressive_api"
        assert data["risk_category"] == "aggressive"
        assert data["risk_score"] >= 70  # Should be high
        
        # Validate aggressive trading parameters
        params = data["trading_parameters"]
        assert params["max_risk_per_trade"] > 0.02
        assert params["leverage_limit"] > 1.5
    
    def test_assessment_validation_errors(self):
        """Test assessment endpoint validation."""
        # Test empty user_id
        response = self.session.post(f"{self.base_url}/assess", json={
            "user_id": "",
            "responses": self._get_valid_responses()
        })
        assert response.status_code == 422
        
        # Test empty responses
        response = self.session.post(f"{self.base_url}/assess", json={
            "user_id": "test_user",
            "responses": {}
        })
        assert response.status_code == 422
        
        # Test missing user_id
        response = self.session.post(f"{self.base_url}/assess", json={
            "responses": self._get_valid_responses()
        })
        assert response.status_code == 422
    
    def test_assessment_invalid_responses(self):
        """Test assessment with invalid response values."""
        invalid_responses = self._get_valid_responses()
        invalid_responses["experience_level"] = "Invalid experience level"
        
        assessment_data = {
            "user_id": "test_invalid",
            "responses": invalid_responses
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        assert response.status_code == 400
        data = response.json()
        # Check if error message is in 'detail' or 'message' field
        error_message = data.get("detail", data.get("message", ""))
        assert "Invalid" in error_message
    
    def test_get_profile_after_assessment(self):
        """Test retrieving profile after assessment."""
        # First create a profile
        assessment_data = {
            "user_id": "test_get_profile_api",
            "responses": self._get_valid_responses()
        }
        
        assess_response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        assert assess_response.status_code == 200
        
        # Then retrieve it
        get_response = self.session.get(f"{self.base_url}/profile/test_get_profile_api")
        assert get_response.status_code == 200
        
        assess_data = assess_response.json()
        get_data = get_response.json()
        
        # Should match the assessment data
        assert assess_data["user_id"] == get_data["user_id"]
        assert assess_data["risk_category"] == get_data["risk_category"]
        assert assess_data["risk_score"] == get_data["risk_score"]
        # Confidence score should match if both have it, or be acceptable if one is missing
        if assess_data.get("confidence_score") is not None and get_data.get("confidence_score") is not None:
            assert assess_data["confidence_score"] == get_data["confidence_score"]
    
    def test_get_nonexistent_profile(self):
        """Test retrieving non-existent profile."""
        response = self.session.get(f"{self.base_url}/profile/nonexistent_user_api")
        assert response.status_code == 404
        data = response.json()
        assert "Risk profile not found" in data["message"]
    
    def test_update_profile(self):
        """Test updating existing profile."""
        user_id = "test_update_api"
        
        # Create initial conservative profile
        initial_assessment = {
            "user_id": user_id,
            "responses": {
                "experience_level": "No experience - I'm just starting",
                "investment_horizon": "Short-term (less than 1 year)",
                "volatility_comfort": "1",
                "loss_tolerance": "0-5% - I cannot tolerate significant losses",
                "portfolio_percentage": "Over 75% - Most of my portfolio",
                "income_stability": "Unstable - Income is very unpredictable",
                "market_reaction": "Sell everything immediately to prevent further losses",
                "financial_goals": "Capital preservation - Protect my money",
                "age_category": "Over 65 - Retirement/preservation focus",
                "trading_frequency": "Rarely - Long-term buy and hold"
            }
        }
        
        initial_response = self.session.post(f"{self.base_url}/assess", json=initial_assessment)
        assert initial_response.status_code == 200
        initial_data = initial_response.json()
        assert initial_data["risk_category"] == "conservative"
        
        # Update to aggressive profile
        update_data = {
            "responses": {
                "experience_level": "Very experienced - over 10 years",
                "investment_horizon": "Very long-term (over 7 years)",
                "volatility_comfort": "5",
                "loss_tolerance": "30%+ - I can handle very large losses",
                "portfolio_percentage": "Less than 10% - Very small portion",
                "income_stability": "Very stable - Regular salary with job security",
                "market_reaction": "Significantly increase my investment",
                "financial_goals": "Aggressive growth - Maximum returns",
                "age_category": "25-35 - Building wealth phase",
                "trading_frequency": "Frequently - Daily trading"
            }
        }
        
        update_response = self.session.put(f"{self.base_url}/update/{user_id}", json=update_data)
        assert update_response.status_code == 200
        update_result = update_response.json()
        
        assert update_result["user_id"] == user_id
        assert update_result["risk_category"] == "aggressive"
        assert update_result["risk_score"] > initial_data["risk_score"]
    
    def test_trade_validation_flow(self):
        """Test complete trade validation flow."""
        user_id = "test_trade_validation_api"
        
        # Create moderate profile
        assessment_data = {
            "user_id": user_id,
            "responses": self._get_valid_responses()
        }
        
        assess_response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        assert assess_response.status_code == 200
        profile_data = assess_response.json()
        assert profile_data["risk_category"] == "moderate"
        
        # Test valid trade
        valid_trade = {
            "user_id": user_id,
            "trade_size": 800.0,  # 8% of account
            "account_balance": 10000.0,
            "position_risk": 0.015  # 1.5%
        }
        
        valid_response = self.session.post(f"{self.base_url}/validate-trade", json=valid_trade)
        assert valid_response.status_code == 200
        valid_result = valid_response.json()
        
        assert valid_result["approved"] is True
        assert "approved" in valid_result["message"].lower()
        assert valid_result["user_risk_category"] == "moderate"
        assert "risk_parameters" in valid_result
        
        # Test invalid trade (too large position)
        invalid_trade = {
            "user_id": user_id,
            "trade_size": 1500.0,  # 15% of account (exceeds 10% limit for moderate)
            "account_balance": 10000.0,
            "position_risk": 0.015
        }
        
        invalid_response = self.session.post(f"{self.base_url}/validate-trade", json=invalid_trade)
        assert invalid_response.status_code == 200
        invalid_result = invalid_response.json()
        
        assert invalid_result["approved"] is False
        assert "exceeds limit" in invalid_result["message"].lower()
        assert invalid_result["user_risk_category"] == "moderate"
        
        # Test invalid trade (too much risk)
        high_risk_trade = {
            "user_id": user_id,
            "trade_size": 500.0,
            "account_balance": 10000.0,
            "position_risk": 0.03  # 3% risk (exceeds 2% limit for moderate)
        }
        
        high_risk_response = self.session.post(f"{self.base_url}/validate-trade", json=high_risk_trade)
        assert high_risk_response.status_code == 200
        high_risk_result = high_risk_response.json()
        
        assert high_risk_result["approved"] is False
        assert "risk" in high_risk_result["message"].lower()
        assert "exceeds limit" in high_risk_result["message"].lower()
    
    def test_trade_validation_no_profile(self):
        """Test trade validation without profile."""
        trade_data = {
            "user_id": "nonexistent_user_api",
            "trade_size": 1000.0,
            "account_balance": 10000.0,
            "position_risk": 0.02
        }
        
        response = self.session.post(f"{self.base_url}/validate-trade", json=trade_data)
        assert response.status_code == 404
        data = response.json()
        assert "Risk profile not found" in data["message"]
    
    def test_trade_validation_parameter_validation(self):
        """Test trade validation parameter validation."""
        user_id = "test_param_validation"
        
        # Create profile first
        assessment_data = {
            "user_id": user_id,
            "responses": self._get_valid_responses()
        }
        self.session.post(f"{self.base_url}/assess", json=assessment_data)
        
        # Test negative trade size
        negative_trade = {
            "user_id": user_id,
            "trade_size": -1000.0,
            "account_balance": 10000.0,
            "position_risk": 0.02
        }
        
        response = self.session.post(f"{self.base_url}/validate-trade", json=negative_trade)
        assert response.status_code == 422  # Pydantic validation error
        
        # Test negative account balance
        negative_balance = {
            "user_id": user_id,
            "trade_size": 1000.0,
            "account_balance": -10000.0,
            "position_risk": 0.02
        }
        
        response = self.session.post(f"{self.base_url}/validate-trade", json=negative_balance)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_performance_assessment(self):
        """Test assessment performance under load."""
        assessment_data = {
            "user_id": "performance_test_user",
            "responses": self._get_valid_responses()
        }
        
        # Measure single assessment time
        start_time = time.time()
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should complete in less than 1 second
    
    def test_confidence_score_impact(self):
        """Test that confidence scores are calculated and impact recommendations."""
        # Create profile with consistent responses (high confidence)
        consistent_responses = {
            "experience_level": "Very experienced - over 10 years",
            "investment_horizon": "Very long-term (over 7 years)",
            "volatility_comfort": "5",
            "loss_tolerance": "30%+ - I can handle very large losses",
            "portfolio_percentage": "Less than 10% - Very small portion",
            "income_stability": "Very stable - Regular salary with job security",
            "market_reaction": "Significantly increase my investment",
            "financial_goals": "Aggressive growth - Maximum returns",
            "age_category": "25-35 - Building wealth phase",
            "trading_frequency": "Frequently - Daily trading"
        }
        
        assessment_data = {
            "user_id": "high_confidence_user",
            "responses": consistent_responses
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "confidence_score" in data
        assert data["confidence_score"] > 0.7  # Should have high confidence
        assert data["review_recommended"] is False  # High confidence shouldn't need review
    
    def _get_valid_responses(self) -> Dict[str, str]:
        """Get valid moderate responses for testing."""
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


class TestAPIErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.base_url = "http://localhost:8000/api/v1/risk-profile"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = self.session.post(
            f"{self.base_url}/assess",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test handling of missing content-type header."""
        assessment_data = {
            "user_id": "test_user",
            "responses": self._get_valid_responses()
        }
        
        response = self.session.post(
            f"{self.base_url}/assess",
            data=json.dumps(assessment_data)
            # Note: No Content-Type header
        )
        # Should still work with FastAPI's automatic detection, or hit rate limit
        assert response.status_code in [200, 422, 429]
    
    def test_large_payload(self):
        """Test handling of very large payloads."""
        large_responses = self._get_valid_responses()
        large_responses["experience_level"] = "A" * 10000  # Very long response
        
        assessment_data = {
            "user_id": "large_payload_user",
            "responses": large_responses
        }
        
        response = self.session.post(f"{self.base_url}/assess", json=assessment_data)
        # Should handle gracefully (either process or reject cleanly), or hit rate limit
        assert response.status_code in [200, 400, 413, 422, 429]
    
    def _get_valid_responses(self) -> Dict[str, str]:
        """Get valid responses for testing."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])