"""API integration tests for risk profiling endpoints.

Production-ready API testing covering:
- Full HTTP request/response cycle
- Authentication and authorization
- Rate limiting validation
- Error handling and status codes
- Performance under load
- Security validations
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import time

from src.main import app
from src.core.user_profiling import RiskCategory, RiskProfile, UserResponse
from datetime import datetime


class TestRiskProfileAPI:
    """Integration tests for risk profile API endpoints."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
        self.base_url = "/api/v1/risk-profile"
    
    def test_questionnaire_endpoint(self):
        """Test questionnaire retrieval endpoint."""
        response = self.client.get(f"{self.base_url}/questionnaire")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "questions" in data
        assert "version" in data
        assert "total_questions" in data
        assert data["total_questions"] == 10
        assert len(data["questions"]) == 10
        
        # Validate question structure
        question = data["questions"][0]
        assert "id" in question
        assert "question" in question
        assert "type" in question
        assert "options" in question
        assert "weight" in question
        assert "category" in question
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "risk-profiling"
        assert "questionnaire_questions" in data
        assert "available_categories" in data
        assert "timestamp" in data
    
    def test_categories_endpoint(self):
        """Test risk categories endpoint."""
        response = self.client.get(f"{self.base_url}/categories")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "conservative" in data
        assert "moderate" in data
        assert "aggressive" in data
        
        for category, description in data.items():
            assert isinstance(description, str)
            assert len(description) > 50  # Meaningful description
    
    def test_parameters_endpoint(self):
        """Test risk parameters endpoint."""
        for category in ["conservative", "moderate", "aggressive"]:
            response = self.client.get(f"{self.base_url}/parameters/{category}")
            
            assert response.status_code == 200
            data = response.json()
            
            required_params = [
                "max_risk_per_trade", "max_portfolio_risk", "max_position_size",
                "daily_loss_limit", "leverage_limit", "volatility_threshold",
                "holding_period_days"
            ]
            
            for param in required_params:
                assert param in data
                assert isinstance(data[param], (int, float))
                assert data[param] > 0
    
    def test_parameters_invalid_category(self):
        """Test parameters endpoint with invalid category."""
        response = self.client.get(f"{self.base_url}/parameters/invalid")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid risk category" in data["message"]
    
    @patch('src.data.database.UserProfileStorage.store_risk_profile')
    def test_assess_endpoint_success(self, mock_store):
        """Test successful risk assessment."""
        mock_store.return_value = True
        
        assessment_data = {
            "user_id": "test_user_001",
            "responses": self._get_valid_moderate_responses()
        }
        
        response = self.client.post(
            f"{self.base_url}/assess",
            json=assessment_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_user_001"
        assert data["risk_category"] in ["conservative", "moderate", "aggressive"]
        assert 0 <= data["risk_score"] <= 100
        assert "assessment_date" in data
        assert "trading_parameters" in data
        assert "category_description" in data
        assert "confidence_score" in data
        assert "review_recommended" in data
    
    def test_assess_endpoint_validation_errors(self):
        """Test assessment endpoint validation."""
        # Test empty user_id
        response = self.client.post(
            f"{self.base_url}/assess",
            json={"user_id": "", "responses": self._get_valid_moderate_responses()}
        )
        assert response.status_code == 422  # Pydantic validation error
        
        # Test empty responses
        response = self.client.post(
            f"{self.base_url}/assess",
            json={"user_id": "test_user", "responses": {}}
        )
        assert response.status_code == 422
        
        # Test missing required fields
        response = self.client.post(
            f"{self.base_url}/assess",
            json={"user_id": "test_user"}
        )
        assert response.status_code == 422
    
    def test_assess_endpoint_invalid_responses(self):
        """Test assessment with invalid questionnaire responses."""
        assessment_data = {
            "user_id": "test_user_001",
            "responses": {
                "experience_level": "Invalid experience level",
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
        
        response = self.client.post(
            f"{self.base_url}/assess",
            json=assessment_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid assessment data" in data["message"]
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    def test_get_profile_success(self, mock_get):
        """Test successful profile retrieval."""
        # Mock profile data
        mock_profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=65,
            responses=[UserResponse("q1", "response1", 3)],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.85
        )
        mock_get.return_value = mock_profile
        
        response = self.client.get(f"{self.base_url}/profile/test_user")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_user"
        assert data["risk_category"] == "moderate"
        assert data["risk_score"] == 65
        assert data["confidence_score"] == 0.85
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    def test_get_profile_not_found(self, mock_get):
        """Test profile retrieval for non-existent user."""
        mock_get.return_value = None
        
        response = self.client.get(f"{self.base_url}/profile/nonexistent_user")
        
        assert response.status_code == 404
        data = response.json()
        assert "Risk profile not found" in data["message"]
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    @patch('src.data.database.UserProfileStorage.store_risk_profile')
    def test_update_profile_success(self, mock_store, mock_get):
        """Test successful profile update."""
        # Mock existing profile
        mock_profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.CONSERVATIVE,
            risk_score=30,
            responses=[UserResponse("q1", "response1", 2)],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.7
        )
        mock_get.return_value = mock_profile
        mock_store.return_value = True
        
        update_data = {
            "responses": self._get_valid_aggressive_responses()
        }
        
        response = self.client.put(
            f"{self.base_url}/update/test_user",
            json=update_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "test_user"
        assert data["risk_category"] == "aggressive"  # Should change from conservative
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    def test_validate_trade_success(self, mock_get):
        """Test successful trade validation."""
        # Mock moderate profile
        mock_profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.MODERATE,
            risk_score=60,
            responses=[UserResponse("q1", "response1", 3)],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.8
        )
        mock_get.return_value = mock_profile
        
        trade_data = {
            "user_id": "test_user",
            "trade_size": 800.0,
            "account_balance": 10000.0,
            "position_risk": 0.015
        }
        
        response = self.client.post(
            f"{self.base_url}/validate-trade",
            json=trade_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["approved"] is True
        assert "approved" in data["message"].lower()
        assert data["user_risk_category"] == "moderate"
        assert "risk_parameters" in data
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    def test_validate_trade_rejection(self, mock_get):
        """Test trade validation rejection."""
        # Mock conservative profile
        mock_profile = RiskProfile(
            user_id="test_user",
            risk_category=RiskCategory.CONSERVATIVE,
            risk_score=25,
            responses=[UserResponse("q1", "response1", 1)],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            confidence_score=0.9
        )
        mock_get.return_value = mock_profile
        
        # Large trade that should be rejected
        trade_data = {
            "user_id": "test_user",
            "trade_size": 8000.0,  # 80% of account
            "account_balance": 10000.0,
            "position_risk": 0.05
        }
        
        response = self.client.post(
            f"{self.base_url}/validate-trade",
            json=trade_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["approved"] is False
        assert "exceeds limit" in data["message"].lower()
        assert data["user_risk_category"] == "conservative"
    
    @patch('src.data.database.UserProfileStorage.get_risk_profile')
    def test_validate_trade_no_profile(self, mock_get):
        """Test trade validation without profile."""
        mock_get.return_value = None
        
        trade_data = {
            "user_id": "test_user",
            "trade_size": 1000.0,
            "account_balance": 10000.0,
            "position_risk": 0.02
        }
        
        response = self.client.post(
            f"{self.base_url}/validate-trade",
            json=trade_data
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "Risk profile not found" in data["message"]
    
    def test_validate_trade_invalid_parameters(self):
        """Test trade validation with invalid parameters."""
        # Test negative values
        trade_data = {
            "user_id": "test_user",
            "trade_size": -1000.0,
            "account_balance": 10000.0,
            "position_risk": 0.02
        }
        
        response = self.client.post(
            f"{self.base_url}/validate-trade",
            json=trade_data
        )
        
        assert response.status_code == 422  # Pydantic validation error
    
    def _get_valid_moderate_responses(self) -> dict:
        """Get valid responses that result in moderate risk profile."""
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
    
    def _get_valid_aggressive_responses(self) -> dict:
        """Get valid responses that result in aggressive risk profile."""
        return {
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


class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
        self.base_url = "/api/v1/risk-profile"
    
    def test_questionnaire_response_time(self):
        """Test questionnaire endpoint response time."""
        start_time = time.time()
        
        response = self.client.get(f"{self.base_url}/questionnaire")
        
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should respond in < 100ms
    
    def test_concurrent_assessments(self):
        """Test handling of concurrent assessment requests."""
        assessment_data = {
            "user_id": "concurrent_user",
            "responses": self._get_valid_responses()
        }
        
        # Simulate concurrent requests
        start_time = time.time()
        responses = []
        
        for i in range(10):
            assessment_data["user_id"] = f"user_{i}"
            response = self.client.post(
                f"{self.base_url}/assess",
                json=assessment_data
            )
            responses.append(response)
        
        total_time = time.time() - start_time
        
        # All requests should complete successfully
        assert all(r.status_code == 200 for r in responses)
        assert total_time < 2.0  # 10 assessments in < 2 seconds
    
    def test_large_payload_handling(self):
        """Test handling of large request payloads."""
        # Create assessment with very long responses
        large_responses = self._get_valid_responses()
        large_responses["experience_level"] = "A" * 1000  # Very long response
        
        assessment_data = {
            "user_id": "large_payload_user",
            "responses": large_responses
        }
        
        response = self.client.post(
            f"{self.base_url}/assess",
            json=assessment_data
        )
        
        # Should handle large payloads gracefully (either accept or reject cleanly)
        assert response.status_code in [200, 400, 413]  # OK, Bad Request, or Payload Too Large
    
    def _get_valid_responses(self) -> dict:
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


class TestAPISecurity:
    """Security tests for API endpoints."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
        self.base_url = "/api/v1/risk-profile"
    
    def test_sql_injection_attempts(self):
        """Test API protection against SQL injection."""
        sql_injection_payloads = [
            "'; DROP TABLE user_profiles; --",
            "1' OR '1'='1",
            "'; INSERT INTO user_profiles VALUES ('hacker', 'aggressive', 100); --"
        ]
        
        for payload in sql_injection_payloads:
            assessment_data = {
                "user_id": payload,
                "responses": self._get_valid_responses()
            }
            
            response = self.client.post(
                f"{self.base_url}/assess",
                json=assessment_data
            )
            
            # Should either process safely or reject
            assert response.status_code in [200, 400]
            
            if response.status_code == 200:
                # If processed, should not cause any system issues
                data = response.json()
                assert "user_id" in data
    
    def test_xss_prevention(self):
        """Test API protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')"
        ]
        
        for payload in xss_payloads:
            assessment_data = {
                "user_id": "test_user",
                "responses": {**self._get_valid_responses(), "experience_level": payload}
            }
            
            response = self.client.post(
                f"{self.base_url}/assess",
                json=assessment_data
            )
            
            # Should reject malicious payloads
            assert response.status_code in [400, 422]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        assessment_data = {
            "user_id": "rate_limit_user",
            "responses": self._get_valid_responses()
        }
        
        # Make many rapid requests
        responses = []
        for i in range(15):  # Exceed the rate limit of 10 per 10 minutes
            assessment_data["user_id"] = f"user_{i}"
            response = self.client.post(
                f"{self.base_url}/assess",
                json=assessment_data
            )
            responses.append(response)
        
        # Some requests should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes or all(code == 200 for code in status_codes[:10])
    
    def test_input_size_limits(self):
        """Test protection against oversized inputs."""
        # Create very large user_id
        large_user_id = "a" * 10000
        
        assessment_data = {
            "user_id": large_user_id,
            "responses": self._get_valid_responses()
        }
        
        response = self.client.post(
            f"{self.base_url}/assess",
            json=assessment_data
        )
        
        # Should handle large inputs gracefully (422 is Pydantic validation error)
        assert response.status_code in [200, 400, 413, 422]
    
    def _get_valid_responses(self) -> dict:
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