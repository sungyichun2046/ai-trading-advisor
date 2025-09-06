"""Tests for user profiling system."""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
from typing import Dict

from src.core.user_profiling import (
    RiskCategory, QuestionType, UserProfilingEngine, RiskAssessmentQuestionnaire,
    RiskAssessmentScorer, RiskProfile, UserResponse, validate_trading_decision_against_profile
)


class TestRiskAssessmentQuestionnaire:
    """Tests for risk assessment questionnaire."""
    
    def test_initialize_questions(self):
        """Test questionnaire initialization."""
        questionnaire = RiskAssessmentQuestionnaire()
        questions = questionnaire.get_questions()
        
        assert len(questions) > 0
        assert len(questions) == 10  # Expected number of questions
        
        # Check all questions have required fields
        for question in questions:
            assert hasattr(question, 'id')
            assert hasattr(question, 'question')
            assert hasattr(question, 'type')
            assert hasattr(question, 'options')
            assert hasattr(question, 'weight')
            assert hasattr(question, 'category')
            assert len(question.options) > 0
            assert question.weight > 0
    
    def test_get_question_by_id(self):
        """Test retrieving specific question by ID."""
        questionnaire = RiskAssessmentQuestionnaire()
        
        # Test existing question
        question = questionnaire.get_question_by_id("experience_level")
        assert question is not None
        assert question.id == "experience_level"
        assert question.type == QuestionType.MULTIPLE_CHOICE
        
        # Test non-existing question
        question = questionnaire.get_question_by_id("non_existing")
        assert question is None
    
    def test_question_types(self):
        """Test that questions have valid types."""
        questionnaire = RiskAssessmentQuestionnaire()
        questions = questionnaire.get_questions()
        
        valid_types = {QuestionType.MULTIPLE_CHOICE, QuestionType.SCALE, QuestionType.YES_NO}
        
        for question in questions:
            assert question.type in valid_types


class TestRiskAssessmentScorer:
    """Tests for risk assessment scorer."""
    
    def test_score_response(self):
        """Test response scoring."""
        scorer = RiskAssessmentScorer()
        
        # Test valid responses
        assert scorer.score_response("experience_level", "No experience - I'm just starting") == 1
        assert scorer.score_response("experience_level", "Very experienced - over 10 years") == 5
        
        # Test volatility comfort scale
        assert scorer.score_response("volatility_comfort", "1") == 1
        assert scorer.score_response("volatility_comfort", "5") == 5
        
        # Test invalid question
        assert scorer.score_response("invalid_question", "any_response") == 0
        
        # Test invalid response
        assert scorer.score_response("experience_level", "invalid_response") == 0
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        scorer = RiskAssessmentScorer()
        questionnaire = RiskAssessmentQuestionnaire()
        questions = questionnaire.get_questions()
        
        # Create sample responses
        responses = [
            UserResponse("experience_level", "Very experienced - over 10 years", 5),
            UserResponse("investment_horizon", "Very long-term (over 7 years)", 5),
            UserResponse("volatility_comfort", "5", 5)
        ]
        
        # Filter questions to match responses
        relevant_questions = [q for q in questions if q.id in [r.question_id for r in responses]]
        
        weighted_score, confidence = scorer.calculate_weighted_score(responses, relevant_questions)
        
        assert weighted_score > 0
        assert weighted_score <= 5.0
        assert 0 <= confidence <= 1.0
        
        # Test with no responses
        weighted_score, confidence = scorer.calculate_weighted_score([], questions)
        assert weighted_score == 0.0
        assert confidence == 0.0


class TestUserProfilingEngine:
    """Tests for user profiling engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = UserProfilingEngine()
        self.sample_responses = {
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
    
    def test_get_questionnaire(self):
        """Test getting questionnaire."""
        engine = UserProfilingEngine()
        questions = engine.get_questionnaire()
        
        assert len(questions) > 0
        assert all(hasattr(q, 'id') for q in questions)
    
    def test_validate_responses_valid(self):
        """Test validation of valid responses."""
        engine = UserProfilingEngine()
        responses = {
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
        
        is_valid, errors = engine.validate_responses(responses)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_responses_missing(self):
        """Test validation with missing responses."""
        engine = UserProfilingEngine()
        responses = {
            "experience_level": "Very experienced - over 10 years"
            # Missing all other required responses
        }
        
        is_valid, errors = engine.validate_responses(responses)
        assert not is_valid
        assert len(errors) > 0
        assert any("Missing response" in error for error in errors)
    
    def test_validate_responses_invalid_option(self):
        """Test validation with invalid response options."""
        engine = UserProfilingEngine()
        responses = {
            "experience_level": "Invalid experience level",
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
        
        is_valid, errors = engine.validate_responses(responses)
        assert not is_valid
        assert len(errors) > 0
        assert any("Invalid response" in error for error in errors)
    
    def test_assess_risk_profile_conservative(self):
        """Test assessing conservative risk profile."""
        engine = UserProfilingEngine()
        responses = {
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
        
        profile = engine.assess_risk_profile("test_user_conservative", responses)
        
        assert profile.user_id == "test_user_conservative"
        assert profile.risk_category == RiskCategory.CONSERVATIVE
        assert profile.risk_score < 50  # Should be low score
        assert len(profile.responses) == 10
        assert profile.questionnaire_version == "1.0"
    
    def test_assess_risk_profile_aggressive(self):
        """Test assessing aggressive risk profile."""
        engine = UserProfilingEngine()
        responses = {
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
        
        profile = engine.assess_risk_profile("test_user_aggressive", responses)
        
        assert profile.user_id == "test_user_aggressive"
        assert profile.risk_category == RiskCategory.AGGRESSIVE
        assert profile.risk_score > 70  # Should be high score
        assert len(profile.responses) == 10
    
    def test_assess_risk_profile_moderate(self):
        """Test assessing moderate risk profile."""
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
        
        profile = engine.assess_risk_profile("test_user_moderate", responses)
        
        assert profile.user_id == "test_user_moderate"
        assert profile.risk_category == RiskCategory.MODERATE
        assert 40 < profile.risk_score < 75  # Should be medium score
        assert len(profile.responses) == 10
    
    def test_categorize_risk(self):
        """Test risk categorization."""
        engine = UserProfilingEngine()
        
        assert engine.categorize_risk(1.0) == RiskCategory.CONSERVATIVE
        assert engine.categorize_risk(2.4) == RiskCategory.CONSERVATIVE
        assert engine.categorize_risk(2.5) == RiskCategory.MODERATE
        assert engine.categorize_risk(3.6) == RiskCategory.MODERATE
        assert engine.categorize_risk(3.7) == RiskCategory.AGGRESSIVE
        assert engine.categorize_risk(5.0) == RiskCategory.AGGRESSIVE
    
    def test_get_trading_parameters(self):
        """Test getting trading parameters by risk category."""
        engine = UserProfilingEngine()
        
        # Test conservative parameters
        params = engine.get_trading_parameters(RiskCategory.CONSERVATIVE)
        assert params["max_risk_per_trade"] == 0.01
        assert params["max_portfolio_risk"] == 0.10
        assert params["leverage_limit"] == 1.0
        
        # Test aggressive parameters
        params = engine.get_trading_parameters(RiskCategory.AGGRESSIVE)
        assert params["max_risk_per_trade"] == 0.05
        assert params["max_portfolio_risk"] == 0.40
        assert params["leverage_limit"] == 2.0
        
        # Test moderate parameters
        params = engine.get_trading_parameters(RiskCategory.MODERATE)
        assert 0.01 < params["max_risk_per_trade"] < 0.05
        assert 0.10 < params["max_portfolio_risk"] < 0.40
    
    def test_update_risk_profile(self):
        """Test updating risk profile."""
        engine = UserProfilingEngine()
        
        # Create initial profile
        initial_responses = {
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
        
        initial_profile = engine.assess_risk_profile("test_user", initial_responses)
        original_created_at = initial_profile.created_at
        
        # Update with new responses
        new_responses = {
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
        
        updated_profile = engine.update_risk_profile(initial_profile, new_responses)
        
        assert updated_profile.user_id == initial_profile.user_id
        assert updated_profile.risk_category == RiskCategory.AGGRESSIVE
        assert updated_profile.created_at == original_created_at  # Should be preserved
        assert updated_profile.updated_at > original_created_at   # Should be updated
    
    def test_get_risk_profile_summary(self):
        """Test getting risk profile summary."""
        engine = UserProfilingEngine()
        responses = {
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
        
        profile = engine.assess_risk_profile("test_user", responses)
        summary = engine.get_risk_profile_summary(profile)
        
        assert "user_id" in summary
        assert "risk_category" in summary
        assert "risk_score" in summary
        assert "assessment_date" in summary
        assert "questionnaire_version" in summary
        assert "trading_parameters" in summary
        assert "category_description" in summary
        
        assert summary["user_id"] == "test_user"
        assert summary["risk_category"] == RiskCategory.AGGRESSIVE.value


class TestTradingValidation:
    """Tests for trading decision validation."""
    
    def test_validate_trading_decision_conservative_approved(self):
        """Test trading validation for conservative profile - approved trade."""
        # Create conservative profile
        responses = {
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
        
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Test small, conservative trade
        is_valid, message = validate_trading_decision_against_profile(
            profile, 500.0, 10000.0, 0.005  # $500 trade, $10k account, 0.5% risk
        )
        
        assert is_valid
        assert "approved" in message.lower()
    
    def test_validate_trading_decision_conservative_rejected_position_size(self):
        """Test trading validation for conservative profile - rejected for position size."""
        # Create conservative profile
        responses = {
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
        
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Test large trade that exceeds position size limit
        is_valid, message = validate_trading_decision_against_profile(
            profile, 8000.0, 10000.0, 0.005  # $8k trade, $10k account (80%), 0.5% risk
        )
        
        assert not is_valid
        assert "position size" in message.lower()
        assert "exceeds limit" in message.lower()
    
    def test_validate_trading_decision_conservative_rejected_risk(self):
        """Test trading validation for conservative profile - rejected for risk."""
        # Create conservative profile
        responses = {
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
        
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Test trade with high risk
        is_valid, message = validate_trading_decision_against_profile(
            profile, 300.0, 10000.0, 0.03  # $300 trade, $10k account, 3% risk
        )
        
        assert not is_valid
        assert "risk" in message.lower()
        assert "exceeds limit" in message.lower()
    
    def test_validate_trading_decision_aggressive_approved(self):
        """Test trading validation for aggressive profile - approved trade."""
        # Create aggressive profile
        responses = {
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
        
        engine = UserProfilingEngine()
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Test larger, more aggressive trade
        is_valid, message = validate_trading_decision_against_profile(
            profile, 1500.0, 10000.0, 0.04  # $1.5k trade, $10k account, 4% risk
        )
        
        assert is_valid
        assert "approved" in message.lower()


class TestDatabaseIntegration:
    """Tests for database integration (mocked)."""
    
    @mock.patch('src.data.database.UserProfileStorage')
    def test_store_and_retrieve_profile(self, mock_storage_class):
        """Test storing and retrieving risk profile."""
        # Setup mock
        mock_storage = mock_storage_class.return_value
        mock_storage.store_risk_profile.return_value = True
        
        # Create test profile
        engine = UserProfilingEngine()
        responses = {
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
        
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Mock return value for get_risk_profile
        mock_storage.get_risk_profile.return_value = profile
        
        # Test storage
        result = mock_storage.store_risk_profile(profile)
        assert result is True
        
        # Test retrieval
        retrieved_profile = mock_storage.get_risk_profile("test_user")
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == "test_user"
        assert retrieved_profile.risk_category == RiskCategory.AGGRESSIVE
        
        # Verify mock calls
        mock_storage.store_risk_profile.assert_called_once_with(profile)
        mock_storage.get_risk_profile.assert_called_once_with("test_user")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_assess_risk_profile_invalid_responses(self):
        """Test assessment with invalid responses."""
        engine = UserProfilingEngine()
        
        with pytest.raises(ValueError):
            engine.assess_risk_profile("test_user", {})
    
    def test_empty_user_id(self):
        """Test with empty user ID."""
        engine = UserProfilingEngine()
        responses = {
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
        
        # Should raise error for empty user ID with enhanced validation
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            engine.assess_risk_profile("", responses)
    
    def test_extreme_scores(self):
        """Test with extreme score values."""
        engine = UserProfilingEngine()
        
        # Test categorization at boundary values
        assert engine.categorize_risk(0.0) == RiskCategory.CONSERVATIVE
        assert engine.categorize_risk(2.5) == RiskCategory.MODERATE
        assert engine.categorize_risk(3.7) == RiskCategory.AGGRESSIVE
        assert engine.categorize_risk(10.0) == RiskCategory.AGGRESSIVE  # Out of range
    
    def test_validate_negative_trade_values(self):
        """Test validation with negative trade values."""
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
        
        profile = engine.assess_risk_profile("test_user", responses)
        
        # Test with negative values (should raise ValueError with enhanced validation)
        with pytest.raises(ValueError, match="Trade size cannot be negative"):
            validate_trading_decision_against_profile(
                profile, -1000.0, 10000.0, 0.02
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])