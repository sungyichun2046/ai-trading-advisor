"""User profiling and risk tolerance assessment for AI Trading Advisor.

This module provides comprehensive risk profiling capabilities including:
- Multi-dimensional risk assessment questionnaire
- Algorithmic risk categorization (Conservative/Moderate/Aggressive)
- Dynamic trading parameter calculation
- Trade validation against user profiles
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Risk tolerance categories."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class QuestionType(str, Enum):
    """Risk assessment question types."""
    MULTIPLE_CHOICE = "multiple_choice"
    SCALE = "scale"
    YES_NO = "yes_no"


@dataclass
class RiskQuestion:
    """Risk assessment question."""
    id: str
    question: str
    type: QuestionType
    options: List[str]
    weight: float
    category: str


@dataclass
class UserResponse:
    """User response to risk question."""
    question_id: str
    response: str
    score: int


@dataclass
class RiskProfile:
    """User risk profile with comprehensive metadata.
    
    Attributes:
        user_id: Unique identifier for user
        risk_category: Categorized risk level
        risk_score: Numerical risk score (0-100)
        responses: List of user responses to questionnaire
        created_at: Initial profile creation timestamp
        updated_at: Last modification timestamp
        questionnaire_version: Version of questionnaire used
        confidence_score: Algorithm confidence in categorization (0-1)
        last_review_date: Date of last manual review (if any)
    """
    user_id: str
    risk_category: RiskCategory
    risk_score: int
    responses: List[UserResponse]
    created_at: datetime
    updated_at: datetime
    questionnaire_version: str = "1.0"
    confidence_score: Optional[float] = field(default=None)
    last_review_date: Optional[datetime] = field(default=None)
    
    def __post_init__(self):
        """Validate risk profile data after initialization."""
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not 0 <= self.risk_score <= 100:
            raise ValueError("risk_score must be between 0 and 100")
        if self.created_at > self.updated_at:
            raise ValueError("created_at cannot be after updated_at")


class RiskAssessmentQuestionnaire:
    """Risk tolerance assessment questionnaire."""
    
    def __init__(self):
        self.questions = self._initialize_questions()
    
    def _initialize_questions(self) -> List[RiskQuestion]:
        """Initialize risk assessment questions."""
        return [
            RiskQuestion(
                id="experience_level",
                question="How would you describe your investment experience?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "No experience - I'm just starting",
                    "Limited experience - less than 2 years",
                    "Moderate experience - 2-5 years",
                    "Experienced - 5-10 years",
                    "Very experienced - over 10 years"
                ],
                weight=2.0,
                category="experience"
            ),
            RiskQuestion(
                id="investment_horizon",
                question="What is your typical investment time horizon?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Short-term (less than 1 year)",
                    "Medium-term (1-3 years)",
                    "Long-term (3-7 years)",
                    "Very long-term (over 7 years)"
                ],
                weight=1.5,
                category="time_horizon"
            ),
            RiskQuestion(
                id="volatility_comfort",
                question="How comfortable are you with market volatility?",
                type=QuestionType.SCALE,
                options=["1", "2", "3", "4", "5"],
                weight=2.5,
                category="risk_tolerance"
            ),
            RiskQuestion(
                id="loss_tolerance",
                question="What is the maximum loss you could tolerate in a year?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "0-5% - I cannot tolerate significant losses",
                    "5-10% - Small losses are acceptable",
                    "10-20% - Moderate losses are acceptable for higher returns",
                    "20-30% - Large losses are acceptable for potentially high returns",
                    "30%+ - I can handle very large losses"
                ],
                weight=3.0,
                category="loss_tolerance"
            ),
            RiskQuestion(
                id="portfolio_percentage",
                question="What percentage of your total portfolio will this trading account represent?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Less than 10% - Very small portion",
                    "10-25% - Small portion",
                    "25-50% - Moderate portion",
                    "50-75% - Large portion",
                    "Over 75% - Most of my portfolio"
                ],
                weight=2.0,
                category="portfolio_allocation"
            ),
            RiskQuestion(
                id="income_stability",
                question="How stable is your income?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Very stable - Regular salary with job security",
                    "Stable - Regular income with some variability",
                    "Moderate - Income varies but generally consistent",
                    "Variable - Income fluctuates significantly",
                    "Unstable - Income is very unpredictable"
                ],
                weight=1.5,
                category="financial_stability"
            ),
            RiskQuestion(
                id="market_reaction",
                question="If your portfolio dropped 15% in a month, what would you do?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Sell everything immediately to prevent further losses",
                    "Sell some positions to reduce risk",
                    "Hold my positions and wait for recovery",
                    "Buy more at the lower prices",
                    "Significantly increase my investment"
                ],
                weight=2.5,
                category="behavioral_response"
            ),
            RiskQuestion(
                id="financial_goals",
                question="What is your primary investment goal?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Capital preservation - Protect my money",
                    "Income generation - Regular returns",
                    "Moderate growth - Steady appreciation",
                    "Growth - Significant appreciation",
                    "Aggressive growth - Maximum returns"
                ],
                weight=2.0,
                category="investment_goals"
            ),
            RiskQuestion(
                id="age_category",
                question="What is your age category?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Under 25 - Long investment horizon",
                    "25-35 - Building wealth phase",
                    "36-50 - Peak earning years",
                    "51-65 - Pre-retirement planning",
                    "Over 65 - Retirement/preservation focus"
                ],
                weight=1.0,
                category="demographics"
            ),
            RiskQuestion(
                id="trading_frequency",
                question="How often do you plan to actively trade?",
                type=QuestionType.MULTIPLE_CHOICE,
                options=[
                    "Rarely - Long-term buy and hold",
                    "Occasionally - Few trades per month",
                    "Regularly - Weekly trading",
                    "Frequently - Daily trading",
                    "Very frequently - Multiple trades per day"
                ],
                weight=1.5,
                category="trading_behavior"
            )
        ]
    
    def get_questions(self) -> List[RiskQuestion]:
        """Get all risk assessment questions."""
        return self.questions
    
    def get_question_by_id(self, question_id: str) -> Optional[RiskQuestion]:
        """Get a specific question by ID."""
        return next((q for q in self.questions if q.id == question_id), None)


class RiskAssessmentScorer:
    """Scorer for risk assessment responses."""
    
    def __init__(self):
        self.scoring_rules = self._initialize_scoring_rules()
    
    def _initialize_scoring_rules(self) -> Dict[str, Dict[str, int]]:
        """Initialize scoring rules for each question."""
        return {
            "experience_level": {
                "No experience - I'm just starting": 1,
                "Limited experience - less than 2 years": 2,
                "Moderate experience - 2-5 years": 3,
                "Experienced - 5-10 years": 4,
                "Very experienced - over 10 years": 5
            },
            "investment_horizon": {
                "Short-term (less than 1 year)": 2,
                "Medium-term (1-3 years)": 3,
                "Long-term (3-7 years)": 4,
                "Very long-term (over 7 years)": 5
            },
            "volatility_comfort": {
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
            },
            "loss_tolerance": {
                "0-5% - I cannot tolerate significant losses": 1,
                "5-10% - Small losses are acceptable": 2,
                "10-20% - Moderate losses are acceptable for higher returns": 3,
                "20-30% - Large losses are acceptable for potentially high returns": 4,
                "30%+ - I can handle very large losses": 5
            },
            "portfolio_percentage": {
                "Less than 10% - Very small portion": 5,
                "10-25% - Small portion": 4,
                "25-50% - Moderate portion": 3,
                "50-75% - Large portion": 2,
                "Over 75% - Most of my portfolio": 1
            },
            "income_stability": {
                "Very stable - Regular salary with job security": 5,
                "Stable - Regular income with some variability": 4,
                "Moderate - Income varies but generally consistent": 3,
                "Variable - Income fluctuates significantly": 2,
                "Unstable - Income is very unpredictable": 1
            },
            "market_reaction": {
                "Sell everything immediately to prevent further losses": 1,
                "Sell some positions to reduce risk": 2,
                "Hold my positions and wait for recovery": 3,
                "Buy more at the lower prices": 4,
                "Significantly increase my investment": 5
            },
            "financial_goals": {
                "Capital preservation - Protect my money": 1,
                "Income generation - Regular returns": 2,
                "Moderate growth - Steady appreciation": 3,
                "Growth - Significant appreciation": 4,
                "Aggressive growth - Maximum returns": 5
            },
            "age_category": {
                "Under 25 - Long investment horizon": 5,
                "25-35 - Building wealth phase": 4,
                "36-50 - Peak earning years": 3,
                "51-65 - Pre-retirement planning": 2,
                "Over 65 - Retirement/preservation focus": 1
            },
            "trading_frequency": {
                "Rarely - Long-term buy and hold": 2,
                "Occasionally - Few trades per month": 3,
                "Regularly - Weekly trading": 4,
                "Frequently - Daily trading": 5,
                "Very frequently - Multiple trades per day": 4  # Slightly lower due to higher risk
            }
        }
    
    def score_response(self, question_id: str, response: str) -> int:
        """Score a single response."""
        if question_id not in self.scoring_rules:
            logger.warning(f"Unknown question ID: {question_id}")
            return 0
        
        return self.scoring_rules[question_id].get(response, 0)
    
    def calculate_weighted_score(self, responses: List[UserResponse], questions: List[RiskQuestion]) -> Tuple[float, float]:
        """Calculate weighted risk score with confidence measure.
        
        Returns:
            Tuple of (weighted_score, confidence_score)
            - weighted_score: 0.0-5.0 scale
            - confidence_score: 0.0-1.0 based on response consistency
        """
        if not responses or not questions:
            return 0.0, 0.0
            
        total_weighted_score = 0.0
        total_weight = 0.0
        score_variance = []
        
        question_map = {q.id: q for q in questions}
        
        for response in responses:
            question = question_map.get(response.question_id)
            if question and 1 <= response.score <= 5:
                weighted_score = response.score * question.weight
                total_weighted_score += weighted_score
                total_weight += question.weight
                score_variance.append(response.score)
        
        if total_weight == 0:
            return 0.0, 0.0
            
        final_score = total_weighted_score / total_weight
        
        # Calculate confidence based on response consistency
        if len(score_variance) > 1:
            mean_score = sum(score_variance) / len(score_variance)
            variance = sum((x - mean_score) ** 2 for x in score_variance) / len(score_variance)
            # Lower variance = higher confidence
            confidence = max(0.0, min(1.0, 1.0 - (variance / 4.0)))
        else:
            confidence = 0.5  # Moderate confidence with single response
            
        return final_score, confidence


class UserProfilingEngine:
    """Engine for user risk profiling and assessment."""
    
    def __init__(self):
        self.questionnaire = RiskAssessmentQuestionnaire()
        self.scorer = RiskAssessmentScorer()
        self.risk_thresholds = {
            RiskCategory.CONSERVATIVE: (0.0, 2.5),
            RiskCategory.MODERATE: (2.5, 3.7),
            RiskCategory.AGGRESSIVE: (3.7, 5.0)
        }
    
    def get_questionnaire(self) -> List[RiskQuestion]:
        """Get risk assessment questionnaire."""
        return self.questionnaire.get_questions()
    
    def validate_responses(self, responses: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate user responses to questionnaire."""
        errors = []
        questions = self.questionnaire.get_questions()
        question_ids = {q.id for q in questions}
        
        # Check if all required questions are answered
        for question in questions:
            if question.id not in responses:
                errors.append(f"Missing response for question: {question.id}")
                continue
            
            # Validate response is in valid options
            response = responses[question.id]
            if response not in question.options:
                errors.append(f"Invalid response for {question.id}: {response}")
        
        # Check for unexpected responses
        for response_id in responses:
            if response_id not in question_ids:
                errors.append(f"Unknown question ID: {response_id}")
        
        return len(errors) == 0, errors
    
    def assess_risk_profile(self, user_id: str, responses: Dict[str, str]) -> RiskProfile:
        """Assess user risk profile based on responses."""
        # Validate responses
        is_valid, errors = self.validate_responses(responses)
        if not is_valid:
            raise ValueError(f"Invalid responses: {'; '.join(errors)}")
        
        # Score responses
        user_responses = []
        questions = self.questionnaire.get_questions()
        
        for question in questions:
            if question.id in responses:
                response_text = responses[question.id]
                score = self.scorer.score_response(question.id, response_text)
                user_responses.append(UserResponse(
                    question_id=question.id,
                    response=response_text,
                    score=score
                ))
        
        # Calculate weighted score with confidence
        weighted_score, confidence = self.scorer.calculate_weighted_score(user_responses, questions)
        
        # Determine risk category
        risk_category = self.categorize_risk(weighted_score)
        
        # Scale score to 0-100
        scaled_score = min(100, max(0, int(weighted_score * 20)))
        
        # Create risk profile
        now = datetime.now()
        return RiskProfile(
            user_id=user_id,
            risk_category=risk_category,
            risk_score=scaled_score,
            responses=user_responses,
            created_at=now,
            updated_at=now,
            confidence_score=confidence
        )
    
    def categorize_risk(self, weighted_score: float) -> RiskCategory:
        """Categorize risk based on weighted score."""
        for category, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= weighted_score < max_score:
                return category
        
        # Default to aggressive if score is at maximum
        return RiskCategory.AGGRESSIVE
    
    def get_trading_parameters(self, risk_category: RiskCategory, confidence_score: Optional[float] = None) -> Dict[str, float]:
        """Get recommended trading parameters based on risk category and confidence.
        
        Args:
            risk_category: User's risk category
            confidence_score: Algorithm confidence (0-1), used to adjust parameters
            
        Returns:
            Dictionary of trading parameters with confidence-adjusted values
        """
        base_parameters = {
            RiskCategory.CONSERVATIVE: {
                "max_risk_per_trade": 0.01,
                "max_portfolio_risk": 0.10,
                "max_position_size": 0.05,
                "daily_loss_limit": 0.02,
                "leverage_limit": 1.0,
                "volatility_threshold": 0.15,
                "holding_period_days": 30.0,
            },
            RiskCategory.MODERATE: {
                "max_risk_per_trade": 0.02,
                "max_portfolio_risk": 0.20,
                "max_position_size": 0.10,
                "daily_loss_limit": 0.06,
                "leverage_limit": 1.5,
                "volatility_threshold": 0.25,
                "holding_period_days": 14.0,
            },
            RiskCategory.AGGRESSIVE: {
                "max_risk_per_trade": 0.05,
                "max_portfolio_risk": 0.40,
                "max_position_size": 0.20,
                "daily_loss_limit": 0.15,
                "leverage_limit": 2.0,
                "volatility_threshold": 0.40,
                "holding_period_days": 3.0,
            }
        }
        
        params = base_parameters.get(risk_category, base_parameters[RiskCategory.MODERATE]).copy()
        
        # Adjust parameters based on confidence score
        if confidence_score is not None and confidence_score < 0.7:
            # Reduce risk limits for low-confidence assessments
            adjustment_factor = 0.8  # 20% reduction
            risk_params = ['max_risk_per_trade', 'max_portfolio_risk', 'max_position_size', 'daily_loss_limit']
            for param in risk_params:
                if param in params:
                    params[param] = params[param] * adjustment_factor
        
        return params
    
    def update_risk_profile(self, current_profile: RiskProfile, new_responses: Dict[str, str]) -> RiskProfile:
        """Update existing risk profile with new responses."""
        # Create new assessment
        new_profile = self.assess_risk_profile(current_profile.user_id, new_responses)
        
        # Preserve original creation date
        new_profile.created_at = current_profile.created_at
        
        return new_profile
    
    def get_risk_profile_summary(self, profile: RiskProfile) -> Dict[str, any]:
        """Get summary of risk profile for display."""
        trading_params = self.get_trading_parameters(profile.risk_category)
        
        return {
            "user_id": profile.user_id,
            "risk_category": profile.risk_category.value,
            "risk_score": profile.risk_score,
            "assessment_date": profile.updated_at.isoformat(),
            "questionnaire_version": profile.questionnaire_version,
            "trading_parameters": trading_params,
            "category_description": self._get_category_description(profile.risk_category)
        }
    
    def _get_category_description(self, category: RiskCategory) -> str:
        """Get description for risk category."""
        descriptions = {
            RiskCategory.CONSERVATIVE: "Conservative investors prioritize capital preservation and accept lower returns to minimize risk of losses.",
            RiskCategory.MODERATE: "Moderate investors seek balanced growth with measured risk, accepting some volatility for better returns.",
            RiskCategory.AGGRESSIVE: "Aggressive investors pursue maximum returns and are willing to accept high volatility and potential large losses."
        }
        return descriptions.get(category, "")


def validate_trading_decision_against_profile(
    profile: RiskProfile, 
    trade_size: float, 
    account_balance: float,
    position_risk: float
) -> Tuple[bool, str]:
    """Validate trading decision against user risk profile with comprehensive checks.
    
    Args:
        profile: User's risk profile
        trade_size: Proposed trade size in USD
        account_balance: Current account balance in USD
        position_risk: Risk as percentage (0.01 = 1%)
        
    Returns:
        Tuple of (is_valid, explanation_message)
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if account_balance <= 0:
        raise ValueError("Account balance must be positive")
    if trade_size < 0:
        raise ValueError("Trade size cannot be negative")
    if position_risk < 0:
        raise ValueError("Position risk cannot be negative")
        
    engine = UserProfilingEngine()
    params = engine.get_trading_parameters(profile.risk_category, profile.confidence_score)
    
    # Comprehensive validation checks
    position_percentage = trade_size / account_balance if account_balance > 0 else 0
    
    # Check position size limit
    if position_percentage > params["max_position_size"]:
        return False, (
            f"Position size {position_percentage:.1%} exceeds limit "
            f"{params['max_position_size']:.1%} for {profile.risk_category.value} profile"
        )
    
    # Check risk per trade
    if position_risk > params["max_risk_per_trade"]:
        return False, (
            f"Trade risk {position_risk:.1%} exceeds limit "
            f"{params['max_risk_per_trade']:.1%} for {profile.risk_category.value} profile"
        )
    
    # Additional safety checks
    if trade_size > account_balance:
        return False, "Trade size exceeds account balance"
        
    # Low confidence warning
    confidence_warning = ""
    if profile.confidence_score and profile.confidence_score < 0.6:
        confidence_warning = " (Note: Low confidence in risk assessment - consider review)"
    
    return True, f"Trade approved for user risk profile{confidence_warning}"