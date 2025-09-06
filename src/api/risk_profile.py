"""Risk profiling API endpoints for AI Trading Advisor.

Production-ready RESTful API for comprehensive user risk profiling including:
- Risk assessment questionnaire management
- User profile CRUD operations
- Trading decision validation
- Comprehensive error handling and logging
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import asyncio
from contextlib import asynccontextmanager

from src.core.user_profiling import (
    UserProfilingEngine, 
    RiskProfile, 
    RiskCategory, 
    RiskQuestion,
    validate_trading_decision_against_profile
)
from src.data.database import UserProfileStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk-profile", tags=["Risk Profiling"])


# Pydantic models for API request/response
class QuestionResponse(BaseModel):
    """Response model for risk assessment question."""
    id: str
    question: str
    type: str
    options: List[str]
    weight: float
    category: str


class QuestionnaireResponse(BaseModel):
    """Response model for complete questionnaire."""
    questions: List[QuestionResponse]
    version: str = "1.0"
    total_questions: int


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment."""
    user_id: str = Field(..., min_length=1, max_length=100)
    responses: Dict[str, str] = Field(..., min_items=1)
    
    @validator('responses')
    def validate_responses_not_empty(cls, v):
        """Validate that responses are not empty."""
        if not v:
            raise ValueError("Responses cannot be empty")
        for key, value in v.items():
            if not value or not value.strip():
                raise ValueError(f"Response for {key} cannot be empty")
        return v


class RiskProfileResponse(BaseModel):
    """Response model for risk profile with enhanced metadata."""
    user_id: str
    risk_category: str
    risk_score: int
    assessment_date: str
    questionnaire_version: str
    trading_parameters: Dict[str, float]
    category_description: str
    confidence_score: Optional[float] = None
    last_updated: Optional[str] = None
    review_recommended: bool = False


class RiskProfileUpdateRequest(BaseModel):
    """Request model for risk profile update."""
    responses: Dict[str, str] = Field(..., min_items=1)
    
    @validator('responses')
    def validate_responses_not_empty(cls, v):
        """Validate that responses are not empty."""
        if not v:
            raise ValueError("Responses cannot be empty")
        return v


class TradingValidationRequest(BaseModel):
    """Request model for trading decision validation."""
    user_id: str = Field(..., min_length=1, max_length=100)
    trade_size: float = Field(..., gt=0, description="Trade size in USD")
    account_balance: float = Field(..., gt=0, description="Current account balance in USD")
    position_risk: float = Field(..., ge=0, le=1, description="Risk as percentage (0.01 = 1%)")


class TradingValidationResponse(BaseModel):
    """Response model for trading validation."""
    approved: bool
    message: str
    user_risk_category: str
    risk_parameters: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[Dict] = None


# Dependencies with proper error handling
def get_profiling_engine() -> UserProfilingEngine:
    """Get user profiling engine instance."""
    try:
        return UserProfilingEngine()
    except Exception as e:
        logger.error(f"Failed to initialize profiling engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk profiling service temporarily unavailable"
        )


def get_profile_storage() -> UserProfileStorage:
    """Get user profile storage instance with connection validation."""
    try:
        storage = UserProfileStorage()
        # Test connection
        with storage.get_connection():
            pass
        return storage
    except Exception as e:
        logger.error(f"Failed to initialize profile storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )


# Rate limiting helper
class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""
    def __init__(self):
        self.requests = {}
        
    async def is_allowed(self, key: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if request is within rate limit."""
        now = datetime.now().timestamp()
        if key not in self.requests:
            self.requests[key] = []
            
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < window]
        
        if len(self.requests[key]) >= limit:
            return False
            
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter()


@router.get("/questionnaire", response_model=QuestionnaireResponse)
async def get_risk_questionnaire(
    engine: UserProfilingEngine = Depends(get_profiling_engine)
):
    """Get risk assessment questionnaire.
    
    Returns the complete risk assessment questionnaire with all questions
    that users need to answer to determine their risk profile.
    """
    try:
        questions = engine.get_questionnaire()
        question_responses = [
            QuestionResponse(
                id=q.id,
                question=q.question,
                type=q.type.value,
                options=q.options,
                weight=q.weight,
                category=q.category
            )
            for q in questions
        ]
        
        return QuestionnaireResponse(
            questions=question_responses,
            total_questions=len(question_responses)
        )
    
    except Exception as e:
        logger.error(f"Failed to get questionnaire: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questionnaire: {str(e)}"
        )


@router.post("/assess", response_model=RiskProfileResponse)
async def assess_risk_profile(
    request: RiskAssessmentRequest,
    http_request: Request,
    engine: UserProfilingEngine = Depends(get_profiling_engine),
    storage: UserProfileStorage = Depends(get_profile_storage)
):
    """Assess user risk profile with comprehensive validation and logging.
    
    Takes user responses to the risk assessment questionnaire and returns
    a complete risk profile with category, score, and trading parameters.
    Includes rate limiting and audit logging for compliance.
    """
    # Rate limiting
    client_ip = http_request.client.host
    if not await rate_limiter.is_allowed(f"assess_{client_ip}", limit=50, window=600):
        logger.warning(f"Rate limit exceeded for IP {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many assessment requests. Please wait before trying again."
        )
    
    try:
        # Assess risk profile
        profile = engine.assess_risk_profile(request.user_id, request.responses)
        
        # Store profile in database with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if storage.store_risk_profile(profile):
                    break
                elif attempt == max_retries - 1:
                    logger.error(f"Failed to store risk profile after {max_retries} attempts for user {request.user_id}")
                    # Still return profile but log the storage failure
            except Exception as storage_error:
                logger.error(f"Storage attempt {attempt + 1} failed: {storage_error}")
                if attempt == max_retries - 1:
                    # Don't fail the whole request if storage fails
                    pass
                else:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        # Get profile summary
        summary = engine.get_risk_profile_summary(profile)
        
        # Enhanced logging for audit trail
        confidence_str = f"{profile.confidence_score:.3f}" if profile.confidence_score else "N/A"
        logger.info(
            f"Risk profile assessed - User: {request.user_id}, "
            f"Category: {profile.risk_category.value}, "
            f"Score: {profile.risk_score}, "
            f"Confidence: {confidence_str}, "
            f"IP: {client_ip}"
        )
        
        # Determine if manual review is recommended
        review_recommended = (
            profile.confidence_score is not None and profile.confidence_score < 0.6
        ) or profile.risk_score in range(45, 55)  # Boundary cases
        
        return RiskProfileResponse(
            user_id=summary["user_id"],
            risk_category=summary["risk_category"],
            risk_score=summary["risk_score"],
            assessment_date=summary["assessment_date"],
            questionnaire_version=summary["questionnaire_version"],
            trading_parameters=summary["trading_parameters"],
            category_description=summary["category_description"],
            confidence_score=profile.confidence_score,
            last_updated=summary["assessment_date"],
            review_recommended=review_recommended
        )
    
    except ValueError as e:
        logger.warning(f"Invalid risk assessment request from {client_ip}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid assessment data: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Failed to assess risk profile for {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk assessment service temporarily unavailable"
        )


@router.put("/update/{user_id}", response_model=RiskProfileResponse)
async def update_risk_profile(
    user_id: str,
    request: RiskProfileUpdateRequest,
    engine: UserProfilingEngine = Depends(get_profiling_engine),
    storage: UserProfileStorage = Depends(get_profile_storage)
):
    """Update existing risk profile with new responses.
    
    Updates an existing user's risk profile based on new questionnaire responses.
    The original creation date is preserved while the updated date is set to now.
    """
    try:
        # Get existing profile to preserve creation date
        existing_profile = storage.get_risk_profile(user_id)
        
        # Create new assessment
        new_profile = engine.assess_risk_profile(user_id, request.responses)
        
        # Preserve original creation date if profile exists
        if existing_profile:
            new_profile.created_at = existing_profile.created_at
        
        # Store updated profile in database
        if not storage.store_risk_profile(new_profile):
            logger.warning(f"Failed to store updated risk profile for user {user_id}")
            # Continue anyway - return the profile even if storage failed
        
        # Get profile summary
        summary = engine.get_risk_profile_summary(new_profile)
        
        logger.info(f"Risk profile updated for user {user_id}: {new_profile.risk_category.value}")
        
        return RiskProfileResponse(
            user_id=summary["user_id"],
            risk_category=summary["risk_category"],
            risk_score=summary["risk_score"],
            assessment_date=summary["assessment_date"],
            questionnaire_version=summary["questionnaire_version"],
            trading_parameters=summary["trading_parameters"],
            category_description=summary["category_description"],
            confidence_score=new_profile.confidence_score,
            last_updated=summary["assessment_date"]
        )
    
    except ValueError as e:
        logger.warning(f"Invalid risk profile update: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid update data: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Failed to update risk profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )


@router.get("/profile/{user_id}", response_model=RiskProfileResponse)
async def get_risk_profile(
    user_id: str,
    engine: UserProfilingEngine = Depends(get_profiling_engine),
    storage: UserProfileStorage = Depends(get_profile_storage)
):
    """Get existing risk profile for a user.
    
    Retrieves the current risk profile for a user if it exists.
    """
    try:
        # Retrieve profile from database
        profile = storage.get_risk_profile(user_id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Risk profile not found for user {user_id}"
            )
        
        # Get profile summary
        summary = engine.get_risk_profile_summary(profile)
        
        return RiskProfileResponse(
            user_id=summary["user_id"],
            risk_category=summary["risk_category"],
            risk_score=summary["risk_score"],
            assessment_date=summary["assessment_date"],
            questionnaire_version=summary["questionnaire_version"],
            trading_parameters=summary["trading_parameters"],
            category_description=summary["category_description"],
            confidence_score=profile.confidence_score,
            last_updated=summary["assessment_date"]
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get risk profile for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )


@router.post("/validate-trade", response_model=TradingValidationResponse)
async def validate_trading_decision(
    request: TradingValidationRequest,
    http_request: Request,
    engine: UserProfilingEngine = Depends(get_profiling_engine),
    storage: UserProfileStorage = Depends(get_profile_storage)
):
    """Validate trading decision with comprehensive compliance checks.
    
    Performs multi-layered validation including:
    - User risk profile compatibility
    - Position sizing limits
    - Risk management compliance
    - Audit logging for regulatory requirements
    """
    client_ip = http_request.client.host
    
    # Enhanced rate limiting for trading validation
    if not await rate_limiter.is_allowed(f"trade_validation_{client_ip}", limit=50, window=3600):
        logger.warning(f"Trade validation rate limit exceeded for IP {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many trade validation requests. Please wait before trying again."
        )
    
    try:
        # Retrieve profile from database
        profile = storage.get_risk_profile(request.user_id)
        
        if not profile:
            logger.warning(f"Trade validation attempted for user without profile: {request.user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Risk profile not found for user {request.user_id}. Please complete risk assessment first."
            )
        
        # Validate trade against profile with enhanced error handling
        try:
            is_valid, message = validate_trading_decision_against_profile(
                profile, request.trade_size, request.account_balance, request.position_risk
            )
        except ValueError as validation_error:
            logger.warning(f"Invalid trade validation parameters: {validation_error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid trade parameters: {str(validation_error)}"
            )
        
        # Get trading parameters with confidence adjustment
        trading_params = engine.get_trading_parameters(profile.risk_category, profile.confidence_score)
        
        # Comprehensive audit logging
        logger.info(
            f"Trade validation - User: {request.user_id}, "
            f"Result: {'APPROVED' if is_valid else 'REJECTED'}, "
            f"Trade Size: ${request.trade_size:,.2f}, "
            f"Account Balance: ${request.account_balance:,.2f}, "
            f"Position Risk: {request.position_risk:.3%}, "
            f"Risk Category: {profile.risk_category.value}, "
            f"IP: {client_ip}"
        )
        
        return TradingValidationResponse(
            approved=is_valid,
            message=message,
            user_risk_category=profile.risk_category.value,
            risk_parameters=trading_params
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Critical error in trade validation for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trade validation service temporarily unavailable"
        )


@router.get("/categories", response_model=Dict[str, str])
async def get_risk_categories():
    """Get available risk categories and their descriptions.
    
    Returns all available risk categories with descriptions for
    documentation and client information purposes.
    """
    try:
        engine = UserProfilingEngine()
        
        categories = {}
        for category in RiskCategory:
            categories[category.value] = engine._get_category_description(category)
        
        return categories
    
    except Exception as e:
        logger.error(f"Failed to get risk categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve categories: {str(e)}"
        )


@router.get("/parameters/{risk_category}", response_model=Dict[str, float])
async def get_risk_parameters(
    risk_category: str,
    engine: UserProfilingEngine = Depends(get_profiling_engine)
):
    """Get trading parameters for a specific risk category.
    
    Returns the recommended trading parameters (limits, thresholds)
    for a given risk category.
    """
    try:
        # Validate risk category
        try:
            category = RiskCategory(risk_category.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid risk category: {risk_category}. Valid options: {[c.value for c in RiskCategory]}"
            )
        
        parameters = engine.get_trading_parameters(category)
        
        return parameters
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get risk parameters for {risk_category}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parameters: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for risk profiling service."""
    try:
        engine = UserProfilingEngine()
        questions = engine.get_questionnaire()
        
        return {
            "status": "healthy",
            "service": "risk-profiling",
            "questionnaire_questions": len(questions),
            "available_categories": [c.value for c in RiskCategory],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service unhealthy: {str(e)}"
        )