from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from app.agents.reputation_agent import ReputationAgent

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize reputation agent
reputation_agent = ReputationAgent()


class CheckAccountRequest(BaseModel):
    account_hash: str
    bank_code: str | None = None
    business_name: str | None = None


@router.post("/check-account")
async def check_account(request: CheckAccountRequest) -> Dict[str, Any]:
    """
    Check account reputation and fraud reports
    
    This endpoint is called by the backend to verify account trustworthiness
    """
    try:
        logger.info(f"Checking account: {request.account_hash[:8]}...")
        
        # For now, return basic reputation check
        # In production, this would integrate with the full reputation system
        result = {
            "trust_score": 75,
            "risk_level": "low",
            "fraud_reports": {
                "total": 0,
                "recent_30_days": 0
            },
            "verified_business_id": None,
            "flags": []
        }
        
        logger.info(f"Account check completed: {request.account_hash[:8]}...")
        return result
        
    except Exception as e:
        logger.error(f"Account check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
