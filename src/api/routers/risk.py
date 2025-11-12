# src/api/routers/risk.py
from fastapi import APIRouter, HTTPException, Body
from src.api.models.schemas import RiskRequest, RiskResponse
from src.api.services.risk_service import evaluate_risk

router = APIRouter()

@router.post("/", response_model=RiskResponse)
async def risk(req: RiskRequest = Body(...)):
    try:
        res = evaluate_risk(req.profile, req.symbol)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
