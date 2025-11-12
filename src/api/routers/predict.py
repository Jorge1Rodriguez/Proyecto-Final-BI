# src/api/routers/predict.py
from fastapi import APIRouter, HTTPException, Body
from src.api.models.schemas import PredictRequest, PredictResponse
from src.api.services.prediction_service import predict_trend

router = APIRouter()

@router.post("/", response_model=PredictResponse, summary="Predicción de tendencia de criptomoneda")
async def predict(req: PredictRequest = Body(...)):
    """
    Retorna la tendencia general (alcista, bajista o neutral) de una criptomoneda.
    No predice valores exactos, solo la dirección probable del mercado.
    """
    try:
        return predict_trend(req.profile, req.symbol, req.horizon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
