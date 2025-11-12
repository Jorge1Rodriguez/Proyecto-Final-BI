# src/api/models/schemas.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# ==============================
# 🔹 Predicción de tendencias
# ==============================
class PredictRequest(BaseModel):
    profile: str        # 'conservador', 'moderado', 'agresivo'
    symbol: str         # 'BTC' | 'ETH' | 'BNB' (case-insensitive)
    horizon: int        # horizonte en días (ej. 7, 30)


class PredictResponse(BaseModel):
    symbol: str
    horizon: int
    trend: str
    confidence: float
    scaled_change: float
    timestamp: datetime
    predicted_path: Optional[List[float]] = None  # nueva



# ==============================
# 🔹 Evaluación de riesgo
# ==============================
class RiskRequest(BaseModel):
    profile: str
    symbol: str


class RiskResponse(BaseModel):
    symbol: str
    risk_level: str      # 'bajo' | 'medio' | 'alto'
    volatility: float
    max_drawdown: float


# ==============================
# 🔹 Recomendaciones personalizadas
# ==============================
class RecommendationItem(BaseModel):
    symbol: str
    expected_trend: str
    confidence: float
    risk_level: str


class RecommendationsResponse(BaseModel):
    profile: str
    recommendations: List[RecommendationItem]


# ==============================
# 🔹 Historial y salud del sistema
# ==============================
class HistoryQuery(BaseModel):
    user_id: Optional[str] = None
    limit: Optional[int] = 100


class HealthResponse(BaseModel):
    status: str
    timestamp: str
