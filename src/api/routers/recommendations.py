# src/api/routers/recommendations.py
from fastapi import APIRouter
from src.api.services.recommendation_service import recommendations_for_profile

router = APIRouter()

@router.get("/")
async def recommendations(profile: str = "moderado"):
    res = recommendations_for_profile(profile)
    return res
