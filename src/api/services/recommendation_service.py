# src/api/services/recommendation_service.py
from src.api.services.prediction_service import predict_trend
from src.api.services.risk_service import evaluate_risk
from typing import List, Dict

SYMBOLS = ["BTC", "ETH", "BNB"]

def recommendations_for_profile(profile: str) -> Dict:
    recs = []
    for s in SYMBOLS:
        try:
            pred = predict_trend(profile=profile, symbol=s, horizon=30, user_id=None)
            risk = evaluate_risk(profile=profile, symbol=s)
            recs.append({
                "symbol": s,
                "expected_trend": pred["trend"],
                "confidence": pred["confidence"],
                "risk_level": risk["risk_level"]
            })
        except Exception as e:
            recs.append({
                "symbol": s,
                "expected_trend": "error",
                "confidence": 0.0,
                "risk_level": "unknown",
                "error": str(e)
            })
    # filtrar/ordenar según perfil
    if profile.lower() == "conservador":
        # priorizar menor riesgo
        recs = sorted(recs, key=lambda r: (0 if r["risk_level"]=="bajo" else 1, -r["confidence"]))
    elif profile.lower() == "agresivo":
        # priorizar confianza positiva (alcista) y riesgo medio/alto
        recs = sorted(recs, key=lambda r: (-r["confidence"], 0 if r["risk_level"]=="alto" else 1))
    else:
        recs = sorted(recs, key=lambda r: -r["confidence"])

    return {
        "profile": profile,
        "recommendations": recs
    }
