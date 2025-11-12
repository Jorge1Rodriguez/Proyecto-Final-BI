# src/api/main.py
from fastapi import FastAPI
from src.api.routers import predict, risk, recommendations, health, trend_scores


app = FastAPI(
    title="Crypto Prediction API - Proyecto Final BI",
    version="1.0.0",
    description="API para predicción, riesgo y recomendaciones basada en modelos LSTM."
)

app.include_router(predict.router, prefix="/predict")
app.include_router(risk.router, prefix="/risk")
app.include_router(recommendations.router, prefix="/recommendations")
app.include_router(health.router, prefix="/health")
app.include_router(trend_scores.router, prefix="/trend-scores")


@app.get("/")
async def root():
    return {"message": "Crypto Prediction API - Proyecto Final BI"}
