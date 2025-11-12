# src/api/routers/trend_scores.py
from fastapi import APIRouter
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter()
DATA_PROCESSED = Path("data/processed")
SYMBOLS = ["BTC", "ETH", "BNB"]

def compute_trend_score(df: pd.DataFrame):
    # Momentum simple: mean of log_return last 14 days standardized by std
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    recent = df["log_return"].tail(14).dropna()
    if recent.empty:
        return 0.0
    return float(recent.mean() / (recent.std() + 1e-9))

@router.get("/")
async def trend_scores():
    out = {}
    for s in SYMBOLS:
        path = DATA_PROCESSED / f"{s.lower()}_api.csv"
        if not path.exists():
            out[s] = {"error": "no data"}
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        score = compute_trend_score(df)
        out[s] = {"trend_score": score}
    return out
