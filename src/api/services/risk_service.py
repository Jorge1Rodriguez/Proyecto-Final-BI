# src/api/services/risk_service.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
DATA_PROCESSED = Path("C:/Users/jorge/Documents/Universidad/2025B/Inteligencia/Proyecto-Final-BI/data/processed")

def _max_drawdown(ts: pd.Series):
    cum_max = ts.cummax()
    drawdown = (cum_max - ts) / cum_max
    return float(drawdown.max())

def evaluate_risk(profile: str, symbol: str) -> Dict:
    symbol = symbol.upper()
    df_path = DATA_PROCESSED / f"{symbol.lower()}_api.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {df_path}")

    df = pd.read_csv(df_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    # usar log_returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # volatilidad histórica 30 días (std of log_return)
    vol_30 = float(df["log_return"].rolling(30).std().dropna().iloc[-1])
    # drawdown máximo en último año (365 días) o en lo disponible
    window = min(365, len(df))
    recent = df["close"].tail(window)
    max_dd = _max_drawdown(recent)

    # heurística de clasificación
    if vol_30 < 0.02 and max_dd < 0.2:
        risk = "bajo"
    elif vol_30 < 0.05 and max_dd < 0.4:
        risk = "medio"
    else:
        risk = "alto"

    res = {
        "symbol": symbol,
        "risk_level": risk,
        "volatility": round(vol_30, 6),
        "max_drawdown": round(max_dd, 6)
    }

    return res
