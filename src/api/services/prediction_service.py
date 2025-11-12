# src/api/services/prediction_service.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from datetime import datetime
from typing import Dict

from src.api.models.schemas import PredictResponse
from src.utils.rnn_models import TorchRNN as TorchRNNProject, prepare_features, make_windowed_multivariate, invert_scaling

# Directorios base (rutas absolutas)
MODEL_DIR = Path("C:/Users/jorge/Documents/Universidad/2025B/Inteligencia/Proyecto-Final-BI/models")
DATA_PROCESSED = Path("C:/Users/jorge/Documents/Universidad/2025B/Inteligencia/Proyecto-Final-BI/data/processed")

# ======================================================
# 🔹 Carga del modelo y scaler
# ======================================================
def _load_model_and_scaler(symbol: str):
    sym = symbol.lower()
    model_path = MODEL_DIR / f"model_{sym}.pth"
    scaler_path = MODEL_DIR / f"scaler_{sym}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")

    model = TorchRNNProject(
        input_size=6,
        hidden_size=128,
        num_layers=3,
        rnn_type="LSTM",
        dropout=0.2
    )
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()

    scaler = load(scaler_path)
    return model, scaler


# ======================================================
# 🔹 Preparar ventana de entrada
# ======================================================
def _load_latest_window(df: pd.DataFrame, scaler, window=60):
    cols = ["close", "log_return", "volatility", "volume", "ma_7d", "ma_30d"]
    df = df.copy().dropna(subset=cols)
    if len(df) < window:
        raise ValueError(f"No hay suficientes datos para una ventana de {window} días.")

    last = df[cols].tail(window)
    scaled = pd.DataFrame(scaler.transform(last), columns=cols, index=last.index)
    X = scaled.values[np.newaxis, :, :].astype(np.float32)
    return X, last


# ======================================================
# 🔹 Servicio principal de predicción de tendencia
# ======================================================
def predict_trend(profile: str, symbol: str, horizon: int, user_id: str = None) -> Dict:
    symbol = symbol.upper()
    df_path_api = DATA_PROCESSED / f"{symbol.lower()}_api.csv"

    if not df_path_api.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {df_path_api}")

    # 1️⃣ Cargar modelo y scaler
    model, scaler = _load_model_and_scaler(symbol)
    model.eval()

    # 2️⃣ Cargar y preparar datos
    df = pd.read_csv(df_path_api, parse_dates=["date"]).sort_values("date")
    df = prepare_features(df)
    cols = ["close", "log_return", "volatility", "volume", "ma_7d", "ma_30d"]

    df_scaled = pd.DataFrame(scaler.transform(df[cols]), columns=cols, index=df.index)

    # 3️⃣ Crear ventanas (idéntico a tu test)
    X, y = make_windowed_multivariate(df_scaled, "close", window=60)

    # 4️⃣ Predicción
    with torch.no_grad():
        X_t = torch.tensor(X[-horizon:], dtype=torch.float32)
        preds_scaled = model(X_t).cpu().numpy().flatten()

    # 5️⃣ Desescalar para análisis humano
    preds = invert_scaling(preds_scaled, scaler, df_scaled, "close")
    y_real = invert_scaling(y[-horizon:], scaler, df_scaled, "close")

    delta = preds[-1] - y_real[0]
    trend = "alcista" if delta > 0 else "bajista" if delta < 0 else "neutral"
    confidence = float(min(1.0, abs(delta) / y_real[0]))

    result = {
        "symbol": symbol,
        "horizon": horizon,
        "trend": trend,
        "confidence": round(confidence, 3),
        "scaled_change": round(float(delta / y_real[0]), 5),
        "predicted_path": preds.tolist(),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result

