import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_processed(path: str) -> pd.DataFrame:
    """
    Carga un CSV procesado, parsea la columna 'date' a datetime, ordena por fecha
    y devuelve un DataFrame reindexado.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_returns_volatility(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Agrega columnas útiles para análisis:
      - return: pct_change() diario
      - vol_roll_{window}d: volatilidad (std de returns * sqrt(window))
      - close_ma_{window}d: media móvil del cierre
      - volume_ma_{window}d: media móvil del volumen
    Devuelve copia con NaN iniciales eliminados.
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df[f"vol_roll_{window}d"] = df["return"].rolling(window).std() * np.sqrt(window)
    df[f"close_ma_{window}d"] = df["close"].rolling(window).mean()
    if "volume" in df.columns:
        df[f"volume_ma_{window}d"] = df["volume"].rolling(window).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def features_for_clustering(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calcula características agregadas por 'coin' sobre la ventana más reciente 'window'
    Ejemplo de features:
      - close_mean, close_std, return_mean, vol_mean, vol_std, market_cap_mean
    Devuelve DataFrame indexado por coin.
    """
    feats = []
    for coin, g in df.groupby("coin"):
        g = g.sort_values("date").tail(window)
        if g.empty:
            continue
        feats.append({
            "coin": coin,
            "close_mean": g["close"].mean(),
            "close_std": g["close"].std(),
            "return_mean": g["close"].pct_change().mean(),
            "vol_mean": g["volume"].mean() if "volume" in g else np.nan,
            "vol_std": g["volume"].std() if "volume" in g else np.nan,
            "market_cap_mean": g["market_cap"].mean() if "market_cap" in g else np.nan,
        })
    feats_df = pd.DataFrame(feats).set_index("coin")
    return feats_df

def scale_features(df: pd.DataFrame):
    """
    Escala las columnas numéricas con StandardScaler.
    Retorna (df_scaled, scaler)
    """
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return numeric, scaler
    scaled_arr = scaler.fit_transform(numeric.fillna(0))
    scaled = pd.DataFrame(scaled_arr, index=numeric.index, columns=numeric.columns)
    return scaled, scaler
