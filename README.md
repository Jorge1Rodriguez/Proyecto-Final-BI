# Proyecto Final — Análisis y Predicción de Criptomonedas

## Resumen

- Proyecto de curso para el análisis y modelado de series temporales de criptomonedas (Bitcoin, Ethereum, Binance Coin).
- Contiene recolección y procesamiento de datos, análisis (clusterización), modelos RNN/LSTM entrenados y notebooks reproducibles.
- Autor: Jorge (repositorio: `Proyecto-Final-BI`).

## Estructura del repositorio

- `data/`
  - `raw/` — archivos originales descargados (por ejemplo `coin_Bitcoin.csv`, `coin_Ethereum.csv`, `coin_BinanceCoin.csv`).
  - `processed/` — datos preprocesados y consolidados (`*_api.csv`, `*_csv.csv`).
- `notebooks/` — notebooks de Jupyter:
  - `sprint1_data_collection.ipynb` — recolección y preparación de datos.
  - `sprint2_clustering_timeseries.ipynb` — análisis y clusterización de series temporales.
- `src/` — código fuente y utilidades:
  - `src/utils/processing.py` — procesamiento de datos.
  - `src/utils/clustering.py` — funciones y pipeline de clusterización.
  - `src/utils/rnn_models.py` — definiciones de modelos RNN/LSTM y utilidades de entrenamiento.
  - `src/api/`, `src/dashboard/` — módulos (si están activos) para servir API o dashboard.
- `models/` — artefactos de modelos y scalers (por ejemplo `model_btc.pth`, `scaler_btc.joblib`, `btc_scaled.csv`).
- `salidas/` — resultados y gráficos de experimentos (`clusterizacion/`, `rnn/`).
- `requirements.txt` — dependencias de Python.

## Requisitos

- Python 3.9+ recomendado. Ajusta si el `requirements.txt` lo requiere.
- Git (para clonar) y PowerShell en Windows (instrucciones incluidas más abajo).
- Recomendado: GPU para reentrenamiento de modelos grandes.

## Instalación (PowerShell)

1. Clonar el repositorio (si no lo tienes):

```powershell
git clone https://github.com/Jorge1Rodriguez/Proyecto-Final-BI.git
cd Proyecto-Final-BI
```

2. Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. (Opcional) Instalar Jupyter:

```powershell
pip install jupyterlab notebook
```

## Uso — Notebooks

- Abrir Jupyter Lab o Notebook desde la raíz del repo:

```powershell
jupyter lab
```
o

```powershell
jupyter notebook
```

- Ejecutar `notebooks/sprint1_data_collection.ipynb` para reproducir la recolección y limpieza de datos.
- Ejecutar `notebooks/sprint2_clustering_timeseries.ipynb` para análisis y clusterización.

## Descripción de los datos

- Fuentes: CSVs históricos y descargas vía API. Archivos con sufijos `_api.csv` provienen de llamadas a API.
- Columnas típicas: timestamp/date, open, high, low, close, volume, etc.
- Preprocesamiento: limpieza de nulos, normalización/escalado (scalers en `models/`), creación de ventanas para entrenamiento de series temporales.

## Modelos y artefactos

- Modelos y scalers disponibles en `models/` (por ejemplo `model_btc.pth`, `lstm_btc.pth`, `scaler_btc.joblib`, `btc_scaled.csv`).
- Flujo de uso general para inferencia:
  1. Cargar scaler (joblib).
  2. Preprocesar las series (ventanas) con el mismo esquema usado en entrenamiento.
  3. Cargar pesos del modelo (p. ej. `torch.load(...)`) y pasar los tensores.
  4. Desescalar las predicciones y guardar/visualizar los resultados.

## Reproducibilidad

- Usar las mismas versiones en `requirements.txt`.
- Ejecutar las celdas de `sprint1_data_collection.ipynb` para regenerar `data/processed/` antes de entrenar o evaluar.
- Guardar salidas en `salidas/` para comparar experimentos.

## Resultados y salidas

- Gráficos y métricas: `salidas/clusterizacion/` y `salidas/rnn/`.
- Modelos entrenados en `models/` para inferencia rápida.
