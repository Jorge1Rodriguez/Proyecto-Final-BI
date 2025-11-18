import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
from typing import Optional

# clustering imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


st.set_page_config(page_title="Crypto Dashboard", layout="wide")

st.title("Crypto Dashboard")

st.markdown("Interfaz: estadísticas, recomendaciones y visualizaciones interactivas.")

# Sidebar: API base and global symbol (UNICA entrada para base_url)
BASE_URL = st.sidebar.text_input("Base URL de la API", value="http://localhost:8000", key="base_url")
if BASE_URL.endswith("/"):
    BASE_URL = BASE_URL[:-1]

st.sidebar.markdown("---")

@st.cache_data(ttl=60)
def call_get_cached(path: str, params: Optional[dict] = None):
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def call_post(path: str, json_data: Optional[dict] = None):
    try:
        r = requests.post(f"{BASE_URL}{path}", json=json_data, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def load_processed_csv(symbol: str) -> Optional[pd.DataFrame]:
    p = Path("data/processed") / f"{symbol.lower()}_api.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"] if "date" in pd.read_csv(p, nrows=0).columns else None)
        return df
    except Exception:
        return None


tabs = st.tabs(["Overview", "Recommendations", "Clustering", "Predict", "Risk", "Health"])

# ------------------ OVERVIEW ------------------
with tabs[0]:
    st.header("Overview")
    symbol = st.selectbox("Símbolo", ["BTC", "ETH", "BNB"], index=0, key="overview_symbol")
    df = load_processed_csv(symbol)
    if df is None:
        st.warning("No se encontró CSV preprocesado en `data/processed/`. Ejecuta la recolección/preprocesamiento.")
    else:
        # asegurar fecha y ordenar
        if "date" in df.columns:
            df = df.sort_values("date")
        # mostrar sólo lo básico: último precio y gráfico de cierre (últimos 90 días)
        if "close" in df.columns:
            last_price = df["close"].dropna().iloc[-1]
            last_date = pd.to_datetime(df["date"]).dropna().iloc[-1] if "date" in df.columns else None
            cols = st.columns(2)
            cols[0].metric("Último precio", f"{last_price:.2f}")
            cols[1].metric("Última fecha", str(last_date.date()) if last_date is not None else "N/A")

            # gráfico simple: últimos 90 días si hay suficientes datos
            if "date" in df.columns:
                df_plot = df.copy()
                df_plot = df_plot.dropna(subset=["date", "close"]).sort_values("date")
                df_plot = df_plot.tail(90)
                fig = px.line(df_plot, x="date", y="close", title=f"{symbol} - Precio de cierre (últimos {len(df_plot)} días)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("El CSV procesado no contiene la columna `close`.")

# ------------------ RECOMMENDATIONS ------------------
with tabs[1]:
    st.header("Recomendaciones")
    profile = st.selectbox("Perfil usuario", ["conservador", "moderado", "agresivo"], index=1, key="rec_profile")
    if st.button("Cargar recomendaciones", key="load_recs"):
        res = call_get_cached("/recommendations/", params={"profile": profile})
        if isinstance(res, dict) and "recommendations" in res:
            recs = res["recommendations"]
        elif isinstance(res, list):
            recs = res
        else:
            # try direct dict of symbols
            recs = []
            if isinstance(res, dict):
                # transform to list
                for k, v in res.items():
                    if isinstance(v, dict) and "expected_trend" in v:
                        recs.append({"symbol": k, **v})

        if not recs:
            st.warning("No se encontraron recomendaciones o respuesta inesperada from API.")
        else:
            df_rec = pd.DataFrame(recs)
            st.dataframe(df_rec)
            # chart
            if "symbol" in df_rec.columns and "confidence" in df_rec.columns:
                fig = px.bar(df_rec.sort_values("confidence", ascending=False), x="symbol", y="confidence", color="risk_level" if "risk_level" in df_rec.columns else None, title="Recomendaciones por confianza")
                st.plotly_chart(fig, use_container_width=True)

            # download
            csv = df_rec.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", data=csv, file_name=f"recommendations_{profile}.csv", mime="text/csv")

# ------------------ CLUSTERING ------------------
with tabs[2]:
    st.header("Clusterización")
    symbol = st.selectbox("Símbolo para clusterizar", ["BTC", "ETH", "BNB"], index=0, key="tab_cluster_symbol")
    method = st.selectbox("Método", ["kmeans", "dbscan", "hierarchical"], index=0, key="tab_cluster_method")
    features = st.multiselect("Features", ["open", "high", "low", "close", "volume", "market_cap"], default=["open", "high", "low", "close", "volume"], key="tab_cluster_features")
    k = st.number_input("k (kmeans/jerárquico)", min_value=2, max_value=10, value=3, key="tab_cluster_k")
    eps = st.number_input("eps (DBSCAN)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="tab_cluster_eps")
    min_samples = st.number_input("min_samples (DBSCAN)", min_value=1, max_value=20, value=5, key="tab_cluster_min_samples")

    if st.button("Ejecutar clustering", key="run_cluster"):
        df_local = load_processed_csv(symbol)
        if df_local is None:
            st.error("CSV procesado no encontrado en `data/processed`.")
        else:
            if not set(features).issubset(set(df_local.columns)):
                st.error("Algunas features no están en el CSV procesado.")
            else:
                X = df_local[features].dropna()
                scaler = StandardScaler()
                Xs = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
                if method == "kmeans":
                    model = KMeans(n_clusters=int(k), random_state=0)
                    labels = model.fit_predict(Xs)
                elif method == "dbscan":
                    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                    labels = model.fit_predict(Xs)
                else:
                    model = AgglomerativeClustering(n_clusters=int(k))
                    labels = model.fit_predict(Xs)

                pca = PCA(n_components=2)
                Xp = pca.fit_transform(Xs)
                df_viz = pd.DataFrame(Xp, columns=["PC1", "PC2"], index=Xs.index)
                df_viz["cluster"] = labels.astype(str)
                    # Solo usar hover_data si la columna existe en df_viz
                fig = px.scatter(df_viz, x="PC1", y="PC2", color="cluster", title=f"{symbol} clusters ({method})")
                st.plotly_chart(fig, use_container_width=True)
                st.write(pd.Series(labels).value_counts().rename_axis("cluster").reset_index(name="count"))

# ------------------ PREDICT ------------------
with tabs[3]:
    st.header("Predicción")
    profile = st.selectbox("Perfil (predict)", ["conservador", "moderado", "agresivo"], index=1, key="tab_pred_profile")
    symbol = st.selectbox("Símbolo (predict)", ["BTC", "ETH", "BNB"], index=0, key="tab_pred_symbol")
    horizon = st.number_input("Horizonte (días)", min_value=1, max_value=365, value=7, key="tab_pred_horizon")
    if st.button("Predecir (API)", key="tab_pred_run"):
        payload = {"profile": profile, "symbol": symbol, "horizon": int(horizon)}
        res = call_post("/predict/", json_data=payload)
        if res.get("error"):
            st.error(res["error"])
        else:
            st.table(pd.DataFrame([res]).T.rename(columns={0: "value"}))
            predicted = res.get("predicted_path")
            if predicted:
                # Mostrar solo la tendencia predicha
                pred_dates = pd.date_range(start=datetime.today(), periods=len(predicted), freq="D")
                df_pred = pd.DataFrame({"date": pred_dates, "predicted": predicted})
                fig = px.line(df_pred, x="date", y="predicted", title=f"{symbol} Tendencia Predicha")
                st.plotly_chart(fig, use_container_width=True)

# ------------------ RISK ------------------
with tabs[4]:
    st.header("Riesgo")
    profile = st.selectbox("Perfil (risk)", ["conservador", "moderado", "agresivo"], index=1, key="tab_risk_profile")
    symbol = st.selectbox("Símbolo (risk)", ["BTC", "ETH", "BNB"], index=0, key="tab_risk_symbol")
    if st.button("Evaluar riesgo", key="tab_risk_run"):
        with st.spinner("Evaluando riesgo, por favor espera..."):
            res = call_post("/risk/", json_data={"profile": profile, "symbol": symbol})

        if not res:
            st.error("Respuesta vacía de la API de riesgo.")
        elif isinstance(res, dict) and res.get("error"):
            st.error(res.get("error"))
        else:
            # Si la API devuelve un dict con medidas comunes, mostrarlas como métricas
            if isinstance(res, dict):
                risk_score = res.get("risk_score") or res.get("score") or res.get("risk")
                var = res.get("var") or res.get("value_at_risk")
                volatility = res.get("volatility") or res.get("volatility_pct")

                cols = st.columns(3)
                if risk_score is not None:
                    cols[0].metric("Risk score", f"{risk_score}")
                else:
                    cols[0].write("")

                if var is not None:
                    cols[1].metric("Value at Risk", f"{var}")
                else:
                    cols[1].write("")

                if volatility is not None:
                    try:
                        cols[2].metric("Volatility", f"{float(volatility):.4f}")
                    except Exception:
                        cols[2].write(str(volatility))
                else:
                    cols[2].write("")

                # Mostrar detalles en forma tabular
                try:
                    df_res = pd.json_normalize(res)
                    st.markdown("**Detalle de la evaluación de riesgo**")
                    # Si es un solo registro, mostrar en forma vertical para facilitar lectura
                    if df_res.shape[0] == 1:
                        st.dataframe(df_res.T.rename(columns={0: 'value'}))
                    else:
                        st.dataframe(df_res)
                except Exception:
                    st.write(res)

                # Si la API devuelve una serie de retornos simulados, graficarla
                sim_key_candidates = ["simulated_returns", "sim_returns", "simulations", "returns"]
                for key in sim_key_candidates:
                    if key in res and isinstance(res[key], (list, tuple)) and len(res[key]) > 0:
                        try:
                            sr = pd.Series(res[key])
                            fig = px.histogram(sr, nbins=50, title="Distribución de retornos simulados")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                        break

                # Mostrar sugerencias si las provee la API
                if "suggestions" in res:
                    st.markdown("**Sugerencias / Acciones recomendadas**")
                    sug = res["suggestions"]
                    if isinstance(sug, list):
                        for s in sug:
                            st.write(f"- {s}")
                    else:
                        st.write(sug)

            elif isinstance(res, list):
                st.markdown("**Listado de resultados de riesgo**")
                st.dataframe(pd.DataFrame(res))
            else:
                st.json(res)

# ------------------ HEALTH ------------------
with tabs[5]:
    st.header("Health")
    if st.button("Comprobar API"):
        res = call_get_cached("/")
        st.json(res)

st.sidebar.markdown("---")
st.sidebar.info("Ejecución: `uvicorn src.api.main:app --reload --port 8000` para la API y `streamlit run src/dashboard/streamlit_app.py` para este dashboard.")
