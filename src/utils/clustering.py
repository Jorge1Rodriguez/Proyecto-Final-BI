import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def kmeans_cluster(X: pd.DataFrame, k: int = 3, random_state: int = 42):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return model, labels, score

def dbscan_cluster(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return model, labels, n_clusters

def hierarchical_cluster(X: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels

def plot_clusters_2d(X2: pd.DataFrame, labels, title="Clusters 2D"):
    plt.figure(figsize=(6,4))
    df = X2.copy()
    df["label"] = labels
    sns.scatterplot(x=df.columns[0], y=df.columns[1], hue="label", data=df, palette="tab10", s=120)
    plt.title(title)
    plt.show()

def cluster_each_coin(X: pd.DataFrame, original_df: pd.DataFrame, method="kmeans", k=3, eps=0.5, min_samples=5, linkage="ward"):
    """
    Aplicar clusterización a cada moneda (index debe indicar moneda o puede ser agregado como columna).
    Retorna dataframe original con columna 'cluster' con identificador de cluster para cada fila.
    """
    cluster_assignments = pd.Series(index=original_df.index, dtype=int)

    coins = np.unique(original_df['coin']) if 'coin' in original_df.columns else X.index.unique()

    for coin in coins:
        # Obtener subset para la moneda
        idx = original_df[original_df['coin'] == coin].index
        data_coin = X.loc[idx]
        
        # No clusterizar si menos de 2 filas
        if data_coin.shape[0] < 2:
            cluster_assignments.loc[idx] = -1
            continue

        if method == "kmeans":
            model, labels, score = kmeans_cluster(data_coin, k=k)
            cluster_assignments.loc[idx] = labels
        elif method == "dbscan":
            model, labels, n_clusters = dbscan_cluster(data_coin, eps=eps, min_samples=min_samples)
            cluster_assignments.loc[idx] = labels
        elif method == "hierarchical":
            model, labels = hierarchical_cluster(data_coin, n_clusters=k, linkage=linkage)
            cluster_assignments.loc[idx] = labels
        else:
            raise ValueError(f"Método de clusterización no soportado: {method}")

    # Agregar columna cluster al dataframe original
    result_df = original_df.copy()
    result_df['cluster'] = cluster_assignments

    return result_df
