import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from extraction.tmdb_api import load_data_v2

def prepare_clustering_data(df):
    features = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "annee",
        "nombre_genres",
        "nombre_keywords"
    ]
    X = df[features].copy()
    print("VARIABLES UTILISEES: ")
    print(features)
    print("VALEURS MANQUANTES: ")
    print(X.isna().sum())
    X = X.fillna(X.median())
    return X

def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("STANDARDISATION: ")
    print(f"shape: {X_scaled.shape}")
    return X_scaled, scaler

def test_k_values(X_scaled, k_values=range(2, 8)):
    results = []
    print("TEST DES VALEURS DE K: ")
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(
            X_scaled, labels
        )
        results.append({
            "K": k,
            "Silhouette": score
        })
        print(f"K: {k} | silhouette score: {score}")
    res_df = pd.DataFrame(results)
    return res_df


def select_best_k(res_df):
    br = res_df.loc[
        res_df["Silhouette"].idxmax()
    ]

    bk = int(br["K"])
    bs = br["Silhouette"]
    print("MEILLEUR K: ")
    print("Meilleur K: ", bk)
    print(f"Meilleur Silhouette Score: {bs:.4f}")
    return bk

def apply_kmeans(X_scaled, df, bk):
    kmeans = KMeans(
        n_clusters=bk,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["cluster"] = labels
    print("CLUSTERS: ")
    print(df["cluster"].value_counts().sort_index())
    return df, kmeans

def interpret_clusters(df):
    features = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "annee",
        "nombre_genres",
        "nombre_keywords"
    ]

    print("CARACTERISTIQUE DES CLUSTERS: ")
    cluster_mean = df.groupby("cluster")[features].mean()
    print(cluster_mean.round(2))
    return cluster_mean

def plot_clusters(df, output_dir="../data/processed/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="popularity",
        y="vote_average",
        hue="cluster",
        palette="viridis",
        s=60,
    )
    plt.title("Cluster de films")
    plt.xlabel("Popularité")
    plt.ylabel("Note moyenne")
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        "clusters_movies.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_silhouette(df, output_dir="../data/processed/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="K",
        y="Silhouette",
        marker="o",
    )
    plt.title("Silhouette Score selon K")
    plt.xlabel("Nombre de clusters(K)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        "silhouette_score.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    print("CLUSTERING - MOVIE INTELLIGENCE")
    path = "../data/processed/movies_clean.csv"
    df = load_data_v2(path)
    X = prepare_clustering_data(df)
    X_scaled, scaler = standardize_data(X)
    results_df = test_k_values(X_scaled, k_values=range(2, 8))
    print("\nResultats: ")
    print(results_df.round(4))
    plot_silhouette(results_df)
    best_k = select_best_k(results_df)
    df_clustered, kmeans = apply_kmeans(X_scaled,df,best_k)
    cluster_means = interpret_clusters(df_clustered)
    plot_clusters(df_clustered)
    print("CLUSTERING TERMINE")
    return (df_clustered, results_df, cluster_means, kmeans, scaler)


if __name__ == "__main__":
    (df_clustered, results_df, cluster_means, kmeans, scaler) = main()