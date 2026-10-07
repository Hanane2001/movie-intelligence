import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DISTRIBUTION DES NOTES
def plot_01_vote_avg(df, sp=None):
    median = df["vote_average"].median()
    plt.figure(figsize=(8, 5))
    plt.hist(df["vote_average"].dropna(), bins=20, edgecolor="black")
    plt.axvline(median, color="red", linestyle="--", linewidth=1.5, label=f"Médiane : {median:.2f}")
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de films")
    plt.title("Distribution des notes des films", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 2. DISTRIBUTION DE LA POPULARITÉ
def plot_02_popularity(df, sp=None):
    median = df["popularity"].median()
    plt.figure(figsize=(8, 5))
    plt.hist(df["popularity"].dropna(), bins=20, edgecolor="black")
    plt.axvline(median, color="red", linestyle="--", linewidth=1.5, label=f"Médiane : {median:.2f}")
    plt.xlabel("Popularité")
    plt.ylabel("Nombre de films")
    plt.title("Distribution de la popularité des films", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 3. FILMS PAR GENRE
def plot_03_genres(df, sp=None):
    genre_counts = (df["genres"].explode().dropna().value_counts().sort_values(ascending=True))
    plt.figure(figsize=(9, 6))
    genre_counts.plot(kind="barh", edgecolor="black")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genre")
    plt.title("Nombre de films par genre", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 4. SORTIES PAR ANNÉE
def plot_04_movies_year(df, sp=None):
    year_counts = (df["release_date"].dropna().dt.year.value_counts().sort_index()) # ou year_counts = (df["annee"].dropna().value_counts().sort_index())
    plt.figure(figsize=(12, 5))
    plt.plot(year_counts.index, year_counts.values, marker="o")
    plt.xlabel("Année")
    plt.ylabel("Nombre de films")
    plt.title("Nombre de films sortis par année", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 5. DISTRIBUTION DE LA DURÉE
def plot_05_runtime(df, sp=None):
    median = df["runtime"].median()
    plt.figure(figsize=(8, 5))
    plt.hist(df["runtime"].dropna(), bins=30, edgecolor="black")
    plt.axvline(median, color="red", linestyle="--", linewidth=1.5, label=f"Médiane : {median:.0f} min")
    plt.xlabel("Durée (minutes)")
    plt.ylabel("Nombre de films")
    plt.title("Distribution de la durée des films", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 6. BUDGET / REVENUS
def plot_06_budget_rev(df, sp=None):
    data = df[["budget", "revenue"]].copy()
    data = data[(data["budget"] > 0) & (data["revenue"] > 0)]
    plt.figure(figsize=(9, 6))
    plt.scatter(data["budget"], data["revenue"], alpha=0.5)
    plt.xlabel("Budget")
    plt.ylabel("Revenus")
    plt.title("Relation entre le budget et les revenus", fontsize=14, fontweight="bold")
    max_value = max(data["budget"].max(), data["revenue"].max())
    plt.plot([0, max_value], [0, max_value], linestyle="--", label="Revenus = Budget")
    plt.legend()
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 7. VOTES / POPULARITÉ
def plot_07_votes_pop(df, sp=None):
    data = df[["vote_count", "popularity"]].dropna()
    plt.figure(figsize=(9, 6))
    plt.scatter(data["vote_count"], data["popularity"], alpha=0.5)
    plt.xlabel("Nombre de votes")
    plt.ylabel("Popularité")
    plt.title("Relation entre le nombre de votes et la popularité", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 8. FILMS PAR DÉCENNIE
def plot_08_decade(df, sp=None):
    if "decennie" in df.columns:
        decade_counts = (df["decennie"].dropna().value_counts().sort_index())
    else:
        if "annee" in df.columns:
            years = df["annee"]
        else:
            years = df["release_date"].dt.year
        decade_counts = ((years // 10 * 10).dropna().astype(int).value_counts().sort_index())
    plt.figure(figsize=(9, 5))
    decade_counts.plot(kind="bar", edgecolor="black")
    plt.xlabel("Décennie")
    plt.ylabel("Nombre de films")
    plt.title("Nombre de films par décennie", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 9. BOXPLOTS
def plot_09_boxplots(df, sp=None):
    columns = ["vote_average", "popularity", "runtime"]
    data = df[columns].copy()
    plt.figure(figsize=(10, 6))
    data.boxplot()
    plt.title("Boxplots des principales variables numériques", fontsize=14, fontweight="bold")
    plt.ylabel("Valeur")
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# 10. MATRICE DE CORRÉLATION
def plot_10_corr(df, sp=None):
    columns = ["budget", "revenue", "popularity", "vote_average", "vote_count", "runtime", "annee"]
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if sp:
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.show()


# EXÉCUTION DE TOUTES LES VISUALISATIONS
def run_eda(df, od="../notebooks/eda_figs"):
    os.makedirs(od, exist_ok=True)
    plot_01_vote_avg(df, f"{od}/01_vote_avg.png")
    plot_02_popularity(df, f"{od}/02_popularity.png")
    plot_03_genres(df, f"{od}/03_genres.png")
    plot_04_movies_year(df, f"{od}/04_year.png")
    plot_05_runtime(df, f"{od}/05_runtime.png")
    plot_06_budget_rev(df, f"{od}/06_budget_rev.png")
    plot_07_votes_pop(df, f"{od}/07_votes_pop.png")
    plot_08_decade(df, f"{od}/08_decade.png")
    plot_09_boxplots(df, f"{od}/09_boxplots.png")
    plot_10_corr(df, f"{od}/10_corr.png")
    print(f"\nAll charts saved to: {os.path.abspath(od)}/")
    return df
