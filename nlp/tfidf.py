import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from extraction.tmdb_api import load_data_v3 

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text) # supprimer les caracteres speciaux
    text = re.sub(r"\s+", " ", text) # supprimer les espaces multiples
    return text.strip()

def prepare_text(df):
    df = df.copy()
    df["overview"] = df["overview"].fillna("")
    df["overview_clean"] = df["overview"].apply(clean_text)
    return df

def create_tfidf(df, max_features=5000, ngram_range=(1, 2), stop_words="english"):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    X_tfidf = vec.fit_transform(df["overview_clean"])
    return X_tfidf, vec

def analyze_tfidf(X_tfidf, vectorizer):
    print("Analyse TF-IDF:")
    print("Dimensions de la matrice:", X_tfidf.shape)
    print ("Nombre de documents:", X_tfidf.shape[0])
    print("Nombre de features:", X_tfidf.shape[1])
    print("Nombre de termes dans le vocabulaire:", len(vectorizer.get_feature_names_out()))

def top_terms(X_tfidf, vec, n=20):
    terms = vec.get_feature_names_out()
    mean_scores = np.asarray(X_tfidf.mean(axis=0)).ravel()
    ind = mean_scores.argsort()[::-1][:n]
    print(f"Top {n} termes representatifs: ")
    for i in ind:
        print(f"{terms[i]:25s} -> {mean_scores[i]:.4f}")

def experiment_tfidf(df):
    print("Experimentation max_features: ")
    for m in [1000, 3000, 5000, 10000]:
        vec = TfidfVectorizer(max_features=m, ngram_range=(1, 1), stop_words="english")
        X = vec.fit_transform(df["overview_clean"])
        print(f"max_features={m} -> {X.shape}")
    print("Experimentation ngram_range: ")
    for ng in [(1, 1), (1, 2), (1, 3)]:
        vec = TfidfVectorizer(max_features=5000, ngram_range=ng, stop_words="english")
        X = vec.fit_transform(df["overview_clean"])
        print(f"ngram_range={ng} -> {X.shape}")


def main_TF_IDF():
    path = "../data/features/movies_feature.csv"
    df = load_data_v3(path)
    print("Shape initiale: ", df.shape)
    df = prepare_text(df)
    print("Overviews vides: ", (df["overview_clean"] == "").sum())
    X_tfidf, vectorizer = create_tfidf(df, max_features=5000, ngram_range=(1, 2), stop_words="english")
    analyze_tfidf(X_tfidf, vectorizer)
    top_terms(X_tfidf, vectorizer, n=20)
    experiment_tfidf(df)
