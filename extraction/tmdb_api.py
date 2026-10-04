import os
import json
import requests as rq
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
API_KEY = os.getenv("TMDB_API_KEY")

def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    res = rq.get(url, params={"api_key": API_KEY}, timeout=30)
    res.raise_for_status()
    return res.json()

def get_movie_keywords(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/keywords"
    res = rq.get(url, params={"api_key": API_KEY }, timeout=30)
    res.raise_for_status()
    data = res.json()
    return data.get("keywords", [])

def load_data(url, total_movies=3000):
    out = "data/raw/movies_raw.json"
    os.makedirs("data/raw", exist_ok=True)

    try:
        print("Recuperation des films...")
        movies = []
        page = 1
        while len(movies) < total_movies:
            res = rq.get(url, params={"api_key": API_KEY, "page": page}, timeout=30)
            res.raise_for_status()
            data = res.json()

            for movie in data["results"]:

                movie_id = movie["id"]
                details = get_movie_details(movie_id)
                keywords = get_movie_keywords(movie_id)
                smv = {
                    "movie_id": movie_id,
                    "title": details.get("title"),
                    "overview": details.get("overview"),
                    "release_date": details.get("release_date"),
                    "runtime": details.get("runtime"),
                    "original_language": details.get("original_language"),
                    "genres": details.get("genres"),
                    "keywords": keywords,
                    "budget": details.get("budget"),
                    "revenue": details.get("revenue"),
                    "popularity": details.get("popularity"),
                    "vote_average": details.get("vote_average"),
                    "vote_count": details.get("vote_count")
                }
                movies.append(smv)
                if len(movies) >= total_movies:
                    break
            print(f"Film recupere: {len(movies)}")
            page += 1
        movies = movies[:total_movies]
        print("\nPremiers films:")
        for movie in movies[:5]:
            print(movie["movie_id"], "-", movie["title"])

        with open(out, "w", encoding="utf-8") as f:
            json.dump(movies, f, ensure_ascii=False, indent=4)

        print("\nExtraction terminee")
        print(f"Saved file: {out}")
        print(f"Nombre de films: {len(movies)}")
        print(f"Nombre demande: {total_movies}")

    except rq.Timeout:
        print("Erreur: timeout")
    except rq.HTTPError as e:
        print(f"Erreur HTTP: {e}")
    except rq.RequestException as e:
        print(f"Erreur dans la requete: {e}")
    except Exception as e:
        print(f"Erreur: {e}")

def Analyse_data(df):
    print("=== Shape ===")
    print(df.shape)
    print("=== Info ===")
    print(df.info())
    print("=== Describe ===")
    print(df.describe())

def Nombre_duplicate(df):
    res = df.duplicated().sum()
    print("Nombre de doublons:", res)
    if res > 0:
        df = df.drop_duplicates()
        print("doublons suprimes")
    return df

def feature(df):
    df["annee"] = df["release_date"].dt.year
    df["mois"] = df["release_date"].dt.month
    df["decennie"] = pd.cut(df["release_date"].dt.year, bins=[1979, 2000, 2010, 2020, 2026], labels=["1980-2000", "2001-2010", "2011-2020", "2021-2026"])
    # ou df["decennie"] = (((df["annee"] // 10) * 10).astype("Int64").astype(str) + "s")
    df["nombre_genres"] = df["genres"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["nombre_keywords"] = df["keywords"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["category"] = pd.cut(df["runtime"], bins=[0, 90, 120, np.inf], labels=["Court", "Moyen", "Long"], include_lowest=True)
    return df

def fix_missing_val(df):
    cols = ["budget", "revenue"]

    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        for period in df["decennie"].dropna().unique():
            mask = df["decennie"] == period
            median_val = df.loc[mask, col].median()
            missing_mask = mask & df[col].isna()
            missing_count = missing_mask.sum()
            if missing_count > 0 and not pd.isna(median_val):
                random_val = np.random.normal(loc=median_val, scale=median_val * 0.10, size=missing_count)
                random_val = np.maximum(random_val, 0)
                df.loc[missing_mask, col] = random_val
            print(f"{col} | {period} | median = {median_val:.2f} | missing = {missing_count}")
    return df

def Nombre_Null(df):
    res = df.isna().sum()
    print("Valeurs nulles par colonne:")
    print(res)
    return df

def split_data(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Colonnes numeriques: {len(num_cols)}")
    print(f"{num_cols}")
    print(f"Colonnes categorielles: {len(cat_cols)}")
    print(f"{cat_cols}")
    return num_cols, cat_cols

def Nettoyage_data(df):
    Analyse_data(df)
    df = Nombre_duplicate(df)
    df = Nombre_Null(df)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["genres"] = df["genres"].apply(
        lambda x: [g["name"] for g in x] if isinstance(x, list) else []
    )
    df = fix_missing_val(df)
    num_cols, cat_cols = split_data(df)
    return df

def save_data_clean(df, out_dir="../data/processed"):
    name = "movies_clean"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"data clean sauvgarde: {path}")


if __name__ == "__main__":
    load_data(f"{BASE_URL}/discover/movie")