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

def load_data_v2(path):
    print("Chargement des donnees...")
    data = pd.read_json(path)
    df = pd.DataFrame(data)
    # df = pd.read_csv(path)
    # print("Shape:", df.shape)
    return df

def load_data_v3(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df

def main_E():
    source = f"{BASE_URL}/discover/movie"
    load_data(source)
    destination = "../data/raw/movies_raw.json"
    df = load_data_v2(destination)
    Analyse_data(df)