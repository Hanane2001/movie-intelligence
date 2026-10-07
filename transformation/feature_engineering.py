import os
import pandas as pd
import numpy as np
from extraction.tmdb_api import load_data_v3, Analyse_data
from transformation.cleaning import fix_missing_val, _parse_list_column

def create_date_features(df):
    df["annee"] = df["release_date"].dt.year
    df["mois"] = df["release_date"].dt.month
    df["decennie"] = pd.cut(df["release_date"].dt.year, bins=[0, 1979, 2000, 2010, 2020, 2030], labels=["<1980", "1980-2000", "2001-2010", "2011-2020", "2021-2030"], include_lowest=True)
    # ou df["decennie"] = (((df["annee"] // 10) * 10).astype("Int64").astype(str) + "s")
    return df

def split_data(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Colonnes numeriques: {len(num_cols)}")
    print(f"{num_cols}")
    print(f"Colonnes categorielles: {len(cat_cols)}")
    print(f"{cat_cols}")
    return num_cols, cat_cols

def create_genre_features(df):
    df["nombre_genres"] = df["genres"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df

def create_keyword_features(df):
    df["nombre_keywords"] = df["keywords"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df

def create_runtime_category(df):
    df["category"] = pd.cut(df["runtime"], bins=[0, 90, 120, np.inf], labels=["Court", "Moyen", "Long"], include_lowest=True)
    return df

def save_data_feature(df, out_dir="../data/features"):
    name = "movies_feature"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"data feature sauvgarde: {path}")

def feature(df):
    print("FEATURES TEMPORELLES:")
    df = create_date_features(df)
    print("FEATURES GENRES")
    df = create_genre_features(df)
    print("FEATURES KEYWORDS")
    df = create_keyword_features(df)
    print("CATEGORIE DE DUREE")
    df = create_runtime_category(df)
    return df

def main_FE():
    source = "../data/processed/movies_clean.csv"
    df = load_data_v3(source)
    df["genres"] = df["genres"].apply(_parse_list_column)
    df["keywords"] = df["keywords"].apply(_parse_list_column)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    dt = feature(df)
    dt = fix_missing_val(dt)
    Analyse_data(dt)
    save_data_feature(dt)
    return dt