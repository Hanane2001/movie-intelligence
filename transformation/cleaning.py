import pandas as pd
import numpy as np
import os
from extraction.tmdb_api import Analyse_data

def Nombre_Null(df):
    res = df.isna().sum()
    print("Valeurs nulles par colonne:")
    print(res)
    return df

def Nombre_duplicate(df):
    res = df.duplicated().sum()
    print("Nombre de doublons:", res)
    if res > 0:
        df = df.drop_duplicates()
        print("doublons suprimes")
    return df

def convert_dates(df):
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df

def clean_genres(df):
    df["genres"] = df["genres"].apply(
        lambda x: [g["name"] for g in x] if isinstance(x, list) else []
    )
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

def Nettoyage_data(df):
    print("\n========== ANALYSE DES DONNEES ==========")
    Analyse_data(df)
    print("\n========== DOUBLONS ==========")
    df = Nombre_duplicate(df)
    print("\n========== VALEURS NULLLES ==========")
    df = Nombre_Null(df)
    print("\n========== CONVERSION DES DATES ==========")
    df = convert_dates(df)
    print("\n========== NETTOYAGE DES GENRES ==========")
    df = clean_genres(df)
    return df

def save_data_clean(df, out_dir="../data/processed"):
    name = "movies_clean"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"data clean sauvgarde: {path}")