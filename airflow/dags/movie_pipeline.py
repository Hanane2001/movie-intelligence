import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT_ROOT))


def task_extraction(**context):
    from extraction.tmdb_api import load_data, BASE_URL, load_data_v2, Analyse_data

    source = f"{BASE_URL}/discover/movie"
    load_data(source, total_movies=3000)

    destination = str(PROJECT_ROOT / "data" / "raw" / "movies_raw.json")
    df = load_data_v2(destination)
    Analyse_data(df)
    return df.shape[0]


def task_cleaning(**context):
    from extraction.tmdb_api import load_data_v2
    from transformation.cleaning import Nettoyage_data, save_data_clean

    source = str(PROJECT_ROOT / "data" / "raw" / "movies_raw.json")
    df = load_data_v2(source)
    df_clean = Nettoyage_data(df)
    save_data_clean(df_clean, out_dir=str(PROJECT_ROOT / "data" / "processed"))
    return df_clean.shape[0]


def task_features(**context):
    import pandas as pd
    from extraction.tmdb_api import load_data_v3
    from transformation.cleaning import _parse_list_column, fix_missing_val
    from transformation.feature_engineering import feature, save_data_feature

    source = str(PROJECT_ROOT / "data" / "processed" / "movies_clean.csv")
    df = load_data_v3(source)

    df["genres"] = df["genres"].apply(_parse_list_column)
    df["keywords"] = df["keywords"].apply(_parse_list_column)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    df = feature(df)
    df = fix_missing_val(df)
    save_data_feature(df, out_dir=str(PROJECT_ROOT / "data" / "features"))
    return df.shape[0]


def task_load_mongodb(**context):
    import ast
    import pandas as pd
    from extraction.tmdb_api import load_data_v3
    from mongodb.connection_db import connect_db

    source = str(PROJECT_ROOT / "data" / "features" / "movies_feature.csv")
    df = load_data_v3(source)

    def _reparse(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return []
        return []

    df["genres"] = df["genres"].apply(_reparse)
    df["keywords"] = df["keywords"].apply(_reparse)

    client, db, collection = connect_db()
    if collection is None:
        raise RuntimeError("MongoDB non disponible")

    collection.delete_many({})
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    if records:
        res = collection.insert_many(records)
        print(f"{len(res.inserted_ids)} documents inseres dans MongoDB")
    print("Total en base :", collection.count_documents({}))


def task_nlp(**context):
    from extraction.tmdb_api import load_data_v3
    from nlp.tfidf import prepare_text, create_tfidf, analyze_tfidf, top_terms

    source = str(PROJECT_ROOT / "data" / "features" / "movies_feature.csv")
    df = load_data_v3(source)
    df = prepare_text(df)
    X_tfidf, vectorizer = create_tfidf(
        df, max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    analyze_tfidf(X_tfidf, vectorizer)
    top_terms(X_tfidf, vectorizer, n=20)


def task_classification(**context):
    from extraction.tmdb_api import load_data_v3
    from ml.classification import (
        create_target, prepare_data, split_train_test,
        create_preprocessor, create_models, train_models,
        evaluate_all_models, create_results_dataframe,
        plot_confusion_matrices,
    )

    source = str(PROJECT_ROOT / "data" / "features" / "movies_feature.csv")
    df = load_data_v3(source)
    df = create_target(df)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    preprocessor = create_preprocessor(X)
    models = create_models(preprocessor)
    models = train_models(models, X_train, y_train)
    all_results, cms = evaluate_all_models(models, X_test, y_test)
    create_results_dataframe(all_results)
    plot_confusion_matrices(cms, output_dir=str(PROJECT_ROOT / "data" / "figures"))


def task_clustering(**context):
    from extraction.tmdb_api import load_data_v3
    from ml.clustering import (
        prepare_clustering_data, standardize_data, test_k_values,
        select_best_k, apply_kmeans, interpret_clusters,
        plot_silhouette, plot_clusters,
    )

    source = str(PROJECT_ROOT / "data" / "features" / "movies_feature.csv")
    df = load_data_v3(source)
    X = prepare_clustering_data(df)
    X_scaled, scaler = standardize_data(X)
    results_df = test_k_values(X_scaled, k_values=range(2, 8))
    plot_silhouette(results_df, output_dir=str(PROJECT_ROOT / "data" / "figures"))
    best_k = select_best_k(results_df)
    df_clustered, _ = apply_kmeans(X_scaled, df, best_k)
    interpret_clusters(df_clustered)
    plot_clusters(df_clustered, output_dir=str(PROJECT_ROOT / "data" / "figures"))


default_args = {
    "owner": "movie_intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="movie_intelligence_pipeline",
    description="Extraction -> Nettoyage -> Features -> MongoDB -> NLP -> ML",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["movie", "tmdb", "ml"],
) as dag:

    extraction = PythonOperator(
        task_id="extraction",
        python_callable=task_extraction,
    )

    cleaning = PythonOperator(
        task_id="cleaning",
        python_callable=task_cleaning,
    )

    features = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_features,
    )

    mongodb = PythonOperator(
        task_id="load_mongodb",
        python_callable=task_load_mongodb,
    )

    nlp = PythonOperator(
        task_id="nlp_tfidf",
        python_callable=task_nlp,
    )

    classification = PythonOperator(
        task_id="classification",
        python_callable=task_classification,
    )

    clustering = PythonOperator(
        task_id="clustering",
        python_callable=task_clustering,
    )

    extraction >> cleaning >> features >> mongodb >> nlp >> classification >> clustering