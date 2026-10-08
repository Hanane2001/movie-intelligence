import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ajoute le projet au path Python
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def run_extraction():
    from extraction.tmdb_api import main_E
    main_E()
    print("Extraction terminée")

def run_cleaning():
    from transformation.cleaning import main_T
    main_T()
    print("Nettoyage terminé")

def run_features():
    from transformation.feature_engineering import main_FE
    main_FE()
    print("Feature engineering terminé")

def run_mongodb():
    import pandas as pd
    from extraction.tmdb_api import load_data_v3
    from mongodb.connection_db import connect_db

    df = load_data_v3("data/features/movies_feature.csv")

    client, db, collection = connect_db()
    if collection is None:
        raise RuntimeError("MongoDB non disponible")

    collection.delete_many({})
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    if records:
        collection.insert_many(records)
        print(f"{len(records)} films insérés dans MongoDB")

def run_nlp():
    from nlp.tfidf import main_TF_IDF
    main_TF_IDF()
    print("TF-IDF terminé")

def run_classification():
    from ml.classification import main_CL
    main_CL()
    print("Classification terminée")

def run_clustering():
    from ml.clustering import main_CU
    main_CU()
    print("Clustering terminé")

default_args = {
    "owner": "movie_intelligence",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="movie_intelligence_pipeline",
    description="Pipeline complet : Extraction → ML",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["movie", "tmdb", "ml"],
) as dag:

    extraction = PythonOperator(task_id="extraction", python_callable=run_extraction)
    cleaning = PythonOperator(task_id="cleaning", python_callable=run_cleaning)
    features = PythonOperator(task_id="features", python_callable=run_features)
    mongodb = PythonOperator(task_id="mongodb", python_callable=run_mongodb)
    nlp = PythonOperator(task_id="nlp", python_callable=run_nlp)
    classification = PythonOperator(task_id="classification", python_callable=run_classification)
    clustering = PythonOperator(task_id="clustering", python_callable=run_clustering)

    extraction >> cleaning >> features >> mongodb >> nlp >> classification >> clustering