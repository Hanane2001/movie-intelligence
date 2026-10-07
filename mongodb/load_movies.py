import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from extraction.tmdb_api import (Analyse_data, Nettoyage_data, load_data_v2)
from mongodb.connection_db import connect_db

source = "data/raw/movies_raw.json"
# data = pd.read_json(source)
# df = pd.DataFrame(data)

df = load_data_v2(source)

df = Nettoyage_data(df)
Analyse_data(df)
doct = df.to_dict(orient="records")
print(doct[:2])

client, db, collection = connect_db()
collection.delete_many({})

if doct:
    res = collection.insert_many(doct)
    print(f"{len(res.inserted_ids)} films inseres dans mongodb")
else:
    print("Aucun film a inserer")
print("Nombre de documents :", collection.count_documents({}))