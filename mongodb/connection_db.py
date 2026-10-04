import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

def connect_db():
    myClient = pymongo.MongoClient(os.getenv("MONGODB_URI"))
    try:
        myClient.admin.command("ping")
        print("Connexion MongoDB reussie")
        mydb = myClient["cin_mind"]
        mycol = mydb["cleaned_data"]
        print("Databases:", myClient.list_database_names())
        print("Collections:", mydb.list_collection_names())
        return myClient, mydb, mycol

    except Exception as e:
        print("Erreur de connexion:", e)
        return None, None, None