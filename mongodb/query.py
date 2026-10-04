import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pymongo
import os
from mongodb.connection_db import connect_db

client, db, collection = connect_db()

# recuperer les 5 films
movies = collection.find().limit(5)
for movie in movies:
    print(movie["movie_id"])

# recuperer les films avec une budget superieure a 45000000
movies_gt = collection.find({
    "budget": {"$gt": 90000000} # $gt => greater than
})
for movie in movies_gt:
    print(movie["title"])

# 500000 <= movies >= 90000000
nb_movies_gt = collection.count_documents({
    "budget" : {"$gte": 500000, "$lte": 90000000} # $eq => equal also there $lt => less than, $in => any value in array, $ne => all values not equal a, 
})
print(nb_movies_gt)

# Compter les films dans cette plage
count = collection.count_documents({
    "budget": {
        "$gte": 500000,
        "$lte": 90000000
    }
})
print("Nombre de films:", count)

# Films du genre Drama
movies = collection.aggregate([{
    "$match": {"genres": "Drama"}
}])
for m in movies:
    print(m["title"])

# Films avec une note supérieure ou égale à 8
movies = collection.find({
    "vote_average": {"$gte": 8}
})
for movie in movies:
    print(movie["title"], movie["vote_average"])

# Les 10 films avec le plus de votes
movies = collection.find().sort(
    "vote_count",
    pymongo.DESCENDING
).limit(10)

for movie in movies:
    print(movie["title"], movie["vote_count"])

# demander à MongoDB de retourner uniquement certaines informations avec $project (0 => non affiche, 1 => affiche)
movies = collection.aggregate([{
        "$project": {
            "_id": 0,
            "title": 1,
            "vote_average": 1,
            "popularity": 1
        }}
])
for movie in movies:
    print(movie)

# trouver les films Drama et les trier par note
movies = collection.aggregate([{
        "$match": {"genres": "Drama"}},{
        "$sort": {"vote_average": -1}}
])
for movie in movies:
    print(movie["title"], movie["vote_average"])


movies = collection.aggregate([{
        "$match": {
            "genres": "Drama",
            "vote_average": {"$gte": 7}
        }},{
        "$project": {
            "_id": 0,
            "title": 1,
            "vote_average": 1,
            "popularity": 1
        }},{
        "$sort": {
            "vote_average": -1
        }},{
        "$limit": 10
}])
for movie in movies:
    print(movie)

# compter combien de films appartiennent à chaque genre
genres = collection.aggregate([
    {
        "$unwind": "$genres"},{
        "$group": {
            "_id": "$genres",
            "nombre_films": {
                "$sum": 1
            }
        }},{
        "$sort": {
            "nombre_films": -1
        }
    }
])

for genre in genres:
    print(genre["_id"], genre["nombre_films"])


# budget moyen par genre
result = collection.aggregate([
    {
        "$unwind": "$genres"
    },{
        "$group": {
            "_id": "$genres",
            "budget_moyen": {
                "$avg": "$budget"
            },
            "nombre_films": {
                "$sum": 1
            }
        }
    },{
        "$sort": {
            "budget_moyen": -1
        }
    }
])

for genre in result:
    print(genre["_id"], round(genre["budget_moyen"], 2), genre["nombre_films"])