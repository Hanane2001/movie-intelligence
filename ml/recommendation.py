import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from extraction.tmdb_api import load_data_v3
from nlp.tfidf import prepare_text, create_tfidf


def calculate_similarity(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print("SIMILARITE COSINUS: ")
    print("Shape :", similarity_matrix.shape)
    return similarity_matrix


def recommend_movies(title, df, similarity_matrix, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        print(f"\nFilm introuvable: {title}")
        return pd.DataFrame()

    movie_index = matches.index[0]
    similarity_scores = similarity_matrix[movie_index]
    similar_indices = np.argsort(similarity_scores)[::-1]
    recommendations = []
    for index in similar_indices:
        if index == movie_index:
            continue
        recommendations.append({
            "title": df.loc[index, "title"],
            "genres": df.loc[index, "genres"],
            "annee": df.loc[index, "annee"],
            "similarity_score": similarity_scores[index]
        })
        if len(recommendations) >= top_n:
            break
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df


def display_recommendations(title, recommendations_df):
    print(f"\nRECOMMANDATIONS POUR: {title}")
    if recommendations_df.empty:
        print("Aucune recommandation trouvée")
        return

    for i, row in recommendations_df.iterrows():
        print(f"\n{i + 1}. {row['title']}")
        print(f"Genre: {row['genres']}")
        print(f"Année: {row['annee']}")
        print(f"Similarité: {row['similarity_score']:.4f}")


def main_R():
    print("RECOMMANDATION DE FILMS")
    path = "../data/features/movies_feature.csv"
    df = load_data_v3(path)
    df = df.reset_index(drop=True)
    df = prepare_text(df)
    tfidf_matrix, vectorizer = create_tfidf(df, max_features=5000, ngram_range=(1, 2), stop_words="english")
    similarity_matrix = calculate_similarity(tfidf_matrix)
    movie_title = input("\nEntrez le titre d'un film: ")
    recommendations = recommend_movies(movie_title, df, similarity_matrix, top_n=5)
    display_recommendations(movie_title, recommendations)
    return (df, tfidf_matrix, similarity_matrix, recommendations)
