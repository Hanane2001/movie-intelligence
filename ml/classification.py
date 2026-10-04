import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def load_data():
    path = "../data/processed/movies_clean.csv"
    print("Chargement des donnees...")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df

def create_target(df):
    print("Creation de la cible: ")
    threshold = df["vote_count"].median()
    print("Seuil choisi:", threshold)
    df["high_engagement"] = (df["vote_count"] >= threshold).astype(int)
    print("Repartition des classes:")
    print(df["high_engagement"].value_counts())
    print("Pourcentage des classes:")
    print(df["high_engagement"].value_counts(normalize=True) * 100)
    return df

def prepare_data(df):
    print("Preparation X et y: ")
    features = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "annee",
        "mois",
        "nombre_genres",
        "nombre_keywords",
        "original_language"
    ]
    X = df[features].copy()
    y = df["high_engagement"]
    print("Features utilisees:")
    print(X.columns.tolist())
    print("Target:")
    print("high_engagement")
    if "vote_count" in X.columns:
        print("ATTENTION: vote_count est present dans X!")
    else:
        print("OK: vote_count n'est pas utilise comme feature")
    return X, y

def split_data(X, y):
    print("Train / Test Split: ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    num_cols = [
        "runtime",
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "annee",
        "mois",
        "nombre_genres",
        "nombre_keywords"
    ]
    cat_cols = [
        "original_language"
    ]
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    return preprocessor

def create_models(preprocessor):
    logistic_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    random_forest_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    svm_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearSVC(random_state=42, max_iter=5000))
    ])
    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "Linear SVM": svm_model
    }
    return models

def train_models(models, X_train, y_train):
    print("Entrainement des modeles: ")
    for name, model in models.items():
        print(f"Entrainement: {name}")
        model.fit(X_train, y_train)
        print("Termine")
    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }
    return results, cm

def evaluate_all_models(models, X_test, y_test):
    print("EVALUATION DES MODELES: ")
    all_results = {}
    confusion_matrices = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        results, cm = evaluate_model(model, X_test, y_test)
        all_results[name] = results
        confusion_matrices[name] = cm
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"Precision: {results['Precision']:.4f}")
        print(f"Recall: {results['Recall']:.4f}")
        print(f"F1-score: {results['F1-score']:.4f}")
        print(f"ROC-AUC: {results['ROC-AUC']:.4f}")
        print("\nMatrice de confusion:")
        print(cm)
    return all_results, confusion_matrices

def create_results_dataframe(all_results):
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    print("COMPARAISON DES MODELES: ")
    print(results_df)
    return results_df

def plot_confusion_matrices(confusion_matrices, output_dir="../data/processed/figures"):

    os.makedirs(output_dir, exist_ok=True)
    for name, cm in confusion_matrices.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Faible", "Élevé" ], yticklabels=["Faible", "Élevé" ])
        plt.xlabel("Prédiction")
        plt.ylabel("Valeur réelle")
        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_").replace("-", "")
        path = os.path.join(output_dir, f"confusion_matrix_{filename}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Matrice sauvegardee: {path}")

def main():
    print("CLASSIFICATION - MOVIE INTELLIGENCE: ")
    df = load_data()
    df = create_target(df)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = create_preprocessor()
    models = create_models(preprocessor)
    models = train_models(models, X_train, y_train)
    all_results, confusion_matrices = evaluate_all_models(models, X_test, y_test)
    results_df = create_results_dataframe(all_results)
    plot_confusion_matrices(confusion_matrices)
    print("CLASSIFICATION TERMINEE")
    return models, results_df

if __name__ == "__main__":
    models, results_df = main()