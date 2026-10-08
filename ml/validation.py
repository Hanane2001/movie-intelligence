import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from extraction.tmdb_api import load_data_v3
from ml.classification import (
    create_target,
    prepare_data,
    split_train_test,
    create_preprocessor,
    create_models,
    evaluate_model
)

def cross_validate_models(models, X, y):
    print("CROSS VALIDATION: ")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    resl = []
    for name, model in models.items():
        print(f"Validation du modele: {name}")
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        res = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-score": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean()
        }
        resl.append(res)
    res_df = pd.DataFrame(resl)
    print("RESULTATS CROSS VALIDATION: ")
    print(res_df.round(4))
    return res_df

def grid_search_random_forest(random_forest_model, X, y):
    print("GRID SEARCH - RANDOM FOREST")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "model__n_estimators": [100, 200, 300, 500, 1000],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    print("recherche la meilleurs params...")
    grid_search.fit(X, y)
    print("Meilleur params: ")
    print(grid_search.best_params_)
    print("Meilleur F1-score moyen: ")
    print(f"{grid_search.best_score_:.4f}")
    return grid_search

def main_V():
    print("VALIDATION ET OPTIMISATION")
    path = "../data/features/movies_feature.csv"
    df = load_data_v3(path)
    df = create_target(df)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    preprocessor = create_preprocessor(X)
    models = create_models(preprocessor)
    cv_results = cross_validate_models(models, X_train, y_train)
    random_forest_model = models["Random Forest"]
    grid_search = grid_search_random_forest(random_forest_model, X_train, y_train)
    best_model = grid_search.best_estimator_
    optimized_results, cm = evaluate_model(best_model, X_test, y_test)
    print("\nRESULTATS DU RANDOM FOREST OPTIMISE:")
    for metric, value in optimized_results.items():
        print(f"{metric}: {value:.4f}")
    print("\nMatrice de confusion:")
    print(cm)
    print("VALIDATION ET OPTIMISATION TERMINEES")
    return cv_results, grid_search, optimized_results


