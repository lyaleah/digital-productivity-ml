import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

from config import CV_FOLDS, CV_RESULTS_PATH, MODEL_OUTPUT_PATH
from data import load_data, split_data
from features import build_preprocessor
from model import get_models


def main():

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    models = get_models()

    results = []

    best_score = np.inf
    best_model_pipeline = None

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error"
        )

        rmse_scores = -scores
        mean_rmse = rmse_scores.mean()

        results.append({
            "model": name,
            "cv_rmse_mean": mean_rmse
        })

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_model_pipeline = pipeline

    # save CV results
    results_df = pd.DataFrame(results)
    results_df.to_csv(CV_RESULTS_PATH, index=False)

    # use all train sets to fit best model
    best_model_pipeline.fit(X_train, y_train)

    # save model
    joblib.dump(best_model_pipeline, MODEL_OUTPUT_PATH)

    print("Training complete.")
    print(results_df)


if __name__ == "__main__":
    main()
