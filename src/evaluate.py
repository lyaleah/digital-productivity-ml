import os
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

from config import MODEL_OUTPUT_PATH, TEST_METRICS_PATH
from data import load_data, split_data


def main():

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = joblib.load(MODEL_OUTPUT_PATH)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # save metrics
    metrics = {
        "rmse": rmse,
        "r2": r2
    }

    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    # Residual plot
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.savefig("outputs/figures/residual_plot.png")
    plt.close()

    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_rmse, label="Train RMSE")
    plt.plot(train_sizes, val_rmse, label="Validation RMSE")
    plt.xlabel("Training Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Learning Curve")
    plt.savefig("outputs/figures/learning_curve.png")
    plt.close()

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
