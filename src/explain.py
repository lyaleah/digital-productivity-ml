import os
import joblib
import shap
import matplotlib.pyplot as plt
from data import load_data, split_data
from config import MODEL_OUTPUT_PATH


def main():

    os.makedirs("outputs/figures", exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = joblib.load(MODEL_OUTPUT_PATH)

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_processed = preprocessor.transform(X_train)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_processed)

    shap.summary_plot(
        shap_values,
        X_processed,
        show=False
    )

    plt.savefig("outputs/figures/shap_summary.png")
    plt.close()

    print("SHAP explanation saved.")


if __name__ == "__main__":
    main()
