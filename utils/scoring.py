import uuid
import shap
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error
from .mlflow_utils import get_or_create_experiment

from utils.shared import load_model


def get_trained_model_by_name(model_name, data_dim: int):
    return load_model(f"models/{model_name}_{data_dim}.joblib")


def score_model(experiment_name, run_name, model, X_test, y_test: pd.Series):
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name + "_" + str(uuid.uuid4())[:5],
    ):
        y_pred = model.predict(X_test)

        df_pred = pd.DataFrame({"prediction": y_pred}, index=X_test.index)

        # Calculate and log SHAP feature importance
        try:

            model_name = model.__class__.__name__.lower()

            print(model_name)

            if (
                model_name.startswith("decisiontree")
                or model_name.startswith("randomforest")
                or (model_name == "xgbregressor")
                or (model_name == "lgbmregressor")
            ):
                explainer = shap.TreeExplainer(model, X_test.sample(100))
                shap_values = explainer(X_test, check_additivity=False)
            elif model_name.startswith("mlpregressor"):
                explainer = shap.Explainer(model.predict, X_test)
                shap_values = explainer(X_test)
            else:
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)

            fig, ax = plt.subplots()
            fig.set_size_inches(8, max(6, 0.3 * len(X_test.columns)))
            ax = shap.plots.bar(shap_values, ax=ax, show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            mlflow.log_figure(fig, "shap.png")
            plt.savefig(f"artifacts/scoring/shap/{run_name}.png")
            plt.close(fig)
        except Exception as e:
            print(f"SHAP computation skipped due to: {e}")

        rmse = root_mean_squared_error(
            y_test.to_numpy(), df_pred["prediction"].to_numpy()
        )
        mlflow.log_metric("rmse", rmse)

    return df_pred
