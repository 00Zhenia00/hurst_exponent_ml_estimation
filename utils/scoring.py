import uuid
import shap
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error
from .mlflow_utils import get_or_create_experiment

from utils.shared import load_model
from utils.rs_estimator import RSEstimator
from utils.dfa_estimator import DFAEstimator


def get_trained_model_by_name(model_name, data_dim: int):
    if model_name == "RS":
        return RSEstimator()
    elif model_name == "DFA":
        return DFAEstimator()
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

        # -------- SHAP FEATURE IMPORTANCE (PAPER-READY) --------
        try:
            model_name = model.__class__.__name__.lower()

            if (
                model_name.startswith("decisiontree")
                or model_name.startswith("randomforest")
                or model_name == "xgbregressor"
                or model_name == "lgbmregressor"
            ):
                explainer = shap.TreeExplainer(model, X_test.sample(100))
                shap_values = explainer(X_test, check_additivity=False)

            elif (
                model_name.startswith("mlpregressor")
                or model_name.startswith("cnnregressor")
                or model_name.startswith("rnnregressor")
            ):
                explainer = shap.Explainer(model.predict, X_test)
                shap_values = explainer(X_test)

            else:
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)

            # Publication-friendly matplotlib style
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            })

            fig, ax = plt.subplots(
                figsize=(6.5, max(4.0, 0.28 * len(X_test.columns)))
            )

            shap.plots.bar(
                shap_values,
                ax=ax,
                show=False,
                # color="black",      # grayscale-safe
            )

            # 3. Iterate through the bars (patches) and change their color
            #    Note: We iterate in reverse or check the artist type to be safe
            for patch in ax.patches:
                patch.set_facecolor('#000000')  # Set black color

            for txt in ax.texts:
                txt.set_color("black")

            # Axis labels (titles go in captions)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_ylabel("Feature")

            ax.tick_params(direction="in", which="both")
            fig.tight_layout()

            # Log & save (vector preferred)
            mlflow.log_figure(fig, "shap_feature_importance.png")
            fig.savefig(
                f"artifacts/scoring/shap/{run_name}.png",
                bbox_inches="tight"
            )
            plt.close(fig)

        except Exception as e:
            print(f"SHAP computation skipped due to: {e}")

        # -------- METRICS --------
        rmse = root_mean_squared_error(
            y_test.to_numpy(),
            df_pred["prediction"].to_numpy()
        )
        mlflow.log_metric("rmse", rmse)

        residuals = y_test.to_numpy() - df_pred["prediction"].to_numpy()
        error_std = np.std(residuals, ddof=0)
        mlflow.log_metric("error_std", float(error_std))

    return df_pred


# def score_model(experiment_name, run_name, model, X_test, y_test: pd.Series):
#     experiment_id = get_or_create_experiment(experiment_name)
#     mlflow.set_experiment(experiment_id=experiment_id)

#     with mlflow.start_run(
#         experiment_id=experiment_id,
#         run_name=run_name + "_" + str(uuid.uuid4())[:5],
#     ):
#         y_pred = model.predict(X_test)

#         df_pred = pd.DataFrame({"prediction": y_pred}, index=X_test.index)

#         # Calculate and log SHAP feature importance
#         try:

#             model_name = model.__class__.__name__.lower()

#             print(model_name)

#             if (
#                 model_name.startswith("decisiontree")
#                 or model_name.startswith("randomforest")
#                 or (model_name == "xgbregressor")
#                 or (model_name == "lgbmregressor")
#             ):
#                 explainer = shap.TreeExplainer(model, X_test.sample(100))
#                 shap_values = explainer(X_test, check_additivity=False)
#             elif (
#                 model_name.startswith("mlpregressor")
#                 or model_name.startswith("cnnregressor")
#                 or model_name.startswith("rnnregressor")
#             ):
#                 explainer = shap.Explainer(model.predict, X_test)
#                 shap_values = explainer(X_test)
#             else:
#                 explainer = shap.Explainer(model, X_test)
#                 shap_values = explainer(X_test)

#             fig, ax = plt.subplots()
#             fig.set_size_inches(8, max(6, 0.3 * len(X_test.columns)))
#             ax = shap.plots.bar(shap_values, ax=ax, show=False)
#             ax.set_title("SHAP Feature Importance")
#             ax.set_xlabel("Absolute SHAP value")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "shap.png")
#             plt.savefig(f"artifacts/scoring/shap/{run_name}.png")
#             plt.close(fig)
#         except Exception as e:
#             print(f"SHAP computation skipped due to: {e}")

#         rmse = root_mean_squared_error(
#             y_test.to_numpy(), df_pred["prediction"].to_numpy()
#         )
#         mlflow.log_metric("rmse", rmse)

#         # Calculate residuals and log their standard deviation (error std)
#         residuals = y_test.to_numpy() - df_pred["prediction"].to_numpy()
#         error_std = np.std(residuals, ddof=0)
#         mlflow.log_metric("error_std", float(error_std))

#     return df_pred
