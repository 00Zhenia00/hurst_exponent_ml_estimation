import json
import uuid
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .mlflow_utils import get_or_create_experiment


def get_tuned_hyperparameters(model_name: str, data_dim: int) -> dict:
    """
    Loads and returns the tuned hyperparameters for a given model and data dimension.
    Args:
        model_name (str): The name of the model.
        data_dim (int): The dimension of the data.
    Returns:
        dict: A dictionary containing the tuned hyperparameters.
    """
    try:
        # TODO: Path magic string should be replaced with a config variable
        with open(f"hyperparams/{model_name}_{data_dim}.json", "r") as f:
            hyperparameters = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"get_tuned_hyperparameters(): Tuned hyperparameters file for model {model_name} \
            with data dimension {data_dim} not found!"
        )

    # if model_name == "RNN" or model_name == "CNN":
    #     result_hyperparameters = {}
    #     # Add `model__` prefix to each hyperparameter for KerasRegressor compatibility
    #     for hyperparameter_name, value in hyperparameters.items():
    #         if hyperparameter_name not in ["batch_size", "epochs"]:
    #             result_hyperparameters[f"model__{hyperparameter_name}"] = value
    #         else:
    #             result_hyperparameters[hyperparameter_name] = value
    #     hyperparameters = result_hyperparameters

    return hyperparameters


def train_model(experiment_name, run_name, estimator, hyperparams, X_train, y_train):
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name + "_" + str(uuid.uuid4())[:5],
    ):
        estimator.set_params(**hyperparams)
        estimator.fit(X_train, y_train)

        # ------------------ PLOT STYLE ------------------
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        # ------------------ LINEAR MODELS ------------------
        if hasattr(estimator, "coef_"):
            coefficients = estimator.coef_
            coef_df = pd.DataFrame({
                "feature": X_train.columns.tolist(),
                "coefficient": coefficients.tolist(),
            }).sort_values('coefficient', ascending=False)

            fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.28*len(coef_df))))
            ax.barh(coef_df['feature'], coef_df['coefficient'], color="black")
            ax.set_xlabel("Coefficient")
            ax.set_ylabel("Feature")
            ax.tick_params(direction="in", which="both")
            fig.tight_layout()
            mlflow.log_figure(fig, "coefficients.png")
            fig.savefig(f"artifacts/training/feature_importance/{run_name}_coefficients.png", bbox_inches="tight")
            plt.close(fig)

        # ------------------ XGBOOST ------------------
        elif hasattr(estimator, "get_booster"):
            booster = estimator.get_booster()
            for importance_type in ["weight", "gain"]:
                importance_dict = booster.get_score(importance_type=importance_type)
                importance_df = pd.DataFrame({
                    "feature": list(importance_dict.keys()),
                    "importance": list(importance_dict.values())
                }).sort_values('importance', ascending=False)

                fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.28*len(importance_df))))
                ax.barh(importance_df['feature'], importance_df['importance'], color="black")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                ax.tick_params(direction="in", which="both")
                fig.tight_layout()
                filename = f"{importance_type}.png"
                mlflow.log_figure(fig, filename)
                fig.savefig(f"artifacts/training/feature_importance/{run_name}_{filename}", bbox_inches="tight")
                plt.close(fig)

        # ------------------ LIGHTGBM ------------------
        elif hasattr(estimator, "booster_"):
            booster = estimator.booster_
            for importance_type in ["split", "gain"]:
                importance_array = booster.feature_importance(importance_type=importance_type)
                importance_df = pd.DataFrame({
                    "feature": booster.feature_name(),
                    "importance": importance_array.tolist()
                }).sort_values('importance', ascending=False)

                fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.28*len(importance_df))))
                ax.barh(importance_df['feature'], importance_df['importance'], color="black")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                ax.tick_params(direction="in", which="both")
                fig.tight_layout()
                filename = f"{importance_type}.png"
                mlflow.log_figure(fig, filename)
                fig.savefig(f"artifacts/training/feature_importance/{run_name}_{filename}", bbox_inches="tight")
                plt.close(fig)

        # ------------------ DECISION TREE / RANDOM FOREST ------------------
        elif hasattr(estimator, "feature_importances_"):
            feature_importances = estimator.feature_importances_
            importance_df = pd.DataFrame({
                "feature": X_train.columns.tolist(),
                "importance": feature_importances.tolist(),
            }).sort_values('importance', ascending=False)

            fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.28*len(importance_df))))
            ax.barh(importance_df['feature'], importance_df['importance'], color="black")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.tick_params(direction="in", which="both")
            fig.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            fig.savefig(f"artifacts/training/feature_importance/{run_name}.png", bbox_inches="tight")
            plt.close(fig)

        # ------------------ LOG MODEL AND PARAMETERS ------------------
        mlflow.sklearn.log_model(estimator, name="model")
        mlflow.log_params(hyperparams)

    return estimator



# def train_model(experiment_name, run_name, estimator, hyperparams, X_train, y_train):
#     experiment_id = get_or_create_experiment(experiment_name)
#     mlflow.set_experiment(experiment_id=experiment_id)

#     with mlflow.start_run(
#         experiment_id=experiment_id,
#         run_name=run_name + "_" + str(uuid.uuid4())[:5],
#     ):
#         estimator.set_params(**hyperparams)
#         estimator.fit(X_train, y_train)

#         # Linear models
#         if hasattr(estimator, "coef_"):
#             coefficients = estimator.coef_
#             # Log coefficients as dataframe
#             coef_df = pd.DataFrame({
#                 "feature": X_train.columns.tolist(),
#                 "coefficient": coefficients.tolist(),
#             }).sort_values('coefficient', ascending=False)
#             # mlflow.log_table(data=coef_df, artifact_file="coefficients_table.json")

#             fig, ax = plt.subplots(constrained_layout=True)
#             fig.set_size_inches(8, max(6, 0.3 * len(coef_df)))
#             ax.barh(coef_df['feature'], coef_df['coefficient'])
#             ax.set_title("Model Coefficients")
#             ax.set_xlabel("Coefficient")
#             ax.set_ylabel("Feature")
#             # plt.tight_layout()
#             mlflow.log_figure(fig, "coefficients.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}.png")
#             plt.close(fig)
#         # xgboost
#         elif hasattr(estimator, "get_booster"):
#             booster = estimator.get_booster()

#             weight_importance_dict = booster.get_score(importance_type='weight')
#             gain_importance_dict = booster.get_score(importance_type='gain')

#             weight_importance_df = pd.DataFrame({
#                 "feature": list(weight_importance_dict.keys()),
#                 "importance": list(weight_importance_dict.values()),
#             }).sort_values('importance', ascending=False)

#             gain_importance_df = pd.DataFrame({
#                 "feature": list(gain_importance_dict.keys()),
#                 "importance": list(gain_importance_dict.values()),
#             }).sort_values('importance', ascending=False)

#             fig, ax = plt.subplots()
#             fig.set_size_inches(8, max(6, 0.3 * len(weight_importance_df)))
#             ax.barh(weight_importance_df['feature'], weight_importance_df['importance'])
#             ax.set_title("Feature Importances (weight)")
#             ax.set_xlabel("Importance")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "weight_feature_importance.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}_weight.png")
#             plt.close(fig)

#             fig, ax = plt.subplots()
#             fig.set_size_inches(8, max(6, 0.3 * len(gain_importance_df)))
#             ax.barh(gain_importance_df['feature'], gain_importance_df['importance'])
#             ax.set_title("Feature Importances (gain)")
#             ax.set_xlabel("Importance")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "gain_feature_importance.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}_gain.png")
#             plt.close(fig)
#         # lightgbm
#         elif hasattr(estimator, "booster_"):
#             booster = estimator.booster_

#             split_importance_dict = booster.feature_importance(importance_type='split')
#             gain_importance_dict = booster.feature_importance(importance_type='gain')

#             feature_names = booster.feature_name()

#             split_importance_df = pd.DataFrame({
#                 "feature": feature_names,
#                 "importance": split_importance_dict.tolist(),
#             }).sort_values('importance', ascending=False)

#             gain_importance_df = pd.DataFrame({
#                 "feature": feature_names,
#                 "importance": gain_importance_dict.tolist(),
#             }).sort_values('importance', ascending=False)

#             fig, ax = plt.subplots()
#             fig.set_size_inches(8, max(6, 0.3 * len(split_importance_df)))
#             ax.barh(split_importance_df['feature'], split_importance_df['importance'])
#             ax.set_title("Feature Importances (split)")
#             ax.set_xlabel("Importance")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "split_feature_importance.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}_split.png")
#             plt.close(fig)

#             fig, ax = plt.subplots()
#             fig.set_size_inches(8, max(6, 0.3 * len(gain_importance_df)))
#             ax.barh(gain_importance_df['feature'], gain_importance_df['importance'])
#             ax.set_title("Feature Importances (gain)")
#             ax.set_xlabel("Importance")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "gain_feature_importance.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}_gain.png")
#             plt.close(fig)
#         # decision tree and random forest
#         elif hasattr(estimator, "feature_importances_"):
#             feature_importances = estimator.feature_importances_
#             # Log feature importances as dataframe
#             importance_df = pd.DataFrame({
#                 "feature": X_train.columns.tolist(),
#                 "importance": feature_importances.tolist(),
#             }).sort_values('importance', ascending=False) 
#             # mlflow.log_table(data=importance_df, artifact_file="feature_importance_table.json")

#             fig, ax = plt.subplots()
#             # Set figure size based on number of features
#             fig.set_size_inches(8, max(6, 0.3 * len(importance_df)))
#             ax.barh(importance_df['feature'], importance_df['importance'])
#             ax.set_title("Feature Importances")
#             ax.set_xlabel("Importance")
#             ax.set_ylabel("Feature")
#             plt.tight_layout()
#             mlflow.log_figure(fig, "feature_importance.png")
#             fig.savefig(f"artifacts/training/feature_importance/{run_name}.png")
#             plt.close(fig)
    
#         mlflow.sklearn.log_model(estimator, name="model")
#         mlflow.log_params(hyperparams)

#     return estimator
