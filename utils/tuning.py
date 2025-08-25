import time
import uuid
import mlflow
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from .mlflow_utils import get_or_create_experiment


def get_objective(
    estimator,
    hyperparams_getter_func,
    scoring_func,
    direction,
    X_train,
    y_train,
    cv_folds=None,
    X_val=None,
    y_val=None,
    cpus_to_use=-1,
):
    """
    Creates an Optuna objective function for hyperparameter optimization with MLflow logging.

    Args:
        estimator: The machine learning estimator (model) to optimize.
        hyperparams_getter_func: Function that takes a trial and returns a dict of hyperparameters.
        scoring_func: Function to evaluate model predictions (e.g., RMSE, accuracy).
        direction: Optimization direction, either 'minimize' or 'maximize'.
        X_train: Training features as a numpy array or pandas DataFrame.
        y_train: Training targets as a numpy array or pandas Series.
        cv_folds: (Optional) Number of cross-validation folds. If None, uses validation set.
        X_val: (Optional) Validation features, required if cv_folds is None.
        y_val: (Optional) Validation targets, required if cv_folds is None.
        cpus_to_use: (Optional) Number of CPUs to use for parallelization in cross_val_score. Default is -1 (all CPUs).

    Returns:
        function: An objective function to be used with Optuna's study.optimize().

    Raises:
        ValueError: If neither cv_folds nor (X_val and y_val) are provided.
    """
    if not cv_folds and (X_val is None or y_val is None):
        raise ValueError(
            "get_objective(): Either cv_folds or X_val and y_val must be provided!"
        )

    def objective(trial):
        with mlflow.start_run(nested=True):
            hyperparams = hyperparams_getter_func(trial)
            estimator.set_params(**hyperparams)

            result = None

            if not cv_folds:
                y_pred = estimator.fit(X_train, y_train).predict(X_val)
                result = scoring_func(y_val, y_pred)
            else:
                scorer = make_scorer(
                    scoring_func, greater_is_better=(direction == "maximize")
                )
                result = cross_val_score(
                    estimator=estimator,
                    X=X_train,
                    y=y_train,
                    cv=cv_folds,
                    scoring=scorer,
                    n_jobs=(
                        None if hasattr(estimator, "n_jobs")
                        else cpus_to_use
                    )
                ).mean()

                if direction != "maximize":
                    result = (
                        -result
                    )  # Invert results of the sklearn.metrics.make_scorer scorer

            mlflow.log_params(hyperparams)
            mlflow.log_metric("rmse", result)

        return result

    return objective


def tune_hyperparameters(
    experiment_name,
    run_name,
    estimator,
    hyperparams_getter_func,
    scoring_func,
    direction,
    X_train,
    y_train,
    cv_folds=None,
    X_val=None,
    y_val=None,
    n_trials=30,
    cpus_to_use=-1,
) -> optuna.study.Study:
    """
    Runs hyperparameter optimization using Optuna and logs results to MLflow.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name for the MLflow run.
        estimator: The machine learning estimator (model) to optimize.
        hyperparams_getter_func: Function that takes a trial and returns a dict of hyperparameters.
        scoring_func: Function to evaluate model predictions (e.g., RMSE, accuracy).
        direction (str): Optimization direction, either 'minimize' or 'maximize'.
        X_train: Training features as a numpy array or pandas DataFrame.
        y_train: Training targets as a numpy array or pandas Series.
        cv_folds (int, optional): Number of cross-validation folds. If None, uses validation set.
        X_val (optional): Validation features, required if cv_folds is None.
        y_val (optional): Validation targets, required if cv_folds is None.
        n_trials (int, optional): Number of Optuna trials. Defaults to 30.
        cpus_to_use (int, optional): Number of CPUs to use for parallelization in cross_val_score. Default is -1 (all CPUs).

    Returns:
        optuna.study.Study: The Optuna study object containing optimization results.
    """
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    objective = get_objective(
        estimator=estimator,
        hyperparams_getter_func=hyperparams_getter_func,
        scoring_func=scoring_func,
        direction=direction,
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
        X_val=X_val,
        y_val=y_val,
        cpus_to_use=cpus_to_use
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name + "_" + str(uuid.uuid4())[:5],
        nested=True,
    ):
        start_time = time.time()
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        elapsed_time = time.time() - start_time

        run_info = {
            "feature_count_train": X_train.shape[1],
            "rows_count_train": X_train.shape[0],
            "feature_count_val": X_val.shape[1] if X_val else None,
            "rows_count_val": X_val.shape[0] if X_val else None,
            "tuning_time": f"{int(elapsed_time//3600)}h {int(elapsed_time//60)}m {int(elapsed_time%60)}s",
        }
        mlflow.log_dict(run_info, "run_info.json")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)
        mlflow.log_figure(
            optuna.visualization.plot_optimization_history(study),
            artifact_file="optimization_history.png",
        )

    return study


def get_ridge_hyperparams(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-5, 1e5, log=True)}


def get_lasso_hyperparams(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-5, 1e5, log=True)}


def get_decision_tree_hyperparams(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def get_random_forest_hyperparams(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def get_lightgbm_hyperparams(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=20),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50, step=5),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
    }


def get_xgboost_hyperparams(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50, step=5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10, log=True),
    }


def get_hyperparams_getter_function_by_model_name(model_name):
    """
    Returns the hyperparameter getter function for a given model name.

    Args:
        model_name (str): The name of the model. Supported values are
            'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'LightGBM', 'XGBoost'.

    Returns:
        function: A function that takes an Optuna trial and returns a dictionary of hyperparameters for the specified model.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if model_name == "Ridge":
        return get_ridge_hyperparams
    elif model_name == "Lasso":
        return get_lasso_hyperparams
    elif model_name == "DecisionTree":
        return get_decision_tree_hyperparams
    elif model_name == "RandomForest":
        return get_random_forest_hyperparams
    elif model_name == "LightGBM":
        return get_lightgbm_hyperparams
    elif model_name == "XGBoost":
        return get_xgboost_hyperparams
    else:
        raise ValueError(
            f"get_hyperparams_getter_function_by_model_name(): Model {model_name} is not supported!"
        )
