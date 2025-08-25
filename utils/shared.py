from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def get_model_by_name(model_name: str, random_state: int = 42, n_jobs: int = -1):
    """
    Returns an instantiated model based on the provided model name.

    Args:
        model_name (str): The name of the model to instantiate. Supported models are:
            "Ridge", "Lasso", "DecisionTree", "RandomForest", "LightGBM", "XGBoost".
        random_state (int): Random seed for reproducibility. Default is 42.
        n_jobs (int): Number of CPU cores to use for models that support parallel processing.
            Default is -1 (use all available cores).

    Returns:
        An instantiated model object corresponding to the provided model name.

    Raises:
        ValueError: If the provided model name is not supported.
    """
    if model_name == "Ridge":
        return Ridge(random_state=random_state)
    elif model_name == "Lasso":
        return Lasso(random_state=random_state)
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(random_state=random_state)
    elif model_name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "LightGBM":
        return LGBMRegressor(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=n_jobs)
    else:
        raise ValueError(f"get_model_by_name(): Model {model_name} is not supported!")
