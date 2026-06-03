import joblib
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def get_model_by_name(model_name: str, random_state: int = 42, n_jobs: int = -1, input_shape=None):
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
    elif model_name == "MLP":
        return MLPRegressor(random_state=random_state)
    elif model_name == "RNN":
        from utils.keras_models import RNNRegressor
        return RNNRegressor(model__input_shape=input_shape, verbose=1, random_state=random_state)
    elif model_name == "CNN":
        from utils.keras_models import CNNRegressor
        return CNNRegressor(model__input_shape=input_shape, verbose=1, random_state=random_state)
    else:
        raise ValueError(f"get_model_by_name(): Model {model_name} is not supported!")
