import joblib
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers


def build_rnn_model(input_shape, n_units=64, n_layers=1, dropout=0.0, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for _ in range(n_layers):
        model.add(layers.SimpleRNN(n_units, activation="tanh", return_sequences=(_ < n_layers - 1)))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_cnn_model(input_shape, n_filters=32, n_conv_layers=2, kernel_size=3, dense_units=64, dropout=0.2, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for _ in range(n_conv_layers):
        model.add(layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(1))  # output layer for regression

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


def save_model(model, path):
    """Saves the model to the specified path using joblib."""
    joblib.dump(model, path)


def load_model(path):
    """Loads the model from the specified path using joblib."""
    return joblib.load(path)


def get_model_by_name(model_name: str, random_state: int = 42, n_jobs: int = -1, input_shape=None):
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
    elif model_name == "MLP":
        return MLPRegressor(random_state=random_state)
    elif model_name == "RNN":
        if input_shape is None:
            raise ValueError("get_model_by_name(): For RNN model, 'input_shape' must be provided.")
        return KerasRegressor(model=build_rnn_model, model__input_shape=input_shape, verbose=1, random_state=random_state)
    elif model_name == "CNN":
        if input_shape is None:
            raise ValueError("For CNN model, 'input_shape' must be provided.")
        return KerasRegressor(model=build_cnn_model, model__input_shape=input_shape, verbose=1, random_state=random_state)
    else:
        raise ValueError(f"get_model_by_name(): Model {model_name} is not supported!")
