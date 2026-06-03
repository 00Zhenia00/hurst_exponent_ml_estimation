from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasRegressor


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
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


class RNNRegressor(KerasRegressor):
    def __init__(
        self,
        n_units=64,
        n_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        epochs=20,
        batch_size=32,
        verbose=0,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            model=build_rnn_model,
            n_units=n_units,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


class CNNRegressor(KerasRegressor):
    def __init__(
        self,
        n_filters=32,
        n_conv_layers=2,
        kernel_size=3,
        dense_units=64,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=20,
        batch_size=32,
        verbose=0,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            model=build_cnn_model,
            n_filters=n_filters,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
