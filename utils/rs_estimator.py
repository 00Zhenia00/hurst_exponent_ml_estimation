from sklearn.base import BaseEstimator, RegressorMixin
from nolds import hurst_rs
import numpy as np
import pandas as pd


class RSEstimator(BaseEstimator, RegressorMixin):
    """
    A scikit-learn API wrapper for the R/S estimator of the Hurst exponent
    using nolds.hurst_rs.
    """

    def __init__(self):
        # no hyperparameters, but keep init for sklearn compatibility
        pass

    def fit(self, X, y=None):
        # Nothing to train; return estimator unchanged
        return self

    def predict(self, X):
        """
        X: pd.DataFrame or np.ndarray (samples × time-series)
        Returns: np.ndarray of Hurst estimates for each row
        """
        if isinstance(X, pd.DataFrame):
            data = X.to_numpy()
        else:
            data = np.asarray(X)

        predictions = np.apply_along_axis(hurst_rs, 1, data)
        return predictions
