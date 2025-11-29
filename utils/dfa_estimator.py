from sklearn.base import BaseEstimator, RegressorMixin
from nolds import dfa
import numpy as np
import pandas as pd


def correct_dfa_output(hurst_exp: float):
    """Clip the DFA-based estimate to (0.001, 0.999)."""
    if hurst_exp >= 1:
        return 0.999
    elif hurst_exp <= 0:
        return 0.001
    return hurst_exp


class DFAEstimator(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper around the DFA estimator.
    Computes H ≈ DFA(x) - 1, with boundary correction.
    """

    def __init__(self):
        pass  # sklearn compatibility

    def fit(self, X, y=None):
        # No training needed
        return self

    def predict(self, X):
        """
        X: pd.DataFrame or np.ndarray, each row is a time series.
        Returns: np.ndarray of Hurst exponent estimates.
        """
        if isinstance(X, pd.DataFrame):
            data = X.to_numpy()
        else:
            data = np.asarray(X)

        def estimate_row(row):
            raw = dfa(row) - 1
            return correct_dfa_output(raw)

        predictions = np.apply_along_axis(estimate_row, 1, data)
        return predictions
