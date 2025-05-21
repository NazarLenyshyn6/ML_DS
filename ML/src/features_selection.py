"""Module containing functionality for features selection."""

import numpy as np
import pandas as pd

from Internals.utility import check_dtype

class VarianceTreshold:
    """
    Removes features with variance lower than the specified threshold.

    Args:
        treshold (int | float): Minimum variance required for a feature to be kept.

    Raises:
        TypeError: If 'treshold' is not an int or float.
    """

    def __init__(self, treshold: int | float = 0):
        if not isinstance(treshold, (int, float)):
            raise TypeError('Treshold must be integer of float number.')
        self.treshold = treshold

    def __repr__(self) -> str:
        return f'VarianceTreshold(treshold={self.treshold})'
    
    @staticmethod
    def _variance(X: np.ndarray):
        """Calculate variance for each feature (column-wise) in the input array.

        Args:
            X (np.ndarray): Array to calculate feature variances from.

        Returns:
            np.ndarray: Variance of each feature.
        """
        X_mean = X.mean(0, keepdims=True)
        return np.mean((X - X_mean)**2, axis=0, keepdims=True)
    
    def _check_if_fit(self):
        if not hasattr(self, '_selected_features'):
            raise RuntimeError('VarianceTreshold has to be fit. Call .fit() first.')
    
    def get_features_names_out(self):
        """Get names of features that meet the variance threshold.

        Returns:
            pd.Index: Names of selected features.

        Raises:
            RuntimeError: If called before fitting.
        """
        self._check_if_fit()
        return self._features[self._selected_features]

    def fit(self, X: pd.DataFrame):
        """Compute variance of features and select those above the threshold.

        Args:
            X (pd.DataFrame): Input data to fit on.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        check_dtype(X, 'X', pd.DataFrame)
        features_variance = self._variance(X.values)
        self._selected_features = np.where(features_variance > self.treshold, True, False).reshape(-1)
        self._features = X.columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only features that were above the variance threshold during fitting.

        Args:
            X (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If X's features differ from those seen during fitting.
            RuntimeError: If called before fitting.
        """
        self._check_if_fit()
        check_dtype(X, 'X', pd.DataFrame)
        if not all(X.columns == self._features):
            raise ValueError('Features of DataFrame X must be exacly the same as seen during fitting.')
        return X.loc[:, self.get_features_names_out()]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Fit to the data, then return only the features that meet the variance threshold.

        Args:
            X (pd.DataFrame): Data to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        self.fit(X)
        return self.transform(X)