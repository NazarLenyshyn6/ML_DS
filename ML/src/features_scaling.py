"""Module which implements functionality for features scaling."""

import numpy as np

from Internals.utility import check_array_dims
from Internals.utility import check_dtype

class StandardScaler:
    """Scale features across the second dimension to zero mean and unit variance.

    Attributes:
        mean (np.ndarray): Mean values across features (axis=0).
        std (np.ndarray): Standard deviations across features (axis=0).
        is_fit (bool): Indicates whether the scaler has been fit.
    """

    def __init__(self):
        self.mean =  None
        self.std = None
        self.is_fit = False

    def __repr__(self):
        return f'StandardScaler(is_fit={self.is_fit})'

    def fit(self, X: np.ndarray) -> None:
        """Compute the mean and standard deviation for each feature.

        Args:
            X (np.ndarray): 2D array with shape (n_samples, n_features).

        Raises:
            TypeError: If X is not a NumPy array.
            ValueError: If X is not 2-dimensional.
        """

        check_dtype(X, 'X', np.ndarray)
        check_array_dims(X, 2)
        self.mean = X.mean(0, keepdims=True)
        self.std =  X.std(0, keepdims=True)
        self.is_fit = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to zero mean and unit variance using fit parameters.

        Args:
            X (np.ndarray): 2D array to scale.

        Returns:
            np.ndarray: Scaled array.

        Raises:
            TypeError: If X is not a NumPy array.
            ValueError: If X is not 2-dimensional.
            ValueError: If X second dimention does not equal to fit parameters second dimention.
            RuntimeError: If fit() has not been called before transform().
        """
        check_dtype(X, 'X', np.ndarray)
        check_array_dims(X, 2)
        if X.shape[1] != self.mean.shape[1]:
            raise ValueError(f'X second dimention must be equal to {self.min.shape[1]}')
        if not self.is_fit:
            raise RuntimeError('StandardScaler must be fit before calling transform.')
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X (np.ndarray): 2D array to scale.

        Returns:
            np.ndarray: Scaled array.
        """
        self.fit(X)
        return self.transform(X)
    
class MinMaxScaler:
    """Scale features across the second dimention to the range between 0 and 1.

    Attributes:
        min: Min values across the features (axis=0).
        max: Max values across the features (axis=0).
        is_fit: Indicates whether the scaler has been fit.
    
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.is_fit = False

    def __repr__(self) -> str:
        return f'MinMaxScaler(is_fit={self.is_fit})'

    def fit(self, X: np.ndarray) -> None:
        """Compute min and max values for each feature.

        Args:
            X (np.ndarray): 2D array with shape (n_samples, n_features).

        Raises:
            TypeError: If X is not a NumPy array.
            ValueError: If X is not 2-dimensional.
        """

        check_dtype(X, 'X', np.ndarray)
        check_array_dims(X, 2)
        self.min = X.min(0, keepdims=True)
        self.max = X.max(0, keepdims=True)
        self.is_fit = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to the range between 0 and 1 using fit parameters.

        Args:
            X (np.ndarray): 2D array to scale.

        Returns:
            np.ndarray: Scaled array.

        Raises:
            TypeError: If X is not a NumPy array.
            ValueError: If X is not 2-dimensional.
            ValueError: If X second dimention does not equal to fit parameters second dimention.
            RuntimeError: If fit() has not been called before transform().
        """
        check_dtype(X, 'X', np.ndarray)
        check_array_dims(X, 2)
        if X.shape[1] != self.min.shape[1]:
            raise ValueError(f'X second dimention must be equal to {self.min.shape[1]}')
        if not self.is_fit:
            raise RuntimeError('MinMaxScaler must be fit before calling transform.')
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X (np.ndarray): 2D array to scale.

        Returns:
            np.ndarray: Scaled array.
        """
        self.fit(X)
        return self.transform(X)