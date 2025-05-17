"""Module which contain utilities function for ML and DL projects."""

from typing import Any, Callable
import inspect

import numpy as np
import functools


def shuffle_arrays(*arrays: np.ndarray, random_state: int = None) -> list[np.ndarray]:
    """Shuffle multiple arrays in the same order each.

    Args:
        arrays: Arrays to shuffle. All has to have the same number of samples (first dimention).
        random_state: Controls the shuffling for reprodusible results.

    Raises:
        ValueError: If not arrays are provided.
        ValueError: If any input is  not a NumPy array.
        ValueError: If nput arrays have different number of samples.

    Returns:
        list[np.array]: A list of arrays shuffled in the same order each.
    """
    if not arrays:
        raise ValueError('At least one array must be provided.')
    
    if not all(isinstance(array, np.ndarray) for array in arrays):
        raise ValueError("All arrays must be instance of np.ndarray.")
    
    n_samples = arrays[0].shape[0]
    if not all(array.shape[0] == n_samples for array in arrays):
        raise ValueError("All arrays must have the same number of samples")
    
    rng = np.random.default_rng(seed=random_state) if random_state is not None else None
    indices = np.arange(n_samples)
    if rng:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    return [array[indices] for array in arrays]


class ValidateType:
    """Decorator to validate types of arguments provided to a function.
    
    Args:
        expected_type: (tuple[str, Any] | list[tuple[str, Any]]):  expected type for provided argument.
    """
    
    def __init__(self, expected_type: tuple[str, Any] | list[tuple[str, Any]]):
        self.expected_type = expected_type
    
    @staticmethod
    def _validate_arg(arg_name: Any, expected_type: Any, kwargs: dict) -> None:
        arg = kwargs[arg_name]
        if not isinstance(arg, expected_type) and not (inspect.isclass(arg) and 
                                                       issubclass(arg, expected_type)):
            raise TypeError(f'Invalid input type for {arg_name}: expected: {expected_type}')    
      
    @staticmethod      
    def _check_missing_arguments(expected_type, provided_type) -> None:
        # wrap expected type into list to avoid separate logic for not list case
        if not isinstance(expected_type, list):
            expected_type = [expected_type]
            
        for arg, _ in expected_type:
            if not arg in provided_type:
                raise KeyError(f'Required keyword argument {arg} is not found.')
                    
    def __call__(self, func: Callable) -> Callable:
        """Decorator call method that applies validation logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._check_missing_arguments(self.expected_type, kwargs)
            
            if isinstance(self.expected_type, list):
                for arg, expected_type in self.expected_type:
                    self._validate_arg(arg, expected_type, kwargs)   
            else:
                self._validate_arg(self.expected_type[0], self.expected_type[1], kwargs)   
            return func(*args, **kwargs)
        return wrapper
    
def validate_fit_inputs(X: np.ndarray, y: np.ndarray):
    """Validate input types and shapes or arrays provided for model training.

    Args:
        X: Features array.
        y: Targets array

    Raises:
        TypeError: If X or y is not NumPy array.
        ValueError: If X or y is not 2 dimentional array.
        ValueError: If X or y has different number of samples (first dimention).
        ValueError: If second dimention of y array is not equal to 1.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be NumPy arrays.")
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("X and y must be 2D arrays.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if y.shape[1] != 1:
        raise ValueError("y must have shape (n_samples, 1).")

def validate_predict_input(X: np.ndarray, w: np.ndarray):
    """Validate input type and shape of array provided for prediction.

    Args:
        X: Features  array.
        w: Weights array for matrix multiplication with X.

    Raises:
        TypeError:  If X is not NumPy array.
        ValueError: If X is not 2 dimentional array.
        ValueError: If X second dimentions does not match with w first dimention.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"X must have shape (n_samples, {w.shape[0]}).")
    
def check_array_dims(array: np.ndarray, required_n_dims: int):
    """Validate if NumPy array has required number of dimensions.

    Args:
        array: NumPy array for validation.
        required_n_dims: Required number of dimensions.

    Raises:
        ValueError: If number of dimensions in input array differs from required.
    """
    if not array.ndim == required_n_dims:
        raise ValueError(f'Array must have {required_n_dims} dimentions, got instead: {array.ndim}')
    
def check_dtype(input: Any, input_name: str, required_dtype: Any):
    """Validate input type.

    Args:
        input: Object to validate.
        input_name: Name to use in the error message.
        required_dtype: Expected data type.

    Raises:
        TypeError: If input does not match expected type.
    """
    if not isinstance(input, required_dtype):
        raise TypeError(f'{input_name} has to be of type {required_dtype}, got instead: {type(input)}')

