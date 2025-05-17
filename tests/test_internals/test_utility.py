import re

import pytest
import numpy as np
import contextlib

from Internals.utility import shuffle_arrays
from Internals.utility import validate_fit_inputs
from Internals.utility import validate_predict_input
from Internals.utility import check_array_dims
from Internals.utility import check_dtype


def test_shuffle_arrays():
    X = np.random.randn(100, 1)
    X_shuffled_1 = shuffle_arrays(X, random_state=22)
    X_shuffled_2 = shuffle_arrays(X, random_state=22)
    assert all(X_shuffled_1[0] == X_shuffled_2[0])


@pytest.mark.parametrize(
        'X, Y, raised_exception', 
        [
            (
                np.random.randn(10, 2), 
                np.random.randn(10, 1), 
                contextlib.nullcontext()
                ), 
            (
                np.random.randn(10, 2), 
                10, 
                pytest.raises(TypeError, match="X and y must be NumPy arrays.")
                ),
            (
                np.random.randn(10,2,2), 
                np.random.randn(10,2), 
                pytest.raises(ValueError, match="X and y must be 2D arrays.")
                ),
            (
                np.random.randn(1,2), 
                np.random.randn(10,2), 
                pytest.raises(ValueError, match="X and y must have the same number of samples.")
                ),
            (
                np.random.randn(1,2), 
                np.random.randn(1,2), 
                pytest.raises(ValueError, match=re.escape("y must have shape (n_samples, 1)."))),
            ]
        )
def test_validate_fit_inputs(X, Y, raised_exception):
    with raised_exception:
        validate_fit_inputs(X, Y)

@pytest.mark.parametrize(
        'X, w, raised_exception', 
        [
            (
                np.random.randn(10, 2),
                np.random.randn(2, 1), 
                contextlib.nullcontext()), 
            (
                1, 
                np.random.randn(2, 1), 
                pytest.raises(TypeError, match="X must be a NumPy array.")
                ), 
            (
                np.random.randn(10, 1, 1), np.random.randn(1, 1), 
                pytest.raises(ValueError, match=re.escape("X must be a 2D array."))
                ),
            (
                np.random.randn(10, 2), 
                np.random.randn(3, 1), 
                pytest.raises(ValueError, match=re.escape("X must have shape (n_samples, 3)."))
                ), 
            ]
        )
def test_validate_predict_input(X, w, raised_exception):
    with raised_exception:
        validate_predict_input(X, w)

@pytest.mark.parametrize(
        'array, required_n_dims, raised_exception', 
        [
            (np.random.randn(10, 2), 2, contextlib.nullcontext()), 
            (np.random.randn(10, 2), 1, pytest.raises(ValueError))
            ]
        )
def test_check_array_dims(array, required_n_dims, raised_exception):
    with raised_exception:
        check_array_dims(array, required_n_dims)

@pytest.mark.parametrize(
        'input, input_name, required_dtype, raised_exception', 
        [
            (1, 'number', int, contextlib.nullcontext()), 
            ('1', 'number', int,  pytest.raises(TypeError))
            ]
        )
def test_check_dtype(input, input_name,  required_dtype, raised_exception):
    with raised_exception:
        check_dtype(input, input_name, required_dtype)