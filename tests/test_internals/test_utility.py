import pytest
import numpy as np
import contextlib

from Internals.utility import validate_fit_inputs
from Internals.utility import validate_predict_input
from Internals.utility import check_array_dims
from Internals.utility import check_dtype


@pytest.mark.parametrize(
        'X, Y, has_exception, exception_message', 
        [
            (np.random.randn(10, 2), np.random.randn(10, 1), False, ""), 
            (np.random.randn(10, 2), 10, True, "X and y must be NumPy arrays."),
            (np.random.randn(10,2,2), np.random.randn(10,2), True, "X and y must be 2D arrays."),
            (np.random.randn(1,2), np.random.randn(10,2), True, "X and y must have the same number of samples."),
            (np.random.randn(1,2), np.random.randn(1,2), True, "y must have shape (n_samples, 1)."),
            ]
        )
def test_validate_fit_inputs(X, Y, has_exception, exception_message):
    if has_exception:
        try:
            validate_fit_inputs(X, Y)
        except Exception as e:
            assert str(e) == exception_message
    else:
        validate_fit_inputs(X, Y)

@pytest.mark.parametrize(
        'X, w, has_exception, exception_message', 
        [
            (np.random.randn(10, 2), np.random.randn(2, 1), False, ""), 
            (1, np.random.randn(2, 1), True, "X must be a NumPy array."), 
            (np.random.randn(10, 2), np.random.randn(3, 1), True, "X must have shape (n_samples, 3)."), 
            ]
        )
def test_validate_predict_input(X, w, has_exception, exception_message):
    if has_exception:
        try:
            validate_predict_input(X, w)
        except Exception as e:
            assert str(e) == exception_message
    else:
        ...

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