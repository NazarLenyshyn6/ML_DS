import re

import pytest
import numpy as np
import contextlib

from ML.src.model_validation import train_test_split
from ML.src.model_validation import cross_validate
from ML.src.linear_regression import LinearRegression
from ML.src.linear_regression import ParamsInitialization
from ML.src.linear_regression import MSE


@pytest.fixture(scope='module')
def model():
    return LinearRegression(params_initialization=ParamsInitialization.CONSTANT)

@pytest.fixture(scope='module')
def loss_fn():
    return MSE()

@pytest.mark.parametrize(
        'X, Y, train_size, test_size, shuffle, random_state, raised_exception', 
        [
            (
                np.random.randn(10, 1), 
                np.random.randn(10, 1), 
                0.8, 
                None, 
                True, 
                None, 
                contextlib.nullcontext()
                ), 
            (
                np.random.randn(10, 1), 
                np.random.randn(10, 1), 
                0.8, 
                None, 
                False, 
                None, 
                contextlib.nullcontext()
                ),
            (
                None, 
                None, 
                0.8, 
                None, 
                False, 
                None, 
                pytest.raises(ValueError, match="At least one set must be provided.")
                ),
            (
                np.random.randn(10, 1), 
                np.random.randn(10, 1), 
                None, 
                None, 
                False, 
                None, 
                pytest.raises(ValueError, match="Either train_size or test_size must be specified.")
                ),
            (
                np.random.randn(8, 1), 
                np.random.randn(10, 1), 
                0.8, 
                None, 
                False, 
                None, 
                pytest.raises(ValueError, match=re.escape("All input sets must have the same number of samples. Got instead: [8, 10]"))
                ),
            (
                np.random.randn(10, 1), 
                np.random.randn(10, 1), 
                2, 
                None, 
                False, 
                None, 
                pytest.raises(ValueError, match=re.escape("train_size must be between 0 and 1. Got: 2"))
                ),
            ]
        )
def test_train_test_split(X, Y, train_size, test_size, shuffle, random_state, raised_exception):
    with raised_exception:
        if X is not None and Y is not None:
            split = train_test_split(X, Y, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state) 
            X_train, X_test, Y_train, Y_test =  split
            assert type(split) == list
            assert len(split) ==  4
            assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
            assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
        else:
            train_test_split(train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)


def test_train_test_split_reproducibility():
    X = np.random.randn(100, 1)
    Y = np.random.randn(100, 1)
    split1 = train_test_split(X, Y, train_size=0.7, random_state=42)
    split2 = train_test_split(X, Y, train_size=0.7, random_state=42)
    for arr1, arr2 in zip(split1, split2):
        np.testing.assert_array_equal(arr1, arr2)

@pytest.mark.parametrize(
    'X, Y, raised_exception', 
    [
        (
            np.random.randn(10, 1), 
            np.random.randn(10), 
            contextlib.nullcontext()
            ), 
        (
            1, 
            np.random.randn(10, 1), 
            pytest.raises(ValueError, match="Both X and Y must be NumPy arrays.")
            ),
        (
            np.random.randn(10, 1), 
            1, 
            pytest.raises(ValueError, match="Both X and Y must be NumPy arrays.")
            ),
        (
            np.random.randn(10, 1), 
            np.random.randn(11, 1), 
            pytest.raises(ValueError, match=re.escape("X and Y must have the same number of samples (fist dimention)."))
            ),
        ]
    )
def test_cross_validate(X, Y, model, loss_fn, raised_exception):
    with raised_exception:
        cross_validation_result = cross_validate(X=X, Y=Y, model=model, loss_fn=loss_fn)
        assert type(cross_validation_result) == np.ndarray
        assert len(cross_validation_result) == 3

def test_cross_validate_reproducibility(model, loss_fn):
    np.random.seed(1)
    X = np.random.randn(100, 1)
    Y = np.random.randn(100)
    cross_validation_result_1 = cross_validate(X=X, Y=Y, model=model, loss_fn=loss_fn, shuffle=True, random_state=22)
    cross_validation_result_2 = cross_validate(X=X, Y=Y, model=model, loss_fn=loss_fn, shuffle=True, random_state=22)
    assert np.mean(cross_validation_result_1) == np.mean(cross_validation_result_2)
