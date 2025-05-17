import pytest
import numpy as np
import contextlib
import pydantic

from ML.src import linear_regression


@pytest.mark.parametrize(
        'loss, early_stopping, X, y, verbose, raise_exception',
        [
            (
                linear_regression.RegressionLoss.MSE, 
                linear_regression.EarlyStopping.GRADIENT_NORM, 
                np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
                np.array([2, 4, 6, 8]),
                False,
                contextlib.nullcontext()
                ),
            (
                linear_regression.RegressionLoss.MAE, 
                linear_regression.EarlyStopping.GRADIENT_NORM, 
                np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
                np.array([2, 4, 6, 8]),
                False,
                contextlib.nullcontext()
                ),
            (
                linear_regression.RegressionLoss.MSE, 
                linear_regression.EarlyStopping.PARAMETERS_CHANGE, 
                np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
                np.array([2, 4, 6, 8]),
                False,
                contextlib.nullcontext()
                ),
            (
                linear_regression.RegressionLoss.MSE, 
                linear_regression.EarlyStopping.FUNCTION_CHANGE, 
                np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
                np.array([2, 4, 6, 8]),
                False,
                contextlib.nullcontext()
                ),
            (
                linear_regression.RegressionLoss.MSE, 
                linear_regression.EarlyStopping.FUNCTION_CHANGE, 
                np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
                np.array([2, 4, 6, 8]),
                True,
                contextlib.nullcontext()
                ),
            (
                linear_regression.RegressionLoss.MSE, 
                linear_regression.EarlyStopping.FUNCTION_CHANGE, 
                [np.array([[1, 1], [2, 2], [3, 3], [4, 4]])],
                np.array([2, 4, 6, 8]),
                True,
                pytest.raises(TypeError)
                ),
            (
                11, 
                linear_regression.EarlyStopping.FUNCTION_CHANGE, 
                [np.array([[1, 1], [2, 2], [3, 3], [4, 4]])],
                np.array([2, 4, 6, 8]),
                True,
                pytest.raises(pydantic.ValidationError)
                ),
            ]
        )
def test_linear_regression(loss, early_stopping, X, y, verbose, raise_exception):
    with raise_exception:
        model = linear_regression.LinearRegression(learning_rate=0.001,
                                                   n_iters=1000,
                                                   loss=loss,
                                                   early_stopping=early_stopping,
                                                   early_stopping_treshold=0.00001
                                                   )
        model.fit(X=X, y=y, verbose=verbose)
        model.predict(X=X)
    