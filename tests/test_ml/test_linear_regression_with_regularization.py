import re

import pytest
import numpy as np
import contextlib
import pydantic

from ML.src.linear_regression_with_regularization import Ridge
from ML.src.linear_regression_with_regularization import Lasso


@pytest.mark.parametrize(
        'model, model_config, raised_exception', 
        [
            (Lasso, None, contextlib.nullcontext()), 
            (Ridge, None, contextlib.nullcontext()),
            (Lasso, {'learning_rate':0.001}, contextlib.nullcontext()),
            (Ridge, {'learning_rate':0.001}, contextlib.nullcontext()),
            (Lasso, {'learning_rate':'null'}, pytest.raises(pydantic.ValidationError)),
            (Ridge, {'learning_rate':'null'}, pytest.raises(pydantic.ValidationError)),
            ]
        )
def test_linear_regression_with_regularization_initialization(model, model_config,  raised_exception):
    with raised_exception:
        initialized_model = model(**model_config) if model_config else model()

@pytest.mark.parametrize(
        'model, X, Y, raised_exception', 
        [
            (Lasso(), np.random.randn(100, 2), np.random.randn(100, 1), contextlib.nullcontext()), 
            (Ridge(), np.random.randn(100, 2), np.random.randn(100, 1), contextlib.nullcontext()),
            (Lasso(), 1, np.random.randn(100, 1), pytest.raises(TypeError, match="X and y must be NumPy arrays.")), 
            (Ridge(), 1, np.random.randn(100, 1), pytest.raises(TypeError, match="X and y must be NumPy arrays.")),
            (Lasso(), np.random.randn(100, 2), 1, pytest.raises(TypeError, match="X and y must be NumPy arrays.")), 
            (Ridge(), np.random.randn(100, 2), 1, pytest.raises(TypeError, match="X and y must be NumPy arrays.")),
            (Lasso(), np.random.randn(100, 2), np.random.randn(100, 2, 1), pytest.raises(ValueError, match="X and y must be 2D arrays.")), 
            (Ridge(), np.random.randn(100, 2), np.random.randn(100, 2, 1), pytest.raises(ValueError, match="X and y must be 2D arrays.")),
            (Lasso(), np.random.randn(100, 2, 1), np.random.randn(100, 2), pytest.raises(ValueError, match="X and y must be 2D arrays.")), 
            (Ridge(), np.random.randn(100, 2, 1), np.random.randn(100, 2), pytest.raises(ValueError, match="X and y must be 2D arrays.")),
            (Lasso(), np.random.randn(101, 2), np.random.randn(100, 2), pytest.raises(ValueError, match="X and y must have the same number of samples.")), 
            (Ridge(), np.random.randn(101, 2), np.random.randn(100, 2), pytest.raises(ValueError, match="X and y must have the same number of samples.")),
            (Lasso(), np.random.randn(100, 2), np.random.randn(100, 2), pytest.raises(ValueError, match=re.escape("y must have shape (n_samples, 1)."))), 
            (Ridge(), np.random.randn(100, 2), np.random.randn(100, 2), pytest.raises(ValueError,  match=re.escape("y must have shape (n_samples, 1)."))),
            ]
            )
def test_linear_regression_with_regularization_fit(model, X, Y, raised_exception):
    with raised_exception:
        model.fit(X, Y)

@pytest.mark.parametrize(
        'model, X, raised_exception', 
        [
            (Lasso(), np.random.randn(10, 2), contextlib.nullcontext()), 
            (Ridge(), np.random.randn(10, 2), contextlib.nullcontext()),
            (Lasso(), 1, pytest.raises(TypeError, match="X must be a NumPy array.")), 
            (Ridge(), 1, pytest.raises(TypeError, match="X must be a NumPy array.")),
            (Lasso(), np.random.randn(10, 2, 2), pytest.raises(ValueError, match="X must be a 2D array.")), 
            (Ridge(), np.random.randn(10, 2, 2), pytest.raises(ValueError, match="X must be a 2D array.")),
            (Lasso(), np.random.randn(10, 4), pytest.raises(ValueError, match=re.escape("X must have shape (n_samples, 2)."))), 
            (Ridge(), np.random.randn(10, 4), pytest.raises(ValueError, match=re.escape("X must have shape (n_samples, 2)."))),
            ]
        )
def test_linear_regression_with_regularization_predict(model, X, raised_exception):
    np.random.seed(1)
    X_train, Y_train = np.random.randn(2, 2), np.random.randn(2,1)
    model.fit(X_train, Y_train)
    with raised_exception:
        prediction = model.predict(X)
        assert prediction.shape == (X.shape[0], 1)


@pytest.mark.parametrize('model', [Lasso(), Ridge()])
def test_unfit_linear_regression_with_regularization(model):
    np.random.seed(1)
    X = np.random.randn(100, 1)
    with pytest.raises(ValueError):
        model.predict(X)