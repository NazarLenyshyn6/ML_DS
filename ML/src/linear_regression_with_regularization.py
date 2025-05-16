"""Module containing implementation of Linear Regressions with regularization."""

import numpy as np
import pydantic

from Internals.utility  import  validate_fit_inputs, validate_predict_input


class Ridge(pydantic.BaseModel):
    """Linear regression model with  L2-regularization.

    Attributes:
        learning_rate: Size of converge step.
        n_iters: Number of iterations for gradient descent.
        alpha: Regularization strenght.
        bias: Whether to include bias or not. Default True.

    Raises:
        ValidationError: If any of provided attributes does not correspond to required data type.
    """
    model_config   = pydantic.ConfigDict(arbitrary_types_allowed=True)
    
    learning_rate: float = pydantic.Field(default=0.0001)
    n_iters: int = pydantic.Field(default=10)
    alpha: int | float = pydantic.Field(default=1)
    bias: bool  = pydantic.Field(default=True)
    w: np.ndarray = pydantic.Field(default=None)
    b: np.ndarray  =  pydantic.Field(default=None)

    def __repr__(self):
        return f'Ridge(learning_rate={self.learning_rate}, n_iters={self.n_iters}, alpha={self.alpha})'

    def _step(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculates partial derivative with respect to weight and bias for MSE loss with L2 regularization and perform one parameters update.

        Args:
            X: Features array.
            y: Targets array.
        """
        pred  = self.predict(X=X)
        dw = (-2 / y.shape[0]) * X.T @ (y - pred) +  2 * self.w * self.alpha
        db = np.sum((-2 / y.shape[0]) * (y - pred), 0, keepdims=True) if self.bias else 0

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Linear Regression with L2 regularization to features X and targets Y.

        Args:
            X: Features array.
            y: Targets array.

        Raises:
            TypeError: If X or y is not NumPy array.
            ValueError: If X or y is not 2 dimentional NumPy array or X and Y has different number of samples (first dimention).
            ValueError: If last dimention of target array 1 is not equal to 1.
        """
        validate_fit_inputs(X, y)
        self.w, self.b = np.random.randn(X.shape[1], 1), np.zeros((1,1))
        for iteration in range(self.n_iters):
            self._step(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass of Linear Regression.

        Args:
            X: features array

        Raises:
            TypeError: If X is not NumPy array.
            ValueError: If making prediction before fitting the model.

        Returns:
            np.ndarray: Predicted targets array.
        """
        if self.w is None:
            raise ValueError("Model has not been fitted yet. Call `.fit()` first.")

        validate_predict_input(X, self.w)
        return X @ self.w + self.b
    

class Lasso(pydantic.BaseModel):
    """Linear regression model with  L1-regularization.

    Attributes:
        learning_rate: Size of converge step.
        n_iters: Number of iterations for gradient descent.
        alpha: Regularization strenght.
        bias: Whether to include bias or not. Default True.

    Raises:
        ValidationError: If any of provided attributes does not correspond to required data type.
    """
    model_config   = pydantic.ConfigDict(arbitrary_types_allowed=True)
    
    learning_rate: float = pydantic.Field(default=0.0001)
    n_iters: int = pydantic.Field(default=10)
    alpha: int | float = pydantic.Field(default=1)
    bias: bool  = pydantic.Field(default=True)
    w: np.ndarray = pydantic.Field(default=None)
    b: np.ndarray  =  pydantic.Field(default=None)

    def __repr__(self):
        return f'Lasso(learning_rate={self.learning_rate}, n_iters={self.n_iters}, alpha={self.alpha})'

    def _step(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculates partial derivative with respect to weight and bias for MSE loss with L1 regularization and perform one parameters update.

        Args:
            X: Features array.
            y: Targets array.
        """
        pred  = self.predict(X=X)
        dw = (-2 / y.shape[0]) * X.T @ (y - pred) +  np.sign(self.w) * self.alpha
        db = np.sum((-2 / y.shape[0]) * (y - pred), 0, keepdims=True) if self.bias else 0

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Linear Regression with L2 regularization to features X and targets Y.

        Args:
            X: Features array.
            y: Targets array.

        Raises:
            TypeError: If X or y is not NumPy array.
            ValueError: If X or y is not 2 dimentional NumPy array or X and Y has different number of samples (first dimention).
            ValueError: If last dimention of target array 1 is not equal to 1.
        """
        validate_fit_inputs(X, y)
        self.w, self.b = np.random.randn(X.shape[1], 1), np.zeros((1,1))
        for iteration in range(self.n_iters):
            self._step(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass of Linear Regression.

        Args:
            X: features array

        Raises:
            TypeError: If X is not NumPy array.
            ValueError: If making prediction before fitting the model.

        Returns:
            np.ndarray: Predicted targets array.
        """
        if self.w is None:
            raise ValueError("Model has not been fitted yet. Call `.fit()` first.")

        validate_predict_input(X, self.w)
        return X @ self.w + self.b