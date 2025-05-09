"""Module  on  which  we will cover linear models  in  Machine  Learning."""

"""Linear models is a rules of linear transition from elements of one set to another.
That is basicly linear function from set X to set Y.
The are usefull when we make assumtion the some problem solution Y linearly depend of our knowled X,
so we try to find that linear rule, and that is exacly when linear models are useful.
When promble solution has not linear relation with knowled there is absolutly no gist in trying to find
linear rule of transition from knowled set to solution set (fit linear model).

"""

from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import override

import numpy as np

from loss_function_for_regression_task import MSE, MAE


class LinearModel(ABC):
    """Interface class for linear models."""
    
    @abstractmethod
    def __init__(self, input_dimentionality: int) -> None:
        """Initialize n-dimentional linear model."""
        self._validate_input_type(input=input_dimentionality, required_type=int)
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> None:
        """Perform function transformation from input vector to real number."""
        self._validate_input_type(input=X, required_type=np.ndarray)
        
    @staticmethod
    def _validate_input_type(input: Any, required_type: Any) -> None:
        """Validates input type if corresponds to requried type.
        
        Args:
			input: Data to validate.
			required_type: Required data type.
   
		Raises:
			TypeError: When data type does not corresponds to required data type.
        """
        if not isinstance(input, required_type):
            raise TypeError(f'Argument has to be of type {required_type}, got instead: {type(input)}')
                
                
class LinearRegression(LinearModel):
    """Capture linear relation between input and output.
    
    Formula:
    f(X) = W * X + b
    X - input features vector
    W - internal weights, same dimentionality as input features vector
    B - internal scalar feature.
    """
    
    @override
    def __init__(self, input_dimentionality: int):
        rng = np.random.default_rng(seed=42)
        self._weights = rng.random(input_dimentionality)
        self._bias = rng.random(1)
        
    def __repr__(self) -> str:
        return f'LinearRegression(input_dimentionality={self._weights.shape[0]})'
    
    def _validate_input_shape(self, input: np.ndarray) -> None:
        """Validate if input type has the same dimentionality as initialized model.
        
        Args:
			input: Data to validate.
   
		Raises:
			ValueError: When data has any other dimentionality that initialized model.
        """
        if not input.shape[-1] == self._weights.shape[0]:
            raise ValueError(
                f'Data has to be of dimentionality: {self._weights.shape}, got instead: {input.shape}'
                )
        
    @override
    def predict(self, X: np.ndarray):
        super().predict(X)
        self._validate_input_shape(X)
        return (X @ self._weights.T + self._bias)
        
        
if __name__ == '__main__':
    model = LinearRegression(input_dimentionality=5)
    print(model)
    
    X = np.random.randn(25, 5) # [2,2] -> 3.28
    y = np.ones(25)
    prediction  = model.predict(X)
    mse = MSE()
    mae = MAE()
    print(f'Input: {X}  | Target: {y}')
    print(f'Prediction: {prediction}')
    print(f'MSE: {mse(y, prediction)}')
    print(f'MAE: {mae(y, prediction)}')
