"""Module on which we will define and implement loss function for regression problem."""

"""Loss function for regression problem:

Mean Squared Error (MSE) = f(x1, x2) = (x1 - x2)^2
Mean Absoluted Error (MAE) = f(x1, x2) = |x1 - x2|

Main Characteristic:
MSE - loss function which suites well when we can not allow our model to make big mistakes, 
whe it critical for us to remove huge output results difference, that loss function will suit 
well because mse will grow in n^2 manner when loss goes to infitity, so to minimise that loss function
it is critical to keep loss low, so when we have to have model which strictly low loss MSE is perfect
loss function to find such model function.

MAE - loss function which suites well when we care less how big loss is, when for us small loss and big
loss means the same, that is because MAE in difference to MSE grows in n manner so big loss and small loss
will contribute is the same manner.
"""

from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any

import numpy as np
from  sklearn.metrics import  mean_squared_error, mean_absolute_error


class LossFunction(ABC):
    """Interface class for all loss functions."""
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        self._validate_inputs([y_true, y_pred])
        """Average loss function value of all set samples."""
        ...
        
    @staticmethod
    def _validate_inputs(inputs: Any)  -> None:
        """Helper function to validate if input type is instance of numpy array.
        
        Args:
			input: Data to validate.
   
		Raises:
			TypeError: When input  data is  not instance of numpy  array
        """
        for input in inputs:
            if  not isinstance(input, np.ndarray):
                raise TypeError(f'Input has to be np.ndarray got instead: {type(input)}')
        
        
class MSE(LossFunction):
    """Mease Squeared Error loss function.
    
    Formula:
    f(y_true, y_pred) = (y_true - y_pred)^2
    y_true  -  True target label.
    y_pred  -  Target  labels which is output from ML model.
    """
    
    @override
    def __call__(self,  y_true: np.ndarray, y_pred: np.ndarray) ->  int  | float:
        super().__call__(y_true, y_pred)
        return np.mean((y_true -  y_pred) **  2)
    
    def __repr__(self) ->  str:
        return f'MSE()'
    
    
class MAE(LossFunction):
    """Mean Absolute  Error loss function.
    
    Formula:
    f(y_true, y_pred) = |y_true -  y_pred|
    y_true - True  target  label.
    y_pred - Target  labels which is output  from  ML  model.
    """
    
    @override
    def  __call__(self, y_true: np.ndarray, y_pred:  np.ndarray)  -> int  | float:
        super().__call__(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    def __repr__(self) -> str:
        return f'MAE()'
    
    
if __name__ == '__main__':
    mse = MSE()
    mae  = MAE()
    print(mse)
    print(mae)
    
    y_true  = np.array([1,1,1,1,1])
    y_pred =  np.array([0.9,0.2,0.39,1,1])
    print(f'MSE: {mse(y_true, y_pred)}')
    print(f'MAE: {mae(y_true, y_pred)}')
    print(f'Sklearn MAE: {mean_squared_error(y_true, y_pred)}')
    print(f'Sklearn MAE: {mean_absolute_error(y_true, y_pred)}')