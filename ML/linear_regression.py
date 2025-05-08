"""Module which implements Linear Regression Model."""

import random 
import enum
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any

import numpy as np

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLossFunctionName(enum.Enum):
    MSE = 'mse'
    MAE = 'mae'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLossFunctionI(ABC):
    @abstractmethod
    def __call__(self,  y_true, y_pred):
        ...


class MSE:
    def __call__(self, y_true, y_pred):
        return np.mean((y_true -  y_pred)**2)
    
    def __repr__(self):
        return f'MSE()'
    

class MAE:
    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def __repr__(self):
        return f'MAE'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionGradientI(ABC):
    @abstractmethod
    def __call__(self, x, y, prediction, bias):
        ...

class MSEGradient:
    def  __call__(self, x, y, prediction, bias):
        return (-2*x*(y - prediction),  -2*(y - prediction)) if bias else (-2*x*(y - prediction),  np.array([0.0])) 
    
    def __repr__(self):
        return f'MSEGradient()'
    
class MAEGradient:
    def  __call__(self, x, y, prediction, bias):
        sign = np.sign(y - prediction)
        return (-sign*x,  np.array([-sign.item()])) if bias else (-sign*x, np.array([0.0]))
    
    def __repr__(self):
        return f'MAEGradient()'
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Model(ABC):
    """Base class for machine learning models."""
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to set X."""
        ...
        
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass of a model."""
        ...
        
class LinearRegression(Model):
    _loss_functions: dict[RegressionLossFunctionI, RegressionGradientI] =  {
        RegressionLossFunctionName.MSE: {'loss_fn': MSE(), 'grad': MSEGradient()},
        RegressionLossFunctionName.MAE: {'loss_fn': MAE(), 'grad': MAEGradient()}
        }
    def __init__(
            self, 
            learning_rate: int | float =   0.001, 
            n_iters: int = 10, 
            bias:  bool = True, 
            loss: RegressionLossFunctionName = RegressionLossFunctionName.MSE
            ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.bias = bias
        self.loss = loss
        self._loss_fn = self._loss_functions[self.loss]['loss_fn']
        self._grad = self._loss_functions[self.loss]['grad']

    def __repr__(self):
        return f'LinearRegression(learning_rate={self.learning_rate}, n_iters={self.n_iters}, bias={self.bias}, loss={self.loss})'
    
    @classmethod
    def _register_loss_function(cls, 
                                loss_function_name: RegressionLossFunctionName,
                                loss_function: RegressionLossFunctionI, 
                                grad: RegressionGradientI
                                ):
        cls._loss_functions[loss_function_name]  =  {'loss_fn': loss_function(), 'grad': grad()}

    def  _initialize_params(self, input_dim):
        self.w, self.b = np.random.uniform(-1, 1, input_dim), np.array([0.0])
    
    def _step(self, X, y):
        grads = [self._grad(x, y, model.predict(x), model.bias) for x, y in zip(X,y)]
        dw, db = [sum(param) for param in zip(*grads)]
        self.w -= self.learning_rate * dw  / X.shape[0]
        self.b -= self.learning_rate * db   /  X.shape[0]

    def fit(self, X, y, verbose: bool = True):
        self._initialize_params(X.shape[-1])
        for _ in range(self.n_iters):
            if verbose: 
                print(f'Loss: {self._loss_fn(y, model.predict(X))}')
            self._step(X, y)
        
    def predict(self, x):
        return np.dot(x, self.w) + self.b
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = LinearRegression(n_iters=200, loss=RegressionLossFunctionName.MAE)
    X = np.array([
        [1, 1, 1], 
        [2, 2, 2], 
        [3, 3, 3], 
        [4, 4, 3]
        ])
    y = np.array([6, 9, 3, 2])
    model.fit(X, y)
    print(model.predict([3,3,3]))

  
 