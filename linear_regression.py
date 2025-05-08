"""Module which implements Linear Regression Model."""

import random 
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import functools


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
        
        
class LinearRegression(Model, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    learning_rate: int | float = Field(default=0.001)
    n_iters: int = Field(default=2)
    bias: bool = Field(default=True)
    w: np.ndarray = Field(default=np.random.uniform(-1, 1, 1))
    b: np.ndarray = Field(default=np.random.uniform(-1, 1, 1))
    
    def _loss(self, x, y):
        return (y - self.predict(x))**2
    
    def _grad(self, x, y):
        loss = self._loss(x, y)
        return (-2*x*(y-loss), -2*(y-loss)) if self.bias else (-2*x*(y-loss), np.array([0]))
    
    def _step(self, X, y):
        dw, db = np.array([0]), np.array([0])
        for x in X:
            grad = self._grad(x, y)
            print(grad)
            dw += grad[0]
            db += grad[-1]
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db    
    
    def fit(self, X, y):
        print(self.w)
        for _ in range(self.n_iters):
            self._step(X, y)
        print(self.w)
        
    def predict(self, x):
        return x * self.w + self.b
        
if __name__ == '__main__':
    model = LinearRegression()
    X = np.array([1, 2, 3, 4])
    y = np.array([6, 9, 3, 2])
    model.fit(X, y)
    prediction = [model.predict(x) for x in X]
    # print(prediction)
  
 