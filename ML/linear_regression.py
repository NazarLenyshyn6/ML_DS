"""Module which implements Linear Regression Model."""

import random 
import enum
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any

import numpy as np

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLoss(enum.Enum):
    """Enumerates avaliable  regression loss functions."""

    MSE = 'mse'
    MAE = 'mae'
    SINMSE = 'sinmse'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLossFunctionI(ABC):
    """Defines interface of regression loss function."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates difference between true target label and predicted label with defined loss function."""

class MSE(RegressionLossFunctionI):
    """Implements mean squared error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates mean  squeared error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean((y_true -  y_pred)^2).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean  squeared error difference between true target label and predicted label.
        """
        return np.mean((y_true -  y_pred)**2)
    
    def __repr__(self)  -> str:
        return f'MSE()'
    

class MAE(RegressionLossFunctionI):
    """Implements mean absolute error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates mean  absolute error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean(np.abs(y_true -  y_pred)).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean absolute error difference between true target label and predicted label.
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def __repr__(self) -> str:
        return f'MAE'
    
class SinMSE(RegressionLossFunctionI):
    """Implements mean squared sinus of mean squeared error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates  mean squared sinus of mean squeared error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean(sin((y_true -  y_pred)^2)^2).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean squared sinus of mean squeared error difference between true target label and predicted label
        """
        return np.mean(np.sin((y_true - y_pred)**2)**2)
    
    def __repr__(self) -> str:
        return 'SinMSE()'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionGradientI(ABC):
    """Defines interface of loss function gradient calculation."""

    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        ...

class MSEGradient:
    """Calculate gradient of  MSE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulatespartial derivatives of  MSE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  MSE loss function.
        """
        if bias:
            return ((-2*X.T) @ (y - prediction) / X.shape[0], -2*(y - prediction)/ X.shape[0])
        else:
            return ((-2*X.T) @ (y - prediction)/ X.shape[0], np.zeros(X.shape[0]))

    def __repr__(self) -> str:
        return f'MSEGradient()'
    
class MAEGradient:
    """Calculate gradient of  MAE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulatespartial derivatives of  MAE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  MAE loss function.
        """
        sign = np.sign(y - prediction)
        return (X.T @ (-sign),  np.array([-sign.item()])) if bias else (X.T @ (-sign), np.array([0.0]))
    
    def __repr__(self) -> str:
        return f'MAEGradient()'
    
class SinMSEGradient:
    """Calculate gradient of  SinMSE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulates partial derivatives of  SinMSE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  SinMSE loss function.
        """
        mse_loss = (y - prediction)**2
        if bias:
            return  (X.T @ (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0], (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0])
        else:
            return   (X.T @ (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0], np.zeros(X.shape[0]))
        
    def __repr__(self) -> str:
        return f'SinMSEGradient()'
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class EarlyStopping(enum.Enum):
    GRADIENT_NORM = 'Gradient norm'
    PARAMETERS_CHANGE = 'Parameters change'
    FUNCTION_CHANGE =  'Function change'

class EarlyStoppingI(ABC):
    """Defines interface for early stopping of  gradient  descent."""

    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        ...

class GradientNormStopping(EarlyStoppingI):
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        for partial_derivative in grad:
            gradient_norm = np.abs(np.linalg.norm(partial_derivative))
            if gradient_norm != 0 and gradient_norm < treshold:
                return True
        return False

    def __repr__(self) -> str:
        return f'GradientNormStopping()'
    
class ParametersChangeStopping(EarlyStoppingI):
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        for param_before_step, param_after_step in zip(params_before_step, params_after_step):
            param_change = np.mean(np.abs(param_after_step - param_before_step))
            if param_change != 0 and param_change < treshold:
                return True
        return False
    
    def __repr__(self) -> str:
        return f'ParametersChangeStopping()'
    
class FunctionChangeStopping(EarlyStoppingI):
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        function_change = np.mean(np.abs(prediction_after - prediction_before))
        return True if function_change != 0 and  function_change < treshold else False


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
    """Implements linear regression machine learning model. Forward pass: f(x) = x * w  + b.

    Args:
        learning_rate: converage step.
        n_iters: Number of backprogation iterations.
        bias: Indicate if include  bias parameter in forward pass or not.
        loss:  Define which loss function will be minimized.
    """
    _losses: dict[RegressionLossFunctionI, dict[str, RegressionLossFunctionI | RegressionGradientI]] =  {
        RegressionLoss.MSE: {'loss_fn': MSE(), 'grad': MSEGradient()},
        RegressionLoss.MAE: {'loss_fn': MAE(), 'grad': MAEGradient()}
        }
    
    _early_stoppings : dict[str, EarlyStoppingI] = {
        EarlyStopping.GRADIENT_NORM:  GradientNormStopping(),
        EarlyStopping.PARAMETERS_CHANGE: ParametersChangeStopping(),
        EarlyStopping.FUNCTION_CHANGE: FunctionChangeStopping()
        }

    def __init__(
            self, 
            learning_rate: int | float = 0.001, 
            n_iters: int = 10, 
            bias: bool = True, 
            loss: RegressionLoss = RegressionLoss.MSE,
            early_stopping = EarlyStopping.GRADIENT_NORM,
            early_stopping_treshold: int  | float = 0.00001
            ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.bias = bias
        self.loss = loss
        self.early_stopping = early_stopping
        self.early_stopping_treshold = early_stopping_treshold
        self._loss_fn = self._losses[self.loss]['loss_fn']
        self._grad = self._losses[self.loss]['grad']
        self._early_stopping = self._early_stoppings[self.early_stopping]

    def __repr__(self):
        return f'LinearRegression(learning_rate={self.learning_rate}, n_iters={self.n_iters}, bias={self.bias}, loss={self.loss})'
    
    @classmethod
    def _register_loss(cls, 
                       loss_function_name: RegressionLoss,
                       loss_function: RegressionLossFunctionI, 
                       grad: RegressionGradientI
                       ) -> None:
        """Register new loss which can be optimized to fine optimizal weight of the model.

        Args:
            loss_function_name: Key on  which  new  loss function will be stored.
            loss_function: Object which will calculate loss with defined formula.
            grad: Object which calculates gradies of provided loss function  with respect to model  parameters.
        """
        cls._losses[loss_function_name]  =  {'loss_fn': loss_function(), 'grad': grad()}

    @classmethod
    def _register_early_stopping(cls, early_stopping_name: EarlyStopping, early_stopping: EarlyStoppingI):
        cls._early_stoppings[early_stopping_name] = early_stopping()

    def  _initialize_params(self, input_dims: tuple[int, int]) -> None:
        """Initialize model weights and bias with respect to input  vector dimentionality.

        Args:
            input_dims: Dimentionalties of input  vector.
        """
        self.w, self.b = np.random.uniform(-1, 1, input_dims[-1]), np.zeros(input_dims[0])
    
    def _step(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Perform one step of gradient descent with respect  to provided loss function.

        Args:
            X: Set to fit model to.
            y: Targets.
        """
        params_before_step = [self.w.copy(),  self.b.copy()]
        prediction_before = self.predict(X)
        dw, db = self._grad(X, y, self.predict(X), self.bias)
        self.w -= self.learning_rate * dw
        self.b -= self.b - self.learning_rate * db
        grad = [dw, db]
        params_after_step = [self.w, self.b]
        prediction_after = self.predict(X)
        return self._early_stopping(params_before_step=params_before_step, 
                                    grad=grad,
                                    params_after_step=params_after_step,
                                    prediction_before=prediction_before,
                                    prediction_after=prediction_after,
                                    treshold=self.early_stopping_treshold
                                    )

    @override
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Iterativaly perfrom gradient descent steps with respect to specified  loss  function.

        Args:
            X: Set to fit model to.
            y: Targets.
            verbose: Show loss function value on each  iteration, if True.
        """

        self._initialize_params(X.shape)
        for iteration in range(self.n_iters):
            if verbose: print(f'Loss {iteration}: {self._loss_fn(y, self.predict(X))}')
            if self._step(X, y): break

    @override  
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass of linear regression.

        Args:
            X: input vector for forward pass.
        
        Returns:
            np.ndarray: Result of forward pass with input matrix.
        """
        return X @ self.w + self.b if self.bias else X @ self.w
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Register custom loss function to LinearRegression
    LinearRegression._register_loss(loss_function_name=RegressionLoss.SINMSE,
                                             loss_function=SinMSE,
                                             grad=SinMSEGradient)
    
    # Initialize model with specific loss function
    model = LinearRegression()

    # Initialize  training  set
    X = np.array([
        [1, 1], 
        [2, 2], 
        [3, 3], 
        [4, 4]
        ])
    y = np.array([6, 9, 3, 2])

    # Fit the model and make prediction.
    model.fit(X, y)
    print(model.predict(X))